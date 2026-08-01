import unittest
from types import SimpleNamespace

from conversation.conversation_manager import ConversationManager
from conversation.prompts import (
    POLICY_SERVICE_UNAVAILABLE_RESPONSE,
    _recommending_policy_instructions,
)
from conversation.state import ConversationState
from evaluation.validate_rag import evaluate_response
from ingestion.models import PolicyMetadata
from rag.grounding import INSUFFICIENT_EVIDENCE_RESPONSE, format_retrieved_context
from rag.models import RAGResponse, RetrievedChunk
from rag.retriever import PolicyRetriever
from recommendation.policy import Policy
from recommendation.policy_identity import PolicyIdentityResolver, policy_display_labels
from recommendation.recommendation_engine import RecommendationEngine


class PolicyCodeCompatibilityTests(unittest.TestCase):
    def _legacy_dict(self):
        return {
            "policy_id": "POL-1",
            "policy_name": "Legacy Cover",
            "insurer": "Legacy Insurance",
            "plan_type": "Individual",
            "premium": 1000,
            "min_age": 18,
            "max_age": 65,
            "sum_insured": 100000,
            "smoker_allowed": True,
            "covers_diabetes": False,
            "covers_hypertension": False,
            "parents_allowed": False,
            "children_allowed": False,
        }

    def test_legacy_policy_without_code_loads_and_round_trips(self):
        policy = Policy.from_dict(self._legacy_dict())
        self.assertIsNone(policy.policy_code)
        self.assertIn("policy_code", policy.to_dict())
        self.assertIsNone(policy.to_dict()["policy_code"])

    def test_legacy_metadata_without_code_loads_and_round_trips(self):
        metadata = PolicyMetadata.from_dict(self._legacy_dict())
        self.assertIsNone(metadata.policy_code)
        self.assertIsNone(metadata.to_dict()["policy_code"])


class IdentityResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policies = RecommendationEngine().policies
        cls.resolver = PolicyIdentityResolver(cls.policies)

    def resolve(self, message, recent=()):
        return self.resolver.resolve(message, recent_policies=recent)

    def test_exact_number_has_highest_precedence(self):
        result = self.resolve("premium for SLI/FHS/2026/00792144")
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.matched_by, "policy_id")
        self.assertEqual(result.policy.policy_id, "SLI/FHS/2026/00792144")

    def test_product_code_accepts_punctuation_and_spoken_spacing(self):
        punctuated = self.resolve("waiting period for SFHS-2026")
        spoken = self.resolve("waiting period for S F H S 2026")
        self.assertEqual(punctuated.policy.policy_id, "SLI/FHS/2026/00792144")
        self.assertEqual(spoken.policy.policy_id, "SLI/FHS/2026/00792144")
        self.assertEqual(spoken.matched_by, "policy_code")

    def test_identifier_near_misses_do_not_select_a_known_policy(self):
        for message in (
            "waiting period for SLTP-20260",
            "premium for SLI/FHS/2026/007921440",
            "coverage for SFHS-2026X",
        ):
            with self.subTest(message=message):
                self.assertEqual(self.resolve(message).status, "none")

    def test_sentence_period_after_identifier_keeps_the_exact_match(self):
        code = self.resolve("Use policy code SFHS-2026. What does it cover?")
        number = self.resolve(
            "Use policy number SLI/FHS/2026/00792144. What does it cover?"
        )
        self.assertEqual(code.status, "matched")
        self.assertEqual(code.policy.policy_id, "SLI/FHS/2026/00792144")
        self.assertEqual(number.status, "matched")
        self.assertEqual(number.policy.policy_id, "SLI/FHS/2026/00792144")

    def test_ordinal_uses_recent_recommendation_order(self):
        result = self.resolve("what about the second policy", self.policies[:3])
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.policy.policy_id, self.policies[1].policy_id)

    def test_bare_one_is_not_silently_treated_as_the_first_option(self):
        ambiguous_reference = self.resolve(
            "Which one has maternity coverage?",
            self.policies[:3],
        )
        explicit_reference = self.resolve(
            "What about option one?",
            self.policies[:3],
        )
        self.assertEqual(ambiguous_reference.status, "ambiguous")
        self.assertEqual(explicit_reference.status, "matched")
        self.assertEqual(explicit_reference.policy.policy_id, self.policies[0].policy_id)

    def test_unique_full_name_and_insurer_shorthand_resolve_from_full_catalog(self):
        by_name = self.resolve("maximum age for VitalCare Family Health Shield")
        by_insurer = self.resolve("premium for ApexCare")
        self.assertEqual(by_name.policy.policy_id, "VCH/FL/2026/00193572")
        self.assertEqual(by_insurer.policy.policy_id, "ACH/EL/2026/00721904")

    def test_duplicate_name_and_shared_insurer_are_ambiguous(self):
        by_name = self.resolve("waiting period for SecureLife Family Health Suraksha")
        by_insurer = self.resolve("what does SecureLife cover")
        self.assertEqual(by_name.status, "ambiguous")
        self.assertEqual(len(by_name.candidates), 2)
        self.assertEqual(by_insurer.status, "ambiguous")

    def test_conflicting_identifiers_never_select_first_match(self):
        result = self.resolve(
            "Use TSG/HP/2026/00562481 for ApexCare Elevate Health Plan"
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.matched_by, "conflicting_identifiers")

        multiple_codes = self.resolve(
            "Use SLI/FHS/2026/00792144, code SFHS-2026, but code SLFH-2026"
        )
        self.assertEqual(multiple_codes.status, "ambiguous")


class ConversationIdentityTests(unittest.TestCase):
    def test_duplicate_bare_name_clears_stale_scope_and_requests_clarification(self):
        manager = ConversationManager(session_id="identity-test")
        duplicate = [
            policy
            for policy in manager.rec_engine.policies
            if policy.policy_name == "SecureLife Family Health Suraksha"
        ]
        manager.last_recommended_policies = duplicate
        manager._set_active_policy(duplicate[0])

        selected = manager.prepare_policy_context(
            "What is the waiting period for SecureLife Family Health Suraksha?"
        )

        self.assertIsNone(selected)
        self.assertIsNone(manager.active_policy_id)
        self.assertTrue(manager.policy_selection_required)
        self.assertEqual(len(manager.policy_resolution_candidates), 2)
        prompt = manager.build_system_prompt()
        self.assertIn("code SLFH-2026", prompt)
        self.assertIn("code SFHS-2026", prompt)

    def test_short_generic_question_does_not_inherit_active_policy(self):
        manager = ConversationManager(session_id="identity-test")
        manager._set_active_policy(manager.rec_engine.policies[0])
        manager.policy_discussion_turns_remaining = 2
        self.assertIsNone(manager.prepare_policy_context("what is insurance?"))
        self.assertIsNone(manager.active_policy_id)
        self.assertFalse(manager.policy_selection_required)

    def test_exact_code_selects_the_correct_duplicate(self):
        manager = ConversationManager(session_id="identity-test")
        selected = manager.prepare_policy_context(
            "What is the waiting period for S F H S 2026?"
        )
        self.assertEqual(selected, "SLI/FHS/2026/00792144")
        self.assertFalse(manager.policy_selection_required)

    def test_unique_catalog_name_resolves_even_when_not_recently_recommended(self):
        manager = ConversationManager(session_id="identity-test")
        manager.last_recommended_policies = []
        selected = manager.prepare_policy_context(
            "What is the maximum age for VitalCare Family Health Shield?"
        )
        self.assertEqual(selected, "VCH/FL/2026/00193572")

    def test_active_scope_is_kept_only_for_a_genuine_short_follow_up(self):
        manager = ConversationManager(session_id="identity-test")
        policy = manager.rec_engine.policies[0]
        manager._set_active_policy(policy)
        manager.policy_discussion_turns_remaining = 2
        self.assertEqual(manager.prepare_policy_context("what about dialysis"), policy.policy_id)

        self.assertIsNone(
            manager.prepare_policy_context(
                "What is a waiting period in health insurance generally?"
            )
        )
        self.assertIsNone(manager.active_policy_id)


class IdentityPresentationAndRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policies = RecommendationEngine().policies

    def test_only_duplicate_marketing_names_gain_code_and_number(self):
        duplicate = [
            policy
            for policy in self.policies
            if policy.policy_name == "SecureLife Family Health Suraksha"
        ]
        labels = policy_display_labels(duplicate)
        self.assertTrue(all("policy number" in label for label in labels))
        unique = next(policy for policy in self.policies if policy.policy_name.startswith("ApexCare"))
        unique_prompt = _recommending_policy_instructions([unique])
        self.assertIn("ApexCare Elevate Health Plan by", unique_prompt)
        self.assertNotIn("code AC/EHP/2026/D2", unique_prompt)

    def test_detail_only_ranking_retains_requested_code_or_number_fact(self):
        retriever = PolicyRetriever.__new__(PolicyRetriever)
        retriever._policy_identity_by_id = {
            "SLI/FHS/2026/00792144": {
                "policy_id": "SLI/FHS/2026/00792144",
                "policy_code": "SFHS-2026",
                "policy_name": "SecureLife Family Health Suraksha",
            }
        }
        policy_id = "SLI/FHS/2026/00792144"
        faq_query = retriever.ranking_query_for(
            "For SecureLife Family Health Suraksha, policy code SFHS-2026, what is the waiting period?",
            policy_id,
        )
        code_query = retriever.ranking_query_for(
            "What is the policy code for SecureLife Family Health Suraksha?",
            policy_id,
        )
        self.assertEqual(faq_query.lower(), "what is the waiting period?")
        self.assertIn("policy code", code_query.lower())
        self.assertNotIn("securelife", code_query.lower())

    def test_grounding_header_contains_name_code_and_exact_number(self):
        chunk = RetrievedChunk(
            "c1", "POL/123", "Example Cover", 0, "Evidence", 0.9, "EX-1"
        )
        context = format_retrieved_context([chunk])
        self.assertIn("Policy: Example Cover", context)
        self.assertIn("Policy Code: EX-1", context)
        self.assertIn("Policy Number: POL/123", context)

    def test_evaluation_rejects_same_name_with_wrong_policy_id(self):
        wrong = RetrievedChunk(
            "c1", "wrong-id", "SecureLife Family Health Suraksha", 0, "Evidence", 0.9
        )
        response = RAGResponse("answer", [wrong], ["c1"])
        passed, reason = evaluate_response(
            {
                "expect_unaware": False,
                "expected_policy": "SecureLife Family Health Suraksha",
                "expected_policy_id": "right-id",
                "expected_groups": [],
            },
            response,
        )
        self.assertFalse(passed)
        self.assertIn("policy ID", reason)

    def test_retrieval_error_uses_temporary_service_message_not_unaware(self):
        manager = ConversationManager(session_id="identity-test")
        manager.state = ConversationState.ANSWERING_POLICY_QUESTIONS
        prompt = manager.build_system_prompt(retrieval_error=True)
        self.assertIn(POLICY_SERVICE_UNAVAILABLE_RESPONSE, prompt)
        self.assertNotIn(INSUFFICIENT_EVIDENCE_RESPONSE, prompt)


if __name__ == "__main__":
    unittest.main()
