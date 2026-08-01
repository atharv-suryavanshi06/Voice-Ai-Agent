import unittest
from types import SimpleNamespace

from conversation.conversation_manager import ConversationManager, EmailDeliveryState
from conversation.prompts import _email_verification_instructions
from conversation.amounts import format_indian_currency_for_speech
from conversation.customer_profile import parse_spoken_number, normalize_stt_aliases
from conversation.state import ConversationState
from conversation.question_flow import Question
from evaluation.validate_rag import evaluate_response
from rag.grounding import INSUFFICIENT_EVIDENCE_RESPONSE
from rag.models import RAGResponse, RetrievedChunk
from rag.rag_pipeline import RAGPipeline
from rag.retriever import PolicyRetriever
from rag.service import RAGService


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query, top_k=5, policy_id_filter=None):
        self.calls.append((query, top_k, policy_id_filter))
        return list(self.chunks)


class FakeEmailService:
    def __init__(self, configured=True, result=True):
        self.configured = configured
        self.result = result
        self.calls = 0

    def is_configured(self):
        return self.configured

    def send_policy_recommendation_email(self, **_kwargs):
        self.calls += 1
        return self.result


class RAGBehaviorTests(unittest.TestCase):
    def test_retriever_rejects_below_threshold_chunks(self):
        low = RetrievedChunk("c-low", "p1", "Policy", 0, "weak", 0.01)
        retriever = PolicyRetriever.__new__(PolicyRetriever)
        retriever.candidate_k = 5
        retriever.min_relevance_score = 0.05
        retriever.enable_tracing = False
        retriever._get_vector_candidates = lambda *_args, **_kwargs: [low]
        retriever._get_bm25_candidates = lambda *_args, **_kwargs: []
        retriever._reciprocal_rank_fusion = lambda *_args, **_kwargs: [low]
        retriever.reranker = SimpleNamespace(rerank=lambda *_args, **_kwargs: [low])

        self.assertEqual(retriever.retrieve("irrelevant", top_k=1), [])

    def test_pipeline_uses_supplied_canonical_service(self):
        retriever = FakeRetriever([])
        service = RAGService(retriever)
        pipeline = RAGPipeline(retriever=retriever, rag_service=service, client=None)
        response = pipeline.answer_question("irrelevant question")

        self.assertIs(pipeline.rag_service, service)
        self.assertEqual(response.answer, INSUFFICIENT_EVIDENCE_RESPONSE)
        self.assertEqual(len(retriever.calls), 1)

    def test_irrelevant_retrieval_cannot_pass_validation(self):
        chunk = RetrievedChunk("c1", "p1", "Wrong Policy", 0, "irrelevant", 0.9)
        response = RAGResponse(
            answer=INSUFFICIENT_EVIDENCE_RESPONSE,
            retrieved_chunks=[chunk],
            sources=["c1"],
        )
        passed, _ = evaluate_response({"expect_unaware": True}, response)
        self.assertFalse(passed)

    def test_broader_candidates_only_for_multi_fact_questions(self):
        retriever = PolicyRetriever.__new__(PolicyRetriever)
        retriever.candidate_k = 15
        retriever.min_relevance_score = 0.05
        retriever.enable_tracing = False
        retriever.reranker = SimpleNamespace(rerank=lambda _q, chunks, top_k: chunks[:top_k])
        calls = []
        retriever._get_vector_candidates = lambda _q, count, _policy: calls.append(("vector", count)) or []
        retriever._get_bm25_candidates = lambda _q, count, _policy: calls.append(("bm25", count)) or []

        retriever.retrieve("What is the policy number and sum insured?", top_k=5)
        self.assertEqual(calls, [("vector", 25), ("bm25", 25)])

        calls.clear()
        retriever.retrieve("What is the waiting period?", top_k=5)
        self.assertEqual(calls, [("vector", 15), ("bm25", 15)])

    def test_catalog_filter_excludes_unlisted_vectors(self):
        retriever = PolicyRetriever.__new__(PolicyRetriever)
        retriever._active_policy_ids = {"listed"}
        self.assertTrue(retriever._is_active_catalog_policy("listed"))
        self.assertFalse(retriever._is_active_catalog_policy("legacy"))


class PolicyConversationRoutingTests(unittest.TestCase):
    def _policy(self, policy_id, policy_name, insurer="Example Insurance"):
        return SimpleNamespace(policy_id=policy_id, policy_name=policy_name, insurer=insurer)

    def test_voice_detail_question_without_punctuation_retrieves(self):
        manager = ConversationManager(session_id="routing-test")
        self.assertTrue(manager.should_retrieve_policy_context("what is the ambulance limit"))

    def test_short_follow_up_uses_recent_policy_context(self):
        manager = ConversationManager(session_id="routing-test")
        manager.policy_discussion_turns_remaining = 2
        self.assertTrue(manager.should_retrieve_policy_context("what about dialysis"))

    def test_ordinal_reference_selects_recommended_policy(self):
        manager = ConversationManager(session_id="routing-test")
        manager.last_recommended_policies = [
            self._policy("policy-1", "First Cover"),
            self._policy("policy-2", "Second Cover"),
        ]
        self.assertEqual(manager.prepare_policy_context("what is the policy code for the second policy"), "policy-2")
        self.assertFalse(manager.policy_selection_required)

    def test_ambiguous_this_policy_requests_clarification(self):
        manager = ConversationManager(session_id="routing-test")
        manager.last_recommended_policies = [
            self._policy("policy-1", "First Cover"),
            self._policy("policy-2", "Second Cover"),
        ]
        self.assertIsNone(manager.prepare_policy_context("what is the policy number for this policy"))
        self.assertTrue(manager.policy_selection_required)

class SpeechAmountFormattingTests(unittest.TestCase):
    def test_formats_indian_currency_as_words(self):
        self.assertEqual(format_indian_currency_for_speech(7_000_000), "seventy lakh rupees")
        self.assertEqual(format_indian_currency_for_speech(26_000), "twenty-six thousand rupees")
        self.assertEqual(
            format_indian_currency_for_speech(46_302),
            "forty-six thousand three hundred and two rupees",
        )

    def test_normalizes_common_stt_amount_aliases(self):
        self.assertEqual(normalize_stt_aliases("75 black rupees"), "75 lakh rupees")
        self.assertEqual(parse_spoken_number("75 black rupees"), 7_500_000)
        self.assertEqual(parse_spoken_number("seventy five lac rupees"), 7_500_000)

    def test_does_not_change_non_amount_uses_of_black(self):
        self.assertEqual(normalize_stt_aliases("I own one black car"), "I own one black car")


class UnrecognizedAnswerTests(unittest.TestCase):
    def test_unrecognized_name_is_repeated_instead_of_advancing(self):
        manager = ConversationManager(session_id="retry-test")
        manager.start_conversation()
        prompt = manager.process_user_message("could you repeat that")
        self.assertIsNone(manager.profile.name)
        self.assertEqual(manager.last_unrecognized_question.field_name, "name")
        self.assertIn("sorry, i didn't catch that", prompt.lower())
        self.assertIn("may i have your name", prompt.lower())

    def test_unrecognized_required_field_is_repeated(self):
        manager = ConversationManager(session_id="retry-test")
        manager.state = ConversationState.COLLECTING_INFORMATION
        manager.pending_question = Question("budget", "budget", "Ask what annual premium budget they have in mind.")
        prompt = manager.process_user_message("I am not sure")
        self.assertIsNone(manager.profile.budget)
        self.assertEqual(manager.last_unrecognized_question.field_name, "budget")
        self.assertIn("sorry, i didn't catch that", prompt.lower())

    def test_correction_to_an_earlier_field_does_not_trigger_retry(self):
        manager = ConversationManager(session_id="retry-test")
        manager.state = ConversationState.COLLECTING_INFORMATION
        manager.profile.age = 30
        manager.profile.family_members = 1
        manager.pending_question = Question("smoker", "smoking / tobacco habit", "Ask whether they smoke or use tobacco.")

        prompt = manager.process_user_message("I want a family floater plan")

        self.assertEqual(manager.profile.family_members, 4)
        self.assertIsNone(manager.last_unrecognized_question)
        self.assertNotIn("sorry, i didn't catch that", prompt.lower())
        self.assertIn("smok", prompt.lower())


class EmailTruthfulnessTests(unittest.TestCase):
    def _manager(self):
        manager = ConversationManager(session_id="email-test")
        manager.profile.email = "person@example.com"
        manager.profile.email_confirmed = True
        manager.rec_engine = SimpleNamespace(
            recommend=lambda _profile, limit=3: [SimpleNamespace(policy_id="p1", policy_name="Policy")]
        )
        return manager

    def test_success_and_failure_states_drive_spoken_prompt(self):
        manager = self._manager()
        self.assertTrue(manager.maybe_trigger_email(FakeEmailService(result=True)))
        self.assertEqual(manager.email_state, EmailDeliveryState.SENT)
        manager.state = ConversationState.ENDING_CALL
        self.assertIn("delivery succeeded", manager.build_system_prompt().lower())

        failed = self._manager()
        self.assertFalse(failed.maybe_trigger_email(FakeEmailService(result=False)))
        self.assertEqual(failed.email_state, EmailDeliveryState.FAILED)
        failed.state = ConversationState.ENDING_CALL
        prompt = failed.build_system_prompt().lower()
        self.assertIn("delivery failed", prompt)
        self.assertNotIn("details have been sent", prompt)

    def test_disabled_invalid_and_absent_email_never_claim_success(self):
        disabled = self._manager()
        service = FakeEmailService(configured=False, result=False)
        self.assertFalse(disabled.maybe_trigger_email(service))
        self.assertEqual(disabled.email_state, EmailDeliveryState.DISABLED)

        invalid = self._manager()
        invalid.profile.email = "invalid"
        self.assertFalse(invalid.maybe_trigger_email(FakeEmailService()))
        self.assertEqual(invalid.email_state, EmailDeliveryState.INVALID)

        absent = self._manager()
        absent.profile.email = None
        self.assertFalse(absent.maybe_trigger_email(FakeEmailService()))
        self.assertEqual(absent.email_state, EmailDeliveryState.NOT_REQUESTED)

    def test_email_verification_prompt_formats_spelling(self):
        prompt = _email_verification_instructions("person@example.com")
        self.assertNotIn("{spelled}", prompt)

    def test_split_email_fragments_are_combined_and_confirmed(self):
        manager = ConversationManager(session_id="email-fragments")
        manager._extract_information("My email is test zero six zero four")
        self.assertIsNone(manager.profile.pending_email)

        manager._extract_information("at gmail dot com")
        self.assertEqual(manager.profile.pending_email, "test0604@gmail.com")
        self.assertFalse(manager.profile.email_confirmed)

        manager._extract_information("yes, that is correct")
        self.assertEqual(manager.profile.email, "test0604@gmail.com")
        self.assertTrue(manager.profile.email_confirmed)

    def test_corrected_split_email_replaces_the_pending_address(self):
        manager = ConversationManager(session_id="email-fragments")
        manager.profile.pending_email = "old@example.com"
        manager._extract_information("No, it is")
        manager._extract_information("b h a u zero six zero four")
        manager._extract_information("gmail dot com")
        self.assertEqual(manager.profile.pending_email, "bhau0604@gmail.com")

    def test_provider_introduction_is_not_merged_into_email_username(self):
        manager = ConversationManager(session_id="email-fragments")
        manager.profile.pending_email = "incorrect@example.com"
        manager._extract_information("Gmail is")
        manager._extract_information("b h a u zero six zero four")
        manager._extract_information("at the rate gmail dot com")
        self.assertEqual(manager.profile.pending_email, "bhau0604@gmail.com")


if __name__ == "__main__":
    unittest.main()
