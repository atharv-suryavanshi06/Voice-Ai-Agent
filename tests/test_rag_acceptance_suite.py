import unittest
from collections import Counter

from conversation.customer_profile import CustomerProfile
from evaluation.validate_rag_60 import (
    KNOWN_HIGH_RISK_FAQS,
    _has_labeled_identifier,
    build_cases,
)
from recommendation.recommendation_engine import RecommendationEngine


class FixedRAGAcceptanceSuiteTests(unittest.TestCase):
    def test_suite_has_stable_category_counts_and_exact_policy_ids(self):
        cases = build_cases()
        self.assertEqual(len(cases), 60)
        self.assertEqual(
            Counter(case["kind"] for case in cases),
            {"code": 9, "faq": 45, "negative": 6},
        )
        positive = [case for case in cases if case["kind"] != "negative"]
        self.assertTrue(all(case.get("policy_id") for case in positive))
        code_cases = [case for case in cases if case["kind"] == "code"]
        self.assertTrue(all(case.get("policy_code") for case in code_cases))
        self.assertTrue(all("code and policy number" in case["query"] for case in code_cases))
        self.assertEqual(len({case["id"] for case in cases}), 60)

    def test_every_policy_has_one_code_case_and_five_faq_cases(self):
        cases = build_cases()
        code_ids = [case["policy_id"] for case in cases if case["kind"] == "code"]
        faq_ids = [case["policy_id"] for case in cases if case["kind"] == "faq"]
        self.assertEqual(len(code_ids), len(set(code_ids)))
        for policy_id in code_ids:
            self.assertEqual(faq_ids.count(policy_id), 5)

    def test_reconstructed_high_risk_regressions_are_pinned(self):
        cases = build_cases()
        selected = {
            (case["policy_id"], case["source_faq_number"])
            for case in cases
            if case["kind"] == "faq" and case.get("known_high_risk")
        }
        expected = {
            (policy_id, question_number)
            for policy_id, question_numbers in KNOWN_HIGH_RISK_FAQS.items()
            for question_number in question_numbers
        }
        self.assertEqual(len(expected), 11)
        self.assertEqual(selected, expected)

    def test_policy_code_and_number_are_independently_labeled(self):
        policy_number = "SLFH/2026/0518291"
        policy_code = "SLFH-2026"
        number_only = f"The policy number is {policy_number}."
        complete = (
            f"Policy code: {policy_code}. "
            f"Policy number: {policy_number}."
        )
        number_mislabeled_as_code = (
            f"Policy code: {policy_number}. "
            f"Policy number: {policy_number}."
        )
        self.assertFalse(
            _has_labeled_identifier(
                number_only,
                r"(?:policy|product)\s+code|code",
                policy_code,
            )
        )
        self.assertTrue(
            _has_labeled_identifier(
                complete,
                r"(?:policy|product)\s+code|code",
                policy_code,
            )
        )
        self.assertFalse(
            _has_labeled_identifier(
                number_mislabeled_as_code,
                r"(?:policy|product)\s+code|code",
                policy_code,
            )
        )
        self.assertTrue(
            _has_labeled_identifier(
                complete,
                r"policy\s+(?:number|no\.?|id)",
                policy_number,
            )
        )


class RecommendationOrderingRegressionTests(unittest.TestCase):
    def test_fixed_customer_profiles_keep_policy_id_ordering(self):
        engine = RecommendationEngine()
        profiles_and_expected_ids = (
            (
                CustomerProfile(
                    age=35, family_members=4, parents_included=False,
                    children_included=True, existing_diseases=["diabetes"],
                    smoker=False, budget=50_000, coverage_required=1_000_000,
                ),
                [
                    "TSG/HP/2026/00562481",
                    "ACH/EL/2026/00721904",
                    "VCH/FL/2026/00193572",
                ],
            ),
            (
                CustomerProfile(
                    age=30, family_members=1, parents_included=False,
                    children_included=False, existing_diseases=[],
                    smoker=False, budget=20_000, coverage_required=1_000_000,
                ),
                [
                    "SLTP/2026/0417832",
                    "WNH/FP/2026/00378965",
                    "SLI/SIHS/2026/00458231",
                ],
            ),
            (
                CustomerProfile(
                    age=55, family_members=4, parents_included=True,
                    children_included=True, existing_diseases=["hypertension"],
                    smoker=False, budget=40_000, coverage_required=2_000_000,
                ),
                ["TSG/HP/2026/00562481", "VCH/FL/2026/00193572"],
            ),
        )

        for profile, expected_ids in profiles_and_expected_ids:
            with self.subTest(expected_ids=expected_ids):
                self.assertEqual(
                    [policy.policy_id for policy in engine.recommend(profile, limit=3)],
                    expected_ids,
                )


if __name__ == "__main__":
    unittest.main()
