"""Deterministic validation for the same canonical RAG service used in production."""

from __future__ import annotations

import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if sys.platform == "win32" and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rag.grounding import INSUFFICIENT_EVIDENCE_RESPONSE
from rag.rag_pipeline import RAGPipeline
from rag.retriever import PolicyRetriever
from rag.vector_store import PolicyVectorStore


VALIDATION_SUITE: List[Dict[str, Any]] = [
    {
        "id": "TC-01",
        "category": "Fact Retrieval (TrustShield)",
        "query": "What is the policy number and sum insured for TrustShield Health Suraksha?",
        "expected_policy": "TrustShield Health Suraksha",
        "expected_policy_id": "TSG/HP/2026/00562481",
        "policy_id_filter": "TSG/HP/2026/00562481",
        "ground_truth": "TrustShield_Health_Suraksha_Policy_Document.md",
        "expected_groups": [
            ["TSG/HP/2026/00562481"],
            ["50,00,000", "50 lakh", "5000000"],
        ],
        "expect_unaware": False,
    },
    {
        "id": "TC-02",
        "category": "Medical Condition Coverage",
        "query": "Does TrustShield Health Suraksha cover pre-existing diabetes and hypertension?",
        "expected_policy": "TrustShield Health Suraksha",
        "expected_policy_id": "TSG/HP/2026/00562481",
        "policy_id_filter": "TSG/HP/2026/00562481",
        "ground_truth": "TrustShield_Health_Suraksha_Policy_Document.md",
        "expected_groups": [["diabetes", "pre-existing", "disease"], ["hypertension", "pre-existing", "disease"], ["cover", "waiting period", "yes"]],
        "expect_unaware": False,
        "allow_unaware": True,
    },
    {
        "id": "TC-03",
        "category": "Fact Retrieval (ApexCare)",
        "query": "What is the sum insured and premium for ApexCare Elevate Health Plan?",
        "expected_policy": "ApexCare Elevate Health Plan",
        "expected_policy_id": "ACH/EL/2026/00721904",
        "policy_id_filter": "ACH/EL/2026/00721904",
        "ground_truth": "ApexCare_Elevate_Health_Plan_Policy_Document.md",
        "expected_groups": [
            ["75,00,000", "75 lakh", "7500000"],
            ["40,828", "40828"],
        ],
        "expect_unaware": False,
    },
    {
        "id": "TC-04",
        "category": "Fact Retrieval (VitalCare)",
        "query": "What is the maximum age limit for VitalCare Family Health Shield?",
        "expected_policy": "VitalCare Family Health Shield",
        "expected_policy_id": "VCH/FL/2026/00193572",
        "policy_id_filter": "VCH/FL/2026/00193572",
        "ground_truth": "VitalCare_Family_Health_Shield_Policy_Document.md",
        "expected_groups": [["no maximum", "no upper", "lifetime renewability"]],
        "expect_unaware": False,
    },
    {
        "id": "TC-05",
        "category": "Negative Guardrail Test",
        "query": "Does the policy cover damages to my personal electronic smartphone or car accidental repairs?",
        "expected_groups": [],
        "expect_unaware": True,
    },
    {
        "id": "TC-06",
        "category": "Negative Guardrail Test",
        "query": "What is the claim process for Moon Base Alpha space travel insurance?",
        "expected_groups": [],
        "expect_unaware": True,
    },
]


def _catalog_policy_number_cases() -> List[Dict[str, Any]]:
    """Exercise named policy-number retrieval for every currently published policy.

    Markdown ground truth exists for only part of the catalogue. Policy IDs in
    the published catalogue are nevertheless authoritative structured facts,
    so these cases close the coverage gap without inventing missing documents.
    """
    catalog_path = os.path.join(ROOT_DIR, "recommendation", "policy_catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    return [
        {
            "id": f"CAT-{index:02d}",
            "category": "Catalog Policy Number (Voice-style)",
            "query": f"what is the policy number for {policy['policy_name']}",
            "expected_policy": policy["policy_name"],
            "expected_policy_id": str(policy["policy_id"]),
            "policy_id_filter": str(policy["policy_id"]),
            "expected_groups": [[str(policy["policy_id"])]] ,
            "expect_unaware": False,
            "catalog_backed": True,
        }
        for index, policy in enumerate(catalog, 1)
        if policy.get("_ingestion_status", "active") == "active"
    ]


def _ground_truth_text(test_case: Dict[str, Any]) -> str:
    filename = test_case.get("ground_truth")
    if not filename:
        return ""
    path = os.path.join(ROOT_DIR, "Data", filename)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().lower()


def evaluate_response(test_case: Dict[str, Any], response: Any) -> Tuple[bool, str]:
    """Validate facts, accepted evidence source, and insufficient-evidence behavior."""
    answer = (response.answer or "").strip()
    answer_lower = answer.lower()
    chunks = list(response.retrieved_chunks or [])

    if test_case.get("expect_unaware"):
        if answer != INSUFFICIENT_EVIDENCE_RESPONSE:
            return False, f"Expected canonical fallback, received: {answer!r}"
        if chunks:
            return False, "Irrelevant evidence passed the configured relevance threshold"
        return True, "Correct canonical fallback with no accepted evidence"

    if answer == INSUFFICIENT_EVIDENCE_RESPONSE and test_case.get("allow_unaware"):
        return True, "Correct canonical fallback for unlisted condition query"

    if not chunks:
        return False, "No accepted evidence was returned"

    expected_policy_id = str(test_case.get("expected_policy_id", "") or "")
    if expected_policy_id:
        if not any(chunk.policy_id == expected_policy_id for chunk in chunks):
            return False, f"Expected source policy ID was absent: {expected_policy_id}"
        if any(chunk.policy_id != expected_policy_id for chunk in chunks):
            return False, f"Wrong-policy evidence was returned with scoped ID: {expected_policy_id}"
    else:
        # Backward compatibility for ad-hoc external test cases that have not
        # yet adopted exact policy IDs.
        expected_policy = test_case.get("expected_policy", "").lower()
        if expected_policy and not any(expected_policy in chunk.policy_name.lower() for chunk in chunks):
            return False, f"Expected source policy was absent: {test_case['expected_policy']}"

    ground_truth = _ground_truth_text(test_case)
    for group in test_case.get("expected_groups", []):
        lowered = [value.lower() for value in group]
        if ground_truth and not any(value in ground_truth for value in lowered):
            return False, f"Evaluation fact is absent from Markdown ground truth: {group}"
        if not any(value in answer_lower for value in lowered):
            return False, f"Answer missed expected factual group: {group}"

    return True, "Expected facts and source policy were validated"


def run_rag_validation() -> None:
    print("=" * 80)
    print("VOICE AI AGENT - CANONICAL RAG VALIDATION")
    print("=" * 80)

    vector_store = PolicyVectorStore()
    retriever = PolicyRetriever(vector_store=vector_store)
    pipeline = RAGPipeline(retriever=retriever)
    retriever.retrieve("warmup test", top_k=1)

    test_cases = [*VALIDATION_SUITE, *_catalog_policy_number_cases()]
    results = []
    for idx, test_case in enumerate(test_cases):
        if idx > 0:
            time.sleep(3.0)
        start = time.perf_counter()
        response = pipeline.answer_question(
            test_case["query"],
            policy_id=test_case.get("policy_id_filter"),
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        passed, reason = evaluate_response(test_case, response)
        top_score = response.retrieved_chunks[0].similarity_score if response.retrieved_chunks else 0.0
        results.append((test_case["id"], passed, top_score, duration_ms, reason))
        print(f"[{test_case['id']}] {'PASS' if passed else 'FAIL'} - {reason}")

    passed_count = sum(1 for _, passed, *_ in results if passed)
    print("=" * 80)
    print(f"PASSED: {passed_count}/{len(results)}")
    if passed_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    run_rag_validation()
