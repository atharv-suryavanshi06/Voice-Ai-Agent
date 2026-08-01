"""Fixed 60-question live acceptance suite for a candidate RAG collection.

The suite intentionally uses the exact policy ID for every positive query. Policy
reference resolution is covered by offline tests; this script measures retrieval,
grounding, guardrails, and answer quality without allowing cross-policy evidence.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if sys.platform == "win32" and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv(ROOT_DIR / ".env")

from rag.grounding import INSUFFICIENT_EVIDENCE_RESPONSE
from rag.rag_pipeline import RAGPipeline
from rag.retriever import PolicyRetriever
from rag.vector_store import PolicyVectorStore


FAQ_PATTERN = re.compile(
    r"\*\*Q(\d+)\.\s*(.*?)\?\*\*\s*\n(.*?)(?=\n\*\*Q\d+\.|\n#{1,6}\s|\Z)",
    re.DOTALL,
)

# Reconstructed from the earlier run summary because that run's raw result
# artifact was not retained.  These are the eleven specifically reported
# weak/high-risk FAQ areas; they are pinned before deterministic fill so every
# future 60-question run keeps exercising them.
KNOWN_HIGH_RISK_FAQS: dict[str, tuple[int, ...]] = {
    "ACH/EL/2026/00721904": (17,),       # exclusions
    "TSG/HP/2026/00562481": (7,),        # pre-existing disease waiting period
    "SLFH/2026/0518291": (12, 17),       # No Claim Bonus; renewal grace
    "SLTP/2026/0417832": (12,),          # monthly premium mode
    "WNH/FP/2026/00378965": (24,),       # Wellness Points expiry
    "SLI/SIHS/2026/00458231": (1, 17),   # claim mode; Free Look Period
    "VCH/FL/2026/00193572": (13,),       # senior-citizen / maximum age
    "SLI/FHS/2026/00792144": (7, 19),    # cashless hospital; Free Look Period
}
NEGATIVE_QUERIES = (
    "Does any policy cover repairs to my mobile phone screen?",
    "Which policy covers a space shuttle engine explosion?",
    "Can I claim insurance for damage to my private car?",
    "Does the plan pay for cryptocurrency investment losses?",
    "What benefit covers a holiday cancellation on Mars?",
    "Can this health policy reimburse restaurant bills?",
)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _has_labeled_identifier(answer: str, label_pattern: str, value: str) -> bool:
    """Require a distinct labeled identifier instead of a normalized substring."""
    clean_answer = re.sub(r"[*_`#]", " ", answer)
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    if not tokens:
        return False
    identifier = r"[^a-z0-9]*".join(re.escape(token) for token in tokens)
    identifier_end = r"(?![a-z0-9]|[/_-][^a-z0-9]*[a-z0-9]|\.[a-z0-9])"
    label_then_value = rf"(?:{label_pattern})\s*(?:\(\s*uin\s*\))?\s*(?::|-|is)?\s*{identifier}{identifier_end}"
    value_then_label = rf"(?<![a-z0-9]){identifier}\s+(?:is\s+)?(?:the\s+)?(?:{label_pattern})(?![a-z0-9])"
    return bool(re.search(label_then_value, clean_answer, re.IGNORECASE)) or bool(
        re.search(value_then_label, clean_answer, re.IGNORECASE)
    )


def _document_index(catalog: list[dict[str, Any]]) -> dict[str, tuple[Path, str]]:
    active_ids = {str(item["policy_id"]) for item in catalog}
    matches: dict[str, list[tuple[Path, str]]] = {policy_id: [] for policy_id in active_ids}
    unknown: list[str] = []
    for path in sorted((ROOT_DIR / "Data").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        embedded = [policy_id for policy_id in active_ids if policy_id in text]
        if len(embedded) == 1:
            matches[embedded[0]].append((path, text))
        elif embedded:
            unknown.append(f"{path.name}: multiple policy IDs {embedded}")
    problems = [*unknown]
    for policy_id, sources in matches.items():
        if len(sources) != 1:
            problems.append(f"{policy_id}: expected one Markdown source, found {len(sources)}")
    if problems:
        raise ValueError("Invalid Markdown corpus: " + "; ".join(problems))
    return {policy_id: sources[0] for policy_id, sources in matches.items()}


def build_cases() -> list[dict[str, Any]]:
    catalog_path = ROOT_DIR / "recommendation" / "policy_catalog.json"
    catalog = [
        item
        for item in json.loads(catalog_path.read_text(encoding="utf-8"))
        if item.get("_ingestion_status", "active") == "active"
    ]
    sources = _document_index(catalog)
    cases: list[dict[str, Any]] = []
    for index, policy in enumerate(catalog, 1):
        policy_id = str(policy["policy_id"])
        policy_name = str(policy["policy_name"])
        policy_code = str(policy.get("policy_code") or "").strip()
        if not policy_code:
            raise ValueError(f"{policy_id}: active catalog entry is missing policy_code")
        cases.append(
            {
                "id": f"CODE-{index:02d}",
                "kind": "code",
                "query": f"What are the policy code and policy number for {policy_name}?",
                "policy_id": policy_id,
                "policy_code": policy_code,
                "policy_name": policy_name,
                "reference": f"Policy code {policy_code}; policy number {policy_id}",
            }
        )
        source_path, source_text = sources[policy_id]
        faq_pairs: list[tuple[int, str, str]] = []
        for question_number, question, answer in FAQ_PATTERN.findall(source_text):
            clean_answer = " ".join(re.sub(r"[*_`#|]", " ", answer).split())
            if len(clean_answer) >= 20:
                faq_pairs.append(
                    (int(question_number), question.strip() + "?", clean_answer[:1200])
                )
        if len(faq_pairs) < 5:
            raise ValueError(f"{source_path.name}: expected at least five FAQ pairs")
        selected: list[int] = []
        faq_index_by_number = {
            question_number: pair_index
            for pair_index, (question_number, _question, _answer) in enumerate(faq_pairs)
        }
        for question_number in KNOWN_HIGH_RISK_FAQS.get(policy_id, ()):
            if question_number not in faq_index_by_number:
                raise ValueError(
                    f"{source_path.name}: pinned high-risk FAQ Q{question_number} is absent"
                )
            selected.append(faq_index_by_number[question_number])
        for candidate in (
            0,
            len(faq_pairs) // 4,
            len(faq_pairs) // 2,
            (3 * len(faq_pairs)) // 4,
            len(faq_pairs) - 1,
        ):
            if len(selected) >= 5:
                break
            if candidate not in selected:
                selected.append(candidate)
        for candidate in range(len(faq_pairs)):
            if len(selected) >= 5:
                break
            if candidate not in selected:
                selected.append(candidate)
        for faq_index, selected_index in enumerate(selected, 1):
            question_number, question, reference = faq_pairs[selected_index]
            cases.append(
                {
                    "id": f"FAQ-{index:02d}-{faq_index}",
                    "kind": "faq",
                    "query": f"For {policy_name}, {question[0].lower() + question[1:]}",
                    "policy_id": policy_id,
                    "policy_name": policy_name,
                    "reference": reference,
                    "source": source_path.name,
                    "source_faq_number": question_number,
                    "known_high_risk": question_number in KNOWN_HIGH_RISK_FAQS.get(policy_id, ()),
                }
            )
    for index, query in enumerate(NEGATIVE_QUERIES, 1):
        cases.append(
            {
                "id": f"NEG-{index:02d}",
                "kind": "negative",
                "query": query,
                "reference": "No supporting policy evidence; canonical fallback required.",
            }
        )
    if len(cases) != 60:
        raise AssertionError(f"Expected 60 cases, built {len(cases)}")
    return cases


def _answer_with_retry(pipeline: RAGPipeline, query: str, policy_id: str | None):
    for attempt in range(8):
        try:
            return pipeline.answer_question(query, policy_id=policy_id)
        except RuntimeError as exc:
            if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            delay = min(30, 6 * (attempt + 1))
            print(f"Gemini rate limit; retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("Gemini quota retries exhausted")


def _judge_faqs(items: list[dict[str, Any]], model: str) -> dict[str, dict[str, Any]]:
    prompt = (
        "Judge each insurance answer against its reference. Pass only when the answer directly "
        "and correctly answers the question, agrees with the reference, is not the unaware "
        "fallback, and all retrieved chunks use the expected policy ID. Return only a JSON array "
        "of objects with id, passed, and a short reason. ITEMS:\n"
        + json.dumps(items, ensure_ascii=False)
    )
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = None
    for attempt in range(8):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.0},
            )
            break
        except Exception as exc:
            if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            time.sleep(min(30, 6 * (attempt + 1)))
    if response is None:
        raise RuntimeError("Gemini judge quota retries exhausted")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.DOTALL)
    return {item["id"]: item for item in json.loads(raw)}


def run_once(collection_name: str, pace_seconds: float) -> dict[str, Any]:
    cases = build_cases()
    store = PolicyVectorStore(collection_name=collection_name)
    retriever = PolicyRetriever(store, enable_tracing=False)
    pipeline = RAGPipeline(retriever=retriever)
    results: list[dict[str, Any]] = []
    judge_items: list[dict[str, Any]] = []
    for position, case in enumerate(cases, 1):
        started = time.perf_counter()
        expected_id = case.get("policy_id")
        response = _answer_with_retry(pipeline, case["query"], expected_id)
        latency_ms = (time.perf_counter() - started) * 1000.0
        chunks = list(response.retrieved_chunks or [])
        exact_sources = bool(chunks) and all(chunk.policy_id == expected_id for chunk in chunks)
        result = {
            **case,
            "answer": response.answer,
            "latency_ms": round(latency_ms, 1),
            "top_score": round(chunks[0].similarity_score, 4) if chunks else 0.0,
            "accepted_chunks": len(chunks),
            "source_policy_ids": sorted({chunk.policy_id for chunk in chunks}),
        }
        if case["kind"] == "code":
            has_number = _has_labeled_identifier(
                response.answer,
                r"policy\s+(?:number|no\.?|id)",
                expected_id,
            )
            has_code = _has_labeled_identifier(
                response.answer,
                r"(?:policy|product)\s+code|code",
                case["policy_code"],
            )
            result["passed"] = exact_sources and has_number and has_code
            result["reason"] = (
                "exact policy code and number present"
                if result["passed"]
                else "wrong source, missing code, or missing policy number"
            )
        elif case["kind"] == "negative":
            result["passed"] = response.answer.strip() == INSUFFICIENT_EVIDENCE_RESPONSE and not chunks
            result["reason"] = "correct fallback" if result["passed"] else "irrelevant evidence accepted"
        else:
            judge_items.append(
                {
                    "id": case["id"],
                    "question": case["query"],
                    "reference": case["reference"],
                    "answer": response.answer,
                    "exact_policy_sources": exact_sources,
                }
            )
        results.append(result)
        print(f"[{position:02d}/60] {case['id']} chunks={len(chunks)}", flush=True)
        if pace_seconds:
            time.sleep(pace_seconds)
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    judged = _judge_faqs(judge_items, model)
    for result in results:
        if result["kind"] == "faq":
            verdict = judged.get(result["id"], {"passed": False, "reason": "judge result missing"})
            exact_sources = bool(result["accepted_chunks"]) and result["source_policy_ids"] == [
                result["policy_id"]
            ]
            has_answer = result["answer"].strip() != INSUFFICIENT_EVIDENCE_RESPONSE
            result["passed"] = exact_sources and has_answer and bool(verdict["passed"])
            if not exact_sources:
                result["reason"] = "wrong-policy or missing retrieved evidence"
            elif not has_answer:
                result["reason"] = "canonical unaware fallback returned for a supported FAQ"
            else:
                result["reason"] = str(verdict["reason"])
    latencies = sorted(result["latency_ms"] for result in results)
    summary: dict[str, Any] = {
        "collection": collection_name,
        "total": len(results),
        "passed": sum(bool(result["passed"]) for result in results),
        "failed": sum(not bool(result["passed"]) for result in results),
        "average_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": latencies[56],
        "by_kind": {},
    }
    for kind in ("code", "faq", "negative"):
        group = [result for result in results if result["kind"] == kind]
        summary["by_kind"][kind] = {
            "passed": sum(bool(result["passed"]) for result in group),
            "total": len(group),
        }
    return {
        "summary": summary,
        "results": results,
        "failures": [result for result in results if not result["passed"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=os.getenv("RAG_COLLECTION_NAME", "insurance_policies"),
        help="Candidate Chroma collection to validate",
    )
    parser.add_argument(
        "--baseline-collection",
        required=True,
        help="Active collection benchmark; candidate average and P95 must stay within 20%%",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        choices=(2,),
        help="Required consecutive candidate runs (fixed at 2)",
    )
    parser.add_argument("--pace-seconds", type=float, default=4.2)
    parser.add_argument(
        "--output",
        help="Optional JSON evidence path (defaults to a timestamped file under evaluation/results)",
    )
    args = parser.parse_args()
    if args.collection.casefold() == args.baseline_collection.casefold():
        parser.error("--collection must differ from --baseline-collection")
    baseline_report = None
    if args.baseline_collection:
        print(f"=== LATENCY BASELINE: {args.baseline_collection} ===")
        baseline_report = run_once(args.baseline_collection, args.pace_seconds)

    reports = []
    for run_number in range(1, args.repeat + 1):
        print(f"=== RUN {run_number}/{args.repeat}: {args.collection} ===")
        reports.append(run_once(args.collection, args.pace_seconds))
    score_pass = len(reports) == 2 and all(
        report["summary"]["passed"] == 60
        and report["summary"]["by_kind"]["code"]["passed"] == 9
        and report["summary"]["by_kind"]["faq"]["passed"] == 45
        and report["summary"]["by_kind"]["negative"]["passed"] == 6
        for report in reports
    )
    latency_pass = False
    if baseline_report is not None:
        baseline_summary = baseline_report["summary"]
        latency_pass = all(
            report["summary"]["average_latency_ms"] <= 1.2 * baseline_summary["average_latency_ms"]
            and report["summary"]["p95_latency_ms"] <= 1.2 * baseline_summary["p95_latency_ms"]
            for report in reports
        )
    hard_pass = score_pass and latency_pass
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline_report,
        "candidate_runs": reports,
        "score_pass": score_pass,
        "latency_pass": latency_pass,
        "hard_pass": hard_pass,
    }
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT_DIR / output_path
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = ROOT_DIR / "evaluation" / "results" / f"rag_60_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": baseline_report["summary"],
                "candidate_runs": [report["summary"] for report in reports],
                "score_pass": score_pass,
                "latency_pass": latency_pass,
                "hard_pass": hard_pass,
                "evidence_file": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not hard_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
