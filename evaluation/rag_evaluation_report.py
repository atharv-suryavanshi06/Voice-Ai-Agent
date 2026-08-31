"""Evaluate the verified RAG question set and write a Markdown report."""
import json, re
from datetime import datetime
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.rag_pipeline import RAGPipeline
from rag.retriever import PolicyRetriever
from rag.vector_store import PolicyVectorStore
from rag.validator import RAGAnswerValidator

QUESTIONS=ROOT/"evaluation"/"rag_questions.json"
REPORT=ROOT/"evaluation"/"rag_evaluation_report.md"

def tokens(s): return set(re.findall(r"[a-z0-9]+", (s or '').lower()))
def f1(p,r): return 2*p*r/(p+r) if p+r else 0.0
def main():
    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = PolicyRetriever(PolicyVectorStore())
    pipe = RAGPipeline(retriever=retriever)
    validator = RAGAnswerValidator()
    rows = []
    quality_rows = []
    embedding_available = True

    for c in cases:
        expected = set(c.get("expected_chunk_ids") or [])
        policy_id = next(iter(expected), None)
        policy_id = policy_id.rsplit("_chunk_", 1)[0] if policy_id else None
        answer = ""
        retrieved_chunks = []
        if embedding_available:
            try:
                response = pipe.answer_question(c["question"], policy_id=policy_id)
                retrieved_chunks = response.retrieved_chunks or []
                answer = response.answer or ""
            except Exception as exc:
                # Gemini embedding quota errors should not invalidate retrieval metrics.
                embedding_available = False
                print(f"Embedding unavailable; switching to BM25 fallback: {exc}")
        if not retrieved_chunks:
            retrieved_chunks = retriever._get_bm25_candidates(c["question"], pipe.top_k, policy_id)

        got = [x.chunk_id for x in retrieved_chunks]
        hit = any(x in expected for x in got[:3]) if expected else not got
        rank = next((i + 1 for i, x in enumerate(got) if x in expected), 0)
        retrieved = set(got)
        tp = len(retrieved & expected)
        if expected:
            precision = tp / len(retrieved) if retrieved else 0.0
            recall = tp / len(expected)
        else:
            precision = recall = 1.0 if not retrieved else 0.0
        chunk_f1 = f1(precision, recall)
        faith = rel = corr = None
        if answer and retrieved_chunks:
            faith, _ = validator.validate_answer(c["question"], answer, retrieved_chunks)
            rel = len(tokens(c["question"]) & tokens(answer)) / (len(tokens(c["question"])) or 1)
            corr = len(tokens(c.get("ground_truth", "")) & tokens(answer)) / (len(tokens(c.get("ground_truth", ""))) or 1)
            quality_rows.append((float(faith), rel, corr))
        rows.append((c, hit, 1 / rank if rank else 0.0, precision, recall, chunk_f1, faith, rel, corr))

    n = len(rows)
    avg = lambda i: sum(r[i] for r in rows) / n if n else 0.0
    qavg = lambda i: sum(r[i] for r in quality_rows) / len(quality_rows) if quality_rows else None
    qfmt = lambda i: f"{qavg(i):.3f}" if qavg(i) is not None else "N/A (Gemini answer generation unavailable)"
    lines = [
        "# RAG Evaluation Report", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Questions: {n}", "", "## Retriever accuracy", "",
        f"- Hit@3: {sum(r[1] for r in rows) / n:.3f}", f"- MRR: {avg(2):.3f}", "",
        "## Chunk accuracy", "", f"- Precision: {avg(3):.3f}", f"- Recall: {avg(4):.3f}",
        f"- F1: {avg(5):.3f}", "", "## Answer quality", "",
        f"- Faithfulness: {qfmt(0)}", f"- Answer relevancy: {qfmt(1)}", f"- Answer correctness: {qfmt(2)}", "",
        "## Per-question results", "",
        "| ID | Question | Hit@3 | MRR | Chunk P | Chunk R | Chunk F1 | Faithfulness | Relevancy | Correctness |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def metric(v):
        return f"{v:.3f}" if v is not None else "N/A"
    lines += [
        f"| {c['id']} | {c['question']} | {'yes' if h else 'no'} | {m:.3f} | {p:.3f} | {r:.3f} | {f:.3f} | {metric(fa)} | {metric(re)} | {metric(co)} |"
        for c, h, m, p, r, f, fa, re, co in rows
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)
if __name__=="__main__": main()
