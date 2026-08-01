# RAG validation

Run the current validation suite from the repository root:

```powershell
python evaluation/validate_rag.py
```

Production and evaluation use the same `RAGService`, hybrid `PolicyRetriever`, relevance threshold, grounding rules, and insufficient-evidence response.

Positive cases validate:

- expected factual groups in the generated answer;
- the expected policy among accepted source chunks;
- that the expected facts exist in the corresponding Markdown ground-truth document under `Data/`.

Negative cases pass only when no retrieved chunk meets `RAG_MIN_RELEVANCE_SCORE` and the answer is exactly `Sorry, I am unaware of it.` A non-empty retrieval is never sufficient for a pass.

The script reports the observed score and end-to-end duration for the current run. The repository does not claim a fixed accuracy, zero-hallucination rate, WER, or latency result. Rerun validation whenever documents, embeddings, models, prompts, or the relevance threshold change.
