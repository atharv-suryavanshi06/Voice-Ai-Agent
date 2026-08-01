# RAG validation

Run the smoke validation suite from the repository root:

```powershell
python evaluation/validate_rag.py
```

Production and evaluation use the same `RAGService`, hybrid `PolicyRetriever`, relevance threshold, grounding rules, and insufficient-evidence response.

Positive cases validate:

- expected factual groups in the generated answer;
- the expected exact policy ID among accepted source chunks, with no wrong-policy chunk in a scoped query;
- that the expected facts exist in the corresponding Markdown ground-truth document under `Data/`.

Negative cases pass only when no retrieved chunk meets `RAG_MIN_RELEVANCE_SCORE` and the answer is exactly `Sorry, I am unaware of it.` A non-empty retrieval is never sufficient for a pass.

For a candidate collection, run the hard 60-case gate:

```powershell
python evaluation/validate_rag_60.py `
  --collection insurance_policies_candidate_faq_v1 `
  --baseline-collection insurance_policies
```

This sends the evaluation questions, retrieved excerpts, answers, and FAQ references to Gemini. It always performs two consecutive candidate runs and requires 9/9 policy-code-and-number cases, 45/45 FAQs, 6/6 negative guardrails, exact policy IDs, and average/P95 latency within 20% of the active baseline. A failed gate must not be activated.

The scripts report the observed score and end-to-end duration for the current run. The repository does not claim a fixed accuracy, zero-hallucination rate, WER, or latency result. Rerun validation whenever documents, embeddings, models, prompts, or the relevance threshold change.
