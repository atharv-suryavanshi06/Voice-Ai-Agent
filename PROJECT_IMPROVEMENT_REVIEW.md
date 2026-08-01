# Project Improvement Review

Generated from a full repository index and knowledge-graph review.

> Historical review note (2026-08-01): the selected real-time pipeline, metrics, RAG/evaluation, email truthfulness, lifecycle, ingestion consistency, append-only persistence, and documentation/configuration items from this review have since been implemented. Use `README.md` and `PROJECT_CONTEXT_FOR_CHATGPT.md` for the current system state; retain this file as the evidence-backed review snapshot.

## Review scope

- Repository: `D:\Voice-Ai-Agent`
- Index mode: Full
- Indexed graph: 777 nodes and 2,248 relationships
- Review type: Whole-system product and engineering review
- Code modifications performed during review: None
- Excluded generated directories: virtual environments, caches, `frontend/node_modules`, and `frontend/dist`

This is not a security audit and is not limited to recently changed files. It evaluates the product purpose, implemented capabilities, runtime architecture, user journeys, disconnected or incomplete functionality, reliability, performance, observability, testing, documentation, and developer experience.

## Current system summary

This repository implements an insurance-focused conversational voice agent named **Riya**. It collects customer requirements, retrieves relevant policy information, recommends policies, optionally emails recommendations, stores customer and conversation data, and streams the conversation and processing activity to a browser dashboard.

The documentation also presents a broader enterprise or white-label voice-agent platform. The implemented product, however, is currently tightly coupled to Indian insurance workflows through its prompts, customer profile, question flow, recommendation rules, policy catalogue, and email templates.

### Runtime architecture

```text
Microphone or Twilio
    -> speech-to-text
    -> ConversationManager
    -> policy retrieval and recommendation
    -> Gemini response
    -> text-to-speech
    -> customer

                    -> PostgreSQL
                    -> LangSmith and metrics
                    -> live frontend WebSocket
                    -> recommendation email
```

### Capability inventory

| Capability | Current implementation | Status |
|---|---|---|
| Microphone calls | Pipecat local audio transport, Silero VAD, Deepgram STT, Gemini LLM, and Cartesia TTS | Implemented |
| Telephone calls | FastAPI and Twilio Media Streams | Implemented |
| Conversation control | Deterministic state machine and ordered question flow | Implemented |
| Live frontend | React chat and activity dashboard over the `/events` WebSocket | Implemented |
| Policy retrieval | ChromaDB, BM25, reciprocal-rank fusion, and FlashRank | Implemented |
| Recommendations | JSON policy catalogue with deterministic filtering and scoring | Implemented with a domain correctness gap |
| Document ingestion | PDF extraction/OCR, Gemini metadata extraction, catalogue updates, and vector ingestion | Implemented |
| Persistence | PostgreSQL profiles, conversation JSONB, policy documents, and email logs | Implemented and optional |
| Email | Gmail SMTP recommendation email | Implemented and optional |
| Observability | Custom metrics and LangSmith tracing | Partially reliable |
| Evaluation | Procedural RAG validation scripts | Present but insufficient |
| Automated testing | Focused unit and integration regression tests | Missing |

## What is already implemented well

- Conversation behaviour is explicit instead of relying entirely on the LLM. `ConversationState`, `ALLOWED_TRANSITIONS`, and `REQUIRED_QUESTIONS` make the customer journey understandable and testable.
- The RAG implementation is sensibly layered. Embeddings, vector search, BM25 retrieval, result fusion, and reranking are separated into focused modules.
- Recommendation filtering is deterministic and explainable.
- The ingestion pipeline includes structured metadata extraction and validation instead of blindly embedding PDF text.
- Microphone and Twilio transports use the same overall conversation-management concept.
- Dashboard publication uses bounded, non-blocking queues, reducing the chance that a slow browser directly stalls audio processing.
- PostgreSQL, email, and LangSmith integrations generally degrade gracefully when configuration is absent.
- Complete conversation history is persisted with the customer profile.
- Frontend chat and activity presentation are connected to backend events.

## Confirmed gaps

### 1. Insurance product categories can be mixed

The question flow and recommendation email describe health insurance, but neither `Policy` nor `PolicyMetadata` contains a product category or line of business. `RecommendationEngine.filter_policies` therefore cannot distinguish health, life, and motor products.

A read-only execution using a health-style customer profile returned `SecureLife Term Protect Plus`, a term-life product, as the recommendation.

### 2. Blocking work runs in the real-time async pipeline

PostgreSQL operations, SMTP delivery, Gemini token counting, and LangSmith `run.post()` calls can execute synchronously from `ConversationManagerProcessor.process_frame`. Slow database or network calls can therefore increase conversational latency.

### 3. Dashboard state is global rather than call-scoped

`live_event_hub` maintains one global event history and subscriber list. Events do not contain a session ID, so simultaneous calls can appear in the same frontend feed.

The frontend also appends replayed history after reconnecting without event IDs or deduplication, which can duplicate previously displayed messages.

### 4. Metrics can be shared and double-counted

Manual STT, LLM, and TTS tracking overlaps with Pipecat-generated metric frames. The shared `global_metrics_tracker` contains mutable timing fields that concurrent Twilio calls can overwrite.

### 5. Heavy services are initialized for every Twilio call

The Twilio WebSocket handler creates PostgreSQL, vector store, retriever, and reranker objects for each call. This can repeat schema synchronization, policy-document scans, and FlashRank initialization during call setup.

### 6. Production and evaluation use different RAG paths

Production invokes `PolicyRetriever.retrieve` from the conversation processor, while evaluation primarily exercises `RAGPipeline.answer_question`.

The evaluator also accepts `matches OR retrieved_count > 0`. Retrieving any document can therefore incorrectly count as a successful result even when the answer is wrong.

### 7. Email confirmation can be inaccurate

The ending prompt tells the customer that policy details were sent even when email is disabled or delivery failed. A separate prompt line can expose the literal text `{spelled}` because it is not formatted as an f-string.

### 8. Twilio shutdown can raise an unrelated `NameError`

`active_session_manager` and `active_db_manager` are assigned only in microphone mode but inspected unconditionally during application shutdown.

### 9. Policy ingestion is not atomic

The catalogue JSON is updated before vector embeddings and PostgreSQL storage finish. A later failure can leave the catalogue, ChromaDB, and PostgreSQL representing different datasets.

### 10. Conversation persistence performs redundant work

A user turn can call `save_to_db` inside `ConversationManager.process_user_message` and again from `ConversationManagerProcessor.process_frame`. Each operation rewrites the complete JSONB conversation array.

### 11. Documentation and dependency declarations have drifted

Confirmed examples include:

- Documentation uses `GEMINI_API_KEY`, while application configuration expects `GOOGLE_API_KEY`.
- Setup instructions reference a missing `.env.example`.
- Evaluation commands omit the `evaluation/` directory.
- Some documentation describes Twilio and the dashboard as future or disconnected work even though both are implemented.
- The README makes WER and latency claims not calculated by the current metrics implementation.
- Directly imported packages such as `pydantic`, `Pillow`, `numpy`, and the OCR fallback are not consistently declared.

### 12. There is no meaningful automated regression suite

The repository contains evaluation scripts but no focused tests for conversation transitions, field parsing, recommendation correctness, persistence, multi-session isolation, email-delivery state, or Twilio endpoints.

## Improvement opportunities

| ID | Classification | User value | Effort | Dependencies |
|---|---|---:|---:|---|
| QW-1 | Confirmed gap | High | Small | None |
| QW-2 | Confirmed gap | Medium | Small | Final configuration decisions |
| QW-3 | Confirmed gap | High | Small | None |
| QW-4 | Confirmed gap | High | Small | Representative evaluation cases |
| HI-1 | Confirmed gap | Very high | Medium | Catalogue migration and re-ingestion |
| HI-2 | Confirmed gap | Very high | Medium to large | Background execution and error policy |
| HI-3 | Confirmed gap | High | Medium | Session identity design |
| HI-4 | Confirmed gap | High | Medium | Shared runtime factory |
| HI-5 | Confirmed gap | High | Medium | Retrieval thresholds and test corpus |
| HI-6 | Confirmed gap | High | Medium | Test adapters and fixtures |
| LT-1 | Confirmed gap | High | Large | Ingestion status and transaction model |
| LT-2 | Confirmed gap plus inferred product opportunity | Medium to high | Medium to large | Message schema and history API |
| LT-3 | Inferred opportunity | Potentially very high | Large | Product decision on white-label scope |

## Quick wins

### QW-1: Correct the application lifecycle

- **Classification:** Confirmed gap
- **User value:** High
- **Effort:** Small
- **Dependencies:** None
- **Evidence:** `main.run_microphone_agent` assigns the active session and database variables, while the `main.py` shutdown block reads them for every runtime mode.
- **Recommendation:** Initialize the variables to `None` at module scope and guard shutdown according to the active runtime mode.
- **Expected result:** Twilio startup or configuration failures will no longer be masked by an unrelated shutdown exception.

### QW-2: Reconcile setup documentation and dependencies

- **Classification:** Confirmed gap
- **User value:** Medium
- **Effort:** Small
- **Dependencies:** Agreement on supported OCR packages and environment variable names
- **Evidence:** README instructions conflict with `core.config.validate_api_keys`; `.env.example` is absent; evaluation scripts are under `evaluation/`; directly imported OCR/image dependencies are not consistently declared.
- **Recommendation:** Add a working `.env.example`, standardize on `GOOGLE_API_KEY`, correct evaluation paths, update Twilio/dashboard descriptions, remove unsupported metric claims, and explicitly declare direct dependencies.
- **Expected result:** More reproducible setup and fewer environment-specific failures.

### QW-3: Make email-related responses truthful

- **Classification:** Confirmed gap
- **User value:** High
- **Effort:** Small
- **Dependencies:** None
- **Evidence:** `ConversationManager.maybe_trigger_email` knows whether delivery succeeded, but `_ending_call_instructions` unconditionally claims that delivery occurred.
- **Recommendation:** Generate the final confirmation from the actual `email_sent` result and correct the missing formatted string in `_email_verification_instructions`.
- **Expected result:** Customers will not be told an email was delivered when it was not.

### QW-4: Make RAG validation fail correctly

- **Classification:** Confirmed gap
- **User value:** High
- **Effort:** Small
- **Dependencies:** Representative policy questions and expected facts
- **Evidence:** `evaluation.validate_rag.run_rag_validation` treats any non-empty retrieval as a successful positive case.
- **Recommendation:** Remove the `retrieved_count > 0` success shortcut. Validate expected facts, permitted citations, relevance thresholds, and explicit fallback behaviour.
- **Expected result:** Evaluation results will measure answer correctness instead of merely detecting retrieved text.

## High-impact improvements

### HI-1: Model insurance product category end to end

- **Classification:** Confirmed gap
- **User value:** Very high
- **Effort:** Medium
- **Dependencies:** Catalogue migration and document re-ingestion
- **Evidence:** `PolicyExtractionSchema` recognizes document types, but the mapped policy metadata drops that information. `RecommendationEngine.filter_policies` filters plan type and profile fields without filtering product category.
- **Recommendation:** Add product category or line of business to the extraction schema, `PolicyMetadata`, catalogue records, `Policy`, customer intent, and recommendation filters.
- **Expected result:** A health-insurance conversation cannot recommend term-life or motor products.

### HI-2: Remove synchronous side effects from the audio hot path

- **Classification:** Confirmed gap
- **User value:** Very high
- **Effort:** Medium to large
- **Dependencies:** Bounded background queues, retry policy, and shutdown flushing
- **Evidence:** `ConversationManagerProcessor.process_frame` directly reaches PostgreSQL saves, SMTP delivery, Gemini token counting, and LangSmith `run.post()`.
- **Recommendation:** Move these operations behind bounded asynchronous workers or `asyncio.to_thread`. Prefer native usage metrics or local estimates over synchronous token-counting requests.
- **Expected result:** Lower and more predictable response latency during database or network slowdown.

### HI-3: Isolate dashboard and metrics state by session

- **Classification:** Confirmed gap
- **User value:** High
- **Effort:** Medium
- **Dependencies:** Session ID and event ID design
- **Evidence:** `LiveEventHub._history`, `LiveEventHub._subscribers`, and `global_metrics_tracker` are global. `frontend/src/App.jsx` appends replayed events without filtering or deduplication.
- **Recommendation:** Add `session_id` and `event_id` to every live event, maintain session-scoped histories and subscriptions, give each call its own metrics tracker, and deduplicate frontend replay.
- **Expected result:** Concurrent conversations will not leak into one another, and reconnecting will not duplicate messages.

### HI-4: Introduce a shared runtime and pipeline factory

- **Classification:** Confirmed gap
- **User value:** High
- **Effort:** Medium
- **Dependencies:** Clear separation between application-wide and per-call dependencies
- **Evidence:** `main.run_microphone_agent` and `server.websocket_endpoint` independently construct similar STT, LLM, TTS, RAG, database, email, and metrics pipelines. The Twilio handler repeats construction for every call.
- **Recommendation:** Build one dependency factory for microphone and Twilio modes. Initialize database pools and migrations, the vector store, and FlashRank at application startup while keeping conversation state per call.
- **Expected result:** Faster call startup, fewer repeated scans, and less configuration drift between transports.

### HI-5: Unify production RAG and evaluation

- **Classification:** Confirmed gap
- **User value:** High
- **Effort:** Medium
- **Dependencies:** Relevance thresholds and a representative evaluation corpus
- **Evidence:** Production calls `PolicyRetriever.retrieve`; evaluation primarily calls `RAGPipeline.answer_question`. Current retrieval returns top candidates without a minimum acceptance threshold.
- **Recommendation:** Route production and evaluation through the same retrieval and grounding service. Add product-category filtering, minimum relevance thresholds, and one canonical insufficient-evidence response.
- **Expected result:** Evaluation will predict real voice-agent behaviour and unrelated documents will be less likely to reach the LLM.

### HI-6: Add a focused automated regression suite

- **Classification:** Confirmed gap
- **User value:** High
- **Effort:** Medium
- **Dependencies:** Mock adapters and deterministic fixtures
- **Evidence:** No pytest or unittest regression suite was found. `evaluation/test_relevance_scores.py` is an external-service evaluation script rather than a unit or integration suite.
- **Recommendation:** Start with deterministic tests for state transitions, parsers, policy filtering, email-delivery state, and persistence. Add integration tests for session-isolated WebSockets and mocked Twilio/STT/LLM/TTS flows.
- **Expected result:** Latency and architecture refactoring can proceed without silently changing customer behaviour.

## Longer-term improvements

### LT-1: Make ingestion staged and recoverable

- **Classification:** Confirmed gap
- **User value:** High
- **Effort:** Large
- **Dependencies:** Stable document identity, ingestion status model, and rollback/retry rules
- **Evidence:** `ingestion.process_policy_pdf` writes catalogue information before embeddings, ChromaDB, and PostgreSQL work complete.
- **Recommendation:** Stage extraction and embeddings first, then publish a new catalogue version atomically. Track states such as `pending`, `indexed`, `failed`, and `active`.
- **Expected result:** Failed ingestion cannot leave recommendation and retrieval stores representing different policy datasets.

### LT-2: Store messages as append-only records and expose call history

- **Classification:** Persistence rewrite is a confirmed gap; the history interface is an inferred product opportunity
- **User value:** Medium to high
- **Effort:** Medium to large
- **Dependencies:** Session/message identifiers and a migration from JSONB histories
- **Evidence:** `PostgresDBManager.save_profile` rewrites the complete conversation JSONB array. `get_conversation_history` has no active frontend consumer.
- **Recommendation:** Store one message or activity event per row with session ID, sequence, role, content, and timestamps. Add a read API and optionally allow the dashboard to browse previous calls.
- **Expected result:** Cheaper writes, safer concurrent updates, easier analytics, and useful operational history.

### LT-3: Decide whether the product is insurance-specific or a vertical platform

- **Classification:** Inferred opportunity
- **User value:** Potentially very high
- **Effort:** Large
- **Dependencies:** Explicit product strategy
- **Evidence:** The README describes multiple enterprise use cases, while `conversation.prompts`, `CustomerProfile`, `QuestionFlow`, `RecommendationEngine`, the policy catalogue, and email templates are insurance-specific.
- **Recommendation:** If white-label support remains a goal, extract profile fields, prompts, catalogue schemas, recommendation rules, and templates behind a domain package. Otherwise, position and document the repository clearly as an insurance agent.
- **Expected result:** Either a credible configurable platform or a clearer and more maintainable insurance product.

## Recommended implementation sequence

1. Fix lifecycle and email-truthfulness bugs.
2. Correct documentation, environment examples, and dependency declarations.
3. Add core deterministic tests before structural refactoring.
4. Add product category and migrate or re-ingest the policy catalogue.
5. Create a shared runtime factory and move heavyweight initialization to application startup.
6. Introduce session-scoped frontend events and metrics.
7. Move database, email, tracing, and token-counting work out of the real-time processing path.
8. Unify production and evaluation RAG and establish relevance thresholds.
9. Make ingestion staged and recoverable.
10. Decide the vertical-platform scope, then normalize message storage and expand the dashboard accordingly.

## Evidence index

### Runtime orchestration

- `main.py`
  - `ConversationManagerProcessor.process_frame`
  - `run_microphone_agent`
  - `main`
- `server.py`
  - `websocket_endpoint`
  - `dashboard_events`

### Conversation behaviour

- `conversation/state.py`
  - `ConversationState`
  - `ALLOWED_TRANSITIONS`
  - `can_transition`
- `conversation/question_flow.py`
  - `REQUIRED_QUESTIONS`
- `conversation/manager.py`
  - `ConversationManager.process_user_message`
  - `ConversationManager.save_to_db`
  - `ConversationManager.maybe_trigger_email`
- `conversation/prompts.py`
  - `_ending_call_instructions`
  - `_email_verification_instructions`

### Recommendation correctness

- `recommendation/engine.py`
  - `RecommendationEngine.filter_policies`
- `recommendation/models.py`
  - `Policy`
- `recommendation/policies.json`
  - `SecureLife Term Protect Plus`

### Retrieval and evaluation

- `rag/retriever.py`
  - `PolicyRetriever.retrieve`
  - `PolicyRetriever._get_bm25_candidates`
- `rag/pipeline.py`
  - `RAGPipeline.answer_question`
- `evaluation/validate_rag.py`
  - `run_rag_validation`

### Persistence

- `database/db_manager.py`
  - `PostgresDBManager.__init__`
  - `PostgresDBManager.init_db`
  - `PostgresDBManager.save_profile`
  - `PostgresDBManager.get_conversation_history`
  - `PostgresDBManager.sync_existing_markdown_documents`

### Dashboard and session handling

- `core/live_events.py`
  - `LiveEventHub`
  - `live_event_hub`
  - `publish_message`
- `frontend/src/App.jsx`
  - WebSocket connection, replay, and event append handling

### Observability and latency

- `core/metrics.py`
  - `MetricsTracker.record_llm_request`
  - `MetricsTracker.record_llm_response`
  - `MetricsTracker.record_metrics_frame`
  - `global_metrics_tracker`
- `observability/langsmith_tracer.py`
  - `log_retrieval_event`
  - `log_voice_turn`

### Ingestion consistency

- `ingestion/process_policy_pdf.py`
  - `process_policy_pdf`
- `ingestion/text_extractor.py`
  - OCR fallback and image-processing imports

### Setup and documentation

- `README.md`
- `requirements.txt`
- `PROJECT_CONTEXT_FOR_CHATGPT.md`
- `frontend/README.md`
- `core/config.py`
  - `validate_api_keys`

## Final assessment

The repository already contains a coherent end-to-end voice-agent product rather than a collection of disconnected prototypes. Its strongest foundations are the explicit conversation state machine, hybrid retrieval stack, deterministic recommendation flow, optional persistence, and functioning live dashboard.

The immediate priority should be correctness and isolation: prevent cross-category recommendations, remove inaccurate email confirmations, isolate concurrent sessions, and ensure evaluation measures the production path. Once those safeguards are covered by deterministic tests, the latency-sensitive pipeline can be refactored safely to reuse heavyweight services and move blocking side effects out of real-time audio processing.
