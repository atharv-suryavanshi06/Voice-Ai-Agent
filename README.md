# Voice AI Insurance Advisor

Riya is a real-time insurance-policy voice assistant. It supports local microphone/speaker conversations and Twilio phone calls, follows a deterministic qualification flow, answers policy questions from hybrid RAG evidence, recommends policies from a local catalogue, and can optionally persist conversations and email policy documents.

The browser dashboard displays completed caller/assistant messages and live processing activity. PostgreSQL, Gmail SMTP, Twilio, ngrok, and LangSmith are optional integrations.

## Architecture

```text
Microphone or Twilio
  -> Silero VAD
  -> Deepgram STT
  -> ConversationManager and canonical RAG service
  -> Gemini LLM
  -> Cartesia TTS
  -> caller

Side effects:
  -> bounded PostgreSQL persistence worker
  -> bounded LangSmith trace queue
  -> live dashboard WebSocket
  -> SMTP delivery when a confirmed email requires a result
```

Policy retrieval combines Gemini query embeddings, ChromaDB dense retrieval, BM25 sparse retrieval, reciprocal-rank fusion, and FlashRank reranking. Results below `RAG_MIN_RELEVANCE_SCORE` are rejected consistently in production and evaluation.

## Requirements

- Python 3.10+
- Node.js and npm for the dashboard
- PortAudio/PyAudio support for microphone mode
- API credentials for Deepgram, Google Gemini, and Cartesia

Optional services are PostgreSQL, Gmail SMTP, Twilio, ngrok, and LangSmith.

## Backend installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS/Linux, activate with `source .venv/bin/activate` and copy the environment file with `cp .env.example .env`.

Fill in the three required variables in `.env`:

```env
DEEPGRAM_API_KEY=
GOOGLE_API_KEY=
CARTESIA_API_KEY=
```

All other variables in `.env.example` are optional or have documented defaults. Never commit a populated `.env` file.

## Frontend installation

In a second terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`. The default backend event URL is `ws://localhost:8765/events`; override it with `VITE_EVENTS_URL` when required.

## Run microphone mode

The live dashboard event server is enabled by default:

```powershell
python main.py --mode mic
```

Run without the event server when the dashboard is not needed:

```powershell
python main.py --mode mic --no-dashboard
```

## Run Twilio mode

Configure `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_PHONE_NUMBER`. Set `PUBLIC_URL` to an HTTPS endpoint forwarding to `PORT`, or configure `NGROK_AUTHTOKEN` so the launcher can create a tunnel.

```powershell
python main.py --mode twilio --phone +919876543210
```

Twilio uses `/twiml` for call instructions and `/ws` for the bidirectional media stream. The dashboard remains available over `/events`.

## Optional PostgreSQL

Set either `DATABASE_URL` or the individual `POSTGRES_*` variables. When PostgreSQL is unavailable, calls continue in memory.

The application creates/migrates these tables non-destructively:

- `customer_profiles`
- `conversation_sessions`
- `conversation_messages`
- `policy_documents`
- `policy_ingestions`
- `sent_email_logs`

Existing `customer_profiles.conversation_history` JSONB data is preserved and migrated idempotently into append-only message rows. New history reads prefer `conversation_messages` and fall back to legacy JSONB.

## Optional recommendation email

Configure:

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
GMAIL_SENDER_EMAIL=
GMAIL_APP_PASSWORD=
```

The caller must provide and explicitly confirm an email address. The final spoken statement uses the actual delivery state: sent, failed, disabled, pending, invalid, or not requested.

## Optional LangSmith tracing

Configure `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and `LANGSMITH_TRACING=true`. Trace submission uses a bounded background queue and is flushed during normal shutdown.

## Policy ingestion

Run the interactive PDF picker from the repository root:

```powershell
python run_ingestion.py
```

Ingestion extracts and validates metadata, prepares chunks and embeddings, stages ChromaDB and optional PostgreSQL data, atomically publishes the catalogue, and only then activates the new version. Failures are recorded in `ingestion/ingestion_manifest.json`; the previously active version is retained when replacement ingestion fails.

### Markdown-backed candidate index

Validate all active Markdown sources and preview the isolated index without an API call or Chroma write:

```powershell
python -m ingestion.markdown_reindex
```

The dry run requires exactly one UTF-8 Markdown source for every active catalog policy ID. Identity comes from the labeled policy number inside each document, never its filename. Missing, duplicate, unknown, or multi-ID sources abort before embeddings.

After explicitly approving transmission of the complete Markdown policy text to Google for embeddings, build the separate candidate collection:

```powershell
python -m ingestion.markdown_reindex --execute
```

The default candidate is `insurance_policies_candidate_faq_v1`. The builder refuses the configured active name and any pre-existing candidate name, waits until all embeddings succeed before creating Chroma state, and validates row counts, policy IDs, FAQ chunk provenance, recipe hashes, and embedding settings. It never activates the candidate.

## RAG evaluation

```powershell
python evaluation/validate_rag.py
python evaluation/test_relevance_scores.py
python evaluation/validate_rag_60.py `
  --collection insurance_policies_candidate_faq_v1 `
  --baseline-collection insurance_policies
```

The validation suite uses the same canonical retrieval/acceptance service as production. Positive cases validate expected facts and source policies against the Markdown ground-truth documents. Negative cases require the canonical insufficient-evidence response and zero accepted chunks.

The fixed acceptance command always runs the candidate twice and exits successfully only when both runs score 60/60: policy code and number 9/9, FAQ 45/45, and negative guardrails 6/6. Every positive case enforces the expected policy ID, and candidate average/P95 latency must remain within 20% of the active collection baseline. A timestamped JSON evidence file is written under `evaluation/results/`.

Only after that command succeeds, activate through configuration and restart the process:

```env
RAG_COLLECTION_NAME=insurance_policies_candidate_faq_v1
```

Keep `insurance_policies` for rollback. To roll back, restore `RAG_COLLECTION_NAME=insurance_policies` and restart; do not delete either collection during the canary period. PostgreSQL Markdown synchronization also resolves content by embedded policy ID and accepts a PDF only when its stem exactly matches that verified Markdown source or an exact-ID ingestion recorded it. If that verification is incomplete, document text and attachments for the unverified policy fail closed and are omitted from email.

No fixed latency, zero-hallucination, WER, or accuracy result is guaranteed by this repository. Measure performance in the deployment environment and treat evaluation output as the current result.

## Automated tests

The focused regression suite uses fakes and mocks rather than paid external services:

```powershell
python -m unittest discover -s tests -v
```

## Configuration reference

See `.env.example` for all supported variables and defaults. Application configuration is centralized in `core/config.py`; the canonical Gemini key name is `GOOGLE_API_KEY`.
