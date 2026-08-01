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

## RAG evaluation

```powershell
python evaluation/validate_rag.py
python evaluation/test_relevance_scores.py
```

The validation suite uses the same canonical retrieval/acceptance service as production. Positive cases validate expected facts and source policies against the Markdown ground-truth documents. Negative cases require the canonical insufficient-evidence response and zero accepted chunks.

No fixed latency, zero-hallucination, WER, or accuracy result is guaranteed by this repository. Measure performance in the deployment environment and treat evaluation output as the current result.

## Automated tests

The focused regression suite uses fakes and mocks rather than paid external services:

```powershell
python -m unittest discover -s tests -v
```

## Configuration reference

See `.env.example` for all supported variables and defaults. Application configuration is centralized in `core/config.py`; the canonical Gemini key name is `GOOGLE_API_KEY`.
