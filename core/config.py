"""
config.py

Loads all configuration for the voice agent from environment variables
(populated from a local .env file via python-dotenv). Nothing else in the
app should call os.getenv() directly - it all goes through here.
"""

import os
import sys
import logging

from dotenv import load_dotenv

load_dotenv()

# Suppress verbose google_genai SDK info logs (e.g., AFC max remote calls notices)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)


# --- API keys (required) --------------------------------------------------

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

# --- Deepgram (STT) ---------------------------------------------------------

DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "en")

# --- Google Gemini (LLM) -----------------------------------------------------

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_SYSTEM_PROMPT = os.getenv(
    "GEMINI_SYSTEM_PROMPT",
    "You are a helpful, friendly voice assistant. Keep replies short and "
    "conversational, since they will be spoken out loud.",
)

# Minimum FlashRank score accepted as usable policy evidence. Historical
# irrelevant-query scores in this repository are near zero (~0.0007).
RAG_MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.05"))

# --- Cartesia (TTS) -----------------------------------------------------------

CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-2")
# Default voice is Cartesia's public "British Reading Lady" demo voice.
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121")

# --- PostgreSQL (Database) ---------------------------------------------------

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "voice_ai_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Gmail SMTP (Email Service) -----------------------------------------------

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
GMAIL_SENDER_EMAIL = os.getenv("GMAIL_SENDER_EMAIL", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

# --- Twilio (Telephony Service) ----------------------------------------------

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")
SERVER_PORT = int(os.getenv("PORT", "8765"))

# --- LangSmith (Observability & Tracing) -------------------------------------

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() in ("true", "1", "yes")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "Voice-AI-Agent")

# Ensure environment variables are exposed to langsmith SDK if key is present
if LANGSMITH_API_KEY and LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT



def validate_api_keys() -> None:
    """Fail fast with a clear message if a required API key is missing."""
    required = {
        "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "CARTESIA_API_KEY": CARTESIA_API_KEY,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(
            f"Missing required API key(s) in .env: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in the values.",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_twilio_keys() -> None:
    """Fail fast with a clear message if a required Twilio setting is missing when running in Twilio mode."""
    required = {
        "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
        "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN,
        "TWILIO_PHONE_NUMBER": TWILIO_PHONE_NUMBER,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(
            f"Missing required Twilio configuration in .env: {', '.join(missing)}\n"
            "Please provide your Twilio credentials in .env to use phone service mode.",
            file=sys.stderr,
        )
        sys.exit(1)
