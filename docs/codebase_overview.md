# Pipecat Voice AI Agent: Codebase Overview

This document provides a detailed, simple-to-understand explanation of every file in this project. It explains what each file is made for, what it does, and how it fits into the overall architecture of your Insurance Policy Recommendation Voice Agent.

---

## Core Packages & Structure

### 1. [main.py](file:///d:/Voice-Ai-Agent/main.py)
* **What it is**: The main entry point and coordinator of the entire application.
* **What it does**:
  - Initializes your audio hardware (microphone for input, speaker for output) using the **Pipecat Local Audio Transport**.
  - Sets up the Voice Activity Detection (VAD) system using **Silero VAD** so the bot knows when you start and stop speaking.
  - Wires up the services: **Deepgram** (transcribes your voice to text), **Google Gemini** (analyzes context and generates replies), and **Cartesia** (synthesizes the bot's voice from text).
  - Integrates the **`ConversationManagerProcessor`**, a custom middleware class that intercepts each conversation turn to update the customer's profile, check the state of the call, and rewrite the LLM system prompt dynamically.

### 2. [core/config.py](file:///d:/Voice-Ai-Agent/core/config.py)
* **What it is**: The central configuration manager.
* **What it does**:
  - Loads configuration values and API keys from your environment (populated from `.env`).
  - Runs `validate_api_keys()` on startup, providing a clear error message if keys are missing.
  - Sets default configuration values for models (`gemini-3.1-flash-lite`, `nova-3`, `sonic-2`).

### 3. [core/metrics_tracker.py](file:///d:/Voice-Ai-Agent/core/metrics_tracker.py)
* **What it is**: Telemetry and latency tracking processor.
* **What it does**:
  - Captures real-time TTFB (Time to First Byte) latencies and token/character usage for STT, Gemini LLM, and Cartesia TTS.
  - Generates a latency report upon call completion.

---

## RAG & Recommendation Modules

### 4. `ingestion/`
- [ingestion/pdf_processor.py](file:///d:/Voice-Ai-Agent/ingestion/pdf_processor.py): Orchestrates PDF extraction, metadata parsing, catalog updates, and vector embedding.
- [ingestion/text_extractor.py](file:///d:/Voice-Ai-Agent/ingestion/text_extractor.py): Fast PyMuPDF extraction with PaddleOCR + EasyOCR fallback for scanned PDF pages.
- [ingestion/metadata_extractor.py](file:///d:/Voice-Ai-Agent/ingestion/metadata_extractor.py): 3-Layer document classifier & Gemini anti-hallucination field parser.

### 5. `rag/`
- [rag/vector_store.py](file:///d:/Voice-Ai-Agent/rag/vector_store.py): ChromaDB vector store wrapper.
- [rag/retriever.py](file:///d:/Voice-Ai-Agent/rag/retriever.py): Hybrid search engine (Dense vector search + BM25 keyword search + RRF fusion + FlashRank Cross-Encoder re-ranking).
- [rag/embeddings.py](file:///d:/Voice-Ai-Agent/rag/embeddings.py): Gemini API batch embedding generator (`gemini-embedding-001`).

### 6. `evaluation/`
- [evaluation/validate_rag.py](file:///d:/Voice-Ai-Agent/evaluation/validate_rag.py): Automated RAG validation suite.
- [evaluation/test_relevance_scores.py](file:///d:/Voice-Ai-Agent/evaluation/test_relevance_scores.py): RAG relevance score benchmark script.
