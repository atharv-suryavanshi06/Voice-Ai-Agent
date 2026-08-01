"""
langsmith_tracer.py

Asynchronous, non-blocking LangSmith tracer for real-time Voice AI agent sessions.
Tracks end-to-end turn latency, system prompts, LLM responses, token usage, estimated costs,
and RAG retrieval details (chunk count, chunk IDs, policy IDs, similarity scores).
"""

import os
import time
import math
import logging
import queue
import threading
from typing import Any, Dict, List, Optional
from langsmith import Client
from langsmith.run_trees import RunTree

from core import config

logger = logging.getLogger(__name__)

# Gemini pricing estimates (USD per 1M tokens)
# Gemini 3.1 Flash Lite / 2.0 Flash: $0.075 / 1M prompt tokens, $0.30 / 1M completion tokens
GEMINI_PROMPT_COST_PER_1M = 0.075
GEMINI_COMPLETION_COST_PER_1M = 0.30


def calculate_turn_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates estimated USD cost for a Gemini LLM turn."""
    p_cost = (prompt_tokens / 1_000_000.0) * GEMINI_PROMPT_COST_PER_1M
    c_cost = (completion_tokens / 1_000_000.0) * GEMINI_COMPLETION_COST_PER_1M
    return round(p_cost + c_cost, 8)


class LangSmithTracer:
    """
    Manages non-blocking trace runs to LangSmith.
    Captures prompts, responses, token usage, costs, latencies, and RAG chunk metadata.
    """

    def __init__(self):
        self.api_key = getattr(config, "LANGSMITH_API_KEY", "") or os.getenv("LANGSMITH_API_KEY", "")
        self.project_name = getattr(config, "LANGSMITH_PROJECT", "Voice-AI-Agent") or "Voice-AI-Agent"
        self.enabled = bool(self.api_key and getattr(config, "LANGSMITH_TRACING", True))
        self.client: Optional[Client] = None
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=100)
        self._worker: Optional[threading.Thread] = None

        if self.enabled:
            try:
                self.client = Client(api_key=self.api_key)
                self._worker = threading.Thread(
                    target=self._run_worker,
                    name="langsmith-trace-worker",
                    daemon=True,
                )
                self._worker.start()
                print(f"[LangSmith] Initialized tracer for project '{self.project_name}'.")
            except Exception as e:
                print(f"[LangSmith Warning] Could not initialize Client: {e}")
                self.enabled = False

    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                operation, kwargs = item
                if operation == "retrieval":
                    self._post_retrieval_event(**kwargs)
                elif operation == "voice_turn":
                    self._post_voice_turn(**kwargs)
            except Exception:
                logger.exception("LangSmith background submission failed")
            finally:
                self._queue.task_done()

    def _submit(self, operation: str, kwargs: Dict[str, Any]) -> bool:
        if not self.enabled or not self.client:
            return False
        try:
            self._queue.put_nowait((operation, kwargs))
            return True
        except queue.Full:
            logger.warning(
                "LangSmith trace queue is full; dropping best-effort trace",
                extra={"operation": operation, "session_id": kwargs.get("session_id", "")},
            )
            return False

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for already queued traces without accepting unbounded work."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.02)
        return self._queue.unfinished_tasks == 0

    def log_retrieval_event(
        self,
        query: str,
        retrieved_chunks: List[Any],
        latency_ms: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        self._submit("retrieval", {
            "query": query,
            "retrieved_chunks": list(retrieved_chunks),
            "latency_ms": latency_ms,
            "session_id": session_id,
        })
        return None

    def _post_retrieval_event(
        self,
        query: str,
        retrieved_chunks: List[Any],
        latency_ms: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Logs a RAG hybrid retrieval event to LangSmith.
        Captures query, chunk count, chunk IDs, policy IDs, and similarity scores.
        """
        if not self.enabled or not self.client:
            return None

        try:
            chunk_ids = [getattr(c, "chunk_id", str(i)) for i, c in enumerate(retrieved_chunks)]
            policy_ids = [getattr(c, "policy_id", "unknown") for c in retrieved_chunks]
            similarity_scores = [float(getattr(c, "similarity_score", 0.0)) for c in retrieved_chunks]
            no_of_chunks = len(retrieved_chunks)

            run = RunTree(
                name="rag_hybrid_retrieval",
                run_type="retriever",
                project_name=self.project_name,
                inputs={"query": query},
                outputs={
                    "no_of_chunks_retrieved": no_of_chunks,
                    "chunk_ids": chunk_ids,
                    "policy_ids": policy_ids,
                    "similarity_scores": similarity_scores,
                    "retrieved_chunks_summary": [
                        {
                            "chunk_id": getattr(c, "chunk_id", ""),
                            "policy_id": getattr(c, "policy_id", ""),
                            "policy_name": getattr(c, "policy_name", ""),
                            "similarity_score": getattr(c, "similarity_score", 0.0),
                            "snippet": getattr(c, "chunk_text", "")[:120] + "..." if getattr(c, "chunk_text", None) else ""
                        }
                        for c in retrieved_chunks
                    ]
                },
                extra={
                    "metadata": {
                        "session_id": session_id or "",
                        "latency_ms": round(latency_ms, 2) if latency_ms else 0.0,
                        "no_of_chunks_retrieved": no_of_chunks,
                        "chunk_ids": chunk_ids,
                        "policy_ids": policy_ids,
                        "similarity_scores": similarity_scores,
                    }
                }
            )
            run.post()
            return str(run.id)
        except Exception as e:
            logger.warning(
                "LangSmith retrieval submission failed",
                extra={"session_id": session_id or "", "error_type": type(e).__name__},
            )
            return None

    def log_voice_turn(
        self,
        session_id: str,
        user_message: str,
        system_prompt: str,
        llm_response: str,
        prompt_tokens: int,
        completion_tokens: int,
        end_to_end_latency_ms: float,
        stt_latency_ms: Optional[float] = None,
        llm_ttfb_ms: Optional[float] = None,
        tts_ttfb_ms: Optional[float] = None,
        retrieved_chunks: Optional[List[Any]] = None,
    ) -> Optional[str]:
        self._submit("voice_turn", {
            "session_id": session_id,
            "user_message": user_message,
            "system_prompt": system_prompt,
            "llm_response": llm_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "end_to_end_latency_ms": end_to_end_latency_ms,
            "stt_latency_ms": stt_latency_ms,
            "llm_ttfb_ms": llm_ttfb_ms,
            "tts_ttfb_ms": tts_ttfb_ms,
            "retrieved_chunks": list(retrieved_chunks or []),
        })
        return None

    def _post_voice_turn(
        self,
        session_id: str,
        user_message: str,
        system_prompt: str,
        llm_response: str,
        prompt_tokens: int,
        completion_tokens: int,
        end_to_end_latency_ms: float,
        stt_latency_ms: Optional[float] = None,
        llm_ttfb_ms: Optional[float] = None,
        tts_ttfb_ms: Optional[float] = None,
        retrieved_chunks: Optional[List[Any]] = None,
    ) -> Optional[str]:
        """
        Logs a full end-to-end voice turn run to LangSmith.
        """
        if not self.enabled or not self.client:
            return None

        try:
            total_tokens = prompt_tokens + completion_tokens
            cost_usd = calculate_turn_cost(prompt_tokens, completion_tokens)

            chunk_ids = [getattr(c, "chunk_id", str(i)) for i, c in enumerate(retrieved_chunks)] if retrieved_chunks else []
            policy_ids = [getattr(c, "policy_id", "unknown") for c in retrieved_chunks] if retrieved_chunks else []
            similarity_scores = [float(getattr(c, "similarity_score", 0.0)) for c in retrieved_chunks] if retrieved_chunks else []
            no_of_chunks = len(retrieved_chunks) if retrieved_chunks else 0

            inputs = {
                "user_message": user_message,
                "system_prompt": system_prompt,
            }

            outputs = {
                "llm_response": llm_response,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "estimated_cost_usd": cost_usd,
                "no_of_chunks_retrieved": no_of_chunks,
                "chunk_ids": chunk_ids,
                "policy_ids": policy_ids,
                "similarity_scores": similarity_scores,
            }

            metadata = {
                "session_id": session_id,
                "end_to_end_latency_ms": round(end_to_end_latency_ms, 2),
                "stt_latency_ms": round(stt_latency_ms, 2) if stt_latency_ms else None,
                "llm_ttfb_ms": round(llm_ttfb_ms, 2) if llm_ttfb_ms else None,
                "tts_ttfb_ms": round(tts_ttfb_ms, 2) if tts_ttfb_ms else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": cost_usd,
                "no_of_chunks_retrieved": no_of_chunks,
                "chunk_ids": chunk_ids,
                "policy_ids": policy_ids,
                "similarity_scores": similarity_scores,
            }

            run = RunTree(
                name="voice_agent_turn",
                run_type="chain",
                project_name=self.project_name,
                inputs=inputs,
                outputs=outputs,
                extra={"metadata": metadata}
            )
            run.post()
            return str(run.id)
        except Exception as e:
            logger.warning(
                "LangSmith voice-turn submission failed",
                extra={"session_id": session_id, "error_type": type(e).__name__},
            )
            return None


# Global singleton instance
global_langsmith_tracer = LangSmithTracer()
