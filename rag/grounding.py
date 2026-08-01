"""Canonical grounding rules shared by production and evaluation."""

from __future__ import annotations

from typing import Iterable

from rag.models import RetrievedChunk


INSUFFICIENT_EVIDENCE_RESPONSE = "Sorry, I am unaware of it."

GROUNDING_RULES = (
    "Answer using only the retrieved policy context. Do not use external knowledge or guess. "
    f"If the answer cannot be determined from that context, respond exactly: "
    f"'{INSUFFICIENT_EVIDENCE_RESPONSE}'"
)


def format_retrieved_context(chunks: Iterable[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        policy_code = chunk.policy_code or "not provided"
        blocks.append(
            f"--- Source Chunk ID: {chunk.chunk_id} ---\n"
            f"Policy: {chunk.policy_name}\n"
            f"Policy Code: {policy_code}\n"
            f"Policy Number: {chunk.policy_id}\n"
            f"{chunk.chunk_text}\n"
        )
    return "\n".join(blocks)
