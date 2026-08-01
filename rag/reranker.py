"""
reranker.py

Implements local cross-encoder re-ranking for RAG policy retrieval using FlashRank (ONNX).
Re-ranks candidate chunks based on full query-passage cross-attention scoring.
"""

from __future__ import annotations
import os
from typing import List, Optional

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None
    RerankRequest = None

from rag.models import RetrievedChunk




class PolicyReranker:
    """
    Reranks retrieved policy chunks using a lightweight local ONNX cross-encoder model (FlashRank).
    Extremely fast (~5-10ms) and adds high precision without external API costs.
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2", cache_dir: str = "./.cache"):
        """
        Initializes the FlashRank cross-encoder ranker.

        Args:
            model_name: Name of the FlashRank model to load.
            cache_dir: Directory where the ONNX model is stored/cached.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        """
        Re-ranks candidate RetrievedChunk objects based on cross-encoder similarity with the query.

        Args:
            query: The user's query string.
            candidates: List of candidate RetrievedChunk objects.
            top_k: Number of top re-ranked chunks to return.

        Returns:
            List of RetrievedChunk objects updated with rerank scores and sorted descending.
        """
        if not candidates or not query.strip():
            return candidates[:top_k]

        # Prepare FlashRank passage dicts
        passages = []
        chunk_map = {}
        for idx, chunk in enumerate(candidates):
            chunk_id_str = str(idx)
            passages.append({
                "id": chunk_id_str,
                "text": chunk.chunk_text,
            })
            chunk_map[chunk_id_str] = chunk

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)

        reranked_chunks = []
        for res in results[:top_k]:
            chunk_id_str = res.get("id")
            score = float(res.get("score", 0.0))
            original_chunk = chunk_map[chunk_id_str]

            reranked_chunks.append(RetrievedChunk(
                chunk_id=original_chunk.chunk_id,
                policy_id=original_chunk.policy_id,
                policy_name=original_chunk.policy_name,
                chunk_index=original_chunk.chunk_index,
                chunk_text=original_chunk.chunk_text,
                similarity_score=round(score, 4)
            ))

        return reranked_chunks
