"""Canonical retrieval and evidence-acceptance service."""

from __future__ import annotations

from typing import List, Optional

from rag.models import RetrievedChunk
from rag.retriever import PolicyRetriever


class RAGService:
    """One production/evaluation entry point for accepted policy evidence."""

    def __init__(self, retriever: PolicyRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def retrieve_relevant(
        self,
        query: str,
        policy_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        return self.retriever.retrieve(
            query=query,
            top_k=top_k or self.top_k,
            policy_id_filter=policy_id,
        )
