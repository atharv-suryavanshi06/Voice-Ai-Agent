"""
retriever.py

Implements the Hybrid Search and Re-ranking information retrieval layer for RAG.
Combines Dense Vector Search (ChromaDB + Gemini Embeddings) and Sparse Keyword Search (BM25)
via Reciprocal Rank Fusion (RRF), followed by FlashRank Cross-Encoder Re-ranking.
"""

from __future__ import annotations
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from rag.models import RetrievedChunk


from rag.embeddings import generate_query_embedding
from rag.vector_store import PolicyVectorStore
from rag.reranker import PolicyReranker


logger = logging.getLogger(__name__)


class PolicyRetriever:
    """
    Hybrid Retriever that performs Dense Vector Search + BM25 Sparse Keyword Search,
    fuses candidates via Reciprocal Rank Fusion (RRF), and re-ranks top candidates
    using FlashRank Cross-Encoder for ultra-high precision retrieval.
    """

    def __init__(
        self,
        vector_store: PolicyVectorStore,
        reranker: Optional[PolicyReranker] = None,
        candidate_k: int = 15,
        min_relevance_score: Optional[float] = None,
        metrics_tracker=None,
        enable_tracing: bool = True,
    ):
        """
        Initializes the hybrid retriever.

        Args:
            vector_store: The PolicyVectorStore instance connecting to ChromaDB.
            reranker: Optional PolicyReranker instance. If None, instantiates a default PolicyReranker.
            candidate_k: Number of initial candidates to pull from vector and BM25 search.
        """
        self.vector_store = vector_store
        self.reranker = reranker or PolicyReranker()
        self.candidate_k = candidate_k
        if min_relevance_score is None:
            try:
                from core import config
                min_relevance_score = config.RAG_MIN_RELEVANCE_SCORE
            except Exception:
                min_relevance_score = 0.05
        self.min_relevance_score = float(min_relevance_score)
        self.metrics_tracker = metrics_tracker
        self.enable_tracing = enable_tracing
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_chunks: List[RetrievedChunk] = []
        # The catalogue is the publication authority for policies. Keep older
        # Chroma rows for recovery, but never let unlisted versions compete in
        # normal customer retrieval.
        active_catalog = self._load_active_catalog_entries()
        self._active_policy_ids = {
            str(policy["policy_id"])
            for policy in active_catalog
            if policy.get("policy_id")
        }
        self._policy_identity_by_id = {
            str(policy["policy_id"]): policy
            for policy in active_catalog
            if policy.get("policy_id")
        }

    @staticmethod
    def _load_active_catalog_entries() -> List[Dict[str, Any]]:
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "recommendation",
            "policy_catalog.json",
        )
        try:
            with open(catalog_path, "r", encoding="utf-8") as handle:
                catalog = json.load(handle)
            return [
                dict(policy)
                for policy in catalog
                if policy.get("_ingestion_status", "active") == "active"
                and policy.get("policy_id")
            ]
        except (OSError, ValueError, TypeError, KeyError):
            # A failed catalogue read must not make the retrieval service
            # unavailable. In that exceptional case preserve legacy behavior.
            return []

    @classmethod
    def _load_active_policy_ids(cls) -> Set[str]:
        """Compatibility helper retained for callers and focused tests."""
        return {
            str(policy["policy_id"])
            for policy in cls._load_active_catalog_entries()
            if policy.get("policy_id")
        }

    def _is_active_catalog_policy(self, policy_id: str) -> bool:
        return not self._active_policy_ids or policy_id in self._active_policy_ids

    def _policy_code(self, policy_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        metadata = metadata or {}
        value = metadata.get("policy_code")
        if not value:
            value = getattr(self, "_policy_identity_by_id", {}).get(policy_id, {}).get("policy_code")
        return str(value) if value else None

    @staticmethod
    def _identity_pattern(value: Any) -> Optional[re.Pattern[str]]:
        tokens = re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if not tokens:
            return None
        # Identifiers may be spoken with spaces ("S F H S") or transcribed
        # with different separators, so match punctuation/whitespace flexibly.
        separator = r"[^a-z0-9]*" if any(len(token) <= 4 for token in tokens) else r"[^a-z0-9]+"
        body = separator.join(re.escape(token) for token in tokens)
        return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", flags=re.IGNORECASE)

    def ranking_query_for(self, query: str, policy_id_filter: Optional[str]) -> str:
        """Remove only the resolved policy identity from the ranking query.

        The caller's original question remains untouched and is still used for
        the answer prompt and conversation history.
        """
        if not policy_id_filter:
            return query
        identity = getattr(self, "_policy_identity_by_id", {}).get(str(policy_id_filter))
        if not identity:
            return query

        ranking_query = query
        values = (
            identity.get("policy_name"),
            identity.get("policy_code"),
            identity.get("policy_id"),
        )
        # Remove labelled appositives as one unit (for example
        # "policy code SFHS-2026,").  A genuine question such as "what is the
        # policy code for <name>" has no code value beside the label, so the
        # requested fact remains in the ranking query.
        labelled_values = (
            (r"(?:policy|product)\s+code", identity.get("policy_code")),
            (r"policy\s+(?:number|no\.?)", identity.get("policy_id")),
        )
        for label, value in labelled_values:
            pattern = self._identity_pattern(value)
            if pattern:
                ranking_query = re.sub(
                    rf"\b{label}\s*(?::|is|-)?\s*{pattern.pattern}",
                    " ",
                    ranking_query,
                    flags=re.IGNORECASE,
                )
        for value in values:
            pattern = self._identity_pattern(value)
            if pattern:
                ranking_query = pattern.sub(" ", ranking_query)
        ranking_query = re.sub(r"\b(?:for|of)\s*(?=[,;:])", " ", ranking_query, flags=re.IGNORECASE)
        ranking_query = re.sub(
            r"\b(?:for|of)\s*(?=[?!.]?\s*$)",
            " ",
            ranking_query,
            flags=re.IGNORECASE,
        )
        ranking_query = re.sub(
            r"\b(?:policy|product)\s+(?:code|number)\s*(?=[,;:])",
            " ",
            ranking_query,
            flags=re.IGNORECASE,
        )
        ranking_query = re.sub(r"\s+([,;:?])", r"\1", ranking_query)
        ranking_query = re.sub(r"([,;:])\s*([,;:])", r"\2", ranking_query)
        ranking_query = re.sub(r"\s+", " ", ranking_query).strip(" ,;:-")

        tokens = set(re.findall(r"[a-z0-9]+", ranking_query.casefold()))
        non_detail_tokens = {
            "a", "an", "the", "for", "of", "to", "me", "please", "about",
            "what", "which", "who", "is", "are", "was", "were", "do", "does",
            "can", "could", "would", "will", "you", "tell", "give",
        }
        return ranking_query if tokens - non_detail_tokens else query

    @staticmethod
    def _needs_broader_evidence(query: str) -> bool:
        """Use more *candidates* only for questions that request multiple facts.

        The final prompt remains bounded by ``top_k``. This avoids increasing
        prompt size or normal-turn latency while helping questions such as
        "policy number and sum insured" that may span separate chunks.
        """
        text = query.lower()
        multi_fact_markers = (
            " and ", " as well as ", "both ", "compare", "difference",
            "policy number", "policy code", "claim process",
        )
        return any(marker in text for marker in multi_fact_markers)

    def refresh_bm25_index(self, force: bool = False) -> None:
        """Loads all policy chunks from vector store and builds/refreshes an in-memory BM25 index."""
        data = self.vector_store.collection.get(include=["documents", "metadatas"])
        if not data or not data["ids"]:
            self._bm25_index = None
            self._bm25_chunks = []
            return

        ids = data["ids"]
        # Skip rebuilding if index is already up to date with ChromaDB
        if not force and self._bm25_index is not None and len(self._bm25_chunks) == len(ids):
            return

        documents = data["documents"]
        metadatas = data["metadatas"]

        tokenized_corpus = []
        chunks = []
        for idx in range(len(ids)):
            text = documents[idx]
            metadata = metadatas[idx] or {}
            if metadata.get("ingestion_status", "active") != "active":
                continue
            if not self._is_active_catalog_policy(metadata.get("policy_id", "")):
                continue
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)

            chunks.append(RetrievedChunk(
                chunk_id=ids[idx],
                policy_id=metadata.get("policy_id", ""),
                policy_name=metadata.get("policy_name", ""),
                chunk_index=int(metadata.get("chunk_index", 0)),
                chunk_text=text,
                similarity_score=0.0,
                policy_code=self._policy_code(metadata.get("policy_id", ""), metadata),
            ))

        self._bm25_chunks = chunks
        if tokenized_corpus:
            self._bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self._bm25_index = None


    def _get_vector_candidates(self, query: str, candidate_k: int, policy_id_filter: Optional[str] = None) -> List[RetrievedChunk]:
        """Performs Dense Vector Search in ChromaDB."""
        query_vector = generate_query_embedding(query, metrics_tracker=self.metrics_tracker)
        if policy_id_filter:
            where_filter = {"policy_id": policy_id_filter}
        elif self._active_policy_ids:
            where_filter = {"policy_id": {"$in": sorted(self._active_policy_ids)}}
        else:
            where_filter = None

        # Staged chunks are deliberately invisible. If ingestion is running,
        # ask Chroma for enough extra candidates that staged rows cannot crowd
        # the previously active version out before the Python-side status filter.
        staged = self.vector_store.collection.get(
            where={"ingestion_status": "staging"},
            include=["metadatas"],
        )
        staged_count = len(staged.get("ids") or [])
        collection_count = self.vector_store.collection.count()
        query_k = min(collection_count, candidate_k + staged_count) if collection_count else candidate_k

        results = self.vector_store.collection.query(
            query_embeddings=[query_vector],
            n_results=max(1, query_k),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        chunks = []
        for idx in range(len(ids)):
            metadata = metadatas[idx] or {}
            if metadata.get("ingestion_status", "active") != "active":
                continue
            if not self._is_active_catalog_policy(metadata.get("policy_id", "")):
                continue
            distance = distances[idx]
            similarity_score = max(0.0, 1.0 - distance)

            chunks.append(RetrievedChunk(
                chunk_id=ids[idx],
                policy_id=metadata.get("policy_id", ""),
                policy_name=metadata.get("policy_name", ""),
                chunk_index=int(metadata.get("chunk_index", 0)),
                chunk_text=documents[idx],
                similarity_score=similarity_score,
                policy_code=self._policy_code(metadata.get("policy_id", ""), metadata),
            ))

        return chunks

    def _get_bm25_candidates(self, query: str, candidate_k: int, policy_id_filter: Optional[str] = None) -> List[RetrievedChunk]:
        """Performs Sparse BM25 Keyword Search."""
        self.refresh_bm25_index()


        if self._bm25_index is None or not self._bm25_chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)

        # Pair each chunk with its BM25 score
        scored_pairs = []
        for idx, score in enumerate(scores):
            chunk = self._bm25_chunks[idx]
            if policy_id_filter and chunk.policy_id != policy_id_filter:
                continue
            if score > 0:
                scored_pairs.append((score, chunk))

        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        top_bm25 = []
        for score, chunk in scored_pairs[:candidate_k]:
            top_bm25.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                policy_id=chunk.policy_id,
                policy_name=chunk.policy_name,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text,
                similarity_score=float(score),
                policy_code=chunk.policy_code,
            ))

        return top_bm25

    def _reciprocal_rank_fusion(
        self,
        vector_candidates: List[RetrievedChunk],
        bm25_candidates: List[RetrievedChunk],
        rrf_k: int = 60
    ) -> List[RetrievedChunk]:
        """Combines rankings from vector search and BM25 search using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, RetrievedChunk] = {}

        # 1. Rank vector candidates
        for rank, chunk in enumerate(vector_candidates, 1):
            chunk_lookup[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Rank BM25 candidates
        for rank, chunk in enumerate(bm25_candidates, 1):
            chunk_lookup[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        # Sort fused results descending by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        fused_candidates = []
        for cid in sorted_ids:
            c = chunk_lookup[cid]
            fused_candidates.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                policy_id=c.policy_id,
                policy_name=c.policy_name,
                chunk_index=c.chunk_index,
                chunk_text=c.chunk_text,
                similarity_score=round(rrf_scores[cid], 6),
                policy_code=c.policy_code,
            ))

        return fused_candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        policy_id_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieves top K policy chunks using Hybrid Search + RRF Fusion + Cross-Encoder Re-ranking.

        Args:
            query: The user's question or search query string.
            top_k: Final number of re-ranked chunks to return.
            policy_id_filter: Optional policy ID to filter scope.

        Returns:
            List of top K RetrievedChunk objects sorted by re-ranker precision score.
        """
        start_t = time.perf_counter()
        if not query.strip():
            return []

        # Use a detail-only query for ranking when identity is already exact.
        # The original query remains untouched for generation and tracing.
        ranking_query = self.ranking_query_for(query, policy_id_filter)

        # Step 1: Retrieve candidate lists from Dense Vector and BM25
        candidate_k = self.candidate_k + 10 if self._needs_broader_evidence(ranking_query) else self.candidate_k
        vector_candidates = self._get_vector_candidates(ranking_query, candidate_k, policy_id_filter)
        bm25_candidates = self._get_bm25_candidates(ranking_query, candidate_k, policy_id_filter)

        # Step 2: Combine via Reciprocal Rank Fusion (RRF)
        fused_candidates = self._reciprocal_rank_fusion(vector_candidates, bm25_candidates)

        if not fused_candidates:
            logger.info(
                "Policy retrieval completed",
                extra={
                    "policy_id": policy_id_filter,
                    "candidate_count": 0,
                    "accepted_chunk_ids": [],
                },
            )
            return []

        # Step 3: Re-rank fused candidates using FlashRank Cross-Encoder
        final_reranked = self.reranker.rerank(ranking_query, fused_candidates, top_k=top_k)
        final_reranked = [
            chunk for chunk in final_reranked
            if chunk.similarity_score >= self.min_relevance_score
        ]
        duration_ms = (time.perf_counter() - start_t) * 1000.0
        logger.info(
            "Policy retrieval completed",
            extra={
                "policy_id": policy_id_filter,
                "candidate_count": len(fused_candidates),
                "accepted_chunk_ids": [chunk.chunk_id for chunk in final_reranked],
            },
        )

        # Log retrieval span to LangSmith
        if self.enable_tracing:
            try:
                from observability.langsmith_tracer import global_langsmith_tracer
                global_langsmith_tracer.log_retrieval_event(
                    query=query,
                    retrieved_chunks=final_reranked,
                    latency_ms=duration_ms,
                )
            except Exception:
                pass

        return final_reranked
