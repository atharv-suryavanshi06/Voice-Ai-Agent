"""
retriever.py

Implements the Hybrid Search and Re-ranking information retrieval layer for RAG.
Combines Dense Vector Search (ChromaDB + Gemini Embeddings) and Sparse Keyword Search (BM25)
via Reciprocal Rank Fusion (RRF), followed by FlashRank Cross-Encoder Re-ranking.
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, List, Optional, Set


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from rag.models import RetrievedChunk


from rag.embeddings import generate_query_embedding
from rag.vector_store import PolicyVectorStore
from rag.reranker import PolicyReranker


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
        self._active_policy_ids = self._load_active_policy_ids()

    @staticmethod
    def _load_active_policy_ids() -> Set[str]:
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "recommendation",
            "policy_catalog.json",
        )
        try:
            with open(catalog_path, "r", encoding="utf-8") as handle:
                catalog = json.load(handle)
            return {
                str(policy["policy_id"])
                for policy in catalog
                if policy.get("_ingestion_status", "active") == "active"
                and policy.get("policy_id")
            }
        except (OSError, ValueError, TypeError, KeyError):
            # A failed catalogue read must not make the retrieval service
            # unavailable. In that exceptional case preserve legacy behavior.
            return set()

    def _is_active_catalog_policy(self, policy_id: str) -> bool:
        return not self._active_policy_ids or policy_id in self._active_policy_ids

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
                similarity_score=0.0
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
                similarity_score=similarity_score
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
                similarity_score=float(score)
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
                similarity_score=round(rrf_scores[cid], 6)
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

        # Step 1: Retrieve candidate lists from Dense Vector and BM25
        candidate_k = self.candidate_k + 10 if self._needs_broader_evidence(query) else self.candidate_k
        vector_candidates = self._get_vector_candidates(query, candidate_k, policy_id_filter)
        bm25_candidates = self._get_bm25_candidates(query, candidate_k, policy_id_filter)

        # Step 2: Combine via Reciprocal Rank Fusion (RRF)
        fused_candidates = self._reciprocal_rank_fusion(vector_candidates, bm25_candidates)

        if not fused_candidates:
            return []

        # Step 3: Re-rank fused candidates using FlashRank Cross-Encoder
        final_reranked = self.reranker.rerank(query, fused_candidates, top_k=top_k)
        final_reranked = [
            chunk for chunk in final_reranked
            if chunk.similarity_score >= self.min_relevance_score
        ]
        duration_ms = (time.perf_counter() - start_t) * 1000.0

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
