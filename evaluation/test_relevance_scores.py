"""
test_relevance_scores.py

Detailed Relevance Score Benchmark Script for ChromaDB Vector Search,
BM25 Keyword Search, Reciprocal Rank Fusion (RRF), and FlashRank Cross-Encoder Re-ranking.
"""

import sys
import os
import io
import time
from typing import List

# Ensure workspace root is in sys.path when running script directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rag.vector_store import PolicyVectorStore
from rag.retriever import PolicyRetriever
from rag.embeddings import generate_query_embedding
from rag.service import RAGService


def benchmark_relevance_scores():
    """Runs detailed multi-stage relevance score analysis across ChromaDB and Reranker."""
    print("=" * 90)
    print("           CHROMADB & RAG RELEVANCE SCORE DETAILED BENCHMARK REPORT           ")
    print("=" * 90)

    vs = PolicyVectorStore()
    retriever = PolicyRetriever(vector_store=vs)
    rag_service = RAGService(retriever, top_k=3)
    retriever.refresh_bm25_index()

    test_queries = [
        {
            "query": "What is the room rent capping and ICU limit in TrustShield Health Suraksha?",
            "category": "Exact Policy Query (TrustShield Room Rent)"
        },
        {
            "query": "What are the pre-existing disease waiting period and coverage for diabetes in ApexCare?",
            "category": "Medical Waiting Period (ApexCare)"
        },
        {
            "query": "How do I claim cashless hospitalization reimbursement in RSBN Mediclaim Suraksha?",
            "category": "Claim Process Query (RSBN Mediclaim)"
        },
        {
            "query": "What is the policy for space shuttle engine explosion and galaxy transport repair?",
            "category": "Irrelevant / Out-of-Domain Query"
        }
    ]

    for item in test_queries:
        query = item["query"]
        category = item["category"]

        print(f"\n[QUERY CATEGORY]: {category}")
        print(f"[SEARCH QUERY]   : '{query}'")
        print("-" * 90)

        # 1. Raw ChromaDB Vector Search
        query_vector = generate_query_embedding(query)
        chroma_res = vs.collection.query(
            query_embeddings=[query_vector],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        vector_chunks = retriever._get_vector_candidates(query, candidate_k=5)
        bm25_chunks = retriever._get_bm25_candidates(query, candidate_k=5)
        fused_chunks = retriever._reciprocal_rank_fusion(vector_candidates=vector_chunks, bm25_candidates=bm25_chunks)
        reranked_chunks = retriever.reranker.rerank(query, candidates=fused_chunks, top_k=3)
        accepted_chunks = rag_service.retrieve_relevant(query, top_k=3)

        print(f"{'Rank':<5} | {'Stage':<18} | {'Chunk ID':<35} | {'Score':<8} | {'Policy Name'}")
        print("-" * 90)

        # Vector Top Chunks
        for idx, c in enumerate(vector_chunks[:3], 1):
            print(f"#{idx:<4} | Vector (Chroma)  | {c.chunk_id:<35} | {c.similarity_score:<8.4f} | {c.policy_name}")

        # BM25 Top Chunks
        for idx, c in enumerate(bm25_chunks[:3], 1):
            print(f"#{idx:<4} | BM25 (Keyword)  | {c.chunk_id:<35} | {c.similarity_score:<8.4f} | {c.policy_name}")

        # Reranked Top Chunks
        for idx, c in enumerate(reranked_chunks[:3], 1):
            print(f"#{idx:<4} | Final CrossEnc | {c.chunk_id:<35} | {c.similarity_score:<8.4f} | {c.policy_name}")

        print("-" * 90)
        if reranked_chunks:
            top_chunk = reranked_chunks[0]
            print(f"[TOP MATCHED TEXT SNIPPET]: {top_chunk.chunk_text[:160]}...\n")
        print(
            f"[CANONICAL ACCEPTANCE]: {len(accepted_chunks)} chunk(s) at threshold "
            f"{retriever.min_relevance_score:.4f}\n"
        )

    print("=" * 90)
    print("                           RELEVANCE SCORE ANALYSIS COMPLETE                           ")
    print("=" * 90)


if __name__ == "__main__":
    benchmark_relevance_scores()
