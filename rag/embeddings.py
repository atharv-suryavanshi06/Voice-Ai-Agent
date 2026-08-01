"""
embeddings.py

Implements vector embedding generation for policy text chunks using the
Google Gemini Embedding API and the modern google-genai SDK.
"""

import os
import time
from typing import List
try:
    from google import genai
except ImportError:
    genai = None

from rag.models import Chunk


# Configure Google GenAI Client
try:
    from core import config
    api_key = config.GOOGLE_API_KEY
except ImportError:
    api_key = os.getenv("GOOGLE_API_KEY")

client = None
if api_key and genai is not None:
    client = genai.Client(api_key=api_key)



def generate_embeddings(
    chunks: List[Chunk], 
    model_name: str = "gemini-embedding-001"
) -> List[Chunk]:
    """
    Generates 3072-dimensional vector embeddings for a list of Chunk objects
    using Google's stable Gemini embedding model with batch processing.

    Args:
        chunks: A list of Chunk objects returned by the chunker.
        model_name: Name of the Gemini embedding model to use (default: gemini-embedding-001).

    Returns:
        The same list of Chunk objects, with their `embedding` fields populated.
    """
    if not chunks:
        return []

    if not api_key or not client:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. Please set the GOOGLE_API_KEY environment variable "
            "or populate it in your .env file."
        )

    # Process in batches of 16 for high speed with rate-limit retry protection
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        texts = [c.chunk_text for c in batch_chunks]
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=texts
                )
                if response.embeddings and len(response.embeddings) == len(batch_chunks):
                    for chunk, emb in zip(batch_chunks, response.embeddings):
                        chunk.embedding = emb.values
                    break
                else:
                    # Fallback if response embeddings count does not match batch count
                    for chunk in batch_chunks:
                        res = client.models.embed_content(
                            model=model_name,
                            contents=chunk.chunk_text
                        )
                        if res.embeddings:
                            chunk.embedding = res.embeddings[0].values
                    break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "rate limit" in err_msg.lower():
                    wait_time = 10 * (attempt + 1)
                    print(f"  -> Gemini embedding rate limit hit (429). Retrying in {wait_time}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to generate embeddings for chunk batch {i}-{i+len(batch_chunks)}: {e}") from e

        # Gentle pause between batches to prevent triggering Gemini API RPM rate limits
        time.sleep(1.5)

    return chunks




def generate_query_embedding(
    query: str,
    model_name: str = "gemini-embedding-001",
    metrics_tracker=None,
) -> List[float]:
    """
    Generates a 3072-dimensional vector embedding for a single query string.

    Args:
        query: The raw query string to embed.
        model_name: Name of the Gemini embedding model to use (default: gemini-embedding-001).

    Returns:
        A list of floats representing the embedding vector.
    """
    import time
    if not query.strip():
        raise ValueError("Query string cannot be empty.")

    if not api_key or not client:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. Please set the GOOGLE_API_KEY environment variable "
            "or populate it in your .env file."
        )

    start_time = time.perf_counter()
    try:
        response = client.models.embed_content(
            model=model_name,
            contents=query
        )
        duration_sec = time.perf_counter() - start_time
        try:
            if metrics_tracker is None:
                from core.metrics_tracker import global_metrics_tracker
                metrics_tracker = global_metrics_tracker
            metrics_tracker.record_embedding_call(query=query, duration_sec=duration_sec)

        except Exception:
            pass

        if response.embeddings:
            return response.embeddings[0].values
        else:
            raise ValueError("No embeddings returned for the query.")
    except Exception as e:
        raise RuntimeError(f"Failed to generate query embedding: {e}") from e
