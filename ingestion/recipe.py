"""Deterministic identities for reproducible policy indexing recipes."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple


DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
INGESTION_RECIPE_VERSION = 1


def describe_chunker(chunker: Any) -> Dict[str, Any]:
    """Describe the chunking behavior that materially affects vector rows."""
    version = getattr(chunker, "VERSION", None)
    if not version:
        chunker_type = type(chunker)
        version = f"{chunker_type.__module__}.{chunker_type.__qualname__}"
    return {
        "chunking_version": str(version),
        "chunk_size": getattr(chunker, "chunk_size", None),
        "chunk_overlap": getattr(chunker, "chunk_overlap", None),
    }


def build_ingestion_recipe(
    *,
    source_hash: str,
    source_format: str,
    chunker: Any,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    """Build the canonical recipe whose hash identifies one physical index revision."""
    if not source_hash:
        raise ValueError("source_hash cannot be empty")
    if not source_format:
        raise ValueError("source_format cannot be empty")
    if not embedding_model:
        raise ValueError("embedding_model cannot be empty")
    return {
        "recipe_version": INGESTION_RECIPE_VERSION,
        "source_hash": source_hash.lower(),
        "source_format": source_format.lower(),
        **describe_chunker(chunker),
        "embedding_model": embedding_model,
    }


def hash_ingestion_recipe(recipe: Dict[str, Any]) -> str:
    """Hash a recipe using stable JSON encoding."""
    encoded = json.dumps(
        recipe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_ingestion_identity(**kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    """Return ``(ingestion_id, recipe)`` for the supplied source and settings."""
    recipe = build_ingestion_recipe(**kwargs)
    return hash_ingestion_recipe(recipe), recipe
