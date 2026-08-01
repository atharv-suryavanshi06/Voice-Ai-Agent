"""Build a validated Markdown-backed Chroma candidate without touching production.

Running this module without ``--execute`` is a local-only dry run: it validates
source identity and chunking but does not call an embedding API or write Chroma.
An executed build refuses to reuse the configured active collection or any
pre-existing candidate collection. Collection activation is intentionally not
implemented here.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from ingestion.recipe import DEFAULT_EMBEDDING_MODEL, build_ingestion_identity
from ingestion.source_locator import (
    MarkdownPolicySource,
    MarkdownSourceValidationError,
    resolve_markdown_policy_sources,
)
from rag.chunker import SemanticChunker
from rag.models import Chunk
from rag.vector_store import PolicyVectorStore


DEFAULT_CANDIDATE_COLLECTION = "insurance_policies_candidate_faq_v1"
logger = logging.getLogger(__name__)


@dataclass
class CandidatePolicyIndex:
    source: MarkdownPolicySource
    chunks: List[Chunk]
    recipe_hash: str
    recipe: Dict[str, Any]

    @property
    def provenance(self) -> Dict[str, Any]:
        return {
            "source_filename": self.source.path.name,
            "source_hash": self.source.source_hash,
            "source_format": "markdown",
            **self.recipe,
            "ingestion_recipe_hash": self.recipe_hash,
            "policy_code": self.source.policy_code,
        }


@dataclass
class CandidateIndexPlan:
    collection_name: str
    active_collection_name: str
    policies: List[CandidatePolicyIndex]
    embedding_model: str

    @property
    def total_chunks(self) -> int:
        return sum(len(item.chunks) for item in self.policies)

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": "candidate",
            "collection_name": self.collection_name,
            "active_collection_name": self.active_collection_name,
            "embedding_model": self.embedding_model,
            "policy_count": len(self.policies),
            "total_chunks": self.total_chunks,
            "policies": [
                {
                    "policy_id": item.source.policy_id,
                    "policy_code": item.source.policy_code,
                    "source_filename": item.source.path.name,
                    "source_hash": item.source.source_hash,
                    "recipe_hash": item.recipe_hash,
                    "chunk_count": len(item.chunks),
                }
                for item in self.policies
            ],
        }


def load_catalog(catalog_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(catalog_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarkdownSourceValidationError(f"Cannot load policy catalog '{path}': {exc}") from exc
    if not isinstance(payload, list):
        raise MarkdownSourceValidationError("Policy catalog must be a JSON array")
    return [dict(entry) for entry in payload if isinstance(entry, Mapping)]


def validate_candidate_collection_name(candidate_name: str, active_name: str) -> str:
    candidate = candidate_name.strip()
    if not candidate:
        raise ValueError("Candidate collection name cannot be empty")
    if candidate.casefold() == active_name.strip().casefold():
        raise ValueError(
            f"Candidate collection '{candidate}' is the configured active collection; choose a new name"
        )
    return candidate


def create_candidate_plan(
    *,
    data_dir: Union[str, Path],
    catalog_entries: Iterable[Union[str, Mapping[str, Any]]],
    collection_name: str,
    active_collection_name: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunker: Optional[SemanticChunker] = None,
) -> CandidateIndexPlan:
    """Validate sources and prepare chunks without embeddings or Chroma writes."""
    candidate_name = validate_candidate_collection_name(collection_name, active_collection_name)
    effective_chunker = chunker or SemanticChunker()
    sources = resolve_markdown_policy_sources(data_dir, catalog_entries)
    policies: List[CandidatePolicyIndex] = []
    for source in sources.values():
        chunks = effective_chunker.split_text_to_chunks(
            source.text,
            source.policy_id,
            source.policy_name,
        )
        if not chunks:
            raise MarkdownSourceValidationError(f"{source.path.name}: source produced no chunks")
        recipe_hash, recipe = build_ingestion_identity(
            source_hash=source.source_hash,
            source_format="markdown",
            chunker=effective_chunker,
            embedding_model=embedding_model,
        )
        policies.append(CandidatePolicyIndex(source, chunks, recipe_hash, recipe))
    return CandidateIndexPlan(
        collection_name=candidate_name,
        active_collection_name=active_collection_name,
        policies=policies,
        embedding_model=embedding_model,
    )


def validate_written_candidate(
    plan: CandidateIndexPlan,
    collection: Any,
) -> Dict[str, Any]:
    """Read and validate candidate structure without changing collection state."""
    written = collection.get(include=["metadatas"])
    ids = list(written.get("ids") or [])
    metadatas = list(written.get("metadatas") or [])
    if len(ids) != plan.total_chunks or len(metadatas) != plan.total_chunks:
        raise RuntimeError(
            f"Candidate verification failed: expected {plan.total_chunks} rows, found {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise RuntimeError("Candidate verification failed: duplicate physical row IDs")

    expected_ids = {
        f"{chunk.chunk_id}__{policy.recipe_hash[:16]}"
        for policy in plan.policies
        for chunk in policy.chunks
    }
    if set(ids) != expected_ids:
        raise RuntimeError("Candidate verification failed: physical row IDs differ from the plan")

    policy_plan = {policy.source.policy_id: policy for policy in plan.policies}
    expected_by_policy = Counter(
        policy.source.policy_id
        for policy in plan.policies
        for _chunk in policy.chunks
    )
    actual_by_policy: Counter = Counter()
    required_provenance = {
        "source_filename",
        "source_hash",
        "source_format",
        "recipe_version",
        "chunking_version",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "ingestion_recipe_hash",
        "policy_code",
    }
    for raw_metadata in metadatas:
        metadata = dict(raw_metadata or {})
        policy_id = str(metadata.get("policy_id") or "")
        if policy_id not in policy_plan:
            raise RuntimeError(f"Candidate verification failed: unplanned policy ID '{policy_id}'")
        actual_by_policy[policy_id] += 1
        missing = required_provenance - set(metadata)
        if missing:
            raise RuntimeError(
                f"Candidate verification failed: incomplete row metadata "
                f"(missing={sorted(missing)})"
            )
        if metadata.get("ingestion_status") != "active":
            raise RuntimeError("Candidate verification failed: a row is not active")

        policy = policy_plan[policy_id]
        expected_provenance = policy.provenance
        for key in required_provenance:
            if metadata.get(key) != expected_provenance.get(key):
                raise RuntimeError(
                    f"Candidate verification failed: {key} mismatch for policy '{policy_id}'"
                )

    if actual_by_policy != expected_by_policy:
        raise RuntimeError(
            f"Candidate verification failed: policy row counts differ "
            f"(expected {dict(expected_by_policy)}, found {dict(actual_by_policy)})"
        )
    return {
        "row_count": len(ids),
        "policy_count": len(actual_by_policy),
        "policy_row_counts": dict(actual_by_policy),
        "verified": True,
    }


def build_candidate_collection(
    plan: CandidateIndexPlan,
    *,
    db_path: Union[str, Path] = "chroma_db",
    embedding_generator: Optional[Callable[..., List[Chunk]]] = None,
    vector_store_factory: Callable[..., PolicyVectorStore] = PolicyVectorStore,
    collection_exists: Optional[Callable[[str, str], bool]] = None,
) -> Dict[str, Any]:
    """Embed and write a new candidate collection; never activate or delete one."""
    validate_candidate_collection_name(plan.collection_name, plan.active_collection_name)
    db_path_text = str(db_path)
    exists = collection_exists or PolicyVectorStore.collection_exists
    if exists(db_path_text, plan.collection_name):
        raise FileExistsError(
            f"Candidate collection '{plan.collection_name}' already exists; choose a new collection name"
        )

    if embedding_generator is None:
        from rag.embeddings import generate_embeddings
        embedding_generator = generate_embeddings

    all_chunks = [chunk for policy in plan.policies for chunk in policy.chunks]
    embedded = embedding_generator(all_chunks, model_name=plan.embedding_model)
    if embedded is not all_chunks and len(embedded or []) != len(all_chunks):
        raise RuntimeError("Embedding generation returned an unexpected number of chunks")
    if not all_chunks or any(chunk.embedding is None for chunk in all_chunks):
        raise RuntimeError("Embedding generation did not populate every candidate chunk")

    # The collection is created only after every embedding succeeds. Therefore
    # extraction/validation/API failures cannot create or alter a vector store.
    store = vector_store_factory(
        db_path=db_path_text,
        collection_name=plan.collection_name,
        create_only=True,
    )
    try:
        for policy in plan.policies:
            store.insert_chunks(
                policy.chunks,
                provenance=policy.provenance,
                id_suffix=policy.recipe_hash[:16],
            )

        collection = getattr(store, "collection", None)
        if collection is None:
            raise RuntimeError("Candidate verification failed: vector store does not expose its collection")
        structural_validation = validate_written_candidate(plan, collection)
    except Exception:
        # ``create_only=True`` proves this invocation owns the candidate name.
        # Remove only that newly created, incomplete candidate so an identical
        # retry is possible; the configured active collection is never touched.
        client = getattr(store, "client", None)
        if client is not None and hasattr(client, "delete_collection"):
            try:
                client.delete_collection(name=plan.collection_name)
            except Exception:
                logger.exception(
                    "Failed to remove incomplete candidate collection",
                    extra={"collection_name": plan.collection_name},
                )
        raise

    summary = plan.summary()
    summary["written"] = True
    summary["activated"] = False
    summary["structural_validation"] = structural_validation
    summary["verified"] = structural_validation["verified"]
    return summary


def _default_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "Data", root / "recommendation" / "policy_catalog.json", root / "chroma_db"


def main(argv: Optional[Sequence[str]] = None) -> int:
    data_default, catalog_default, db_default = _default_paths()
    try:
        from core import config
        active_collection = config.RAG_COLLECTION_NAME
    except (ImportError, AttributeError):
        active_collection = "insurance_policies"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(data_default))
    parser.add_argument("--catalog", default=str(catalog_default))
    parser.add_argument("--db-path", default=str(db_default))
    parser.add_argument("--candidate-collection", default=DEFAULT_CANDIDATE_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the embedding API and write the new candidate collection. Does not activate it.",
    )
    args = parser.parse_args(argv)

    plan = create_candidate_plan(
        data_dir=args.data_dir,
        catalog_entries=load_catalog(args.catalog),
        collection_name=args.candidate_collection,
        active_collection_name=active_collection,
        embedding_model=args.embedding_model,
    )
    if args.execute:
        result = build_candidate_collection(plan, db_path=args.db_path)
    else:
        result = plan.summary()
        result["written"] = False
        result["activated"] = False
        result["note"] = "Dry run only: no embedding API call and no Chroma write was performed."
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
