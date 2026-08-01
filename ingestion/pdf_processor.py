"""Consistent, staged policy-document ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Callable, Optional, Tuple

from ingestion.lifecycle import IngestionJournal
from ingestion.metadata_extractor import parse_metadata_from_text
from ingestion.models import PolicyMetadata
from ingestion.text_extractor import extract_text_from_pdf
from rag.chunker import SemanticChunker
from rag.embeddings import generate_embeddings
from rag.vector_store import PolicyVectorStore


def _read_catalog(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def _atomic_write_catalog(path: str, catalog: list) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="policy-catalog-", suffix=".json.tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(catalog, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _staged_catalog(catalog: list, metadata: PolicyMetadata, ingestion_id: str) -> list:
    result = [
        dict(entry) for entry in catalog
        if not (
            entry.get("policy_id") == metadata.policy_id
            and entry.get("_ingestion_status") == "staging"
        )
    ]
    new_entry = metadata.to_dict()
    new_entry["_ingestion_status"] = "staging"
    new_entry["_ingestion_id"] = ingestion_id
    result.append(new_entry)
    return result


def _active_catalog(catalog: list, metadata: PolicyMetadata, ingestion_id: str) -> list:
    result = [dict(entry) for entry in catalog if entry.get("policy_id") != metadata.policy_id]
    new_entry = metadata.to_dict()
    new_entry["_ingestion_status"] = "active"
    new_entry["_ingestion_id"] = ingestion_id
    result.append(new_entry)
    return result


def process_policy_pdf(
    pdf_path: str,
    catalog_path: Optional[str] = None,
    *,
    text_extractor: Callable[[str], str] = extract_text_from_pdf,
    metadata_parser: Callable[[str], PolicyMetadata] = parse_metadata_from_text,
    chunker: Optional[SemanticChunker] = None,
    embedding_generator: Callable[[list], list] = generate_embeddings,
    vector_store: Optional[PolicyVectorStore] = None,
    db_manager: Any = None,
    journal_path: Optional[str] = None,
) -> Tuple[str, PolicyMetadata]:
    """Prepare every store first, then publish one retry-safe policy version."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    catalog_path = catalog_path or os.path.join(base_dir, "recommendation", "policy_catalog.json")
    journal_path = journal_path or os.path.join(base_dir, "ingestion", "ingestion_manifest.json")
    journal = IngestionJournal(journal_path)

    with open(pdf_path, "rb") as handle:
        pdf_bytes = handle.read()
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    ingestion_id = document_hash
    journal.update(
        ingestion_id,
        "pending",
        source_path=os.path.abspath(pdf_path),
        document_hash=document_hash,
    )

    policy_id = "unknown"
    staged_vectors = False
    catalog_published = False
    previous_catalog = []
    active_db = db_manager
    previous_db_document = None
    db_activated = False

    try:
        previous_catalog = _read_catalog(catalog_path)
        print("Step 1: Extracting text from PDF...")
        text = text_extractor(pdf_path)
        if not text or not text.strip():
            raise ValueError("No policy text was extracted")
        journal.update(ingestion_id, "extracted", extracted_characters=len(text))

        print("Step 2: Parsing and validating policy metadata...")
        metadata = metadata_parser(text)
        policy_id = metadata.policy_id
        journal.update(
            ingestion_id,
            "validated",
            policy_id=policy_id,
            policy_name=metadata.policy_name,
        )

        print("Step 3: Preparing semantic chunks and embeddings...")
        effective_chunker = chunker or SemanticChunker()
        chunks = effective_chunker.split_text_to_chunks(text, metadata.policy_id, metadata.policy_name)
        if not chunks:
            raise ValueError("Policy text produced no chunks")
        embedded_chunks = embedding_generator(chunks)
        if not embedded_chunks or any(chunk.embedding is None for chunk in embedded_chunks):
            raise RuntimeError("Embedding generation did not populate every policy chunk")

        effective_vector_store = vector_store or PolicyVectorStore()
        effective_vector_store.stage_policy_chunks(policy_id, embedded_chunks, ingestion_id)
        staged_vectors = True

        if active_db is None:
            from database.db_manager import PostgresDBManager
            active_db = PostgresDBManager()
        if getattr(active_db, "enabled", False):
            previous_db_document = active_db.get_policy_document(policy_id)
            staged = active_db.stage_policy_document(
                ingestion_id=ingestion_id,
                policy_id=policy_id,
                metadata=metadata.to_dict(),
                document_text=text,
                pdf_path=os.path.abspath(pdf_path),
                pdf_bytes=pdf_bytes,
            )
            if not staged:
                raise RuntimeError("PostgreSQL rejected the staged policy document")

        journal.update(ingestion_id, "indexed", policy_id=policy_id, chunk_count=len(embedded_chunks))

        # Publish a hidden staging record first. RecommendationEngine ignores it,
        # so the previous active catalogue version remains visible during commit.
        staged_catalog = _staged_catalog(previous_catalog, metadata, ingestion_id)
        _atomic_write_catalog(catalog_path, staged_catalog)
        catalog_published = True

        effective_vector_store.activate_staged_policy(
            policy_id,
            ingestion_id,
            remove_previous=False,
        )
        if getattr(active_db, "enabled", False):
            if not active_db.activate_staged_policy_document(ingestion_id):
                raise RuntimeError("PostgreSQL failed to activate the staged policy document")
            db_activated = True

        # This final atomic replacement is the production publication point.
        _atomic_write_catalog(catalog_path, _active_catalog(previous_catalog, metadata, ingestion_id))

        # The new version is now complete in every required store. Retire the
        # previous vector version last so a failed replacement never removes it.
        try:
            effective_vector_store.remove_previous_policy_versions(policy_id, ingestion_id)
        except Exception as cleanup_error:
            # The new version is already active everywhere. Retaining an older
            # vector version is recoverable and safer than rolling back a fully
            # published database/catalogue version.
            print(f"Warning: old vector-version cleanup will need retry: {cleanup_error}")
        try:
            journal.update(ingestion_id, "active", policy_id=policy_id, chunk_count=len(embedded_chunks))
        except Exception as journal_error:
            print(f"Warning: policy is active but the ingestion journal could not be finalized: {journal_error}")
        print(f"Policy '{metadata.policy_name}' is active in catalog, vector store, and configured database.")
        return text, metadata

    except Exception as exc:
        error_context = f"{type(exc).__name__}: {exc}"
        if staged_vectors:
            try:
                effective_vector_store.delete_ingestion(ingestion_id)
            except Exception:
                pass
        if catalog_published:
            try:
                _atomic_write_catalog(catalog_path, previous_catalog)
            except Exception:
                pass
        if db_activated and active_db is not None:
            try:
                if previous_db_document:
                    active_db.save_policy_document(
                        policy_id=previous_db_document["policy_id"],
                        policy_name=previous_db_document["policy_name"],
                        document_text=previous_db_document.get("document_text") or "",
                        insurer=previous_db_document.get("insurer"),
                        plan_type=previous_db_document.get("plan_type"),
                        premium=previous_db_document.get("premium"),
                        sum_insured=previous_db_document.get("sum_insured"),
                        pdf_path=previous_db_document.get("pdf_path"),
                        pdf_bytes=previous_db_document.get("pdf_data"),
                    )
                else:
                    active_db.delete_policy_document(policy_id)
            except Exception:
                pass
        if active_db is not None and getattr(active_db, "enabled", False):
            active_db.mark_policy_ingestion_failed(ingestion_id, error_context)
        journal.update(
            ingestion_id,
            "failed",
            policy_id=policy_id,
            error=error_context[:2000],
        )
        raise RuntimeError(f"Policy ingestion failed before activation: {error_context}") from exc
