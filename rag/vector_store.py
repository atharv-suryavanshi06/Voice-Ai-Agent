"""
vector_store.py

Implements the vector database storage layer for RAG using ChromaDB.
Enables inserting, deleting, and updating policy chunks and their embeddings.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional

try:
    import chromadb
except ImportError:
    chromadb = None

from rag.models import Chunk




class PolicyVectorStore:
    """
    Manages persistent storage and querying of policy document chunks and their
    Gemini embeddings using ChromaDB.
    """

    def __init__(
        self,
        db_path: str = "chroma_db",
        collection_name: Optional[str] = None,
        *,
        create_only: bool = False,
    ):
        """
        Initializes the persistent ChromaDB client and loads/creates the collection.

        Args:
            db_path: Path to the persistent database folder.
            collection_name: Name of the collection to get or create. Defaults
                to ``RAG_COLLECTION_NAME`` (``insurance_policies`` when unset).
            create_only: Fail if the named collection already exists. Candidate
                builders use this to close the preflight/create race window.
        """
        if chromadb is None:
            raise RuntimeError("chromadb is required to use PolicyVectorStore")
        if collection_name is None:
            try:
                from core import config
                collection_name = config.RAG_COLLECTION_NAME
            except (ImportError, AttributeError):
                collection_name = os.getenv("RAG_COLLECTION_NAME", "insurance_policies")
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        # Ensure parent directories exist
        os.makedirs(db_path, exist_ok=True)

        self.db_path = os.path.abspath(db_path)
        self.collection_name = collection_name.strip()
        self.client = chromadb.PersistentClient(path=db_path)
        # Note: Since embeddings are computed upstream by Gemini, we set embedding_function to None.
        # This prevents Chroma from attempting to compute embeddings with its default local models.
        collection_factory = (
            self.client.create_collection if create_only else self.client.get_or_create_collection
        )
        self.collection = collection_factory(name=self.collection_name, embedding_function=None)

    @staticmethod
    def collection_exists(db_path: str, collection_name: str) -> bool:
        """Check for a collection without creating the requested collection."""
        if chromadb is None:
            raise RuntimeError("chromadb is required to inspect vector collections")
        if not Path(db_path).is_dir():
            return False
        client = chromadb.PersistentClient(path=db_path)
        from chromadb.errors import NotFoundError
        try:
            client.get_collection(name=collection_name, embedding_function=None)
        except NotFoundError:
            return False
        return True

    @staticmethod
    def _sanitize_provenance(provenance: Optional[Mapping[str, Any]]) -> dict:
        """Return Chroma-compatible scalar provenance metadata."""
        clean = {}
        for key, value in (provenance or {}).items():
            if value is None or value == "":
                continue
            if isinstance(value, Path):
                value = str(value)
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"Unsupported Chroma metadata value for '{key}': {type(value).__name__}")
            clean[str(key)] = value
        return clean

    def insert_chunks(
        self,
        chunks: List[Chunk],
        *,
        provenance: Optional[Mapping[str, Any]] = None,
        id_suffix: Optional[str] = None,
    ) -> None:
        """
        Stores a list of embedded Chunk objects in the collection.
        Uses upsert to prevent duplicate insertions for the same chunk_id.

        Args:
            chunks: List of Chunk objects with populated embeddings.
        """
        if not chunks:
            return

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        common_metadata = self._sanitize_provenance(provenance)
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} is missing its vector embedding.")

            ids.append(f"{chunk.chunk_id}__{id_suffix}" if id_suffix else chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.chunk_text)
            metadata = dict(common_metadata)
            metadata.update({
                "policy_id": chunk.policy_id,
                "policy_name": chunk.policy_name,
                "chunk_index": chunk.chunk_index,
                "ingestion_status": "active",
            })
            metadatas.append(metadata)

        # Upsert adds new items or updates existing items with the same id,
        # naturally preventing duplicate insertions.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def delete_policy_chunks(self, policy_id: str) -> None:
        """
        Deletes all chunks belonging to a specific policy_id from the collection.

        Args:
            policy_id: The ID of the policy to delete.
        """
        self.collection.delete(where={"policy_id": policy_id})

    def update_policy_chunks(self, policy_id: str, chunks: List[Chunk]) -> None:
        """
        Atomically updates a policy's chunks by deleting all existing chunks for
        the policy_id and inserting the new list of embedded chunks.

        Args:
            policy_id: The ID of the policy being updated.
            chunks: The new list of embedded Chunk objects.
        """
        # First, remove any existing chunks for this policy to prevent orphans
        self.delete_policy_chunks(policy_id)
        
        # Then, insert the new chunks
        self.insert_chunks(chunks)

    def stage_policy_chunks(
        self,
        policy_id: str,
        chunks: List[Chunk],
        ingestion_id: str,
        *,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """Upsert a retry-safe vector version that retrieval must not expose yet."""
        if not chunks:
            raise ValueError("Cannot stage a policy without chunks")
        ids = [f"{chunk.chunk_id}__{ingestion_id[:16]}" for chunk in chunks]
        embeddings = []
        documents = []
        metadatas = []
        common_metadata = self._sanitize_provenance(provenance)
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} is missing its vector embedding.")
            embeddings.append(chunk.embedding)
            documents.append(chunk.chunk_text)
            metadata = dict(common_metadata)
            metadata.update({
                "policy_id": policy_id,
                "policy_name": chunk.policy_name,
                "chunk_index": chunk.chunk_index,
                "ingestion_status": "staging",
                "ingestion_id": ingestion_id,
            })
            metadatas.append(metadata)
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return ids

    def activate_staged_policy(
        self,
        policy_id: str,
        ingestion_id: str,
        remove_previous: bool = True,
    ) -> None:
        """Activate a staged version, optionally retaining the prior version."""
        staged = self.collection.get(
            where={"ingestion_id": ingestion_id},
            include=["metadatas"],
        )
        staged_ids = list(staged.get("ids") or [])
        if not staged_ids:
            raise RuntimeError(f"No staged vector chunks found for ingestion '{ingestion_id}'")
        staged_metadata = []
        for metadata in staged.get("metadatas") or []:
            updated = dict(metadata or {})
            updated["ingestion_status"] = "active"
            staged_metadata.append(updated)
        self.collection.update(ids=staged_ids, metadatas=staged_metadata)

        if remove_previous:
            self.remove_previous_policy_versions(policy_id, ingestion_id)

    def remove_previous_policy_versions(self, policy_id: str, ingestion_id: str) -> None:
        all_policy = self.collection.get(where={"policy_id": policy_id}, include=["metadatas"])
        obsolete_ids = []
        for chunk_id, metadata in zip(all_policy.get("ids") or [], all_policy.get("metadatas") or []):
            metadata = metadata or {}
            if metadata.get("ingestion_id") != ingestion_id and metadata.get("ingestion_status", "active") == "active":
                obsolete_ids.append(chunk_id)
        if obsolete_ids:
            self.collection.delete(ids=obsolete_ids)

    def delete_ingestion(self, ingestion_id: str) -> None:
        """Compensating cleanup for a staged or failed vector version."""
        staged = self.collection.get(where={"ingestion_id": ingestion_id}, include=["metadatas"])
        ids = list(staged.get("ids") or [])
        if ids:
            self.collection.delete(ids=ids)
