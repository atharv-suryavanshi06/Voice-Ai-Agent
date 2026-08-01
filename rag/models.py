"""
models.py

Defines the Chunk data model representing a semantic text segment of a policy.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

@dataclass
class Chunk:
    """Represents a single text chunk of an insurance policy."""
    chunk_id: str         # Unique identifier (e.g., "{policy_id}_chunk_{chunk_index}")
    policy_id: str        # Policy identifier from catalog
    policy_name: str      # Name of the policy
    chunk_index: int      # 0-based sequence index
    chunk_text: str       # Extracted text content of the chunk
    embedding: Optional[List[float]] = None  # 3072-dimensional vector embedding
    policy_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Chunk object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Instantiates a Chunk object from a dictionary."""
        return cls(
            chunk_id=str(data["chunk_id"]),
            policy_id=str(data["policy_id"]),
            policy_name=str(data["policy_name"]),
            chunk_index=int(data["chunk_index"]),
            chunk_text=str(data["chunk_text"]),
            embedding=data.get("embedding"),
            policy_code=data.get("policy_code"),
        )


@dataclass
class RetrievedChunk:
    """Represents a retrieved policy chunk with an associated similarity score."""
    chunk_id: str
    policy_id: str
    policy_name: str
    chunk_index: int
    chunk_text: str
    similarity_score: float
    policy_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the RetrievedChunk object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievedChunk":
        """Instantiates a RetrievedChunk object from a dictionary."""
        return cls(
            chunk_id=str(data["chunk_id"]),
            policy_id=str(data["policy_id"]),
            policy_name=str(data["policy_name"]),
            chunk_index=int(data["chunk_index"]),
            chunk_text=str(data["chunk_text"]),
            similarity_score=float(data["similarity_score"]),
            policy_code=data.get("policy_code"),
        )


@dataclass
class RAGResponse:
    """Represents the complete output returned by the RAG Pipeline."""
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    sources: List[str]
    policy_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the RAGResponse object to a dictionary."""
        return {
            "answer": self.answer,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "sources": self.sources,
            "policy_id": self.policy_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGResponse":
        """Instantiates a RAGResponse object from a dictionary."""
        return cls(
            answer=str(data["answer"]),
            retrieved_chunks=[RetrievedChunk.from_dict(c) for c in data["retrieved_chunks"]],
            sources=list(data["sources"]),
            policy_id=data.get("policy_id")
        )
