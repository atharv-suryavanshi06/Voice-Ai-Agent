"""
prompt_builder.py

Implements prompt formatting templates and configurations for grounding policy question answering
in retrieved context chunks.
"""

from typing import List
from rag.models import RetrievedChunk
from rag.grounding import GROUNDING_RULES, format_retrieved_context


class RAGPromptBuilder:
    """
    Constructs structured system and user prompts for RAG-based question answering,
    ensuring Gemini answers strictly based on the provided context.
    """

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are an insurance policy assistant. Answer accurately and concisely.\n\n"
        f"Ground Rules:\n{GROUNDING_RULES}"
    )

    def build_user_prompt(self, question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """
        Builds the user prompt combining the question and delimited context chunks.

        Args:
            question: The user's search query or question.
            retrieved_chunks: List of RetrievedChunk objects containing relevant text.

        Returns:
            A formatted string prompt for the user instruction.
        """
        if not retrieved_chunks:
            context_text = "No relevant policy documents were retrieved."
        else:
            context_text = format_retrieved_context(retrieved_chunks)

        return (
            f"Retrieved Policy Context:\n"
            f"=========================================\n"
            f"{context_text}\n"
            f"=========================================\n\n"
            f"User Question: {question}\n\n"
            f"Answer:"
        )
