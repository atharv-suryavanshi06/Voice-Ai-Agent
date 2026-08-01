"""
rag_pipeline.py

Orchestrates the Retrieval-Augmented Generation (RAG) pipeline to retrieve
policy context and answer user queries deterministically.
"""

from __future__ import annotations
import os
from typing import Optional

try:
    from google import genai
except ImportError:
    genai = None

from rag.models import RAGResponse

from rag.retriever import PolicyRetriever
from rag.prompt_builder import RAGPromptBuilder
from rag.service import RAGService
from rag.grounding import INSUFFICIENT_EVIDENCE_RESPONSE


class RAGPipeline:
    """
    RAG Pipeline orchestrator that coordinates retrieving relevant document chunks,
    formatting them via a prompt builder, and querying Google's Gemini LLM to generate
    grounded, fact-based answers.
    """

    def __init__(
        self,
        retriever: PolicyRetriever,
        client: Optional[genai.Client] = None,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        model_name: Optional[str] = None,
        top_k: int = 5,
        rag_service: Optional[RAGService] = None,
    ):
        """
        Initializes the RAG Pipeline.

        Args:
            retriever: An initialized PolicyRetriever instance.
            client: Optional Google GenAI client. Reconstructed if None using config.
            prompt_builder: Optional RAGPromptBuilder instance. Defaults to default builder.
            model_name: Name of the Gemini model to use. Defaults to config.GEMINI_MODEL.
            top_k: Default number of relevant chunks to retrieve.
        """
        self.retriever = retriever
        self.rag_service = rag_service or RAGService(retriever, top_k=top_k)
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.top_k = top_k

        # 1. Resolve GenAI Client (Dependency Injection fallback)
        if client:
            self.client = client
        else:
            try:
                from core import config
                api_key = config.GOOGLE_API_KEY
            except ImportError:
                api_key = os.getenv("GOOGLE_API_KEY")
            
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = None

        # 2. Resolve Model Name
        if model_name:
            self.model_name = model_name
        else:
            try:
                from core import config
                self.model_name = config.GEMINI_MODEL
            except ImportError:
                self.model_name = "gemini-3.1-flash-lite"


    def answer_question(
        self,
        question: str,
        policy_id: Optional[str] = None
    ) -> RAGResponse:
        """
        Retrieves relevant policy chunks for the user's question and invokes Gemini
        to generate a grounded answer strictly based on the retrieved context.

        Args:
            question: The user's query or question string.
            policy_id: Optional policy ID to restrict the retriever's search scope.

        Returns:
            A RAGResponse containing the answer, retrieved chunks, and source IDs.
        """
        if not question.strip():
            return RAGResponse(
                answer="Please provide a valid question.",
                retrieved_chunks=[],
                sources=[],
                policy_id=policy_id
            )

        # 1. Retrieve the top K relevant policy chunks
        retrieved_chunks = self.rag_service.retrieve_relevant(
            query=question,
            top_k=self.top_k,
            policy_id=policy_id,
        )

        # 2. Short-circuit: If no context was retrieved, return the fallback message immediately
        # (saves API costs and ensures deterministic fallback)
        if not retrieved_chunks:
            return RAGResponse(
                answer=INSUFFICIENT_EVIDENCE_RESPONSE,
                retrieved_chunks=[],
                sources=[],
                policy_id=policy_id
            )

        # 3. Construct system instructions and user prompts
        system_instruction = self.prompt_builder.DEFAULT_SYSTEM_INSTRUCTION
        user_prompt = self.prompt_builder.build_user_prompt(question, retrieved_chunks)

        if not self.client:
            raise ValueError(
                "Google GenAI client is not initialized. Please set the GOOGLE_API_KEY "
                "environment variable or check your config file."
            )

        # 4. Generate grounded content from Gemini using temperature=0.0
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.0  # Zero temperature for deterministic, fact-grounded responses
                }
            )
            answer = response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer from Gemini: {e}") from e

        # 5. Extract sources
        sources = [c.chunk_id for c in retrieved_chunks]

        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            sources=sources,
            policy_id=policy_id
        )
