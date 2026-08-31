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
from rag.validator import RAGAnswerValidator


class RAGPipeline:
    """
    RAG Pipeline orchestrator that coordinates retrieving relevant document chunks,
    formatting them via a prompt builder, querying Google's Gemini LLM to generate
    grounded, fact-based answers, and validating answers against ground truth.
    """

    def __init__(
        self,
        retriever: PolicyRetriever,
        client: Optional[genai.Client] = None,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        validator: Optional[RAGAnswerValidator] = None,
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
            validator: Optional RAGAnswerValidator instance. Defaults to default validator.
            model_name: Name of the Gemini model to use. Defaults to config.GEMINI_MODEL.
            top_k: Default number of relevant chunks to retrieve.
        """
        self.retriever = retriever
        self.rag_service = rag_service or RAGService(retriever, top_k=top_k)
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.validator = validator or RAGAnswerValidator()
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
            
            if api_key and genai is not None:
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
        Retrieves relevant policy chunks for the user's question, invokes Gemini
        to generate a grounded answer, and validates the answer against ground truth.

        Args:
            question: The user's query or question string.
            policy_id: Optional policy ID to restrict the retriever's search scope.

        Returns:
            A RAGResponse containing the answer, retrieved chunks, source IDs, and validation status.
        """
        if not question.strip():
            return RAGResponse(
                answer="Please provide a valid question.",
                retrieved_chunks=[],
                sources=[],
                policy_id=policy_id,
                is_valid=False,
                was_reframed=False,
            )

        was_reframed = False
        query_to_search = question

        # 1. Retrieve the top K relevant policy chunks
        retrieved_chunks = self.rag_service.retrieve_relevant(
            query=query_to_search,
            top_k=self.top_k,
            policy_id=policy_id,
        )

        # 1b. Reframing Retry: If no context retrieved initially, reframe query once
        if not retrieved_chunks:
            reframed_q = self.validator.reframe_question(question, policy_id=policy_id)
            if reframed_q and reframed_q != question.strip():
                query_to_search = reframed_q
                retrieved_chunks = self.rag_service.retrieve_relevant(
                    query=query_to_search,
                    top_k=self.top_k,
                    policy_id=policy_id,
                )
                if retrieved_chunks:
                    was_reframed = True

        # Short-circuit: If still no context retrieved, return canonical fallback
        if not retrieved_chunks:
            return RAGResponse(
                answer=INSUFFICIENT_EVIDENCE_RESPONSE,
                retrieved_chunks=[],
                sources=[],
                policy_id=policy_id,
                is_valid=False,
                was_reframed=was_reframed,
            )

        # 2. Construct system instructions and user prompts
        system_instruction = self.prompt_builder.DEFAULT_SYSTEM_INSTRUCTION
        user_prompt = self.prompt_builder.build_user_prompt(query_to_search, retrieved_chunks)

        if not self.client:
            raise ValueError(
                "Google GenAI client is not initialized. Please set the GOOGLE_API_KEY "
                "environment variable or check your config file."
            )

        # 3. Generate grounded content from Gemini using temperature=0.0 with backoff on rate limits
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
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                import time
                time.sleep(3.5)
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=user_prompt,
                        config={
                            "system_instruction": system_instruction,
                            "temperature": 0.0
                        }
                    )
                    answer = response.text.strip()
                except Exception as retry_e:
                    raise RuntimeError(f"Failed to generate answer from Gemini after retry: {retry_e}") from retry_e
            else:
                raise RuntimeError(f"Failed to generate answer from Gemini: {e}") from e

        # 4. Validate generated answer against retrieved ground truth chunks
        is_valid, _reason = self.validator.validate_answer(
            question=query_to_search,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )

        # 5. Query Reframing Retry Strategy: If validation failed and query was not yet reframed, retry once
        if not is_valid and not was_reframed:
            reframed_q = self.validator.reframe_question(question, policy_id=policy_id)
            if reframed_q and reframed_q != query_to_search:
                retry_chunks = self.rag_service.retrieve_relevant(
                    query=reframed_q,
                    top_k=self.top_k,
                    policy_id=policy_id,
                )
                if retry_chunks:
                    retry_prompt = self.prompt_builder.build_user_prompt(reframed_q, retry_chunks)
                    try:
                        retry_resp = self.client.models.generate_content(
                            model=self.model_name,
                            contents=retry_prompt,
                            config={
                                "system_instruction": system_instruction,
                                "temperature": 0.0
                            }
                        )
                        retry_answer = retry_resp.text.strip()
                        is_valid_retry, _ = self.validator.validate_answer(
                            question=reframed_q,
                            answer=retry_answer,
                            retrieved_chunks=retry_chunks,
                        )
                        if is_valid_retry:
                            return RAGResponse(
                                answer=retry_answer,
                                retrieved_chunks=retry_chunks,
                                sources=[c.chunk_id for c in retry_chunks],
                                policy_id=policy_id,
                                is_valid=True,
                                was_reframed=True,
                            )
                    except Exception:
                        pass

        # 6. Final return: if validation failed after retry, return canonical fallback
        if not is_valid:
            return RAGResponse(
                answer=INSUFFICIENT_EVIDENCE_RESPONSE,
                retrieved_chunks=retrieved_chunks,
                sources=[c.chunk_id for c in retrieved_chunks],
                policy_id=policy_id,
                is_valid=False,
                was_reframed=was_reframed,
            )

        sources = [c.chunk_id for c in retrieved_chunks]
        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            sources=sources,
            policy_id=policy_id,
            is_valid=True,
            was_reframed=was_reframed,
        )
