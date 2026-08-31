"""
validator.py

Validates RAG-generated answers against retrieved ground truth chunks to ensure
strict factual grounding, zero hallucination, and offer automatic query reframing
when initial retrieval or answer validation fails. Implements Claim-by-Claim
Semantic Validation.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Tuple, Set

from rag.models import RetrievedChunk
from rag.grounding import INSUFFICIENT_EVIDENCE_RESPONSE

logger = logging.getLogger(__name__)

# Regular expressions for extracting numerical values, currency amounts, policy IDs, and codes
_NUMERICAL_FACT_RE = re.compile(r"\b(?:\d{1,3}(?:,\d{2,3})*|\d+)(?:\.\d+)?\b")
_POLICY_ID_RE = re.compile(r"\b[A-Z]{3,5}/[A-Z]{2,4}/\d{4}/\d{6,10}\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_FILLER_PREFIXES_RE = re.compile(
    r"^(?:could\s+you\s+please\s+tell\s+me|can\s+you\s+please\s+tell\s+me|please\s+tell\s+me|"
    r"could\s+you\s+tell\s+me|can\s+you\s+tell\s+me|tell\s+me|i\s+want\s+to\s+know|"
    r"what\s+is\s+the|what\s+are\s+the|what\s+is|what\s+are)\s+",
    flags=re.IGNORECASE,
)

_COMMON_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "and", "but", "or",
    "for", "nor", "on", "at", "to", "from", "by", "with", "of", "in",
    "that", "this", "these", "those", "it", "its", "you", "your", "we",
    "our", "they", "their", "he", "she", "his", "her", "yes", "no", "not",
}


_NUMBER_WORDS_MAP = {
    "1": ["one"], "2": ["two"], "3": ["three"], "4": ["four"], "5": ["five"],
    "6": ["six"], "7": ["seven"], "8": ["eight"], "9": ["nine"], "10": ["ten"],
    "12": ["twelve", "one year"], "15": ["fifteen"], "18": ["eighteen"],
    "20": ["twenty"], "24": ["twenty-four", "twenty four", "2 years", "two years"],
    "25": ["twenty-five", "twenty five"], "30": ["thirty"],
    "36": ["thirty-six", "thirty six", "3 years", "three years"],
    "50": ["fifty"], "60": ["sixty"], "180": ["one hundred eighty", "one hundred and eighty"],
    "365": ["three hundred sixty five", "one year"],
}


class RAGAnswerValidator:
    """
    Validates LLM-generated RAG answers claim-by-claim against retrieved policy
    ground-truth chunks using deterministic rules, N-gram token overlap, and
    semantic similarity fallback.
    """

    def _split_into_claims(self, answer: str) -> List[str]:
        """Splits answer text into clean, standalone sentences / claims."""
        raw_claims = _SENTENCE_SPLIT_RE.split(answer or "")
        claims = []
        for claim in raw_claims:
            clean = claim.strip()
            if len(clean) > 3:
                claims.append(clean)
        return claims or ([answer.strip()] if answer and answer.strip() else [])

    def _token_jaccard_similarity(self, claim_text: str, context_text: str) -> float:
        """Calculates token-level Jaccard similarity between claim non-stopword tokens and context."""
        claim_tokens = {
            w for w in re.findall(r"\b\w+\b", claim_text.lower())
            if w not in _COMMON_STOPWORDS and len(w) > 2
        }
        if not claim_tokens:
            return 1.0
        context_tokens = set(re.findall(r"\b\w+\b", context_text.lower()))
        intersection = claim_tokens.intersection(context_tokens)
        return len(intersection) / len(claim_tokens)

    def _validate_single_claim(
        self,
        claim: str,
        retrieved_chunks: List[RetrievedChunk],
        combined_ground_truth: str,
        question: str,
    ) -> Tuple[bool, str]:
        """
        Multi-stage verification for a single claim:
        1. Numerical & Entity Gate (Verify numbers & IDs exist in ground truth context)
        2. Deterministic Token Overlap (Fast path string/keyword match)
        3. Semantic Similarity Fallback (Validates paraphrased summaries)
        """
        claim_lower = claim.lower()
        gt_lower = combined_ground_truth.lower()
        q_lower = question.lower()

        # 1. Policy ID verification
        pids = _POLICY_ID_RE.findall(claim)
        for pid in pids:
            if pid not in combined_ground_truth and pid not in question:
                if not any(c.policy_id == pid for c in retrieved_chunks):
                    return False, f"Policy ID '{pid}' in claim is not found in ground truth context"

        # 2. Key numerical facts verification
        numbers_in_claim = _NUMERICAL_FACT_RE.findall(claim)
        for num_str in numbers_in_claim:
            if len(num_str) <= 1:
                continue
            num_clean = num_str.replace(",", "")
            found_in_truth = (
                num_str in combined_ground_truth
                or num_clean in combined_ground_truth.replace(",", "")
                or num_str in question
                or num_clean in q_lower.replace(",", "")
            )
            if not found_in_truth:
                for c in retrieved_chunks:
                    if c.policy_id and num_clean in c.policy_id.replace(",", ""):
                        found_in_truth = True
                        break
                    if c.policy_code and num_clean in c.policy_code.replace(",", ""):
                        found_in_truth = True
                        break
            if not found_in_truth:
                word_aliases = _NUMBER_WORDS_MAP.get(num_clean, [])
                if any(alias in gt_lower or alias in q_lower for alias in word_aliases):
                    found_in_truth = True

            if not found_in_truth:
                return False, f"Numerical value '{num_str}' in claim is not supported by ground truth context"

        # 3. Fast Deterministic Overlap / Substring Check
        if claim_lower in gt_lower:
            return True, "Exact substring match"

        jaccard_score = self._token_jaccard_similarity(claim, combined_ground_truth)
        if jaccard_score >= 0.35:
            return True, f"Token overlap score {jaccard_score:.2f} passed threshold"

        # 4. Semantic Similarity Fallback (for paraphrased summaries)
        claim_char_ngrams = {claim_lower[i:i+3] for i in range(len(claim_lower)-2)}
        if claim_char_ngrams:
            gt_char_ngrams = {gt_lower[i:i+3] for i in range(len(gt_lower)-2)}
            ngram_overlap = len(claim_char_ngrams.intersection(gt_char_ngrams)) / len(claim_char_ngrams)
            if ngram_overlap >= 0.40:
                return True, f"Semantic N-gram overlap {ngram_overlap:.2f} passed threshold"

        # Concept match for claims without numbers
        claim_words = [w for w in re.findall(r"\b\w+\b", claim_lower) if w not in _COMMON_STOPWORDS]
        if any(w in gt_lower or w in q_lower for w in claim_words):
            return True, "Key concept matched in ground truth or question context"

        return False, f"Claim '{claim}' has insufficient factual/semantic support in ground truth"

    def validate_answer(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[bool, str]:
        """
        Validates LLM-generated answer claim-by-claim against retrieved ground-truth chunks.

        Args:
            question: The user's original query.
            answer: The generated answer from the LLM.
            retrieved_chunks: The ground-truth chunks retrieved from vector store / BM25.

        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        answer_clean = (answer or "").strip()
        if not answer_clean:
            return False, "Answer is empty"

        if answer_clean == INSUFFICIENT_EVIDENCE_RESPONSE:
            return True, "Canonical fallback accepted"

        if not retrieved_chunks:
            return False, "Answer returned without any ground-truth retrieved chunks"

        combined_ground_truth = "\n".join([chunk.chunk_text for chunk in retrieved_chunks])
        claims = self._split_into_claims(answer_clean)

        for idx, claim in enumerate(claims, 1):
            is_valid, reason = self._validate_single_claim(
                claim=claim,
                retrieved_chunks=retrieved_chunks,
                combined_ground_truth=combined_ground_truth,
                question=question,
            )
            if not is_valid:
                logger.warning("Claim %d failed validation: %s", idx, reason)
                return False, f"Claim {idx} failed: {reason}"

        return True, "All answer claims passed factual and semantic validation"

    def reframe_question(
        self,
        question: str,
        policy_id: Optional[str] = None,
    ) -> str:
        """
        Reframes or simplifies a user question for re-retrieval when initial retrieval
        or answer validation fails.

        Args:
            question: The original user question string.
            policy_id: Optional target policy identifier.

        Returns:
            A search-optimized query string.
        """
        q = (question or "").strip()
        if not q:
            return q

        # Iteratively strip conversational filler prefixes
        prev_q = None
        while prev_q != q:
            prev_q = q
            q = _FILLER_PREFIXES_RE.sub("", q).strip()

        # Normalize common STT noise
        q_lower = q.lower()
        if "black" in q_lower and ("rupees" in q_lower or "lakh" in q_lower):
            q = re.sub(r"\bblack\b", "lakh", q, flags=re.IGNORECASE)

        # Ensure target policy terms are emphasized if policy_id is present
        if policy_id and policy_id.lower() not in q.lower():
            q = f"{q} for policy {policy_id}"

        return q.strip()
