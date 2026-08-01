"""
chunker.py

Implements hierarchical semantic chunking of policy text. Splits documents
by paragraph, sentence, and word boundaries, maintaining a configurable chunk size
and contextual overlap without cutting sentences or headings in half.
"""

import re
from typing import List, Optional
from rag.models import Chunk


CHUNKING_VERSION = "semantic-faq-v1"


class _FAQUnit(str):
    """Internal marker for a question/answer unit that must stay standalone."""


class SemanticChunker:
    """
    Splits policy text into semantically meaningful chunks, preserving
    paragraph/heading boundaries and sentence structures wherever possible.
    """

    VERSION = CHUNKING_VERSION
    _FAQ_QUESTION_RE = re.compile(
        r"(?im)^[ \t]*(?:\*\*)?Q[ \t]*\d+[ \t]*[.)][^\r\n]*(?:\r?\n|$)"
    )
    _MARKDOWN_SECTION_RE = re.compile(
        r"(?m)^(?:[ \t]*---+[ \t]*\r?\n(?:[ \t]*\r?\n)*)?(?=#{1,6}[ \t]+\S)"
    )
    _PLAIN_DEFINITIONS_RE = re.compile(
        r"(?im)^(?:section[ \t]+)?(?:\d+(?:\.\d+)*[.)]?[ \t]+)?definitions?[ \t]*$"
    )

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the chunker.

        Args:
            chunk_size: Maximum character length of each chunk.
            chunk_overlap: Maximum character overlap between consecutive chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_plain_text_into_units(self, text: str, limit: Optional[int] = None) -> List[str]:
        """
        Apply the original hierarchical splitter to text that is not an FAQ.

        Each unit is either under the chunk_size limit or split down to individual words.
        Separators are preserved in the units so that the original text can be
        reconstructed exactly.
        """
        effective_limit = self.chunk_size if limit is None else limit
        units = []

        # Step 1: Split into paragraphs
        paragraphs = text.split("\n\n")
        for i, p in enumerate(paragraphs):
            # Re-add paragraph spacing to all but the last paragraph
            p_text = p + "\n\n" if i < len(paragraphs) - 1 else p
            if not p_text.strip():
                continue

            if len(p_text) <= effective_limit:
                units.append(p_text)
                continue

            # Step 2: Paragraph too large, split into sentences
            # Split by punctuation (. ! ?) followed by whitespace
            sentences = re.split(r"(?<=[.!?])\s+", p_text)
            for j, s in enumerate(sentences):
                # Re-add spacing between sentences
                s_text = s + " " if j < len(sentences) - 1 else s
                if not s_text.strip():
                    continue

                if len(s_text) <= effective_limit:
                    units.append(s_text)
                    continue

                # Step 3: Sentence too large, split into words
                words = s_text.split(" ")
                for k, w in enumerate(words):
                    # Re-add spacing between words
                    w_text = w + " " if k < len(words) - 1 else w
                    if not w_text:
                        continue
                    units.append(w_text)

        return units

    def _faq_pair_end(self, text: str, answer_start: int, default_end: int) -> int:
        """Find the next non-FAQ section boundary, if one precedes the next question."""
        answer_region = text[answer_start:default_end]
        boundaries = []
        for pattern in (self._MARKDOWN_SECTION_RE, self._PLAIN_DEFINITIONS_RE):
            match = pattern.search(answer_region)
            if match:
                boundaries.append(answer_start + match.start())
        return min(boundaries, default=default_end)

    def _split_long_faq_pair(self, pair: str, question_length: int) -> List[str]:
        """Split only a long answer and repeat its question on every continuation."""
        question = pair[:question_length]
        answer = pair[question_length:]
        available = self.chunk_size - len(question)
        if available <= 0 or not answer:
            # Splitting the question itself would destroy the question/answer contract.
            return [_FAQUnit(pair)]

        answer_units = self._split_plain_text_into_units(answer, limit=available)
        fragments: List[str] = []
        current: List[str] = []
        current_len = 0
        for unit in answer_units:
            if current and current_len + len(unit) > available:
                fragments.append("".join(current))
                current = []
                current_len = 0
            current.append(unit)
            current_len += len(unit)
            if current_len >= available:
                fragments.append("".join(current))
                current = []
                current_len = 0
        if current:
            fragments.append("".join(current))

        return [_FAQUnit(question + fragment) for fragment in fragments] or [_FAQUnit(pair)]

    def _split_into_units(self, text: str) -> List[str]:
        """
        Split text while preserving each line-start ``Q<number>.`` FAQ pair.

        Documents without a valid FAQ marker use the legacy splitter verbatim,
        which keeps their chunk text and identifiers backward compatible.
        """
        questions = list(self._FAQ_QUESTION_RE.finditer(text))
        if not questions:
            return self._split_plain_text_into_units(text)

        units: List[str] = []
        cursor = 0
        for index, question_match in enumerate(questions):
            if question_match.start() < cursor:
                continue

            if cursor < question_match.start():
                units.extend(self._split_plain_text_into_units(text[cursor:question_match.start()]))

            next_question = questions[index + 1].start() if index + 1 < len(questions) else len(text)
            pair_end = self._faq_pair_end(text, question_match.end(), next_question)
            pair = text[question_match.start():pair_end]
            question_length = question_match.end() - question_match.start()
            if len(pair) <= self.chunk_size:
                units.append(_FAQUnit(pair))
            else:
                units.extend(self._split_long_faq_pair(pair, question_length))
            cursor = pair_end

        if cursor < len(text):
            units.extend(self._split_plain_text_into_units(text[cursor:]))
        return units

    def split_text_to_chunks(self, text: str, policy_id: str, policy_name: str) -> List[Chunk]:
        """
        Processes the input text and splits it into a list of Chunk objects
        conforming to the configured chunk_size and chunk_overlap parameters.

        Args:
            text: The raw extracted policy text.
            policy_id: The ID of the policy (e.g., POL_IND_01).
            policy_name: The name of the policy (e.g., Star Health Assure Individual).

        Returns:
            A list of Chunk objects containing metadata and text.
        """
        if not text.strip():
            return []

        units = self._split_into_units(text)

        chunks = []
        current_units = []
        current_len = 0
        chunk_index = 0
        last_i = -1  # Tracks the split unit index to prevent infinite loops

        i = 0
        while i < len(units):
            unit = units[i]
            unit_len = len(unit)

            # FAQ pairs are deliberately isolated so retrieval never receives a
            # question without its answer or an answer diluted by another FAQ.
            if isinstance(unit, _FAQUnit):
                if current_units:
                    chunk_text = "".join(current_units)
                    chunks.append(Chunk(
                        chunk_id=f"{policy_id}_chunk_{chunk_index}",
                        policy_id=policy_id,
                        policy_name=policy_name,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text
                    ))
                    chunk_index += 1
                    current_units = []
                    current_len = 0
                chunks.append(Chunk(
                    chunk_id=f"{policy_id}_chunk_{chunk_index}",
                    policy_id=policy_id,
                    policy_name=policy_name,
                    chunk_index=chunk_index,
                    chunk_text=str(unit)
                ))
                chunk_index += 1
                last_i = -1
                i += 1
                continue

            # Edge case: a single unit is larger than chunk_size (should be extremely rare)
            if unit_len > self.chunk_size:
                # Finalize any accumulated text in current_units
                if current_units:
                    chunk_text = "".join(current_units)
                    chunks.append(Chunk(
                        chunk_id=f"{policy_id}_chunk_{chunk_index}",
                        policy_id=policy_id,
                        policy_name=policy_name,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text
                    ))
                    chunk_index += 1
                    current_units = []
                    current_len = 0

                # Add the large unit as its own standalone chunk
                chunks.append(Chunk(
                    chunk_id=f"{policy_id}_chunk_{chunk_index}",
                    policy_id=policy_id,
                    policy_name=policy_name,
                    chunk_index=chunk_index,
                    chunk_text=unit
                ))
                chunk_index += 1
                i += 1
                continue

            if current_len + unit_len <= self.chunk_size:
                current_units.append(unit)
                current_len += unit_len
                i += 1
            else:
                # If we backtracked in the previous step and still cannot fit the unit
                # (meaning the overlap units + current unit > chunk_size), we must
                # clear the overlap to prevent an infinite loop.
                if i == last_i and current_units:
                    current_units = []
                    current_len = 0
                    continue

                # Finalize the current chunk
                chunk_text = "".join(current_units)
                chunks.append(Chunk(
                    chunk_id=f"{policy_id}_chunk_{chunk_index}",
                    policy_id=policy_id,
                    policy_name=policy_name,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text
                ))
                chunk_index += 1

                # Record the split index to detect infinite loops
                last_i = i

                # Backtrack to calculate overlapping units
                overlap_units = []
                overlap_len = 0
                for u in reversed(current_units):
                    if overlap_len + len(u) <= self.chunk_overlap:
                        overlap_units.insert(0, u)
                        overlap_len += len(u)
                    else:
                        break

                # Start the next chunk with overlapping units
                current_units = overlap_units
                current_len = overlap_len
                # Do NOT increment i, so the current unit is processed in the next iteration

        # Finalize any remaining text in current_units
        if current_units:
            chunk_text = "".join(current_units)
            # Avoid duplicate chunks if the final chunk text is identical to the previous chunk
            if not chunks or chunk_text != chunks[-1].chunk_text:
                chunks.append(Chunk(
                    chunk_id=f"{policy_id}_chunk_{chunk_index}",
                    policy_id=policy_id,
                    policy_name=policy_name,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text
                ))

        return chunks
