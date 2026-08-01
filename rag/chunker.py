"""
chunker.py

Implements hierarchical semantic chunking of policy text. Splits documents
by paragraph, sentence, and word boundaries, maintaining a configurable chunk size
and contextual overlap without cutting sentences or headings in half.
"""

import re
from typing import List, Optional
from rag.models import Chunk


class SemanticChunker:
    """
    Splits policy text into semantically meaningful chunks, preserving
    paragraph/heading boundaries and sentence structures wherever possible.
    """

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

    def _split_into_units(self, text: str) -> List[str]:
        """
        Hierarchically splits the input text into a list of atomic text units.
        Each unit is either under the chunk_size limit or split down to individual words.
        Separators are preserved in the units so that the original text can be
        reconstructed exactly.
        """
        units = []

        # Step 1: Split into paragraphs
        paragraphs = text.split("\n\n")
        for i, p in enumerate(paragraphs):
            # Re-add paragraph spacing to all but the last paragraph
            p_text = p + "\n\n" if i < len(paragraphs) - 1 else p
            if not p_text.strip():
                continue

            if len(p_text) <= self.chunk_size:
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

                if len(s_text) <= self.chunk_size:
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
