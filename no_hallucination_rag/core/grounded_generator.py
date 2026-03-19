"""
GroundedGenerator
=================
Generates answers *only* from retrieved context chunks.
Never uses parametric (LLM) knowledge.

Design
------
For each top-k retrieved chunk, the generator:
  1. Identifies the sentence(s) in the chunk most relevant to the query
     (by Jaccard overlap).
  2. Wraps them in a citation template:
       "According to [source]: <relevant_sentence>"
  3. Concatenates these grounded sentences into a final answer.

The result is a fully traceable, citation-backed response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Citation:
    source: str       # source id / title
    sentence: str     # the exact sentence drawn from the source
    doc_id: str = ""


@dataclass
class GeneratedAnswer:
    """Output of GroundedGenerator."""
    query: str
    answer: str                    # full formatted answer text
    citations: List[Citation]      # traceability chain
    source_chunks_used: List[str]  # doc_ids that contributed


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 8]


class GroundedGenerator:
    """
    Template-based grounded answer generator.

    Parameters
    ----------
    top_k_chunks
        How many retrieved chunks to include in the answer.
    sentences_per_chunk
        Max sentences to extract from each chunk.
    min_relevance
        Minimum Jaccard score for a sentence to be included.
    """

    def __init__(
        self,
        top_k_chunks: int = 3,
        sentences_per_chunk: int = 2,
        min_relevance: float = 0.05,
    ) -> None:
        self.top_k_chunks = top_k_chunks
        self.sentences_per_chunk = sentences_per_chunk
        self.min_relevance = min_relevance

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict],
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer.

        Parameters
        ----------
        query
            The user's original question.
        retrieved_chunks
            List of dicts, each with:
              - ``content`` (str): chunk text
              - ``source``  (str, optional): file name / URL
              - ``title``   (str, optional): human-readable title
              - ``doc_id``  (str, optional): unique identifier

        Returns
        -------
        GeneratedAnswer
        """
        if not retrieved_chunks:
            return GeneratedAnswer(
                query=query,
                answer="No relevant sources were found for this query.",
                citations=[],
                source_chunks_used=[],
            )

        query_tokens = set(_tokenize(query))
        chunks = retrieved_chunks[: self.top_k_chunks]

        citations: List[Citation] = []
        answer_parts: List[str] = []
        used_ids: List[str] = []

        for chunk in chunks:
            content = chunk.get("content", "")
            source = chunk.get("source") or chunk.get("title") or chunk.get("doc_id", "unknown")
            doc_id = chunk.get("doc_id", source)

            sentences = _split_sentences(content)
            if not sentences:
                continue

            # Score each sentence by relevance to query
            scored = []
            for sent in sentences:
                sent_tokens = set(_tokenize(sent))
                score = _jaccard(query_tokens, sent_tokens)
                scored.append((score, sent))

            scored.sort(key=lambda x: x[0], reverse=True)

            # Pick top sentences above minimum relevance
            selected = [
                s for score, s in scored[: self.sentences_per_chunk]
                if score >= self.min_relevance
            ]

            if not selected:
                # Fall back: take the first sentence regardless of score
                selected = [scored[0][1]] if scored else []

            for sent in selected:
                line = f"According to [{source}]: {sent}"
                answer_parts.append(line)
                citations.append(Citation(source=source, sentence=sent, doc_id=doc_id))

            if selected:
                used_ids.append(doc_id)

        if not answer_parts:
            answer = "The retrieved sources did not contain relevant sentences for this query."
        else:
            answer = "\n".join(answer_parts)

        return GeneratedAnswer(
            query=query,
            answer=answer,
            citations=citations,
            source_chunks_used=used_ids,
        )
