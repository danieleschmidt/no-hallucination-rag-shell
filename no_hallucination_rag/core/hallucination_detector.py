"""
HallucinationDetector
=====================
Scores each sentence in an answer for grounding in the source documents.

Algorithm
---------
For each sentence S in the answer:
  1. Tokenize S and each source chunk.
  2. Compute Jaccard similarity between the token sets.
  3. Take the max similarity over all source chunks as the grounding score.

Score interpretation
  1.0 — sentence is heavily supported by sources
  0.0 — sentence has no lexical overlap with any source
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class SentenceScore:
    """Grounding score for a single sentence."""
    sentence: str
    score: float          # [0, 1]
    best_source: str      # source id / title of the most-overlapping chunk
    is_grounded: bool     # True if score >= threshold


@dataclass
class DetectionResult:
    """Full result for an answer."""
    answer: str
    sentence_scores: List[SentenceScore]
    overall_score: float              # mean of sentence scores
    ungrounded_sentences: List[str]   # sentences below threshold
    is_hallucination_free: bool       # True when every sentence is grounded


def _tokenize(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(t for t in text.split() if len(t) > 1)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering empties."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]


class HallucinationDetector:
    """
    Scores an answer's sentences against source documents using
    Jaccard token overlap.  No LLM required.
    """

    def __init__(self, grounding_threshold: float = 0.15) -> None:
        """
        Parameters
        ----------
        grounding_threshold
            Minimum Jaccard score for a sentence to be considered grounded.
            0.15 works well for concise template-generated answers; raise to
            0.25 for stricter checks.
        """
        self.grounding_threshold = grounding_threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score_answer(
        self,
        answer: str,
        source_chunks: List[Dict],
    ) -> DetectionResult:
        """
        Score every sentence in *answer* against *source_chunks*.

        Parameters
        ----------
        answer
            The generated text to evaluate.
        source_chunks
            List of dicts with at least a ``content`` key and optionally
            ``source`` / ``title`` / ``doc_id`` for attribution.

        Returns
        -------
        DetectionResult
        """
        sentences = _split_sentences(answer)
        if not sentences:
            return DetectionResult(
                answer=answer,
                sentence_scores=[],
                overall_score=0.0,
                ungrounded_sentences=[],
                is_hallucination_free=True,
            )

        source_tokens = [
            (
                chunk.get("source") or chunk.get("title") or chunk.get("doc_id", f"src-{i}"),
                _tokenize(chunk.get("content", "")),
            )
            for i, chunk in enumerate(source_chunks)
        ]

        sentence_scores: List[SentenceScore] = []
        for sent in sentences:
            sent_tokens = _tokenize(sent)
            best_score = 0.0
            best_src = ""
            for src_id, src_tokens in source_tokens:
                j = _jaccard(sent_tokens, src_tokens)
                if j > best_score:
                    best_score = j
                    best_src = src_id

            sentence_scores.append(
                SentenceScore(
                    sentence=sent,
                    score=round(best_score, 4),
                    best_source=best_src,
                    is_grounded=best_score >= self.grounding_threshold,
                )
            )

        overall = (
            sum(s.score for s in sentence_scores) / len(sentence_scores)
            if sentence_scores else 0.0
        )

        ungrounded = [s.sentence for s in sentence_scores if not s.is_grounded]

        return DetectionResult(
            answer=answer,
            sentence_scores=sentence_scores,
            overall_score=round(overall, 4),
            ungrounded_sentences=ungrounded,
            is_hallucination_free=len(ungrounded) == 0,
        )

    def score_sentence(self, sentence: str, source_chunks: List[Dict]) -> SentenceScore:
        """Convenience method to score a single sentence."""
        result = self.score_answer(sentence + ".", source_chunks)
        if result.sentence_scores:
            return result.sentence_scores[0]
        return SentenceScore(
            sentence=sentence, score=0.0, best_source="", is_grounded=False
        )
