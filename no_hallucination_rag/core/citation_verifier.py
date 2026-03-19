"""
CitationVerifier
================
For each claim in a generated answer, retrieves the cited source chunk
and verifies the claim is textually supported.

Support check
-------------
A claim is considered *supported* if it meets either condition:
  1. Substring match — the claim (normalised) appears verbatim in the chunk.
  2. Token-overlap (Jaccard) — the Jaccard similarity between the claim
     tokens and chunk tokens exceeds ``overlap_threshold``.

Both signals are reported so callers can apply their own threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class VerificationResult:
    """Verification result for a single claim."""
    claim: str
    source: str                 # cited source label
    is_supported: bool
    support_type: str           # "substring", "token_overlap", or "none"
    overlap_score: float        # Jaccard score [0, 1]
    matched_text: str = ""      # snippet from the source that supports the claim


@dataclass
class VerificationReport:
    """Aggregate report across all claims in an answer."""
    answer: str
    results: List[VerificationResult]
    supported_count: int
    unsupported_count: int
    overall_support_rate: float   # [0, 1]
    all_supported: bool


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokenize(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(t for t in text.split() if len(t) > 1)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 5]


def _find_matching_snippet(claim: str, content: str, window: int = 120) -> str:
    """Return a short snippet from *content* that best overlaps with *claim*."""
    claim_tokens = _tokenize(claim)
    sentences = _split_sentences(content)
    if not sentences:
        return content[:window]

    best_score = -1.0
    best_sent = ""
    for s in sentences:
        score = _jaccard(claim_tokens, _tokenize(s))
        if score > best_score:
            best_score = score
            best_sent = s

    return best_sent[:window]


# ------------------------------------------------------------------ #
# CitationVerifier
# ------------------------------------------------------------------ #

class CitationVerifier:
    """
    Verifies that each claim in an answer is textually supported by its
    cited source chunk.

    Parameters
    ----------
    overlap_threshold
        Minimum Jaccard score to count as token-overlap support.
    """

    def __init__(self, overlap_threshold: float = 0.20) -> None:
        self.overlap_threshold = overlap_threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def verify_answer(
        self,
        answer: str,
        source_chunks: List[Dict],
    ) -> VerificationReport:
        """
        Verify all claims in *answer* against *source_chunks*.

        The generator is expected to produce lines like:
            "According to [source_label]: <sentence>"
        or plain sentences.  Both are handled.

        Parameters
        ----------
        answer
            The generated answer text.
        source_chunks
            List of dicts with ``content``, ``source``/``title``/``doc_id``.

        Returns
        -------
        VerificationReport
        """
        # Build a lookup: source_label → content
        chunk_map: Dict[str, str] = {}
        for chunk in source_chunks:
            label = (
                chunk.get("source")
                or chunk.get("title")
                or chunk.get("doc_id", "")
            )
            if label:
                existing = chunk_map.get(label, "")
                chunk_map[label] = existing + " " + chunk.get("content", "")

        claims = self._extract_claims(answer, chunk_map)
        results: List[VerificationResult] = []

        for claim_text, cited_source in claims:
            result = self.verify_claim(claim_text, cited_source, chunk_map)
            results.append(result)

        supported = sum(1 for r in results if r.is_supported)
        total = len(results)

        return VerificationReport(
            answer=answer,
            results=results,
            supported_count=supported,
            unsupported_count=total - supported,
            overall_support_rate=round(supported / total, 4) if total else 1.0,
            all_supported=(supported == total),
        )

    def verify_claim(
        self,
        claim: str,
        cited_source: str,
        chunk_map: Dict[str, str],
    ) -> VerificationResult:
        """
        Verify a single claim against a specific source.

        Parameters
        ----------
        claim
            The sentence / claim to verify.
        cited_source
            The label of the source that should support the claim.
        chunk_map
            Mapping of source_label → full content string.
        """
        if cited_source not in chunk_map:
            # Try partial match
            cited_source = self._fuzzy_match_source(cited_source, chunk_map)

        content = chunk_map.get(cited_source, "")

        if not content:
            return VerificationResult(
                claim=claim,
                source=cited_source,
                is_supported=False,
                support_type="none",
                overlap_score=0.0,
                matched_text="",
            )

        norm_claim = _normalize(claim)
        norm_content = _normalize(content)

        # 1. Substring check
        if norm_claim in norm_content or _normalize(claim[:60]) in norm_content:
            snippet = _find_matching_snippet(claim, content)
            return VerificationResult(
                claim=claim,
                source=cited_source,
                is_supported=True,
                support_type="substring",
                overlap_score=1.0,
                matched_text=snippet,
            )

        # 2. Token-overlap (Jaccard)
        jaccard = _jaccard(_tokenize(claim), _tokenize(content))
        snippet = _find_matching_snippet(claim, content)

        return VerificationResult(
            claim=claim,
            source=cited_source,
            is_supported=jaccard >= self.overlap_threshold,
            support_type="token_overlap" if jaccard >= self.overlap_threshold else "none",
            overlap_score=round(jaccard, 4),
            matched_text=snippet if jaccard >= self.overlap_threshold else "",
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _extract_claims(
        self, answer: str, chunk_map: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """
        Extract (claim_text, cited_source) pairs from the answer.

        Supports the GroundedGenerator template:
            "According to [source]: sentence"
        and plain sentences (attributed to the best-matching source).
        """
        claims: List[Tuple[str, str]] = []
        lines = answer.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Template format: "According to [source]: sentence"
            m = re.match(r"According to \[(.+?)\]:\s*(.*)", line, re.IGNORECASE)
            if m:
                source_label = m.group(1).strip()
                claim_text = m.group(2).strip()
                claims.append((claim_text, source_label))
            else:
                # Plain sentence — attribute to best-matching source
                best_source, _ = self._best_matching_source(line, chunk_map)
                if best_source:
                    claims.append((line, best_source))

        return claims

    def _best_matching_source(
        self, text: str, chunk_map: Dict[str, str]
    ) -> Tuple[str, float]:
        """Return (source_label, score) for the highest-overlap source."""
        text_tokens = _tokenize(text)
        best_label = ""
        best_score = 0.0
        for label, content in chunk_map.items():
            score = _jaccard(text_tokens, _tokenize(content))
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    def _fuzzy_match_source(self, label: str, chunk_map: Dict[str, str]) -> str:
        """Try to find a source key that partially matches *label*."""
        label_lower = label.lower()
        for key in chunk_map:
            if label_lower in key.lower() or key.lower() in label_lower:
                return key
        return label
