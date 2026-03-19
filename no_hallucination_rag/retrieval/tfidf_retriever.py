"""
TF-IDF based document retriever.
Pure Python + numpy — no external ML dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Document:
    """A document in the index."""
    doc_id: str
    content: str
    source: str = ""
    title: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with a relevance score."""
    doc_id: str
    content: str
    source: str
    title: str
    score: float
    metadata: Dict = field(default_factory=dict)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _term_freq(tokens: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    n = len(tokens) or 1
    return {t: c / n for t, c in tf.items()}


class TFIDFRetriever:
    """
    Simple TF-IDF retrieval over an in-memory corpus.
    """

    def __init__(self) -> None:
        self._docs: List[Document] = []
        # inverted index: term → set of doc indices
        self._index: Dict[str, List[int]] = {}
        # per-document term frequencies
        self._doc_tfs: List[Dict[str, float]] = []
        # idf cache (rebuilt after indexing)
        self._idf: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def add_document(self, doc: Document) -> None:
        idx = len(self._docs)
        self._docs.append(doc)
        tokens = _tokenize(doc.content)
        tf = _term_freq(tokens)
        self._doc_tfs.append(tf)
        for term in tf:
            self._index.setdefault(term, []).append(idx)
        self._rebuild_idf()

    def add_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            idx = len(self._docs)
            self._docs.append(doc)
            tokens = _tokenize(doc.content)
            tf = _term_freq(tokens)
            self._doc_tfs.append(tf)
            for term in tf:
                self._index.setdefault(term, []).append(idx)
        self._rebuild_idf()

    def _rebuild_idf(self) -> None:
        N = len(self._docs) or 1
        self._idf = {
            term: math.log((N + 1) / (len(postings) + 1)) + 1
            for term, postings in self._index.items()
        }

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not self._docs:
            return []

        q_tokens = _tokenize(query)
        q_tf = _term_freq(q_tokens)

        # Score every document that shares at least one term
        candidate_indices: set = set()
        for term in q_tf:
            candidate_indices.update(self._index.get(term, []))

        scores: Dict[int, float] = {}
        for idx in candidate_indices:
            doc_tf = self._doc_tfs[idx]
            score = 0.0
            for term, qtf in q_tf.items():
                idf = self._idf.get(term, 1.0)
                score += qtf * doc_tf.get(term, 0.0) * idf * idf
            scores[idx] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            doc = self._docs[idx]
            results.append(
                RetrievedChunk(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    source=doc.source,
                    title=doc.title,
                    score=score,
                    metadata=doc.metadata,
                )
            )
        return results
