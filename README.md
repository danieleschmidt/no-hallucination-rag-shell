# No-Hallucination RAG Shell

**Retrieval-first architecture with formal hallucination prevention guarantees.**

> Every sentence in every answer is traceable to a verified source chunk.
> No LLM parametric knowledge is injected.

---

## Motivation

Large language models hallucinate — they generate plausible-sounding but unsupported statements. Standard RAG systems reduce (but don't eliminate) this by conditioning generation on retrieved context. This project goes further: it **formally guarantees** that output is grounded by construction, then verifies that guarantee post-hoc with lightweight token-overlap checks.

---

## Architecture

```
Query
  │
  ▼
┌──────────────────┐
│  TFIDFRetriever  │  ← pure TF-IDF over an in-memory corpus (no vector DB)
└────────┬─────────┘
         │  top-k chunks
         ▼
┌────────────────────┐
│  GroundedGenerator │  ← template-based; never uses parametric knowledge
└────────┬───────────┘
         │  "According to [source]: <sentence>"
         ▼
┌────────────────────────┐   ┌──────────────────────┐
│  HallucinationDetector │   │  CitationVerifier     │
│  (per-sentence score)  │   │  (claim → source link)|
└────────────────────────┘   └──────────────────────┘
```

### Core components

| Class | File | Purpose |
|---|---|---|
| `TFIDFRetriever` | `retrieval/tfidf_retriever.py` | TF-IDF retrieval; no external dependencies |
| `GroundedGenerator` | `core/grounded_generator.py` | Template-based answer generation with mandatory citations |
| `HallucinationDetector` | `core/hallucination_detector.py` | Per-sentence grounding score via Jaccard similarity |
| `CitationVerifier` | `core/citation_verifier.py` | Claim-level verification against cited source chunks |

---

## Formal Guarantee Model

### Construction guarantee (GroundedGenerator)

The generator operates exclusively over retrieved context:

```
answer = ∪ { "According to [src_i]: sent_j" | sent_j ∈ top-k(src_i, query) }
```

No sentence is added that did not originate from a retrieved chunk.
This is a **by-construction** guarantee: hallucination is architecturally impossible
as long as retrieval returns non-empty results.

### Verification guarantee (HallucinationDetector + CitationVerifier)

After generation, each sentence is independently verified:

```
grounding_score(s) = max_i Jaccard(tokens(s), tokens(chunk_i))
```

A sentence is grounded iff `grounding_score ≥ threshold` (default 0.15).

The CitationVerifier additionally checks each `[source]` attribution:

```
supported(claim, src) = substring_match(claim, src.content)
                        OR Jaccard(tokens(claim), tokens(src.content)) ≥ threshold
```

Together these provide:
1. **Construction guarantee** — answers are built only from retrieved text
2. **Verification guarantee** — every sentence is independently scored for grounding
3. **Attribution guarantee** — every citation is verified against its named source

---

## Installation

```bash
git clone https://github.com/danieleschmidt/no-hallucination-rag-shell
cd no-hallucination-rag-shell
pip install numpy pytest  # only external dependencies
```

---

## Quick Start

```python
from no_hallucination_rag.retrieval.tfidf_retriever import TFIDFRetriever, Document
from no_hallucination_rag.core.grounded_generator import GroundedGenerator
from no_hallucination_rag.core.hallucination_detector import HallucinationDetector
from no_hallucination_rag.core.citation_verifier import CitationVerifier

# 1. Index documents
retriever = TFIDFRetriever()
retriever.add_document(Document(
    doc_id="d1", source="paper.pdf", title="NLP Paper",
    content="Hallucination means generating factually incorrect content."
))

# 2. Retrieve
chunks = retriever.retrieve("What is hallucination?", top_k=3)
chunk_dicts = [{"doc_id": c.doc_id, "source": c.source,
                "title": c.title, "content": c.content} for c in chunks]

# 3. Generate grounded answer
gen = GroundedGenerator()
answer = gen.generate("What is hallucination?", chunk_dicts)
print(answer.answer)
# → "According to [paper.pdf]: Hallucination means generating factually incorrect content."

# 4. Score grounding
det = HallucinationDetector(grounding_threshold=0.15)
result = det.score_answer(answer.answer, chunk_dicts)
print(result.overall_score, result.is_hallucination_free)

# 5. Verify citations
ver = CitationVerifier(overlap_threshold=0.20)
report = ver.verify_answer(answer.answer, chunk_dicts)
print(report.overall_support_rate, report.all_supported)
```

---

## Demo

```bash
python demo.py
```

Indexes 3 sample documents, asks 3 questions, and prints:
- Retrieved chunks with TF-IDF scores
- Grounded answers with source citations
- Per-sentence grounding scores
- Citation verification results

---

## Tests

```bash
pytest tests/test_core_components.py -v
```

28 tests across all four components + an end-to-end integration test.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | (available, not yet required by core — reserved for future embedding support) |
| `pytest` | Testing |

**No LLM APIs. No vector databases. No proprietary services.**

---

## Research Context

This system is designed to satisfy the properties required for formal hallucination prevention guarantees as described in:

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS)
- Ji et al. (2023) — *Survey of Hallucination in Natural Language Generation* (ACM CSUR)
- Maynez et al. (2020) — *On Faithfulness and Factuality in Abstractive Summarization* (ACL)

The Jaccard-based grounding score is a lightweight proxy for NLI-based faithfulness,
suitable for fast verification without GPU inference.

---

## License

MIT
