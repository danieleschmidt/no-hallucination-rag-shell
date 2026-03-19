#!/usr/bin/env python3
"""
demo.py — No-Hallucination RAG Shell demonstration
====================================================

Indexes 3 sample documents, asks 3 questions, and shows:
  - Retrieved chunks
  - Grounded answer with citations
  - Per-sentence grounding scores (HallucinationDetector)
  - Citation verification results (CitationVerifier)
"""

from no_hallucination_rag.retrieval.tfidf_retriever import TFIDFRetriever, Document
from no_hallucination_rag.core.grounded_generator import GroundedGenerator
from no_hallucination_rag.core.hallucination_detector import HallucinationDetector
from no_hallucination_rag.core.citation_verifier import CitationVerifier

# ─────────────────────────────────────────────
# 1. Sample documents
# ─────────────────────────────────────────────
DOCUMENTS = [
    Document(
        doc_id="doc1",
        source="ai_safety_report_2024.pdf",
        title="AI Safety Report 2024",
        content=(
            "Large language models can produce plausible-sounding but factually incorrect "
            "statements, a phenomenon known as hallucination. Retrieval-augmented generation "
            "(RAG) systems address this by grounding model outputs in verified source documents. "
            "Empirical evaluations show that RAG reduces hallucination rates by 40–60 percent "
            "compared to purely parametric generation."
        ),
    ),
    Document(
        doc_id="doc2",
        source="rag_survey_acl2023.pdf",
        title="RAG Survey ACL 2023",
        content=(
            "Retrieval-augmented generation combines a dense retrieval component with a "
            "conditional text generator. The retriever selects the top-k relevant passages "
            "from a knowledge corpus using BM25 or dense embeddings. The generator then "
            "conditions on both the query and retrieved passages. Citation verification "
            "is the process of checking whether each generated claim is supported by a "
            "retrieved passage, typically using natural language inference or token overlap."
        ),
    ),
    Document(
        doc_id="doc3",
        source="hallucination_taxonomy_emnlp2023.pdf",
        title="Hallucination Taxonomy EMNLP 2023",
        content=(
            "Hallucinations in NLP systems are categorised as intrinsic (contradicting the "
            "source) or extrinsic (adding unsupported information). Formal hallucination "
            "prevention guarantees require that every sentence in the output be traceable to "
            "a specific source passage. Jaccard similarity and substring matching are "
            "lightweight approaches to verify sentence-level grounding without an NLI model."
        ),
    ),
]

QUESTIONS = [
    "What is hallucination in language models and how does RAG address it?",
    "How does citation verification work in RAG systems?",
    "What is the difference between intrinsic and extrinsic hallucinations?",
]

# ─────────────────────────────────────────────
# 2. Build index
# ─────────────────────────────────────────────
retriever = TFIDFRetriever()
retriever.add_documents(DOCUMENTS)
print("=" * 70)
print("  No-Hallucination RAG Shell — Demo")
print("=" * 70)
print(f"\n✔ Indexed {len(DOCUMENTS)} documents\n")

generator = GroundedGenerator(top_k_chunks=3, sentences_per_chunk=2, min_relevance=0.05)
detector = HallucinationDetector(grounding_threshold=0.12)
verifier = CitationVerifier(overlap_threshold=0.18)

# ─────────────────────────────────────────────
# 3. Run each query
# ─────────────────────────────────────────────
for q_idx, query in enumerate(QUESTIONS, 1):
    print(f"{'─' * 70}")
    print(f"Q{q_idx}: {query}")
    print()

    # Retrieve
    chunks = retriever.retrieve(query, top_k=3)
    chunk_dicts = [
        {
            "doc_id": c.doc_id,
            "source": c.source,
            "title": c.title,
            "content": c.content,
            "score": c.score,
        }
        for c in chunks
    ]

    print(f"  Retrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"    [{c.score:.4f}] {c.source}")
    print()

    # Generate grounded answer
    generated = generator.generate(query, chunk_dicts)
    print("  Answer:")
    for line in generated.answer.split("\n"):
        print(f"    {line}")
    print()

    # Grounding scores (HallucinationDetector)
    detection = detector.score_answer(generated.answer, chunk_dicts)
    print(f"  Grounding scores (threshold={detector.grounding_threshold}):")
    for ss in detection.sentence_scores:
        status = "✔" if ss.is_grounded else "✘"
        print(f"    {status} [{ss.score:.4f}] {ss.sentence[:80]!r}")
    print(f"  Overall grounding score: {detection.overall_score:.4f}")
    print(f"  Hallucination-free: {detection.is_hallucination_free}")
    if detection.ungrounded_sentences:
        print("  Ungrounded sentences:")
        for s in detection.ungrounded_sentences:
            print(f"    ✘ {s[:100]!r}")
    print()

    # Citation verification
    report = verifier.verify_answer(generated.answer, chunk_dicts)
    print(f"  Citation verification ({report.supported_count}/{len(report.results)} supported):")
    for vr in report.results:
        status = "✔" if vr.is_supported else "✘"
        snippet = f" → '{vr.matched_text[:60]}'" if vr.matched_text else ""
        print(f"    {status} [{vr.support_type}|{vr.overlap_score:.3f}] "
              f"source={vr.source!r}{snippet}")
    print(f"  All citations verified: {report.all_supported}")
    print()

print("=" * 70)
print("Demo complete.")
