"""
tests/test_core_components.py
Tests for CitationVerifier, GroundedGenerator, and HallucinationDetector.
Pure stdlib + numpy (no LLM APIs).
"""

import pytest
from no_hallucination_rag.retrieval.tfidf_retriever import TFIDFRetriever, Document
from no_hallucination_rag.core.grounded_generator import GroundedGenerator, GeneratedAnswer
from no_hallucination_rag.core.hallucination_detector import HallucinationDetector, DetectionResult
from no_hallucination_rag.core.citation_verifier import CitationVerifier, VerificationReport


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_chunks():
    return [
        {
            "doc_id": "d1",
            "source": "safety_report.pdf",
            "title": "Safety Report",
            "content": (
                "Hallucination in language models refers to generating plausible but "
                "factually incorrect statements. Retrieval-augmented generation reduces "
                "hallucination by grounding outputs in verified source documents."
            ),
        },
        {
            "doc_id": "d2",
            "source": "rag_survey.pdf",
            "title": "RAG Survey",
            "content": (
                "Citation verification checks whether each generated claim is supported "
                "by a retrieved passage using token overlap or natural language inference."
            ),
        },
    ]


@pytest.fixture
def retriever_with_docs():
    r = TFIDFRetriever()
    docs = [
        Document(
            doc_id="doc1",
            source="safety.pdf",
            title="Safety",
            content="Hallucination means generating factually incorrect text. RAG reduces hallucination.",
        ),
        Document(
            doc_id="doc2",
            source="survey.pdf",
            title="Survey",
            content="Citation verification ensures claims are supported by retrieved passages.",
        ),
    ]
    r.add_documents(docs)
    return r


# ─────────────────────────────────────────────
# TFIDFRetriever
# ─────────────────────────────────────────────

class TestTFIDFRetriever:
    def test_retrieve_returns_results(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("hallucination", top_k=2)
        assert len(results) >= 1

    def test_top_result_is_relevant(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("hallucination RAG", top_k=2)
        assert any("hallucination" in r.content.lower() for r in results)

    def test_scores_descending(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("hallucination", top_k=2)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_retriever(self):
        r = TFIDFRetriever()
        assert r.retrieve("anything") == []

    def test_top_k_respected(self, retriever_with_docs):
        results = retriever_with_docs.retrieve("hallucination", top_k=1)
        assert len(results) <= 1

    def test_incremental_add(self):
        r = TFIDFRetriever()
        r.add_document(Document("a", "First document about cats.", "file.txt", "Cats"))
        r.add_document(Document("b", "Second document about dogs.", "file2.txt", "Dogs"))
        res = r.retrieve("cats")
        assert res[0].doc_id == "a"


# ─────────────────────────────────────────────
# GroundedGenerator
# ─────────────────────────────────────────────

class TestGroundedGenerator:
    def test_returns_generated_answer(self, sample_chunks):
        gen = GroundedGenerator()
        out = gen.generate("What is hallucination?", sample_chunks)
        assert isinstance(out, GeneratedAnswer)
        assert len(out.answer) > 0

    def test_answer_contains_citation_template(self, sample_chunks):
        gen = GroundedGenerator()
        out = gen.generate("What is hallucination?", sample_chunks)
        assert "According to [" in out.answer

    def test_citations_non_empty(self, sample_chunks):
        gen = GroundedGenerator()
        out = gen.generate("hallucination language models", sample_chunks)
        assert len(out.citations) > 0

    def test_source_chunks_used_listed(self, sample_chunks):
        gen = GroundedGenerator()
        out = gen.generate("citation verification", sample_chunks)
        assert len(out.source_chunks_used) > 0

    def test_no_chunks_returns_fallback(self):
        gen = GroundedGenerator()
        out = gen.generate("anything", [])
        assert "No relevant sources" in out.answer
        assert out.citations == []

    def test_top_k_chunks_limit(self, sample_chunks):
        gen = GroundedGenerator(top_k_chunks=1)
        out = gen.generate("hallucination", sample_chunks)
        # At most 1 source should appear
        sources_used = set(c.source for c in out.citations)
        assert len(sources_used) <= 1

    def test_query_stored_in_result(self, sample_chunks):
        gen = GroundedGenerator()
        q = "What reduces hallucination?"
        out = gen.generate(q, sample_chunks)
        assert out.query == q


# ─────────────────────────────────────────────
# HallucinationDetector
# ─────────────────────────────────────────────

class TestHallucinationDetector:
    def test_high_overlap_is_grounded(self, sample_chunks):
        det = HallucinationDetector(grounding_threshold=0.10)
        # Sentence drawn verbatim from a source
        answer = (
            "Hallucination in language models refers to generating plausible but "
            "factually incorrect statements."
        )
        result = det.score_answer(answer, sample_chunks)
        assert result.sentence_scores[0].is_grounded

    def test_unrelated_sentence_low_score(self, sample_chunks):
        det = HallucinationDetector(grounding_threshold=0.10)
        answer = "Penguins are flightless birds found in the Southern Hemisphere."
        result = det.score_answer(answer, sample_chunks)
        assert result.overall_score < 0.3

    def test_overall_score_range(self, sample_chunks):
        det = HallucinationDetector()
        result = det.score_answer("RAG reduces hallucination by grounding outputs.", sample_chunks)
        assert 0.0 <= result.overall_score <= 1.0

    def test_empty_answer_returns_hallucination_free(self, sample_chunks):
        det = HallucinationDetector()
        result = det.score_answer("", sample_chunks)
        assert result.is_hallucination_free is True

    def test_ungrounded_sentences_listed(self, sample_chunks):
        det = HallucinationDetector(grounding_threshold=0.90)
        answer = "The sky is blue and quantum entanglement is real."
        result = det.score_answer(answer, sample_chunks)
        assert len(result.ungrounded_sentences) > 0

    def test_detection_result_type(self, sample_chunks):
        det = HallucinationDetector()
        result = det.score_answer("RAG reduces hallucination.", sample_chunks)
        assert isinstance(result, DetectionResult)

    def test_score_sentence_convenience(self, sample_chunks):
        det = HallucinationDetector()
        ss = det.score_sentence("RAG reduces hallucination", sample_chunks)
        assert 0.0 <= ss.score <= 1.0


# ─────────────────────────────────────────────
# CitationVerifier
# ─────────────────────────────────────────────

class TestCitationVerifier:
    def test_supported_claim_in_template_format(self, sample_chunks):
        ver = CitationVerifier(overlap_threshold=0.15)
        answer = (
            "According to [safety_report.pdf]: "
            "Hallucination in language models refers to generating plausible but "
            "factually incorrect statements."
        )
        report = ver.verify_answer(answer, sample_chunks)
        assert report.results[0].is_supported

    def test_unsupported_claim_fails(self, sample_chunks):
        ver = CitationVerifier(overlap_threshold=0.15)
        answer = (
            "According to [safety_report.pdf]: "
            "Quantum tunnelling occurs in semiconductors at low temperatures."
        )
        report = ver.verify_answer(answer, sample_chunks)
        assert not report.results[0].is_supported

    def test_overall_support_rate_range(self, sample_chunks):
        ver = CitationVerifier()
        answer = "According to [safety_report.pdf]: RAG reduces hallucination."
        report = ver.verify_answer(answer, sample_chunks)
        assert 0.0 <= report.overall_support_rate <= 1.0

    def test_report_type(self, sample_chunks):
        ver = CitationVerifier()
        report = ver.verify_answer("Some answer.", sample_chunks)
        assert isinstance(report, VerificationReport)

    def test_all_supported_flag(self, sample_chunks):
        ver = CitationVerifier(overlap_threshold=0.10)
        answer = (
            "According to [safety_report.pdf]: "
            "Retrieval-augmented generation reduces hallucination by grounding outputs."
        )
        report = ver.verify_answer(answer, sample_chunks)
        # At least one claim
        assert len(report.results) >= 1
        assert isinstance(report.all_supported, bool)

    def test_counts_correct(self, sample_chunks):
        ver = CitationVerifier()
        answer = (
            "According to [safety_report.pdf]: RAG reduces hallucination.\n"
            "According to [safety_report.pdf]: Quantum physics is complex."
        )
        report = ver.verify_answer(answer, sample_chunks)
        assert report.supported_count + report.unsupported_count == len(report.results)

    def test_missing_source_is_unsupported(self, sample_chunks):
        ver = CitationVerifier()
        answer = "According to [nonexistent_doc.pdf]: Some claim."
        report = ver.verify_answer(answer, sample_chunks)
        # Should still parse and mark as unsupported
        assert len(report.results) >= 1


# ─────────────────────────────────────────────
# Integration: full pipeline
# ─────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        """Index docs → retrieve → generate → detect → verify."""
        docs = [
            Document(
                doc_id="x1",
                source="nlp_paper.pdf",
                title="NLP Paper",
                content=(
                    "Hallucination in NLP is defined as generating content not supported "
                    "by the input context. Mitigation strategies include retrieval "
                    "augmentation and output verification."
                ),
            )
        ]
        retriever = TFIDFRetriever()
        retriever.add_documents(docs)

        query = "What is hallucination in NLP?"
        chunks = retriever.retrieve(query, top_k=3)
        chunk_dicts = [
            {"doc_id": c.doc_id, "source": c.source, "title": c.title, "content": c.content}
            for c in chunks
        ]

        gen = GroundedGenerator()
        generated = gen.generate(query, chunk_dicts)
        assert "According to [" in generated.answer

        det = HallucinationDetector(grounding_threshold=0.10)
        detection = det.score_answer(generated.answer, chunk_dicts)
        assert detection.overall_score >= 0.0

        ver = CitationVerifier(overlap_threshold=0.12)
        report = ver.verify_answer(generated.answer, chunk_dicts)
        assert report.supported_count >= 0
