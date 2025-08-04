"""
Comprehensive tests for FactualRAG system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from no_hallucination_rag.core.factual_rag import FactualRAG, RAGResponse
from no_hallucination_rag.core.error_handling import ValidationError, RetrievalError


class TestFactualRAG:
    """Test suite for FactualRAG class."""
    
    @pytest.fixture
    def rag_system(self):
        """Create RAG system for testing."""
        return FactualRAG(
            factuality_threshold=0.9,
            enable_caching=False,  # Disable for deterministic tests
            enable_optimization=False,
            enable_metrics=False
        )
    
    @pytest.fixture
    def mock_sources(self):
        """Mock source documents."""
        return [
            {
                "id": "doc_1",
                "title": "AI Safety Guidelines",
                "url": "https://example.com/ai-safety",
                "content": "AI systems must undergo safety testing before deployment.",
                "date": "2023-10-30",
                "authority_score": 0.95,
                "relevance_score": 0.8
            },
            {
                "id": "doc_2",
                "title": "Governance Framework",
                "url": "https://example.com/governance",
                "content": "Companies must file impact assessments for AI applications.",
                "date": "2025-01-01",
                "authority_score": 0.9,
                "relevance_score": 0.7
            }
        ]
    
    def test_initialization(self):
        """Test RAG system initialization."""
        rag = FactualRAG(
            factuality_threshold=0.95,
            governance_mode="strict",
            max_sources=10,
            enable_security=True,
            enable_metrics=True
        )
        
        assert rag.factuality_threshold == 0.95
        assert rag.governance_mode == "strict"
        assert rag.max_sources == 10
        assert rag.enable_security is True
        assert rag.enable_metrics is True
        assert rag.retriever is not None
        assert rag.source_ranker is not None
        assert rag.factuality_detector is not None
        assert rag.governance_checker is not None
    
    def test_query_basic_flow(self, rag_system, mock_sources):
        """Test basic query processing flow."""
        # Mock components
        rag_system.retriever.retrieve = Mock(return_value=mock_sources)
        rag_system.source_ranker.rank = Mock(return_value=mock_sources)
        rag_system.factuality_detector.verify_answer = Mock(return_value=0.95)
        rag_system.governance_checker.check_compliance = Mock(
            return_value=Mock(is_compliant=True, details={}, violations=[], recommendations=[])
        )
        
        # Execute query
        response = rag_system.query("What are AI safety requirements?")
        
        # Assertions
        assert isinstance(response, RAGResponse)
        assert response.factuality_score == 0.95
        assert response.governance_compliant is True
        assert len(response.sources) == 2
        assert response.answer is not None
        assert response.timestamp is not None
        
        # Verify component calls
        rag_system.retriever.retrieve.assert_called_once()
        rag_system.source_ranker.rank.assert_called_once()
        rag_system.factuality_detector.verify_answer.assert_called_once()
        rag_system.governance_checker.check_compliance.assert_called_once()
    
    def test_query_no_sources(self, rag_system):
        """Test query handling when no sources found."""
        rag_system.retriever.retrieve = Mock(return_value=[])
        
        response = rag_system.query("Unknown topic")
        
        assert response.factuality_score == 0.0
        assert "no reliable sources were found" in response.answer
        assert len(response.sources) == 0
    
    def test_query_insufficient_sources(self, rag_system, mock_sources):
        """Test query handling when insufficient sources found."""
        rag_system.retriever.retrieve = Mock(return_value=mock_sources[:1])  # Only 1 source
        rag_system.source_ranker.rank = Mock(return_value=mock_sources[:1])
        
        response = rag_system.query("Test query", min_sources=2)
        
        assert response.factuality_score == 0.0
        assert "require at least 2 for a confident answer" in response.answer
    
    def test_query_low_factuality(self, rag_system, mock_sources):
        """Test query handling when factuality score is too low."""
        rag_system.retriever.retrieve = Mock(return_value=mock_sources)
        rag_system.source_ranker.rank = Mock(return_value=mock_sources)
        rag_system.factuality_detector.verify_answer = Mock(return_value=0.5)  # Low score
        
        response = rag_system.query("Test query", min_factuality_score=0.9)
        
        assert response.factuality_score == 0.5
        assert "factuality score (0.50) is below the required threshold" in response.answer
    
    def test_query_with_client_context(self, rag_system, mock_sources):
        """Test query with client context for security."""
        rag_system.retriever.retrieve = Mock(return_value=mock_sources)
        rag_system.source_ranker.rank = Mock(return_value=mock_sources)
        rag_system.factuality_detector.verify_answer = Mock(return_value=0.95)
        rag_system.governance_checker.check_compliance = Mock(
            return_value=Mock(is_compliant=True, details={}, violations=[], recommendations=[])
        )
        
        client_context = {
            "client_ip": "192.168.1.1",
            "user_id": "test_user",
            "api_key": "test_key"
        }
        
        response = rag_system.query("Test query", client_context=client_context)
        
        assert isinstance(response, RAGResponse)
        assert response.factuality_score == 0.95
    
    def test_input_validation_error(self, rag_system):
        """Test input validation error handling."""
        # Mock validation to fail
        rag_system.input_validator.validate_query = Mock(
            return_value=Mock(is_valid=False, errors=["Input too long"])
        )
        
        response = rag_system.query("x" * 3000)  # Very long input
        
        assert "Request validation failed" in response.answer
        assert response.factuality_score == 0.0
    
    def test_retrieval_error_handling(self, rag_system):
        """Test retrieval error handling."""
        rag_system.retriever.retrieve = Mock(side_effect=Exception("Retrieval failed"))
        
        response = rag_system.query("Test query")
        
        assert "error occurred while processing" in response.answer.lower()
        assert response.factuality_score == 0.0
    
    @pytest.mark.asyncio
    async def test_async_query(self, rag_system, mock_sources):
        """Test async query processing."""
        # Mock components for async
        rag_system.retriever.retrieve = Mock(return_value=mock_sources)
        rag_system.source_ranker.rank = Mock(return_value=mock_sources)
        rag_system.factuality_detector.verify_answer = Mock(return_value=0.95)
        rag_system.governance_checker.check_compliance = Mock(
            return_value=Mock(is_compliant=True, details={}, violations=[], recommendations=[])
        )
        
        response = await rag_system.aquery("What are AI safety requirements?")
        
        assert isinstance(response, RAGResponse)
        assert response.factuality_score == 0.95
    
    @pytest.mark.asyncio
    async def test_async_batch_query(self, rag_system, mock_sources):
        """Test async batch query processing."""
        # Mock components
        rag_system.retriever.retrieve = Mock(return_value=mock_sources)
        rag_system.source_ranker.rank = Mock(return_value=mock_sources)
        rag_system.factuality_detector.verify_answer = Mock(return_value=0.95)
        rag_system.governance_checker.check_compliance = Mock(
            return_value=Mock(is_compliant=True, details={}, violations=[], recommendations=[])
        )
        
        questions = [
            "What are AI safety requirements?",
            "What are governance requirements?",
            "What are compliance standards?"
        ]
        
        responses = await rag_system.aquery_batch(questions, batch_size=2)
        
        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, RAGResponse)
            assert response.factuality_score >= 0.0
    
    def test_system_health(self, rag_system):
        """Test system health monitoring."""
        health = rag_system.get_system_health()
        
        assert "status" in health
        assert "timestamp" in health
        assert "components" in health
        assert health["components"]["retriever"] is True
        assert health["components"]["source_ranker"] is True
        assert health["components"]["factuality_detector"] is True
        assert health["components"]["governance_checker"] is True
    
    def test_models_loaded(self, rag_system):
        """Test models loaded check."""
        models = rag_system.models_loaded()
        
        assert isinstance(models, dict)
        assert "retriever" in models
        assert "factuality_detector" in models
        assert "governance_checker" in models
        assert "source_ranker" in models
    
    def test_knowledge_bases_list(self, rag_system):
        """Test knowledge base listing."""
        kb_list = rag_system.list_knowledge_bases()
        
        assert isinstance(kb_list, list)
        assert len(kb_list) > 0
    
    def test_citation_generation(self, rag_system, mock_sources):
        """Test citation generation."""
        citations = rag_system._generate_citations(mock_sources)
        
        assert len(citations) == 2
        assert "[1] AI Safety Guidelines - https://example.com/ai-safety" in citations[0]
        assert "[2] Governance Framework - https://example.com/governance" in citations[1]
    
    def test_answer_generation(self, rag_system, mock_sources):
        """Test answer generation from sources."""
        answer = rag_system._generate_answer("What are requirements?", mock_sources)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "based on the available sources" in answer.lower()
    
    def test_error_response_creation(self, rag_system):
        """Test error response creation."""
        response = rag_system._create_error_response("test query", "test error")
        
        assert isinstance(response, RAGResponse)
        assert "test error" in response.answer
        assert response.factuality_score == 0.0
        assert response.governance_compliant is False
    
    def test_query_caching_disabled(self, rag_system, mock_sources):
        """Test query processing with caching disabled."""
        rag_system.cache_manager = None  # Ensure caching is disabled
        
        rag_system.retriever.retrieve = Mock(return_value=mock_sources)
        rag_system.source_ranker.rank = Mock(return_value=mock_sources)
        rag_system.factuality_detector.verify_answer = Mock(return_value=0.95)
        rag_system.governance_checker.check_compliance = Mock(
            return_value=Mock(is_compliant=True, details={}, violations=[], recommendations=[])
        )
        
        response = rag_system.query("Test query")
        
        assert isinstance(response, RAGResponse)
        assert response.factuality_score == 0.95
    
    def test_shutdown(self, rag_system):
        """Test graceful shutdown."""
        # Should not raise any exceptions
        rag_system.shutdown()
    
    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test valid parameters
        rag = FactualRAG(
            factuality_threshold=0.95,
            max_sources=20,
            min_sources=3
        )
        
        assert rag.factuality_threshold == 0.95
        assert rag.max_sources == 20
        assert rag.min_sources == 3
    
    def test_concurrent_processing_fallback(self, rag_system):
        """Test fallback when concurrent processing is disabled."""
        rag_system.enable_concurrency = False
        rag_system.async_processor = None
        
        # Should still work without concurrency features
        rag_system.retriever.retrieve = Mock(return_value=[])
        response = rag_system.query("Test query")
        
        assert isinstance(response, RAGResponse)


class TestFactualRAGIntegration:
    """Integration tests for FactualRAG system."""
    
    def test_end_to_end_query_processing(self):
        """Test complete end-to-end query processing."""
        rag = FactualRAG(
            factuality_threshold=0.8,
            enable_caching=False,
            enable_optimization=False,
            enable_metrics=False,
            enable_security=False
        )
        
        # This uses the mock knowledge base from HybridRetriever
        response = rag.query("What are AI safety requirements?")
        
        assert isinstance(response, RAGResponse)
        assert response.factuality_score >= 0.0
        assert response.answer is not None
        assert response.timestamp is not None
        assert len(response.sources) >= 0
    
    def test_governance_compliance_integration(self):
        """Test governance compliance integration."""
        rag = FactualRAG(
            governance_mode="strict",
            enable_caching=False,
            enable_optimization=False,
            enable_metrics=False,
            enable_security=False
        )
        
        response = rag.query("What are the penalties for AI violations?")
        
        assert isinstance(response, RAGResponse)
        # Should have governance compliance checked
        assert isinstance(response.governance_compliant, bool)
        assert "governance_details" in response.verification_details
    
    def test_multi_language_query_handling(self):
        """Test handling of non-English queries."""
        rag = FactualRAG(
            enable_caching=False,
            enable_optimization=False,
            enable_metrics=False,
            enable_security=False
        )
        
        # Test with different character sets (should be handled gracefully)
        queries = [
            "¿Cuáles son los requisitos de seguridad de IA?",  # Spanish
            "IA安全要求是什么？",  # Chinese
            "Quelles sont les exigences de sécurité IA?",  # French
        ]
        
        for query in queries:
            response = rag.query(query)
            assert isinstance(response, RAGResponse)
            # Should not crash, even if quality is low
            assert response.factuality_score >= 0.0