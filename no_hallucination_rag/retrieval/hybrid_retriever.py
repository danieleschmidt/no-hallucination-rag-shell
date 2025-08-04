"""
Hybrid retrieval combining dense and sparse retrieval methods.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os


class HybridRetriever:
    """Combines dense and sparse retrieval for comprehensive source discovery."""
    
    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        data_path: str = "data/knowledge_bases"
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
        
        # Mock knowledge base for Generation 1
        self.mock_knowledge_base = self._create_mock_knowledge_base()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant sources using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters for retrieval
            
        Returns:
            List of relevant source documents
        """
        try:
            # For Generation 1, use mock retrieval
            # Generation 2 will implement actual dense/sparse retrieval
            
            results = self._mock_retrieve(query, top_k, filters)
            
            self.logger.info(f"Retrieved {len(results)} sources for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {e}")
            return []
    
    def _mock_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Mock retrieval for Generation 1 demonstration."""
        
        query_lower = query.lower()
        matching_docs = []
        
        # Simple keyword matching against mock knowledge base
        for doc in self.mock_knowledge_base:
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()
            
            # Calculate simple relevance score
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            title_words = set(title_lower.split())
            
            content_matches = len(query_words.intersection(content_words))
            title_matches = len(query_words.intersection(title_words))
            
            # Boost title matches
            relevance_score = content_matches + (title_matches * 2)
            
            if relevance_score > 0:
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = relevance_score
                matching_docs.append(doc_copy)
        
        # Sort by relevance and return top_k
        matching_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return matching_docs[:top_k]
    
    def _create_mock_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create mock knowledge base for demonstration."""
        
        mock_docs = [
            {
                "id": "doc_1",
                "title": "White House Executive Order on AI Safety",
                "url": "https://whitehouse.gov/ai-executive-order",
                "content": "The Executive Order on Safe, Secure, and Trustworthy Artificial Intelligence establishes new standards for AI safety and security. Key requirements include safety testing for AI systems above specified compute thresholds, mandatory red-team testing before deployment, and transparency reports for high-risk AI applications. Companies must file impact assessments and undergo algorithmic bias audits annually.",
                "date": "2023-10-30",
                "type": "executive_order",
                "authority_score": 1.0,
                "citations": 247
            },
            {
                "id": "doc_2", 
                "title": "NIST AI Risk Management Framework 2.0",
                "url": "https://nist.gov/ai-risk-framework-2025",
                "content": "The NIST AI Risk Management Framework provides a comprehensive approach for managing AI risks. The 2025 update includes enhanced transparency requirements, documentation standards, and governance mechanisms. Organizations must implement risk assessment methodologies, maintain audit trails, and ensure algorithmic accountability through regular evaluations.",
                "date": "2025-07-01",
                "type": "framework",
                "authority_score": 0.95,
                "citations": 156
            },
            {
                "id": "doc_3",
                "title": "AI Watermarking and Provenance Standards",
                "url": "https://federal-register.gov/2025/07/15/ai-watermark-standards",
                "content": "New federal requirements mandate machine-readable watermarks for all AI-generated content. Synthetic media must include provenance markers that identify the generating system, creation timestamp, and modification history. Compliance is required by January 2026 for all commercial AI systems.",
                "date": "2025-07-15",
                "type": "regulation",
                "authority_score": 0.9,
                "citations": 89
            },
            {
                "id": "doc_4",
                "title": "Employment AI Bias Audit Requirements",
                "url": "https://eeoc.gov/ai-bias-audit-requirements-2025",
                "content": "The Equal Employment Opportunity Commission requires annual bias audits for AI systems used in hiring, promotion, or employment decisions. Audits must test for disparate impact across protected classes and include statistical significance testing. Results must be publicly reported and corrective actions documented.",
                "date": "2025-06-01",
                "type": "guidance",
                "authority_score": 0.85,
                "citations": 67
            },
            {
                "id": "doc_5",
                "title": "Generative AI Safety Research Overview",
                "url": "https://arxiv.org/abs/2023.12345",
                "content": "Recent advances in large language model safety focus on reducing hallucinations and improving factual accuracy. Key techniques include retrieval-augmented generation, factual consistency checking, and source attribution. Research shows that hybrid retrieval methods combined with factuality detectors can reduce hallucination rates by up to 85%.",
                "date": "2023-12-15",
                "type": "research_paper",
                "authority_score": 0.7,
                "citations": 134
            },
            {
                "id": "doc_6",
                "title": "International AI Governance Coordination",
                "url": "https://oecd.org/ai/governance-coordination-2025",
                "content": "The OECD AI Governance Framework establishes international coordination mechanisms for AI safety standards. Member countries commit to shared risk assessment methodologies, cross-border incident reporting, and harmonized compliance requirements. The framework includes provisions for technology transfer restrictions and export controls.",
                "date": "2025-03-20",
                "type": "international_framework",
                "authority_score": 0.9,
                "citations": 98
            }
        ]
        
        return mock_docs