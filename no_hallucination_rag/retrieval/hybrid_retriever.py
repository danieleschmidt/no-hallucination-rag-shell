```python
"""
Hybrid retrieval combining dense and sparse retrieval methods.
Generation 1: Basic functionality with graceful fallbacks.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    # Fallback for Generation 1 - stub implementations
    ML_AVAILABLE = False
    np = None
import pickle
import hashlib
import random
import math


class HybridRetriever:
    """Combines dense and sparse retrieval for comprehensive source discovery."""
    
    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        data_path: str = "data/knowledge_bases",
        embedding_model: str = "all-mpnet-base-v2"
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.data_path = data_path
        self.embedding_model_name = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize demo knowledge base
        self.knowledge_base = self._create_demo_knowledge_base()
        
        # Initialize models if available
        self._initialize_models()
        
        # Build indices if models are available
        self._build_indices()
        
        self.logger.info(f"HybridRetriever initialized with {len(self.knowledge_base)} demo sources")
    
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
            # Use semantic retrieval if ML dependencies are available
            if ML_AVAILABLE and self.embedding_model is not None:
                results = self._semantic_retrieve(query, top_k, filters)
            else:
                # Fall back to simple keyword-based retrieval
                results = self._keyword_retrieve(query, top_k, filters)
            
            self.logger.info(f"Retrieved {len(results)} sources for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {e}")
            return []
    
    def _keyword_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        
        for doc in self.knowledge_base:
            # Simple scoring based on keyword overlap
            content_words = set(doc['content'].lower().split())
            title_words = set(doc['title'].lower().split())
            
            # Calculate overlap scores
            content_overlap = len(query_words.intersection(content_words))
            title_overlap = len(query_words.intersection(title_words)) * 2  # Title words weighted higher
            
            total_score = content_overlap + title_overlap
            
            if total_score > 0:
                scored_docs.append((doc, total_score))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]
    
    def _initialize_models(self):
        """Initialize embedding model and vectorizers."""
        if not ML_AVAILABLE:
            self.logger.warning("ML dependencies not available, using mock retrieval")
            self.embedding_model = None
            self.tfidf_vectorizer = None
            return
            
        try:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Initialize TF-IDF vectorizer for sparse retrieval
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            self.embedding_model = None
            self.tfidf_vectorizer = None
    
    def _build_indices(self):
        """Build FAISS index and TF-IDF matrix from documents."""
        if not ML_AVAILABLE or self.embedding_model is None:
            self.documents = self.knowledge_base.copy()
            return
            
        try:
            self.documents = self.knowledge_base.copy()
            
            # Extract text for indexing
            texts = [doc["title"] + " " + doc["content"] for doc in self.documents]
            
            # Build dense embeddings and FAISS index
            self.logger.info("Building dense embeddings...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype(np.float32))
            
            # Build TF-IDF matrix for sparse retrieval
            self.logger.info("Building TF-IDF matrix...")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            self.logger.info(f"Indices built successfully for {len(self.documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to build indices: {e}")
            self.faiss_index = None
            self.tfidf_matrix = None
    
    def _semantic_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform semantic retrieval using dense and sparse methods."""
        if not ML_AVAILABLE or self.embedding_model is None:
            return self._keyword_retrieve(query, top_k, filters)
        
        try:
            # Dense retrieval using FAISS
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            dense_scores, dense_indices = self.faiss_index.search(
                query_embedding.astype(np.float32), 
                min(top_k * 2, len(self.documents))
            )
            
            # Sparse retrieval using TF-IDF
            query_tfidf = self.tfidf_vectorizer.transform([query])
            sparse_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            sparse_indices = np.argsort(sparse_scores)[::-1][:min(top_k * 2, len(self.documents))]
            
            # Combine scores
            combined_scores = {}
            
            for idx, score in zip(dense_indices[0], dense_scores[0]):
                combined_scores[idx] = score * self.dense_weight
            
            for idx in sparse_indices:
                if idx in combined_scores:
                    combined_scores[idx] += sparse_scores[idx] * self.sparse_weight
                else:
                    combined_scores[idx] = sparse_scores[idx] * self.sparse_weight
            
            # Sort by combined score and return top_k
            sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            results = [self.documents[idx] for idx, _ in sorted_indices[:top_k]]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic retrieval: {e}")
            return self._keyword_retrieve(query, top_k, filters)
    
    def _create_demo_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create demonstration knowledge base for Generation 1."""
        
        demo_docs = [
            {
                "id": "ai_governance_1",
                "title": "AI Governance Framework 2025",
                "content": "The AI governance framework establishes principles for responsible AI development, including transparency, accountability, and fairness requirements. Organizations must implement risk assessment procedures and maintain audit trails for AI system decisions.",
                "url": "https://example.org/ai-governance-framework-2025",
                "source": "AI Policy Institute",
                "date": "2025-01-15",
                "tags": ["governance", "ai", "policy", "framework"],
                "authority_score": 0.95
            },
            {
                "id": "factual_ai_1", 
                "title": "Factual AI Systems and Hallucination Prevention",
                "content": "Factual AI systems use retrieval-augmented generation to ground responses in verifiable sources. Key techniques include source verification, claim extraction, and consistency checking across multiple authoritative documents.",
                "url": "https://example.org/factual-ai-systems",
                "source": "AI Research Journal",
                "date": "2025-02-01",
                "tags": ["factual", "ai", "hallucination", "rag"],
                "authority_score": 0.92
            },
            {
                "id": "quantum_computing_1",
                "title": "Quantum-Inspired Task Planning",
                "content": "Quantum-inspired algorithms apply quantum mechanical principles like superposition and entanglement to classical optimization problems. Task planning systems can use these concepts to explore multiple solution paths simultaneously.",
                "url": "https://example.org/quantum-task-planning",
                "source": "Quantum Computing Review",
                "date": "2025-01-20",
                "tags": ["quantum", "planning", "optimization", "algorithms"],
                "authority_score": 0.88
            },
            {
                "id": "rag_systems_1",
                "title": "Retrieval-Augmented Generation Best Practices",
                "content": "RAG systems combine retrieval and generation for factual responses. Best practices include using hybrid retrieval methods, implementing source ranking, and maintaining knowledge base quality through regular updates.",
                "url": "https://example.org/rag-best-practices",
                "source": "NLP Engineering Guide",
                "date": "2025-01-10",
                "tags": ["rag", "retrieval", "generation", "nlp"],
                "authority_score": 0.90
            },
            {
                "id": "security_ai_1",
                "title": "AI Security and Zero-Trust Architectures",
                "content": "AI systems require robust security measures including input validation, output filtering, and access controls. Zero-trust architectures ensure that all AI interactions are authenticated and authorized.",
                "url": "https://example.org/ai-security-zero-trust",
                "source": "Cybersecurity Institute",
                "date": "2025-01-25",
                "tags": ["security", "ai", "zero-trust", "authentication"],
                "authority_score": 0.94
            },
            {
                "id": "performance_1",
                "title": "High-Performance RAG System Optimization",
                "content": "Optimizing RAG systems involves caching strategies, parallel processing, and efficient vector indexing. Performance improvements can be achieved through batch processing and smart cache invalidation policies.",
                "url": "https://example.org/rag-performance-optimization",
                "source": "Systems Engineering Journal",
                "date": "2025-02-05",
                "tags": ["performance", "optimization", "caching", "indexing"],
                "authority_score": 0.87
            },
            {
                "id": "compliance_1",
                "title": "GDPR Compliance for AI Systems",
                "content": "AI systems must comply with GDPR requirements including data minimization, purpose limitation, and user consent. Organizations must implement privacy-by-design principles and maintain detailed processing records.",
                "url": "https://example.org/gdpr-ai-compliance",
                "source": "Privacy Law Review",
                "date": "2025-01-30",
                "tags": ["gdpr", "privacy", "compliance", "data-protection"],
                "authority_score": 0.96
            },
            {
                "id": "monitoring_1",
                "title": "AI System Monitoring and Observability",
                "content": "Comprehensive monitoring includes performance metrics, accuracy tracking, and bias detection. Observability platforms should provide real-time dashboards and automated alerting for system anomalies.",
                "url": "https://example.org/ai-system-monitoring",
                "source": "DevOps Intelligence",
                "date": "2025-01-18",
                "tags": ["monitoring", "observability", "metrics", "alerts"],
                "authority_score": 0.85
            }
        ]
        
        # Add some randomization to make retrieval more realistic
        for doc in demo_docs:
            doc["relevance_score"] = random.uniform(0.7, 1.0)
            doc["last_accessed"] = datetime.utcnow().isoformat()
        
        return demo_docs
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the knowledge base."""
        for doc in documents:
            if "id" not in doc:
                doc["id"] = hashlib.md5(doc.get("content", "").encode()).hexdigest()[:16]
            
            # Ensure required fields
            doc.setdefault("title", "Untitled")
            doc.setdefault("content", "")
            doc.setdefault("tags", [])
            doc.setdefault("authority_score", 0.5)
            doc.setdefault("relevance_score", 0.5)
            doc.setdefault("last_accessed", datetime.utcnow().isoformat())
        
        self.knowledge_base.extend(documents)
        
        # Rebuild indices if ML is available
        if ML_AVAILABLE and self.embedding_model is not None:
            self._build_indices()
        
        self.logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        return {
            "total_documents": len(self.knowledge_base),
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "data_path": self.data_path,
            "embedding_model": self.embedding_model_name,
            "ml_available": ML_AVAILABLE,
            "generation": 1
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check system health."""
        return {
            "knowledge_base_loaded": len(self.knowledge_base) > 0,
            "retrieval_available": True,
            "ml_dependencies": ML_AVAILABLE,
            "embedding_model_loaded": self.embedding_model is not None if ML_AVAILABLE else False,
            "demo_mode": True
        }
```
