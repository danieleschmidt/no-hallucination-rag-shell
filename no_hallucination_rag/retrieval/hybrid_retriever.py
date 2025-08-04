"""
Hybrid retrieval combining dense and sparse retrieval methods.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib

from ..knowledge.vector_store import VectorStore, VectorSearchResult


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
        
        # Initialize real embedding model
        self.embedding_model = None
        self.faiss_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        
        # Initialize vector store for advanced retrieval
        self.vector_store = None
        try:
            self.vector_store = VectorStore(
                name="hybrid_retriever",
                embedding_model=embedding_model,
                index_type="IVF"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize vector store: {e}")
        
        # Initialize with mock knowledge base and real models
        self.mock_knowledge_base = self._create_mock_knowledge_base()
        self._initialize_models()
        self._build_indices()
    
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
            # Use vector store if available, otherwise semantic retrieval, then mock
            if self.vector_store is not None and self.vector_store.stats.total_vectors > 0:
                results = self._vector_store_retrieve(query, top_k, filters)
            elif self.embedding_model is not None and self.faiss_index is not None:
                results = self._semantic_retrieve(query, top_k, filters)
            else:
                self.logger.warning("Falling back to mock retrieval - models not loaded")
                results = self._mock_retrieve(query, top_k, filters)
            
            self.logger.info(f"Retrieved {len(results)} sources for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {e}")
            # Fallback to mock retrieval on error
            return self._mock_retrieve(query, top_k, filters)
    
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
    
    def _initialize_models(self):
        """Initialize embedding model and vectorizers."""
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
        try:
            if self.embedding_model is None:
                return
                
            self.documents = self.mock_knowledge_base.copy()
            
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
        
        # Get dense retrieval results
        dense_results = self._dense_retrieve(query, top_k * 2)  # Get more for reranking
        
        # Get sparse retrieval results
        sparse_results = self._sparse_retrieve(query, top_k * 2)
        
        # Combine and rerank results
        combined_results = self._combine_results(dense_results, sparse_results, top_k)
        
        # Apply filters if provided
        if filters:
            combined_results = self._apply_filters(combined_results, filters)
        
        return combined_results[:top_k]
    
    def _dense_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Dense retrieval using sentence embeddings."""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.faiss_index.search(
                query_embedding.astype(np.float32), 
                min(top_k, len(self.documents))
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # Valid index
                    doc = self.documents[idx].copy()
                    doc["dense_score"] = float(score)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Dense retrieval error: {e}")
            return []
    
    def _sparse_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Sparse retrieval using TF-IDF."""
        try:
            # Transform query
            query_vec = self.tfidf_vectorizer.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include if similarity > 0
                    doc = self.documents[idx].copy()
                    doc["sparse_score"] = float(similarities[idx])
                    results.append(doc)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sparse retrieval error: {e}")
            return []
    
    def _combine_results(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Combine dense and sparse results using weighted fusion."""
        
        # Create lookup dictionaries
        dense_lookup = {doc["id"]: doc for doc in dense_results}
        sparse_lookup = {doc["id"]: doc for doc in sparse_results}
        
        # Get all unique document IDs
        all_doc_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())
        
        combined_results = []
        for doc_id in all_doc_ids:
            # Get base document
            doc = dense_lookup.get(doc_id) or sparse_lookup.get(doc_id)
            doc = doc.copy()
            
            # Calculate combined score
            dense_score = dense_lookup.get(doc_id, {}).get("dense_score", 0.0)
            sparse_score = sparse_lookup.get(doc_id, {}).get("sparse_score", 0.0)
            
            # Normalize scores to [0, 1] range
            normalized_dense = max(0, dense_score)  # Already normalized
            normalized_sparse = max(0, sparse_score)
            
            # Weighted combination
            combined_score = (
                self.dense_weight * normalized_dense + 
                self.sparse_weight * normalized_sparse
            )
            
            doc["combined_score"] = combined_score
            doc["dense_score"] = dense_score
            doc["sparse_score"] = sparse_score
            
            combined_results.append(doc)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined_results
    
    def _apply_filters(
        self, 
        results: List[Dict[str, Any]], 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply filters to results."""
        filtered_results = []
        
        for doc in results:
            include = True
            
            for filter_key, filter_value in filters.items():
                if filter_key in doc:
                    if isinstance(filter_value, list):
                        if doc[filter_key] not in filter_value:
                            include = False
                            break
                    else:
                        if doc[filter_key] != filter_value:
                            include = False
                            break
            
            if include:
                filtered_results.append(doc)
        
        return filtered_results
    
    def _vector_store_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Retrieve using vector store with hybrid search."""
        
        try:
            # Use vector store hybrid search
            vector_results = self.vector_store.hybrid_search(
                query=query,
                top_k=top_k,
                vector_weight=self.dense_weight,
                keyword_weight=self.sparse_weight,
                filters=filters
            )
            
            # Convert vector results to standard format
            results = []
            for result in vector_results:
                # Extract metadata for standard format
                metadata = result.metadata
                
                doc_result = {
                    "id": result.chunk_id,
                    "content": result.content,
                    "title": metadata.get("title", "Unknown"),
                    "url": metadata.get("url", ""),
                    "date": metadata.get("date", ""),
                    "type": metadata.get("type", "document"),
                    "authority_score": metadata.get("authority_score", 0.8),
                    "citations": metadata.get("citations", 0),
                    "combined_score": result.similarity_score,
                    "chunk_index": metadata.get("chunk_index", 0),
                    "token_count": metadata.get("token_count", 0)
                }
                results.append(doc_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector store retrieval error: {e}")
            return []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add new documents to the retrieval system."""
        try:
            self.documents.extend(documents)
            
            # Add to vector store if available
            if self.vector_store is not None:
                # Convert documents to chunks for vector store
                from ..knowledge.document_processor import DocumentChunk
                
                chunks = []
                for doc in documents:
                    # Create a simple chunk from the document
                    chunk = DocumentChunk(
                        id=doc.get("id", f"chunk_{len(chunks)}"),
                        content=doc.get("content", ""),
                        metadata=doc,
                        chunk_index=0,
                        token_count=len(doc.get("content", "").split())
                    )
                    chunks.append(chunk)
                
                self.vector_store.add_chunks(chunks)
                self.logger.info(f"Added {len(chunks)} chunks to vector store")
            
            # Also build traditional indices as fallback
            self._build_indices()
            self.logger.info(f"Added {len(documents)} documents to retrieval system")
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
    
    def add_chunks_to_vector_store(self, chunks: List):
        """Add processed document chunks to vector store."""
        if self.vector_store is not None:
            try:
                added_count = self.vector_store.add_chunks(chunks)
                self.logger.info(f"Added {added_count} chunks to vector store")
                return added_count
            except Exception as e:
                self.logger.error(f"Error adding chunks to vector store: {e}")
                return 0
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        stats = {
            "num_documents": len(self.documents),
            "embedding_model": self.embedding_model_name,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "faiss_index_ready": self.faiss_index is not None,
            "tfidf_matrix_ready": self.tfidf_matrix is not None,
            "vector_store_available": self.vector_store is not None
        }
        
        # Add vector store stats if available
        if self.vector_store is not None:
            stats["vector_store_stats"] = self.vector_store.get_stats()
        
        return stats
    
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