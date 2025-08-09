"""
Vector database integration for efficient semantic search.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import json
import pickle
import os
from datetime import datetime
from pathlib import Path
import hashlib

# Vector database libraries with fallback
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    faiss = None
    SentenceTransformer = None

from .document_processor import DocumentChunk


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    embedding: Optional[List[float]] = None


@dataclass
class IndexStats:
    """Vector index statistics."""
    total_vectors: int
    dimension: int
    index_type: str
    memory_usage_mb: float
    build_time: float
    last_updated: str


class VectorStore:
    """High-performance vector database for semantic search."""
    
    def __init__(
        self,
        name: str,
        embedding_model: str = "all-mpnet-base-v2",
        index_type: str = "IVF",
        storage_path: str = "data/vector_stores",
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10,  # Number of clusters to search
    ):
        self.name = name
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.storage_path = Path(storage_path)
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.chunk_map = {}  # Maps vector ID to chunk info
        self.dimension = None
        
        # Statistics
        self.stats = IndexStats(
            total_vectors=0,
            dimension=0,
            index_type=index_type,
            memory_usage_mb=0.0,
            build_time=0.0,
            last_updated=datetime.utcnow().isoformat()
        )
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        if not ML_AVAILABLE:
            self.logger.warning("ML dependencies not available - VectorStore disabled")
            self.embedding_model = None
            self.dimension = 384  # Default dimension
            self.stats.dimension = self.dimension
            return
            
        try:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Get embedding dimension
            test_embedding = self.embedding_model.encode(["test"])
            self.dimension = test_embedding.shape[1]
            self.stats.dimension = self.dimension
            
            self.logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            self.dimension = 384  # Default fallback
    
    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of chunks added
        """
        if not ML_AVAILABLE or not self.embedding_model:
            self.logger.warning("Vector store not available - skipping chunk addition")
            return 0
            
        import time
        start_time = time.time()
        
        if not chunks:
            return 0
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Generate embeddings in batches
        all_embeddings = []
        chunk_texts = [chunk.content for chunk in chunks]
        
        for i in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        
        # Initialize or extend FAISS index
        if self.faiss_index is None:
            self._initialize_faiss_index(embeddings.shape[1])
        
        # Add embeddings to index
        start_id = len(self.chunk_map)
        self.faiss_index.add(embeddings)
        
        # Update chunk mapping
        for i, chunk in enumerate(chunks):
            vector_id = start_id + i
            self.chunk_map[vector_id] = {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index
            }
            
            # Store embedding in chunk object
            chunk.embedding = embeddings[i].tolist()
        
        # Update statistics
        self.stats.total_vectors = len(self.chunk_map)
        self.stats.build_time = time.time() - start_time
        self.stats.last_updated = datetime.utcnow().isoformat()
        
        self.logger.info(
            f"Added {len(chunks)} chunks in {self.stats.build_time:.2f}s. "
            f"Total vectors: {self.stats.total_vectors}"
        )
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[VectorSearchResult]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results ordered by similarity
        """
        if self.faiss_index is None or len(self.chunk_map) == 0:
            self.logger.warning("Vector store is empty")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).astype(np.float32)
            
            # Search in FAISS index
            # Request more results to account for filtering
            search_k = min(top_k * 3, len(self.chunk_map))
            similarities, indices = self.faiss_index.search(query_embedding, search_k)
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                    
                chunk_info = self.chunk_map.get(idx)
                if chunk_info is None:
                    continue
                
                # Apply similarity threshold
                if similarity < min_similarity:
                    continue
                
                # Apply filters if provided
                if filters and not self._matches_filters(chunk_info["metadata"], filters):
                    continue
                
                result = VectorSearchResult(
                    chunk_id=chunk_info["chunk_id"],
                    content=chunk_info["content"],
                    metadata=chunk_info["metadata"],
                    similarity_score=float(similarity)
                )
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
            
            self.logger.debug(f"Vector search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            vector_weight: Weight for vector similarity scores
            keyword_weight: Weight for keyword matching scores
            filters: Optional metadata filters
            
        Returns:
            List of search results with combined scores
        """
        # Get vector search results
        vector_results = self.search(query, top_k * 2, filters)
        
        # Calculate keyword scores for all chunks
        keyword_scores = self._calculate_keyword_scores(query, vector_results)
        
        # Combine scores
        combined_results = []
        for i, result in enumerate(vector_results):
            keyword_score = keyword_scores.get(result.chunk_id, 0.0)
            
            # Normalize scores to [0, 1]
            normalized_vector_score = max(0, result.similarity_score)
            normalized_keyword_score = keyword_score
            
            # Calculate combined score
            combined_score = (
                vector_weight * normalized_vector_score +
                keyword_weight * normalized_keyword_score
            )
            
            # Update result with combined score
            result.similarity_score = combined_score
            combined_results.append(result)
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return combined_results[:top_k]
    
    def _initialize_faiss_index(self, dimension: int):
        """Initialize FAISS index based on configuration."""
        self.logger.info(f"Initializing FAISS index: {self.index_type}")
        
        if self.index_type == "Flat":
            # Brute force, exact search
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
        elif self.index_type == "IVF":
            # Inverted file index for faster approximate search
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            
            # Need to train the index (will be done when we have enough vectors)
            self.index_needs_training = True
            
        elif self.index_type == "HNSW":
            # Hierarchical navigable small world for very fast search
            self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
            self.faiss_index.hnsw.efConstruction = 40
            self.faiss_index.hnsw.efSearch = 16
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        self.logger.info(f"FAISS index initialized: {type(self.faiss_index).__name__}")
    
    def _calculate_keyword_scores(
        self, 
        query: str, 
        results: List[VectorSearchResult]
    ) -> Dict[str, float]:
        """Calculate keyword matching scores for results."""
        query_words = set(query.lower().split())
        scores = {}
        
        for result in results:
            content_words = set(result.content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = query_words.intersection(content_words)
            union = query_words.union(content_words)
            
            if union:
                jaccard_score = len(intersection) / len(union)
            else:
                jaccard_score = 0.0
            
            scores[result.chunk_id] = jaccard_score
        
        return scores
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save vector store to disk.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where vector store was saved
        """
        if filepath is None:
            filepath = self.storage_path / f"{self.name}_vectorstore"
        else:
            filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            if self.faiss_index is not None:
                index_path = filepath / "faiss.index"
                faiss.write_index(self.faiss_index, str(index_path))
            
            # Save chunk mapping
            chunk_map_path = filepath / "chunk_map.pkl"
            with open(chunk_map_path, 'wb') as f:
                pickle.dump(self.chunk_map, f)
            
            # Save metadata and stats
            metadata = {
                "name": self.name,
                "embedding_model": self.embedding_model_name,
                "index_type": self.index_type,
                "dimension": self.dimension,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "stats": {
                    "total_vectors": self.stats.total_vectors,
                    "dimension": self.stats.dimension,
                    "index_type": self.stats.index_type,
                    "memory_usage_mb": self.stats.memory_usage_mb,
                    "build_time": self.stats.build_time,
                    "last_updated": self.stats.last_updated
                },
                "saved_at": datetime.utcnow().isoformat()
            }
            
            metadata_path = filepath / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved vector store to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load(self, filepath: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            filepath: Path to saved vector store
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Vector store not found: {filepath}")
        
        try:
            # Load metadata
            metadata_path = filepath / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify compatibility
            if metadata["embedding_model"] != self.embedding_model_name:
                self.logger.warning(
                    f"Embedding model mismatch: expected {self.embedding_model_name}, "
                    f"found {metadata['embedding_model']}"
                )
            
            # Load FAISS index
            index_path = filepath / "faiss.index"
            if index_path.exists():
                self.faiss_index = faiss.read_index(str(index_path))
                self.faiss_index.nprobe = self.nprobe  # Set search parameters
            
            # Load chunk mapping
            chunk_map_path = filepath / "chunk_map.pkl"
            with open(chunk_map_path, 'rb') as f:
                self.chunk_map = pickle.load(f)
            
            # Update stats
            stats_data = metadata.get("stats", {})
            self.stats = IndexStats(
                total_vectors=stats_data.get("total_vectors", len(self.chunk_map)),
                dimension=stats_data.get("dimension", self.dimension),
                index_type=stats_data.get("index_type", self.index_type),
                memory_usage_mb=stats_data.get("memory_usage_mb", 0.0),
                build_time=stats_data.get("build_time", 0.0),
                last_updated=stats_data.get("last_updated", datetime.utcnow().isoformat())
            )
            
            self.logger.info(
                f"Loaded vector store from {filepath}. "
                f"Vectors: {self.stats.total_vectors}, Dimension: {self.stats.dimension}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            raise
    
    def optimize_index(self) -> None:
        """Optimize the index for better performance."""
        if self.faiss_index is None or len(self.chunk_map) == 0:
            return
        
        try:
            if self.index_type == "IVF" and hasattr(self, 'index_needs_training'):
                # Train IVF index if it hasn't been trained yet
                if len(self.chunk_map) >= self.nlist:
                    self.logger.info("Training IVF index...")
                    # Get all embeddings for training
                    all_embeddings = []
                    for chunk_info in self.chunk_map.values():
                        if "embedding" in chunk_info:
                            all_embeddings.append(chunk_info["embedding"])
                    
                    if all_embeddings:
                        training_data = np.array(all_embeddings).astype(np.float32)
                        self.faiss_index.train(training_data)
                        self.index_needs_training = False
                        self.logger.info("IVF index training completed")
            
        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        memory_usage = 0.0
        if self.faiss_index is not None:
            # Estimate memory usage (approximate)
            memory_usage = (self.stats.total_vectors * self.stats.dimension * 4) / (1024 * 1024)  # 4 bytes per float32
        
        self.stats.memory_usage_mb = memory_usage
        
        return {
            "name": self.name,
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "total_vectors": self.stats.total_vectors,
            "dimension": self.stats.dimension,
            "memory_usage_mb": self.stats.memory_usage_mb,
            "index_ready": self.faiss_index is not None,
            "build_time": self.stats.build_time,
            "last_updated": self.stats.last_updated,
            "search_parameters": {
                "nlist": self.nlist,
                "nprobe": self.nprobe
            }
        }
    
    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """
        Delete vectors matching the given filters.
        
        Args:
            filters: Metadata filters to match
            
        Returns:
            Number of vectors deleted
        """
        # Find matching vector IDs
        to_delete = []
        for vector_id, chunk_info in self.chunk_map.items():
            if self._matches_filters(chunk_info["metadata"], filters):
                to_delete.append(vector_id)
        
        if not to_delete:
            return 0
        
        # Remove from chunk map
        for vector_id in to_delete:
            del self.chunk_map[vector_id]
        
        # Note: FAISS doesn't support efficient deletion
        # In production, we'd need to rebuild the index
        # For now, we'll mark it as needing rebuild
        self.logger.warning(
            f"Deleted {len(to_delete)} vectors from mapping. "
            "Index rebuild recommended for optimal performance."
        )
        
        # Update stats
        self.stats.total_vectors = len(self.chunk_map)
        self.stats.last_updated = datetime.utcnow().isoformat()
        
        return len(to_delete)
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        self.faiss_index = None
        self.chunk_map.clear()
        self.stats.total_vectors = 0
        self.stats.last_updated = datetime.utcnow().isoformat()
        self.logger.info("Vector store cleared")


class VectorStoreManager:
    """Manage multiple vector stores."""
    
    def __init__(self, storage_path: str = "data/vector_stores"):
        self.storage_path = Path(storage_path)
        self.stores: Dict[str, VectorStore] = {}
        self.logger = logging.getLogger(__name__)
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def create_store(
        self,
        name: str,
        embedding_model: str = "all-mpnet-base-v2",
        index_type: str = "IVF"
    ) -> VectorStore:
        """Create a new vector store."""
        if name in self.stores:
            raise ValueError(f"Vector store '{name}' already exists")
        
        store = VectorStore(
            name=name,
            embedding_model=embedding_model,
            index_type=index_type,
            storage_path=str(self.storage_path)
        )
        
        self.stores[name] = store
        self.logger.info(f"Created vector store: {name}")
        
        return store
    
    def get_store(self, name: str) -> Optional[VectorStore]:
        """Get an existing vector store."""
        return self.stores.get(name)
    
    def load_store(self, name: str, filepath: Optional[str] = None) -> VectorStore:
        """Load a vector store from disk."""
        if filepath is None:
            filepath = self.storage_path / f"{name}_vectorstore"
        
        # Create store instance
        store = VectorStore(name=name, storage_path=str(self.storage_path))
        store.load(str(filepath))
        
        self.stores[name] = store
        self.logger.info(f"Loaded vector store: {name}")
        
        return store
    
    def list_stores(self) -> List[str]:
        """List all available vector stores."""
        return list(self.stores.keys())
    
    def delete_store(self, name: str) -> bool:
        """Delete a vector store."""
        if name in self.stores:
            del self.stores[name]
            
            # Delete files from disk
            store_path = self.storage_path / f"{name}_vectorstore"
            if store_path.exists():
                import shutil
                shutil.rmtree(store_path)
            
            self.logger.info(f"Deleted vector store: {name}")
            return True
        
        return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all vector stores."""
        return {name: store.get_stats() for name, store in self.stores.items()}