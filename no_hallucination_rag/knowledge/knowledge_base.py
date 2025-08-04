"""
Knowledge base management for RAG system.
"""

import logging
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Document:
    """Document stored in knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any]
    chunks: List[str] = None
    embedding: List[float] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


class KnowledgeBase:
    """Manages document storage and retrieval for RAG system."""
    
    def __init__(
        self,
        name: str,
        storage_path: str = "data/knowledge_bases"
    ):
        self.name = name
        self.storage_path = storage_path
        self.documents: Dict[str, Document] = {}
        self.metadata = {
            "name": name,
            "created": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "document_count": 0
        }
        self.logger = logging.getLogger(__name__)
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
    
    def add_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add document to knowledge base.
        
        Args:
            content: Document content
            doc_id: Optional document ID (auto-generated if not provided)
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = f"doc_{len(self.documents) + 1}"
        
        if metadata is None:
            metadata = {}
        
        # Add default metadata
        metadata.update({
            "added": datetime.utcnow().isoformat(),
            "length": len(content),
            "source": metadata.get("source", "unknown")
        })
        
        # Create document
        document = Document(
            id=doc_id,
            content=content,
            metadata=metadata
        )
        
        # Process document (chunking, embedding in Generation 2)
        document.chunks = self._chunk_document(content)
        
        # Store document
        self.documents[doc_id] = document
        self.metadata["document_count"] = len(self.documents)
        
        self.logger.info(f"Added document {doc_id} to knowledge base {self.name}")
        
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents in knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of matching documents with relevance scores
        """
        results = []
        query_lower = query.lower()
        
        for doc_id, document in self.documents.items():
            # Apply filters if provided
            if filters and not self._matches_filters(document.metadata, filters):
                continue
            
            # Simple text matching for Generation 1
            content_lower = document.content.lower()
            
            # Calculate relevance score
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            
            overlap = len(query_words.intersection(content_words))
            relevance_score = overlap / max(len(query_words), 1)
            
            if relevance_score > 0:
                results.append({
                    "id": doc_id,
                    "content": document.content,
                    "metadata": document.metadata,
                    "relevance_score": relevance_score,
                    "title": document.metadata.get("title", f"Document {doc_id}"),
                    "url": document.metadata.get("url", ""),
                    "date": document.metadata.get("date", ""),
                    "type": document.metadata.get("type", "document")
                })
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]
    
    def build_index(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        index_type: str = "hnsw",
        ef_construction: int = 200
    ) -> None:
        """Build search index for fast retrieval."""
        # Placeholder for Generation 1
        # Generation 2 will implement vector indexing
        self.logger.info(f"Index building placeholder - will implement in Generation 2")
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save knowledge base to disk.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where knowledge base was saved
        """
        if filepath is None:
            filepath = os.path.join(self.storage_path, f"{self.name}.json")
        
        # Prepare data for serialization
        kb_data = {
            "metadata": self.metadata,
            "documents": {}
        }
        
        for doc_id, document in self.documents.items():
            kb_data["documents"][doc_id] = {
                "id": document.id,
                "content": document.content,
                "metadata": document.metadata,
                "chunks": document.chunks
            }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(kb_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved knowledge base to {filepath}")
        return filepath
    
    def load(self, filepath: str) -> None:
        """Load knowledge base from disk."""
        with open(filepath, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        self.metadata = kb_data["metadata"]
        self.documents = {}
        
        for doc_id, doc_data in kb_data["documents"].items():
            document = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                chunks=doc_data.get("chunks", [])
            )
            self.documents[doc_id] = document
        
        self.logger.info(f"Loaded knowledge base from {filepath}")
    
    def _chunk_document(self, content: str, chunk_size: int = 512) -> List[str]:
        """Chunk document for retrieval."""
        # Simple chunking for Generation 1
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches filters."""
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        total_content_length = sum(len(doc.content) for doc in self.documents.values())
        
        return {
            "name": self.name,
            "document_count": len(self.documents),
            "total_content_length": total_content_length,
            "average_document_length": total_content_length / max(len(self.documents), 1),
            "created": self.metadata.get("created"),
            "last_modified": datetime.utcnow().isoformat()
        }