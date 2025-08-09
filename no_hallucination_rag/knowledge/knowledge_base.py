"""
Knowledge base management for RAG systems.
Generation 1: Basic knowledge base with in-memory storage.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime


class KnowledgeBase:
    """Simple knowledge base for document storage and retrieval."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.documents = []
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"KnowledgeBase '{name}' initialized (Generation 1)")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the knowledge base."""
        self.documents.extend(documents)
        self.logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the knowledge base."""
        return self.documents
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Simple search in knowledge base."""
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc in self.documents:
            content = doc.get("content", "").lower()
            title = doc.get("title", "").lower()
            
            content_words = set(content.split())
            title_words = set(title.split())
            
            content_score = len(query_words.intersection(content_words))
            title_score = len(query_words.intersection(title_words)) * 2
            
            total_score = content_score + title_score
            if total_score > 0:
                scored_docs.append((doc, total_score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "name": self.name,
            "document_count": len(self.documents),
            "generation": 1
        }