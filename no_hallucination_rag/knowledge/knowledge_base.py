"""
Knowledge base management for RAG system.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import json
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from .document_processor import DocumentProcessor, DocumentChunk, ProcessingStats


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
        storage_path: str = "data/knowledge_bases",
        processor_config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.storage_path = storage_path
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, DocumentChunk] = {}  # Store processed chunks
        self.metadata = {
            "name": name,
            "created": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "document_count": 0,
            "chunk_count": 0
        }
        self.logger = logging.getLogger(__name__)
        
        # Initialize document processor
        self.processor = DocumentProcessor(**(processor_config or {}))
        
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
    
    def ingest_documents(
        self,
        sources: List[Union[str, Dict[str, Any]]],
        batch_size: int = 10,
        max_workers: int = 4
    ) -> ProcessingStats:
        """
        Ingest multiple documents from files or URLs.
        
        Args:
            sources: List of file paths/URLs or dicts with 'path' and 'metadata'
            batch_size: Number of documents to process in each batch
            max_workers: Number of parallel workers
            
        Returns:
            Processing statistics
        """
        # Normalize sources to dict format
        normalized_sources = []
        for source in sources:
            if isinstance(source, str):
                normalized_sources.append({"path": source, "metadata": {}})
            else:
                normalized_sources.append(source)
        
        total_stats = ProcessingStats()
        
        # Process in batches
        for i in range(0, len(normalized_sources), batch_size):
            batch = normalized_sources[i:i + batch_size]
            
            self.logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")
            
            # Process batch with document processor
            batch_stats = self.processor.process_batch(batch, max_workers)
            
            # Add chunks to knowledge base
            self._add_processed_chunks(batch_stats, batch)
            
            # Aggregate statistics
            total_stats.total_documents += batch_stats.total_documents
            total_stats.successful_documents += batch_stats.successful_documents
            total_stats.failed_documents += batch_stats.failed_documents
            total_stats.total_chunks += batch_stats.total_chunks
            total_stats.total_tokens += batch_stats.total_tokens
            total_stats.processing_time += batch_stats.processing_time
            total_stats.errors.extend(batch_stats.errors)
        
        self.logger.info(
            f"Document ingestion complete: {total_stats.successful_documents}/{total_stats.total_documents} "
            f"documents, {total_stats.total_chunks} chunks, {total_stats.total_tokens} tokens"
        )
        
        return total_stats
    
    def add_document_from_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add document from file with advanced processing.
        
        Args:
            file_path: Path to document file
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        # Process document with advanced processor
        chunks = self.processor.process_document(file_path, metadata)
        
        if not chunks:
            raise ValueError(f"Failed to process document: {file_path}")
        
        # Create document ID from file path
        file_path_obj = Path(file_path)
        doc_id = f"doc_{file_path_obj.stem}_{len(self.documents) + 1}"
        
        # Combine chunk content for document
        full_content = "\n\n".join(chunk.content for chunk in chunks)
        
        # Create document with original metadata from first chunk
        base_metadata = chunks[0].metadata if chunks else {}
        base_metadata.update(metadata or {})
        
        document = Document(
            id=doc_id,
            content=full_content,
            metadata=base_metadata,
            chunks=[chunk.content for chunk in chunks]
        )
        
        # Store document and chunks
        self.documents[doc_id] = document
        
        for chunk in chunks:
            chunk.metadata["document_id"] = doc_id
            self.chunks[chunk.id] = chunk
        
        # Update metadata
        self.metadata["document_count"] = len(self.documents)
        self.metadata["chunk_count"] = len(self.chunks)
        
        self.logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        
        return doc_id
    
    def add_document_from_url(
        self,
        url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add document from URL with advanced processing.
        
        Args:
            url: URL to document
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        # Process document from URL
        chunks = self.processor.process_document(url, metadata)
        
        if not chunks:
            raise ValueError(f"Failed to process document from URL: {url}")
        
        # Create document ID from URL
        doc_id = f"doc_url_{len(self.documents) + 1}"
        
        # Combine chunk content for document
        full_content = "\n\n".join(chunk.content for chunk in chunks)
        
        # Create document with metadata from first chunk
        base_metadata = chunks[0].metadata if chunks else {}
        base_metadata.update(metadata or {})
        
        document = Document(
            id=doc_id,
            content=full_content,
            metadata=base_metadata,
            chunks=[chunk.content for chunk in chunks]
        )
        
        # Store document and chunks
        self.documents[doc_id] = document
        
        for chunk in chunks:
            chunk.metadata["document_id"] = doc_id
            self.chunks[chunk.id] = chunk
        
        # Update metadata
        self.metadata["document_count"] = len(self.documents)
        self.metadata["chunk_count"] = len(self.chunks)
        
        self.logger.info(f"Added document {doc_id} from URL with {len(chunks)} chunks")
        
        return doc_id
    
    def _add_processed_chunks(self, stats: ProcessingStats, batch: List[Dict[str, Any]]):
        """Add processed chunks from batch processing."""
        # This would be called after batch processing
        # For now, we'll process each document individually
        for doc_info in batch:
            try:
                path = doc_info["path"]
                metadata = doc_info.get("metadata", {})
                
                if path.startswith(('http://', 'https://')):
                    self.add_document_from_url(path, metadata)
                else:
                    self.add_document_from_file(path, metadata)
                    
            except Exception as e:
                self.logger.error(f"Failed to add document {doc_info.get('path', 'unknown')}: {e}")
    
    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search document chunks for more precise retrieval.
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            filters: Optional metadata filters
            
        Returns:
            List of matching chunks with relevance scores
        """
        results = []
        query_lower = query.lower()
        
        for chunk_id, chunk in self.chunks.items():
            # Apply filters if provided
            if filters and not self._matches_filters(chunk.metadata, filters):
                continue
            
            # Simple text matching for Generation 1
            content_lower = chunk.content.lower()
            
            # Calculate relevance score
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            
            overlap = len(query_words.intersection(content_words))
            relevance_score = overlap / max(len(query_words), 1)
            
            if relevance_score > 0:
                # Get parent document info
                doc_id = chunk.metadata.get("document_id", "unknown")
                parent_doc = self.documents.get(doc_id)
                
                results.append({
                    "id": chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "relevance_score": relevance_score,
                    "title": chunk.metadata.get("title", parent_doc.metadata.get("title", f"Chunk {chunk_id}") if parent_doc else f"Chunk {chunk_id}"),
                    "url": chunk.metadata.get("url", parent_doc.metadata.get("url", "") if parent_doc else ""),
                    "date": chunk.metadata.get("date", parent_doc.metadata.get("date", "") if parent_doc else ""),
                    "type": chunk.metadata.get("type", "chunk"),
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count
                })
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]
    
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
        total_chunk_tokens = sum(chunk.token_count for chunk in self.chunks.values())
        
        return {
            "name": self.name,
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "total_content_length": total_content_length,
            "total_chunk_tokens": total_chunk_tokens,
            "average_document_length": total_content_length / max(len(self.documents), 1),
            "average_chunk_tokens": total_chunk_tokens / max(len(self.chunks), 1),
            "processor_stats": self.processor.get_stats(),
            "created": self.metadata.get("created"),
            "last_modified": datetime.utcnow().isoformat()
        }