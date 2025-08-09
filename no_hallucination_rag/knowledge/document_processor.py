"""
Advanced document processing and ingestion pipeline.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import hashlib
import json
import re
import os
from datetime import datetime
from pathlib import Path
import tempfile
import requests
import mimetypes

# Document processing libraries with fallbacks
try:
    import PyPDF2
    from docx import Document as DocxDocument
    import markdown
    from bs4 import BeautifulSoup
    import pandas as pd
    from PIL import Image
    import pytesseract  # OCR for images
    DOC_PROCESSING_AVAILABLE = True
except ImportError:
    DOC_PROCESSING_AVAILABLE = False
    # Mock imports for Generation 1
    PyPDF2 = None
    DocxDocument = None
    markdown = None
    BeautifulSoup = None
    pd = None
    Image = None
    pytesseract = None


@dataclass
class DocumentChunk:
    """Processed document chunk."""
    id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    token_count: int
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"chunk_{content_hash}_{self.chunk_index}"


@dataclass
class ProcessingStats:
    """Document processing statistics."""
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DocumentProcessor:
    """Advanced document processor with multi-format support."""
    
    def __init__(
        self,
        chunk_strategy: str = "semantic",
        chunk_size: int = 512,
        overlap: int = 50,
        max_file_size_mb: int = 100,
        supported_formats: Optional[List[str]] = None
    ):
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_file_size_mb = max_file_size_mb
        self.supported_formats = supported_formats or [
            "pdf", "docx", "txt", "md", "html", "csv", "json", "png", "jpg", "jpeg"
        ]
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer for accurate token counting
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize tokenizer for token counting."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer: {e}. Using word count approximation.")
            self.tokenizer = None
    
    def process_document(
        self,
        document_path_or_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process a single document into chunks.
        
        Args:
            document_path_or_url: Path to local file or URL
            metadata: Additional metadata for the document
            
        Returns:
            List of processed document chunks
        """
        try:
            # Determine if input is URL or file path
            if document_path_or_url.startswith(('http://', 'https://')):
                return self._process_url(document_path_or_url, metadata)
            else:
                return self._process_file(document_path_or_url, metadata)
                
        except Exception as e:
            self.logger.error(f"Failed to process document {document_path_or_url}: {e}")
            return []
    
    def process_batch(
        self,
        documents: List[Dict[str, Any]],
        max_workers: int = 4
    ) -> ProcessingStats:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of document dicts with 'path' and optional 'metadata'
            max_workers: Number of parallel workers
            
        Returns:
            Processing statistics
        """
        import concurrent.futures
        import time
        
        start_time = time.time()
        stats = ProcessingStats()
        all_chunks = []
        
        def process_single(doc_info):
            try:
                path = doc_info["path"]
                metadata = doc_info.get("metadata", {})
                chunks = self.process_document(path, metadata)
                return {"success": True, "chunks": chunks, "path": path}
            except Exception as e:
                return {"success": False, "error": str(e), "path": doc_info.get("path", "unknown")}
        
        # Process documents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single, documents))
        
        # Collect statistics
        stats.total_documents = len(documents)
        
        for result in results:
            if result["success"]:
                stats.successful_documents += 1
                chunks = result["chunks"]
                all_chunks.extend(chunks)
                stats.total_chunks += len(chunks)
                stats.total_tokens += sum(chunk.token_count for chunk in chunks)
            else:
                stats.failed_documents += 1
                stats.errors.append(f"{result['path']}: {result['error']}")
        
        stats.processing_time = time.time() - start_time
        
        self.logger.info(
            f"Batch processing complete: {stats.successful_documents}/{stats.total_documents} "
            f"documents processed, {stats.total_chunks} chunks, {stats.total_tokens} tokens"
        )
        
        return stats
    
    def _process_file(self, file_path: str, metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """Process a local file."""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")
        
        # Determine file type
        file_extension = file_path.suffix.lower().lstrip('.')
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Extract text based on file type
        if file_extension == "pdf":
            text = self._extract_pdf_text(file_path)
        elif file_extension == "docx":
            text = self._extract_docx_text(file_path)
        elif file_extension in ["txt", "md"]:
            text = self._extract_text_file(file_path)
        elif file_extension == "html":
            text = self._extract_html_text(file_path)
        elif file_extension == "csv":
            text = self._extract_csv_text(file_path)
        elif file_extension == "json":
            text = self._extract_json_text(file_path)
        elif file_extension in ["png", "jpg", "jpeg"]:
            text = self._extract_image_text(file_path)
        else:
            raise ValueError(f"No processor for file type: {file_extension}")
        
        # Build metadata
        doc_metadata = {
            "source_path": str(file_path),
            "file_type": file_extension,
            "file_size_bytes": file_path.stat().st_size,
            "processed_at": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        # Create chunks
        chunks = self._create_chunks(text, doc_metadata)
        
        self.logger.info(f"Processed {file_path.name}: {len(chunks)} chunks, {len(text)} characters")
        
        return chunks
    
    def _process_url(self, url: str, metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """Process a document from URL."""
        
        try:
            # Download the content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            try:
                # Determine file type from Content-Type or URL
                content_type = response.headers.get('content-type', '')
                file_extension = self._guess_extension_from_content_type(content_type, url)
                
                # Add URL to filename for proper processing
                final_temp_path = f"{temp_path}.{file_extension}"
                os.rename(temp_path, final_temp_path)
                
                # Add URL metadata
                url_metadata = {
                    "source_url": url,
                    "content_type": content_type,
                    "downloaded_at": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
                
                # Process the temporary file
                chunks = self._process_file(final_temp_path, url_metadata)
                
                return chunks
                
            finally:
                # Clean up temporary file
                for temp_file_path in [temp_path, final_temp_path]:
                    try:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                    except OSError:
                        pass
                        
        except requests.RequestException as e:
            raise ValueError(f"Failed to download from URL {url}: {e}")
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text_parts = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to process PDF {file_path}: {e}")
            raise
        
        return "\n\n".join(text_parts)
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = DocxDocument(file_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            self.logger.error(f"Failed to process DOCX {file_path}: {e}")
            raise
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text or markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # If it's markdown, convert to text
            if file_path.suffix.lower() == '.md':
                html = markdown.markdown(content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to process text file {file_path}: {e}")
            raise
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            return soup.get_text()
            
        except Exception as e:
            self.logger.error(f"Failed to process HTML {file_path}: {e}")
            raise
    
    def _extract_csv_text(self, file_path: Path) -> str:
        """Extract text from CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to readable text
            text_parts = [f"CSV file with {len(df)} rows and {len(df.columns)} columns"]
            text_parts.append(f"Columns: {', '.join(df.columns)}")
            
            # Add sample rows as text
            for idx, row in df.head(10).iterrows():  # First 10 rows
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                text_parts.append(f"Row {idx + 1}: {row_text}")
            
            if len(df) > 10:
                text_parts.append(f"... and {len(df) - 10} more rows")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to process CSV {file_path}: {e}")
            raise
    
    def _extract_json_text(self, file_path: Path) -> str:
        """Extract text from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Convert JSON to readable text
            def json_to_text(obj, prefix=""):
                if isinstance(obj, dict):
                    parts = []
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            parts.append(f"{prefix}{key}:")
                            parts.append(json_to_text(value, prefix + "  "))
                        else:
                            parts.append(f"{prefix}{key}: {value}")
                    return "\n".join(parts)
                elif isinstance(obj, list):
                    parts = []
                    for i, item in enumerate(obj):
                        parts.append(f"{prefix}[{i}]:")
                        parts.append(json_to_text(item, prefix + "  "))
                    return "\n".join(parts)
                else:
                    return f"{prefix}{obj}"
            
            return json_to_text(data)
            
        except Exception as e:
            self.logger.error(f"Failed to process JSON {file_path}: {e}")
            raise
    
    def _extract_image_text(self, file_path: Path) -> str:
        """Extract text from image using OCR."""
        try:
            # Use OCR to extract text from image
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                return f"Image file: {file_path.name} (no extractable text)"
            
            return f"[OCR extracted text from {file_path.name}]\n{text}"
            
        except Exception as e:
            self.logger.warning(f"OCR failed for {file_path}: {e}")
            return f"Image file: {file_path.name} (OCR extraction failed)"
    
    def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create chunks from extracted text."""
        
        if self.chunk_strategy == "semantic":
            return self._semantic_chunking(text, metadata)
        elif self.chunk_strategy == "fixed":
            return self._fixed_chunking(text, metadata)
        elif self.chunk_strategy == "sentence":
            return self._sentence_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunk strategy: {self.chunk_strategy}")
    
    def _semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create semantically coherent chunks."""
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = self._count_tokens(test_chunk)
            
            if token_count <= self.chunk_size or not current_chunk:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk = DocumentChunk(
                        id="",
                        content=current_chunk.strip(),
                        metadata=metadata.copy(),
                        chunk_index=chunk_index,
                        token_count=self._count_tokens(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunk = DocumentChunk(
                id="",
                content=current_chunk.strip(),
                metadata=metadata.copy(),
                chunk_index=chunk_index,
                token_count=self._count_tokens(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create fixed-size chunks with overlap."""
        
        # Simple word-based chunking
        words = text.split()
        chunks = []
        chunk_index = 0
        
        i = 0
        while i < len(words):
            # Create chunk of approximately chunk_size tokens
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk = DocumentChunk(
                id="",
                content=chunk_text,
                metadata=metadata.copy(),
                chunk_index=chunk_index,
                token_count=self._count_tokens(chunk_text)
            )
            chunks.append(chunk)
            
            # Move forward with overlap
            i += max(1, self.chunk_size - self.overlap)
            chunk_index += 1
        
        return chunks
    
    def _sentence_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create chunks based on sentence boundaries."""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            token_count = self._count_tokens(test_chunk)
            
            if token_count <= self.chunk_size or not current_chunk:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk = DocumentChunk(
                        id="",
                        content=current_chunk.strip(),
                        metadata=metadata.copy(),
                        chunk_index=chunk_index,
                        token_count=self._count_tokens(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunk = DocumentChunk(
                id="",
                content=current_chunk.strip(),
                metadata=metadata.copy(),
                chunk_index=chunk_index,
                token_count=self._count_tokens(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback to word count approximation
        return len(text.split())
    
    def _guess_extension_from_content_type(self, content_type: str, url: str) -> str:
        """Guess file extension from content type or URL."""
        
        # Try to get extension from content type
        extension = mimetypes.guess_extension(content_type.split(';')[0])
        if extension:
            return extension.lstrip('.')
        
        # Try to get extension from URL
        url_path = Path(url)
        if url_path.suffix:
            return url_path.suffix.lstrip('.')
        
        # Default to text
        return "txt"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "max_file_size_mb": self.max_file_size_mb,
            "supported_formats": self.supported_formats,
            "tokenizer_available": self.tokenizer is not None
        }