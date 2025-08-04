"""
Text generation component with LLM integration for answer synthesis.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True


class TextGenerator:
    """Generates factual answers from retrieved sources."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        config: Optional[GenerationConfig] = None,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.config = config or GenerationConfig()
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model."""
        try:
            # Try to load a lightweight model first
            self.logger.info(f"Loading text generation model: {self.model_name}")
            
            # For Generation 1, use a pipeline approach
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_full_text=False
            )
            
            self.logger.info("Text generation model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.warning("Falling back to template-based generation")
            self.generator = None
    
    def generate_answer(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        require_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a factual answer from sources.
        
        Args:
            query: User query
            sources: Retrieved and ranked sources
            require_citations: Whether to include citations
            
        Returns:
            Dict with generated answer and metadata
        """
        try:
            # Use LLM generation if available, otherwise template-based
            if self.generator is not None:
                return self._llm_generate(query, sources, require_citations)
            else:
                return self._template_generate(query, sources, require_citations)
                
        except Exception as e:
            self.logger.error(f"Error in answer generation: {e}")
            return self._fallback_generate(query, sources)
    
    def _llm_generate(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        require_citations: bool
    ) -> Dict[str, Any]:
        """Generate answer using LLM."""
        
        # Construct prompt with sources
        prompt = self._build_rag_prompt(query, sources, require_citations)
        
        try:
            # Generate with the model
            generation_kwargs = {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
                "do_sample": self.config.do_sample,
                "num_beams": self.config.num_beams,
                "early_stopping": self.config.early_stopping,
                "pad_token_id": 50256  # GPT-2 pad token
            }
            
            outputs = self.generator(prompt, **generation_kwargs)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"].strip()
                
                # Post-process the generated text
                processed_answer = self._post_process_answer(generated_text, sources)
                
                return {
                    "answer": processed_answer["text"],
                    "citations": processed_answer["citations"],
                    "generation_method": "llm",
                    "model_name": self.model_name,
                    "confidence": 0.8  # Default LLM confidence
                }
            else:
                return self._template_generate(query, sources, require_citations)
                
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return self._template_generate(query, sources, require_citations)
    
    def _template_generate(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        require_citations: bool
    ) -> Dict[str, Any]:
        """Generate answer using template-based approach."""
        
        if not sources:
            return {
                "answer": "I don't have sufficient reliable sources to answer this question.",
                "citations": [],
                "generation_method": "template",
                "confidence": 0.0
            }
        
        # Extract key information from sources
        key_points = self._extract_key_points(query, sources)
        
        # Build structured answer
        answer_parts = []
        citations = []
        
        if len(key_points) > 0:
            answer_parts.append(f"Based on authoritative sources:")
            
            for i, point in enumerate(key_points[:5], 1):  # Limit to top 5 points
                answer_parts.append(f"\n{i}. {point['text']}")
                
                if require_citations:
                    source = point['source']
                    citation = f"[{i}] {source.get('url', source.get('title', 'Unknown source'))}"
                    citations.append(citation)
                    
        else:
            answer_parts.append("The available sources do not contain sufficient information to provide a comprehensive answer to your question.")
        
        # Add source summary
        if len(sources) > 1:
            answer_parts.append(f"\n\nThis information is based on {len(sources)} authoritative sources.")
        
        return {
            "answer": " ".join(answer_parts),
            "citations": citations,
            "generation_method": "template",
            "confidence": min(0.9, len(key_points) * 0.2)
        }
    
    def _fallback_generate(self, query: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback generation when other methods fail."""
        return {
            "answer": "I apologize, but I'm unable to generate a reliable answer at this time. Please try rephrasing your question.",
            "citations": [],
            "generation_method": "fallback",
            "confidence": 0.0
        }
    
    def _build_rag_prompt(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        require_citations: bool
    ) -> str:
        """Build RAG prompt for LLM generation."""
        
        prompt_parts = [
            "You are a factual AI assistant. Answer the question using only the provided sources.",
            "Be accurate, concise, and cite your sources."
        ]
        
        if require_citations:
            prompt_parts.append("Include numbered citations [1], [2], etc. for each claim.")
        
        prompt_parts.append("\nSources:")
        
        for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
            title = source.get('title', 'Unknown')
            content = source.get('content', '')[:300]  # Truncate content
            prompt_parts.append(f"\n[{i}] {title}: {content}")
        
        prompt_parts.append(f"\nQuestion: {query}")
        prompt_parts.append("\nAnswer:")
        
        return "\n".join(prompt_parts)
    
    def _extract_key_points(self, query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key points from sources relevant to the query."""
        
        key_points = []
        query_words = set(query.lower().split())
        
        for source in sources:
            content = source.get('content', '')
            sentences = self._split_into_sentences(content)
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                
                if overlap >= 2 and len(sentence) > 50:  # Minimum relevance and length
                    key_points.append({
                        'text': sentence.strip(),
                        'source': source,
                        'relevance': overlap
                    })
        
        # Sort by relevance and remove duplicates
        key_points.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Remove near-duplicates
        unique_points = []
        for point in key_points:
            is_duplicate = False
            for existing in unique_points:
                if self._calculate_similarity(point['text'], existing['text']) > 0.8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        return unique_points[:10]  # Return top 10 unique points
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (Jaccard similarity)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _post_process_answer(self, text: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Post-process generated answer to extract citations."""
        
        # Extract citations from text
        citation_pattern = r'\[(\d+)\]'
        citations_found = re.findall(citation_pattern, text)
        
        # Build citation list
        citations = []
        for i, citation_num in enumerate(set(citations_found), 1):
            try:
                source_idx = int(citation_num) - 1
                if 0 <= source_idx < len(sources):
                    source = sources[source_idx]
                    citation = f"[{i}] {source.get('url', source.get('title', 'Unknown source'))}"
                    citations.append(citation)
            except (ValueError, IndexError):
                continue
        
        return {
            "text": text,
            "citations": citations
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "model_name": self.model_name,
            "model_loaded": self.generator is not None,
            "device": self.device,
            "config": {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
        }