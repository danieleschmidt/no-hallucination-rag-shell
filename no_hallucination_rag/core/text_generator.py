"""
Text generation component with LLM integration for answer synthesis.
Generation 1: Template-based generation with rule-based synthesis.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
import random
from datetime import datetime


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
        model_name: str = "rule_based_generator",
        config: Optional[GenerationConfig] = None,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.config = config or GenerationConfig()
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Generation 1: Template-based answer synthesis
        self.answer_templates = self._init_answer_templates()
        
        self.logger.info("TextGenerator initialized (Generation 1: Template-based)")
    
    def generate_answer(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        require_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer from query and sources.
        
        Args:
            query: User's question
            sources: Retrieved source documents
            require_citations: Whether to include citations
            
        Returns:
            Dict containing generated answer and metadata
        """
        try:
            if not sources:
                return {
                    "answer": "I cannot provide an answer as no reliable sources were found.",
                    "confidence": 0.0,
                    "sources_used": 0,
                    "generation_method": "fallback"
                }
            
            # Extract key information from sources
            key_info = self._extract_key_information(query, sources)
            
            # Select appropriate template
            template = self._select_template(query, key_info)
            
            # Generate answer using template
            answer = self._apply_template(template, query, key_info, sources)
            
            # Add citations if required
            if require_citations:
                answer = self._add_citations(answer, sources)
            
            # Calculate confidence based on source quality
            confidence = self._calculate_confidence(key_info, sources)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources_used": len(sources),
                "generation_method": "template_based",
                "key_concepts": key_info.get("concepts", [])
            }
            
        except Exception as e:
            self.logger.error(f"Error in answer generation: {e}")
            return {
                "answer": f"An error occurred while generating the answer: {e}",
                "confidence": 0.0,
                "sources_used": len(sources),
                "generation_method": "error_fallback"
            }
    
    def _extract_key_information(
        self, 
        query: str, 
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract key information from sources relevant to query."""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        key_info = {
            "concepts": [],
            "facts": [],
            "definitions": [],
            "numbers": [],
            "dates": [],
            "authorities": []
        }
        
        for source in sources:
            content = source.get("content", "")
            title = source.get("title", "")
            
            # Extract concepts (words that appear in both query and source)
            content_words = set(content.lower().split())
            common_words = query_words.intersection(content_words)
            key_info["concepts"].extend(common_words)
            
            # Extract numbers and dates
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', content)
            key_info["numbers"].extend(numbers)
            
            dates = re.findall(r'\b\d{4}\b', content)
            key_info["dates"].extend(dates)
            
            # Extract definitions (sentences with "is", "are", "means")
            sentences = content.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['is ', 'are ', 'means ', 'refers to']):
                    if len(sentence.strip()) < 200:  # Keep definitions concise
                        key_info["definitions"].append(sentence.strip())
            
            # Extract factual statements (sentences with authoritative language)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(auth in sentence_lower for auth in ['according to', 'research shows', 'study found']):
                    key_info["facts"].append(sentence.strip())
            
            # Track authorities
            source_name = source.get("source", "")
            if source_name and source_name not in key_info["authorities"]:
                key_info["authorities"].append(source_name)
        
        # Remove duplicates and limit items
        for key in key_info:
            if isinstance(key_info[key], list):
                key_info[key] = list(set(key_info[key]))[:5]  # Limit to 5 items each
        
        return key_info
    
    def _select_template(self, query: str, key_info: Dict[str, Any]) -> str:
        """Select appropriate answer template based on query type."""
        
        query_lower = query.lower()
        
        # Question type classification
        if query_lower.startswith(('what is', 'what are')):
            return 'definition'
        elif query_lower.startswith(('how to', 'how do')):
            return 'procedure'
        elif query_lower.startswith(('why', 'what causes')):
            return 'explanation'
        elif query_lower.startswith(('when', 'what year')):
            return 'temporal'
        elif query_lower.startswith(('where')):
            return 'location'
        elif '?' in query:
            return 'general_question'
        else:
            return 'informative'
    
    def _apply_template(
        self, 
        template_type: str, 
        query: str, 
        key_info: Dict[str, Any], 
        sources: List[Dict[str, Any]]
    ) -> str:
        """Apply selected template to generate answer."""
        
        templates = self.answer_templates.get(template_type, self.answer_templates['default'])
        selected_template = random.choice(templates)
        
        # Prepare template variables
        concepts = ", ".join(key_info.get("concepts", [])[:3])
        
        # Get best source content
        best_content = ""
        if sources:
            best_source = max(sources, key=lambda x: x.get("authority_score", 0.5))
            best_content = best_source.get("content", "")[:300]
        
        # Get facts and definitions
        facts = ". ".join(key_info.get("facts", [])[:2])
        definitions = ". ".join(key_info.get("definitions", [])[:2])
        
        # Numbers and dates
        numbers = ", ".join(key_info.get("numbers", [])[:3])
        dates = ", ".join(key_info.get("dates", [])[:3])
        
        # Authorities
        authorities = ", ".join(key_info.get("authorities", [])[:2])
        
        try:
            # Apply template
            answer = selected_template.format(
                query=query,
                concepts=concepts or "relevant topics",
                content=best_content or "available information",
                facts=facts or "supporting evidence",
                definitions=definitions or "key concepts",
                numbers=numbers or "statistical data",
                dates=dates or "temporal information",
                authorities=authorities or "authoritative sources"
            )
            
            # Clean up empty sections
            answer = re.sub(r'\.\s*\.', '.', answer)  # Remove double periods
            answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
            
            return answer.strip()
            
        except Exception as e:
            self.logger.error(f"Template application failed: {e}")
            return f"Based on the available sources, {best_content}"
    
    def _add_citations(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Add citation markers to the answer."""
        
        if not sources:
            return answer
        
        # Simple approach: add citation numbers at the end of sentences
        sentences = answer.split('. ')
        
        for i, sentence in enumerate(sentences):
            if sentence.strip() and i < len(sources):
                citation_num = (i % len(sources)) + 1
                sentences[i] = f"{sentence} [{citation_num}]"
        
        return '. '.join(sentences)
    
    def _calculate_confidence(
        self, 
        key_info: Dict[str, Any], 
        sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for generated answer."""
        
        confidence = 0.5  # Base confidence
        
        # Boost for multiple sources
        if len(sources) >= 2:
            confidence += 0.2
        
        # Boost for authoritative sources
        avg_authority = sum(s.get("authority_score", 0.5) for s in sources) / len(sources)
        confidence += (avg_authority - 0.5) * 0.3
        
        # Boost for factual content
        if key_info.get("facts"):
            confidence += 0.1
        
        if key_info.get("numbers") or key_info.get("dates"):
            confidence += 0.1
        
        # Boost for definitions
        if key_info.get("definitions"):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _init_answer_templates(self) -> Dict[str, List[str]]:
        """Initialize answer templates for different question types."""
        
        return {
            'definition': [
                "Based on the available sources, {concepts} refers to {definitions}. {content}",
                "{definitions}. According to {authorities}, {facts}",
                "The key aspects of {concepts} include {definitions}. {content}"
            ],
            'procedure': [
                "According to the sources, the process involves {content}. {facts}",
                "Based on {authorities}, the recommended approach is: {content}",
                "The available information suggests {content}. {facts}"
            ],
            'explanation': [
                "The sources indicate that {facts}. {content}",
                "According to {authorities}, this occurs because {content}",
                "The available evidence suggests {facts}. {content}"
            ],
            'temporal': [
                "Based on the available information, the timeframe is {dates}. {content}",
                "According to {authorities}, this occurred in {dates}. {facts}",
                "The sources indicate timing around {dates}. {content}"
            ],
            'location': [
                "According to the sources, {content}",
                "The available information indicates {facts}",
                "Based on {authorities}, {content}"
            ],
            'general_question': [
                "Based on the available sources, {content}. {facts}",
                "According to {authorities}, {content}",
                "The information suggests {facts}. {content}"
            ],
            'informative': [
                "The available sources indicate that {content}. {facts}",
                "According to {authorities}, {content}",
                "Based on the information available, {facts}. {content}"
            ],
            'default': [
                "Based on the available sources: {content}",
                "According to the information available: {facts}",
                "The sources indicate: {content}"
            ]
        }
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get text generation statistics."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "temperature": self.config.temperature,
            "generation": 1,
            "method": "template_based"
        }