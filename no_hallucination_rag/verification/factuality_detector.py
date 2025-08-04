"""
Factuality detection and scoring for RAG responses.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json


@dataclass 
class FactualityResult:
    """Result of factuality verification."""
    is_supported: bool
    confidence: float
    evidence: List[str]
    contradictions: List[str] = None
    
    def __post_init__(self):
        if self.contradictions is None:
            self.contradictions = []


class FactualityDetector:
    """Detects and scores factual accuracy of generated answers."""
    
    def __init__(
        self,
        governance_dataset: str = "whitehouse_2025",
        calibration: str = "temperature_scaled",
        threshold: float = 0.8
    ):
        self.governance_dataset = governance_dataset
        self.calibration = calibration
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize fact patterns for Generation 1
        self.fact_patterns = self._load_fact_patterns()
    
    def verify_answer(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Verify factuality of answer against sources.
        
        Args:
            question: Original query
            answer: Generated answer to verify  
            sources: Supporting sources
            
        Returns:
            Factuality confidence score [0.0, 1.0]
        """
        try:
            # Extract claims from answer
            claims = self._extract_claims(answer)
            
            if not claims:
                return 0.5  # Neutral score for no extractable claims
            
            # Verify each claim against sources
            claim_scores = []
            for claim in claims:
                verification = self.verify_claim(claim, sources)
                claim_scores.append(verification.confidence)
            
            # Calculate overall factuality score
            if claim_scores:
                factuality_score = sum(claim_scores) / len(claim_scores)
            else:
                factuality_score = 0.0
            
            self.logger.info(
                f"Verified {len(claims)} claims, "
                f"factuality score: {factuality_score:.3f}"
            )
            
            return factuality_score
            
        except Exception as e:
            self.logger.error(f"Error in factuality verification: {e}")
            return 0.0
    
    def verify_claim(
        self,
        claim: str,
        sources: List[Dict[str, Any]]
    ) -> FactualityResult:
        """
        Verify individual claim against sources.
        
        Args:
            claim: Factual claim to verify
            sources: Source documents
            
        Returns:
            FactualityResult with verification details
        """
        try:
            evidence = []
            contradictions = []
            support_scores = []
            
            claim_lower = claim.lower()
            
            for source in sources:
                content = source.get("content", "").lower()
                
                # Simple fact checking using keyword/pattern matching
                # In Generation 2, this will use advanced NLI models
                
                support_score = self._calculate_claim_support(claim_lower, content)
                support_scores.append(support_score)
                
                if support_score > 0.7:
                    evidence.append(self._extract_supporting_text(claim, source))
                elif support_score < 0.3:
                    contradictions.append(self._extract_contradicting_text(claim, source))
            
            # Determine overall support
            if support_scores:
                avg_support = sum(support_scores) / len(support_scores)
                is_supported = avg_support > self.threshold
                confidence = avg_support
            else:
                is_supported = False
                confidence = 0.0
            
            return FactualityResult(
                is_supported=is_supported,
                confidence=confidence,
                evidence=evidence,
                contradictions=contradictions
            )
            
        except Exception as e:
            self.logger.error(f"Error verifying claim: {e}")
            return FactualityResult(
                is_supported=False,
                confidence=0.0,
                evidence=[],
                contradictions=[]
            )
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Simple sentence splitting for Generation 1
        # Generation 2 will use dependency parsing and claim detection
        
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                # Simple heuristics for factual claims
                if any(pattern in sentence.lower() for pattern in [
                    'must', 'require', 'establish', 'mandate', 'according to',
                    'percent', '%', 'by', 'include', 'contain', 'state'
                ]):
                    claims.append(sentence)
        
        return claims
    
    def _calculate_claim_support(self, claim: str, content: str) -> float:
        """Calculate how well content supports a claim."""
        # Simple keyword overlap for Generation 1
        claim_words = set(claim.split())
        content_words = set(content.split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
        claim_words -= common_words
        content_words -= common_words
        
        if not claim_words:
            return 0.0
        
        # Calculate overlap score
        overlap = len(claim_words.intersection(content_words))
        overlap_ratio = overlap / len(claim_words)
        
        # Look for negation patterns that might contradict
        negation_patterns = ['not', 'no', 'never', 'cannot', 'does not', 'do not']
        has_negation = any(pattern in content for pattern in negation_patterns)
        
        if has_negation and overlap_ratio > 0.5:
            # High overlap with negation suggests contradiction
            return 0.2
        
        return overlap_ratio
    
    def _extract_supporting_text(self, claim: str, source: Dict[str, Any]) -> str:
        """Extract text that supports the claim."""
        content = source.get("content", "")
        title = source.get("title", "")
        url = source.get("url", "")
        
        # Find most relevant sentence
        sentences = re.split(r'[.!?]+', content)
        claim_words = set(claim.lower().split())
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                sentence_words = set(sentence.lower().split())
                overlap = len(claim_words.intersection(sentence_words))
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence
        
        if best_sentence:
            return f"{best_sentence[:200]}... (Source: {title})"
        else:
            return f"Supporting evidence from: {title}"
    
    def _extract_contradicting_text(self, claim: str, source: Dict[str, Any]) -> str:
        """Extract text that contradicts the claim."""
        content = source.get("content", "")
        title = source.get("title", "")
        
        # Look for sentences with negation and keyword overlap
        sentences = re.split(r'[.!?]+', content)
        claim_words = set(claim.lower().split())
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            overlap = len(claim_words.intersection(sentence_words))
            
            # Check for contradiction patterns
            if overlap > 2 and any(neg in sentence_lower for neg in ['not', 'no', 'never', 'cannot']):
                return f"Potential contradiction: {sentence[:200]}... (Source: {title})"
        
        return f"Potential contradiction in: {title}"
    
    def _load_fact_patterns(self) -> Dict[str, Any]:
        """Load fact checking patterns."""
        # Simplified patterns for Generation 1
        return {
            "date_patterns": [
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b'
            ],
            "numeric_patterns": [
                r'\b\d+(\.\d+)?\s*(percent|%)\b',
                r'\b\d+(\.\d+)?\s*(million|billion|trillion)\b',
                r'\b\$\d+(\.\d+)?\b'
            ],
            "authority_patterns": [
                r'\baccording to\b',
                r'\bas stated by\b',
                r'\breported by\b',
                r'\bin a (study|report|document)\b'
            ]
        }