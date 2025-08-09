"""
Factuality detection and scoring for RAG responses.
Generation 1: Basic implementation with rule-based scoring.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json
import random


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
        threshold: float = 0.8,
        nli_model: str = "microsoft/deberta-large-mnli",
        similarity_model: str = "all-mpnet-base-v2"
    ):
        self.governance_dataset = governance_dataset
        self.calibration = calibration
        self.threshold = threshold
        self.nli_model_name = nli_model
        self.similarity_model_name = similarity_model
        self.logger = logging.getLogger(__name__)
        
        # Generation 1: Simple rule-based patterns
        self.factual_patterns = self._init_factual_patterns()
        
        self.logger.info("FactualityDetector initialized (Generation 1: Rule-based)")
    
    def verify_answer(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Verify factuality of an answer against sources.
        
        Args:
            question: Original question
            answer: Generated answer to verify
            sources: Source documents used for generation
            
        Returns:
            Factuality score between 0.0 and 1.0
        """
        try:
            # Extract claims from answer
            claims = self._extract_claims(answer)
            
            if not claims:
                return 0.5  # Neutral score if no claims found
            
            # Verify each claim against sources
            claim_scores = []
            for claim in claims:
                score = self._verify_claim(claim, sources)
                claim_scores.append(score)
            
            # Calculate overall factuality score
            overall_score = sum(claim_scores) / len(claim_scores) if claim_scores else 0.0
            
            self.logger.debug(f"Factuality verification: {len(claims)} claims, average score: {overall_score:.3f}")
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            self.logger.error(f"Error in factuality verification: {e}")
            return 0.0
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract verifiable claims from answer text."""
        # Simple sentence-based claim extraction for Generation 1
        sentences = re.split(r'[.!?]+', answer)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Skip questions and commands
            if sentence.endswith('?') or sentence.startswith(('Please', 'Consider', 'Note that')):
                continue
            
            # Skip vague statements
            vague_patterns = ['might', 'could', 'perhaps', 'possibly', 'generally', 'often']
            if any(pattern in sentence.lower() for pattern in vague_patterns):
                continue
            
            claims.append(sentence)
        
        return claims
    
    def _verify_claim(self, claim: str, sources: List[Dict[str, Any]]) -> float:
        """Verify a single claim against source documents."""
        
        if not sources:
            return 0.0
        
        claim_lower = claim.lower()
        
        # Check for direct text overlap with sources
        max_overlap_score = 0.0
        
        for source in sources:
            content = source.get('content', '').lower()
            title = source.get('title', '').lower()
            
            # Simple word overlap scoring
            claim_words = set(claim_lower.split())
            content_words = set(content.split())
            title_words = set(title.split())
            
            # Calculate overlap ratios
            content_overlap = len(claim_words.intersection(content_words)) / max(len(claim_words), 1)
            title_overlap = len(claim_words.intersection(title_words)) / max(len(claim_words), 1)
            
            # Weight title matches higher
            overlap_score = (content_overlap * 0.7) + (title_overlap * 0.3)
            max_overlap_score = max(max_overlap_score, overlap_score)
        
        # Apply factual pattern bonuses
        pattern_bonus = self._check_factual_patterns(claim)
        
        final_score = min(1.0, max_overlap_score + pattern_bonus)
        
        return final_score
    
    def _check_factual_patterns(self, claim: str) -> float:
        """Check claim against factual patterns for bonus scoring."""
        claim_lower = claim.lower()
        bonus = 0.0
        
        # Bonus for specific dates, numbers, citations
        if re.search(r'\b\d{4}\b', claim):  # Years
            bonus += 0.1
        
        if re.search(r'\b\d+%\b', claim):  # Percentages
            bonus += 0.1
        
        # Bonus for authoritative language
        if any(pattern in claim_lower for pattern in ['according to', 'research shows', 'study finds']):
            bonus += 0.1
        
        # Penalty for hedging language
        if any(pattern in claim_lower for pattern in ['believe', 'think', 'assume', 'probably']):
            bonus -= 0.1
        
        return max(-0.2, min(0.3, bonus))
    
    def _init_factual_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for factual language recognition."""
        return {
            'authoritative': [
                'according to', 'research shows', 'study finds', 'data indicates',
                'analysis reveals', 'evidence suggests', 'statistics show'
            ],
            'hedging': [
                'might', 'could', 'perhaps', 'possibly', 'probably', 'likely',
                'believe', 'think', 'assume', 'suppose', 'generally', 'often'
            ],
            'specific': [
                r'\b\d{4}\b',  # Years
                r'\b\d+%\b',   # Percentages
                r'\b\d+\.\d+\b',  # Decimals
                r'\$\d+',      # Dollar amounts
            ]
        }
    
    def verify_claim_detailed(
        self,
        claim: str,
        sources: List[Dict[str, Any]]
    ) -> FactualityResult:
        """Perform detailed verification of a single claim."""
        
        score = self._verify_claim(claim, sources)
        
        # Generate evidence and contradictions
        evidence = []
        contradictions = []
        
        for source in sources:
            content = source.get('content', '')
            title = source.get('title', 'Unknown')
            
            # Simple heuristic for evidence vs contradiction
            claim_words = set(claim.lower().split())
            content_words = set(content.lower().split())
            
            overlap = len(claim_words.intersection(content_words)) / max(len(claim_words), 1)
            
            if overlap > 0.3:
                evidence.append(f"{title}: {content[:200]}...")
            elif overlap < 0.1 and len(content) > 50:
                # Very low overlap might indicate contradiction
                contradictions.append(f"{title}: Potential conflicting information")
        
        return FactualityResult(
            is_supported=score >= self.threshold,
            confidence=score,
            evidence=evidence[:3],  # Limit to top 3 evidence items
            contradictions=contradictions[:2]  # Limit to top 2 contradictions
        )
    
    def batch_verify_claims(
        self,
        claims: List[str],
        sources: List[Dict[str, Any]]
    ) -> List[FactualityResult]:
        """Verify multiple claims in batch."""
        results = []
        
        for claim in claims:
            result = self.verify_claim_detailed(claim, sources)
            results.append(result)
        
        return results
    
    def get_factuality_stats(self) -> Dict[str, Any]:
        """Get factuality detector statistics."""
        return {
            "governance_dataset": self.governance_dataset,
            "calibration": self.calibration,
            "threshold": self.threshold,
            "nli_model": self.nli_model_name,
            "similarity_model": self.similarity_model_name,
            "generation": 1,
            "method": "rule_based"
        }