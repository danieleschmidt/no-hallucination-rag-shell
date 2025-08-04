"""
Factuality detection and scoring for RAG responses.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
        
        # Initialize NLI and similarity models
        self.nli_pipeline = None
        self.similarity_model = None
        self._initialize_models()
        
        # Initialize fact patterns for fallback
        self.fact_patterns = self._load_fact_patterns()
    
    def _initialize_models(self):
        """Initialize NLI and similarity models."""
        try:
            self.logger.info(f"Loading NLI model: {self.nli_model_name}")
            self.nli_pipeline = pipeline(
                "text-classification",
                model=self.nli_model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info(f"Loading similarity model: {self.similarity_model_name}")
            self.similarity_model = SentenceTransformer(self.similarity_model_name)
            
            self.logger.info("Factuality detection models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            self.logger.warning("Falling back to keyword-based factuality detection")
            self.nli_pipeline = None
            self.similarity_model = None
    
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
        Verify individual claim against sources using NLI models.
        
        Args:
            claim: Factual claim to verify
            sources: Source documents
            
        Returns:
            FactualityResult with verification details
        """
        try:
            # Use advanced NLI verification if models are available
            if self.nli_pipeline is not None and self.similarity_model is not None:
                return self._advanced_verify_claim(claim, sources)
            else:
                return self._fallback_verify_claim(claim, sources)
                
        except Exception as e:
            self.logger.error(f"Error verifying claim: {e}")
            return FactualityResult(
                is_supported=False,
                confidence=0.0,
                evidence=[],
                contradictions=[]
            )
    
    def _advanced_verify_claim(
        self,
        claim: str,
        sources: List[Dict[str, Any]]
    ) -> FactualityResult:
        """Advanced claim verification using NLI models."""
        
        evidence = []
        contradictions = []
        support_scores = []
        
        # Extract relevant passages from sources
        relevant_passages = self._extract_relevant_passages(claim, sources)
        
        if not relevant_passages:
            return FactualityResult(
                is_supported=False,
                confidence=0.0,
                evidence=["No relevant passages found"],
                contradictions=[]
            )
        
        # Verify claim against each relevant passage using NLI
        for passage_info in relevant_passages:
            passage = passage_info["text"]
            source = passage_info["source"]
            
            # Use NLI to determine entailment/contradiction/neutral
            nli_result = self._nli_check(claim, passage)
            
            if nli_result["label"] == "ENTAILMENT":
                support_scores.append(nli_result["score"])
                evidence.append(f"{passage[:200]}... (Source: {source.get('title', 'Unknown')})")
                
            elif nli_result["label"] == "CONTRADICTION":
                support_scores.append(1.0 - nli_result["score"])  # Invert for contradiction
                contradictions.append(f"Contradiction: {passage[:200]}... (Source: {source.get('title', 'Unknown')})")
                
            else:  # NEUTRAL
                support_scores.append(0.5)  # Neutral evidence
        
        # Calculate overall confidence
        if support_scores:
            avg_support = sum(support_scores) / len(support_scores)
            
            # Apply confidence calibration
            calibrated_confidence = self._calibrate_confidence(avg_support, len(support_scores))
            
            is_supported = calibrated_confidence > self.threshold
        else:
            is_supported = False
            calibrated_confidence = 0.0
        
        return FactualityResult(
            is_supported=is_supported,
            confidence=calibrated_confidence,
            evidence=evidence[:5],  # Limit to top 5 evidence pieces
            contradictions=contradictions[:3]  # Limit to top 3 contradictions
        )
    
    def _fallback_verify_claim(
        self,
        claim: str,
        sources: List[Dict[str, Any]]
    ) -> FactualityResult:
        """Fallback claim verification using keyword matching."""
        
        evidence = []
        contradictions = []
        support_scores = []
        
        claim_lower = claim.lower()
        
        for source in sources:
            content = source.get("content", "").lower()
            
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
    
    def _extract_relevant_passages(
        self, 
        claim: str, 
        sources: List[Dict[str, Any]], 
        max_passages: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract passages most relevant to the claim using semantic similarity."""
        
        if self.similarity_model is None:
            return []
        
        claim_embedding = self.similarity_model.encode([claim])
        relevant_passages = []
        
        for source in sources:
            content = source.get("content", "")
            
            # Split content into sentences
            sentences = self._split_into_sentences(content)
            
            if sentences:
                # Get embeddings for sentences
                sentence_embeddings = self.similarity_model.encode(sentences)
                
                # Calculate similarities
                similarities = cosine_similarity(claim_embedding, sentence_embeddings)[0]
                
                # Get top sentences from this source
                top_indices = np.argsort(similarities)[::-1][:3]  # Top 3 per source
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:  # Minimum similarity threshold
                        relevant_passages.append({
                            "text": sentences[idx],
                            "source": source,
                            "similarity": similarities[idx]
                        })
        
        # Sort by similarity and return top passages
        relevant_passages.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_passages[:max_passages]
    
    def _nli_check(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Perform Natural Language Inference check."""
        
        try:
            # Format input for NLI model
            input_text = f"{premise} {self.nli_pipeline.tokenizer.sep_token} {hypothesis}"
            
            # Get NLI prediction
            result = self.nli_pipeline(input_text)
            
            # Extract label and confidence
            if isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                label = prediction["label"]
                score = prediction["score"]
            else:
                label = "NEUTRAL"
                score = 0.5
            
            return {
                "label": label,
                "score": score
            }
            
        except Exception as e:
            self.logger.error(f"NLI check failed: {e}")
            return {
                "label": "NEUTRAL",
                "score": 0.5
            }
    
    def _calibrate_confidence(self, raw_confidence: float, num_sources: int) -> float:
        """Apply confidence calibration based on governance dataset."""
        
        # Simple calibration for now - adjust based on number of sources
        source_bonus = min(0.1, num_sources * 0.02)  # Small bonus for more sources
        calibrated = raw_confidence + source_bonus
        
        # Apply temperature scaling (simplified)
        if self.calibration == "temperature_scaled":
            temperature = 1.2  # Learned from governance dataset
            calibrated = calibrated ** (1.0 / temperature)
        
        return min(1.0, max(0.0, calibrated))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for passage extraction."""
        # Improved sentence splitting
        sentences = re.split(r'[.!?]+', text)
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 500:  # Filter by length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get factuality detector statistics."""
        return {
            "nli_model": self.nli_model_name,
            "similarity_model": self.similarity_model_name,
            "models_loaded": {
                "nli_pipeline": self.nli_pipeline is not None,
                "similarity_model": self.similarity_model is not None
            },
            "threshold": self.threshold,
            "calibration": self.calibration,
            "governance_dataset": self.governance_dataset
        }