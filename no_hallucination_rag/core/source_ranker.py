"""
Source ranking and scoring for RAG retrieval.
Generation 1: Simple scoring based on authority and relevance.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime


class SourceRanker:
    """Ranks and scores sources for relevance and authority."""
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        ensemble_method: str = "weighted_vote"
    ):
        self.models = models or ["authority_score", "relevance_score", "recency_score"]
        self.ensemble_method = ensemble_method
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("SourceRanker initialized (Generation 1: Rule-based)")
    
    def rank(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        factors: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank sources by relevance and authority.
        
        Args:
            query: Search query
            sources: List of source documents
            top_k: Number of top sources to return
            factors: Ranking factors to use
            
        Returns:
            Ranked list of sources
        """
        if not sources:
            return []
        
        factors = factors or ["relevance", "authority", "recency"]
        
        # Score each source
        scored_sources = []
        for source in sources:
            score = self._calculate_source_score(query, source, factors)
            source["ranking_score"] = score
            scored_sources.append((source, score))
        
        # Sort by score
        scored_sources.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k sources
        ranked_sources = [source for source, score in scored_sources]
        if top_k:
            ranked_sources = ranked_sources[:top_k]
        
        self.logger.debug(f"Ranked {len(sources)} sources, returning {len(ranked_sources)}")
        
        return ranked_sources
    
    def _calculate_source_score(
        self,
        query: str,
        source: Dict[str, Any],
        factors: List[str]
    ) -> float:
        """Calculate overall score for a source."""
        
        scores = {}
        
        # Relevance score (keyword overlap)
        if "relevance" in factors:
            scores["relevance"] = self._calculate_relevance_score(query, source)
        
        # Authority score (predefined or calculated)
        if "authority" in factors:
            scores["authority"] = source.get("authority_score", 0.5)
        
        # Recency score (based on date)
        if "recency" in factors:
            scores["recency"] = self._calculate_recency_score(source)
        
        # Consistency score (internal coherence)
        if "consistency" in factors:
            scores["consistency"] = self._calculate_consistency_score(source)
        
        # Weighted average
        weights = {
            "relevance": 0.4,
            "authority": 0.3,
            "recency": 0.2,
            "consistency": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for factor, score in scores.items():
            weight = weights.get(factor, 0.25)
            total_score += score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 0.01)
    
    def _calculate_relevance_score(self, query: str, source: Dict[str, Any]) -> float:
        """Calculate relevance score based on query-source similarity."""
        
        query_words = set(query.lower().split())
        
        # Check title relevance (weighted higher)
        title = source.get("title", "").lower()
        title_words = set(title.split())
        title_overlap = len(query_words.intersection(title_words)) / max(len(query_words), 1)
        
        # Check content relevance
        content = source.get("content", "").lower()
        content_words = set(content.split())
        content_overlap = len(query_words.intersection(content_words)) / max(len(query_words), 1)
        
        # Check tags relevance
        tags = source.get("tags", [])
        tag_words = set([tag.lower() for tag in tags])
        tag_overlap = len(query_words.intersection(tag_words)) / max(len(query_words), 1)
        
        # Combined relevance score
        relevance_score = (title_overlap * 0.5) + (content_overlap * 0.3) + (tag_overlap * 0.2)
        
        return min(1.0, relevance_score)
    
    def _calculate_recency_score(self, source: Dict[str, Any]) -> float:
        """Calculate recency score based on source date."""
        
        date_str = source.get("date")
        if not date_str:
            return 0.5  # Neutral score for unknown dates
        
        try:
            # Simple date parsing
            if len(date_str) == 4:  # Year only
                source_year = int(date_str)
                current_year = datetime.now().year
                years_old = current_year - source_year
                
                # Exponential decay with 5-year half-life
                return max(0.1, 1.0 * (0.5 ** (years_old / 5.0)))
            
            # For full dates, could add more sophisticated parsing
            return 0.7  # Default for partial date formats
            
        except (ValueError, TypeError):
            return 0.5  # Neutral score for unparseable dates
    
    def _calculate_consistency_score(self, source: Dict[str, Any]) -> float:
        """Calculate internal consistency score for the source."""
        
        content = source.get("content", "")
        title = source.get("title", "")
        
        if not content or not title:
            return 0.5
        
        # Simple consistency checks
        score = 0.5  # Base score
        
        # Length consistency (not too short, not excessively long)
        content_length = len(content)
        if 100 <= content_length <= 2000:
            score += 0.2
        elif content_length < 50:
            score -= 0.2
        
        # Title-content consistency
        title_words = set(title.lower().split())
        content_words = set(content.lower().split())
        title_content_overlap = len(title_words.intersection(content_words)) / max(len(title_words), 1)
        
        if title_content_overlap > 0.3:
            score += 0.2
        elif title_content_overlap < 0.1:
            score -= 0.1
        
        # Basic quality indicators
        if source.get("source") and source.get("url"):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def get_ranking_explanation(
        self,
        query: str,
        source: Dict[str, Any],
        factors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get detailed explanation of source ranking."""
        
        factors = factors or ["relevance", "authority", "recency", "consistency"]
        
        explanation = {
            "source_id": source.get("id", "unknown"),
            "title": source.get("title", "Untitled"),
            "overall_score": source.get("ranking_score", 0.0),
            "factor_scores": {}
        }
        
        for factor in factors:
            if factor == "relevance":
                explanation["factor_scores"]["relevance"] = self._calculate_relevance_score(query, source)
            elif factor == "authority":
                explanation["factor_scores"]["authority"] = source.get("authority_score", 0.5)
            elif factor == "recency":
                explanation["factor_scores"]["recency"] = self._calculate_recency_score(source)
            elif factor == "consistency":
                explanation["factor_scores"]["consistency"] = self._calculate_consistency_score(source)
        
        return explanation
    
    def batch_rank_sources(
        self,
        queries: List[str],
        sources: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """Rank sources for multiple queries."""
        
        results = []
        for query in queries:
            ranked_sources = self.rank(query, sources, top_k)
            results.append(ranked_sources)
        
        return results
    
    def get_ranking_stats(self) -> Dict[str, Any]:
        """Get ranking statistics."""
        return {
            "models": self.models,
            "ensemble_method": self.ensemble_method,
            "generation": 1,
            "method": "rule_based"
        }