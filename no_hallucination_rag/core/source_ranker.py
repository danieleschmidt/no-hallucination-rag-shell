"""
Multi-factor source ranking for factuality optimization.
"""

import logging
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RankingFactors:
    """Factors used for source ranking."""
    relevance: float = 0.3
    recency: float = 0.2
    authority: float = 0.3
    consistency: float = 0.2


class SourceRanker:
    """Ranks sources using multiple factuality-optimized factors."""
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        ensemble_method: str = "weighted_vote",
        ranking_factors: Optional[RankingFactors] = None
    ):
        self.models = models or ["basic"]  # Simplified for Generation 1
        self.ensemble_method = ensemble_method
        self.ranking_factors = ranking_factors or RankingFactors()
        self.logger = logging.getLogger(__name__)
        
        # Authority scoring database (simplified)
        self.authority_scores = {
            "gov": 1.0,
            "edu": 0.9,
            "org": 0.8,
            "com": 0.6,
            "unknown": 0.3
        }
    
    def rank(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        factors: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank sources by multiple factors for factuality optimization.
        
        Args:
            query: User query for relevance scoring
            sources: List of source documents
            top_k: Number of top sources to return
            factors: Specific factors to use for ranking
            
        Returns:
            Ranked list of sources with scores
        """
        if not sources:
            return []
        
        factors = factors or ["relevance", "recency", "authority", "consistency"]
        
        # Calculate scores for each factor
        scored_sources = []
        for source in sources:
            scores = {}
            
            if "relevance" in factors:
                scores["relevance"] = self._calculate_relevance_score(query, source)
            
            if "recency" in factors:
                scores["recency"] = self._calculate_recency_score(source)
            
            if "authority" in factors:
                scores["authority"] = self._calculate_authority_score(source)
            
            if "consistency" in factors:
                scores["consistency"] = self._calculate_consistency_score(source, sources)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(scores)
            
            # Add scoring metadata to source
            enhanced_source = source.copy()
            enhanced_source.update({
                "ranking_scores": scores,
                "composite_score": composite_score
            })
            
            scored_sources.append(enhanced_source)
        
        # Sort by composite score
        ranked_sources = sorted(
            scored_sources,
            key=lambda x: x["composite_score"],
            reverse=True
        )
        
        # Return top_k if specified
        if top_k:
            ranked_sources = ranked_sources[:top_k]
        
        self.logger.info(f"Ranked {len(sources)} sources, returning top {len(ranked_sources)}")
        
        return ranked_sources
    
    def _calculate_relevance_score(self, query: str, source: Dict[str, Any]) -> float:
        """Calculate relevance score between query and source."""
        # Simplified relevance scoring for Generation 1
        # In Generation 2, this will use embedding similarity
        
        query_lower = query.lower()
        content = source.get("content", "").lower()
        title = source.get("title", "").lower()
        
        # Simple keyword matching
        query_words = set(query_lower.split())
        content_words = set(content.split())
        title_words = set(title.split())
        
        content_overlap = len(query_words.intersection(content_words)) / max(len(query_words), 1)
        title_overlap = len(query_words.intersection(title_words)) / max(len(query_words), 1)
        
        # Weight title matches higher
        relevance_score = (content_overlap * 0.7) + (title_overlap * 0.3)
        
        return min(relevance_score, 1.0)
    
    def _calculate_recency_score(self, source: Dict[str, Any]) -> float:
        """Calculate recency score based on publication date."""
        try:
            date_str = source.get("date", "")
            if not date_str:
                return 0.5  # Unknown date gets neutral score
            
            # Parse date (simplified - assumes ISO format)
            pub_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            current_date = datetime.now(pub_date.tzinfo)
            
            # Calculate age in days
            age_days = (current_date - pub_date).days
            
            # Score based on age (newer is better)
            if age_days <= 30:
                return 1.0
            elif age_days <= 90:
                return 0.9
            elif age_days <= 365:
                return 0.7
            elif age_days <= 365 * 2:
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.5  # Default for unparseable dates
    
    def _calculate_authority_score(self, source: Dict[str, Any]) -> float:
        """Calculate authority score based on source domain and type."""
        url = source.get("url", "")
        
        # Extract domain type
        domain_type = "unknown"
        for suffix in [".gov", ".edu", ".org", ".com"]:
            if suffix in url:
                domain_type = suffix[1:]  # Remove dot
                break
        
        base_score = self.authority_scores.get(domain_type, 0.3)
        
        # Additional authority factors
        citation_count = source.get("citations", 0)
        if citation_count > 100:
            base_score *= 1.2
        elif citation_count > 10:
            base_score *= 1.1
        
        # Check for official sources
        official_indicators = ["whitehouse", "nist", "ieee", "iso", "oecd"]
        if any(indicator in url.lower() for indicator in official_indicators):
            base_score *= 1.3
        
        return min(base_score, 1.0)
    
    def _calculate_consistency_score(
        self, source: Dict[str, Any], all_sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate consistency score with other sources."""
        # Simplified consistency check for Generation 1
        # In Generation 2, this will use semantic similarity
        
        if len(all_sources) <= 1:
            return 1.0  # Single source gets full score
        
        source_content = source.get("content", "").lower()
        source_words = set(source_content.split())
        
        consistency_scores = []
        for other_source in all_sources:
            if other_source.get("url") == source.get("url"):
                continue  # Skip self
            
            other_content = other_source.get("content", "").lower()
            other_words = set(other_content.split())
            
            # Simple Jaccard similarity
            intersection = source_words.intersection(other_words)
            union = source_words.union(other_words)
            
            if len(union) > 0:
                similarity = len(intersection) / len(union)
                consistency_scores.append(similarity)
        
        # Return average consistency with other sources
        if consistency_scores:
            return statistics.mean(consistency_scores)
        else:
            return 0.5  # Neutral score if no other sources
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite score from individual factor scores."""
        composite = 0.0
        total_weight = 0.0
        
        factor_weights = {
            "relevance": self.ranking_factors.relevance,
            "recency": self.ranking_factors.recency,
            "authority": self.ranking_factors.authority,
            "consistency": self.ranking_factors.consistency
        }
        
        for factor, score in scores.items():
            weight = factor_weights.get(factor, 0.0)
            composite += score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite /= total_weight
        
        return composite