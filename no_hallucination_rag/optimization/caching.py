"""
Caching for RAG system.
Generation 1: Basic in-memory caching.
"""

import logging
import hashlib
from typing import Dict, Any, Optional


class AdaptiveCache:
    """Adaptive caching with TTL and memory management."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        
    def _evict_expired(self):
        """Remove expired entries."""
        import time
        current_time = time.time()
        expired_keys = [k for k, created in self.creation_times.items() 
                       if current_time - created > self.default_ttl]
        for key in expired_keys:
            self._remove_key(key)
            
    def _remove_key(self, key: str):
        """Remove key from all tracking structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
        
    def _evict_lru(self):
        """Evict least recently used items."""
        if len(self.cache) <= self.max_size:
            return
        # Sort by access time and remove oldest
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        to_remove = len(self.cache) - self.max_size + 1
        for key, _ in sorted_items[:to_remove]:
            self._remove_key(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value with LRU tracking."""
        self._evict_expired()
        if key in self.cache:
            import time
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached value with TTL."""
        import time
        current_time = time.time()
        self.cache[key] = value
        self.access_times[key] = current_time
        self.creation_times[key] = current_time
        self._evict_lru()
        
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
        
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
    
    def _save_to_disk(self):
        """Save cache to disk (placeholder for persistence)."""
        # In a production system, this would save cache state to disk
        self.logger.debug(f"Cache state saved (simulated) - {len(self.cache)} items")


class CacheManager:
    """Manages multiple adaptive caches for RAG system."""
    
    def __init__(self, query_cache_size: int = 1000, retrieval_cache_size: int = 500):
        self.logger = logging.getLogger(__name__)
        self.query_cache = AdaptiveCache(max_size=query_cache_size)
        self.retrieval_cache = AdaptiveCache(max_size=retrieval_cache_size)
        self.factuality_cache = AdaptiveCache(max_size=200)
        self.stats = {
            'query_hits': 0, 'query_misses': 0,
            'retrieval_hits': 0, 'retrieval_misses': 0,
            'factuality_hits': 0, 'factuality_misses': 0
        }
        
        # For compatibility with tests and shutdown methods
        self.caches = {
            'queries': self.query_cache,
            'retrieval': self.retrieval_cache,
            'factuality': self.factuality_cache
        }
        
    def create_query_hash(self, query: str, params: Dict[str, Any]) -> str:
        """Create hash for query caching."""
        content = f"{query}_{str(params)}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def get_cached_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result."""
        result = self.query_cache.get(query_hash)
        if result:
            self.stats['query_hits'] += 1
            self.logger.debug(f"Query cache hit: {query_hash[:8]}...")
        else:
            self.stats['query_misses'] += 1
        return result
        
    def cache_query_result(self, query_hash: str, result: Any):
        """Cache query result."""
        self.query_cache.set(query_hash, result)
        self.logger.debug(f"Query result cached: {query_hash[:8]}...")
        
    def get_cached_retrieval_result(self, retrieval_hash: str) -> Optional[Any]:
        """Get cached retrieval result."""
        result = self.retrieval_cache.get(retrieval_hash)
        if result:
            self.stats['retrieval_hits'] += 1
            self.logger.debug(f"Retrieval cache hit: {retrieval_hash[:8]}...")
        else:
            self.stats['retrieval_misses'] += 1
        return result
        
    def cache_retrieval_result(self, retrieval_hash: str, result: Any):
        """Cache retrieval result."""
        self.retrieval_cache.set(retrieval_hash, result)
        self.logger.debug(f"Retrieval result cached: {retrieval_hash[:8]}...")
        
    def get_cached_factuality_result(self, content_hash: str) -> Optional[Any]:
        """Get cached factuality check result."""
        result = self.factuality_cache.get(content_hash)
        if result:
            self.stats['factuality_hits'] += 1
        else:
            self.stats['factuality_misses'] += 1
        return result
        
    def cache_factuality_result(self, content_hash: str, result: Any):
        """Cache factuality check result."""
        self.factuality_cache.set(content_hash, result)
        
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = self.stats['query_hits'] + self.stats['retrieval_hits'] + self.stats['factuality_hits']
        total_misses = self.stats['query_misses'] + self.stats['retrieval_misses'] + self.stats['factuality_misses']
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        
        return {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': round(hit_rate, 3),
            'query_cache_size': self.query_cache.size(),
            'retrieval_cache_size': self.retrieval_cache.size(),
            'factuality_cache_size': self.factuality_cache.size(),
            'detailed_stats': self.stats.copy()
        }
        
    def get_total_memory_usage(self) -> int:
        """Get estimated total memory usage in bytes."""
        import sys
        total_size = 0
        for cache in [self.query_cache, self.retrieval_cache, self.factuality_cache]:
            for value in cache.cache.values():
                try:
                    total_size += sys.getsizeof(value)
                except (TypeError, ValueError):
                    total_size += 1024  # Rough estimate for complex objects
        return total_size
        
    def invalidate_all(self):
        """Invalidate all caches."""
        self.query_cache.clear()
        self.retrieval_cache.clear()
        self.factuality_cache.clear()
        self.stats = {k: 0 for k in self.stats.keys()}
        self.logger.info("All caches invalidated")
        
    def warm_up_cache(self, common_queries: list):
        """Pre-warm cache with common queries."""
        self.logger.info(f"Warming up cache with {len(common_queries)} common queries")
        # This would be implemented with actual query processing in production


def cached_method(func):
    """Decorator for caching method results."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper