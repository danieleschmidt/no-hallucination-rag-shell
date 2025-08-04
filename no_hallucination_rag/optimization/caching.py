"""
Advanced caching system with adaptive policies and TTL management.
"""

import logging
import time
import hashlib
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from enum import Enum
import pickle
import os


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class AdaptiveCache:
    """High-performance adaptive cache with multiple eviction policies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = 3600,  # 1 hour
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        persistence_path: Optional[str] = None
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.policy = policy
        self.persistence_path = persistence_path
        
        # Cache storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.access_frequency: Dict[str, int] = defaultdict(int)  # For LFU
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        
        self.logger = logging.getLogger(__name__)
        
        # Load persisted cache if available
        if self.persistence_path:
            self._load_from_disk()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self.entries.get(key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access info
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.access_frequency[key] += 1
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.move_to_end(key)
            
            self.stats.hits += 1
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None
    ) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if we need to make space
            if not self._ensure_capacity(size_bytes):
                return False
            
            # Create entry
            current_time = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self.entries:
                self._remove_entry(key)
            
            # Add new entry
            self.entries[key] = entry
            self.access_order[key] = current_time
            self.access_frequency[key] = 1
            
            # Update stats
            self.stats.entry_count += 1
            self.stats.size_bytes += size_bytes
            
            # Background cleanup if needed
            self._maybe_cleanup()
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.entries.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.stats = CacheStats()
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching pattern."""
        with self._lock:
            keys_to_remove = []
            for key in self.entries:
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                size_bytes=self.stats.size_bytes,
                entry_count=self.stats.entry_count
            )
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        with self._lock:
            return {
                "total_bytes": self.stats.size_bytes,
                "total_mb": self.stats.size_bytes / (1024 * 1024),
                "entry_count": self.stats.entry_count,
                "avg_entry_size": self.stats.size_bytes / max(self.stats.entry_count, 1),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": self.stats.size_bytes / self.max_memory_bytes,
                "max_entries": self.max_size,
                "entry_utilization": self.stats.entry_count / self.max_size
            }
    
    def _ensure_capacity(self, new_entry_size: int) -> bool:
        """Ensure cache has capacity for new entry."""
        # Check if single entry is too large
        if new_entry_size > self.max_memory_bytes:
            return False
        
        # Evict entries until we have space
        while (
            self.stats.entry_count >= self.max_size or
            self.stats.size_bytes + new_entry_size > self.max_memory_bytes
        ):
            if not self._evict_one():
                return False  # Nothing left to evict
        
        return True
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self.entries:
            return False
        
        if self.policy == CachePolicy.LRU:
            key_to_evict = next(iter(self.access_order))
        elif self.policy == CachePolicy.LFU:
            key_to_evict = min(self.access_frequency, key=self.access_frequency.get)
        elif self.policy == CachePolicy.TTL:
            # Evict oldest entry by creation time
            key_to_evict = min(self.entries, key=lambda k: self.entries[k].created_at)
        else:  # ADAPTIVE
            key_to_evict = self._adaptive_eviction()
        
        self._remove_entry(key_to_evict)
        self.stats.evictions += 1
        return True
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction considering multiple factors."""
        best_key = None
        best_score = float('inf')
        
        current_time = time.time()
        
        for key, entry in self.entries.items():
            # Calculate eviction score (lower is better)
            age_factor = entry.age_seconds / 3600  # Normalize to hours
            access_factor = 1.0 / max(entry.access_count, 1)  # Prefer frequently accessed
            recency_factor = (current_time - entry.last_accessed) / 3600  # Recent access
            size_factor = entry.size_bytes / (1024 * 1024)  # Prefer smaller entries
            
            # Combined score
            score = age_factor + access_factor + recency_factor + (size_factor * 0.1)
            
            if score < best_score:
                best_score = score
                best_key = key
        
        return best_key or next(iter(self.entries))
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update bookkeeping."""
        if key not in self.entries:
            return
        
        entry = self.entries[key]
        
        # Update stats
        self.stats.entry_count -= 1
        self.stats.size_bytes -= entry.size_bytes
        
        # Remove from data structures
        del self.entries[key]
        if key in self.access_order:
            del self.access_order[key]
        if key in self.access_frequency:
            del self.access_frequency[key]
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            else:
                return 1024  # Default estimate
    
    def _maybe_cleanup(self) -> None:
        """Perform background cleanup if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = []
        current_time = time.time()
        
        for key, entry in self.entries.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _save_to_disk(self) -> None:
        """Save cache to disk for persistence."""
        if not self.persistence_path:
            return
        
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            cache_data = {
                'entries': {
                    key: {
                        'value': entry.value,
                        'created_at': entry.created_at,
                        'ttl': entry.ttl
                    }
                    for key, entry in self.entries.items()
                    if not entry.is_expired
                }
            }
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save cache to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not os.path.exists(self.persistence_path):
            return
        
        try:
            with open(self.persistence_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            current_time = time.time()
            
            for key, data in cache_data.get('entries', {}).items():
                # Check if entry should be loaded (not expired)
                created_at = data['created_at']
                ttl = data['ttl']
                
                if ttl and current_time - created_at > ttl:
                    continue  # Skip expired entries
                
                # Recreate entry
                entry = CacheEntry(
                    key=key,
                    value=data['value'],
                    created_at=created_at,
                    last_accessed=current_time,
                    access_count=1,
                    ttl=ttl,
                    size_bytes=self._calculate_size(data['value'])
                )
                
                self.entries[key] = entry
                self.access_order[key] = current_time
                self.access_frequency[key] = 1
                
                self.stats.entry_count += 1
                self.stats.size_bytes += entry.size_bytes
            
            self.logger.info(f"Loaded {len(self.entries)} entries from cache disk")
            
        except Exception as e:
            self.logger.error(f"Failed to load cache from disk: {e}")


class CacheManager:
    """Manages multiple specialized caches for different components."""
    
    def __init__(self):
        self.caches: Dict[str, AdaptiveCache] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized caches
        self._initialize_caches()
    
    def _initialize_caches(self) -> None:
        """Initialize component-specific caches."""
        # Query results cache - short TTL, moderate size
        self.caches['queries'] = AdaptiveCache(
            max_size=500,
            max_memory_mb=50,
            default_ttl=1800,  # 30 minutes
            policy=CachePolicy.ADAPTIVE,
            persistence_path="data/cache/queries.pkl"
        )
        
        # Source retrieval cache - longer TTL, larger size
        self.caches['retrieval'] = AdaptiveCache(
            max_size=2000,
            max_memory_mb=200,
            default_ttl=7200,  # 2 hours
            policy=CachePolicy.LFU,
            persistence_path="data/cache/retrieval.pkl"
        )
        
        # Factuality scores cache - very long TTL
        self.caches['factuality'] = AdaptiveCache(
            max_size=1000,
            max_memory_mb=20,
            default_ttl=86400,  # 24 hours
            policy=CachePolicy.TTL,
            persistence_path="data/cache/factuality.pkl"
        )
        
        # Governance results cache - long TTL
        self.caches['governance'] = AdaptiveCache(
            max_size=500,
            max_memory_mb=10,
            default_ttl=43200,  # 12 hours
            policy=CachePolicy.LRU,
            persistence_path="data/cache/governance.pkl"
        )
    
    def get_cache(self, cache_name: str) -> Optional[AdaptiveCache]:
        """Get cache by name."""
        return self.caches.get(cache_name)
    
    def cache_query_result(self, query_hash: str, result: Any, ttl: Optional[float] = None) -> None:
        """Cache query result."""
        cache = self.caches.get('queries')
        if cache:
            cache.put(f"query:{query_hash}", result, ttl)
    
    def get_cached_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result."""
        cache = self.caches.get('queries')
        if cache:
            return cache.get(f"query:{query_hash}")
        return None
    
    def cache_retrieval_result(self, query_hash: str, sources: List[Dict], ttl: Optional[float] = None) -> None:
        """Cache retrieval result."""
        cache = self.caches.get('retrieval')
        if cache:
            cache.put(f"retrieval:{query_hash}", sources, ttl)
    
    def get_cached_retrieval_result(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached retrieval result."""
        cache = self.caches.get('retrieval')
        if cache:
            return cache.get(f"retrieval:{query_hash}")
        return None
    
    def get_all_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        return {
            name: cache.get_stats()
            for name, cache in self.caches.items()
        }
    
    def get_total_memory_usage(self) -> Dict[str, Any]:
        """Get total memory usage across all caches."""
        total_bytes = 0
        total_entries = 0
        
        cache_details = {}
        
        for name, cache in self.caches.items():
            memory_info = cache.get_memory_usage()
            cache_details[name] = memory_info
            total_bytes += memory_info['total_bytes']
            total_entries += memory_info['entry_count']
        
        return {
            'total_memory_mb': total_bytes / (1024 * 1024),
            'total_entries': total_entries,
            'cache_details': cache_details
        }
    
    def invalidate_all(self) -> None:
        """Invalidate all caches."""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("All caches invalidated")
    
    def create_query_hash(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create consistent hash for query caching."""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Include relevant parameters
        cache_params = {
            'query': normalized_query,
            'params': params or {}
        }
        
        # Create hash
        cache_key = json.dumps(cache_params, sort_keys=True)
        return hashlib.md5(cache_key.encode()).hexdigest()


def cached_method(cache_name: str, ttl: Optional[float] = None):
    """Decorator for caching method results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            # Get cache manager from instance
            cache_manager = getattr(self, 'cache_manager', None)
            if not cache_manager:
                return func(self, *args, **kwargs)
            
            cache = cache_manager.get_cache(cache_name)
            if not cache:
                return func(self, *args, **kwargs)
            
            # Create cache key
            key_data = {
                'method': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(self, *args, **kwargs)
            cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator