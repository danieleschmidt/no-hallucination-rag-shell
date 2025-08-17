
"""
Advanced multi-level caching system with intelligent eviction.
"""

import time
import threading
import hashlib
import pickle
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import weakref


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None  # seconds
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at.timestamp() > self.ttl
    
    def touch(self):
        """Update access information."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class CacheEvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        """Select cache entries for eviction."""
        pass


class AdvancedCacheManager:
    """Multi-level cache manager with L1, L2, L3 caches."""
    
    def __init__(self):
        self.l1_cache = {}  # Fast in-memory cache
        self.l2_cache = {}  # Larger in-memory cache
        self.l3_cache = {}  # Persistent cache
        
    def set_l1(self, key: str, value: Any):
        """Set value in L1 cache."""
        self.l1_cache[key] = value
        
    def get_l1(self, key: str) -> Any:
        """Get value from L1 cache."""
        return self.l1_cache.get(key)
        
    def set_with_ttl(self, key: str, value: Any, ttl: int):
        """Set value with TTL."""
        import time
        self.l2_cache[key] = {
            "value": value,
            "expires": time.time() + ttl
        }
        
    def get(self, key: str) -> Any:
        """Get value, checking TTL."""
        import time
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if time.time() > entry["expires"]:
                del self.l2_cache[key]
                return None
            return entry["value"]
        return self.l1_cache.get(key)


class LRUEviction(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        # Sort by last accessed time, oldest first
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].last_accessed)
        return [key for key, _ in sorted_entries[:target_count]]


class LFUEviction(CacheEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        # Sort by access count, lowest first
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].access_count)
        return [key for key, _ in sorted_entries[:target_count]]


class TTLEviction(CacheEvictionPolicy):
    """Time-To-Live eviction policy."""
    
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        # Find expired entries first
        expired = [key for key, entry in entries.items() if entry.is_expired()]
        
        if len(expired) >= target_count:
            return expired[:target_count]
        
        # If not enough expired entries, fall back to LRU
        lru = LRUEviction()
        remaining_needed = target_count - len(expired)
        non_expired = {k: v for k, v in entries.items() if k not in expired}
        additional = lru.select_victims(non_expired, remaining_needed)
        
        return expired + additional


class AdaptiveCacheEviction(CacheEvictionPolicy):
    """Adaptive eviction policy that learns from access patterns."""
    
    def __init__(self):
        self.access_patterns = defaultdict(list)  # key -> list of access times
        self.prediction_window = 3600  # 1 hour
    
    def select_victims(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        now = time.time()
        
        # Score entries based on predicted future access
        scored_entries = []
        
        for key, entry in entries.items():
            # Calculate access frequency in recent window
            recent_accesses = [
                t for t in self.access_patterns.get(key, [])
                if now - t < self.prediction_window
            ]
            
            # Simple prediction: recent frequency + recency boost
            frequency_score = len(recent_accesses) / max(1, self.prediction_window / 3600)
            recency_score = 1.0 / max(1, now - entry.last_accessed.timestamp())
            
            # Combined score (lower = more likely to evict)
            score = frequency_score + recency_score * 0.1
            scored_entries.append((key, score))
        
        # Sort by score (ascending - lowest scores evicted first)
        scored_entries.sort(key=lambda x: x[1])
        return [key for key, _ in scored_entries[:target_count]]
    
    def record_access(self, key: str):
        """Record access for pattern learning."""
        now = time.time()
        self.access_patterns[key].append(now)
        
        # Keep only recent history
        cutoff = now - self.prediction_window * 2
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]


class MultiLevelCache:
    """Multi-level cache with different storage tiers."""
    
    def __init__(
        self,
        l1_size: int = 1000,      # In-memory hot cache
        l2_size: int = 10000,     # In-memory warm cache
        l3_size: int = 100000,    # Persistent cache
        eviction_policy: str = "adaptive"
    ):
        self.l1_cache = OrderedDict()  # Hot - fastest access
        self.l2_cache = OrderedDict()  # Warm - fast access
        self.l3_cache = {}             # Cold - persistent
        
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        
        self.lock = threading.RLock()
        self.stats = CacheStats()
        self.logger = logging.getLogger(__name__)
        
        # Setup eviction policy
        if eviction_policy == "lru":
            self.eviction_policy = LRUEviction()
        elif eviction_policy == "lfu":
            self.eviction_policy = LFUEviction()
        elif eviction_policy == "ttl":
            self.eviction_policy = TTLEviction()
        else:  # adaptive
            self.eviction_policy = AdaptiveCacheEviction()
        
        # Start background maintenance
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
    
    def get(self, key: str) -> Any:
        """Get value from cache with promotion between levels."""
        with self.lock:
            # Try L1 first (hottest)
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not entry.is_expired():
                    entry.touch()
                    self.stats.record_hit("l1")
                    
                    # Record access for adaptive eviction
                    if isinstance(self.eviction_policy, AdaptiveCacheEviction):
                        self.eviction_policy.record_access(key)
                    
                    return entry.value
                else:
                    del self.l1_cache[key]
            
            # Try L2 (warm)
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if not entry.is_expired():
                    entry.touch()
                    self.stats.record_hit("l2")
                    
                    # Promote to L1
                    self._promote_to_l1(key, entry)
                    
                    if isinstance(self.eviction_policy, AdaptiveCacheEviction):
                        self.eviction_policy.record_access(key)
                    
                    return entry.value
                else:
                    del self.l2_cache[key]
            
            # Try L3 (persistent)
            if key in self.l3_cache:
                entry = self.l3_cache[key]
                if not entry.is_expired():
                    entry.touch()
                    self.stats.record_hit("l3")
                    
                    # Promote to L2
                    self._promote_to_l2(key, entry)
                    
                    if isinstance(self.eviction_policy, AdaptiveCacheEviction):
                        self.eviction_policy.record_access(key)
                    
                    return entry.value
                else:
                    del self.l3_cache[key]
            
            # Not found in any level
            self.stats.record_miss()
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache, starting from L1."""
        with self.lock:
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl=ttl,
                size_bytes=self._estimate_size(value)
            )
            
            # Always start in L1
            self._put_l1(key, entry)
            self.stats.record_put("l1")
    
    def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry to L1 cache."""
        # Remove from lower levels
        self.l2_cache.pop(key, None)
        self.l3_cache.pop(key, None)
        
        # Add to L1
        self._put_l1(key, entry)
    
    def _promote_to_l2(self, key: str, entry: CacheEntry):
        """Promote entry to L2 cache."""
        # Remove from L3
        self.l3_cache.pop(key, None)
        
        # Add to L2
        self._put_l2(key, entry)
    
    def _put_l1(self, key: str, entry: CacheEntry):
        """Put entry in L1 with eviction if needed."""
        if len(self.l1_cache) >= self.l1_size:
            self._evict_from_l1(1)
        
        self.l1_cache[key] = entry
    
    def _put_l2(self, key: str, entry: CacheEntry):
        """Put entry in L2 with eviction if needed."""
        if len(self.l2_cache) >= self.l2_size:
            self._evict_from_l2(1)
        
        self.l2_cache[key] = entry
    
    def _evict_from_l1(self, count: int):
        """Evict entries from L1, demoting to L2."""
        if not self.l1_cache:
            return
        
        victims = self.eviction_policy.select_victims(self.l1_cache, count)
        
        for key in victims:
            if key in self.l1_cache:
                entry = self.l1_cache.pop(key)
                # Demote to L2 unless expired
                if not entry.is_expired():
                    self._put_l2(key, entry)
    
    def _evict_from_l2(self, count: int):
        """Evict entries from L2, demoting to L3."""
        if not self.l2_cache:
            return
        
        victims = self.eviction_policy.select_victims(self.l2_cache, count)
        
        for key in victims:
            if key in self.l2_cache:
                entry = self.l2_cache.pop(key)
                # Demote to L3 unless expired
                if not entry.is_expired():
                    if len(self.l3_cache) >= self.l3_size:
                        self._evict_from_l3(1)
                    self.l3_cache[key] = entry
    
    def _evict_from_l3(self, count: int):
        """Evict entries from L3 (final eviction)."""
        if not self.l3_cache:
            return
        
        victims = self.eviction_policy.select_victims(self.l3_cache, count)
        
        for key in victims:
            self.l3_cache.pop(key, None)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return len(value) * 50  # Rough estimate
            elif isinstance(value, dict):
                return len(value) * 100  # Rough estimate
            else:
                return 1000  # Default estimate
    
    def _maintenance_loop(self):
        """Background maintenance for cache cleanup."""
        while True:
            try:
                with self.lock:
                    # Clean up expired entries
                    self._cleanup_expired()
                    
                    # Update statistics
                    self.stats.update_size_stats(
                        l1_size=len(self.l1_cache),
                        l2_size=len(self.l2_cache),
                        l3_size=len(self.l3_cache)
                    )
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Cache maintenance error: {e}")
                time.sleep(60)
    
    def _cleanup_expired(self):
        """Remove expired entries from all levels."""
        # Clean L1
        expired_l1 = [k for k, v in self.l1_cache.items() if v.is_expired()]
        for key in expired_l1:
            del self.l1_cache[key]
        
        # Clean L2
        expired_l2 = [k for k, v in self.l2_cache.items() if v.is_expired()]
        for key in expired_l2:
            del self.l2_cache[key]
        
        # Clean L3
        expired_l3 = [k for k, v in self.l3_cache.items() if v.is_expired()]
        for key in expired_l3:
            del self.l3_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "levels": {
                    "l1": {"size": len(self.l1_cache), "max_size": self.l1_size},
                    "l2": {"size": len(self.l2_cache), "max_size": self.l2_size},
                    "l3": {"size": len(self.l3_cache), "max_size": self.l3_size}
                },
                "performance": self.stats.get_stats(),
                "eviction_policy": type(self.eviction_policy).__name__
            }
    
    def clear(self):
        """Clear all cache levels."""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l3_cache.clear()
            self.stats.reset()


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits_l1: int = 0
    hits_l2: int = 0
    hits_l3: int = 0
    misses: int = 0
    puts_l1: int = 0
    puts_l2: int = 0
    puts_l3: int = 0
    current_l1_size: int = 0
    current_l2_size: int = 0
    current_l3_size: int = 0
    
    def record_hit(self, level: str):
        """Record cache hit."""
        if level == "l1":
            self.hits_l1 += 1
        elif level == "l2":
            self.hits_l2 += 1
        elif level == "l3":
            self.hits_l3 += 1
    
    def record_miss(self):
        """Record cache miss."""
        self.misses += 1
    
    def record_put(self, level: str):
        """Record cache put operation."""
        if level == "l1":
            self.puts_l1 += 1
        elif level == "l2":
            self.puts_l2 += 1
        elif level == "l3":
            self.puts_l3 += 1
    
    def update_size_stats(self, l1_size: int, l2_size: int, l3_size: int):
        """Update current size statistics."""
        self.current_l1_size = l1_size
        self.current_l2_size = l2_size
        self.current_l3_size = l3_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics summary."""
        total_hits = self.hits_l1 + self.hits_l2 + self.hits_l3
        total_requests = total_hits + self.misses
        
        return {
            "hit_rate": total_hits / max(1, total_requests),
            "miss_rate": self.misses / max(1, total_requests),
            "total_requests": total_requests,
            "hits_by_level": {
                "l1": self.hits_l1,
                "l2": self.hits_l2,
                "l3": self.hits_l3
            },
            "level_hit_rates": {
                "l1": self.hits_l1 / max(1, total_requests),
                "l2": self.hits_l2 / max(1, total_requests),
                "l3": self.hits_l3 / max(1, total_requests)
            }
        }
    
    def reset(self):
        """Reset all statistics."""
        self.hits_l1 = 0
        self.hits_l2 = 0
        self.hits_l3 = 0
        self.misses = 0
        self.puts_l1 = 0
        self.puts_l2 = 0
        self.puts_l3 = 0


class CacheManager:
    """Global cache management with multiple cache instances."""
    
    def __init__(self):
        self.caches: Dict[str, MultiLevelCache] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_cache(self, name: str, **kwargs) -> MultiLevelCache:
        """Get or create a named cache."""
        if name not in self.caches:
            self.caches[name] = MultiLevelCache(**kwargs)
            self.logger.info(f"Created cache: {name}")
        
        return self.caches[name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            name: cache.get_stats()
            for name, cache in self.caches.items()
        }
    
    def clear_all(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("Cleared all caches")


# Decorators for easy caching

def cached(cache_name: str = "default", ttl: Optional[float] = None, 
          key_func: Optional[Callable] = None):
    """Decorator to add caching to functions."""
    
    cache_manager = CacheManager()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{hashlib.md5(arg_str.encode()).hexdigest()}"
            
            # Try to get from cache
            cache = cache_manager.get_cache(cache_name)
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
global_cache_manager = CacheManager()
