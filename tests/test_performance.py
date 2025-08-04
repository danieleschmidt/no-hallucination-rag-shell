"""
Performance and optimization tests.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch

from no_hallucination_rag.optimization.caching import AdaptiveCache, CacheManager
from no_hallucination_rag.optimization.concurrent_processing import AsyncRAGProcessor
from no_hallucination_rag.optimization.performance_optimizer import PerformanceOptimizer


class TestAdaptiveCache:
    """Test suite for adaptive caching."""
    
    @pytest.fixture
    def cache(self):
        """Create cache for testing."""
        return AdaptiveCache(
            max_size=100,
            max_memory_mb=10,
            default_ttl=3600
        )
    
    def test_cache_put_get(self, cache):
        """Test basic cache put and get operations."""
        cache.put("key1", "value1")
        
        result = cache.get("key1")
        assert result == "value1"
    
    def test_cache_miss(self, cache):
        """Test cache miss behavior."""
        result = cache.get("nonexistent_key")
        assert result is None
    
    def test_cache_ttl_expiration(self, cache):
        """Test TTL-based expiration."""
        cache.put("key1", "value1", ttl=0.1)  # 0.1 second TTL
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("key1") is None
    
    def test_cache_size_limit(self, cache):
        """Test cache size limit enforcement."""
        # Fill cache beyond limit
        for i in range(150):  # Max size is 100
            cache.put(f"key{i}", f"value{i}")
        
        stats = cache.get_stats()
        assert stats.entry_count <= 100  # Should not exceed max size
    
    def test_cache_memory_limit(self):
        """Test memory limit enforcement."""
        cache = AdaptiveCache(max_size=1000, max_memory_mb=1)  # 1MB limit
        
        # Try to store large values
        large_value = "x" * 100000  # 100KB string
        for i in range(20):  # Would exceed 1MB if all stored
            cache.put(f"key{i}", large_value)
        
        stats = cache.get_stats()
        # Should have evicted some entries to stay under memory limit
        assert stats.size_bytes <= 1024 * 1024  # 1MB
    
    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction policy."""
        cache.policy = cache.policy.LRU
        
        # Fill cache to capacity
        for i in range(100):
            cache.put(f"key{i}", f"value{i}")
        
        # Access first key to make it recently used
        cache.get("key0")
        
        # Add one more item to trigger eviction
        cache.put("new_key", "new_value")
        
        # First key should still be there (recently used)
        assert cache.get("key0") == "value0"
        
        # Some other key should have been evicted
        stats = cache.get_stats()
        assert stats.entry_count <= 100
    
    def test_cache_pattern_invalidation(self, cache):
        """Test pattern-based cache invalidation."""
        # Add some entries
        cache.put("user:123:profile", "profile_data")
        cache.put("user:123:settings", "settings_data")
        cache.put("user:456:profile", "other_profile")
        cache.put("product:789", "product_data")
        
        # Invalidate all user:123 entries
        invalidated = cache.invalidate_pattern("user:123")
        assert invalidated == 2
        
        # Should be gone
        assert cache.get("user:123:profile") is None
        assert cache.get("user:123:settings") is None
        
        # Others should remain
        assert cache.get("user:456:profile") == "other_profile"
        assert cache.get("product:789") == "product_data"
    
    def test_cache_statistics(self, cache):
        """Test cache statistics collection."""
        # Generate some hits and misses
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.entry_count == 2
        assert stats.hit_rate == 2/3
    
    def test_memory_usage_tracking(self, cache):
        """Test memory usage tracking."""
        cache.put("small", "x")
        cache.put("large", "x" * 1000)
        
        usage = cache.get_memory_usage()
        
        assert usage["entry_count"] == 2
        assert usage["total_bytes"] > 1000  # Should account for large string
        assert usage["avg_entry_size"] > 0


class TestCacheManager:
    """Test suite for cache manager."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager for testing."""
        return CacheManager()
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initializes with correct caches."""
        assert "queries" in cache_manager.caches
        assert "retrieval" in cache_manager.caches
        assert "factuality" in cache_manager.caches
        assert "governance" in cache_manager.caches
    
    def test_query_caching(self, cache_manager):
        """Test query result caching."""
        query_hash = cache_manager.create_query_hash("test query", {"param": "value"})
        
        # Cache result
        test_result = {"answer": "test answer", "score": 0.95}
        cache_manager.cache_query_result(query_hash, test_result)
        
        # Retrieve result
        cached_result = cache_manager.get_cached_query_result(query_hash)
        assert cached_result == test_result
    
    def test_retrieval_caching(self, cache_manager):
        """Test retrieval result caching."""
        query_hash = cache_manager.create_query_hash("test query")
        
        # Cache sources
        test_sources = [{"id": "doc1", "content": "test content"}]
        cache_manager.cache_retrieval_result(query_hash, test_sources)
        
        # Retrieve sources
        cached_sources = cache_manager.get_cached_retrieval_result(query_hash)
        assert cached_sources == test_sources
    
    def test_query_hash_consistency(self, cache_manager):
        """Test query hash consistency."""
        # Same query should produce same hash
        hash1 = cache_manager.create_query_hash("test query", {"a": 1, "b": 2})
        hash2 = cache_manager.create_query_hash("test query", {"b": 2, "a": 1})  # Different order
        
        assert hash1 == hash2  # Should be same due to sorting
        
        # Different query should produce different hash
        hash3 = cache_manager.create_query_hash("different query", {"a": 1, "b": 2})
        assert hash1 != hash3
    
    def test_cache_statistics(self, cache_manager):
        """Test cache statistics aggregation."""
        # Add some data to different caches
        cache_manager.cache_query_result("hash1", {"result": "test"})
        cache_manager.cache_retrieval_result("hash2", [{"doc": "test"}])
        
        stats = cache_manager.get_all_stats()
        
        assert "queries" in stats
        assert "retrieval" in stats
        assert stats["queries"].entry_count >= 1
        assert stats["retrieval"].entry_count >= 1
    
    def test_memory_usage_aggregation(self, cache_manager):
        """Test memory usage aggregation across caches."""
        # Add data to caches
        cache_manager.cache_query_result("hash1", {"large_result": "x" * 1000})
        
        usage = cache_manager.get_total_memory_usage()
        
        assert "total_memory_mb" in usage
        assert "total_entries" in usage
        assert "cache_details" in usage
        assert usage["total_entries"] >= 1
    
    def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation."""
        # Add data
        cache_manager.cache_query_result("hash1", {"result": "test"})
        
        # Invalidate all
        cache_manager.invalidate_all()
        
        # Should be gone
        result = cache_manager.get_cached_query_result("hash1")
        assert result is None


class TestAsyncRAGProcessor:
    """Test suite for async RAG processor."""
    
    @pytest.fixture
    def processor(self):
        """Create async processor for testing."""
        return AsyncRAGProcessor()
    
    @pytest.mark.asyncio
    async def test_single_query_processing(self, processor):
        """Test processing single query."""
        # Mock RAG system
        mock_rag = Mock()
        mock_rag.aquery = Mock(return_value={"answer": "test", "score": 0.95})
        
        result = await processor._process_single_query("test query", mock_rag, "task1")
        
        assert result == {"answer": "test", "score": 0.95}
        mock_rag.aquery.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_multiple_query_processing(self, processor):
        """Test processing multiple queries."""
        # Mock RAG system
        mock_rag = Mock()
        mock_rag.aquery = Mock(side_effect=[
            {"answer": "answer1", "score": 0.95},
            {"answer": "answer2", "score": 0.90}
        ])
        
        queries = ["query1", "query2"]
        results = await processor.process_multiple_queries(queries, mock_rag, batch_size=2)
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].result == {"answer": "answer1", "score": 0.95}
        assert results[1].result == {"answer": "answer2", "score": 0.90}
    
    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling in async processing."""
        # Mock RAG system that raises exception
        mock_rag = Mock()
        mock_rag.aquery = Mock(side_effect=Exception("Processing failed"))
        
        results = await processor.process_multiple_queries(["query"], mock_rag)
        
        assert len(results) == 1
        assert results[0].success is False
        assert isinstance(results[0].error, Exception)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor):
        """Test batch processing with size limits."""
        # Mock RAG system
        mock_rag = Mock()
        mock_rag.aquery = Mock(return_value={"answer": "test", "score": 0.95})
        
        # Process 5 queries with batch size 2
        queries = [f"query{i}" for i in range(5)]
        results = await processor.process_multiple_queries(queries, mock_rag, batch_size=2)
        
        assert len(results) == 5
        assert all(result.success for result in results)
        
        # Should have made 5 calls total
        assert mock_rag.aquery.call_count == 5
    
    @pytest.mark.asyncio
    async def test_source_merging_union(self, processor):
        """Test source merging with union strategy."""
        sources1 = [{"url": "http://example.com/1", "content": "content1"}]
        sources2 = [{"url": "http://example.com/2", "content": "content2"}]
        sources3 = [{"url": "http://example.com/1", "content": "content1"}]  # Duplicate
        
        all_sources = sources1 + sources2 + sources3
        merged = processor._merge_sources_union(all_sources)
        
        # Should have 2 unique sources (duplicates removed)
        assert len(merged) == 2
        urls = [s["url"] for s in merged]
        assert "http://example.com/1" in urls
        assert "http://example.com/2" in urls


class TestPerformanceOptimizer:
    """Test suite for performance optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create performance optimizer for testing."""
        return PerformanceOptimizer(
            monitoring_window_minutes=1,  # Short window for testing
            optimization_interval_minutes=1,
            min_samples_for_optimization=5
        )
    
    def test_metric_recording(self, optimizer):
        """Test metric recording."""
        optimizer.record_metric("test_metric", 1.5, {"label": "value"})
        
        assert len(optimizer.metrics) == 1
        metric = optimizer.metrics[0]
        assert metric.name == "test_metric"
        assert metric.value == 1.5
        assert metric.labels["label"] == "value"
    
    def test_query_performance_recording(self, optimizer):
        """Test query performance recording."""
        optimizer.record_query_performance(
            response_time=2.5,
            factuality_score=0.95,
            source_count=5,
            success=True,
            query_type="safety"
        )
        
        # Should record multiple metrics
        assert len(optimizer.metrics) >= 4  # response_time, factuality_score, source_count, successful_queries
        
        metric_names = [m.name for m in optimizer.metrics]
        assert "response_time" in metric_names
        assert "factuality_score" in metric_names
        assert "source_count" in metric_names
        assert "successful_queries" in metric_names
    
    def test_performance_analysis(self, optimizer):
        """Test performance analysis."""
        # Record some performance data
        for i in range(10):
            optimizer.record_query_performance(
                response_time=2.0 + i * 0.1,
                factuality_score=0.9 + i * 0.01,
                source_count=5 + i,
                success=True
            )
        
        profile = optimizer._analyze_current_performance()
        
        assert profile.avg_response_time > 2.0
        assert profile.avg_factuality_score > 0.9
        assert profile.avg_source_count >= 5
        assert profile.throughput_qps >= 0
    
    def test_optimization_suggestions_response_time(self, optimizer):
        """Test optimization suggestions for response time."""
        # Create profile with high response time
        mock_profile = Mock()
        mock_profile.avg_response_time = 6.0  # High response time
        mock_profile.avg_factuality_score = 0.95
        mock_profile.throughput_qps = 2.0
        mock_profile.resource_usage = {"cpu_utilization": 0.5}
        
        suggestions = optimizer._suggest_response_time_optimizations(mock_profile)
        
        assert len(suggestions) > 0
        # Should suggest reducing retrieval top_k or similar
        param_names = [s.parameter for s in suggestions]
        assert any("retrieval" in param or "cache" in param or "timeout" in param 
                  for param in param_names)
    
    def test_optimization_suggestions_quality(self, optimizer):
        """Test optimization suggestions for quality."""
        # Create profile with low factuality
        mock_profile = Mock()
        mock_profile.avg_response_time = 2.0
        mock_profile.avg_factuality_score = 0.85  # Low factuality
        mock_profile.throughput_qps = 5.0
        mock_profile.resource_usage = {"cpu_utilization": 0.5}
        
        suggestions = optimizer._suggest_quality_optimizations(mock_profile)
        
        assert len(suggestions) > 0
        # Should suggest increasing retrieval sources or factuality ensemble
        param_names = [s.parameter for s in suggestions]
        assert any("retrieval_top_k" in param or "factuality" in param 
                  for param in param_names)
    
    def test_parameter_updates(self, optimizer):
        """Test parameter updates from optimizations."""
        # Create high-confidence suggestion
        suggestion = Mock()
        suggestion.parameter = "retrieval_top_k"
        suggestion.suggested_value = 25
        suggestion.confidence = 0.8
        
        applied = optimizer._apply_optimizations([suggestion])
        
        assert len(applied) == 1
        assert optimizer.current_parameters["retrieval_top_k"] == 25
        assert "retrieval_top_k" in optimizer.parameter_history
    
    def test_current_parameters_access(self, optimizer):
        """Test getting current parameters."""
        params = optimizer.get_current_parameters()
        
        assert isinstance(params, dict)
        assert "retrieval_top_k" in params
        assert "factuality_threshold" in params
        assert "cache_ttl_queries" in params
    
    def test_performance_summary(self, optimizer):
        """Test performance summary generation."""
        # Add some metrics
        for i in range(10):
            optimizer.record_metric("response_time", 2.0 + i * 0.1)
            optimizer.record_metric("factuality_score", 0.9 + i * 0.01)
        
        summary = optimizer.get_performance_summary()
        
        assert "monitoring_window_minutes" in summary
        assert "total_metrics" in summary
        assert "avg_response_time" in summary
        assert "avg_factuality_score" in summary
        assert "current_parameters" in summary
    
    def test_optimization_state_persistence(self, optimizer, tmp_path):
        """Test saving and loading optimization state."""
        # Update some parameters
        optimizer.current_parameters["test_param"] = 42
        
        # Save state
        state_file = tmp_path / "opt_state.json"
        optimizer.save_optimization_state(str(state_file))
        
        # Create new optimizer and load state
        new_optimizer = PerformanceOptimizer()
        new_optimizer.load_optimization_state(str(state_file))
        
        assert new_optimizer.current_parameters["test_param"] == 42
    
    def test_auto_optimization_lifecycle(self, optimizer):
        """Test auto-optimization start and stop."""
        # Start optimization
        optimizer.start_auto_optimization()
        assert optimizer._optimization_thread is not None
        assert optimizer._optimization_thread.is_alive()
        
        # Stop optimization
        optimizer.stop_auto_optimization()
        time.sleep(0.1)  # Give thread time to stop
        assert optimizer._stop_optimization.is_set()


class TestPerformanceIntegration:
    """Integration tests for performance components."""
    
    @pytest.mark.asyncio
    async def test_cache_and_async_integration(self):
        """Test caching with async processing."""
        cache_manager = CacheManager()
        processor = AsyncRAGProcessor()
        
        # Mock RAG system
        mock_rag = Mock()
        mock_rag.aquery = Mock(return_value={"answer": "cached_result", "score": 0.95})
        
        # Process query
        query = "test query"
        query_hash = cache_manager.create_query_hash(query)
        
        # First call should hit RAG system
        results = await processor.process_multiple_queries([query], mock_rag)
        assert len(results) == 1
        assert results[0].success
        
        # Cache the result
        cache_manager.cache_query_result(query_hash, results[0].result)
        
        # Second call should hit cache (if integrated properly)
        cached_result = cache_manager.get_cached_query_result(query_hash)
        assert cached_result == results[0].result
    
    def test_performance_monitoring_with_caching(self):
        """Test performance monitoring with caching enabled."""
        optimizer = PerformanceOptimizer()
        cache_manager = CacheManager()
        
        # Simulate cache hits and misses
        for i in range(5):
            # Cache miss (slower)
            optimizer.record_query_performance(
                response_time=3.0,
                factuality_score=0.95,
                source_count=5,
                success=True,
                query_type="cache_miss"
            )
            
            # Cache hit (faster)
            optimizer.record_query_performance(
                response_time=0.1,
                factuality_score=0.95,
                source_count=5,
                success=True,
                query_type="cache_hit"
            )
        
        summary = optimizer.get_performance_summary()
        
        # Should show mixed performance
        assert summary["avg_response_time"] < 3.0  # Improved by cache hits
        assert summary["total_metrics"] >= 20  # Multiple metrics per query
    
    def test_end_to_end_performance_optimization(self):
        """Test end-to-end performance optimization cycle."""
        optimizer = PerformanceOptimizer(min_samples_for_optimization=3)
        
        # Simulate poor performance
        for i in range(5):
            optimizer.record_query_performance(
                response_time=5.0,  # Slow
                factuality_score=0.85,  # Low quality
                source_count=3,
                success=True
            )
        
        # Force optimization
        result = optimizer.force_optimization()
        
        assert "optimization_timestamp" in result
        assert "parameters_updated" in result
        
        # Check that parameters were adjusted
        params = optimizer.get_current_parameters()
        # Some parameters should have been optimized
        assert len(optimizer.performance_history) > 0