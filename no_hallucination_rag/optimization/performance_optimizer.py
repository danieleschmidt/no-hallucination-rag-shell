"""
Performance optimization and auto-tuning for RAG system.
"""

import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import json
import os


@dataclass
class PerformanceMetric:
    """Performance metric with timestamp."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion with confidence score."""
    component: str
    parameter: str
    current_value: Any
    suggested_value: Any
    reason: str
    confidence: float
    estimated_improvement: float


@dataclass
class PerformanceProfile:
    """Performance profile for a specific workload."""
    workload_type: str
    query_patterns: List[str]
    avg_response_time: float
    avg_factuality_score: float
    avg_source_count: float
    throughput_qps: float
    resource_usage: Dict[str, float]
    optimization_suggestions: List[OptimizationSuggestion]


class PerformanceOptimizer:
    """Automatically tunes RAG system performance based on usage patterns."""
    
    def __init__(
        self,
        monitoring_window_minutes: int = 60,
        optimization_interval_minutes: int = 30,
        min_samples_for_optimization: int = 100
    ):
        self.monitoring_window = timedelta(minutes=monitoring_window_minutes)
        self.optimization_interval = timedelta(minutes=optimization_interval_minutes)
        self.min_samples = min_samples_for_optimization
        
        # Performance data storage
        self.metrics: deque = deque(maxlen=10000)
        self.performance_history: List[PerformanceProfile] = []
        
        # Optimization state
        self.last_optimization = datetime.utcnow()
        self.current_parameters: Dict[str, Any] = {}
        self.parameter_history: Dict[str, List[Tuple[datetime, Any]]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background optimization
        self._optimization_thread = None
        self._stop_optimization = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default parameters
        self._initialize_default_parameters()
    
    def _initialize_default_parameters(self) -> None:
        """Initialize default performance parameters."""
        self.current_parameters = {
            # Retrieval parameters
            'retrieval_top_k': 20,
            'retrieval_timeout': 5.0,
            'retrieval_batch_size': 10,
            
            # Ranking parameters
            'ranking_factors': {
                'relevance': 0.3,
                'recency': 0.2,
                'authority': 0.3,
                'consistency': 0.2
            },
            
            # Factuality parameters
            'factuality_threshold': 0.95,
            'factuality_ensemble_size': 3,
            'factuality_timeout': 10.0,
            
            # Caching parameters
            'cache_ttl_queries': 1800,  # 30 minutes
            'cache_ttl_retrieval': 7200,  # 2 hours
            'cache_max_size': 1000,
            
            # Concurrency parameters
            'max_concurrent_queries': 10,
            'max_concurrent_retrievals': 5,
            'thread_pool_size': 8,
            
            # Generation parameters
            'max_answer_length': 2000,
            'generation_timeout': 15.0
        }
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
    
    def record_query_performance(
        self,
        response_time: float,
        factuality_score: float,
        source_count: int,
        success: bool,
        query_type: str = "general",
        error_type: Optional[str] = None
    ) -> None:
        """Record comprehensive query performance data."""
        labels = {
            "query_type": query_type,
            "success": str(success).lower()
        }
        
        if error_type:
            labels["error_type"] = error_type
        
        self.record_metric("response_time", response_time, labels)
        self.record_metric("factuality_score", factuality_score, labels)
        self.record_metric("source_count", source_count, labels)
        
        if success:
            self.record_metric("successful_queries", 1.0, labels)
        else:
            self.record_metric("failed_queries", 1.0, labels)
    
    def start_auto_optimization(self) -> None:
        """Start background auto-optimization."""
        if self._optimization_thread and self._optimization_thread.is_alive():
            return
        
        self._stop_optimization.clear()
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop,
            name="PerformanceOptimizer",
            daemon=True
        )
        self._optimization_thread.start()
        
        self.logger.info("Started auto-optimization")
    
    def stop_auto_optimization(self) -> None:
        """Stop background auto-optimization."""
        self._stop_optimization.set()
        
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5.0)
        
        self.logger.info("Stopped auto-optimization")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while not self._stop_optimization.is_set():
            try:
                # Wait for optimization interval
                if self._stop_optimization.wait(self.optimization_interval.total_seconds()):
                    break
                
                # Check if we have enough data
                if len(self.metrics) < self.min_samples:
                    continue
                
                # Perform optimization
                self._perform_optimization()
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
    
    def _perform_optimization(self) -> None:
        """Perform system optimization based on metrics."""
        with self._lock:
            # Analyze current performance
            current_profile = self._analyze_current_performance()
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(current_profile)
            
            # Apply high-confidence suggestions
            applied_optimizations = self._apply_optimizations(suggestions)
            
            # Record optimization
            if applied_optimizations:
                self.performance_history.append(current_profile)
                self.last_optimization = datetime.utcnow()
                
                self.logger.info(
                    f"Applied {len(applied_optimizations)} optimizations: "
                    f"{[opt.parameter for opt in applied_optimizations]}"
                )
    
    def _analyze_current_performance(self) -> PerformanceProfile:
        """Analyze current system performance."""
        cutoff_time = datetime.utcnow() - self.monitoring_window
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return self._create_empty_profile()
        
        # Group metrics by type
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)
        
        # Calculate performance statistics
        response_times = metric_groups.get('response_time', [1.0])
        factuality_scores = metric_groups.get('factuality_score', [0.95])
        source_counts = metric_groups.get('source_count', [5])
        successful_queries = sum(metric_groups.get('successful_queries', []))
        total_queries = successful_queries + sum(metric_groups.get('failed_queries', []))
        
        # Calculate throughput
        time_window_seconds = self.monitoring_window.total_seconds()
        throughput_qps = total_queries / time_window_seconds if time_window_seconds > 0 else 0
        
        # Estimate resource usage (simplified)
        resource_usage = {
            'cpu_utilization': min(0.8, statistics.mean(response_times) / 5.0),
            'memory_utilization': min(0.9, len(recent_metrics) / 1000.0),
            'cache_hit_rate': 0.7  # Placeholder
        }
        
        return PerformanceProfile(
            workload_type="mixed",
            query_patterns=["general"],
            avg_response_time=statistics.mean(response_times),
            avg_factuality_score=statistics.mean(factuality_scores),
            avg_source_count=statistics.mean(source_counts),
            throughput_qps=throughput_qps,
            resource_usage=resource_usage,
            optimization_suggestions=[]
        )
    
    def _generate_optimization_suggestions(
        self, 
        profile: PerformanceProfile
    ) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on performance profile."""
        suggestions = []
        
        # Response time optimizations
        if profile.avg_response_time > 3.0:
            suggestions.extend(self._suggest_response_time_optimizations(profile))
        
        # Throughput optimizations
        if profile.throughput_qps < 5.0:
            suggestions.extend(self._suggest_throughput_optimizations(profile))
        
        # Quality optimizations
        if profile.avg_factuality_score < 0.9:
            suggestions.extend(self._suggest_quality_optimizations(profile))
        
        # Resource optimizations
        if profile.resource_usage.get('cpu_utilization', 0) > 0.8:
            suggestions.extend(self._suggest_resource_optimizations(profile))
        
        return suggestions
    
    def _suggest_response_time_optimizations(
        self, 
        profile: PerformanceProfile
    ) -> List[OptimizationSuggestion]:
        """Suggest optimizations to improve response time."""
        suggestions = []
        
        # Reduce retrieval top_k if response time is high
        current_top_k = self.current_parameters.get('retrieval_top_k', 20)
        if current_top_k > 10 and profile.avg_response_time > 5.0:
            suggestions.append(OptimizationSuggestion(
                component="retrieval",
                parameter="retrieval_top_k",
                current_value=current_top_k,
                suggested_value=max(10, current_top_k - 5),
                reason="Reduce retrieval time by fetching fewer sources",
                confidence=0.8,
                estimated_improvement=0.3
            ))
        
        # Increase cache TTL to improve hit rate
        current_ttl = self.current_parameters.get('cache_ttl_queries', 1800)
        if current_ttl < 3600:
            suggestions.append(OptimizationSuggestion(
                component="caching",
                parameter="cache_ttl_queries",
                current_value=current_ttl,
                suggested_value=min(3600, current_ttl * 1.5),
                reason="Increase cache hit rate to reduce processing time",
                confidence=0.7,
                estimated_improvement=0.2
            ))
        
        # Reduce factuality timeout for faster processing
        current_timeout = self.current_parameters.get('factuality_timeout', 10.0)
        if current_timeout > 5.0 and profile.avg_factuality_score > 0.95:
            suggestions.append(OptimizationSuggestion(
                component="factuality",
                parameter="factuality_timeout",
                current_value=current_timeout,
                suggested_value=max(5.0, current_timeout * 0.8),
                reason="Reduce factuality verification time while maintaining quality",
                confidence=0.6,
                estimated_improvement=0.15
            ))
        
        return suggestions
    
    def _suggest_throughput_optimizations(
        self, 
        profile: PerformanceProfile
    ) -> List[OptimizationSuggestion]:
        """Suggest optimizations to improve throughput."""
        suggestions = []
        
        # Increase concurrency
        current_concurrent = self.current_parameters.get('max_concurrent_queries', 10)
        if current_concurrent < 20 and profile.resource_usage.get('cpu_utilization', 0) < 0.7:
            suggestions.append(OptimizationSuggestion(
                component="concurrency",
                parameter="max_concurrent_queries",
                current_value=current_concurrent,
                suggested_value=min(20, current_concurrent + 5),
                reason="Increase concurrency to improve throughput",
                confidence=0.7,
                estimated_improvement=0.4
            ))
        
        # Increase thread pool size
        current_pool_size = self.current_parameters.get('thread_pool_size', 8)
        if current_pool_size < 16:
            suggestions.append(OptimizationSuggestion(
                component="threading",
                parameter="thread_pool_size",
                current_value=current_pool_size,
                suggested_value=min(16, current_pool_size + 2),
                reason="Increase thread pool for better parallel processing",
                confidence=0.6,
                estimated_improvement=0.2
            ))
        
        return suggestions
    
    def _suggest_quality_optimizations(
        self, 
        profile: PerformanceProfile
    ) -> List[OptimizationSuggestion]:
        """Suggest optimizations to improve quality."""
        suggestions = []
        
        # Increase retrieval top_k for better source coverage
        current_top_k = self.current_parameters.get('retrieval_top_k', 20)
        if current_top_k < 30 and profile.avg_factuality_score < 0.9:
            suggestions.append(OptimizationSuggestion(
                component="retrieval",
                parameter="retrieval_top_k",
                current_value=current_top_k,
                suggested_value=min(30, current_top_k + 5),
                reason="Retrieve more sources to improve factuality",
                confidence=0.8,
                estimated_improvement=0.1
            ))
        
        # Increase factuality ensemble size
        current_ensemble = self.current_parameters.get('factuality_ensemble_size', 3)
        if current_ensemble < 5:
            suggestions.append(OptimizationSuggestion(
                component="factuality",
                parameter="factuality_ensemble_size",
                current_value=current_ensemble,
                suggested_value=min(5, current_ensemble + 1),
                reason="Use more factuality models for better verification",
                confidence=0.7,
                estimated_improvement=0.05
            ))
        
        return suggestions
    
    def _suggest_resource_optimizations(
        self, 
        profile: PerformanceProfile
    ) -> List[OptimizationSuggestion]:
        """Suggest optimizations to reduce resource usage."""
        suggestions = []
        
        # Reduce batch size to lower memory usage
        current_batch_size = self.current_parameters.get('retrieval_batch_size', 10)
        if current_batch_size > 5:
            suggestions.append(OptimizationSuggestion(
                component="retrieval",
                parameter="retrieval_batch_size",
                current_value=current_batch_size,
                suggested_value=max(5, current_batch_size - 2),
                reason="Reduce batch size to lower memory usage",
                confidence=0.6,
                estimated_improvement=0.1
            ))
        
        # Reduce cache size to free memory
        current_cache_size = self.current_parameters.get('cache_max_size', 1000)
        if current_cache_size > 500:
            suggestions.append(OptimizationSuggestion(
                component="caching",
                parameter="cache_max_size",
                current_value=current_cache_size,
                suggested_value=max(500, int(current_cache_size * 0.8)),
                reason="Reduce cache size to free memory",
                confidence=0.5,
                estimated_improvement=0.05
            ))
        
        return suggestions
    
    def _apply_optimizations(
        self, 
        suggestions: List[OptimizationSuggestion]
    ) -> List[OptimizationSuggestion]:
        """Apply high-confidence optimization suggestions."""
        applied = []
        
        for suggestion in suggestions:
            # Only apply high-confidence suggestions
            if suggestion.confidence >= 0.7:
                # Update parameter
                self.current_parameters[suggestion.parameter] = suggestion.suggested_value
                
                # Record parameter change
                self.parameter_history[suggestion.parameter].append(
                    (datetime.utcnow(), suggestion.suggested_value)
                )
                
                applied.append(suggestion)
        
        return applied
    
    def _create_empty_profile(self) -> PerformanceProfile:
        """Create empty performance profile."""
        return PerformanceProfile(
            workload_type="unknown",
            query_patterns=[],
            avg_response_time=1.0,
            avg_factuality_score=0.95,
            avg_source_count=5,
            throughput_qps=0.0,
            resource_usage={},
            optimization_suggestions=[]
        )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimized parameters."""
        with self._lock:
            return self.current_parameters.copy()
    
    def get_optimization_history(self) -> List[PerformanceProfile]:
        """Get optimization history."""
        with self._lock:
            return self.performance_history.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        cutoff_time = datetime.utcnow() - self.monitoring_window
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"status": "no_data"}
        
        # Group metrics
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)
        
        response_times = metric_groups.get('response_time', [])
        factuality_scores = metric_groups.get('factuality_score', [])
        
        return {
            "monitoring_window_minutes": self.monitoring_window.total_seconds() / 60,
            "total_metrics": len(recent_metrics),
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            "avg_factuality_score": statistics.mean(factuality_scores) if factuality_scores else 0,
            "last_optimization": self.last_optimization.isoformat(),
            "optimizations_applied": len(self.performance_history),
            "current_parameters": self.current_parameters
        }
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate optimization and return results."""
        self._perform_optimization()
        
        return {
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "parameters_updated": self.current_parameters,
            "performance_summary": self.get_performance_summary()
        }
    
    def save_optimization_state(self, filepath: str) -> None:
        """Save optimization state to file."""
        state = {
            "current_parameters": self.current_parameters,
            "parameter_history": {
                param: [(timestamp.isoformat(), value) for timestamp, value in history]
                for param, history in self.parameter_history.items()
            },
            "performance_history": [
                {
                    "workload_type": profile.workload_type,
                    "avg_response_time": profile.avg_response_time,
                    "avg_factuality_score": profile.avg_factuality_score,
                    "throughput_qps": profile.throughput_qps,
                    "resource_usage": profile.resource_usage
                }
                for profile in self.performance_history
            ],
            "last_optimization": self.last_optimization.isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Saved optimization state to {filepath}")
    
    def load_optimization_state(self, filepath: str) -> None:
        """Load optimization state from file."""
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_parameters = state.get("current_parameters", {})
            
            # Restore parameter history
            self.parameter_history.clear()
            for param, history in state.get("parameter_history", {}).items():
                self.parameter_history[param] = [
                    (datetime.fromisoformat(timestamp), value)
                    for timestamp, value in history
                ]
            
            # Restore last optimization time
            if "last_optimization" in state:
                self.last_optimization = datetime.fromisoformat(state["last_optimization"])
            
            self.logger.info(f"Loaded optimization state from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load optimization state: {e}")