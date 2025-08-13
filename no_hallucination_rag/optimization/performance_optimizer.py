
"""
Advanced performance optimization system with adaptive tuning.
"""

import time
import threading
import logging
import statistics
import psutil
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import json
import numpy as np


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    timestamp: datetime
    component: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of performance optimization."""
    parameter_name: str
    old_value: Any
    new_value: Any
    improvement: float
    confidence: float
    applied: bool = False


class PerformanceOptimizer(ABC):
    """Abstract base class for performance optimizers."""
    
    @abstractmethod
    def optimize(self, metrics: List[PerformanceMetric]) -> List[OptimizationResult]:
        """Optimize based on performance metrics."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        pass
    
    @abstractmethod
    def apply_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Apply optimization parameters."""
        pass


class CacheOptimizer(PerformanceOptimizer):
    """Optimizer for cache parameters."""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters with ranges
        self.parameter_ranges = {
            "l1_size": (100, 10000),
            "l2_size": (1000, 100000),
            "l3_size": (10000, 1000000),
            "eviction_threshold": (0.7, 0.95)
        }
        
        # Current parameters
        self.parameters = {
            "l1_size": 1000,
            "l2_size": 10000,
            "l3_size": 100000,
            "eviction_threshold": 0.8
        }
        
        # Performance tracking
        self.performance_data = deque(maxlen=1000)
        self.optimization_history = []
        self.auto_optimization_enabled = False
        
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        return self.parameters.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_data:
            return {"status": "no_data"}
        
        response_times = [d.get("response_time", 0) for d in self.performance_data]
        factuality_scores = [d.get("factuality_score", 0) for d in self.performance_data]
        
        return {
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "avg_factuality": statistics.mean(factuality_scores) if factuality_scores else 0,
            "total_queries": len(self.performance_data),
            "success_rate": sum(1 for d in self.performance_data if d.get("success", False)) / len(self.performance_data)
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history
    
    def record_query_performance(self, **kwargs):
        """Record query performance data."""
        self.performance_data.append({
            "timestamp": datetime.now(),
            **kwargs
        })
    
    def start_auto_optimization(self):
        """Start automatic optimization."""
        self.auto_optimization_enabled = True
    
    def stop_auto_optimization(self):
        """Stop automatic optimization."""
        self.auto_optimization_enabled = False
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force optimization run."""
        if not self.performance_data:
            return {"status": "no_data"}
        
        # Simple optimization: adjust cache sizes based on hit rates
        current_params = self.parameters.copy()
        
        # Simulate optimization
        if len(self.performance_data) > 10:
            avg_response_time = statistics.mean(
                d.get("response_time", 0) for d in list(self.performance_data)[-10:]
            )
            
            if avg_response_time > 2.0:  # If slow, increase cache
                self.parameters["l1_size"] = min(
                    self.parameters["l1_size"] * 1.1, 
                    self.parameter_ranges["l1_size"][1]
                )
        
        optimization_result = {
            "timestamp": datetime.now().isoformat(),
            "old_parameters": current_params,
            "new_parameters": self.parameters.copy(),
            "improvement_estimate": 0.05  # 5% improvement estimate
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def save_optimization_state(self, filepath: str):
        """Save optimization state to file."""
        state = {
            "parameters": self.parameters,
            "optimization_history": self.optimization_history[-10:],  # Last 10
            "performance_summary": self.get_performance_summary()
        }
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def optimize(self, metrics: List[PerformanceMetric]) -> List[OptimizationResult]:
        """Optimize cache parameters based on hit rates and response times."""
        results = []
        
        # Filter cache-related metrics
        cache_metrics = [m for m in metrics if m.component == "cache"]
        
        if not cache_metrics:
            return results
        
        # Calculate average hit rate
        hit_rate_metrics = [m for m in cache_metrics if m.name == "hit_rate"]
        if hit_rate_metrics:
            avg_hit_rate = statistics.mean(m.value for m in hit_rate_metrics)
            
            # If hit rate is low, increase cache sizes
            if avg_hit_rate < 0.7:  # Less than 70% hit rate
                for size_param in ["l1_size", "l2_size", "l3_size"]:
                    old_value = self.parameters[size_param]
                    min_val, max_val = self.parameter_ranges[size_param]
                    new_value = min(max_val, int(old_value * 1.2))  # Increase by 20%
                    
                    if new_value != old_value:
                        improvement = (new_value - old_value) / old_value
                        confidence = min(0.8, (0.7 - avg_hit_rate) * 2)  # Higher confidence for lower hit rates
                        
                        results.append(OptimizationResult(
                            parameter_name=size_param,
                            old_value=old_value,
                            new_value=new_value,
                            improvement=improvement,
                            confidence=confidence
                        ))
            
            # If hit rate is very high, we might be over-caching
            elif avg_hit_rate > 0.95:
                # Consider slightly reducing cache sizes to free memory
                for size_param in ["l3_size", "l2_size"]:  # Start with larger caches
                    old_value = self.parameters[size_param]
                    min_val, max_val = self.parameter_ranges[size_param]
                    new_value = max(min_val, int(old_value * 0.9))  # Reduce by 10%
                    
                    if new_value != old_value:
                        improvement = 0.1  # Small improvement in memory usage
                        confidence = 0.3   # Low confidence for reduction
                        
                        results.append(OptimizationResult(
                            parameter_name=size_param,
                            old_value=old_value,
                            new_value=new_value,
                            improvement=improvement,
                            confidence=confidence
                        ))
                        break  # Only adjust one parameter at a time
        
        # Optimize eviction threshold based on memory pressure
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 85:  # High memory pressure
            old_threshold = self.parameters["eviction_threshold"]
            new_threshold = max(0.7, old_threshold - 0.1)
            
            if new_threshold != old_threshold:
                results.append(OptimizationResult(
                    parameter_name="eviction_threshold",
                    old_value=old_threshold,
                    new_value=new_threshold,
                    improvement=0.15,  # Memory reduction benefit
                    confidence=0.7
                ))
        
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current cache parameters."""
        return self.parameters.copy()
    
    def apply_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Apply cache parameters."""
        try:
            # Update internal parameters
            for key, value in parameters.items():
                if key in self.parameters:
                    self.parameters[key] = value
            
            # Apply to cache manager (implementation depends on cache manager API)
            if hasattr(self.cache_manager, 'update_parameters'):
                self.cache_manager.update_parameters(parameters)
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply cache parameters: {e}")
            return False


class ConcurrencyOptimizer(PerformanceOptimizer):
    """Optimizer for concurrency parameters."""
    
    def __init__(self, concurrency_manager):
        self.concurrency_manager = concurrency_manager
        self.logger = logging.getLogger(__name__)
        
        self.parameter_ranges = {
            "thread_pool_size": (1, 50),
            "connection_pool_size": (5, 100),
            "queue_size": (100, 10000),
            "batch_size": (1, 100)
        }
        
        self.parameters = {
            "thread_pool_size": 10,
            "connection_pool_size": 20,
            "queue_size": 1000,
            "batch_size": 10
        }
    
    def optimize(self, metrics: List[PerformanceMetric]) -> List[OptimizationResult]:
        """Optimize concurrency parameters based on utilization and throughput."""
        results = []
        
        # Filter concurrency-related metrics
        concurrency_metrics = [m for m in metrics if m.component in ["thread_pool", "connection_pool", "queue"]]
        
        if not concurrency_metrics:
            return results
        
        # Optimize thread pool size based on utilization
        thread_metrics = [m for m in concurrency_metrics if m.component == "thread_pool"]
        if thread_metrics:
            utilization_metrics = [m for m in thread_metrics if m.name == "utilization"]
            if utilization_metrics:
                avg_utilization = statistics.mean(m.value for m in utilization_metrics)
                
                if avg_utilization > 0.85:  # High utilization - scale up
                    old_size = self.parameters["thread_pool_size"]
                    min_val, max_val = self.parameter_ranges["thread_pool_size"]
                    new_size = min(max_val, int(old_size * 1.3))  # Increase by 30%
                    
                    if new_size != old_size:
                        improvement = (new_size - old_size) / old_size
                        confidence = min(0.8, (avg_utilization - 0.85) * 4)
                        
                        results.append(OptimizationResult(
                            parameter_name="thread_pool_size",
                            old_value=old_size,
                            new_value=new_size,
                            improvement=improvement,
                            confidence=confidence
                        ))
                
                elif avg_utilization < 0.3:  # Low utilization - scale down
                    old_size = self.parameters["thread_pool_size"]
                    min_val, max_val = self.parameter_ranges["thread_pool_size"]
                    new_size = max(min_val, int(old_size * 0.8))  # Decrease by 20%
                    
                    if new_size != old_size:
                        improvement = 0.1  # Resource efficiency improvement
                        confidence = 0.5   # Lower confidence for scaling down
                        
                        results.append(OptimizationResult(
                            parameter_name="thread_pool_size",
                            old_value=old_size,
                            new_value=new_size,
                            improvement=improvement,
                            confidence=confidence
                        ))
        
        # Optimize connection pool size based on connection pressure
        connection_metrics = [m for m in concurrency_metrics if m.component == "connection_pool"]
        if connection_metrics:
            pressure_metrics = [m for m in connection_metrics if m.name == "connection_pressure"]
            if pressure_metrics:
                avg_pressure = statistics.mean(m.value for m in pressure_metrics)
                
                if avg_pressure > 0.8:  # High pressure - increase pool
                    old_size = self.parameters["connection_pool_size"]
                    min_val, max_val = self.parameter_ranges["connection_pool_size"]
                    new_size = min(max_val, int(old_size * 1.25))  # Increase by 25%
                    
                    if new_size != old_size:
                        improvement = (new_size - old_size) / old_size
                        confidence = min(0.7, (avg_pressure - 0.8) * 3)
                        
                        results.append(OptimizationResult(
                            parameter_name="connection_pool_size",
                            old_value=old_size,
                            new_value=new_size,
                            improvement=improvement,
                            confidence=confidence
                        ))
        
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current concurrency parameters."""
        return self.parameters.copy()
    
    def apply_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Apply concurrency parameters."""
        try:
            for key, value in parameters.items():
                if key in self.parameters:
                    self.parameters[key] = value
            
            # Apply to concurrency manager
            if hasattr(self.concurrency_manager, 'update_parameters'):
                self.concurrency_manager.update_parameters(parameters)
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply concurrency parameters: {e}")
            return False


class SystemOptimizer(PerformanceOptimizer):
    """Optimizer for system-level parameters."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.parameter_ranges = {
            "gc_threshold": (100, 10000),
            "buffer_size": (1024, 1024*1024),
            "timeout_seconds": (1, 300)
        }
        
        self.parameters = {
            "gc_threshold": 1000,
            "buffer_size": 8192,
            "timeout_seconds": 30
        }
    
    def optimize(self, metrics: List[PerformanceMetric]) -> List[OptimizationResult]:
        """Optimize system parameters based on overall performance."""
        results = []
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        # Optimize based on resource usage
        if cpu_usage > 80:  # High CPU usage
            # Reduce GC frequency to save CPU
            old_threshold = self.parameters["gc_threshold"]
            min_val, max_val = self.parameter_ranges["gc_threshold"]
            new_threshold = min(max_val, int(old_threshold * 1.5))
            
            if new_threshold != old_threshold:
                results.append(OptimizationResult(
                    parameter_name="gc_threshold",
                    old_value=old_threshold,
                    new_value=new_threshold,
                    improvement=0.1,  # CPU reduction benefit
                    confidence=0.6
                ))
        
        if memory_usage > 85:  # High memory usage
            # Increase GC frequency to free memory
            old_threshold = self.parameters["gc_threshold"]
            min_val, max_val = self.parameter_ranges["gc_threshold"]
            new_threshold = max(min_val, int(old_threshold * 0.7))
            
            if new_threshold != old_threshold:
                results.append(OptimizationResult(
                    parameter_name="gc_threshold",
                    old_value=old_threshold,
                    new_value=new_threshold,
                    improvement=0.15,  # Memory reduction benefit
                    confidence=0.7
                ))
        
        # Optimize buffer sizes based on I/O patterns
        io_metrics = [m for m in metrics if m.name in ["read_time", "write_time"]]
        if io_metrics:
            avg_io_time = statistics.mean(m.value for m in io_metrics)
            
            if avg_io_time > 0.1:  # Slow I/O - increase buffer size
                old_buffer = self.parameters["buffer_size"]
                min_val, max_val = self.parameter_ranges["buffer_size"]
                new_buffer = min(max_val, old_buffer * 2)
                
                if new_buffer != old_buffer:
                    results.append(OptimizationResult(
                        parameter_name="buffer_size",
                        old_value=old_buffer,
                        new_value=new_buffer,
                        improvement=0.2,
                        confidence=0.6
                    ))
        
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current system parameters."""
        return self.parameters.copy()
    
    def apply_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Apply system parameters."""
        try:
            for key, value in parameters.items():
                if key in self.parameters:
                    self.parameters[key] = value
            
            # Apply system-level changes (implementation specific)
            self.logger.info(f"Applied system parameters: {parameters}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply system parameters: {e}")
            return False


class AdaptivePerformanceManager:
    """Adaptive performance management with multiple optimizers."""
    
    def __init__(self):
        self.optimizers: Dict[str, PerformanceOptimizer] = {}
        self.metrics_history: deque = deque(maxlen=10000)
        self.optimization_history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.optimization_interval = 300  # 5 minutes
        self.min_confidence_threshold = 0.5
        self.optimization_active = False
        
        # A/B testing for optimization validation
        self.ab_test_duration = 600  # 10 minutes
        self.current_ab_test = None
    
    def add_optimizer(self, name: str, optimizer: PerformanceOptimizer):
        """Add performance optimizer."""
        with self.lock:
            self.optimizers[name] = optimizer
            self.logger.info(f"Added optimizer: {name}")
    
    def record_metric(self, metric: PerformanceMetric):
        """Record performance metric."""
        with self.lock:
            self.metrics_history.append(metric)
    
    def start_optimization(self):
        """Start automatic performance optimization."""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        optimization_thread.start()
        self.logger.info("Started adaptive performance optimization")
    
    def stop_optimization(self):
        """Stop automatic performance optimization."""
        self.optimization_active = False
        self.logger.info("Stopped adaptive performance optimization")
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate optimization cycle."""
        with self.lock:
            return self._run_optimization_cycle()
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.optimization_active:
            try:
                time.sleep(self.optimization_interval)
                
                with self.lock:
                    self._run_optimization_cycle()
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                time.sleep(60)  # Back off on errors
    
    def _run_optimization_cycle(self) -> Dict[str, Any]:
        """Run one optimization cycle."""
        if not self.metrics_history:
            return {"status": "no_metrics"}
        
        # Get recent metrics for analysis
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"status": "no_recent_metrics"}
        
        optimization_results = {}
        applied_optimizations = []
        
        # Run each optimizer
        for name, optimizer in self.optimizers.items():
            try:
                results = optimizer.optimize(recent_metrics)
                
                # Apply high-confidence optimizations
                for result in results:
                    if result.confidence >= self.min_confidence_threshold:
                        # Get current parameters
                        current_params = optimizer.get_parameters()
                        
                        # Create new parameters with optimization
                        new_params = current_params.copy()
                        new_params[result.parameter_name] = result.new_value
                        
                        # Apply optimization
                        if optimizer.apply_parameters(new_params):
                            result.applied = True
                            applied_optimizations.append({
                                "optimizer": name,
                                "parameter": result.parameter_name,
                                "old_value": result.old_value,
                                "new_value": result.new_value,
                                "improvement": result.improvement,
                                "confidence": result.confidence
                            })
                            
                            self.logger.info(
                                f"Applied optimization: {name}.{result.parameter_name} "
                                f"{result.old_value} -> {result.new_value} "
                                f"(confidence: {result.confidence:.2f})"
                            )
                
                optimization_results[name] = [
                    {
                        "parameter": r.parameter_name,
                        "improvement": r.improvement,
                        "confidence": r.confidence,
                        "applied": r.applied
                    }
                    for r in results
                ]
                
            except Exception as e:
                self.logger.error(f"Optimizer {name} failed: {e}")
                optimization_results[name] = {"error": str(e)}
        
        # Record optimization cycle
        cycle_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_analyzed": len(recent_metrics),
            "optimizations": optimization_results,
            "applied": applied_optimizations
        }
        
        self.optimization_history.append(cycle_record)
        
        # Limit history size
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-50:]
        
        return cycle_record
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self.lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            # Calculate overall performance trends
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
            
            # Group by component and metric name
            grouped_metrics = defaultdict(list)
            for metric in recent_metrics:
                key = f"{metric.component}.{metric.name}"
                grouped_metrics[key].append(metric.value)
            
            # Calculate statistics for each metric group
            metric_stats = {}
            for key, values in grouped_metrics.items():
                if len(values) >= 3:  # Need at least 3 data points
                    metric_stats[key] = {
                        "count": len(values),
                        "avg": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0
                    }
            
            return {
                "total_metrics": len(self.metrics_history),
                "recent_metrics": len(recent_metrics),
                "metric_stats": metric_stats,
                "optimization_cycles": len(self.optimization_history),
                "last_optimization": self.optimization_history[-1] if self.optimization_history else None,
                "optimizers": list(self.optimizers.keys()),
                "optimization_active": self.optimization_active
            }
    
    def get_current_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get current parameters from all optimizers."""
        with self.lock:
            return {
                name: optimizer.get_parameters()
                for name, optimizer in self.optimizers.items()
            }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        with self.lock:
            return self.optimization_history.copy()


# Global performance manager
global_performance_manager = AdaptivePerformanceManager()
