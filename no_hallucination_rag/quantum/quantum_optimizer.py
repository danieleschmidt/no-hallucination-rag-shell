"""
Advanced performance optimizer for quantum task planning systems.
"""

import logging
import time
import threading
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import psutil
import gc

from .quantum_planner import QuantumTaskPlanner, QuantumTask, TaskState
from .superposition_tasks import SuperpositionTaskManager
from .entanglement_dependencies import EntanglementDependencyGraph


class OptimizationStrategy(Enum):
    """Optimization strategies for quantum systems."""
    CONSERVATIVE = "conservative"   # Safe optimizations, maintain accuracy
    BALANCED = "balanced"          # Balance between performance and accuracy
    AGGRESSIVE = "aggressive"      # Maximum performance, may trade accuracy
    ADAPTIVE = "adaptive"         # Dynamically adjust based on conditions


@dataclass
class PerformanceMetrics:
    """Performance metrics for quantum operations."""
    operation_name: str
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    error_count: int = 0
    last_execution: Optional[datetime] = None
    throughput: float = 0.0  # Operations per second
    
    def update(self, execution_time: float, success: bool = True) -> None:
        """Update metrics with new execution data."""
        self.execution_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.execution_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.last_execution = datetime.utcnow()
        
        if not success:
            self.error_count += 1
        
        # Calculate throughput (ops/sec) over last minute
        if self.execution_count > 0:
            self.throughput = 60.0 / self.avg_time if self.avg_time > 0 else 0.0


@dataclass
class ResourceUsage:
    """System resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_mb: float
    network_io_mb: float
    quantum_coherence_load: float = 0.0
    entanglement_complexity: float = 0.0


class QuantumCache:
    """High-performance cache optimized for quantum operations."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            value, timestamp = self.cache[key]
            
            # Check TTL
            if (datetime.utcnow() - timestamp).total_seconds() > self.ttl_seconds:
                del self.cache[key]
                self.access_times.pop(key, None)
                self.miss_count += 1
                return None
            
            self.access_times[key] = datetime.utcnow()
            self.hit_count += 1
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (value, datetime.utcnow())
            self.access_times[key] = datetime.utcnow()
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries."""
        with self.lock:
            if pattern is None:
                count = len(self.cache)
                self.cache.clear()
                self.access_times.clear()
                return count
            
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.access_times.pop(key, None)
            
            return len(keys_to_remove)
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self.access_times:
            return
        
        # Remove 10% of items (LRU)
        evict_count = max(1, len(self.cache) // 10)
        
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        for key in sorted_keys[:evict_count]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }


class QuantumOptimizer:
    """
    Advanced optimizer for quantum task planning systems.
    
    Provides performance optimization, adaptive caching, resource management,
    and auto-scaling capabilities for quantum operations.
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        enable_auto_optimization: bool = True,
        cache_size: int = 10000,
        max_workers: int = None
    ):
        self.strategy = strategy
        self.enable_auto_optimization = enable_auto_optimization
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        
        # Performance tracking
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.resource_history: deque = deque(maxlen=1000)
        
        # Caching layers
        self.task_cache = QuantumCache(max_size=cache_size // 4, ttl_seconds=1800)
        self.superposition_cache = QuantumCache(max_size=cache_size // 4, ttl_seconds=900)
        self.entanglement_cache = QuantumCache(max_size=cache_size // 4, ttl_seconds=1200)
        self.computation_cache = QuantumCache(max_size=cache_size // 4, ttl_seconds=600)
        
        # Thread pools
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.io_executor = ThreadPoolExecutor(max_workers=min(64, self.max_workers * 2))
        
        # Optimization state
        self.optimization_enabled = True
        self.adaptive_thresholds = {
            "high_load_cpu": 80.0,
            "high_load_memory": 85.0,
            "low_cache_hit_rate": 0.7,
            "high_error_rate": 0.05
        }
        
        # Auto-optimization thread
        self._optimization_thread = None
        self._stop_optimization = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        if enable_auto_optimization:
            self.start_auto_optimization()
        
        self.logger.info(f"Quantum Optimizer initialized with {strategy.value} strategy")
    
    def optimize_task_creation(
        self,
        planner: QuantumTaskPlanner,
        creation_func: Callable,
        *args,
        **kwargs
    ) -> QuantumTask:
        """Optimize task creation with caching and batching."""
        
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key("create_task", args, kwargs)
            
            # Check cache first
            cached_task = self.task_cache.get(cache_key)
            if cached_task and self._should_use_cached_result("create_task"):
                self._update_metrics("create_task_cached", time.time() - start_time, True)
                return cached_task
            
            # Create task with optimization
            if self.strategy == OptimizationStrategy.AGGRESSIVE:
                # Use thread pool for CPU-intensive operations
                future = self.cpu_executor.submit(creation_func, *args, **kwargs)
                task = future.result(timeout=10.0)  # 10 second timeout
            else:
                task = creation_func(*args, **kwargs)
            
            # Cache result
            if task:
                self.task_cache.put(cache_key, task)
            
            self._update_metrics("create_task", time.time() - start_time, True)
            return task
            
        except Exception as e:
            self._update_metrics("create_task", time.time() - start_time, False)
            self.logger.error(f"Task creation optimization failed: {e}")
            # Fallback to direct creation
            return creation_func(*args, **kwargs)
    
    def optimize_superposition_operations(
        self,
        superposition_manager: SuperpositionTaskManager,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """Optimize superposition operations with smart caching."""
        
        start_time = time.time()
        
        try:
            cache_key = self._generate_cache_key(f"superposition_{operation}", args, kwargs)
            
            # Check cache for expensive operations
            if operation in ["calculate_superposition_entropy", "get_superposition_purity", "evolve_superposition"]:
                cached_result = self.superposition_cache.get(cache_key)
                if cached_result is not None:
                    self._update_metrics(f"superposition_{operation}_cached", time.time() - start_time, True)
                    return cached_result
            
            # Execute operation
            method = getattr(superposition_manager, operation)
            result = method(*args, **kwargs)
            
            # Cache result if appropriate
            if result is not None and operation in ["calculate_superposition_entropy", "get_superposition_purity"]:
                self.superposition_cache.put(cache_key, result)
            
            self._update_metrics(f"superposition_{operation}", time.time() - start_time, True)
            return result
            
        except Exception as e:
            self._update_metrics(f"superposition_{operation}", time.time() - start_time, False)
            self.logger.error(f"Superposition {operation} optimization failed: {e}")
            raise
    
    def optimize_entanglement_operations(
        self,
        entanglement_graph: EntanglementDependencyGraph,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """Optimize entanglement graph operations."""
        
        start_time = time.time()
        
        try:
            # Use caching for expensive graph operations
            if operation in ["find_entanglement_path", "get_entanglement_statistics", "calculate_cluster_coherence"]:
                cache_key = self._generate_cache_key(f"entanglement_{operation}", args, kwargs)
                
                cached_result = self.entanglement_cache.get(cache_key)
                if cached_result is not None:
                    self._update_metrics(f"entanglement_{operation}_cached", time.time() - start_time, True)
                    return cached_result
                
                # Execute with potential parallelization
                if self.strategy == OptimizationStrategy.AGGRESSIVE and operation == "get_entanglement_statistics":
                    result = self._parallel_entanglement_stats(entanglement_graph)
                else:
                    method = getattr(entanglement_graph, operation)
                    result = method(*args, **kwargs)
                
                # Cache result
                self.entanglement_cache.put(cache_key, result)
            else:
                method = getattr(entanglement_graph, operation)
                result = method(*args, **kwargs)
            
            self._update_metrics(f"entanglement_{operation}", time.time() - start_time, True)
            return result
            
        except Exception as e:
            self._update_metrics(f"entanglement_{operation}", time.time() - start_time, False)
            self.logger.error(f"Entanglement {operation} optimization failed: {e}")
            raise
    
    def optimize_quantum_sequence(
        self,
        planner: QuantumTaskPlanner,
        available_time: timedelta,
        optimization_hints: Optional[Dict[str, Any]] = None
    ) -> List[QuantumTask]:
        """Optimize quantum task sequence selection with advanced algorithms."""
        
        start_time = time.time()
        
        try:
            cache_key = self._generate_cache_key("optimal_sequence", [available_time], optimization_hints or {})
            
            # Check cache
            cached_sequence = self.computation_cache.get(cache_key)
            if cached_sequence and self._should_use_cached_result("optimal_sequence"):
                self._update_metrics("optimal_sequence_cached", time.time() - start_time, True)
                return cached_sequence
            
            # Get base sequence
            base_sequence = planner.get_optimal_task_sequence(available_time)
            
            # Apply optimization strategy
            if self.strategy == OptimizationStrategy.AGGRESSIVE:
                optimized_sequence = self._aggressive_sequence_optimization(
                    planner, base_sequence, available_time, optimization_hints
                )
            elif self.strategy == OptimizationStrategy.ADAPTIVE:
                optimized_sequence = self._adaptive_sequence_optimization(
                    planner, base_sequence, available_time
                )
            else:
                optimized_sequence = base_sequence
            
            # Cache result
            if optimized_sequence:
                self.computation_cache.put(cache_key, optimized_sequence)
            
            self._update_metrics("optimal_sequence", time.time() - start_time, True)
            return optimized_sequence
            
        except Exception as e:
            self._update_metrics("optimal_sequence", time.time() - start_time, False)
            self.logger.error(f"Sequence optimization failed: {e}")
            return planner.get_optimal_task_sequence(available_time)
    
    async def optimize_parallel_execution(
        self,
        planner: QuantumTaskPlanner,
        tasks: List[QuantumTask],
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute tasks in parallel with optimal resource utilization."""
        
        start_time = time.time()
        max_concurrent = max_concurrent or min(len(tasks), self.max_workers)
        
        try:
            # Group tasks by dependencies and entanglement
            task_groups = self._analyze_task_dependencies(planner, tasks)
            
            # Execute groups in parallel where possible
            results = {"tasks_executed": [], "tasks_failed": [], "execution_groups": []}
            
            for group_idx, task_group in enumerate(task_groups):
                if len(task_group) == 1:
                    # Single task - execute directly
                    task_result = await self._execute_single_task_async(planner, task_group[0])
                    if task_result["success"]:
                        results["tasks_executed"].append(task_result)
                    else:
                        results["tasks_failed"].append(task_result)
                else:
                    # Multiple tasks - execute in parallel
                    group_results = await self._execute_task_group_parallel(planner, task_group, max_concurrent)
                    results["tasks_executed"].extend(group_results["executed"])
                    results["tasks_failed"].extend(group_results["failed"])
                
                results["execution_groups"].append({
                    "group_id": group_idx,
                    "task_count": len(task_group),
                    "parallel_execution": len(task_group) > 1
                })
            
            execution_time = time.time() - start_time
            results["total_execution_time"] = execution_time
            results["parallelization_factor"] = len(tasks) / execution_time if execution_time > 0 else 0
            
            self._update_metrics("parallel_execution", execution_time, True)
            return results
            
        except Exception as e:
            self._update_metrics("parallel_execution", time.time() - start_time, False)
            self.logger.error(f"Parallel execution optimization failed: {e}")
            raise
    
    def start_auto_optimization(self) -> None:
        """Start automatic optimization background thread."""
        
        if self._optimization_thread and self._optimization_thread.is_alive():
            return
        
        self._stop_optimization.clear()
        self._optimization_thread = threading.Thread(target=self._auto_optimization_loop, daemon=True)
        self._optimization_thread.start()
        
        self.logger.info("Auto-optimization started")
    
    def stop_auto_optimization(self) -> None:
        """Stop automatic optimization."""
        
        self._stop_optimization.set()
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5.0)
        
        self.logger.info("Auto-optimization stopped")
    
    def _auto_optimization_loop(self) -> None:
        """Main auto-optimization loop."""
        
        while not self._stop_optimization.wait(60.0):  # Check every minute
            try:
                # Collect system metrics
                resource_usage = self._collect_resource_metrics()
                self.resource_history.append(resource_usage)
                
                # Analyze performance trends
                self._analyze_performance_trends()
                
                # Apply adaptive optimizations
                if self.strategy == OptimizationStrategy.ADAPTIVE:
                    self._apply_adaptive_optimizations(resource_usage)
                
                # Cleanup expired cache entries
                self._cleanup_caches()
                
                # Garbage collection if memory usage is high
                if resource_usage.memory_percent > 85.0:
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"Auto-optimization loop error: {e}")
    
    def _collect_resource_metrics(self) -> ResourceUsage:
        """Collect current system resource usage."""
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / (1024 * 1024),
                disk_io_mb=(disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024) if disk_io else 0,
                network_io_mb=(net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024) if net_io else 0
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect resource metrics: {e}")
            return ResourceUsage(0, 0, 0, 0, 0)
    
    def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and adjust thresholds."""
        
        if len(self.resource_history) < 10:
            return
        
        recent_resources = list(self.resource_history)[-10:]
        
        # Calculate trends
        avg_cpu = sum(r.cpu_percent for r in recent_resources) / len(recent_resources)
        avg_memory = sum(r.memory_percent for r in recent_resources) / len(recent_resources)
        
        # Adjust cache sizes based on memory usage
        if avg_memory > 80.0:
            self._reduce_cache_sizes(0.8)
        elif avg_memory < 50.0:
            self._increase_cache_sizes(1.2)
        
        # Adjust worker counts based on CPU usage
        if avg_cpu > 90.0 and self.max_workers > 2:
            self.max_workers = max(2, int(self.max_workers * 0.8))
        elif avg_cpu < 50.0 and self.max_workers < psutil.cpu_count():
            self.max_workers = min(psutil.cpu_count(), int(self.max_workers * 1.2))
    
    def _apply_adaptive_optimizations(self, resource_usage: ResourceUsage) -> None:
        """Apply adaptive optimizations based on current conditions."""
        
        # High CPU load - reduce parallelization
        if resource_usage.cpu_percent > self.adaptive_thresholds["high_load_cpu"]:
            self.strategy = OptimizationStrategy.CONSERVATIVE
        
        # High memory load - be more aggressive with caching
        elif resource_usage.memory_percent > self.adaptive_thresholds["high_load_memory"]:
            self._cleanup_caches()
            self.strategy = OptimizationStrategy.CONSERVATIVE
        
        # Low resource usage - can be more aggressive
        elif (resource_usage.cpu_percent < 30.0 and resource_usage.memory_percent < 50.0):
            self.strategy = OptimizationStrategy.AGGRESSIVE
        
        else:
            self.strategy = OptimizationStrategy.BALANCED
    
    def _cleanup_caches(self) -> None:
        """Clean up expired cache entries."""
        
        for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache]:
            cache.invalidate()  # This will clean up expired entries
    
    def _reduce_cache_sizes(self, factor: float) -> None:
        """Reduce cache sizes by given factor."""
        
        for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache]:
            new_size = int(cache.max_size * factor)
            if new_size < cache.max_size:
                cache.max_size = max(100, new_size)
                # Evict excess entries
                while len(cache.cache) > cache.max_size:
                    cache._evict_lru()
    
    def _increase_cache_sizes(self, factor: float) -> None:
        """Increase cache sizes by given factor."""
        
        max_total_size = 20000  # Global limit
        current_total = sum(cache.max_size for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache])
        
        if current_total < max_total_size:
            for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache]:
                new_size = int(cache.max_size * factor)
                cache.max_size = min(max_total_size // 4, new_size)
    
    def _generate_cache_key(self, operation: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for operation."""
        
        import hashlib
        
        # Create deterministic string from args and kwargs
        key_parts = [operation]
        
        for arg in args:
            if hasattr(arg, 'id'):
                key_parts.append(str(arg.id))
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _should_use_cached_result(self, operation: str) -> bool:
        """Determine if cached result should be used based on current conditions."""
        
        # Always use cache in conservative mode
        if self.strategy == OptimizationStrategy.CONSERVATIVE:
            return True
        
        # Check cache hit rates
        total_hits = sum(cache.hit_count for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache])
        total_requests = total_hits + sum(cache.miss_count for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache])
        
        hit_rate = total_hits / total_requests if total_requests > 0 else 1.0
        
        # If hit rate is low, be more selective about using cache
        if hit_rate < self.adaptive_thresholds["low_cache_hit_rate"]:
            return operation in ["optimal_sequence", "entanglement_statistics"]
        
        return True
    
    def _aggressive_sequence_optimization(
        self,
        planner: QuantumTaskPlanner,
        base_sequence: List[QuantumTask],
        available_time: timedelta,
        hints: Optional[Dict[str, Any]]
    ) -> List[QuantumTask]:
        """Apply aggressive optimization to task sequence."""
        
        if not base_sequence:
            return base_sequence
        
        # Use genetic algorithm for optimization
        return self._genetic_algorithm_optimization(planner, base_sequence, available_time, hints)
    
    def _adaptive_sequence_optimization(
        self,
        planner: QuantumTaskPlanner,
        base_sequence: List[QuantumTask],
        available_time: timedelta
    ) -> List[QuantumTask]:
        """Apply adaptive optimization based on current system state."""
        
        if not base_sequence:
            return base_sequence
        
        # Analyze current resource usage
        if len(self.resource_history) > 0:
            recent_usage = self.resource_history[-1]
            
            # If system is under high load, prefer simpler optimization
            if recent_usage.cpu_percent > 80.0:
                return self._simple_reorder_optimization(base_sequence)
            else:
                return self._genetic_algorithm_optimization(planner, base_sequence, available_time)
        
        return base_sequence
    
    def _genetic_algorithm_optimization(
        self,
        planner: QuantumTaskPlanner,
        tasks: List[QuantumTask],
        available_time: timedelta,
        hints: Optional[Dict[str, Any]] = None,
        generations: int = 50,
        population_size: int = 20
    ) -> List[QuantumTask]:
        """Use genetic algorithm to optimize task sequence."""
        
        if len(tasks) < 2:
            return tasks
        
        try:
            # Initialize population with random permutations
            population = []
            for _ in range(population_size):
                sequence = tasks.copy()
                np.random.shuffle(sequence)
                population.append(sequence)
            
            # Evolution loop
            for generation in range(generations):
                # Calculate fitness for each sequence
                fitness_scores = []
                for sequence in population:
                    fitness = self._calculate_sequence_fitness(sequence, available_time, hints)
                    fitness_scores.append(fitness)
                
                # Select parents (tournament selection)
                new_population = []
                for _ in range(population_size):
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)
                    
                    # Crossover
                    child = self._crossover(parent1, parent2)
                    
                    # Mutation
                    if np.random.random() < 0.1:  # 10% mutation rate
                        child = self._mutate(child)
                    
                    new_population.append(child)
                
                population = new_population
            
            # Return best sequence
            final_fitness = [self._calculate_sequence_fitness(seq, available_time, hints) for seq in population]
            best_index = np.argmax(final_fitness)
            
            return population[best_index]
            
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization failed: {e}")
            return tasks
    
    def _simple_reorder_optimization(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Simple reordering optimization based on priority and dependencies."""
        
        # Sort by priority and creation time
        return sorted(tasks, key=lambda t: (-t.priority.value, t.created_at))
    
    def _calculate_sequence_fitness(
        self,
        sequence: List[QuantumTask],
        available_time: timedelta,
        hints: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate fitness score for a task sequence."""
        
        fitness = 0.0
        current_time = timedelta()
        
        for i, task in enumerate(sequence):
            # Check if task fits in available time
            if current_time + task.estimated_duration > available_time:
                break
            
            # Priority score (higher priority = higher fitness)
            fitness += task.priority.value * 10
            
            # Earlier in sequence = bonus
            fitness += (len(sequence) - i) * 2
            
            # Quantum properties bonus
            if task.is_coherent():
                fitness += 5
            
            if task.entangled_tasks:
                fitness += len(task.entangled_tasks) * 2
            
            current_time += task.estimated_duration
        
        # Penalty for unused time
        unused_time = available_time - current_time
        fitness -= unused_time.total_seconds() / 3600  # Penalty for each unused hour
        
        return fitness
    
    def _tournament_selection(self, population: List[List[QuantumTask]], fitness_scores: List[float]) -> List[QuantumTask]:
        """Tournament selection for genetic algorithm."""
        
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index].copy()
    
    def _crossover(self, parent1: List[QuantumTask], parent2: List[QuantumTask]) -> List[QuantumTask]:
        """Order crossover for genetic algorithm."""
        
        if len(parent1) <= 2:
            return parent1.copy()
        
        # Order crossover (OX)
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions with parent2 order
        parent2_tasks = [task for task in parent2 if task not in child[start:end]]
        
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = parent2_tasks[j]
                j += 1
        
        return child
    
    def _mutate(self, sequence: List[QuantumTask]) -> List[QuantumTask]:
        """Mutation operation for genetic algorithm."""
        
        if len(sequence) <= 1:
            return sequence
        
        mutated = sequence.copy()
        
        # Swap two random tasks
        i, j = np.random.choice(len(sequence), 2, replace=False)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def _analyze_task_dependencies(self, planner: QuantumTaskPlanner, tasks: List[QuantumTask]) -> List[List[QuantumTask]]:
        """Analyze task dependencies and group for parallel execution."""
        
        # Create dependency graph
        task_map = {task.id: task for task in tasks}
        groups = []
        remaining_tasks = set(tasks)
        
        while remaining_tasks:
            current_group = []
            
            # Find tasks with no unresolved dependencies
            for task in list(remaining_tasks):
                dependencies_resolved = all(
                    dep_id not in task_map or task_map[dep_id] not in remaining_tasks
                    for dep_id in task.dependencies
                )
                
                if dependencies_resolved:
                    current_group.append(task)
            
            if not current_group:
                # Circular dependency or other issue - add remaining tasks to avoid infinite loop
                current_group = list(remaining_tasks)
            
            groups.append(current_group)
            remaining_tasks -= set(current_group)
        
        return groups
    
    async def _execute_single_task_async(self, planner: QuantumTaskPlanner, task: QuantumTask) -> Dict[str, Any]:
        """Execute single task asynchronously."""
        
        try:
            # Simulate task execution
            loop = asyncio.get_event_loop()
            
            # Run in thread pool to avoid blocking
            result = await loop.run_in_executor(
                self.cpu_executor,
                self._execute_task_sync,
                planner,
                task
            )
            
            return {"success": True, "task_id": task.id, "result": result}
            
        except Exception as e:
            return {"success": False, "task_id": task.id, "error": str(e)}
    
    def _execute_task_sync(self, planner: QuantumTaskPlanner, task: QuantumTask) -> Dict[str, Any]:
        """Synchronous task execution."""
        
        # Execute the task
        execution_results = planner.execute_task_sequence([task])
        return execution_results
    
    async def _execute_task_group_parallel(
        self,
        planner: QuantumTaskPlanner,
        task_group: List[QuantumTask],
        max_concurrent: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute group of tasks in parallel."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(task):
            async with semaphore:
                return await self._execute_single_task_async(planner, task)
        
        # Execute all tasks in group concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in task_group],
            return_exceptions=True
        )
        
        executed = []
        failed = []
        
        for result in results:
            if isinstance(result, Exception):
                failed.append({"error": str(result)})
            elif result["success"]:
                executed.append(result)
            else:
                failed.append(result)
        
        return {"executed": executed, "failed": failed}
    
    def _parallel_entanglement_stats(self, entanglement_graph: EntanglementDependencyGraph) -> Dict[str, Any]:
        """Calculate entanglement statistics in parallel."""
        
        try:
            # Use thread pool for parallel computation of different metrics
            futures = []
            
            # Submit different calculations to thread pool
            futures.append(self.cpu_executor.submit(len, entanglement_graph.graph.nodes()))
            futures.append(self.cpu_executor.submit(len, entanglement_graph.graph.edges()))
            futures.append(self.cpu_executor.submit(lambda: sum(dict(entanglement_graph.graph.degree()).values()) / max(1, len(entanglement_graph.graph.nodes()))))
            
            # Wait for results
            total_tasks = futures[0].result(timeout=5.0)
            total_entanglements = futures[1].result(timeout=5.0)
            avg_degree = futures[2].result(timeout=5.0)
            
            return {
                "total_tasks": total_tasks,
                "total_entanglements": total_entanglements,
                "average_degree": avg_degree,
                "computation_method": "parallel"
            }
            
        except Exception as e:
            self.logger.warning(f"Parallel stats computation failed: {e}")
            # Fallback to sequential
            return entanglement_graph.get_entanglement_statistics()
    
    def _update_metrics(self, operation: str, execution_time: float, success: bool) -> None:
        """Update performance metrics."""
        
        if operation not in self.metrics:
            self.metrics[operation] = PerformanceMetrics(operation)
        
        self.metrics[operation].update(execution_time, success)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        
        # Cache statistics
        cache_stats = {
            "task_cache": self.task_cache.get_stats(),
            "superposition_cache": self.superposition_cache.get_stats(),
            "entanglement_cache": self.entanglement_cache.get_stats(),
            "computation_cache": self.computation_cache.get_stats()
        }
        
        # Performance metrics summary
        perf_summary = {}
        for op_name, metrics in self.metrics.items():
            perf_summary[op_name] = {
                "avg_time": metrics.avg_time,
                "min_time": metrics.min_time,
                "max_time": metrics.max_time,
                "execution_count": metrics.execution_count,
                "error_count": metrics.error_count,
                "throughput": metrics.throughput
            }
        
        # Resource usage trend
        resource_trend = {}
        if len(self.resource_history) > 0:
            recent_resources = list(self.resource_history)[-10:]
            resource_trend = {
                "avg_cpu": sum(r.cpu_percent for r in recent_resources) / len(recent_resources),
                "avg_memory": sum(r.memory_percent for r in recent_resources) / len(recent_resources),
                "samples": len(recent_resources)
            }
        
        # Overall cache hit rate
        total_hits = sum(cache.hit_count for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache])
        total_requests = total_hits + sum(cache.miss_count for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache])
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "optimization_strategy": self.strategy.value,
            "auto_optimization_enabled": self.enable_auto_optimization,
            "cache_statistics": cache_stats,
            "overall_cache_hit_rate": overall_hit_rate,
            "performance_metrics": perf_summary,
            "resource_trends": resource_trend,
            "thread_pool_stats": {
                "cpu_workers": self.max_workers,
                "io_workers": min(64, self.max_workers * 2)
            },
            "optimization_recommendations": self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current metrics."""
        
        recommendations = []
        
        # Cache hit rate recommendations
        total_hits = sum(cache.hit_count for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache])
        total_requests = total_hits + sum(cache.miss_count for cache in [self.task_cache, self.superposition_cache, self.entanglement_cache, self.computation_cache])
        hit_rate = total_hits / total_requests if total_requests > 0 else 1.0
        
        if hit_rate < 0.5:
            recommendations.append("Consider increasing cache sizes or adjusting TTL values")
        
        # Resource usage recommendations
        if len(self.resource_history) > 5:
            recent_cpu = [r.cpu_percent for r in list(self.resource_history)[-5:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            
            if avg_cpu > 80:
                recommendations.append("High CPU usage detected - consider reducing parallelization")
            elif avg_cpu < 30:
                recommendations.append("Low CPU usage - can increase parallelization")
        
        # Error rate recommendations
        total_errors = sum(metrics.error_count for metrics in self.metrics.values())
        total_operations = sum(metrics.execution_count for metrics in self.metrics.values())
        error_rate = total_errors / total_operations if total_operations > 0 else 0.0
        
        if error_rate > 0.05:
            recommendations.append("High error rate detected - review error handling and validation")
        
        return recommendations
    
    def shutdown(self) -> None:
        """Shutdown the optimizer and clean up resources."""
        
        self.logger.info("Shutting down Quantum Optimizer...")
        
        # Stop auto-optimization
        self.stop_auto_optimization()
        
        # Shutdown thread pools
        self.cpu_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
        
        # Clear caches
        self.task_cache.invalidate()
        self.superposition_cache.invalidate()
        self.entanglement_cache.invalidate()
        self.computation_cache.invalidate()
        
        self.logger.info("Quantum Optimizer shutdown complete")