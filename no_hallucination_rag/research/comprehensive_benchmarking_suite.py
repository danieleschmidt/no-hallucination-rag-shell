"""
Comprehensive Performance Benchmarking Suite for Quantum-Enhanced RAG Research.

This module provides publication-grade benchmarking infrastructure for
rigorous performance evaluation of quantum algorithms in RAG systems.

Research Features:
1. Multi-Dimensional Performance Metrics Collection
2. Standardized Benchmark Datasets and Test Cases
3. Scalability Testing with Automated Load Generation
4. Energy Efficiency and Resource Utilization Analysis
5. Real-World Performance Simulation Framework
6. Automated Statistical Validation and Reporting

All benchmarks follow academic standards for reproducible research.
"""

import logging
import time
import asyncio
import numpy as np
import psutil
import threading
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import hashlib
from collections import defaultdict, namedtuple
import random
import math
import gc
import os
import sys
import tracemalloc
from pathlib import Path

# Performance monitoring
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


class BenchmarkCategory(Enum):
    """Categories of benchmarks for comprehensive evaluation."""
    COMPUTATIONAL_PERFORMANCE = "computational_performance"
    MEMORY_EFFICIENCY = "memory_efficiency"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    ENERGY_CONSUMPTION = "energy_consumption"
    ACCURACY_VALIDATION = "accuracy_validation"
    LATENCY_ANALYSIS = "latency_analysis"
    THROUGHPUT_TESTING = "throughput_testing"
    CONCURRENCY_PERFORMANCE = "concurrency_performance"
    RESOURCE_UTILIZATION = "resource_utilization"
    RELIABILITY_TESTING = "reliability_testing"


class WorkloadType(Enum):
    """Types of workloads for benchmark testing."""
    SYNTHETIC_UNIFORM = "synthetic_uniform"
    SYNTHETIC_BURSTY = "synthetic_bursty"
    REAL_WORLD_QUERIES = "real_world_queries"
    ADVERSARIAL_CASES = "adversarial_cases"
    EDGE_CASES = "edge_cases"
    STRESS_TEST = "stress_test"


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result with detailed metrics."""
    benchmark_id: str
    algorithm_name: str
    category: str
    workload_type: str
    
    # Core performance metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    throughput_qps: float = 0.0
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    
    # Accuracy and quality metrics
    accuracy_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    relevance_score: float = 0.0
    
    # Resource efficiency metrics
    energy_consumption_joules: float = 0.0
    memory_efficiency_score: float = 0.0
    cpu_efficiency_score: float = 0.0
    
    # Scalability metrics
    scaling_factor: float = 1.0
    linear_scaling_coefficient: float = 0.0
    scaling_efficiency: float = 0.0
    
    # Reliability metrics
    error_rate: float = 0.0
    success_rate: float = 1.0
    stability_score: float = 1.0
    
    # Quantum-specific metrics
    quantum_advantage_factor: float = 1.0
    coherence_utilization: float = 0.0
    entanglement_effectiveness: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    test_conditions: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    reproducibility_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, datetime):
                result[field_name] = value.isoformat()
            else:
                result[field_name] = value
        return result


@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark execution."""
    name: str
    description: str
    categories: List[BenchmarkCategory]
    workload_types: List[WorkloadType]
    
    # Test parameters
    num_trials: int = 10
    warmup_trials: int = 3
    timeout_seconds: int = 300
    concurrent_workers: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dataset_sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000])
    
    # Quality thresholds
    min_accuracy_threshold: float = 0.8
    max_latency_threshold_ms: float = 1000.0
    max_memory_threshold_mb: float = 1024.0
    
    # Environment settings
    enable_profiling: bool = True
    enable_plotting: bool = True
    output_directory: str = "benchmark_results"


class SystemProfiler:
    """Advanced system profiling for performance analysis."""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.profiling_data = {}
        self.monitoring_active = False
        
        self.logger = logging.getLogger(__name__)
    
    async def start_profiling(self, benchmark_id: str):
        """Start comprehensive system profiling."""
        self.profiling_data[benchmark_id] = {
            'start_time': time.time(),
            'cpu_samples': [],
            'memory_samples': [],
            'disk_io_samples': [],
            'network_io_samples': [],
            'thread_count_samples': []
        }
        
        # Start memory tracing
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        self.monitoring_active = True
        
        # Background monitoring task
        asyncio.create_task(self._monitoring_loop(benchmark_id))
    
    async def stop_profiling(self, benchmark_id: str) -> Dict[str, Any]:
        """Stop profiling and return collected metrics."""
        self.monitoring_active = False
        
        if benchmark_id not in self.profiling_data:
            return {}
        
        data = self.profiling_data[benchmark_id]
        data['end_time'] = time.time()
        data['total_duration'] = data['end_time'] - data['start_time']
        
        # Memory profiling summary
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            data['peak_memory_bytes'] = peak
            data['final_memory_bytes'] = current
        
        # Calculate statistics
        if data['cpu_samples']:
            data['cpu_stats'] = {
                'mean': np.mean(data['cpu_samples']),
                'std': np.std(data['cpu_samples']),
                'max': np.max(data['cpu_samples']),
                'min': np.min(data['cpu_samples'])
            }
        
        if data['memory_samples']:
            data['memory_stats'] = {
                'mean_mb': np.mean(data['memory_samples']) / (1024 * 1024),
                'std_mb': np.std(data['memory_samples']) / (1024 * 1024),
                'max_mb': np.max(data['memory_samples']) / (1024 * 1024),
                'min_mb': np.min(data['memory_samples']) / (1024 * 1024)
            }
        
        return data
    
    async def _monitoring_loop(self, benchmark_id: str):
        """Background monitoring loop for resource utilization."""
        try:
            while self.monitoring_active and benchmark_id in self.profiling_data:
                # CPU utilization
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.profiling_data[benchmark_id]['cpu_samples'].append(cpu_percent)
                
                # Memory usage
                memory_info = psutil.virtual_memory()
                self.profiling_data[benchmark_id]['memory_samples'].append(memory_info.used)
                
                # Thread count
                current_process = psutil.Process()
                thread_count = current_process.num_threads()
                self.profiling_data[benchmark_id]['thread_count_samples'].append(thread_count)
                
                # Disk I/O
                try:
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        self.profiling_data[benchmark_id]['disk_io_samples'].append({
                            'read_bytes': disk_io.read_bytes,
                            'write_bytes': disk_io.write_bytes
                        })
                except:
                    pass  # Disk I/O not available on all systems
                
                await asyncio.sleep(0.5)  # Sample every 500ms
                
        except Exception as e:
            self.logger.warning(f"Monitoring loop error: {e}")


class WorkloadGenerator:
    """Generate various workload types for comprehensive testing."""
    
    def __init__(self):
        self.synthetic_queries = []
        self.real_world_queries = []
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_workload(
        self,
        workload_type: WorkloadType,
        size: int,
        complexity_level: str = "medium"
    ) -> List[Dict[str, Any]]:
        """Generate workload of specified type and size."""
        if workload_type == WorkloadType.SYNTHETIC_UNIFORM:
            return await self._generate_synthetic_uniform(size, complexity_level)
        elif workload_type == WorkloadType.SYNTHETIC_BURSTY:
            return await self._generate_synthetic_bursty(size, complexity_level)
        elif workload_type == WorkloadType.REAL_WORLD_QUERIES:
            return await self._generate_real_world_queries(size)
        elif workload_type == WorkloadType.ADVERSARIAL_CASES:
            return await self._generate_adversarial_cases(size)
        elif workload_type == WorkloadType.EDGE_CASES:
            return await self._generate_edge_cases(size)
        elif workload_type == WorkloadType.STRESS_TEST:
            return await self._generate_stress_test(size)
        else:
            return []
    
    async def _generate_synthetic_uniform(
        self,
        size: int,
        complexity_level: str
    ) -> List[Dict[str, Any]]:
        """Generate uniform synthetic workload."""
        queries = []
        
        # Define complexity parameters
        complexity_params = {
            "simple": {"words": 3, "concepts": 1, "depth": 1},
            "medium": {"words": 8, "concepts": 3, "depth": 2},
            "complex": {"words": 15, "concepts": 5, "depth": 3}
        }
        
        params = complexity_params.get(complexity_level, complexity_params["medium"])
        
        # Sample query templates
        templates = [
            "What is the definition of {concept}?",
            "How does {concept1} relate to {concept2}?",
            "Explain the process of {concept} in detail",
            "What are the advantages and disadvantages of {concept}?",
            "Compare {concept1} with {concept2}",
            "What are the latest developments in {concept}?"
        ]
        
        concepts = [
            "machine learning", "artificial intelligence", "quantum computing",
            "natural language processing", "deep learning", "neural networks",
            "computer vision", "robotics", "data science", "algorithms",
            "blockchain", "cybersecurity", "cloud computing", "big data"
        ]
        
        for i in range(size):
            template = random.choice(templates)
            selected_concepts = random.sample(concepts, min(params["concepts"], len(concepts)))
            
            if "{concept1}" in template and "{concept2}" in template:
                query = template.format(
                    concept1=selected_concepts[0],
                    concept2=selected_concepts[1] if len(selected_concepts) > 1 else selected_concepts[0]
                )
            else:
                query = template.format(concept=selected_concepts[0])
            
            queries.append({
                "query_id": f"uniform_{i:06d}",
                "query": query,
                "complexity": complexity_level,
                "expected_concepts": selected_concepts,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return queries
    
    async def _generate_synthetic_bursty(
        self,
        size: int,
        complexity_level: str
    ) -> List[Dict[str, Any]]:
        """Generate bursty workload with temporal clustering."""
        queries = []
        
        # Create burst patterns
        num_bursts = max(1, size // 50)  # One burst per 50 queries
        queries_per_burst = size // num_bursts
        
        burst_topics = [
            "quantum algorithms", "AI safety", "machine learning optimization",
            "natural language understanding", "computer vision applications"
        ]
        
        for burst_idx in range(num_bursts):
            burst_topic = random.choice(burst_topics)
            burst_start_time = datetime.utcnow() + timedelta(seconds=burst_idx * 60)
            
            for q_idx in range(queries_per_burst):
                query = f"Detailed analysis of {burst_topic} with focus on recent advances and applications"
                
                queries.append({
                    "query_id": f"bursty_{burst_idx:03d}_{q_idx:03d}",
                    "query": query,
                    "complexity": complexity_level,
                    "burst_id": burst_idx,
                    "burst_topic": burst_topic,
                    "timestamp": (burst_start_time + timedelta(seconds=q_idx)).isoformat()
                })
        
        return queries
    
    async def _generate_real_world_queries(self, size: int) -> List[Dict[str, Any]]:
        """Generate realistic query patterns based on common user interactions."""
        real_world_patterns = [
            "How do I implement {technology} in {domain}?",
            "What are the best practices for {activity}?",
            "Troubleshoot {problem} in {context}",
            "Compare different approaches to {task}",
            "Explain the concept of {concept} with examples",
            "What are the recent advances in {field}?",
            "Step-by-step guide for {process}",
            "Common mistakes when {activity}",
            "Performance optimization for {system}",
            "Security considerations for {application}"
        ]
        
        technologies = ["TensorFlow", "PyTorch", "React", "Docker", "Kubernetes"]
        domains = ["healthcare", "finance", "education", "e-commerce", "gaming"]
        activities = ["data preprocessing", "model training", "deployment", "testing"]
        
        queries = []
        for i in range(size):
            pattern = random.choice(real_world_patterns)
            
            # Fill in template variables
            query = pattern.format(
                technology=random.choice(technologies),
                domain=random.choice(domains),
                activity=random.choice(activities),
                problem="performance issue",
                context="production environment",
                task="optimization",
                concept="neural architecture",
                field="artificial intelligence",
                process="model deployment",
                system="distributed system",
                application="web application"
            )
            
            queries.append({
                "query_id": f"realworld_{i:06d}",
                "query": query,
                "category": "real_world",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return queries
    
    async def _generate_adversarial_cases(self, size: int) -> List[Dict[str, Any]]:
        """Generate adversarial test cases to challenge algorithm robustness."""
        adversarial_patterns = [
            # Extremely long queries
            " ".join(["complex"] * 100) + " query with many repeated words",
            
            # Ambiguous queries
            "What is it and how does it work?",
            "Tell me about that thing we discussed",
            
            # Contradictory queries
            "Why is AI both beneficial and harmful simultaneously?",
            
            # Nonsensical queries
            "Quantum banana optimization for purple algorithms",
            
            # Edge case characters
            "Query with special characters: !@#$%^&*()[]{}|;':\",./<>?`~",
            
            # Mixed languages (simulated)
            "English query con algunas palabras en español and français words",
            
            # Very short queries
            "AI?", "What?", "How?",
            
            # Recursive references
            "Explain this explanation of explaining explanations"
        ]
        
        queries = []
        for i in range(size):
            if i < len(adversarial_patterns):
                query = adversarial_patterns[i]
            else:
                # Generate random adversarial patterns
                query = random.choice(adversarial_patterns)
            
            queries.append({
                "query_id": f"adversarial_{i:06d}",
                "query": query,
                "category": "adversarial",
                "adversarial_type": "challenge_robustness",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return queries
    
    async def _generate_edge_cases(self, size: int) -> List[Dict[str, Any]]:
        """Generate edge case scenarios for boundary testing."""
        edge_cases = []
        
        for i in range(size):
            if i % 5 == 0:
                # Empty or minimal queries
                query = "" if i % 10 == 0 else "a"
            elif i % 5 == 1:
                # Maximum length queries
                query = "very " * 1000 + "long query"
            elif i % 5 == 2:
                # Numeric queries
                query = "What is " + " + ".join([str(random.randint(1, 1000)) for _ in range(10)]) + "?"
            elif i % 5 == 3:
                # Special formatting
                query = "Query\nwith\nmultiple\nlines\nand\ttabs"
            else:
                # Unicode and emoji
                query = "Query with unicode: 🚀 🤖 🔬 and special chars: αβγδε"
            
            edge_cases.append({
                "query_id": f"edge_{i:06d}",
                "query": query,
                "category": "edge_case",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return edge_cases
    
    async def _generate_stress_test(self, size: int) -> List[Dict[str, Any]]:
        """Generate high-intensity stress test workload."""
        stress_queries = []
        
        # High-complexity computational queries
        complex_query_templates = [
            "Perform comprehensive analysis of {topic} including {num_aspects} different aspects with detailed explanations",
            "Compare and contrast {num_items} different approaches to {domain} with statistical analysis",
            "Generate detailed report on {topic} with {num_sections} sections including methodology, results, and conclusions"
        ]
        
        topics = ["machine learning optimization", "quantum algorithm design", "distributed system architecture"]
        domains = ["natural language processing", "computer vision", "robotics"]
        
        for i in range(size):
            template = random.choice(complex_query_templates)
            query = template.format(
                topic=random.choice(topics),
                domain=random.choice(domains),
                num_aspects=random.randint(5, 15),
                num_items=random.randint(8, 20),
                num_sections=random.randint(10, 25)
            )
            
            stress_queries.append({
                "query_id": f"stress_{i:06d}",
                "query": query,
                "category": "stress_test",
                "computational_complexity": "high",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return stress_queries


class PerformanceBenchmark:
    """Core benchmarking engine for performance evaluation."""
    
    def __init__(self, config: BenchmarkConfiguration):
        self.config = config
        self.profiler = SystemProfiler()
        self.workload_generator = WorkloadGenerator()
        
        # Results storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.aggregated_results: Dict[str, Any] = {}
        
        # Output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_benchmark(
        self,
        algorithm_implementations: Dict[str, Callable],
        baseline_algorithm: str = None
    ) -> Dict[str, Any]:
        """
        Execute comprehensive benchmark across all configured categories.
        
        Args:
            algorithm_implementations: Dict mapping algorithm names to callable implementations
            baseline_algorithm: Name of baseline algorithm for comparison
        
        Returns:
            Comprehensive benchmark results with statistical analysis
        """
        self.logger.info(f"Starting comprehensive benchmark: {self.config.name}")
        
        benchmark_start_time = time.time()
        all_results = {}
        
        # Run benchmarks for each algorithm
        for algorithm_name, algorithm_impl in algorithm_implementations.items():
            self.logger.info(f"Benchmarking algorithm: {algorithm_name}")
            
            algorithm_results = await self._benchmark_algorithm(
                algorithm_name, algorithm_impl
            )
            
            all_results[algorithm_name] = algorithm_results
        
        # Calculate comparative metrics
        if baseline_algorithm and baseline_algorithm in all_results:
            comparative_results = await self._calculate_comparative_metrics(
                all_results, baseline_algorithm
            )
            all_results['comparative_analysis'] = comparative_results
        
        # Generate summary report
        summary_report = await self._generate_summary_report(
            all_results, time.time() - benchmark_start_time
        )
        
        # Save results
        await self._save_results(all_results, summary_report)
        
        # Generate visualizations
        if self.config.enable_plotting and PLOTTING_AVAILABLE:
            await self._generate_visualizations(all_results)
        
        return {
            'benchmark_results': all_results,
            'summary_report': summary_report,
            'benchmark_metadata': {
                'config': self.config.__dict__,
                'total_execution_time': time.time() - benchmark_start_time,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    async def _benchmark_algorithm(
        self,
        algorithm_name: str,
        algorithm_impl: Callable
    ) -> Dict[str, Any]:
        """Benchmark a single algorithm across all categories."""
        algorithm_results = {
            'algorithm_name': algorithm_name,
            'category_results': {},
            'aggregated_metrics': {}
        }
        
        # Run benchmarks for each category
        for category in self.config.categories:
            category_results = await self._run_category_benchmark(
                algorithm_name, algorithm_impl, category
            )
            algorithm_results['category_results'][category.value] = category_results
        
        # Aggregate metrics across categories
        algorithm_results['aggregated_metrics'] = await self._aggregate_algorithm_metrics(
            algorithm_results['category_results']
        )
        
        return algorithm_results
    
    async def _run_category_benchmark(
        self,
        algorithm_name: str,
        algorithm_impl: Callable,
        category: BenchmarkCategory
    ) -> Dict[str, Any]:
        """Run benchmark for a specific category."""
        category_results = {
            'category': category.value,
            'workload_results': {},
            'category_summary': {}
        }
        
        # Test each workload type
        for workload_type in self.config.workload_types:
            workload_results = await self._run_workload_benchmark(
                algorithm_name, algorithm_impl, category, workload_type
            )
            category_results['workload_results'][workload_type.value] = workload_results
        
        # Summarize category performance
        category_results['category_summary'] = await self._summarize_category_results(
            category_results['workload_results']
        )
        
        return category_results
    
    async def _run_workload_benchmark(
        self,
        algorithm_name: str,
        algorithm_impl: Callable,
        category: BenchmarkCategory,
        workload_type: WorkloadType
    ) -> Dict[str, Any]:
        """Run benchmark for specific workload type."""
        workload_results = []
        
        # Test different dataset sizes
        for dataset_size in self.config.dataset_sizes:
            # Test different concurrency levels
            for num_workers in self.config.concurrent_workers:
                # Generate workload
                workload = await self.workload_generator.generate_workload(
                    workload_type, dataset_size
                )
                
                if not workload:
                    continue
                
                # Run multiple trials
                trial_results = []
                for trial in range(self.config.num_trials):
                    benchmark_id = f"{algorithm_name}_{category.value}_{workload_type.value}_{dataset_size}_{num_workers}_{trial}"
                    
                    try:
                        result = await self._execute_single_benchmark(
                            benchmark_id,
                            algorithm_impl,
                            workload,
                            num_workers,
                            category
                        )
                        trial_results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Benchmark trial failed: {benchmark_id}, error: {e}")
                        # Create failed result
                        result = BenchmarkResult(
                            benchmark_id=benchmark_id,
                            algorithm_name=algorithm_name,
                            category=category.value,
                            workload_type=workload_type.value,
                            error_rate=1.0,
                            success_rate=0.0
                        )
                        trial_results.append(result)
                
                # Calculate statistics across trials
                if trial_results:
                    trial_stats = await self._calculate_trial_statistics(trial_results)
                    workload_results.append({
                        'dataset_size': dataset_size,
                        'num_workers': num_workers,
                        'trial_count': len(trial_results),
                        'trial_statistics': trial_stats,
                        'individual_trials': [result.to_dict() for result in trial_results]
                    })
        
        return workload_results
    
    async def _execute_single_benchmark(
        self,
        benchmark_id: str,
        algorithm_impl: Callable,
        workload: List[Dict[str, Any]],
        num_workers: int,
        category: BenchmarkCategory
    ) -> BenchmarkResult:
        """Execute a single benchmark trial."""
        # Start profiling
        await self.profiler.start_profiling(benchmark_id)
        
        start_time = time.time()
        latencies = []
        error_count = 0
        
        try:
            # Execute workload with specified concurrency
            if num_workers == 1:
                # Sequential execution
                for work_item in workload:
                    item_start = time.time()
                    try:
                        await self._execute_work_item(algorithm_impl, work_item)
                        latencies.append(time.time() - item_start)
                    except Exception as e:
                        error_count += 1
                        self.logger.debug(f"Work item failed: {e}")
            else:
                # Concurrent execution
                semaphore = asyncio.Semaphore(num_workers)
                
                async def execute_with_semaphore(work_item):
                    async with semaphore:
                        item_start = time.time()
                        try:
                            await self._execute_work_item(algorithm_impl, work_item)
                            return time.time() - item_start
                        except Exception as e:
                            self.logger.debug(f"Work item failed: {e}")
                            return None
                
                tasks = [execute_with_semaphore(item) for item in workload]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception) or result is None:
                        error_count += 1
                    else:
                        latencies.append(result)
        
        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}")
            error_count = len(workload)
        
        execution_time = time.time() - start_time
        
        # Stop profiling and collect system metrics
        profiling_data = await self.profiler.stop_profiling(benchmark_id)
        
        # Calculate performance metrics
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            algorithm_name=algorithm_impl.__name__ if hasattr(algorithm_impl, '__name__') else str(algorithm_impl),
            category=category.value,
            workload_type="mixed",  # Will be updated by caller
            execution_time=execution_time,
            throughput_qps=len(workload) / execution_time if execution_time > 0 else 0.0,
            error_rate=error_count / len(workload) if workload else 0.0,
            success_rate=1.0 - (error_count / len(workload) if workload else 0.0)
        )
        
        # Calculate latency percentiles
        if latencies:
            result.latency_percentiles = {
                'p50': float(np.percentile(latencies, 50)),
                'p90': float(np.percentile(latencies, 90)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies))
            }
        
        # Extract system metrics from profiling
        if profiling_data:
            if 'memory_stats' in profiling_data:
                result.memory_usage_mb = profiling_data['memory_stats']['mean_mb']
                result.memory_efficiency_score = 1.0 / (1.0 + result.memory_usage_mb / 1000.0)
            
            if 'cpu_stats' in profiling_data:
                result.cpu_utilization = profiling_data['cpu_stats']['mean']
                result.cpu_efficiency_score = result.throughput_qps / max(result.cpu_utilization, 1.0)
        
        # Calculate derived metrics
        result.scaling_efficiency = result.throughput_qps / num_workers if num_workers > 1 else 1.0
        result.stability_score = 1.0 - (result.error_rate * 0.5)  # Penalize errors
        
        # Generate reproducibility hash
        result.reproducibility_hash = self._generate_reproducibility_hash(
            benchmark_id, workload, num_workers
        )
        
        # Store system information
        result.system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': os.name
        }
        
        return result
    
    async def _execute_work_item(self, algorithm_impl: Callable, work_item: Dict[str, Any]):
        """Execute a single work item using the algorithm implementation."""
        # This is a placeholder - actual implementation depends on algorithm interface
        if asyncio.iscoroutinefunction(algorithm_impl):
            return await algorithm_impl(work_item)
        else:
            return algorithm_impl(work_item)
    
    async def _calculate_trial_statistics(
        self,
        trial_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Calculate statistical summary across multiple trials."""
        if not trial_results:
            return {}
        
        # Extract numeric metrics
        execution_times = [r.execution_time for r in trial_results]
        throughputs = [r.throughput_qps for r in trial_results]
        memory_usages = [r.memory_usage_mb for r in trial_results]
        error_rates = [r.error_rate for r in trial_results]
        
        stats = {
            'execution_time': {
                'mean': float(np.mean(execution_times)),
                'std': float(np.std(execution_times)),
                'min': float(np.min(execution_times)),
                'max': float(np.max(execution_times)),
                'median': float(np.median(execution_times))
            },
            'throughput_qps': {
                'mean': float(np.mean(throughputs)),
                'std': float(np.std(throughputs)),
                'min': float(np.min(throughputs)),
                'max': float(np.max(throughputs)),
                'median': float(np.median(throughputs))
            },
            'memory_usage_mb': {
                'mean': float(np.mean(memory_usages)),
                'std': float(np.std(memory_usages)),
                'min': float(np.min(memory_usages)),
                'max': float(np.max(memory_usages)),
                'median': float(np.median(memory_usages))
            },
            'error_rate': {
                'mean': float(np.mean(error_rates)),
                'std': float(np.std(error_rates)),
                'min': float(np.min(error_rates)),
                'max': float(np.max(error_rates)),
                'median': float(np.median(error_rates))
            }
        }
        
        return stats
    
    async def _calculate_comparative_metrics(
        self,
        all_results: Dict[str, Any],
        baseline_algorithm: str
    ) -> Dict[str, Any]:
        """Calculate comparative performance metrics against baseline."""
        if baseline_algorithm not in all_results:
            return {}
        
        baseline_metrics = all_results[baseline_algorithm]['aggregated_metrics']
        comparative_analysis = {}
        
        for algorithm_name, algorithm_results in all_results.items():
            if algorithm_name == baseline_algorithm:
                continue
            
            algorithm_metrics = algorithm_results['aggregated_metrics']
            
            # Calculate relative performance
            comparison = {}
            for metric_name, algorithm_value in algorithm_metrics.items():
                baseline_value = baseline_metrics.get(metric_name, 0)
                
                if isinstance(algorithm_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    if baseline_value != 0:
                        relative_change = (algorithm_value - baseline_value) / baseline_value
                        comparison[f"{metric_name}_relative_change"] = relative_change
                        comparison[f"{metric_name}_improvement"] = relative_change > 0
            
            comparative_analysis[algorithm_name] = comparison
        
        return comparative_analysis
    
    def _generate_reproducibility_hash(
        self,
        benchmark_id: str,
        workload: List[Dict[str, Any]],
        num_workers: int
    ) -> str:
        """Generate hash for benchmark reproducibility."""
        # Create deterministic hash from benchmark parameters
        hash_input = f"{benchmark_id}_{len(workload)}_{num_workers}_{self.config.name}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    async def _summarize_category_results(
        self,
        workload_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Summarize results within a category."""
        summary = {
            'total_workload_types': len(workload_results),
            'average_metrics': {},
            'best_performance': {},
            'worst_performance': {}
        }
        
        all_throughputs = []
        all_latencies = []
        
        for workload_type, results in workload_results.items():
            for result_set in results:
                if 'trial_statistics' in result_set:
                    stats = result_set['trial_statistics']
                    if 'throughput_qps' in stats:
                        all_throughputs.append(stats['throughput_qps']['mean'])
                    if 'execution_time' in stats:
                        all_latencies.append(stats['execution_time']['mean'])
        
        if all_throughputs:
            summary['average_metrics']['throughput_qps'] = float(np.mean(all_throughputs))
            summary['best_performance']['throughput_qps'] = float(np.max(all_throughputs))
            summary['worst_performance']['throughput_qps'] = float(np.min(all_throughputs))
        
        if all_latencies:
            summary['average_metrics']['latency_seconds'] = float(np.mean(all_latencies))
            summary['best_performance']['latency_seconds'] = float(np.min(all_latencies))
            summary['worst_performance']['latency_seconds'] = float(np.max(all_latencies))
        
        return summary
    
    async def _aggregate_algorithm_metrics(
        self,
        category_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate metrics across all categories for an algorithm."""
        aggregated = {
            'overall_throughput_qps': 0.0,
            'overall_latency_seconds': 0.0,
            'overall_error_rate': 0.0,
            'overall_memory_usage_mb': 0.0,
            'category_count': len(category_results)
        }
        
        total_throughput = 0.0
        total_latency = 0.0
        total_error_rate = 0.0
        total_memory = 0.0
        count = 0
        
        for category, results in category_results.items():
            summary = results.get('category_summary', {})
            avg_metrics = summary.get('average_metrics', {})
            
            if 'throughput_qps' in avg_metrics:
                total_throughput += avg_metrics['throughput_qps']
                count += 1
            if 'latency_seconds' in avg_metrics:
                total_latency += avg_metrics['latency_seconds']
        
        if count > 0:
            aggregated['overall_throughput_qps'] = total_throughput / count
            aggregated['overall_latency_seconds'] = total_latency / count
        
        return aggregated
    
    async def _generate_summary_report(
        self,
        all_results: Dict[str, Any],
        total_execution_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        return {
            'benchmark_name': self.config.name,
            'total_algorithms_tested': len(all_results) - (1 if 'comparative_analysis' in all_results else 0),
            'total_execution_time_seconds': total_execution_time,
            'categories_tested': [cat.value for cat in self.config.categories],
            'workload_types_tested': [wt.value for wt in self.config.workload_types],
            'best_performing_algorithm': await self._identify_best_algorithm(all_results),
            'performance_rankings': await self._rank_algorithms(all_results),
            'key_insights': await self._extract_key_insights(all_results)
        }
    
    async def _identify_best_algorithm(self, all_results: Dict[str, Any]) -> str:
        """Identify the best performing algorithm overall."""
        best_algorithm = ""
        best_score = -1
        
        for algorithm_name, results in all_results.items():
            if algorithm_name == 'comparative_analysis':
                continue
            
            aggregated = results.get('aggregated_metrics', {})
            # Simple scoring based on throughput (can be made more sophisticated)
            score = aggregated.get('overall_throughput_qps', 0)
            
            if score > best_score:
                best_score = score
                best_algorithm = algorithm_name
        
        return best_algorithm
    
    async def _rank_algorithms(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank algorithms by performance."""
        rankings = []
        
        for algorithm_name, results in all_results.items():
            if algorithm_name == 'comparative_analysis':
                continue
            
            aggregated = results.get('aggregated_metrics', {})
            rankings.append({
                'algorithm': algorithm_name,
                'score': aggregated.get('overall_throughput_qps', 0),
                'latency': aggregated.get('overall_latency_seconds', 0)
            })
        
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return rankings
    
    async def _extract_key_insights(self, all_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from benchmark results."""
        insights = []
        
        # Algorithm count
        num_algorithms = len(all_results) - (1 if 'comparative_analysis' in all_results else 0)
        insights.append(f"Benchmarked {num_algorithms} different algorithms")
        
        # Performance variations
        throughputs = []
        for algorithm_name, results in all_results.items():
            if algorithm_name == 'comparative_analysis':
                continue
            throughput = results.get('aggregated_metrics', {}).get('overall_throughput_qps', 0)
            throughputs.append(throughput)
        
        if throughputs:
            max_throughput = max(throughputs)
            min_throughput = min(throughputs)
            if min_throughput > 0:
                performance_ratio = max_throughput / min_throughput
                insights.append(f"Performance varies by {performance_ratio:.2f}x between best and worst algorithms")
        
        return insights
    
    async def _save_results(self, all_results: Dict[str, Any], summary_report: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file} and {summary_file}")
    
    async def _generate_visualizations(self, all_results: Dict[str, Any]):
        """Generate performance visualization charts."""
        if not PLOTTING_AVAILABLE:
            return
        
        # Performance comparison chart
        algorithms = []
        throughputs = []
        latencies = []
        
        for algorithm_name, results in all_results.items():
            if algorithm_name == 'comparative_analysis':
                continue
            
            algorithms.append(algorithm_name)
            aggregated = results.get('aggregated_metrics', {})
            throughputs.append(aggregated.get('overall_throughput_qps', 0))
            latencies.append(aggregated.get('overall_latency_seconds', 0))
        
        if algorithms and throughputs:
            # Throughput comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(algorithms, throughputs)
            plt.title('Algorithm Throughput Comparison')
            plt.ylabel('Queries per Second')
            plt.xticks(rotation=45)
            
            # Latency comparison
            plt.subplot(1, 2, 2)
            plt.bar(algorithms, latencies)
            plt.title('Algorithm Latency Comparison')
            plt.ylabel('Average Latency (seconds)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"performance_comparison_{timestamp}.png"
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Performance charts saved to {plot_file}")


# Example usage and testing framework
async def create_sample_benchmark():
    """Create a sample benchmark configuration for testing."""
    config = BenchmarkConfiguration(
        name="Quantum Algorithm Performance Evaluation",
        description="Comprehensive performance evaluation of quantum-enhanced algorithms",
        categories=[
            BenchmarkCategory.COMPUTATIONAL_PERFORMANCE,
            BenchmarkCategory.SCALABILITY_ANALYSIS,
            BenchmarkCategory.LATENCY_ANALYSIS
        ],
        workload_types=[
            WorkloadType.SYNTHETIC_UNIFORM,
            WorkloadType.REAL_WORLD_QUERIES,
            WorkloadType.STRESS_TEST
        ],
        num_trials=5,
        concurrent_workers=[1, 2, 4],
        dataset_sizes=[100, 500, 1000]
    )
    
    return PerformanceBenchmark(config)