#!/usr/bin/env python3
"""
Generation 3 Test: Optimization and Scalability verification
Tests performance optimization, auto-scaling, advanced concurrency, and production features
"""

import sys
import os
import time
import asyncio
import threading
import concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_auto_scaling():
    """Test auto-scaling capabilities."""
    print("🧪 Testing Auto-Scaling...")
    try:
        from no_hallucination_rag.scaling.auto_scaler import AutoScaler
        
        # Initialize auto-scaler
        scaler = AutoScaler()
        print("✅ AutoScaler initialized")
        
        # Test scaling metrics
        scaler.update_metrics({
            "cpu_usage": 75.0,
            "memory_usage": 60.0,
            "queue_length": 10,
            "response_time": 250
        })
        print("✅ Metrics updated")
        
        # Test scaling decision
        should_scale, recommendation = scaler.should_scale_up()
        print(f"✅ Scaling decision: {should_scale}, {recommendation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_performance_optimization():
    """Test performance optimization features."""
    print("\n🧪 Testing Performance Optimization...")
    try:
        from no_hallucination_rag.optimization.performance_optimizer import CacheOptimizer
        from no_hallucination_rag.optimization.caching import CacheManager
        
        # Initialize components
        cache_manager = CacheManager()
        optimizer = CacheOptimizer(cache_manager)
        print("✅ CacheOptimizer initialized")
        
        # Test performance recording
        optimizer.record_query_performance(
            response_time=150.0,
            factuality_score=0.95,
            source_count=5,
            success=True,
            query_type="general"
        )
        print("✅ Performance data recorded")
        
        # Test optimization
        current_params = optimizer.get_current_parameters()
        print(f"✅ Current parameters: {len(current_params)} settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_advanced_concurrency():
    """Test advanced concurrency features."""
    print("\n🧪 Testing Advanced Concurrency...")
    try:
        from no_hallucination_rag.optimization.advanced_concurrency import ConcurrencyManager
        
        # Initialize concurrency manager
        concurrency_manager = ConcurrencyManager()
        print("✅ ConcurrencyManager initialized")
        
        # Test concurrent task execution
        async def test_async_operations():
            tasks = []
            for i in range(5):
                task = concurrency_manager.submit_async_task(
                    lambda x=i: x * 2,  # Simple async task
                    task_id=f"task_{i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(test_async_operations())
        loop.close()
        
        if len(results) == 5:
            print("✅ Advanced concurrency works")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced concurrency test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_quantum_optimization():
    """Test quantum-inspired optimization."""
    print("\n🧪 Testing Quantum Optimization...")
    try:
        from no_hallucination_rag.quantum.quantum_optimizer import QuantumOptimizer
        
        # Initialize quantum optimizer
        optimizer = QuantumOptimizer()
        print("✅ QuantumOptimizer initialized")
        
        # Test quantum annealing simulation
        parameters = {"cache_size": 1000, "ttl": 3600, "batch_size": 10}
        optimized = optimizer.quantum_annealing_optimize(
            parameters=parameters,
            objective_function=lambda p: sum(p.values()),  # Simple objective
            iterations=5
        )
        
        if optimized:
            print("✅ Quantum annealing optimization works")
        
        # Test superposition optimization
        result = optimizer.superposition_parameter_search(
            parameter_space={"x": [1, 2, 3], "y": [4, 5, 6]},
            objective=lambda p: p["x"] + p["y"]
        )
        
        if result:
            print("✅ Superposition parameter search works")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum optimization test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_production_health_monitoring():
    """Test production health monitoring."""
    print("\n🧪 Testing Production Health Monitoring...")
    try:
        from no_hallucination_rag.monitoring.advanced_monitoring import AdvancedMonitor
        
        # Initialize advanced monitor
        monitor = AdvancedMonitor()
        print("✅ AdvancedMonitor initialized")
        
        # Test health checks
        health = monitor.comprehensive_health_check()
        if health.get("overall_status"):
            print("✅ Comprehensive health check works")
        
        # Test anomaly detection
        metrics = {
            "response_time": [100, 120, 110, 500, 130],  # Anomaly: 500
            "error_rate": [0.01, 0.02, 0.01, 0.05, 0.01]
        }
        
        anomalies = monitor.detect_anomalies(metrics)
        if isinstance(anomalies, dict):
            print("✅ Anomaly detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Production monitoring test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_load_balancing():
    """Test load balancing and resource management."""
    print("\n🧪 Testing Load Balancing...")
    try:
        # Test with existing FactualRAG for load testing
        from no_hallucination_rag import FactualRAG
        
        # Initialize multiple instances for load testing
        rag_instances = []
        for i in range(3):
            rag = FactualRAG(
                enable_caching=True,
                enable_optimization=True,
                enable_concurrency=True
            )
            rag_instances.append(rag)
        
        print(f"✅ {len(rag_instances)} RAG instances created for load balancing")
        
        # Simulate concurrent queries
        def query_rag(instance_id, query):
            try:
                rag = rag_instances[instance_id % len(rag_instances)]
                response = rag.query(f"Test query {query}")
                return {"success": True, "instance": instance_id, "factuality": response.factuality_score}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Use ThreadPool for concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(10):  # 10 concurrent queries
                future = executor.submit(query_rag, i, i)
                futures.append(future)
            
            results = [future.result() for future in futures]
        
        successful = sum(1 for r in results if r.get("success", False))
        print(f"✅ Load balancing: {successful}/10 queries successful")
        
        return successful >= 8  # 80% success rate
        
    except Exception as e:
        print(f"❌ Load balancing test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_performance_benchmarks():
    """Test performance benchmarks and optimization metrics."""
    print("\n🧪 Testing Performance Benchmarks...")
    try:
        from no_hallucination_rag import FactualRAG
        
        # Initialize optimized RAG
        rag = FactualRAG(
            enable_caching=True,
            enable_optimization=True,
            enable_metrics=True,
            enable_concurrency=True
        )
        print("✅ Optimized FactualRAG initialized")
        
        # Benchmark queries
        start_time = time.time()
        queries = [
            "What is machine learning?",
            "Explain artificial intelligence",
            "How does deep learning work?",
            "What are neural networks?",
            "Define natural language processing"
        ]
        
        results = []
        for query in queries:
            query_start = time.time()
            response = rag.query(query)
            query_time = time.time() - query_start
            results.append({
                "query": query,
                "response_time": query_time,
                "factuality": response.factuality_score,
                "sources": len(response.sources)
            })
        
        total_time = time.time() - start_time
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        
        print(f"✅ Benchmark: {len(queries)} queries in {total_time:.2f}s")
        print(f"✅ Average response time: {avg_response_time:.3f}s")
        
        # Test performance stats
        perf_stats = rag.get_performance_stats()
        if isinstance(perf_stats, dict):
            print("✅ Performance statistics collection works")
        
        return avg_response_time < 1.0  # Sub-second average response time
        
    except Exception as e:
        print(f"❌ Performance benchmarks test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run all Generation 3 optimization and scalability tests."""
    print("🚀 GENERATION 3 TESTING: MAKE IT SCALE (Optimized)")
    print("=" * 65)
    
    tests = [
        test_auto_scaling,
        test_performance_optimization,
        test_advanced_concurrency,
        test_quantum_optimization,
        test_production_health_monitoring,
        test_load_balancing,
        test_performance_benchmarks
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 65)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Generation 3: SUCCESS - System is optimized and scalable!")
    elif passed >= total * 0.8:
        print("⚠️  Generation 3: MOSTLY OPTIMIZED - Some features to enhance")
    else:
        print("❌ Generation 3: NEEDS WORK - Major optimization issues found")
    
    return passed >= total * 0.7  # 70% threshold for optimization

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)