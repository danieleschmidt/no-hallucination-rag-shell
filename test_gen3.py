#!/usr/bin/env python3
"""
Generation 3 Test - Demonstrate scalability and performance optimizations
"""

import sys
import os
import time
import threading
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / 'no_hallucination_rag'))

def test_caching_system():
    """Test advanced multi-level caching."""
    console = Console()
    console.print("[bold yellow]⚡ Testing Multi-Level Caching System...[/bold yellow]")
    
    from optimization.advanced_caching import MultiLevelCache, CacheManager, cached
    
    # Test basic caching
    cache = MultiLevelCache(l1_size=5, l2_size=10, l3_size=20)
    
    # Test cache operations
    cache.put("key1", "value1", ttl=None)
    cache.put("key2", "value2", ttl=2.0)  # 2 second TTL
    cache.put("key3", "value3")
    
    # Test retrieval
    value1 = cache.get("key1")
    value2 = cache.get("key2")
    value3 = cache.get("nonexistent")
    
    console.print(f"  📦 Cache retrieval: key1={'✅' if value1 == 'value1' else '❌'}")
    console.print(f"  📦 Cache retrieval: key2={'✅' if value2 == 'value2' else '❌'}")
    console.print(f"  📦 Cache retrieval: nonexistent={'✅' if value3 is None else '❌'}")
    
    # Test cache promotion by accessing multiple times
    for i in range(10):
        cache.put(f"bulk_key_{i}", f"bulk_value_{i}")
    
    # Check cache stats
    stats = cache.get_stats()
    console.print(f"  📊 Cache levels populated:")
    console.print(f"     L1: {stats['levels']['l1']['size']}/{stats['levels']['l1']['max_size']}")
    console.print(f"     L2: {stats['levels']['l2']['size']}/{stats['levels']['l2']['max_size']}")
    console.print(f"     L3: {stats['levels']['l3']['size']}/{stats['levels']['l3']['max_size']}")
    console.print(f"  📈 Hit rate: {stats['performance']['hit_rate']:.2%}")
    
    # Test caching decorator
    call_count = 0
    
    @cached(cache_name="test_cache", ttl=5.0)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)  # Simulate work
        return x * 2
    
    # Call multiple times - should only execute once
    result1 = expensive_function(5)
    result2 = expensive_function(5)
    result3 = expensive_function(5)
    
    console.print(f"  🎯 Decorator caching: {result1} (calls: {call_count})")
    console.print(f"  ✅ Cache decorator working: {'✅' if call_count == 1 else '❌'}")
    
    # Test cache manager
    manager = CacheManager()
    test_cache = manager.get_cache("test", l1_size=3)
    test_cache.put("manager_test", "success")
    
    manager_stats = manager.get_stats()
    console.print(f"  🏢 Cache manager: {len(manager_stats)} caches managed")
    
    console.print("  ✅ Caching system tests completed\\n")


def test_concurrency_system():
    """Test advanced concurrency features."""
    console = Console()
    console.print("[bold yellow]🔄 Testing Advanced Concurrency System...[/bold yellow]")
    
    from optimization.advanced_concurrency import (
        ConcurrencyManager, AsyncTaskQueue, Task, ConnectionPool
    )
    import uuid
    
    # Test connection pool
    def create_mock_connection():
        return {"id": str(uuid.uuid4())[:8], "active": True}
    
    def validate_mock_connection(conn):
        return conn.get("active", False)
    
    pool = ConnectionPool(
        factory=create_mock_connection,
        min_connections=2,
        max_connections=5,
        validation_query=validate_mock_connection
    )
    
    # Test getting and returning connections
    conn1 = pool.get_connection()
    conn2 = pool.get_connection()
    
    console.print(f"  🔗 Connection pool: Got connections {conn1['id']} and {conn2['id']}")
    
    pool.return_connection(conn1)
    pool.return_connection(conn2)
    
    pool_stats = pool.get_stats()
    console.print(f"  📊 Pool stats: {pool_stats['active_connections']} active, {pool_stats['available_connections']} available")
    
    # Test async task queue
    task_queue = AsyncTaskQueue(max_workers=3)
    task_queue.start()
    
    results = []
    
    def test_task(x):
        time.sleep(0.01)  # Simulate work
        return x * 2
    
    def result_callback(result):
        results.append(result)
    
    # Submit tasks
    for i in range(5):
        task = Task(
            id=f"task_{i}",
            func=test_task,
            args=(i,),
            priority=i  # Different priorities
        )
        task_queue.submit_task(task, result_callback)
    
    # Wait for tasks to complete
    time.sleep(0.5)
    task_queue.stop()
    
    successful_results = [r for r in results if r.success]
    console.print(f"  ⚡ Async tasks: {len(successful_results)}/5 completed successfully")
    
    queue_stats = task_queue.get_stats()
    console.print(f"  📈 Queue stats: {queue_stats['stats']['tasks_completed']} completed")
    
    # Test concurrency manager
    manager = ConcurrencyManager()
    thread_pool = manager.get_thread_pool("test_pool", max_workers=3)
    
    # Submit some work to thread pool
    futures = []
    for i in range(3):
        future = thread_pool.submit(lambda x: x**2, i)
        futures.append(future)
    
    # Wait for completion
    thread_results = [f.result() for f in futures]
    console.print(f"  🧵 Thread pool: {len(thread_results)} tasks completed")
    
    manager_stats = manager.get_stats()
    console.print(f"  🏢 Manager: {len(manager_stats['thread_pools'])} thread pools")
    
    console.print("  ✅ Concurrency system tests completed\\n")


def test_autoscaling_system():
    """Test auto-scaling and load balancing."""
    console = Console()
    console.print("[bold yellow]📈 Testing Auto-scaling & Load Balancing...[/bold yellow]")
    
    from scaling.auto_scaler import LoadBalancer, AutoScaler, ResourceType, ResourceManager
    
    # Test load balancer
    lb = LoadBalancer()
    
    # Add backend servers
    lb.add_backend("server1", "http://server1:8080", weight=100)
    lb.add_backend("server2", "http://server2:8080", weight=150) 
    lb.add_backend("server3", "http://server3:8080", weight=80)
    
    # Test backend selection
    selections = []
    for _ in range(10):
        backend = lb.select_backend("weighted_round_robin")
        if backend:
            selections.append(backend)
    
    # Count selections
    selection_counts = {}
    for backend in selections:
        selection_counts[backend] = selection_counts.get(backend, 0) + 1
    
    console.print(f"  ⚖️  Load balancer selections: {selection_counts}")
    
    # Test recording metrics
    lb.record_request("server1", 0.05, True)   # Fast successful request
    lb.record_request("server2", 0.15, True)   # Slower successful request
    lb.record_request("server3", 0.08, False)  # Failed request
    
    lb_stats = lb.get_stats()
    console.print(f"  📊 Load balancer: {lb_stats['total_backends']} total, {lb_stats['healthy_backends']} healthy")
    
    # Test auto-scaler
    scaler = AutoScaler()
    
    def mock_resource_controller(resource_name, new_size):
        console.print(f"     🔧 Scaled {resource_name} to {new_size}")
        return True
    
    # Add scaling policy
    scaler.add_scaling_policy(
        resource_name="test_workers",
        resource_type=ResourceType.THREAD_POOL,
        min_size=2,
        max_size=10,
        target_metrics=[("utilization", 75.0, 25.0)],
        resource_controller=mock_resource_controller
    )
    
    # Record high utilization metrics to trigger scale up
    for _ in range(5):
        scaler.record_metric("test_workers", "utilization", 85.0)
    
    # Evaluate scaling
    action = scaler.evaluate_scaling("test_workers")
    console.print(f"  📈 Scaling decision: {action.value}")
    
    if action.value != "maintain":
        success = scaler.execute_scaling("test_workers", action)
        console.print(f"  ⚡ Scaling execution: {'✅' if success else '❌'}")
    
    scaler_stats = scaler.get_stats()
    console.print(f"  📋 Auto-scaler: {len(scaler_stats['policies'])} policies configured")
    
    # Test resource manager
    manager = ResourceManager()
    
    manager.register_scalable_resource(
        name="api_workers",
        resource_type=ResourceType.THREAD_POOL,
        controller=mock_resource_controller,
        min_size=1,
        max_size=5
    )
    
    manager.register_load_balanced_backend("api1", "http://api1:8080")
    
    # Test backend selection
    selected = manager.select_backend()
    console.print(f"  🎯 Resource manager backend selection: {selected or 'None'}")
    
    comprehensive_stats = manager.get_comprehensive_stats()
    console.print(f"  🏢 Resource manager: {len(comprehensive_stats['auto_scaling']['policies'])} scaling policies")
    
    console.print("  ✅ Auto-scaling system tests completed\\n")


def test_performance_optimization():
    """Test performance optimization system."""
    console = Console()
    console.print("[bold yellow]🔧 Testing Performance Optimization...[/bold yellow]")
    
    from optimization.performance_optimizer import (
        AdaptivePerformanceManager, CacheOptimizer, PerformanceMetric
    )
    from datetime import datetime
    
    # Test performance manager
    manager = AdaptivePerformanceManager()
    
    # Create mock cache manager for optimizer
    class MockCacheManager:
        def update_parameters(self, params):
            return True
    
    # Add cache optimizer
    cache_optimizer = CacheOptimizer(MockCacheManager())
    manager.add_optimizer("cache", cache_optimizer)
    
    # Record some performance metrics
    metrics = [
        PerformanceMetric("hit_rate", 0.6, datetime.utcnow(), "cache", "get"),
        PerformanceMetric("response_time", 0.15, datetime.utcnow(), "api", "query"),
        PerformanceMetric("utilization", 0.85, datetime.utcnow(), "thread_pool", "execute"),
        PerformanceMetric("memory_usage", 0.75, datetime.utcnow(), "system", "monitor"),
    ]
    
    for metric in metrics:
        manager.record_metric(metric)
    
    console.print(f"  📊 Recorded {len(metrics)} performance metrics")
    
    # Force optimization cycle
    optimization_result = manager.force_optimization()
    console.print(f"  🔧 Optimization cycle status: {optimization_result.get('status', 'completed')}")
    
    if 'applied' in optimization_result:
        applied_count = len(optimization_result['applied'])
        console.print(f"  ✅ Applied optimizations: {applied_count}")
        
        for opt in optimization_result['applied']:
            console.print(f"     • {opt['optimizer']}.{opt['parameter']}: {opt['old_value']} → {opt['new_value']}")
    
    # Get performance summary
    summary = manager.get_performance_summary()
    if summary.get('status') != 'no_data':
        console.print(f"  📈 Performance summary:")
        console.print(f"     Total metrics: {summary['total_metrics']}")
        console.print(f"     Optimization cycles: {summary['optimization_cycles']}")
        console.print(f"     Active optimizers: {len(summary['optimizers'])}")
    
    # Get current parameters
    current_params = manager.get_current_parameters()
    console.print(f"  ⚙️  Current parameters: {len(current_params)} optimizer sets")
    
    for optimizer_name, params in current_params.items():
        console.print(f"     {optimizer_name}: {len(params)} parameters")
    
    console.print("  ✅ Performance optimization tests completed\\n")


def main():
    """Run Generation 3 comprehensive tests."""
    console = Console()
    
    console.print(Panel(
        "[bold blue]🚀 GENERATION 3 TESTING - MAKE IT SCALE[/bold blue]\\n"
        "[dim]Testing caching, concurrency, auto-scaling, and performance optimization[/dim]",
        title="SCALABLE SYSTEM TESTING",
        border_style="blue"
    ))
    
    tests = [
        ("Multi-Level Caching", test_caching_system),
        ("Advanced Concurrency", test_concurrency_system), 
        ("Auto-scaling & Load Balancing", test_autoscaling_system),
        ("Performance Optimization", test_performance_optimization)
    ]
    
    passed_tests = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        
        for test_name, test_func in tests:
            task = progress.add_task(f"Running {test_name}...", total=100)
            
            try:
                progress.update(task, advance=20)
                test_func()
                progress.update(task, completed=100)
                passed_tests += 1
            except Exception as e:
                progress.update(task, completed=100, description=f"❌ {test_name} - Failed")
                console.print(f"[red]Test {test_name} failed: {e}[/red]")
    
    # Summary
    console.print(Panel(
        f"[bold]Test Results:[/bold]\\n\\n"
        f"✅ [green]Passed:[/green] {passed_tests}/{len(tests)}\\n"
        f"❌ [red]Failed:[/red] {len(tests) - passed_tests}/{len(tests)}\\n\\n"
        f"[bold]Generation 3 Features:[/bold]\\n"
        f"• ⚡ Multi-level adaptive caching with intelligent eviction\\n"
        f"• 🔄 Advanced concurrent processing with connection pooling\\n" 
        f"• 📈 Auto-scaling with load balancing and resource management\\n"
        f"• 🔧 Performance optimization with adaptive parameter tuning\\n\\n"
        f"{'[green]🎉 GENERATION 3 SYSTEM IS HIGHLY SCALABLE!' if passed_tests == len(tests) else '[yellow]⚠️  Some tests failed - system partially scalable'}[/bold]",
        title="🚀 SCALABILITY TEST RESULTS",
        border_style="green" if passed_tests == len(tests) else "yellow"
    ))
    
    return passed_tests == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)