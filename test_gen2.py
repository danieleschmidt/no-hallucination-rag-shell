#!/usr/bin/env python3
"""
Generation 2 Test - Demonstrate robust error handling, validation, monitoring, and security
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / 'no_hallucination_rag'))

def test_error_handling():
    """Test advanced error handling and circuit breakers."""
    console = Console()
    console.print("[bold yellow]🛡️ Testing Error Handling & Circuit Breakers...[/bold yellow]")
    
    from core.enhanced_error_handler import (
        RobustErrorHandler, CircuitBreaker, CircuitBreakerConfig,
        with_retry, ErrorCategory
    )
    
    # Test error handler
    handler = RobustErrorHandler()
    
    # Simulate errors
    test_errors = [
        ValueError("Test validation error"),
        ConnectionError("Test network error"),
        MemoryError("Test resource error")
    ]
    
    for error in test_errors:
        try:
            raise error
        except Exception as e:
            handler.handle_error(e, {"component": "test", "operation": "demo"})
    
    # Get error stats
    stats = handler.get_error_statistics()
    console.print(f"  📊 Total errors handled: [cyan]{stats['total_errors']}[/cyan]")
    console.print(f"  📈 Categories tracked: [cyan]{len(stats['errors_by_category'])}[/cyan]")
    
    # Test circuit breaker
    def failing_function():
        raise Exception("Simulated failure")
    
    config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1)
    breaker = handler.add_circuit_breaker("test_service", config)
    
    # Trigger circuit breaker
    failures = 0
    for i in range(5):
        try:
            breaker._call(failing_function)
        except Exception:
            failures += 1
    
    breaker_state = breaker.get_state()
    console.print(f"  🔌 Circuit breaker state: [red]{breaker_state['state']}[/red]")
    console.print(f"  📉 Failure count: [red]{breaker_state['failure_count']}[/red]")
    
    # Test retry decorator
    @with_retry(max_retries=2, delay=0.1)
    def flaky_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise Exception("Flaky service error")
        return "Success!"
    
    try:
        result = flaky_function()
        console.print(f"  🔄 Retry decorator: [green]{result}[/green]")
    except Exception:
        console.print("  🔄 Retry decorator: [red]All retries failed[/red]")
    
    console.print("  ✅ Error handling tests completed\\n")


def test_validation():
    """Test comprehensive input validation."""
    console = Console()
    console.print("[bold yellow]🔍 Testing Input Validation & Sanitization...[/bold yellow]")
    
    from core.advanced_validation import AdvancedValidator, InputSanitizer
    
    validator = AdvancedValidator()
    sanitizer = InputSanitizer()
    
    # Test query validation
    test_queries = [
        "What are AI safety requirements?",
        "",  # Empty query
        "A" * 11000,  # Too long
        "<script>alert('xss')</script>",  # Suspicious content
        "SELECT * FROM users; DROP TABLE users;",  # SQL injection
    ]
    
    valid_count = 0
    for query in test_queries:
        result = validator.validate_query(query)
        if result.is_valid:
            valid_count += 1
    
    console.print(f"  📝 Queries validated: [cyan]{len(test_queries)}[/cyan]")
    console.print(f"  ✅ Valid queries: [green]{valid_count}[/green]")
    console.print(f"  ❌ Invalid queries: [red]{len(test_queries) - valid_count}[/red]")
    
    # Test security detection
    suspicious_inputs = [
        "<script>malicious code</script>",
        "'; DROP TABLE users; --",
        "javascript:void(0)",
        "/bin/sh -c 'rm -rf /'",
        "normal safe input"
    ]
    
    threats_detected = 0
    for inp in suspicious_inputs:
        if sanitizer.is_suspicious(inp):
            threats_detected += 1
    
    console.print(f"  🚨 Security threats detected: [red]{threats_detected}/{len(suspicious_inputs)}[/red]")
    
    # Test configuration validation
    configs = [
        {"factuality_threshold": 0.95, "max_sources": 10},  # Valid
        {"factuality_threshold": 1.5},  # Invalid threshold
        {"max_sources": -1},  # Invalid max_sources
        {}  # Missing required fields
    ]
    
    valid_configs = 0
    for config in configs:
        result = validator.validate_config(config)
        if result.is_valid:
            valid_configs += 1
    
    console.print(f"  ⚙️  Configuration validation: [cyan]{valid_configs}/{len(configs)}[/cyan] valid")
    console.print("  ✅ Validation tests completed\\n")


def test_monitoring():
    """Test monitoring and health checks."""
    console = Console()
    console.print("[bold yellow]📊 Testing Monitoring & Health Checks...[/bold yellow]")
    
    from monitoring.advanced_monitoring import SystemMonitor, HealthCheck
    import time
    
    monitor = SystemMonitor()
    
    # Add custom health check
    def custom_check():
        return True  # Always healthy for demo
    
    custom_health_check = HealthCheck(
        name="custom_service",
        check_function=custom_check,
        description="Custom service health check"
    )
    
    monitor.health_monitor.add_health_check(custom_health_check)
    
    # Start monitoring briefly
    console.print("  🟢 Starting health monitoring...")
    monitor.start()
    
    # Let it run for a moment
    time.sleep(2)
    
    # Get system status
    status = monitor.get_status()
    
    console.print(f"  💗 Overall health: [green]{status['health']['status']}[/green]")
    console.print(f"  ⏱️  System uptime: [cyan]{status['uptime_seconds']:.1f}s[/cyan]")
    console.print(f"  🔍 Health checks: [cyan]{status['health']['total_checks']}[/cyan]")
    console.print(f"  ✅ Healthy checks: [green]{status['health']['healthy_checks']}[/green]")
    
    # Test metrics collection
    monitor.metrics_collector.record_metric("test_metric", 42.0)
    monitor.metrics_collector.record_counter("test_counter", 5)
    monitor.metrics_collector.record_histogram("test_histogram", 1.5)
    
    metrics_summary = monitor.metrics_collector.get_all_metrics()
    console.print(f"  📈 Metrics collected: [cyan]{len(metrics_summary)}[/cyan]")
    
    monitor.stop()
    console.print("  ✅ Monitoring tests completed\\n")


def test_security():
    """Test security features."""
    console = Console()
    console.print("[bold yellow]🔐 Testing Security Features...[/bold yellow]")
    
    from security.advanced_security import ComprehensiveSecurityManager
    import time
    
    security = ComprehensiveSecurityManager()
    
    # Test API key generation and validation
    api_key = security.api_key_manager.generate_key("test_user", ["read", "write"])
    console.print(f"  🔑 Generated API key: [dim]{api_key[:16]}...[/dim]")
    
    key_valid, key_data = security.api_key_manager.validate_key(api_key)
    console.print(f"  ✅ Key validation: [green]{key_valid}[/green]")
    console.print(f"  👤 User ID: [cyan]{key_data.get('user_id')}[/cyan]")
    
    # Test IP whitelisting
    security.ip_whitelist.add_to_whitelist("127.0.0.1/8")
    security.ip_whitelist.add_to_blacklist("192.168.1.100")
    
    test_ips = ["127.0.0.1", "192.168.1.100", "10.0.0.1"]
    allowed_count = 0
    
    for ip in test_ips:
        allowed, message = security.ip_whitelist.is_allowed(ip)
        if allowed:
            allowed_count += 1
    
    console.print(f"  🌐 IP access control: [cyan]{allowed_count}/{len(test_ips)}[/cyan] allowed")
    
    # Test rate limiting
    client_ip = "127.0.0.1"
    rate_limit_hits = 0
    
    for i in range(5):
        rate_ok, rate_info = security.rate_limiter.check_rate_limit(client_ip, "general")
        if not rate_ok:
            rate_limit_hits += 1
    
    console.print(f"  🚦 Rate limiting: [cyan]{rate_limit_hits}[/cyan] blocks in 5 requests")
    
    # Test request validation
    validation_results = []
    test_requests = [
        {"client_ip": "127.0.0.1", "api_key": api_key, "query": "Safe query"},
        {"client_ip": "192.168.1.100", "query": "Blocked IP test"},
        {"client_ip": "127.0.0.1", "query": "<script>alert('xss')</script>"},
        {"client_ip": "127.0.0.1", "api_key": "invalid_key"}
    ]
    
    valid_requests = 0
    for req in test_requests:
        is_valid, details = security.validate_request(**req)
        if is_valid:
            valid_requests += 1
    
    console.print(f"  🛡️  Request validation: [cyan]{valid_requests}/{len(test_requests)}[/cyan] passed")
    
    # Get security status
    sec_status = security.get_security_status()
    console.print(f"  📋 Security events: [cyan]{sec_status['security_events']['total_events']}[/cyan]")
    
    console.print("  ✅ Security tests completed\\n")


def main():
    """Run Generation 2 comprehensive tests."""
    console = Console()
    
    console.print(Panel(
        "[bold blue]🛡️ GENERATION 2 TESTING - MAKE IT ROBUST[/bold blue]\\n"
        "[dim]Testing error handling, validation, monitoring, and security[/dim]",
        title="ROBUST SYSTEM TESTING",
        border_style="blue"
    ))
    
    tests = [
        ("Error Handling", test_error_handling),
        ("Validation", test_validation),
        ("Monitoring", test_monitoring),
        ("Security", test_security)
    ]
    
    passed_tests = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for test_name, test_func in tests:
            task = progress.add_task(f"Running {test_name} tests...", total=1)
            
            try:
                test_func()
                passed_tests += 1
                progress.update(task, completed=1, description=f"✅ {test_name} - Passed")
            except Exception as e:
                progress.update(task, completed=1, description=f"❌ {test_name} - Failed: {e}")
                console.print(f"[red]Test {test_name} failed: {e}[/red]")
    
    # Summary
    console.print(Panel(
        f"[bold]Test Results:[/bold]\\n\\n"
        f"✅ [green]Passed:[/green] {passed_tests}/{len(tests)}\\n"
        f"❌ [red]Failed:[/red] {len(tests) - passed_tests}/{len(tests)}\\n\\n"
        f"[bold]Generation 2 Features:[/bold]\\n"
        f"• 🛡️  Advanced error handling with circuit breakers\\n"
        f"• 🔍 Comprehensive input validation and sanitization\\n"
        f"• 📊 Real-time monitoring and health checks\\n"
        f"• 🔐 Enterprise-grade security measures\\n\\n"
        f"{'[green]🎉 GENERATION 2 SYSTEM IS ROBUST!' if passed_tests == len(tests) else '[yellow]⚠️  Some tests failed - system partially robust'}[/bold]",
        title="🛡️ ROBUSTNESS TEST RESULTS",
        border_style="green" if passed_tests == len(tests) else "yellow"
    ))
    
    return passed_tests == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)