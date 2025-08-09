
"""
Comprehensive monitoring, health checks, and observability system.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import json
import socket


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], bool]
    description: str
    critical: bool = True
    timeout: int = 5
    interval: int = 60


@dataclass
class HealthStatus:
    """Health check status."""
    name: str
    status: str  # healthy, unhealthy, unknown
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, check_interval: int = 30):
        self.checks: Dict[str, HealthCheck] = {}
        self.statuses: Dict[str, HealthStatus] = {}
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
        # Setup default health checks
        self._setup_default_checks()
    
    def add_health_check(self, check: HealthCheck):
        """Add a health check."""
        self.checks[check.name] = check
        self.logger.info(f"Added health check: {check.name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check."""
        self.checks.pop(name, None)
        self.statuses.pop(name, None)
        self.logger.info(f"Removed health check: {name}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def run_check(self, check_name: str) -> HealthStatus:
        """Run a specific health check."""
        check = self.checks.get(check_name)
        if not check:
            return HealthStatus(
                name=check_name,
                status="unknown",
                last_check=datetime.utcnow(),
                response_time=0,
                error_message="Check not found"
            )
        
        start_time = time.time()
        try:
            # Run check with timeout
            result = self._run_with_timeout(check.check_function, check.timeout)
            response_time = time.time() - start_time
            
            status = HealthStatus(
                name=check_name,
                status="healthy" if result else "unhealthy",
                last_check=datetime.utcnow(),
                response_time=response_time,
                details={"critical": check.critical}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            status = HealthStatus(
                name=check_name,
                status="unhealthy",
                last_check=datetime.utcnow(),
                response_time=response_time,
                error_message=str(e),
                details={"critical": check.critical}
            )
        
        self.statuses[check_name] = status
        return status
    
    def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        for check_name in self.checks:
            results[check_name] = self.run_check(check_name)
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.statuses:
            self.run_all_checks()
        
        healthy_count = sum(1 for s in self.statuses.values() if s.status == "healthy")
        critical_failures = [s for s in self.statuses.values() 
                           if s.status == "unhealthy" and s.details.get("critical", False)]
        
        overall_status = "healthy"
        if critical_failures:
            overall_status = "critical"
        elif healthy_count < len(self.statuses):
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": len(self.statuses),
            "healthy_checks": healthy_count,
            "critical_failures": len(critical_failures),
            "details": {name: {
                "status": status.status,
                "last_check": status.last_check.isoformat(),
                "response_time": status.response_time,
                "error": status.error_message
            } for name, status in self.statuses.items()}
        }
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.run_all_checks()
                
                # Log critical failures
                critical_failures = [s for s in self.statuses.values() 
                                   if s.status == "unhealthy" and s.details.get("critical", False)]
                
                for failure in critical_failures:
                    self.logger.critical(f"Critical health check failed: {failure.name} - {failure.error_message}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(min(self.check_interval, 30))  # Back off on errors
    
    def _run_with_timeout(self, func: Callable, timeout: int):
        """Run function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check timed out after {timeout} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            return func()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _setup_default_checks(self):
        """Setup default system health checks."""
        
        def check_memory():
            """Check memory usage."""
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Less than 90% memory usage
        
        def check_disk():
            """Check disk usage."""
            disk = psutil.disk_usage('/')
            return disk.percent < 95  # Less than 95% disk usage
        
        def check_cpu():
            """Check CPU usage."""
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 95  # Less than 95% CPU usage
        
        def check_file_descriptors():
            """Check file descriptor usage."""
            try:
                process = psutil.Process()
                num_fds = process.num_fds()
                # Check against typical limit (usually 1024 or 4096)
                return num_fds < 800
            except Exception:
                return True  # If can't check, assume healthy
        
        # Add default checks
        self.add_health_check(HealthCheck(
            name="memory",
            check_function=check_memory,
            description="System memory usage",
            critical=True
        ))
        
        self.add_health_check(HealthCheck(
            name="disk",
            check_function=check_disk,
            description="System disk usage",
            critical=True
        ))
        
        self.add_health_check(HealthCheck(
            name="cpu",
            check_function=check_cpu,
            description="System CPU usage",
            critical=False
        ))
        
        self.add_health_check(HealthCheck(
            name="file_descriptors",
            check_function=check_file_descriptors,
            description="Process file descriptors",
            critical=False
        ))


class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, retention_days: int = 7):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_days = retention_days
        self.collectors: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric data point."""
        with self.lock:
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels or {}
            )
            self.metrics[name].append(point)
    
    def record_counter(self, name: str, increment: float = 1, labels: Dict[str, str] = None):
        """Record counter increment."""
        self.record_metric(f"{name}_total", increment, labels)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram value."""
        # Record the value
        self.record_metric(f"{name}", value, labels)
        
        # Calculate and record quantiles periodically
        with self.lock:
            recent_values = [p.value for p in self.metrics[name] 
                           if (datetime.utcnow() - p.timestamp).seconds < 300]  # Last 5 minutes
            
            if len(recent_values) >= 10:
                recent_values.sort()
                n = len(recent_values)
                
                # Record quantiles
                quantiles = [50, 90, 95, 99]
                for q in quantiles:
                    idx = int(n * q / 100)
                    if idx < n:
                        self.record_metric(f"{name}_p{q}", recent_values[idx], labels)
    
    def add_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Add custom metric collector."""
        self.collectors[name] = collector_func
    
    def collect_system_metrics(self):
        """Collect system metrics."""
        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric("system_memory_usage_percent", memory.percent)
        self.record_metric("system_memory_available_bytes", memory.available)
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.record_metric("system_cpu_usage_percent", cpu_percent)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.record_metric("system_disk_usage_percent", disk.percent)
        self.record_metric("system_disk_free_bytes", disk.free)
        
        # Network metrics
        try:
            network = psutil.net_io_counters()
            self.record_metric("system_network_bytes_sent", network.bytes_sent)
            self.record_metric("system_network_bytes_recv", network.bytes_recv)
        except Exception as e:
            self.logger.warning(f"Failed to collect network metrics: {e}")
    
    def get_metric_summary(self, name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get metric summary for specified duration."""
        with self.lock:
            if name not in self.metrics:
                return {"error": "Metric not found"}
            
            cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
            recent_points = [p for p in self.metrics[name] if p.timestamp > cutoff_time]
            
            if not recent_points:
                return {"error": "No recent data"}
            
            values = [p.value for p in recent_points]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else None,
                "timestamp_range": {
                    "start": recent_points[0].timestamp.isoformat(),
                    "end": recent_points[-1].timestamp.isoformat()
                }
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self.lock:
            return {
                name: self.get_metric_summary(name, 60)
                for name in self.metrics.keys()
            }
    
    def _cleanup_loop(self):
        """Cleanup old metrics periodically."""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
                
                with self.lock:
                    for name, points in self.metrics.items():
                        # Remove old points
                        while points and points[0].timestamp < cutoff_time:
                            points.popleft()
                
                time.sleep(3600)  # Run cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")
                time.sleep(300)  # Wait 5 minutes on error


class SystemMonitor:
    """Integrated system monitoring with health checks and metrics."""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.utcnow()
    
    def start(self):
        """Start all monitoring services."""
        self.health_monitor.start_monitoring()
        
        # Setup metrics collection
        self.metrics_collector.add_collector("system", self._collect_system_metrics)
        
        # Start periodic metrics collection
        metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        metrics_thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop(self):
        """Stop all monitoring services."""
        self.health_monitor.stop_monitoring()
        self.logger.info("System monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health = self.health_monitor.get_overall_health()
        uptime = datetime.utcnow() - self.start_time
        
        return {
            "status": health["status"],
            "uptime_seconds": uptime.total_seconds(),
            "health": health,
            "metrics_summary": self.metrics_collector.get_all_metrics(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics."""
        try:
            self.metrics_collector.collect_system_metrics()
            return {"status": 1.0}  # Success
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            return {"status": 0.0}  # Failure
    
    def _metrics_collection_loop(self):
        """Periodic metrics collection loop."""
        while True:
            try:
                # Collect from all registered collectors
                for name, collector in self.metrics_collector.collectors.items():
                    try:
                        metrics = collector()
                        for metric_name, value in metrics.items():
                            self.metrics_collector.record_metric(f"{name}_{metric_name}", value)
                    except Exception as e:
                        self.logger.error(f"Collector {name} failed: {e}")
                
                time.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection loop error: {e}")
                time.sleep(60)  # Back off on errors
