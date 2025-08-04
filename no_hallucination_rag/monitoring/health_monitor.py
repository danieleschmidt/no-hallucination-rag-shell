"""
Comprehensive health monitoring and alerting system.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import json
import statistics


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    description: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: int = 60
    timeout_seconds: int = 30
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    response_time_ms: float
    error: Optional[str] = None


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    component: str
    metric: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(
        self,
        check_interval: int = 30,
        alert_threshold_degraded: float = 0.8,
        alert_threshold_unhealthy: float = 0.6,
        alert_threshold_critical: float = 0.3
    ):
        self.check_interval = check_interval
        self.alert_threshold_degraded = alert_threshold_degraded
        self.alert_threshold_unhealthy = alert_threshold_unhealthy
        self.alert_threshold_critical = alert_threshold_critical
        
        self.logger = logging.getLogger(__name__)
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Results tracking
        self.check_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.recovery_counts: Dict[str, int] = defaultdict(int)
        
        # System metrics tracking
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Alerting
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Monitoring control
        self.monitoring_enabled = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # Overall system health
        self.system_status = HealthStatus.HEALTHY
        self.last_status_change = datetime.utcnow()
        
        # Register default health checks
        self._register_default_health_checks()
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a new health check."""
        with self._lock:
            self.health_checks[health_check.name] = health_check
            self.logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_health_check(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                if name in self.check_results:
                    del self.check_results[name]
                if name in self.failure_counts:
                    del self.failure_counts[name]
                self.logger.info(f"Unregistered health check: {name}")
                return True
            return False
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.monitoring_enabled:
            self.logger.warning("Health monitoring is already running")
            return
        
        self.monitoring_enabled = True
        self._stop_monitoring.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HealthMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        self._stop_monitoring.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._run_health_checks()
                self._update_system_status()
                self._process_metrics()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next check interval
            self._stop_monitoring.wait(self.check_interval)
    
    def _run_health_checks(self) -> None:
        """Run all enabled health checks."""
        for name, health_check in list(self.health_checks.items()):
            if not health_check.enabled:
                continue
            
            try:
                result = self._execute_health_check(health_check)
                self._process_health_check_result(result)
                
            except Exception as e:
                self.logger.error(f"Failed to execute health check {name}: {e}")
                
                # Create failure result
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check execution failed: {e}",
                    details={},
                    timestamp=datetime.utcnow(),
                    response_time_ms=0.0,
                    error=str(e)
                )
                self._process_health_check_result(result)
    
    def _execute_health_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            # Execute check function with timeout
            check_result = health_check.check_function()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine status from result
            status = HealthStatus.HEALTHY
            message = "OK"
            
            if isinstance(check_result, dict):
                status_str = check_result.get("status", "healthy").lower()
                if status_str in ["degraded", "warning"]:
                    status = HealthStatus.DEGRADED
                elif status_str in ["unhealthy", "error"]:
                    status = HealthStatus.UNHEALTHY
                elif status_str in ["critical", "failure"]:
                    status = HealthStatus.CRITICAL
                
                message = check_result.get("message", message)
                details = check_result.get("details", {})
            else:
                details = {"raw_result": check_result}
            
            return HealthCheckResult(
                name=health_check.name,
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=health_check.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"exception": str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    def _process_health_check_result(self, result: HealthCheckResult) -> None:
        """Process health check result and handle alerting."""
        with self._lock:
            # Store result
            self.check_results[result.name].append(result)
            
            # Update failure/recovery counts
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                self.failure_counts[result.name] += 1
                self.recovery_counts[result.name] = 0
                
                # Check if we should alert
                health_check = self.health_checks.get(result.name)
                if health_check and self.failure_counts[result.name] >= health_check.failure_threshold:
                    self._create_health_alert(result)
                    
            else:
                if self.failure_counts[result.name] > 0:
                    self.recovery_counts[result.name] += 1
                    
                    # Check if we should resolve alert
                    health_check = self.health_checks.get(result.name)
                    if health_check and self.recovery_counts[result.name] >= health_check.recovery_threshold:
                        self._resolve_health_alert(result.name)
                        self.failure_counts[result.name] = 0
                        self.recovery_counts[result.name] = 0
    
    def _create_health_alert(self, result: HealthCheckResult) -> None:
        """Create alert for failed health check."""
        alert_id = f"health_check_{result.name}"
        
        # Don't create duplicate alerts
        if alert_id in self.active_alerts:
            return
        
        severity = AlertSeverity.WARNING
        if result.status == HealthStatus.CRITICAL:
            severity = AlertSeverity.CRITICAL
        elif result.status == HealthStatus.UNHEALTHY:
            severity = AlertSeverity.ERROR
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=f"Health Check Failed: {result.name}",
            description=f"Health check '{result.name}' has failed: {result.message}",
            component="HealthMonitor",
            timestamp=result.timestamp,
            metadata={
                "health_check": result.name,
                "failure_count": self.failure_counts[result.name],
                "response_time_ms": result.response_time_ms,
                "details": result.details
            }
        )
        
        self._fire_alert(alert)
    
    def _resolve_health_alert(self, health_check_name: str) -> None:
        """Resolve alert for recovered health check."""
        alert_id = f"health_check_{health_check_name}"
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Resolved health alert: {alert_id}")
            
            # Notify alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
    
    def _update_system_status(self) -> None:
        """Update overall system health status."""
        if not self.check_results:
            return
        
        # Calculate overall health score
        total_checks = 0
        healthy_checks = 0
        
        for name, results in self.check_results.items():
            if not results:
                continue
            
            latest_result = results[-1]
            total_checks += 1
            
            if latest_result.status == HealthStatus.HEALTHY:
                healthy_checks += 1
            elif latest_result.status == HealthStatus.DEGRADED:
                healthy_checks += 0.7  # Partial credit
        
        if total_checks == 0:
            health_score = 1.0
        else:
            health_score = healthy_checks / total_checks
        
        # Determine new status
        new_status = HealthStatus.HEALTHY
        if health_score < self.alert_threshold_critical:
            new_status = HealthStatus.CRITICAL
        elif health_score < self.alert_threshold_unhealthy:
            new_status = HealthStatus.UNHEALTHY
        elif health_score < self.alert_threshold_degraded:
            new_status = HealthStatus.DEGRADED
        
        # Check if status changed
        if new_status != self.system_status:
            old_status = self.system_status
            self.system_status = new_status
            self.last_status_change = datetime.utcnow()
            
            self.logger.warning(f"System health status changed: {old_status.value} -> {new_status.value}")
            
            # Create system-level alert if degraded
            if new_status != HealthStatus.HEALTHY:
                self._create_system_alert(new_status, health_score)
            else:
                self._resolve_system_alert()
    
    def _create_system_alert(self, status: HealthStatus, health_score: float) -> None:
        """Create system-level alert."""
        alert_id = "system_health"
        
        # Don't create duplicate alerts
        if alert_id in self.active_alerts:
            return
        
        severity = AlertSeverity.INFO
        if status == HealthStatus.CRITICAL:
            severity = AlertSeverity.CRITICAL
        elif status == HealthStatus.UNHEALTHY:
            severity = AlertSeverity.ERROR
        elif status == HealthStatus.DEGRADED:
            severity = AlertSeverity.WARNING
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=f"System Health: {status.value.title()}",
            description=f"Overall system health is {status.value} (score: {health_score:.1%})",
            component="System",
            current_value=health_score,
            metadata={
                "health_score": health_score,
                "failed_checks": [
                    name for name, results in self.check_results.items()
                    if results and results[-1].status != HealthStatus.HEALTHY
                ]
            }
        )
        
        self._fire_alert(alert)
    
    def _resolve_system_alert(self) -> None:
        """Resolve system-level alert."""
        alert_id = "system_health"
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info("System health alert resolved")
    
    def _fire_alert(self, alert: Alert) -> None:
        """Fire an alert."""
        with self._lock:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
        
        self.logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.title}")
        
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _process_metrics(self) -> None:
        """Process and analyze system metrics."""
        # This would integrate with the metrics collector
        # For now, just log current status
        pass
    
    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a system metric."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self._lock:
            self.metrics_history[name].append({
                "value": value,
                "timestamp": timestamp
            })
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            # Get latest results for each check
            check_statuses = {}
            for name, results in self.check_results.items():
                if results:
                    latest = results[-1]
                    check_statuses[name] = {
                        "status": latest.status.value,
                        "message": latest.message,
                        "timestamp": latest.timestamp.isoformat(),
                        "response_time_ms": latest.response_time_ms
                    }
            
            return {
                "system_status": self.system_status.value,
                "last_status_change": self.last_status_change.isoformat(),
                "active_alerts": len(self.active_alerts),
                "health_checks": check_statuses,
                "monitoring_enabled": self.monitoring_enabled
            }
    
    def get_alerts(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get system alerts."""
        with self._lock:
            if active_only:
                alerts = list(self.active_alerts.values())
            else:
                alerts = list(self.alert_history)
            
            return [asdict(alert) for alert in alerts]
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks."""
        
        # System resource check
        def check_system_resources() -> Dict[str, Any]:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            status = "healthy"
            issues = []
            
            if cpu_percent > 90:
                status = "critical"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = "unhealthy"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                status = "degraded"
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > 95:
                status = "critical"
                issues.append(f"Critical memory usage: {memory_percent:.1f}%")
            elif memory_percent > 85:
                status = "unhealthy"
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 75:
                status = "degraded"
                issues.append(f"Elevated memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                status = "critical"
                issues.append(f"Critical disk usage: {disk_percent:.1f}%")
            elif disk_percent > 90:
                status = "unhealthy"
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            return {
                "status": status,
                "message": "; ".join(issues) if issues else "System resources normal",
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent
                }
            }
        
        # Application health check
        def check_application_health() -> Dict[str, Any]:
            # This would check application-specific health indicators
            # For now, return healthy
            return {
                "status": "healthy",
                "message": "Application running normally",
                "details": {
                    "uptime_seconds": time.time() - getattr(self, "_start_time", time.time())
                }
            }
        
        # Register checks
        try:
            self.register_health_check(HealthCheck(
                name="system_resources",
                description="Monitor CPU, memory, and disk usage",
                check_function=check_system_resources,
                interval_seconds=30,
                failure_threshold=3
            ))
        except ImportError:
            self.logger.warning("psutil not available, skipping system resource monitoring")
        
        self.register_health_check(HealthCheck(
            name="application_health",
            description="Basic application health check",
            check_function=check_application_health,
            interval_seconds=60,
            failure_threshold=2
        ))
        
        self._start_time = time.time()


# Global health monitor instance
health_monitor = HealthMonitor()


def register_component_health_check(
    component_name: str,
    check_function: Callable[[], Dict[str, Any]],
    interval_seconds: int = 60
) -> None:
    """Register a component health check."""
    health_check = HealthCheck(
        name=f"{component_name}_health",
        description=f"Health check for {component_name}",
        check_function=check_function,
        interval_seconds=interval_seconds
    )
    
    health_monitor.register_health_check(health_check)