"""
Metrics collection and monitoring for RAG system.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import json
import statistics


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(
        self,
        export_to: Optional[str] = None,
        export_interval: int = 60,
        retention_hours: int = 24
    ):
        self.export_to = export_to
        self.export_interval = export_interval
        self.retention_hours = retention_hours
        self.logger = logging.getLogger(__name__)
        
        # Metric storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Metric metadata
        self.metric_labels: Dict[str, Dict[str, str]] = {}
        self.metric_descriptions: Dict[str, str] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Export timer
        self._setup_export_timer()
        
        # Initialize standard metrics
        self._initialize_standard_metrics()
    
    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric."""
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self.counters[metric_key] += value
            self._update_labels(name, labels)
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric value."""
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self.gauges[metric_key] = value
            self._update_labels(name, labels)
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Add value to histogram metric."""
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self.histograms[metric_key].append(MetricValue(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            ))
            self._update_labels(name, labels)
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels)
    
    def track_query_metrics(
        self,
        query: str,
        response_time: float,
        factuality_score: float,
        source_count: int,
        success: bool,
        error_type: Optional[str] = None
    ) -> None:
        """Track comprehensive query metrics."""
        labels = {
            "success": str(success).lower(),
            "error_type": error_type or "none"
        }
        
        # Core metrics
        self.counter("queries_total", 1.0, labels)
        self.histogram("query_response_time_seconds", response_time, labels)
        self.histogram("query_factuality_score", factuality_score, labels)
        self.histogram("query_source_count", source_count, labels)
        
        # Query characteristics
        query_length = len(query)
        query_word_count = len(query.split())
        
        self.histogram("query_length_chars", query_length)
        self.histogram("query_word_count", query_word_count)
        
        # Success/failure tracking
        if success:
            self.counter("successful_queries_total")
            self.histogram("successful_query_factuality", factuality_score)
        else:
            self.counter("failed_queries_total")
            if error_type:
                self.counter("query_errors_by_type", 1.0, {"error_type": error_type})
    
    def track_component_metrics(
        self,
        component: str,
        operation: str,
        duration: float,
        success: bool,
        error_type: Optional[str] = None
    ) -> None:
        """Track component-specific metrics."""
        labels = {
            "component": component,
            "operation": operation,
            "success": str(success).lower()
        }
        
        if error_type:
            labels["error_type"] = error_type
        
        self.counter("component_operations_total", 1.0, labels)
        self.histogram("component_operation_duration_seconds", duration, labels)
        
        if not success:
            self.counter("component_errors_total", 1.0, labels)
    
    def track_retrieval_metrics(
        self,
        query: str,
        sources_found: int,
        retrieval_time: float,
        top_k: int,
        retrieval_method: str
    ) -> None:
        """Track retrieval-specific metrics."""
        labels = {"method": retrieval_method}
        
        self.histogram("retrieval_time_seconds", retrieval_time, labels)
        self.histogram("sources_retrieved", sources_found, labels)
        self.histogram("retrieval_top_k", top_k, labels)
        
        # Calculate retrieval efficiency
        efficiency = sources_found / max(top_k, 1)
        self.histogram("retrieval_efficiency", efficiency, labels)
    
    def track_factuality_metrics(
        self,
        factuality_score: float,
        verification_time: float,
        claim_count: int,
        verified_claims: int
    ) -> None:
        """Track factuality verification metrics."""
        self.histogram("factuality_verification_time_seconds", verification_time)
        self.histogram("factuality_score_distribution", factuality_score)
        self.histogram("claims_per_response", claim_count)
        
        if claim_count > 0:
            verification_rate = verified_claims / claim_count
            self.histogram("claim_verification_rate", verification_rate)
    
    def track_governance_metrics(
        self,
        is_compliant: bool,
        check_time: float,
        violation_count: int,
        policies_checked: List[str]
    ) -> None:
        """Track governance compliance metrics."""
        compliance_labels = {"compliant": str(is_compliant).lower()}
        
        self.counter("governance_checks_total", 1.0, compliance_labels)
        self.histogram("governance_check_time_seconds", check_time)
        self.histogram("governance_violations_per_check", violation_count)
        
        # Track by policy
        for policy in policies_checked:
            policy_labels = {"policy": policy, "compliant": str(is_compliant).lower()}
            self.counter("governance_checks_by_policy", 1.0, policy_labels)
    
    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        with self._lock:
            # Look for histogram data
            histogram_keys = [k for k in self.histograms.keys() if k.startswith(name)]
            if histogram_keys:
                all_values = []
                for key in histogram_keys:
                    values = [mv.value for mv in self.histograms[key]]
                    all_values.extend(values)
                
                if all_values:
                    sorted_values = sorted(all_values)
                    return MetricSummary(
                        count=len(all_values),
                        sum=sum(all_values),
                        min=min(all_values),
                        max=max(all_values),
                        avg=statistics.mean(all_values),
                        p50=self._percentile(sorted_values, 50),
                        p95=self._percentile(sorted_values, 95),
                        p99=self._percentile(sorted_values, 99)
                    )
            
            # Look for counter data
            counter_keys = [k for k in self.counters.keys() if k.startswith(name)]
            if counter_keys:
                total_value = sum(self.counters[k] for k in counter_keys)
                return MetricSummary(
                    count=len(counter_keys),
                    sum=total_value,
                    min=min(self.counters[k] for k in counter_keys),
                    max=max(self.counters[k] for k in counter_keys),
                    avg=total_value / len(counter_keys)
                )
            
            return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    k: [{"value": mv.value, "timestamp": mv.timestamp.isoformat()} 
                        for mv in v] 
                    for k, v in self.histograms.items()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get key health metrics for monitoring."""
        query_summary = self.get_metric_summary("queries_total")
        error_summary = self.get_metric_summary("failed_queries_total") 
        factuality_summary = self.get_metric_summary("query_factuality_score")
        response_time_summary = self.get_metric_summary("query_response_time_seconds")
        
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {}
        }
        
        if query_summary:
            health["metrics"]["total_queries"] = query_summary.sum
            
            if error_summary:
                error_rate = error_summary.sum / max(query_summary.sum, 1)
                health["metrics"]["error_rate"] = error_rate
                
                # Alert if error rate too high
                if error_rate > 0.1:  # 10% error rate
                    health["status"] = "degraded"
        
        if factuality_summary:
            health["metrics"]["avg_factuality_score"] = factuality_summary.avg
            
            # Alert if factuality too low
            if factuality_summary.avg < 0.8:
                health["status"] = "degraded"
        
        if response_time_summary:
            health["metrics"]["avg_response_time"] = response_time_summary.avg
            health["metrics"]["p95_response_time"] = response_time_summary.p95
            
            # Alert if response time too high
            if response_time_summary.p95 > 10.0:  # 10 seconds
                health["status"] = "degraded"
        
        return health
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate metric key with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _update_labels(self, name: str, labels: Optional[Dict[str, str]]) -> None:
        """Update metric labels."""
        if labels:
            if name not in self.metric_labels:
                self.metric_labels[name] = {}
            self.metric_labels[name].update(labels)
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = int(index)
            upper = min(lower + 1, len(sorted_values) - 1)
            weight = index - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    def _initialize_standard_metrics(self) -> None:
        """Initialize standard metrics."""
        self.metric_descriptions.update({
            "queries_total": "Total number of queries processed",
            "query_response_time_seconds": "Query response time in seconds",
            "query_factuality_score": "Factuality score for queries (0-1)",
            "query_source_count": "Number of sources used per query",
            "successful_queries_total": "Total successful queries",
            "failed_queries_total": "Total failed queries",
            "retrieval_time_seconds": "Source retrieval time in seconds",
            "sources_retrieved": "Number of sources retrieved",
            "factuality_verification_time_seconds": "Factuality verification time",
            "governance_checks_total": "Total governance compliance checks",
            "component_operations_total": "Total component operations",
            "component_errors_total": "Total component errors"
        })
    
    def _setup_export_timer(self) -> None:
        """Setup periodic metric export."""
        if self.export_to and self.export_interval > 0:
            def export_worker():
                while True:
                    time.sleep(self.export_interval)
                    try:
                        self._export_metrics()
                    except Exception as e:
                        self.logger.error(f"Metric export failed: {e}")
            
            export_thread = threading.Thread(target=export_worker, daemon=True)
            export_thread.start()
    
    def _export_metrics(self) -> None:
        """Export metrics to configured destination."""
        metrics_data = self.get_all_metrics()
        
        if self.export_to == "prometheus":
            self._export_to_prometheus(metrics_data)
        elif self.export_to == "json":
            self._export_to_json(metrics_data)
        elif self.export_to == "log":
            self._export_to_log(metrics_data)
    
    def _export_to_prometheus(self, metrics_data: Dict[str, Any]) -> None:
        """Export metrics in Prometheus format."""
        # Placeholder for Prometheus export
        self.logger.info("Prometheus export not implemented yet")
    
    def _export_to_json(self, metrics_data: Dict[str, Any]) -> None:
        """Export metrics to JSON file."""
        filename = f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        self.logger.info(f"Metrics exported to {filename}")
    
    def _export_to_log(self, metrics_data: Dict[str, Any]) -> None:
        """Export metrics to log."""
        health_metrics = self.get_health_metrics()
        self.logger.info(f"Health metrics: {health_metrics}")


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]]):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.histogram(self.name, duration, self.labels)