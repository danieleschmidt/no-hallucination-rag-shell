
"""
Auto-scaling and load balancing system for dynamic resource management.
"""

import time
import threading
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    THREAD_POOL = "thread_pool"
    CONNECTION_POOL = "connection_pool"
    PROCESS_POOL = "process_pool"
    CACHE_SIZE = "cache_size"


@dataclass
class ScalingMetric:
    """Metric for scaling decisions."""
    name: str
    current_value: float
    timestamp: datetime
    resource_type: ResourceType
    threshold_up: float = 80.0      # Scale up when above this
    threshold_down: float = 30.0     # Scale down when below this
    weight: float = 1.0             # Importance weight


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: datetime
    resource_name: str
    resource_type: ResourceType
    action: ScalingAction
    old_size: int
    new_size: int
    trigger_metrics: List[ScalingMetric]
    reason: str


class LoadBalancer:
    """Intelligent load balancing for distributing work."""
    
    def __init__(self):
        self.backends: Dict[str, Dict[str, Any]] = {}
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def add_backend(
        self,
        name: str,
        endpoint: str,
        weight: int = 100,
        max_connections: int = 100
    ):
        """Add backend server/resource."""
        with self.lock:
            self.backends[name] = {
                "endpoint": endpoint,
                "weight": weight,
                "max_connections": max_connections,
                "current_connections": 0,
                "enabled": True,
                "health_score": 100.0
            }
            self.logger.info(f"Added backend: {name}")
    
    def remove_backend(self, name: str):
        """Remove backend server/resource."""
        with self.lock:
            if name in self.backends:
                del self.backends[name]
                self.load_history.pop(name, None)
                self.response_times.pop(name, None)
                self.failure_counts.pop(name, 0)
                self.logger.info(f"Removed backend: {name}")
    
    def select_backend(self, strategy: str = "weighted_round_robin") -> Optional[str]:
        """Select best backend based on strategy."""
        with self.lock:
            available_backends = [
                name for name, config in self.backends.items()
                if config["enabled"] and config["current_connections"] < config["max_connections"]
            ]
            
            if not available_backends:
                return None
            
            if strategy == "weighted_round_robin":
                # Simple selection for now
                return available_backends[0]
            
            return available_backends[0]


class AutoScaler:
    """Automatic scaling system for dynamic resource management."""
    
    def __init__(self, scaling_cooldown: int = 300):
        self.scaling_cooldown = scaling_cooldown  # 5 minutes between scaling events
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.scaling_events: List[ScalingEvent] = []
        self.last_scaling: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)
        self.load_balancer = LoadBalancer()
        
        # Current resource sizes
        self.current_sizes = {
            "thread_pool": 4,
            "connection_pool": 10,
            "cache_size": 1000
        }
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Update system metrics for scaling decisions."""
        timestamp = datetime.utcnow()
        
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append((timestamp, value))
            
        self.logger.debug(f"Updated metrics: {metrics}")
        
    def should_scale_up(self) -> Tuple[bool, str]:
        """Determine if system should scale up."""
        # Check recent metrics
        recent_metrics = self._get_recent_metrics(minutes=5)
        
        # Scale up conditions
        if recent_metrics.get("cpu_usage", 0) > 80:
            return True, "High CPU usage detected"
        
        if recent_metrics.get("memory_usage", 0) > 85:
            return True, "High memory usage detected"
            
        if recent_metrics.get("queue_length", 0) > 20:
            return True, "High queue length detected"
            
        if recent_metrics.get("response_time", 0) > 1000:  # > 1 second
            return True, "High response time detected"
            
        return False, "No scaling needed"
    
    def should_scale_down(self) -> Tuple[bool, str]:
        """Determine if system should scale down."""
        recent_metrics = self._get_recent_metrics(minutes=10)
        
        # Scale down conditions (more conservative)
        if (recent_metrics.get("cpu_usage", 100) < 30 and 
            recent_metrics.get("memory_usage", 100) < 40 and
            recent_metrics.get("queue_length", 100) < 5):
            return True, "Low resource utilization detected"
            
        return False, "No scale down needed"
    
    def _get_recent_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """Get average metrics from recent time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent = {}
        
        for metric_name, values in self.metrics_history.items():
            recent_values = [v for t, v in values if t > cutoff]
            if recent_values:
                recent[metric_name] = statistics.mean(recent_values)
                
        return recent
