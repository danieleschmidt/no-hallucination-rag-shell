
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


@dataclass
class ScalingDecision:
    """Decision for scaling action."""
    action: str
    reason: str
    instances_delta: int
    target_instances: int = 0
    trigger_metrics: Dict[str, Any] = None


class PerformanceTracker:
    """Tracks performance metrics for scaling decisions."""
    
    def __init__(self):
        self.scaling_history = []
        self.performance_metrics = {}
        
    def record_scaling_event(self, action: str, old_count: int, new_count: int):
        """Record a scaling event."""
        self.scaling_history.append({
            'timestamp': time.time(),
            'action': action,
            'old_count': old_count,
            'new_count': new_count
        })
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'total_scaling_events': len(self.scaling_history),
            'recent_events': self.scaling_history[-10:]
        }


class LoadPredictor:
    """Predicts future load patterns."""
    
    def __init__(self):
        self.historical_data = []
        
    def predict_future_load(self) -> Optional[Dict[str, float]]:
        """Predict future load based on historical patterns."""
        if len(self.historical_data) < 10:
            return None
            
        # Simple prediction based on recent trend
        recent_loads = self.historical_data[-10:]
        if recent_loads:
            avg_load = sum(recent_loads) / len(recent_loads)
            return {'predicted_cpu': avg_load}
        return None


class SeasonalPatternAnalyzer:
    """Analyzes seasonal patterns in load."""
    
    def __init__(self):
        self.patterns = {}
        
    def analyze_patterns(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze seasonal patterns."""
        return {"patterns": "analysis_pending"}
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
                name for name, info in self.backends.items()
                if info["enabled"] and info["current_connections"] < info["max_connections"]
            ]
            
            if not available_backends:
                return None
            
            if strategy == "weighted_round_robin":
                return self._weighted_round_robin(available_backends)
            elif strategy == "least_connections":
                return self._least_connections(available_backends)
            elif strategy == "response_time":
                return self._fastest_response(available_backends)
            elif strategy == "health_score":
                return self._best_health(available_backends)
            else:
                # Default to round robin
                return available_backends[0]
    
    def _weighted_round_robin(self, backends: List[str]) -> str:
        """Weighted round robin selection."""
        total_weight = sum(self.backends[name]["weight"] for name in backends)
        if total_weight == 0:
            return backends[0]
        
        # Simple implementation - can be improved with better round robin
        weights = [(name, self.backends[name]["weight"]) for name in backends]
        weights.sort(key=lambda x: x[1], reverse=True)
        
        # For simplicity, return highest weight backend
        # In production, implement proper weighted round robin
        return weights[0][0]
    
    def _least_connections(self, backends: List[str]) -> str:
        """Least connections selection."""
        return min(backends, key=lambda name: self.backends[name]["current_connections"])
    
    def _fastest_response(self, backends: List[str]) -> str:
        """Fastest average response time selection."""
        best_backend = backends[0]
        best_avg_time = float('inf')
        
        for name in backends:
            if name in self.response_times and self.response_times[name]:
                avg_time = statistics.mean(self.response_times[name])
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_backend = name
        
        return best_backend
    
    def _best_health(self, backends: List[str]) -> str:
        """Best health score selection."""
        return max(backends, key=lambda name: self.backends[name]["health_score"])
    
    def record_request(self, backend_name: str, response_time: float, success: bool):
        """Record request metrics for a backend."""
        with self.lock:
            if backend_name not in self.backends:
                return
            
            # Record response time
            self.response_times[backend_name].append(response_time)
            
            # Record load
            current_time = time.time()
            self.load_history[backend_name].append(current_time)
            
            # Update failure count
            if not success:
                self.failure_counts[backend_name] += 1
            else:
                # Reset failure count on success
                self.failure_counts[backend_name] = max(0, self.failure_counts[backend_name] - 1)
            
            # Update health score
            self._update_health_score(backend_name)
    
    def _update_health_score(self, backend_name: str):
        """Update health score based on recent performance."""
        if backend_name not in self.backends:
            return
        
        health_score = 100.0
        
        # Factor in failure rate
        failure_rate = self.failure_counts[backend_name] / max(1, len(self.response_times[backend_name]))
        health_score -= failure_rate * 50  # Penalize failures heavily
        
        # Factor in response time
        if self.response_times[backend_name]:
            avg_response_time = statistics.mean(self.response_times[backend_name])
            if avg_response_time > 1.0:  # Penalize slow responses
                health_score -= min(30, avg_response_time * 10)
        
        # Factor in current load
        current_load = self.backends[backend_name]["current_connections"]
        max_load = self.backends[backend_name]["max_connections"]
        load_ratio = current_load / max_load if max_load > 0 else 0
        health_score -= load_ratio * 20  # Penalize high load
        
        self.backends[backend_name]["health_score"] = max(0, min(100, health_score))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            backend_stats = {}
            for name, info in self.backends.items():
                avg_response_time = 0
                if self.response_times[name]:
                    avg_response_time = statistics.mean(self.response_times[name])
                
                # Calculate recent request rate
                now = time.time()
                recent_requests = [
                    t for t in self.load_history[name]
                    if now - t < 60  # Last minute
                ]
                request_rate = len(recent_requests) / 60.0
                
                backend_stats[name] = {
                    "enabled": info["enabled"],
                    "current_connections": info["current_connections"],
                    "max_connections": info["max_connections"],
                    "health_score": info["health_score"],
                    "avg_response_time": avg_response_time,
                    "failure_count": self.failure_counts[name],
                    "request_rate": request_rate
                }
            
            return {
                "backends": backend_stats,
                "total_backends": len(self.backends),
                "healthy_backends": len([
                    name for name, info in self.backends.items()
                    if info["enabled"] and info["health_score"] > 50
                ])
            }


class AutoScaler:
    """Automatic resource scaling based on metrics and policies."""
    
    def __init__(self, scale_up_cooldown: int = 300, scale_down_cooldown: int = 600):
        self.scale_up_cooldown = scale_up_cooldown    # 5 minutes
        self.scale_down_cooldown = scale_down_cooldown  # 10 minutes
        
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.scaling_policies: Dict[str, Dict[str, Any]] = {}
        self.last_scaling_events: Dict[str, datetime] = {}
        self.scaling_history: List[ScalingEvent] = []
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring thread
        self.monitoring_active = False
        self._start_monitoring()
    
    def add_scaling_policy(
        self,
        resource_name: str,
        resource_type: ResourceType,
        min_size: int = 1,
        max_size: int = 20,
        target_metrics: List[Tuple[str, float, float]] = None,  # name, scale_up, scale_down
        scale_up_step: int = 1,
        scale_down_step: int = 1,
        resource_controller: Optional[Callable] = None
    ):
        """Add auto-scaling policy for a resource."""
        target_metrics = target_metrics or [("utilization", 80.0, 30.0)]
        
        policy = {
            "resource_type": resource_type,
            "min_size": min_size,
            "max_size": max_size,
            "current_size": min_size,
            "target_metrics": target_metrics,
            "scale_up_step": scale_up_step,
            "scale_down_step": scale_down_step,
            "resource_controller": resource_controller
        }
        
        with self.lock:
            self.scaling_policies[resource_name] = policy
            self.logger.info(f"Added scaling policy for {resource_name}")
    
    def record_metric(self, resource_name: str, metric_name: str, value: float):
        """Record metric for scaling decisions."""
        with self.lock:
            key = f"{resource_name}:{metric_name}"
            metric = ScalingMetric(
                name=metric_name,
                current_value=value,
                timestamp=datetime.utcnow(),
                resource_type=self.scaling_policies.get(resource_name, {}).get("resource_type", ResourceType.THREAD_POOL)
            )
            self.metrics[key].append(metric)
    
    def evaluate_scaling(self, resource_name: str) -> ScalingAction:
        """Evaluate if resource needs scaling."""
        if resource_name not in self.scaling_policies:
            return ScalingAction.MAINTAIN
        
        policy = self.scaling_policies[resource_name]
        current_size = policy["current_size"]
        
        # Check cooldown periods
        last_event = self.last_scaling_events.get(resource_name)
        if last_event:
            time_since_last = (datetime.utcnow() - last_event).total_seconds()
            if time_since_last < self.scale_up_cooldown:
                return ScalingAction.MAINTAIN
        
        # Collect recent metrics
        scale_up_votes = 0
        scale_down_votes = 0
        trigger_metrics = []
        
        for metric_name, up_threshold, down_threshold in policy["target_metrics"]:
            key = f"{resource_name}:{metric_name}"
            recent_metrics = list(self.metrics[key])[-10:]  # Last 10 readings
            
            if not recent_metrics:
                continue
            
            # Calculate average of recent metrics
            avg_value = sum(m.current_value for m in recent_metrics) / len(recent_metrics)
            
            latest_metric = recent_metrics[-1]
            latest_metric.threshold_up = up_threshold
            latest_metric.threshold_down = down_threshold
            trigger_metrics.append(latest_metric)
            
            # Vote for scaling action
            if avg_value > up_threshold and current_size < policy["max_size"]:
                scale_up_votes += 1
            elif avg_value < down_threshold and current_size > policy["min_size"]:
                scale_down_votes += 1
        
        # Determine action based on votes
        if scale_up_votes > 0:
            return ScalingAction.SCALE_UP
        elif scale_down_votes > 0:
            # More conservative on scaling down
            if scale_down_votes >= len(policy["target_metrics"]) / 2:
                return ScalingAction.SCALE_DOWN
        
        return ScalingAction.MAINTAIN
    
    def execute_scaling(self, resource_name: str, action: ScalingAction) -> bool:
        """Execute scaling action."""
        if resource_name not in self.scaling_policies:
            return False
        
        policy = self.scaling_policies[resource_name]
        old_size = policy["current_size"]
        new_size = old_size
        
        if action == ScalingAction.SCALE_UP:
            new_size = min(policy["max_size"], old_size + policy["scale_up_step"])
        elif action == ScalingAction.SCALE_DOWN:
            new_size = max(policy["min_size"], old_size - policy["scale_down_step"])
        else:
            return False  # No change needed
        
        if new_size == old_size:
            return False  # No change possible
        
        # Execute scaling using resource controller
        if policy["resource_controller"]:
            try:
                success = policy["resource_controller"](resource_name, new_size)
                if not success:
                    return False
            except Exception as e:
                self.logger.error(f"Scaling execution failed for {resource_name}: {e}")
                return False
        
        # Update policy and record event
        with self.lock:
            policy["current_size"] = new_size
            self.last_scaling_events[resource_name] = datetime.utcnow()
            
            # Collect trigger metrics
            trigger_metrics = []
            for metric_name, up_threshold, down_threshold in policy["target_metrics"]:
                key = f"{resource_name}:{metric_name}"
                if self.metrics[key]:
                    trigger_metrics.append(self.metrics[key][-1])
            
            event = ScalingEvent(
                timestamp=datetime.utcnow(),
                resource_name=resource_name,
                resource_type=policy["resource_type"],
                action=action,
                old_size=old_size,
                new_size=new_size,
                trigger_metrics=trigger_metrics,
                reason=f"Automated scaling: {action.value}"
            )
            
            self.scaling_history.append(event)
            
            # Keep history limited
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-500:]
        
        self.logger.info(f"Scaled {resource_name} from {old_size} to {new_size} ({action.value})")
        return True
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while self.monitoring_active:
            try:
                with self.lock:
                    for resource_name in list(self.scaling_policies.keys()):
                        action = self.evaluate_scaling(resource_name)
                        if action != ScalingAction.MAINTAIN:
                            self.execute_scaling(resource_name, action)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaling monitoring error: {e}")
                time.sleep(60)  # Back off on errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self.lock:
            policy_stats = {}
            for name, policy in self.scaling_policies.items():
                recent_events = [
                    e for e in self.scaling_history
                    if e.resource_name == name and 
                    (datetime.utcnow() - e.timestamp).total_seconds() < 86400  # Last 24h
                ]
                
                policy_stats[name] = {
                    "current_size": policy["current_size"],
                    "min_size": policy["min_size"],
                    "max_size": policy["max_size"],
                    "resource_type": policy["resource_type"].value,
                    "events_24h": len(recent_events),
                    "last_scaling": self.last_scaling_events.get(name)
                }
            
            return {
                "policies": policy_stats,
                "total_scaling_events": len(self.scaling_history),
                "monitoring_active": self.monitoring_active
            }


class ResourceManager:
    """Integrated resource management with scaling and load balancing."""
    
    def __init__(self):
        self.auto_scaler = AutoScaler()
        self.load_balancer = LoadBalancer()
        self.logger = logging.getLogger(__name__)
    
    def register_scalable_resource(
        self,
        name: str,
        resource_type: ResourceType,
        controller: Callable,
        **scaling_config
    ):
        """Register a resource for auto-scaling."""
        self.auto_scaler.add_scaling_policy(
            resource_name=name,
            resource_type=resource_type,
            resource_controller=controller,
            **scaling_config
        )
    
    def register_load_balanced_backend(self, name: str, endpoint: str, **config):
        """Register a backend for load balancing."""
        self.load_balancer.add_backend(name, endpoint, **config)
    
    def record_resource_metrics(self, resource_name: str, metrics: Dict[str, float]):
        """Record metrics for resource scaling decisions."""
        for metric_name, value in metrics.items():
            self.auto_scaler.record_metric(resource_name, metric_name, value)
    
    def record_backend_request(self, backend_name: str, response_time: float, success: bool):
        """Record backend request for load balancing decisions."""
        self.load_balancer.record_request(backend_name, response_time, success)
    
    def select_backend(self, strategy: str = "weighted_round_robin") -> Optional[str]:
        """Select best backend for request."""
        return self.load_balancer.select_backend(strategy)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource management statistics."""
        return {
            "auto_scaling": self.auto_scaler.get_stats(),
            "load_balancing": self.load_balancer.get_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global resource manager
global_resource_manager = ResourceManager()
