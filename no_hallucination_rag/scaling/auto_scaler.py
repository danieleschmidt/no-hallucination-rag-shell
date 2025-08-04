"""
Advanced auto-scaling system with intelligent triggers and resource optimization.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics
import json
import asyncio


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingStrategy(Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"          # React to current load
    PREDICTIVE = "predictive"      # Predict future load
    SCHEDULED = "scheduled"        # Scale based on schedule
    HYBRID = "hybrid"             # Combination of strategies


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    COMPUTE_INSTANCES = "compute_instances"
    WORKER_THREADS = "worker_threads"
    CONNECTION_POOLS = "connection_pools"
    CACHE_SIZE = "cache_size"
    BATCH_SIZE = "batch_size"
    CONCURRENCY_LIMIT = "concurrency_limit"


@dataclass
class ScalingMetric:
    """Metric used for scaling decisions."""
    name: str
    current_value: float
    threshold_scale_up: float
    threshold_scale_down: float
    weight: float = 1.0
    aggregation_window_minutes: int = 5
    stability_period_minutes: int = 2


@dataclass
class ScalingRule:
    """Scaling rule definition."""
    name: str
    resource_type: ResourceType
    strategy: ScalingStrategy
    metrics: List[ScalingMetric]
    min_capacity: int = 1
    max_capacity: int = 100
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    cooldown_seconds: int = 300
    enabled: bool = True


@dataclass
class ScalingAction:
    """Scaling action record."""
    action_id: str
    resource_type: ResourceType
    direction: ScalingDirection
    from_capacity: int
    to_capacity: int
    trigger_metrics: Dict[str, float]
    timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class ResourceConfig:
    """Current resource configuration."""
    resource_type: ResourceType
    current_capacity: int
    target_capacity: int
    last_scaling_action: Optional[datetime] = None
    scaling_in_progress: bool = False


class AutoScaler:
    """Advanced auto-scaling system with intelligent triggers."""
    
    def __init__(
        self,
        evaluation_interval: int = 30,
        metric_retention_hours: int = 24,
        enable_predictive_scaling: bool = True,
        enable_cost_optimization: bool = True
    ):
        self.evaluation_interval = evaluation_interval
        self.metric_retention_hours = metric_retention_hours
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_cost_optimization = enable_cost_optimization
        
        self.logger = logging.getLogger(__name__)
        
        # Scaling rules and resources
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.resource_configs: Dict[ResourceType, ResourceConfig] = {}
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=int(metric_retention_hours * 60 / (evaluation_interval / 60)))
        )
        
        # Scaling history
        self.scaling_actions: deque = deque(maxlen=1000)
        
        # Predictive modeling
        self.load_patterns: Dict[str, List[float]] = defaultdict(list)
        self.seasonal_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Control
        self.scaling_enabled = False
        self.scaler_thread: Optional[threading.Thread] = None
        self._stop_scaling = threading.Event()
        self._lock = threading.RLock()
        
        # Scaling callbacks
        self.scaling_callbacks: Dict[ResourceType, Callable[[int, int], bool]] = {}
        
        # Load default scaling rules
        self._load_default_scaling_rules()
    
    def register_scaling_rule(self, rule: ScalingRule) -> None:
        """Register a scaling rule."""
        with self._lock:
            self.scaling_rules[rule.name] = rule
            
            # Initialize resource config if not exists
            if rule.resource_type not in self.resource_configs:
                self.resource_configs[rule.resource_type] = ResourceConfig(
                    resource_type=rule.resource_type,
                    current_capacity=rule.min_capacity,
                    target_capacity=rule.min_capacity
                )
            
            self.logger.info(f"Registered scaling rule: {rule.name}")
    
    def register_scaling_callback(
        self, 
        resource_type: ResourceType, 
        callback: Callable[[int, int], bool]
    ) -> None:
        """Register callback for scaling actions."""
        self.scaling_callbacks[resource_type] = callback
        self.logger.info(f"Registered scaling callback for {resource_type.value}")
    
    def start_scaling(self) -> None:
        """Start auto-scaling."""
        if self.scaling_enabled:
            self.logger.warning("Auto-scaling is already running")
            return
        
        self.scaling_enabled = True
        self._stop_scaling.clear()
        
        self.scaler_thread = threading.Thread(
            target=self._scaling_loop,
            name="AutoScaler",
            daemon=True
        )
        self.scaler_thread.start()
        
        self.logger.info("Auto-scaling started")
    
    def stop_scaling(self) -> None:
        """Stop auto-scaling."""
        if not self.scaling_enabled:
            return
        
        self.scaling_enabled = False
        self._stop_scaling.set()
        
        if self.scaler_thread and self.scaler_thread.is_alive():
            self.scaler_thread.join(timeout=10)
        
        self.logger.info("Auto-scaling stopped")
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a metric value."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self._lock:
            self.metrics_history[metric_name].append({
                "value": value,
                "timestamp": timestamp
            })
    
    def _scaling_loop(self) -> None:
        """Main scaling evaluation loop."""
        while not self._stop_scaling.is_set():
            try:
                self._evaluate_scaling_rules()
                self._update_load_patterns()
                
                if self.enable_predictive_scaling:
                    self._run_predictive_scaling()
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
            
            # Wait for next evaluation
            self._stop_scaling.wait(self.evaluation_interval)
    
    def _evaluate_scaling_rules(self) -> None:
        """Evaluate all scaling rules and make scaling decisions."""
        for rule_name, rule in list(self.scaling_rules.items()):
            if not rule.enabled:
                continue
            
            try:
                self._evaluate_single_rule(rule)
            except Exception as e:
                self.logger.error(f"Error evaluating scaling rule {rule_name}: {e}")
    
    def _evaluate_single_rule(self, rule: ScalingRule) -> None:
        """Evaluate a single scaling rule."""
        resource_config = self.resource_configs.get(rule.resource_type)
        if not resource_config:
            return
        
        # Check cooldown period
        if (resource_config.last_scaling_action and 
            (datetime.utcnow() - resource_config.last_scaling_action).total_seconds() < rule.cooldown_seconds):
            return
        
        # Skip if scaling already in progress
        if resource_config.scaling_in_progress:
            return
        
        # Collect current metric values
        current_metrics = {}
        scale_up_signals = 0
        scale_down_signals = 0
        total_weight = 0
        
        for metric in rule.metrics:
            metric_value = self._get_aggregated_metric(
                metric.name, 
                metric.aggregation_window_minutes
            )
            
            if metric_value is None:
                continue
            
            current_metrics[metric.name] = metric_value
            total_weight += metric.weight
            
            # Check thresholds
            if metric_value >= metric.threshold_scale_up:
                scale_up_signals += metric.weight
            elif metric_value <= metric.threshold_scale_down:
                scale_down_signals += metric.weight
        
        if total_weight == 0:
            return
        
        # Determine scaling direction
        scale_up_ratio = scale_up_signals / total_weight
        scale_down_ratio = scale_down_signals / total_weight
        
        direction = ScalingDirection.NONE
        if scale_up_ratio >= 0.5:  # Majority of weighted metrics suggest scale up
            direction = ScalingDirection.UP
        elif scale_down_ratio >= 0.7:  # Higher threshold for scale down
            direction = ScalingDirection.DOWN
        
        # Execute scaling if needed
        if direction != ScalingDirection.NONE:
            self._execute_scaling(rule, resource_config, direction, current_metrics)
    
    def _get_aggregated_metric(self, metric_name: str, window_minutes: int) -> Optional[float]:
        """Get aggregated metric value over time window."""
        if metric_name not in self.metrics_history:
            return None
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_values = [
            entry["value"] for entry in self.metrics_history[metric_name]
            if entry["timestamp"] >= cutoff_time
        ]
        
        if not recent_values:
            return None
        
        # Use average for aggregation
        return statistics.mean(recent_values)
    
    def _execute_scaling(
        self, 
        rule: ScalingRule, 
        resource_config: ResourceConfig, 
        direction: ScalingDirection,
        trigger_metrics: Dict[str, float]
    ) -> None:
        """Execute scaling action."""
        current_capacity = resource_config.current_capacity
        
        if direction == ScalingDirection.UP:
            new_capacity = min(
                current_capacity + rule.scale_up_increment,
                rule.max_capacity
            )
        else:  # Scale down
            new_capacity = max(
                current_capacity - rule.scale_down_increment,
                rule.min_capacity
            )
        
        if new_capacity == current_capacity:
            return  # No change needed
        
        # Create scaling action record
        action_id = f"scale_{rule.resource_type.value}_{int(time.time())}"
        action = ScalingAction(
            action_id=action_id,
            resource_type=rule.resource_type,
            direction=direction,
            from_capacity=current_capacity,
            to_capacity=new_capacity,
            trigger_metrics=trigger_metrics,
            timestamp=datetime.utcnow()
        )
        
        self.logger.info(
            f"Scaling {rule.resource_type.value} from {current_capacity} to {new_capacity} "
            f"({direction.value}) due to metrics: {trigger_metrics}"
        )
        
        # Mark scaling in progress
        resource_config.scaling_in_progress = True
        resource_config.target_capacity = new_capacity
        
        # Execute scaling callback
        success = True
        error_message = None
        start_time = time.time()
        
        try:
            callback = self.scaling_callbacks.get(rule.resource_type)
            if callback:
                success = callback(current_capacity, new_capacity)
            else:
                self.logger.warning(f"No scaling callback registered for {rule.resource_type.value}")
                success = False
                error_message = "No scaling callback registered"
                
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.error(f"Scaling callback failed: {e}")
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Update action record
        action.success = success
        action.error_message = error_message
        action.execution_time_ms = execution_time_ms
        
        # Update resource config
        if success:
            resource_config.current_capacity = new_capacity
            resource_config.last_scaling_action = datetime.utcnow()
        
        resource_config.scaling_in_progress = False
        
        # Store action
        with self._lock:
            self.scaling_actions.append(action)
    
    def _update_load_patterns(self) -> None:
        """Update load patterns for predictive scaling."""
        if not self.enable_predictive_scaling:
            return
        
        current_time = datetime.utcnow()
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        
        # Update hourly and daily patterns
        for metric_name, history in self.metrics_history.items():
            if not history:
                continue
            
            recent_avg = statistics.mean([entry["value"] for entry in list(history)[-10:]])
            
            # Update hourly pattern
            hourly_key = f"hour_{hour_of_day}"
            if hourly_key not in self.seasonal_patterns[metric_name]:
                self.seasonal_patterns[metric_name][hourly_key] = recent_avg
            else:
                # Exponential moving average
                alpha = 0.1
                self.seasonal_patterns[metric_name][hourly_key] = (
                    alpha * recent_avg + 
                    (1 - alpha) * self.seasonal_patterns[metric_name][hourly_key]
                )
            
            # Update daily pattern
            daily_key = f"day_{day_of_week}"
            if daily_key not in self.seasonal_patterns[metric_name]:
                self.seasonal_patterns[metric_name][daily_key] = recent_avg
            else:
                alpha = 0.05  # Slower adaptation for daily patterns
                self.seasonal_patterns[metric_name][daily_key] = (
                    alpha * recent_avg + 
                    (1 - alpha) * self.seasonal_patterns[metric_name][daily_key]
                )
    
    def _run_predictive_scaling(self) -> None:
        """Run predictive scaling based on historical patterns."""
        # Look ahead 15 minutes
        future_time = datetime.utcnow() + timedelta(minutes=15)
        
        for rule_name, rule in self.scaling_rules.items():
            if rule.strategy not in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
                continue
            
            resource_config = self.resource_configs.get(rule.resource_type)
            if not resource_config or resource_config.scaling_in_progress:
                continue
            
            # Predict load for each metric
            predicted_load = {}
            for metric in rule.metrics:
                prediction = self._predict_metric_value(metric.name, future_time)
                if prediction is not None:
                    predicted_load[metric.name] = prediction
            
            if not predicted_load:
                continue
            
            # Check if predicted load would trigger scaling
            scale_up_signals = 0
            total_weight = 0
            
            for metric in rule.metrics:
                if metric.name not in predicted_load:
                    continue
                
                predicted_value = predicted_load[metric.name]
                total_weight += metric.weight
                
                if predicted_value >= metric.threshold_scale_up:
                    scale_up_signals += metric.weight
            
            if total_weight > 0 and scale_up_signals / total_weight >= 0.6:
                self.logger.info(
                    f"Predictive scaling triggered for {rule.resource_type.value}: "
                    f"predicted load {predicted_load}"
                )
                
                # Execute proactive scale up (smaller increment)
                current_capacity = resource_config.current_capacity
                new_capacity = min(
                    current_capacity + max(1, rule.scale_up_increment // 2),
                    rule.max_capacity
                )
                
                if new_capacity > current_capacity:
                    self._execute_scaling(
                        rule, resource_config, ScalingDirection.UP,
                        {f"predicted_{k}": v for k, v in predicted_load.items()}
                    )
    
    def _predict_metric_value(self, metric_name: str, future_time: datetime) -> Optional[float]:
        """Predict metric value at future time."""
        if metric_name not in self.seasonal_patterns:
            return None
        
        patterns = self.seasonal_patterns[metric_name]
        
        # Get seasonal factors
        hour_key = f"hour_{future_time.hour}"
        day_key = f"day_{future_time.weekday()}"
        
        hourly_factor = patterns.get(hour_key, 1.0)
        daily_factor = patterns.get(day_key, 1.0)
        
        # Get recent baseline
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return None
        
        recent_values = [entry["value"] for entry in list(self.metrics_history[metric_name])[-20:]]
        baseline = statistics.mean(recent_values)
        
        # Simple prediction: baseline * seasonal factors
        prediction = baseline * (hourly_factor / baseline) * 0.7 + baseline * (daily_factor / baseline) * 0.3
        
        return max(0, prediction)
    
    def force_scaling(
        self, 
        resource_type: ResourceType, 
        target_capacity: int,
        reason: str = "Manual scaling"
    ) -> bool:
        """Force scaling to specific capacity."""
        resource_config = self.resource_configs.get(resource_type)
        if not resource_config:
            return False
        
        current_capacity = resource_config.current_capacity
        direction = ScalingDirection.UP if target_capacity > current_capacity else ScalingDirection.DOWN
        
        # Create manual scaling action
        action = ScalingAction(
            action_id=f"manual_scale_{resource_type.value}_{int(time.time())}",
            resource_type=resource_type,
            direction=direction,
            from_capacity=current_capacity,
            to_capacity=target_capacity,
            trigger_metrics={"manual_reason": reason},
            timestamp=datetime.utcnow()
        )
        
        # Execute callback
        success = True
        try:
            callback = self.scaling_callbacks.get(resource_type)
            if callback:
                success = callback(current_capacity, target_capacity)
            
            if success:
                resource_config.current_capacity = target_capacity
                resource_config.last_scaling_action = datetime.utcnow()
            
        except Exception as e:
            success = False
            action.error_message = str(e)
        
        action.success = success
        
        with self._lock:
            self.scaling_actions.append(action)
        
        self.logger.info(
            f"Manual scaling {resource_type.value} from {current_capacity} to {target_capacity}: "
            f"{'success' if success else 'failed'}"
        )
        
        return success
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        with self._lock:
            total_actions = len(self.scaling_actions)
            successful_actions = sum(1 for action in self.scaling_actions if action.success)
            
            recent_actions = [
                action for action in self.scaling_actions
                if action.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ]
            
            resource_status = {}
            for resource_type, config in self.resource_configs.items():
                resource_status[resource_type.value] = {
                    "current_capacity": config.current_capacity,
                    "target_capacity": config.target_capacity,
                    "scaling_in_progress": config.scaling_in_progress,
                    "last_scaling_action": config.last_scaling_action.isoformat() if config.last_scaling_action else None
                }
            
            return {
                "scaling_enabled": self.scaling_enabled,
                "total_scaling_actions": total_actions,
                "successful_actions": successful_actions,
                "success_rate": successful_actions / max(total_actions, 1),
                "recent_actions_24h": len(recent_actions),
                "resource_status": resource_status,
                "active_rules": len([r for r in self.scaling_rules.values() if r.enabled]),
                "predictive_scaling_enabled": self.enable_predictive_scaling
            }
    
    def get_recent_actions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent scaling actions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            recent_actions = [
                asdict(action) for action in self.scaling_actions
                if action.timestamp >= cutoff_time
            ]
        
        return recent_actions
    
    def _load_default_scaling_rules(self) -> None:
        """Load default scaling rules."""
        
        # CPU-based scaling for compute instances
        cpu_rule = ScalingRule(
            name="cpu_scaling",
            resource_type=ResourceType.COMPUTE_INSTANCES,
            strategy=ScalingStrategy.HYBRID,
            metrics=[
                ScalingMetric(
                    name="cpu_usage_percent",
                    current_value=0.0,
                    threshold_scale_up=75.0,
                    threshold_scale_down=25.0,
                    weight=1.0,
                    aggregation_window_minutes=5
                )
            ],
            min_capacity=1,
            max_capacity=20,
            scale_up_increment=2,
            scale_down_increment=1,
            cooldown_seconds=300
        )
        
        # Memory-based scaling
        memory_rule = ScalingRule(
            name="memory_scaling", 
            resource_type=ResourceType.COMPUTE_INSTANCES,
            strategy=ScalingStrategy.REACTIVE,
            metrics=[
                ScalingMetric(
                    name="memory_usage_percent",
                    current_value=0.0,
                    threshold_scale_up=80.0,
                    threshold_scale_down=30.0,
                    weight=0.8
                )
            ],
            min_capacity=1,
            max_capacity=15,
            cooldown_seconds=180
        )
        
        # Request rate-based scaling
        request_rate_rule = ScalingRule(
            name="request_rate_scaling",
            resource_type=ResourceType.WORKER_THREADS,
            strategy=ScalingStrategy.PREDICTIVE,
            metrics=[
                ScalingMetric(
                    name="requests_per_second",
                    current_value=0.0,
                    threshold_scale_up=100.0,
                    threshold_scale_down=20.0,
                    weight=1.0,
                    aggregation_window_minutes=3
                ),
                ScalingMetric(
                    name="avg_response_time_ms",
                    current_value=0.0,
                    threshold_scale_up=5000.0,  # 5 seconds
                    threshold_scale_down=1000.0,  # 1 second
                    weight=0.6
                )
            ],
            min_capacity=5,
            max_capacity=50,
            scale_up_increment=5,
            scale_down_increment=2,
            cooldown_seconds=120
        )
        
        # Register default rules
        self.register_scaling_rule(cpu_rule)
        self.register_scaling_rule(memory_rule)
        self.register_scaling_rule(request_rate_rule)


# Global auto-scaler instance
auto_scaler = AutoScaler()


def register_resource_scaler(
    resource_type: ResourceType,
    scaling_callback: Callable[[int, int], bool]
) -> None:
    """Register a resource scaling callback."""
    auto_scaler.register_scaling_callback(resource_type, scaling_callback)