"""
Comprehensive logging and monitoring for quantum task planning operations.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import os

from .quantum_planner import QuantumTask, TaskState, Priority
from .superposition_tasks import TaskSuperposition
from .entanglement_dependencies import EntanglementBond
from ..monitoring.audit_logger import AuditLevel


class QuantumEventType(Enum):
    """Types of quantum events to log."""
    TASK_CREATED = "task_created"
    TASK_OBSERVED = "task_observed"
    STATE_COLLAPSED = "state_collapsed"
    ENTANGLEMENT_CREATED = "entanglement_created"
    ENTANGLEMENT_BROKEN = "entanglement_broken"
    QUANTUM_GATE_APPLIED = "quantum_gate_applied"
    SUPERPOSITION_EVOLVED = "superposition_evolved"
    COHERENCE_LOST = "coherence_lost"
    INTERFERENCE_DETECTED = "interference_detected"
    BELL_VIOLATION = "bell_violation"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class QuantumLogEntry:
    """Structured log entry for quantum events."""
    timestamp: datetime
    event_type: QuantumEventType
    task_id: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    event_data: Dict[str, Any]
    quantum_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    error_info: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None


class QuantumLogger:
    """
    Advanced logging system for quantum task planning operations.
    
    Provides structured logging, quantum state tracking, performance monitoring,
    and audit trails for all quantum operations.
    """
    
    def __init__(
        self,
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        enable_compression: bool = True,
        max_log_size_mb: int = 100,
        backup_count: int = 5,
        enable_structured_logging: bool = True
    ):
        self.log_level = log_level
        self.log_file = log_file
        self.enable_compression = enable_compression
        self.max_log_size_mb = max_log_size_mb
        self.backup_count = backup_count
        self.enable_structured_logging = enable_structured_logging
        
        # Set up logger
        self.logger = logging.getLogger("quantum_planner")
        self.logger.setLevel(log_level)
        
        # Structured log storage
        self.quantum_logs: List[QuantumLogEntry] = []
        self.max_memory_logs = 10000
        
        # Performance tracking
        self.performance_metrics = {
            "operation_times": {},
            "quantum_states": {},
            "system_health": {},
            "error_counts": {}
        }
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        
        self._setup_logging()
        
        self.logger.info("Quantum Logger initialized")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Create formatter
        if self.enable_structured_logging:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.log_file:
            try:
                # Create log directory if it doesn't exist
                log_path = Path(self.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Set up rotating file handler
                from logging.handlers import RotatingFileHandler
                
                max_bytes = self.max_log_size_mb * 1024 * 1024
                file_handler = RotatingFileHandler(
                    self.log_file,
                    maxBytes=max_bytes,
                    backupCount=self.backup_count
                )
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(formatter)
                
                self.logger.addHandler(file_handler)
                
            except Exception as e:
                self.logger.error(f"Failed to set up file logging: {e}")
    
    def log_task_created(
        self,
        task: QuantumTask,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log task creation event."""
        
        event_data = {
            "task_id": task.id,
            "title": task.title,
            "priority": task.priority.value,
            "state": task.state.value,
            "coherence_time": task.coherence_time,
            "probability_amplitude": task.probability_amplitude,
            "dependencies_count": len(task.dependencies),
            "tags": list(task.tags)
        }
        
        quantum_state = self._capture_task_quantum_state(task)
        
        self._log_quantum_event(
            QuantumEventType.TASK_CREATED,
            task.id,
            user_id,
            session_id,
            event_data,
            quantum_state,
            performance_metrics or {}
        )
        
        # Traditional log
        self.logger.info(
            f"Task created: {task.title} (ID: {task.id[:8]}) "
            f"Priority: {task.priority.value} State: {task.state.value}"
        )
    
    def log_task_observed(
        self,
        task: QuantumTask,
        collapsed_state: TaskState,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        observation_time: Optional[float] = None
    ) -> None:
        """Log task observation and state collapse."""
        
        event_data = {
            "task_id": task.id,
            "original_state": task.state.value,
            "collapsed_state": collapsed_state.value,
            "observation_time_ms": observation_time * 1000 if observation_time else None,
            "coherence_lost": not task.is_coherent(),
            "entangled_tasks": list(task.entangled_tasks)
        }
        
        quantum_state = self._capture_task_quantum_state(task)
        
        performance_metrics = {}
        if observation_time:
            performance_metrics["observation_time"] = observation_time
        
        self._log_quantum_event(
            QuantumEventType.TASK_OBSERVED,
            task.id,
            user_id,
            session_id,
            event_data,
            quantum_state,
            performance_metrics
        )
        
        self.logger.info(
            f"Task observed: {task.title} (ID: {task.id[:8]}) "
            f"{task.state.value} -> {collapsed_state.value}"
        )
    
    def log_entanglement_created(
        self,
        task_id1: str,
        task_id2: str,
        entanglement_type: str,
        correlation_strength: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Log entanglement creation."""
        
        event_data = {
            "task_id1": task_id1,
            "task_id2": task_id2,
            "entanglement_type": entanglement_type,
            "correlation_strength": correlation_strength,
            "bond_created": True
        }
        
        quantum_state = {
            "entangled_pair": [task_id1, task_id2],
            "correlation_type": entanglement_type,
            "strength": correlation_strength
        }
        
        self._log_quantum_event(
            QuantumEventType.ENTANGLEMENT_CREATED,
            f"{task_id1}+{task_id2}",
            user_id,
            session_id,
            event_data,
            quantum_state,
            {}
        )
        
        self.logger.info(
            f"Entanglement created: {task_id1[:8]} ↔ {task_id2[:8]} "
            f"Type: {entanglement_type} Strength: {correlation_strength:.3f}"
        )
    
    def log_quantum_gate_applied(
        self,
        task_id: str,
        gate_type: str,
        before_state: Dict[TaskState, float],
        after_state: Dict[TaskState, float],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> None:
        """Log quantum gate operation."""
        
        event_data = {
            "task_id": task_id,
            "gate_type": gate_type,
            "before_state": {state.value: prob for state, prob in before_state.items()},
            "after_state": {state.value: prob for state, prob in after_state.items()},
            "entropy_before": self._calculate_entropy(before_state),
            "entropy_after": self._calculate_entropy(after_state),
            "purity_before": sum(p**2 for p in before_state.values()),
            "purity_after": sum(p**2 for p in after_state.values())
        }
        
        quantum_state = {
            "gate_operation": gate_type,
            "state_change": "evolved",
            "superposition_modified": True
        }
        
        performance_metrics = {}
        if execution_time:
            performance_metrics["gate_execution_time"] = execution_time
        
        self._log_quantum_event(
            QuantumEventType.QUANTUM_GATE_APPLIED,
            task_id,
            user_id,
            session_id,
            event_data,
            quantum_state,
            performance_metrics
        )
        
        self.logger.info(
            f"Quantum gate applied: {gate_type} on task {task_id[:8]} "
            f"Entropy: {event_data['entropy_before']:.3f} -> {event_data['entropy_after']:.3f}"
        )
    
    def log_superposition_evolved(
        self,
        task_id: str,
        time_step: float,
        before_amplitudes: Dict[TaskState, float],
        after_amplitudes: Dict[TaskState, float]
    ) -> None:
        """Log superposition evolution over time."""
        
        event_data = {
            "task_id": task_id,
            "time_step": time_step,
            "evolution_type": "schrodinger",
            "amplitude_changes": {
                state.value: after_amplitudes[state] - before_amplitudes.get(state, 0)
                for state in after_amplitudes.keys()
            }
        }
        
        quantum_state = {
            "evolution": "unitary",
            "time_evolution": time_step,
            "coherence_maintained": True
        }
        
        self._log_quantum_event(
            QuantumEventType.SUPERPOSITION_EVOLVED,
            task_id,
            None,
            None,
            event_data,
            quantum_state,
            {"evolution_time": time_step}
        )
        
        self.logger.debug(f"Superposition evolved: task {task_id[:8]} (Δt={time_step:.3f})")
    
    def log_coherence_lost(
        self,
        task_id: str,
        coherence_time: float,
        decoherence_cause: str = "natural"
    ) -> None:
        """Log coherence loss event."""
        
        event_data = {
            "task_id": task_id,
            "coherence_time": coherence_time,
            "decoherence_cause": decoherence_cause,
            "forced_collapse": decoherence_cause != "natural"
        }
        
        quantum_state = {
            "coherence": "lost",
            "decoherence_type": decoherence_cause
        }
        
        self._log_quantum_event(
            QuantumEventType.COHERENCE_LOST,
            task_id,
            None,
            None,
            event_data,
            quantum_state,
            {"coherence_duration": coherence_time}
        )
        
        self.logger.warning(
            f"Coherence lost: task {task_id[:8]} after {coherence_time:.1f}s "
            f"Cause: {decoherence_cause}"
        )
    
    def log_bell_violation(
        self,
        task_id1: str,
        task_id2: str,
        bell_parameter: float,
        violation_strength: float
    ) -> None:
        """Log Bell inequality violation detection."""
        
        event_data = {
            "task_id1": task_id1,
            "task_id2": task_id2,
            "bell_parameter": bell_parameter,
            "violation_strength": violation_strength,
            "classical_bound": 2.0,
            "quantum_confirmed": abs(bell_parameter) > 2.0
        }
        
        quantum_state = {
            "non_locality": "detected",
            "entanglement": "confirmed",
            "bell_test": "violated"
        }
        
        self._log_quantum_event(
            QuantumEventType.BELL_VIOLATION,
            f"{task_id1}+{task_id2}",
            None,
            None,
            event_data,
            quantum_state,
            {"bell_parameter": bell_parameter}
        )
        
        self.logger.info(
            f"Bell violation detected: {task_id1[:8]} ↔ {task_id2[:8]} "
            f"S = {bell_parameter:.3f} (violation: {violation_strength:.3f})"
        )
    
    def log_system_error(
        self,
        operation: str,
        error_type: str,
        error_message: str,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        stack_trace: Optional[str] = None
    ) -> None:
        """Log system errors."""
        
        event_data = {
            "operation": operation,
            "error_type": error_type,
            "error_message": error_message,
            "task_id": task_id,
            "recoverable": "recoverable" in error_message.lower()
        }
        
        error_info = {
            "type": error_type,
            "message": error_message,
            "stack_trace": stack_trace
        }
        
        self._log_quantum_event(
            QuantumEventType.SYSTEM_ERROR,
            task_id,
            user_id,
            None,
            event_data,
            {"system_state": "error"},
            {},
            error_info
        )
        
        self.logger.error(
            f"System error in {operation}: {error_type} - {error_message}"
            + (f" (Task: {task_id[:8]})" if task_id else "")
        )
        
        # Update error counts
        self.performance_metrics["error_counts"][error_type] = (
            self.performance_metrics["error_counts"].get(error_type, 0) + 1
        )
    
    def log_performance_degradation(
        self,
        operation: str,
        expected_time: float,
        actual_time: float,
        degradation_factor: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance degradation events."""
        
        event_data = {
            "operation": operation,
            "expected_time_ms": expected_time * 1000,
            "actual_time_ms": actual_time * 1000,
            "degradation_factor": degradation_factor,
            "performance_impact": "high" if degradation_factor > 2.0 else "medium",
            "context": context or {}
        }
        
        performance_metrics = {
            "expected_time": expected_time,
            "actual_time": actual_time,
            "degradation": degradation_factor
        }
        
        self._log_quantum_event(
            QuantumEventType.PERFORMANCE_DEGRADATION,
            None,
            None,
            None,
            event_data,
            {"performance": "degraded"},
            performance_metrics
        )
        
        self.logger.warning(
            f"Performance degradation: {operation} took {actual_time:.3f}s "
            f"(expected {expected_time:.3f}s, {degradation_factor:.1f}x slower)"
        )
    
    def _log_quantum_event(
        self,
        event_type: QuantumEventType,
        task_id: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        event_data: Dict[str, Any],
        quantum_state: Dict[str, Any],
        performance_metrics: Dict[str, float],
        error_info: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> None:
        """Log structured quantum event."""
        
        log_entry = QuantumLogEntry(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            task_id=task_id,
            user_id=user_id,
            session_id=session_id,
            event_data=event_data,
            quantum_state=quantum_state,
            performance_metrics=performance_metrics,
            error_info=error_info,
            trace_id=trace_id
        )
        
        # Add to memory logs
        self.quantum_logs.append(log_entry)
        
        # Maintain memory limit
        if len(self.quantum_logs) > self.max_memory_logs:
            # Remove oldest logs
            self.quantum_logs = self.quantum_logs[-self.max_memory_logs:]
        
        # Write structured log to file if enabled
        if self.log_file and self.enable_structured_logging:
            self._write_structured_log(log_entry)
        
        # Update performance metrics
        self._update_performance_metrics(event_type, performance_metrics)
        
        # Add to audit trail
        audit_entry = {
            "timestamp": log_entry.timestamp.isoformat(),
            "event": event_type.value,
            "task_id": task_id,
            "user_id": user_id,
            "data": event_data
        }
        self.audit_trail.append(audit_entry)
        
        # Maintain audit trail limit
        if len(self.audit_trail) > 1000:
            self.audit_trail = self.audit_trail[-1000:]
    
    def _write_structured_log(self, log_entry: QuantumLogEntry) -> None:
        """Write structured log entry to file."""
        
        try:
            # Convert to JSON
            log_dict = asdict(log_entry)
            log_dict["timestamp"] = log_entry.timestamp.isoformat()
            log_dict["event_type"] = log_entry.event_type.value
            
            # Write to structured log file
            structured_log_file = f"{self.log_file}.quantum.jsonl"
            
            with open(structured_log_file, 'a') as f:
                f.write(json.dumps(log_dict) + '\n')
            
        except Exception as e:
            self.logger.error(f"Failed to write structured log: {e}")
    
    def _update_performance_metrics(
        self,
        event_type: QuantumEventType,
        metrics: Dict[str, float]
    ) -> None:
        """Update performance metrics."""
        
        event_key = event_type.value
        
        if event_key not in self.performance_metrics["operation_times"]:
            self.performance_metrics["operation_times"][event_key] = []
        
        # Add execution time if available
        if "execution_time" in metrics or "observation_time" in metrics:
            exec_time = metrics.get("execution_time") or metrics.get("observation_time", 0)
            self.performance_metrics["operation_times"][event_key].append(exec_time)
            
            # Keep only recent measurements
            if len(self.performance_metrics["operation_times"][event_key]) > 100:
                self.performance_metrics["operation_times"][event_key] = (
                    self.performance_metrics["operation_times"][event_key][-100:]
                )
    
    def _capture_task_quantum_state(self, task: QuantumTask) -> Dict[str, Any]:
        """Capture quantum state information for a task."""
        
        return {
            "state": task.state.value,
            "probability_amplitude": task.probability_amplitude,
            "coherence_time": task.coherence_time,
            "is_coherent": task.is_coherent(),
            "entangled_count": len(task.entangled_tasks),
            "quantum_phase": hash(task.id) % 360,  # Simplified phase representation
            "created_elapsed": (datetime.utcnow() - task.created_at).total_seconds()
        }
    
    def _calculate_entropy(self, probabilities: Dict[TaskState, float]) -> float:
        """Calculate von Neumann entropy of probability distribution."""
        
        import math
        
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_logs = [
            log for log in self.quantum_logs
            if log.timestamp > cutoff_time
        ]
        
        # Count events by type
        event_counts = {}
        for log in recent_logs:
            event_type = log.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Calculate average operation times
        avg_times = {}
        for operation, times in self.performance_metrics["operation_times"].items():
            if times:
                avg_times[operation] = sum(times) / len(times)
        
        # Error statistics
        error_logs = [log for log in recent_logs if log.event_type == QuantumEventType.SYSTEM_ERROR]
        error_rate = len(error_logs) / max(1, len(recent_logs))
        
        # Quantum state statistics
        coherent_tasks = sum(
            1 for log in recent_logs
            if log.quantum_state.get("coherence") == "maintained" 
            or log.quantum_state.get("is_coherent") is True
        )
        
        return {
            "time_window_hours": time_window_hours,
            "total_events": len(recent_logs),
            "event_distribution": event_counts,
            "average_operation_times": avg_times,
            "error_rate": error_rate,
            "error_count": len(error_logs),
            "quantum_statistics": {
                "coherent_tasks": coherent_tasks,
                "decoherence_events": event_counts.get("coherence_lost", 0),
                "entanglement_operations": event_counts.get("entanglement_created", 0),
                "bell_violations": event_counts.get("bell_violation", 0)
            },
            "system_health": "healthy" if error_rate < 0.1 else "degraded"
        }
    
    def get_audit_trail(self, task_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit trail for task or system."""
        
        if task_id:
            filtered_trail = [
                entry for entry in self.audit_trail
                if entry.get("task_id") == task_id
            ]
        else:
            filtered_trail = self.audit_trail
        
        # Return most recent entries
        return sorted(
            filtered_trail,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def export_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json"
    ) -> str:
        """Export logs to file."""
        
        # Filter logs by time range
        filtered_logs = self.quantum_logs
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_logs_export_{timestamp}.{format}"
        
        try:
            if format == "json":
                log_data = []
                for log in filtered_logs:
                    log_dict = asdict(log)
                    log_dict["timestamp"] = log.timestamp.isoformat()
                    log_dict["event_type"] = log.event_type.value
                    log_data.append(log_dict)
                
                with open(filename, 'w') as f:
                    json.dump(log_data, f, indent=2)
            
            elif format == "csv":
                import csv
                
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        "timestamp", "event_type", "task_id", "user_id",
                        "event_data", "quantum_state", "performance_metrics"
                    ])
                    
                    # Data
                    for log in filtered_logs:
                        writer.writerow([
                            log.timestamp.isoformat(),
                            log.event_type.value,
                            log.task_id or "",
                            log.user_id or "",
                            json.dumps(log.event_data),
                            json.dumps(log.quantum_state),
                            json.dumps(log.performance_metrics)
                        ])
            
            # Compress if enabled
            if self.enable_compression:
                compressed_filename = f"{filename}.gz"
                with open(filename, 'rb') as f_in:
                    with gzip.open(compressed_filename, 'wb') as f_out:
                        f_out.writelines(f_in)
                
                # Remove uncompressed file
                os.remove(filename)
                filename = compressed_filename
            
            self.logger.info(f"Logs exported to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            raise
    
    def cleanup_logs(self, older_than_days: int = 30) -> int:
        """Clean up old log entries."""
        
        cutoff_time = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Count logs to be removed
        old_logs = [log for log in self.quantum_logs if log.timestamp < cutoff_time]
        removed_count = len(old_logs)
        
        # Remove old logs
        self.quantum_logs = [log for log in self.quantum_logs if log.timestamp >= cutoff_time]
        
        # Clean audit trail
        self.audit_trail = [
            entry for entry in self.audit_trail
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]
        
        self.logger.info(f"Cleaned up {removed_count} old log entries")
        return removed_count
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics."""
        
        return {
            "total_logs": len(self.quantum_logs),
            "memory_usage_mb": len(str(self.quantum_logs)) / (1024 * 1024),
            "audit_trail_entries": len(self.audit_trail),
            "log_level": logging.getLevelName(self.log_level),
            "structured_logging": self.enable_structured_logging,
            "compression_enabled": self.enable_compression,
            "oldest_log": min(log.timestamp for log in self.quantum_logs).isoformat() if self.quantum_logs else None,
            "newest_log": max(log.timestamp for log in self.quantum_logs).isoformat() if self.quantum_logs else None
        }