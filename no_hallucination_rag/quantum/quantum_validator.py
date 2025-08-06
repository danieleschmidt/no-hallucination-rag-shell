"""
Validation and error handling for quantum task planning operations.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime, timedelta

from .quantum_planner import QuantumTask, TaskState, Priority
from .entanglement_dependencies import EntanglementType
from ..core.validation import ValidationResult
from ..core.error_handling import ValidationError, ErrorContext


class QuantumValidationError(Exception):
    """Specific validation errors for quantum operations."""
    pass


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QuantumValidationResult:
    """Result of quantum operation validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    severity: ValidationSeverity = ValidationSeverity.INFO
    context: Optional[Dict[str, Any]] = None


class QuantumValidator:
    """
    Comprehensive validation for quantum task planning operations.
    
    Validates quantum states, entanglement operations, and task properties
    to ensure system consistency and prevent quantum decoherence.
    """
    
    def __init__(self, max_tasks: int = 1000, max_entanglements: int = 10000):
        self.max_tasks = max_tasks
        self.max_entanglements = max_entanglements
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.title_pattern = re.compile(r'^[a-zA-Z0-9\s\-_.,!?()]{1,200}$')
        self.description_max_length = 2000
        self.min_coherence_time = 60.0  # seconds
        self.max_coherence_time = 86400.0  # 24 hours
        
    def validate_task_creation(
        self,
        title: str,
        description: str = "",
        priority: Optional[Priority] = None,
        due_date: Optional[datetime] = None,
        dependencies: Optional[Set[str]] = None,
        tags: Optional[Set[str]] = None,
        estimated_duration: Optional[timedelta] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QuantumValidationResult:
        """Validate task creation parameters."""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Title validation
        if not title or not title.strip():
            errors.append("Task title cannot be empty")
        elif len(title) > 200:
            errors.append("Task title cannot exceed 200 characters")
        elif not self.title_pattern.match(title):
            errors.append("Task title contains invalid characters")
        
        # Description validation
        if len(description) > self.description_max_length:
            errors.append(f"Description cannot exceed {self.description_max_length} characters")
        
        # Priority validation
        if priority and not isinstance(priority, Priority):
            errors.append("Invalid priority value")
        elif priority == Priority.IONIZED:
            warnings.append("Ionized priority should be used sparingly for critical tasks only")
        
        # Due date validation
        if due_date:
            if due_date <= datetime.utcnow():
                warnings.append("Due date is in the past")
            elif due_date > datetime.utcnow() + timedelta(days=365):
                warnings.append("Due date is more than a year away")
        
        # Dependencies validation
        if dependencies:
            if len(dependencies) > 20:
                warnings.append("Task has many dependencies, consider breaking it down")
            
            for dep_id in dependencies:
                if not self._is_valid_task_id(dep_id):
                    errors.append(f"Invalid dependency task ID: {dep_id}")
        
        # Tags validation
        if tags:
            if len(tags) > 10:
                warnings.append("Too many tags, consider consolidating")
            
            for tag in tags:
                if len(tag) > 50:
                    errors.append(f"Tag '{tag}' exceeds 50 characters")
                elif not re.match(r'^[a-zA-Z0-9\-_]+$', tag):
                    errors.append(f"Tag '{tag}' contains invalid characters")
        
        # Duration validation
        if estimated_duration:
            if estimated_duration <= timedelta():
                errors.append("Estimated duration must be positive")
            elif estimated_duration > timedelta(days=30):
                warnings.append("Estimated duration is very long, consider breaking task down")
        
        # Context validation
        if context:
            if len(str(context)) > 5000:
                warnings.append("Task context is very large")
        
        # Suggestions
        if not description and not context:
            suggestions.append("Consider adding a description or context for better task management")
        
        if not due_date and priority in [Priority.THIRD_EXCITED, Priority.IONIZED]:
            suggestions.append("High priority tasks should have due dates")
        
        severity = self._determine_severity(errors, warnings)
        
        return QuantumValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            severity=severity,
            context={"operation": "task_creation"}
        )
    
    def validate_quantum_state_transition(
        self,
        current_state: TaskState,
        target_state: TaskState,
        task_context: Optional[Dict[str, Any]] = None
    ) -> QuantumValidationResult:
        """Validate quantum state transitions."""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Valid state transitions
        valid_transitions = {
            TaskState.SUPERPOSITION: [TaskState.COLLAPSED, TaskState.ENTANGLED, TaskState.FAILED],
            TaskState.COLLAPSED: [TaskState.COMPLETED, TaskState.FAILED, TaskState.ENTANGLED],
            TaskState.ENTANGLED: [TaskState.COMPLETED, TaskState.FAILED, TaskState.COLLAPSED],
            TaskState.COMPLETED: [],  # Terminal state
            TaskState.FAILED: [TaskState.SUPERPOSITION, TaskState.COLLAPSED]  # Can retry
        }
        
        if target_state not in valid_transitions.get(current_state, []):
            errors.append(f"Invalid state transition: {current_state.value} -> {target_state.value}")
        
        # Specific transition warnings
        if current_state == TaskState.COMPLETED and target_state != TaskState.COMPLETED:
            warnings.append("Transitioning from completed state may lose progress")
        
        if current_state == TaskState.SUPERPOSITION and target_state == TaskState.FAILED:
            warnings.append("Direct collapse to failed state - consider intermediate steps")
        
        # Context-specific validations
        if task_context:
            coherence_time = task_context.get('coherence_time', 0)
            if current_state == TaskState.SUPERPOSITION and coherence_time <= 0:
                errors.append("Superposition tasks must have positive coherence time")
            
            probability_amplitude = task_context.get('probability_amplitude', 1.0)
            if not 0.0 <= probability_amplitude <= 1.0:
                errors.append("Probability amplitude must be between 0 and 1")
        
        severity = self._determine_severity(errors, warnings)
        
        return QuantumValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            severity=severity,
            context={"operation": "state_transition", "from": current_state.value, "to": target_state.value}
        )
    
    def validate_entanglement_creation(
        self,
        task_id1: str,
        task_id2: str,
        entanglement_type: EntanglementType,
        correlation_strength: float,
        existing_entanglements: Optional[Dict[Tuple[str, str], Any]] = None
    ) -> QuantumValidationResult:
        """Validate entanglement creation."""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Task ID validation
        if not self._is_valid_task_id(task_id1):
            errors.append(f"Invalid task ID: {task_id1}")
        
        if not self._is_valid_task_id(task_id2):
            errors.append(f"Invalid task ID: {task_id2}")
        
        if task_id1 == task_id2:
            errors.append("Cannot entangle task with itself")
        
        # Correlation strength validation
        if not 0.0 <= correlation_strength <= 1.0:
            errors.append("Correlation strength must be between 0.0 and 1.0")
        elif correlation_strength < 0.1:
            warnings.append("Very low correlation strength may not provide meaningful entanglement")
        elif correlation_strength > 0.95 and entanglement_type != EntanglementType.BELL_STATE:
            warnings.append("Very high correlation strength may cause over-coupling")
        
        # Entanglement type validation
        if entanglement_type == EntanglementType.BELL_STATE and correlation_strength < 0.8:
            warnings.append("Bell states typically require high correlation strength (>0.8)")
        
        if entanglement_type == EntanglementType.ANTI_CORRELATED and correlation_strength > 0.5:
            warnings.append("Anti-correlated entanglement with high positive correlation may be contradictory")
        
        # Check for existing entanglement
        if existing_entanglements:
            bond_key1 = tuple(sorted([task_id1, task_id2]))
            bond_key2 = tuple(sorted([task_id2, task_id1]))
            
            if bond_key1 in existing_entanglements or bond_key2 in existing_entanglements:
                errors.append("Entanglement already exists between these tasks")
            
            # Check for excessive entanglements
            task1_entanglements = sum(1 for key in existing_entanglements.keys() if task_id1 in key)
            task2_entanglements = sum(1 for key in existing_entanglements.keys() if task_id2 in key)
            
            if task1_entanglements > 10:
                warnings.append(f"Task {task_id1} has many entanglements ({task1_entanglements})")
            
            if task2_entanglements > 10:
                warnings.append(f"Task {task_id2} has many entanglements ({task2_entanglements})")
        
        # Suggestions
        if correlation_strength == 1.0:
            suggestions.append("Perfect correlation (1.0) is rare in quantum systems")
        
        if entanglement_type == EntanglementType.GHZ_STATE:
            suggestions.append("GHZ states work best with multiple tasks - consider creating a cluster")
        
        severity = self._determine_severity(errors, warnings)
        
        return QuantumValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            severity=severity,
            context={
                "operation": "entanglement_creation",
                "task_ids": [task_id1, task_id2],
                "entanglement_type": entanglement_type.value
            }
        )
    
    def validate_superposition_probabilities(
        self,
        state_probabilities: Dict[TaskState, float]
    ) -> QuantumValidationResult:
        """Validate quantum superposition probability distribution."""
        
        errors = []
        warnings = []
        suggestions = []
        
        if not state_probabilities:
            errors.append("Superposition must have at least one state")
            return QuantumValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                severity=ValidationSeverity.ERROR
            )
        
        # Check probability values
        for state, probability in state_probabilities.items():
            if not isinstance(probability, (int, float)):
                errors.append(f"Probability for state {state.value} must be numeric")
            elif probability < 0.0:
                errors.append(f"Probability for state {state.value} cannot be negative")
            elif probability > 1.0:
                errors.append(f"Probability for state {state.value} cannot exceed 1.0")
        
        # Check normalization
        total_probability = sum(state_probabilities.values())
        if abs(total_probability - 1.0) > 1e-6:
            if total_probability == 0.0:
                errors.append("Total probability cannot be zero")
            else:
                warnings.append(f"Probabilities should sum to 1.0 (current: {total_probability:.6f})")
        
        # Check for meaningful superposition
        non_zero_states = sum(1 for p in state_probabilities.values() if p > 1e-6)
        if non_zero_states == 1:
            warnings.append("Superposition with only one non-zero state is effectively collapsed")
        
        # Check for uniform distribution
        if len(state_probabilities) > 1:
            expected_uniform = 1.0 / len(state_probabilities)
            if all(abs(p - expected_uniform) < 1e-3 for p in state_probabilities.values()):
                suggestions.append("Uniform distribution detected - consider if this represents true superposition")
        
        # Check for extreme probabilities
        for state, prob in state_probabilities.items():
            if prob > 0.99:
                warnings.append(f"State {state.value} has very high probability ({prob:.3f})")
        
        severity = self._determine_severity(errors, warnings)
        
        return QuantumValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            severity=severity,
            context={"operation": "superposition_validation", "total_probability": total_probability}
        )
    
    def validate_quantum_gate_operation(
        self,
        task_id: str,
        gate_type: str,
        current_superposition: Dict[TaskState, float]
    ) -> QuantumValidationResult:
        """Validate quantum gate operation."""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Task ID validation
        if not self._is_valid_task_id(task_id):
            errors.append(f"Invalid task ID: {task_id}")
        
        # Gate type validation
        valid_gates = ["hadamard", "h", "pauli_x", "x", "pauli_y", "y", "pauli_z", "z", "phase", "p", "t"]
        if gate_type.lower() not in valid_gates:
            errors.append(f"Unknown quantum gate type: {gate_type}")
        
        # Superposition validation
        superposition_result = self.validate_superposition_probabilities(current_superposition)
        if not superposition_result.is_valid:
            errors.extend(superposition_result.errors)
        
        # Gate-specific warnings
        if gate_type.lower() in ["hadamard", "h"]:
            if len(current_superposition) < 2:
                warnings.append("Hadamard gate requires at least 2 states for meaningful effect")
        
        if gate_type.lower() in ["pauli_x", "x"]:
            if TaskState.COMPLETED not in current_superposition or TaskState.FAILED not in current_superposition:
                warnings.append("Pauli-X gate works best with complementary states (completed/failed)")
        
        # Check if gate operation is meaningful
        max_prob = max(current_superposition.values()) if current_superposition else 0
        if max_prob > 0.99:
            warnings.append("Gate operation on near-collapsed state may have minimal effect")
        
        severity = self._determine_severity(errors, warnings)
        
        return QuantumValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            severity=severity,
            context={"operation": "quantum_gate", "gate_type": gate_type, "task_id": task_id}
        )
    
    def validate_system_limits(
        self,
        current_task_count: int,
        current_entanglement_count: int,
        operation: str
    ) -> QuantumValidationResult:
        """Validate system resource limits."""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Task count limits
        if operation == "create_task" and current_task_count >= self.max_tasks:
            errors.append(f"Maximum task limit reached ({self.max_tasks})")
        elif current_task_count > self.max_tasks * 0.9:
            warnings.append(f"Approaching maximum task limit ({current_task_count}/{self.max_tasks})")
        
        # Entanglement count limits
        if operation == "create_entanglement" and current_entanglement_count >= self.max_entanglements:
            errors.append(f"Maximum entanglement limit reached ({self.max_entanglements})")
        elif current_entanglement_count > self.max_entanglements * 0.9:
            warnings.append(f"Approaching maximum entanglement limit ({current_entanglement_count}/{self.max_entanglements})")
        
        # Performance warnings
        if current_task_count > 100 and current_entanglement_count > current_task_count * 2:
            warnings.append("High entanglement density may impact performance")
        
        # Suggestions
        if current_task_count > self.max_tasks * 0.8:
            suggestions.append("Consider archiving or removing completed tasks")
        
        if current_entanglement_count > self.max_entanglements * 0.8:
            suggestions.append("Consider breaking some entanglements or using clusters")
        
        severity = self._determine_severity(errors, warnings)
        
        return QuantumValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            severity=severity,
            context={
                "operation": "system_limits",
                "current_tasks": current_task_count,
                "current_entanglements": current_entanglement_count
            }
        )
    
    def validate_temporal_constraints(
        self,
        task_due_date: Optional[datetime],
        task_dependencies: Set[str],
        dependency_completion_times: Dict[str, Optional[datetime]]
    ) -> QuantumValidationResult:
        """Validate temporal constraints and dependencies."""
        
        errors = []
        warnings = []
        suggestions = []
        
        if not task_due_date:
            return QuantumValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                severity=ValidationSeverity.INFO
            )
        
        # Check dependency timing
        for dep_id in task_dependencies:
            dep_completion = dependency_completion_times.get(dep_id)
            
            if dep_completion and dep_completion >= task_due_date:
                errors.append(f"Dependency {dep_id} completes after task due date")
            elif dep_completion and (task_due_date - dep_completion).total_seconds() < 3600:
                warnings.append(f"Very tight timing with dependency {dep_id}")
        
        # Check overall timing feasibility
        current_time = datetime.utcnow()
        time_until_due = task_due_date - current_time
        
        if time_until_due.total_seconds() < 0:
            errors.append("Task due date is in the past")
        elif time_until_due.total_seconds() < 3600:
            warnings.append("Less than 1 hour until due date")
        
        # Suggestions
        if len(task_dependencies) > 5 and time_until_due.total_seconds() < 86400:
            suggestions.append("Consider extending due date or reducing dependencies for complex task")
        
        severity = self._determine_severity(errors, warnings)
        
        return QuantumValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            severity=severity,
            context={"operation": "temporal_validation"}
        )
    
    def _is_valid_task_id(self, task_id: str) -> bool:
        """Validate task ID format."""
        if not isinstance(task_id, str):
            return False
        
        # UUID format validation
        uuid_pattern = re.compile(
            r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        )
        
        return bool(uuid_pattern.match(task_id))
    
    def _determine_severity(self, errors: List[str], warnings: List[str]) -> ValidationSeverity:
        """Determine overall validation severity."""
        if errors:
            # Check for critical errors
            critical_keywords = ["limit reached", "cannot", "invalid transition"]
            if any(keyword in error.lower() for error in errors for keyword in critical_keywords):
                return ValidationSeverity.CRITICAL
            return ValidationSeverity.ERROR
        elif warnings:
            return ValidationSeverity.WARNING
        else:
            return ValidationSeverity.INFO
    
    def create_validation_report(
        self,
        validation_results: List[QuantumValidationResult]
    ) -> Dict[str, Any]:
        """Create comprehensive validation report."""
        
        total_errors = sum(len(result.errors) for result in validation_results)
        total_warnings = sum(len(result.warnings) for result in validation_results)
        total_suggestions = sum(len(result.suggestions) for result in validation_results)
        
        # Severity distribution
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = sum(
                1 for result in validation_results if result.severity == severity
            )
        
        # Most common issues
        all_errors = []
        all_warnings = []
        for result in validation_results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        return {
            "summary": {
                "total_validations": len(validation_results),
                "passed_validations": sum(1 for r in validation_results if r.is_valid),
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "total_suggestions": total_suggestions
            },
            "severity_distribution": severity_counts,
            "validation_results": [
                {
                    "is_valid": result.is_valid,
                    "severity": result.severity.value,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "suggestions": result.suggestions,
                    "context": result.context
                }
                for result in validation_results
            ],
            "recommendations": self._generate_recommendations(all_errors, all_warnings)
        }
    
    def _generate_recommendations(
        self,
        errors: List[str],
        warnings: List[str]
    ) -> List[str]:
        """Generate system-level recommendations based on validation results."""
        
        recommendations = []
        
        # Error-based recommendations
        if any("limit reached" in error for error in errors):
            recommendations.append("Consider implementing task archiving or cleanup procedures")
        
        if any("invalid" in error.lower() for error in errors):
            recommendations.append("Review input validation and data sanitization procedures")
        
        # Warning-based recommendations
        if any("many" in warning.lower() for warning in warnings):
            recommendations.append("Consider implementing automatic task organization features")
        
        if any("performance" in warning.lower() for warning in warnings):
            recommendations.append("Monitor system performance and consider optimization")
        
        return recommendations