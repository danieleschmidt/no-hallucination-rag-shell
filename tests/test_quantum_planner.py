"""
Comprehensive tests for quantum task planner.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from no_hallucination_rag.quantum.quantum_planner import (
    QuantumTaskPlanner, QuantumTask, TaskState, Priority
)
from no_hallucination_rag.quantum.superposition_tasks import SuperpositionTaskManager
from no_hallucination_rag.quantum.entanglement_dependencies import (
    EntanglementDependencyGraph, EntanglementType
)
from no_hallucination_rag.quantum.quantum_validator import QuantumValidator
from no_hallucination_rag.quantum.quantum_security import QuantumSecurityManager, QuantumSecurityLevel
from no_hallucination_rag.quantum.quantum_logging import QuantumLogger


class TestQuantumTaskPlanner:
    """Test cases for QuantumTaskPlanner."""
    
    @pytest.fixture
    def planner(self):
        """Create quantum task planner instance."""
        return QuantumTaskPlanner(max_coherence_time=3600.0)
    
    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "title": "Test Quantum Task",
            "description": "A test task for quantum planning",
            "priority": Priority.SECOND_EXCITED,
            "due_date": datetime.utcnow() + timedelta(days=1),
            "dependencies": set(),
            "tags": {"quantum", "test"},
            "estimated_duration": timedelta(hours=2)
        }
    
    def test_create_task(self, planner, sample_task_data):
        """Test task creation."""
        task = planner.create_task(**sample_task_data)
        
        assert task.title == sample_task_data["title"]
        assert task.description == sample_task_data["description"]
        assert task.priority == sample_task_data["priority"]
        assert task.state == TaskState.SUPERPOSITION
        assert task.probability_amplitude == 1.0
        assert task.is_coherent()
        assert task.id in planner.tasks
    
    def test_observe_task_collapse(self, planner, sample_task_data):
        """Test task observation and state collapse."""
        task = planner.create_task(**sample_task_data)
        original_state = task.state
        
        # Observe task multiple times to test collapse
        observed_task = planner.observe_task(task.id)
        
        assert observed_task is not None
        assert observed_task.state != TaskState.SUPERPOSITION
        assert observed_task.state in [TaskState.COLLAPSED, TaskState.ENTANGLED, TaskState.FAILED]
    
    def test_entangle_tasks(self, planner, sample_task_data):
        """Test task entanglement."""
        # Create two tasks
        task1 = planner.create_task(title="Task 1", **{k: v for k, v in sample_task_data.items() if k != "title"})
        task2 = planner.create_task(title="Task 2", **{k: v for k, v in sample_task_data.items() if k != "title"})
        
        # Entangle tasks
        success = planner.entangle_tasks(task1.id, task2.id, 0.8)
        
        assert success
        assert task2.id in task1.entangled_tasks
        assert task1.id in task2.entangled_tasks
        assert (task1.id, task2.id) in planner.interference_matrix or (task2.id, task1.id) in planner.interference_matrix
    
    def test_entangle_same_task_fails(self, planner, sample_task_data):
        """Test that entangling task with itself fails."""
        task = planner.create_task(**sample_task_data)
        
        success = planner.entangle_tasks(task.id, task.id)
        
        assert not success
    
    def test_get_optimal_sequence(self, planner, sample_task_data):
        """Test optimal task sequence generation."""
        # Create multiple tasks
        tasks = []
        for i in range(5):
            task_data = sample_task_data.copy()
            task_data["title"] = f"Task {i}"
            task_data["priority"] = Priority.FIRST_EXCITED if i < 2 else Priority.GROUND_STATE
            tasks.append(planner.create_task(**task_data))
        
        # Observe some tasks to make them available
        for task in tasks[:3]:
            planner.observe_task(task.id)
        
        # Get optimal sequence
        available_time = timedelta(hours=6)
        optimal_tasks = planner.get_optimal_sequence(available_time)
        
        assert len(optimal_tasks) <= 3  # Should fit within time limit
        
        # Higher priority tasks should be first
        if len(optimal_tasks) > 1:
            for i in range(len(optimal_tasks) - 1):
                assert optimal_tasks[i].priority.value >= optimal_tasks[i + 1].priority.value
    
    def test_execute_task_sequence(self, planner, sample_task_data):
        """Test task sequence execution."""
        # Create and observe a task
        task = planner.create_task(**sample_task_data)
        planner.observe_task(task.id)
        
        # Execute sequence
        execution_results = planner.execute_task_sequence([task])
        
        assert execution_results["started_at"] is not None
        assert execution_results["completed_at"] is not None
        assert len(execution_results["tasks_executed"]) == 1
        assert task.state == TaskState.COMPLETED
        assert task.progress == 1.0
    
    def test_quantum_interference_calculation(self, planner, sample_task_data):
        """Test quantum interference calculation."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task_data = sample_task_data.copy()
            task_data["title"] = f"Task {i}"
            tasks.append(planner.create_task(**task_data))
        
        # Entangle some tasks
        planner.entangle_tasks(tasks[0].id, tasks[1].id, 0.8)
        planner.entangle_tasks(tasks[1].id, tasks[2].id, 0.6)
        
        # Calculate interference
        interference_scores = planner.calculate_task_interference()
        
        assert len(interference_scores) == len(tasks)
        for task_id, score in interference_scores.items():
            assert isinstance(score, float)
    
    def test_dependencies_satisfied(self, planner, sample_task_data):
        """Test dependency checking."""
        # Create dependency task
        dep_task_data = sample_task_data.copy()
        dep_task_data["title"] = "Dependency Task"
        dep_task = planner.create_task(**dep_task_data)
        
        # Create main task with dependency
        main_task_data = sample_task_data.copy()
        main_task_data["title"] = "Main Task"
        main_task_data["dependencies"] = {dep_task.id}
        main_task = planner.create_task(**main_task_data)
        
        # Dependencies not satisfied initially
        assert not planner._dependencies_satisfied(main_task.id)
        
        # Complete dependency
        dep_task.state = TaskState.COMPLETED
        
        # Dependencies now satisfied
        assert planner._dependencies_satisfied(main_task.id)
    
    def test_decohere_expired_tasks(self, planner, sample_task_data):
        """Test decoherence of expired tasks."""
        # Create task with very short coherence time
        task = planner.create_task(**sample_task_data)
        task.coherence_time = 0.1  # 0.1 seconds
        task.created_at = datetime.utcnow() - timedelta(seconds=1)  # Created 1 second ago
        
        # Force decoherence check
        planner._decohere_expired_tasks()
        
        # Task should no longer be coherent
        assert not task.is_coherent()
        assert task.state != TaskState.SUPERPOSITION
    
    def test_quantum_state_summary(self, planner, sample_task_data):
        """Test quantum state summary generation."""
        # Create tasks in different states
        task1 = planner.create_task(title="Task 1", **{k: v for k, v in sample_task_data.items() if k != "title"})
        task2 = planner.create_task(title="Task 2", **{k: v for k, v in sample_task_data.items() if k != "title"})
        
        # Change states
        planner.observe_task(task1.id)
        task2.state = TaskState.COMPLETED
        
        # Entangle tasks
        planner.entangle_tasks(task1.id, task2.id)
        
        # Get summary
        summary = planner.get_quantum_state_summary()
        
        assert "total_tasks" in summary
        assert "state_distribution" in summary
        assert "coherent_tasks" in summary
        assert "entanglement_pairs" in summary
        assert "quantum_coherence_ratio" in summary
        
        assert summary["total_tasks"] == 2
    
    def test_delete_task_cleanup(self, planner, sample_task_data):
        """Test task deletion and cleanup."""
        # Create two tasks and entangle them
        task1 = planner.create_task(title="Task 1", **{k: v for k, v in sample_task_data.items() if k != "title"})
        task2 = planner.create_task(title="Task 2", **{k: v for k, v in sample_task_data.items() if k != "title"})
        
        planner.entangle_tasks(task1.id, task2.id)
        
        # Delete first task
        success = planner.delete_task(task1.id)
        
        assert success
        assert task1.id not in planner.tasks
        assert task1.id not in task2.entangled_tasks
        assert task1.id not in planner.entanglement_graph
    
    def test_export_quantum_state(self, planner, sample_task_data):
        """Test quantum state export."""
        # Create task and entanglement
        task1 = planner.create_task(title="Task 1", **{k: v for k, v in sample_task_data.items() if k != "title"})
        task2 = planner.create_task(title="Task 2", **{k: v for k, v in sample_task_data.items() if k != "title"})
        
        planner.entangle_tasks(task1.id, task2.id)
        
        # Export state
        exported_state = planner.export_quantum_state()
        
        assert "tasks" in exported_state
        assert "interference_matrix" in exported_state
        assert "entanglement_graph" in exported_state
        assert "quantum_summary" in exported_state
        assert "exported_at" in exported_state
        
        assert len(exported_state["tasks"]) == 2


class TestSuperpositionTaskManager:
    """Test cases for SuperpositionTaskManager."""
    
    @pytest.fixture
    def manager(self):
        """Create superposition task manager."""
        return SuperpositionTaskManager()
    
    def test_create_superposition(self, manager):
        """Test superposition creation."""
        task_id = "test-task-123"
        initial_states = {
            TaskState.SUPERPOSITION: 0.6,
            TaskState.COLLAPSED: 0.3,
            TaskState.ENTANGLED: 0.1
        }
        
        superposition = manager.create_superposition(task_id, initial_states)
        
        assert superposition.task_id == task_id
        assert len(superposition.state_probabilities) == 3
        
        # Check normalization
        total_prob = sum(superposition.state_probabilities.values())
        assert abs(total_prob - 1.0) < 1e-6
    
    def test_measure_superposition(self, manager):
        """Test superposition measurement and collapse."""
        task_id = "test-task-123"
        initial_states = {
            TaskState.SUPERPOSITION: 0.7,
            TaskState.COLLAPSED: 0.3
        }
        
        manager.create_superposition(task_id, initial_states)
        
        # Measure superposition
        collapsed_state = manager.measure_superposition(task_id)
        
        assert collapsed_state in [TaskState.SUPERPOSITION, TaskState.COLLAPSED]
        
        # After measurement, should be collapsed to single state
        superposition = manager.superpositions[task_id]
        assert superposition.state_probabilities[collapsed_state] == 1.0
    
    def test_superposition_entropy(self, manager):
        """Test superposition entropy calculation."""
        task_id = "test-task-123"
        
        # Maximum entropy (uniform distribution)
        uniform_states = {
            TaskState.SUPERPOSITION: 0.25,
            TaskState.COLLAPSED: 0.25,
            TaskState.ENTANGLED: 0.25,
            TaskState.COMPLETED: 0.25
        }
        
        manager.create_superposition(task_id, uniform_states)
        entropy = manager.calculate_superposition_entropy(task_id)
        
        # Maximum entropy for 4 states is log2(4) = 2
        assert abs(entropy - 2.0) < 0.01
        
        # Minimum entropy (pure state)
        pure_states = {TaskState.COMPLETED: 1.0}
        manager.superpositions[task_id].state_probabilities = pure_states
        
        entropy = manager.calculate_superposition_entropy(task_id)
        assert entropy == 0.0
    
    def test_superposition_purity(self, manager):
        """Test superposition purity calculation."""
        task_id = "test-task-123"
        
        # Pure state should have purity = 1
        pure_states = {TaskState.COMPLETED: 1.0}
        manager.create_superposition(task_id, pure_states)
        
        purity = manager.get_superposition_purity(task_id)
        assert purity == 1.0
        
        # Mixed state should have purity < 1
        mixed_states = {
            TaskState.COMPLETED: 0.5,
            TaskState.FAILED: 0.5
        }
        manager.superpositions[task_id].state_probabilities = mixed_states
        
        purity = manager.get_superposition_purity(task_id)
        assert purity == 0.5  # For equal mixture: 0.5^2 + 0.5^2 = 0.5
    
    def test_create_bell_state(self, manager):
        """Test Bell state creation."""
        task_id1 = "task-1"
        task_id2 = "task-2"
        
        success = manager.create_bell_state(task_id1, task_id2)
        
        assert success
        assert task_id1 in manager.superpositions
        assert task_id2 in manager.superpositions
        
        # Both tasks should have same probability distribution
        sp1 = manager.superpositions[task_id1]
        sp2 = manager.superpositions[task_id2]
        
        assert sp1.state_probabilities[TaskState.COMPLETED] == 0.5
        assert sp1.state_probabilities[TaskState.FAILED] == 0.5
        assert sp2.state_probabilities[TaskState.COMPLETED] == 0.5
        assert sp2.state_probabilities[TaskState.FAILED] == 0.5
    
    def test_quantum_gate_application(self, manager):
        """Test quantum gate application."""
        task_id = "test-task-123"
        initial_states = {
            TaskState.COMPLETED: 1.0
        }
        
        manager.create_superposition(task_id, initial_states)
        
        # Define a simple bit-flip gate
        def pauli_x_gate(state_probs):
            return {
                TaskState.FAILED: state_probs.get(TaskState.COMPLETED, 0),
                TaskState.COMPLETED: state_probs.get(TaskState.FAILED, 0)
            }
        
        success = manager.apply_quantum_gate(task_id, pauli_x_gate)
        
        assert success
        
        superposition = manager.superpositions[task_id]
        assert superposition.state_probabilities[TaskState.FAILED] == 1.0
        assert superposition.state_probabilities.get(TaskState.COMPLETED, 0) == 0.0
    
    def test_superposition_evolution(self, manager):
        """Test quantum superposition time evolution."""
        task_id = "test-task-123"
        initial_states = {
            TaskState.SUPERPOSITION: 0.8,
            TaskState.COLLAPSED: 0.2
        }
        
        manager.create_superposition(task_id, initial_states)
        original_probs = manager.superpositions[task_id].state_probabilities.copy()
        
        # Evolve superposition
        success = manager.evolve_superposition(task_id, 1.0)
        
        assert success
        
        # Probabilities should have changed (slightly) due to evolution
        new_probs = manager.superpositions[task_id].state_probabilities
        assert new_probs != original_probs
        
        # But should still be normalized
        total_prob = sum(new_probs.values())
        assert abs(total_prob - 1.0) < 1e-6


class TestEntanglementDependencyGraph:
    """Test cases for EntanglementDependencyGraph."""
    
    @pytest.fixture
    def graph(self):
        """Create entanglement dependency graph."""
        return EntanglementDependencyGraph()
    
    def test_create_entanglement(self, graph):
        """Test entanglement creation."""
        task_id1 = "task-1"
        task_id2 = "task-2"
        
        success = graph.create_entanglement(
            task_id1, task_id2,
            EntanglementType.SPIN_CORRELATED, 0.8
        )
        
        assert success
        assert graph.graph.has_edge(task_id1, task_id2)
        
        bond_key = tuple(sorted([task_id1, task_id2]))
        assert bond_key in graph.entanglement_bonds
        
        bond = graph.entanglement_bonds[bond_key]
        assert bond.entanglement_type == EntanglementType.SPIN_CORRELATED
        assert bond.correlation_strength == 0.8
    
    def test_measure_entangled_state(self, graph):
        """Test entangled state measurement."""
        task_id1 = "task-1"
        task_id2 = "task-2"
        
        # Create spin-correlated entanglement
        graph.create_entanglement(
            task_id1, task_id2,
            EntanglementType.SPIN_CORRELATED, 0.9
        )
        
        # Measure one task
        correlations = graph.measure_entangled_state(task_id1, TaskState.COMPLETED)
        
        assert len(correlations) == 1
        correlated_task, predicted_state, probability = correlations[0]
        
        assert correlated_task == task_id2
        assert predicted_state == TaskState.COMPLETED  # Spin-correlated should match
        assert probability == 0.9
    
    def test_anti_correlated_entanglement(self, graph):
        """Test anti-correlated entanglement."""
        task_id1 = "task-1"
        task_id2 = "task-2"
        
        # Create anti-correlated entanglement
        graph.create_entanglement(
            task_id1, task_id2,
            EntanglementType.ANTI_CORRELATED, 0.8
        )
        
        # Measure one task as completed
        correlations = graph.measure_entangled_state(task_id1, TaskState.COMPLETED)
        
        assert len(correlations) == 1
        correlated_task, predicted_state, probability = correlations[0]
        
        assert correlated_task == task_id2
        assert predicted_state == TaskState.FAILED  # Anti-correlated should be opposite
        assert probability == 0.8
    
    def test_ghz_state_creation(self, graph):
        """Test GHZ state creation."""
        task_ids = ["task-1", "task-2", "task-3"]
        
        success = graph.create_ghz_state(task_ids)
        
        assert success
        
        # Should have created pairwise entanglements
        expected_pairs = [
            ("task-1", "task-2"),
            ("task-1", "task-3"),
            ("task-2", "task-3")
        ]
        
        for task_id1, task_id2 in expected_pairs:
            assert graph.graph.has_edge(task_id1, task_id2)
        
        # Should have created a cluster
        assert len(graph.clusters) > 0
        ghz_cluster = None
        for cluster_tasks in graph.clusters.values():
            if set(task_ids).issubset(cluster_tasks):
                ghz_cluster = cluster_tasks
                break
        
        assert ghz_cluster is not None
    
    def test_cluster_state_creation(self, graph):
        """Test cluster state creation."""
        task_graph = [
            ("task-1", "task-2"),
            ("task-2", "task-3"),
            ("task-3", "task-4"),
            ("task-1", "task-4")
        ]
        
        success = graph.create_cluster_state(task_graph)
        
        assert success
        
        # All specified connections should exist
        for task_id1, task_id2 in task_graph:
            assert graph.graph.has_edge(task_id1, task_id2)
        
        # Should have created a cluster
        assert len(graph.clusters) > 0
    
    def test_break_entanglement(self, graph):
        """Test entanglement breaking."""
        task_id1 = "task-1"
        task_id2 = "task-2"
        
        # Create entanglement
        graph.create_entanglement(task_id1, task_id2, EntanglementType.SPIN_CORRELATED, 0.8)
        
        # Break entanglement
        success = graph.break_entanglement(task_id1, task_id2)
        
        assert success
        assert not graph.graph.has_edge(task_id1, task_id2)
        
        bond_key = tuple(sorted([task_id1, task_id2]))
        assert bond_key not in graph.entanglement_bonds
    
    def test_entanglement_path_finding(self, graph):
        """Test finding entanglement paths."""
        # Create chain: task1 - task2 - task3
        graph.create_entanglement("task-1", "task-2", EntanglementType.SPIN_CORRELATED, 0.8)
        graph.create_entanglement("task-2", "task-3", EntanglementType.SPIN_CORRELATED, 0.8)
        
        # Find path
        path = graph.find_entanglement_path("task-1", "task-3")
        
        assert path is not None
        assert len(path) == 3
        assert path[0] == "task-1"
        assert path[-1] == "task-3"
        assert "task-2" in path
    
    def test_bell_violation_detection(self, graph):
        """Test Bell inequality violation detection."""
        task_id1 = "task-1"
        task_id2 = "task-2"
        
        # Create entanglement and simulate measurements
        graph.create_entanglement(task_id1, task_id2, EntanglementType.BELL_STATE, 0.95)
        
        # Increase measurement count
        bond_key = tuple(sorted([task_id1, task_id2]))
        graph.entanglement_bonds[bond_key].measurement_count = 10
        
        # Detect Bell violation
        bell_parameter = graph.detect_bell_violations(task_id1, task_id2)
        
        assert bell_parameter is not None
        assert abs(bell_parameter) > 2.0  # Should violate classical bound
    
    def test_entanglement_statistics(self, graph):
        """Test entanglement statistics generation."""
        # Create several entanglements
        graph.create_entanglement("task-1", "task-2", EntanglementType.SPIN_CORRELATED, 0.8)
        graph.create_entanglement("task-2", "task-3", EntanglementType.ANTI_CORRELATED, 0.7)
        graph.create_entanglement("task-3", "task-4", EntanglementType.BELL_STATE, 0.9)
        
        stats = graph.get_entanglement_statistics()
        
        assert stats["total_tasks"] == 4
        assert stats["total_entanglements"] == 3
        assert stats["network_density"] > 0
        assert stats["average_degree"] > 0
        assert "entanglement_types" in stats
        assert "average_correlation_strength" in stats


class TestQuantumValidator:
    """Test cases for QuantumValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create quantum validator."""
        return QuantumValidator()
    
    def test_validate_task_creation_valid(self, validator):
        """Test valid task creation validation."""
        result = validator.validate_task_creation(
            title="Valid Test Task",
            description="A valid test task for quantum planning",
            priority=Priority.SECOND_EXCITED,
            due_date=datetime.utcnow() + timedelta(days=1),
            dependencies=set(),
            tags={"test", "quantum"},
            estimated_duration=timedelta(hours=2)
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_task_creation_invalid_title(self, validator):
        """Test invalid task title validation."""
        result = validator.validate_task_creation(
            title="",  # Empty title
            description="Valid description"
        )
        
        assert not result.is_valid
        assert any("title cannot be empty" in error for error in result.errors)
    
    def test_validate_quantum_state_transition_valid(self, validator):
        """Test valid state transition."""
        result = validator.validate_quantum_state_transition(
            TaskState.SUPERPOSITION,
            TaskState.COLLAPSED
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_quantum_state_transition_invalid(self, validator):
        """Test invalid state transition."""
        result = validator.validate_quantum_state_transition(
            TaskState.COMPLETED,  # Terminal state
            TaskState.SUPERPOSITION  # Cannot transition back
        )
        
        assert not result.is_valid
        assert any("Invalid state transition" in error for error in result.errors)
    
    def test_validate_superposition_probabilities_normalized(self, validator):
        """Test normalized probability validation."""
        probabilities = {
            TaskState.COMPLETED: 0.6,
            TaskState.FAILED: 0.4
        }
        
        result = validator.validate_superposition_probabilities(probabilities)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_superposition_probabilities_unnormalized(self, validator):
        """Test unnormalized probability validation."""
        probabilities = {
            TaskState.COMPLETED: 0.8,
            TaskState.FAILED: 0.5  # Total = 1.3 > 1.0
        }
        
        result = validator.validate_superposition_probabilities(probabilities)
        
        assert not result.is_valid or len(result.warnings) > 0
    
    def test_validate_entanglement_creation_valid(self, validator):
        """Test valid entanglement creation."""
        # Mock valid task IDs (UUIDs)
        task_id1 = "12345678-1234-1234-1234-123456789abc"
        task_id2 = "87654321-4321-4321-4321-cba987654321"
        
        result = validator.validate_entanglement_creation(
            task_id1, task_id2,
            EntanglementType.SPIN_CORRELATED, 0.8
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_entanglement_creation_same_task(self, validator):
        """Test entanglement with same task validation."""
        task_id = "12345678-1234-1234-1234-123456789abc"
        
        result = validator.validate_entanglement_creation(
            task_id, task_id,
            EntanglementType.SPIN_CORRELATED, 0.8
        )
        
        assert not result.is_valid
        assert any("Cannot entangle task with itself" in error for error in result.errors)


class TestQuantumSecurityManager:
    """Test cases for QuantumSecurityManager."""
    
    @pytest.fixture
    def security_manager(self):
        """Create quantum security manager."""
        return QuantumSecurityManager(QuantumSecurityLevel.PROTECTED)
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return QuantumTask(
            title="Test Task",
            description="Test description",
            state=TaskState.SUPERPOSITION,
            priority=Priority.SECOND_EXCITED
        )
    
    def test_secure_task_creation_authorized(self, security_manager, sample_task):
        """Test authorized task creation."""
        user_id = "test-user"
        user_roles = ["user"]
        client_context = {"client_ip": "192.168.1.100"}
        
        success, message, encrypted_task = security_manager.secure_task_creation(
            sample_task, user_id, user_roles, client_context
        )
        
        assert success
        assert "successfully" in message
        assert encrypted_task is not None
    
    def test_secure_task_creation_unauthorized(self, security_manager, sample_task):
        """Test unauthorized task creation."""
        user_id = "test-user"
        user_roles = ["guest"]  # Insufficient role
        client_context = {"client_ip": "192.168.1.100"}
        
        success, message, encrypted_task = security_manager.secure_task_creation(
            sample_task, user_id, user_roles, client_context
        )
        
        assert not success
        assert "Insufficient role privileges" in message
        assert encrypted_task is None
    
    def test_rate_limiting(self, security_manager, sample_task):
        """Test rate limiting functionality."""
        user_id = "test-user"
        user_roles = ["user"]
        client_context = {"client_ip": "192.168.1.100"}
        
        # Make requests up to the limit
        success_count = 0
        for i in range(105):  # Limit is 100 per hour
            success, message, _ = security_manager.secure_task_creation(
                sample_task, user_id, user_roles, client_context
            )
            if success:
                success_count += 1
            else:
                break
        
        # Should hit rate limit before 105 requests
        assert success_count <= 100
    
    def test_generate_secure_task_id(self, security_manager):
        """Test secure task ID generation."""
        task_id = security_manager.generate_secure_task_id()
        
        assert isinstance(task_id, str)
        assert len(task_id) == 36  # UUID format length
        assert task_id.count('-') == 4  # UUID has 4 hyphens
        
        # Generate multiple IDs to test uniqueness
        task_ids = [security_manager.generate_secure_task_id() for _ in range(10)]
        assert len(set(task_ids)) == 10  # All should be unique
    
    def test_security_event_logging(self, security_manager, sample_task):
        """Test security event logging."""
        user_id = "malicious-user"
        user_roles = ["guest"]
        client_context = {"client_ip": "suspicious-ip"}
        
        # Trigger security event
        security_manager.secure_task_creation(sample_task, user_id, user_roles, client_context)
        
        # Check that security event was logged
        assert len(security_manager.security_events) > 0
        
        # Find the relevant event
        event = list(security_manager.security_events.values())[0]
        assert event.user_id == user_id


class TestQuantumLogger:
    """Test cases for QuantumLogger."""
    
    @pytest.fixture
    def logger(self):
        """Create quantum logger."""
        return QuantumLogger(log_level=logging.INFO)
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for logging."""
        return QuantumTask(
            title="Test Task",
            description="Test description",
            state=TaskState.SUPERPOSITION,
            priority=Priority.SECOND_EXCITED
        )
    
    def test_log_task_created(self, logger, sample_task):
        """Test task creation logging."""
        initial_log_count = len(logger.quantum_logs)
        
        logger.log_task_created(
            sample_task,
            user_id="test-user",
            session_id="test-session"
        )
        
        assert len(logger.quantum_logs) == initial_log_count + 1
        
        log_entry = logger.quantum_logs[-1]
        assert log_entry.event_type.value == "task_created"
        assert log_entry.task_id == sample_task.id
        assert log_entry.user_id == "test-user"
    
    def test_log_task_observed(self, logger, sample_task):
        """Test task observation logging."""
        logger.log_task_observed(
            sample_task,
            TaskState.COLLAPSED,
            user_id="test-user"
        )
        
        assert len(logger.quantum_logs) == 1
        
        log_entry = logger.quantum_logs[0]
        assert log_entry.event_type.value == "task_observed"
        assert log_entry.event_data["collapsed_state"] == TaskState.COLLAPSED.value
    
    def test_log_quantum_gate_applied(self, logger):
        """Test quantum gate logging."""
        task_id = "test-task-123"
        before_state = {TaskState.COMPLETED: 1.0}
        after_state = {TaskState.FAILED: 1.0}
        
        logger.log_quantum_gate_applied(
            task_id, "pauli_x", before_state, after_state,
            user_id="test-user", execution_time=0.1
        )
        
        assert len(logger.quantum_logs) == 1
        
        log_entry = logger.quantum_logs[0]
        assert log_entry.event_type.value == "quantum_gate_applied"
        assert log_entry.event_data["gate_type"] == "pauli_x"
        assert "entropy_before" in log_entry.event_data
        assert "entropy_after" in log_entry.event_data
    
    def test_performance_summary(self, logger, sample_task):
        """Test performance summary generation."""
        # Generate some log entries
        logger.log_task_created(sample_task, performance_metrics={"execution_time": 0.1})
        logger.log_task_observed(sample_task, TaskState.COLLAPSED, observation_time=0.05)
        
        summary = logger.get_performance_summary(time_window_hours=1)
        
        assert "total_events" in summary
        assert "event_distribution" in summary
        assert "average_operation_times" in summary
        assert "quantum_statistics" in summary
        assert summary["total_events"] == 2
    
    def test_audit_trail(self, logger, sample_task):
        """Test audit trail functionality."""
        logger.log_task_created(sample_task, user_id="test-user")
        logger.log_task_observed(sample_task, TaskState.COLLAPSED, user_id="test-user")
        
        audit_trail = logger.get_audit_trail(task_id=sample_task.id)
        
        assert len(audit_trail) == 2
        assert all(entry["task_id"] == sample_task.id for entry in audit_trail)
        assert all(entry["user_id"] == "test-user" for entry in audit_trail)
    
    def test_log_cleanup(self, logger, sample_task):
        """Test log cleanup functionality."""
        # Create old log entries by manipulating timestamps
        logger.log_task_created(sample_task)
        
        # Manually set old timestamp
        logger.quantum_logs[0].timestamp = datetime.utcnow() - timedelta(days=40)
        
        initial_count = len(logger.quantum_logs)
        removed_count = logger.cleanup_logs(older_than_days=30)
        
        assert removed_count == 1
        assert len(logger.quantum_logs) == initial_count - removed_count


# Integration tests
class TestQuantumSystemIntegration:
    """Integration tests for the complete quantum system."""
    
    @pytest.fixture
    def quantum_system(self):
        """Create integrated quantum system."""
        planner = QuantumTaskPlanner()
        superposition_manager = SuperpositionTaskManager()
        entanglement_graph = EntanglementDependencyGraph()
        validator = QuantumValidator()
        security_manager = QuantumSecurityManager()
        logger = QuantumLogger()
        
        return {
            "planner": planner,
            "superposition": superposition_manager,
            "entanglement": entanglement_graph,
            "validator": validator,
            "security": security_manager,
            "logger": logger
        }
    
    def test_end_to_end_task_workflow(self, quantum_system):
        """Test complete task workflow from creation to completion."""
        planner = quantum_system["planner"]
        superposition = quantum_system["superposition"]
        logger = quantum_system["logger"]
        
        # Create task
        task = planner.create_task(
            title="Integration Test Task",
            description="End-to-end test",
            priority=Priority.SECOND_EXCITED
        )
        
        # Log creation
        logger.log_task_created(task)
        
        # Create superposition
        initial_states = {
            TaskState.SUPERPOSITION: 0.7,
            TaskState.COLLAPSED: 0.3
        }
        superposition.create_superposition(task.id, initial_states)
        
        # Observe task
        observed_task = planner.observe_task(task.id)
        collapsed_state = superposition.measure_superposition(task.id)
        
        # Log observation
        logger.log_task_observed(observed_task, collapsed_state)
        
        # Execute task
        if observed_task.state in [TaskState.COLLAPSED, TaskState.ENTANGLED]:
            execution_results = planner.execute_task_sequence([observed_task])
            assert len(execution_results["tasks_executed"]) == 1
        
        # Verify logging
        assert len(logger.quantum_logs) >= 2
        
        # Verify final state
        assert task.state == TaskState.COMPLETED
    
    def test_quantum_entanglement_effects(self, quantum_system):
        """Test entanglement effects across the system."""
        planner = quantum_system["planner"]
        entanglement = quantum_system["entanglement"]
        superposition = quantum_system["superposition"]
        
        # Create two tasks
        task1 = planner.create_task(title="Entangled Task 1")
        task2 = planner.create_task(title="Entangled Task 2")
        
        # Create superpositions
        initial_states = {TaskState.COMPLETED: 0.5, TaskState.FAILED: 0.5}
        superposition.create_superposition(task1.id, initial_states)
        superposition.create_superposition(task2.id, initial_states)
        
        # Create entanglement
        planner.entangle_tasks(task1.id, task2.id, 0.9)
        entanglement.create_entanglement(
            task1.id, task2.id,
            EntanglementType.SPIN_CORRELATED, 0.9
        )
        
        # Observe first task
        planner.observe_task(task1.id)
        collapsed_state1 = superposition.measure_superposition(task1.id)
        
        # Check entanglement correlation
        correlations = entanglement.measure_entangled_state(task1.id, collapsed_state1)
        
        assert len(correlations) == 1
        predicted_task, predicted_state, probability = correlations[0]
        assert predicted_task == task2.id
        assert probability == 0.9
    
    def test_security_and_validation_integration(self, quantum_system):
        """Test security and validation working together."""
        validator = quantum_system["validator"]
        security = quantum_system["security"]
        planner = quantum_system["planner"]
        
        # Test valid task creation with security
        validation_result = validator.validate_task_creation(
            title="Secure Test Task",
            description="Security test",
            priority=Priority.SECOND_EXCITED
        )
        
        assert validation_result.is_valid
        
        # Create task through planner
        task = planner.create_task(title="Secure Test Task")
        
        # Test secure access
        success, message, encrypted_task = security.secure_task_creation(
            task, "test-user", ["user"], {"client_ip": "192.168.1.100"}
        )
        
        assert success
        assert encrypted_task is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_quantum_operations(self, quantum_system):
        """Test concurrent quantum operations."""
        planner = quantum_system["planner"]
        
        # Create multiple tasks concurrently
        tasks = []
        for i in range(5):
            task = planner.create_task(title=f"Concurrent Task {i}")
            tasks.append(task)
        
        # Observe tasks concurrently
        observed_tasks = []
        for task in tasks:
            observed_task = planner.observe_task(task.id)
            observed_tasks.append(observed_task)
        
        # All tasks should be observed
        assert len(observed_tasks) == 5
        assert all(task is not None for task in observed_tasks)
        
        # Execute optimal sequence
        available_time = timedelta(hours=10)
        optimal_sequence = planner.get_optimal_task_sequence(available_time)
        
        if optimal_sequence:
            execution_results = planner.execute_task_sequence(optimal_sequence)
            assert len(execution_results["tasks_executed"]) > 0