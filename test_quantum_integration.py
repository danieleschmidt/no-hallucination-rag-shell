"""
Standalone integration test for quantum task planning system.
"""

import uuid
import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import json


class TaskState(Enum):
    """Task states inspired by quantum mechanics."""
    SUPERPOSITION = "superposition"
    COLLAPSED = "collapsed"
    ENTANGLED = "entangled"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(Enum):
    """Task priority levels using quantum energy levels."""
    GROUND_STATE = 1
    FIRST_EXCITED = 2
    SECOND_EXCITED = 3
    THIRD_EXCITED = 4
    IONIZED = 5


class EntanglementType(Enum):
    """Types of quantum entanglement between tasks."""
    BELL_STATE = "bell_state"
    SPIN_CORRELATED = "spin_correlated"
    ANTI_CORRELATED = "anti_correlated"
    GHZ_STATE = "ghz_state"
    CLUSTER_STATE = "cluster_state"


@dataclass
class QuantumTask:
    """A task that exists in quantum superposition until observed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    state: TaskState = TaskState.SUPERPOSITION
    priority: Priority = Priority.GROUND_STATE
    
    probability_amplitude: float = 1.0
    coherence_time: float = 3600.0
    entangled_tasks: Set[str] = field(default_factory=set)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    progress: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def collapse_state(self, observed_state: TaskState) -> None:
        """Collapse the quantum superposition to a definite state."""
        if self.state == TaskState.SUPERPOSITION:
            self.state = observed_state
            self.probability_amplitude = 1.0 if observed_state != TaskState.FAILED else 0.0
    
    def is_coherent(self) -> bool:
        """Check if task is still in coherent superposition."""
        if self.state != TaskState.SUPERPOSITION:
            return False
        
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed < self.coherence_time


@dataclass
class TaskSuperposition:
    """Represents a task existing in multiple states with different probabilities."""
    task_id: str
    state_probabilities: Dict[TaskState, float]
    coherence_coefficient: float = 1.0
    last_measurement: Optional[datetime] = None


@dataclass
class EntanglementBond:
    """Represents an entanglement relationship between tasks."""
    task_id1: str
    task_id2: str
    entanglement_type: EntanglementType
    correlation_strength: float
    created_at: datetime
    last_interaction: Optional[datetime] = None
    measurement_count: int = 0


class QuantumTaskPlanner:
    """Main quantum task planner."""
    
    def __init__(self, max_coherence_time: float = 7200.0):
        self.tasks: Dict[str, QuantumTask] = {}
        self.max_coherence_time = max_coherence_time
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.interference_matrix: Dict[Tuple[str, str], float] = {}
        self.entanglement_graph: Dict[str, Set[str]] = {}
        
        self.logger.info("Quantum Task Planner initialized")
    
    def create_task(self, title: str, description: str = "", priority: Priority = Priority.GROUND_STATE, **kwargs) -> QuantumTask:
        """Create a new quantum task."""
        task = QuantumTask(title=title, description=description, priority=priority, **kwargs)
        task.coherence_time = min(self.max_coherence_time, 3600.0 * priority.value)
        
        self.tasks[task.id] = task
        self.entanglement_graph[task.id] = set()
        
        self.logger.info(f"Created quantum task: {task.title}")
        return task
    
    def observe_task(self, task_id: str) -> Optional[QuantumTask]:
        """Observe a task, collapsing its superposition."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        if task.state == TaskState.SUPERPOSITION:
            if task.dependencies and not self._dependencies_satisfied(task_id):
                task.collapse_state(TaskState.ENTANGLED)
            else:
                probability = task.probability_amplitude ** 2
                if np.random.random() < probability:
                    task.collapse_state(TaskState.COLLAPSED)
                else:
                    task.collapse_state(TaskState.FAILED)
            
            self.logger.info(f"Task {task.title} collapsed to state: {task.state.value}")
        
        return task
    
    def entangle_tasks(self, task_id1: str, task_id2: str, correlation_strength: float = 1.0) -> bool:
        """Create quantum entanglement between two tasks."""
        task1 = self.tasks.get(task_id1)
        task2 = self.tasks.get(task_id2)
        
        if not (task1 and task2) or task_id1 == task_id2:
            return False
        
        task1.entangled_tasks.add(task_id2)
        task2.entangled_tasks.add(task_id1)
        
        self.entanglement_graph[task_id1].add(task_id2)
        self.entanglement_graph[task_id2].add(task_id1)
        
        self.interference_matrix[(task_id1, task_id2)] = correlation_strength
        self.interference_matrix[(task_id2, task_id1)] = correlation_strength
        
        self.logger.info(f"Entangled tasks: {task1.title} ↔ {task2.title}")
        return True
    
    def calculate_task_interference(self) -> Dict[str, float]:
        """Calculate quantum interference effects for all tasks."""
        interference_scores = {}
        
        for task_id, task in self.tasks.items():
            if not task.is_coherent():
                interference_scores[task_id] = 0.0
                continue
            
            total_interference = 0.0
            for other_id, other_task in self.tasks.items():
                if task_id == other_id or not other_task.is_coherent():
                    continue
                
                # Simplified interference calculation
                phase_difference = abs(hash(task_id) - hash(other_id)) % (2 * np.pi)
                interference = task.probability_amplitude * other_task.probability_amplitude * np.cos(phase_difference)
                
                correlation = self.interference_matrix.get((task_id, other_id), 0.0)
                total_interference += interference * correlation
            
            interference_scores[task_id] = total_interference
        
        return interference_scores
    
    def get_optimal_task_sequence(self, available_time: timedelta) -> List[QuantumTask]:
        """Use quantum optimization to find optimal task sequence."""
        self._decohere_expired_tasks()
        
        observable_tasks = []
        for task in self.tasks.values():
            if (task.state in [TaskState.COLLAPSED, TaskState.ENTANGLED] or not task.is_coherent()):
                observable_tasks.append(task)
        
        if not observable_tasks:
            return []
        
        interference_scores = self.calculate_task_interference()
        
        def quantum_priority_score(task: QuantumTask) -> float:
            base_priority = task.priority.value
            interference_boost = interference_scores.get(task.id, 0.0)
            
            time_factor = 1.0
            if task.due_date:
                days_left = (task.due_date - datetime.utcnow()).days
                time_factor = max(0.1, 1.0 / max(1, days_left))
            
            entanglement_factor = 1.0 + 0.1 * len(task.entangled_tasks)
            
            return base_priority * time_factor * entanglement_factor + interference_boost
        
        prioritized_tasks = sorted(observable_tasks, key=quantum_priority_score, reverse=True)
        
        selected_tasks = []
        remaining_time = available_time
        
        for task in prioritized_tasks:
            if (task.estimated_duration <= remaining_time and self._dependencies_satisfied(task.id)):
                selected_tasks.append(task)
                remaining_time -= task.estimated_duration
        
        self.logger.info(f"Quantum optimization selected {len(selected_tasks)} tasks")
        return selected_tasks
    
    def execute_task_sequence(self, tasks: List[QuantumTask]) -> Dict[str, Any]:
        """Execute a sequence of tasks with quantum state tracking."""
        execution_results = {
            'started_at': datetime.utcnow(),
            'tasks_executed': [],
            'tasks_failed': [],
            'quantum_effects_observed': [],
            'total_duration': timedelta()
        }
        
        start_time = time.time()
        
        for task in tasks:
            task_start = time.time()
            
            try:
                self.logger.info(f"Executing task: {task.title}")
                self._check_entanglement_effects(task, execution_results)
                
                execution_time = min(task.estimated_duration.total_seconds(), 2)
                time.sleep(execution_time / 100)  # Accelerated simulation
                
                task.state = TaskState.COMPLETED
                task.progress = 1.0
                
                task_duration = timedelta(seconds=time.time() - task_start)
                execution_results['tasks_executed'].append({
                    'task_id': task.id,
                    'title': task.title,
                    'duration': task_duration,
                    'quantum_state': task.state.value
                })
                
                execution_results['total_duration'] += task_duration
                
            except Exception as e:
                task.state = TaskState.FAILED
                execution_results['tasks_failed'].append({
                    'task_id': task.id,
                    'title': task.title,
                    'error': str(e)
                })
        
        execution_results['completed_at'] = datetime.utcnow()
        execution_results['total_wall_time'] = timedelta(seconds=time.time() - start_time)
        
        return execution_results
    
    def _check_entanglement_effects(self, task: QuantumTask, results: Dict[str, Any]) -> None:
        """Check and apply quantum entanglement effects."""
        for entangled_id in task.entangled_tasks:
            entangled_task = self.tasks.get(entangled_id)
            if not entangled_task:
                continue
            
            if entangled_task.state == TaskState.SUPERPOSITION:
                correlation = self.interference_matrix.get((task.id, entangled_id), 0.5)
                
                if task.state == TaskState.COMPLETED:
                    entangled_task.probability_amplitude = min(1.0, 
                        entangled_task.probability_amplitude + 0.1 * correlation)
                elif task.state == TaskState.FAILED:
                    entangled_task.probability_amplitude = max(0.1,
                        entangled_task.probability_amplitude - 0.1 * correlation)
                
                results['quantum_effects_observed'].append({
                    'type': 'entanglement_correlation',
                    'source_task': task.title,
                    'affected_task': entangled_task.title,
                    'correlation_strength': correlation,
                    'new_amplitude': entangled_task.probability_amplitude
                })
    
    def _dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all task dependencies are satisfied."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.state != TaskState.COMPLETED:
                return False
        
        return True
    
    def _decohere_expired_tasks(self) -> None:
        """Collapse tasks that have exceeded their coherence time."""
        for task in self.tasks.values():
            if task.state == TaskState.SUPERPOSITION and not task.is_coherent():
                task.collapse_state(TaskState.COLLAPSED)
                self.logger.info(f"Task {task.title} decohered and collapsed")
    
    def get_quantum_state_summary(self) -> Dict[str, Any]:
        """Get summary of quantum states across all tasks."""
        state_counts = {state.value: 0 for state in TaskState}
        coherent_tasks = 0
        
        for task in self.tasks.values():
            state_counts[task.state.value] += 1
            if task.is_coherent():
                coherent_tasks += 1
        
        interference_scores = self.calculate_task_interference()
        total_interference = sum(abs(score) for score in interference_scores.values())
        
        return {
            'total_tasks': len(self.tasks),
            'state_distribution': state_counts,
            'coherent_tasks': coherent_tasks,
            'entanglement_pairs': len(self.interference_matrix) // 2,
            'total_quantum_interference': total_interference,
            'quantum_coherence_ratio': coherent_tasks / max(1, len(self.tasks))
        }
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task and clean up entanglements."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Clean up entanglements
        for entangled_id in task.entangled_tasks:
            entangled_task = self.tasks.get(entangled_id)
            if entangled_task:
                entangled_task.entangled_tasks.discard(task_id)
            
            self.interference_matrix.pop((task_id, entangled_id), None)
            self.interference_matrix.pop((entangled_id, task_id), None)
        
        self.entanglement_graph.pop(task_id, None)
        for task_set in self.entanglement_graph.values():
            task_set.discard(task_id)
        
        del self.tasks[task_id]
        return True


class SuperpositionTaskManager:
    """Manages tasks in quantum superposition states."""
    
    def __init__(self):
        self.superpositions: Dict[str, TaskSuperposition] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_superposition(self, task_id: str, initial_states: Dict[TaskState, float]) -> TaskSuperposition:
        """Create a new task superposition with given state probabilities."""
        total_prob = sum(initial_states.values())
        if total_prob == 0:
            raise ValueError("Total probability cannot be zero")
        
        normalized_states = {state: prob / total_prob for state, prob in initial_states.items()}
        
        superposition = TaskSuperposition(task_id=task_id, state_probabilities=normalized_states)
        self.superpositions[task_id] = superposition
        
        return superposition
    
    def measure_superposition(self, task_id: str) -> Optional[TaskState]:
        """Measure/observe a task superposition, collapsing it to a single state."""
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return None
        
        states = list(superposition.state_probabilities.keys())
        probabilities = list(superposition.state_probabilities.values())
        
        collapsed_state = np.random.choice(states, p=probabilities)
        
        superposition.last_measurement = datetime.utcnow()
        superposition.state_probabilities = {collapsed_state: 1.0}
        
        return collapsed_state
    
    def calculate_superposition_entropy(self, task_id: str) -> float:
        """Calculate von Neumann entropy of task superposition."""
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return 0.0
        
        entropy = 0.0
        for probability in superposition.state_probabilities.values():
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def get_superposition_purity(self, task_id: str) -> float:
        """Calculate purity of the superposition state."""
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return 0.0
        
        purity = sum(p**2 for p in superposition.state_probabilities.values())
        return purity


class EntanglementDependencyGraph:
    """Graph-based system for managing quantum entanglement between tasks."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entanglement_bonds: Dict[Tuple[str, str], EntanglementBond] = {}
        self.clusters: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_entanglement(self, task_id1: str, task_id2: str, entanglement_type: EntanglementType, correlation_strength: float) -> bool:
        """Create quantum entanglement between two tasks."""
        if task_id1 == task_id2:
            return False
        
        bond_key = tuple(sorted([task_id1, task_id2]))
        if bond_key in self.entanglement_bonds:
            return False
        
        bond = EntanglementBond(
            task_id1=task_id1, task_id2=task_id2,
            entanglement_type=entanglement_type,
            correlation_strength=correlation_strength,
            created_at=datetime.utcnow()
        )
        
        self.graph.add_edge(task_id1, task_id2, bond=bond, weight=correlation_strength)
        self.entanglement_bonds[bond_key] = bond
        
        return True
    
    def detect_bell_violations(self, task_id1: str, task_id2: str) -> Optional[float]:
        """Detect violations of Bell inequalities."""
        bond_key = tuple(sorted([task_id1, task_id2]))
        bond = self.entanglement_bonds.get(bond_key)
        
        if not bond or bond.measurement_count < 4:
            return None
        
        correlation_coefficient = bond.correlation_strength
        if bond.entanglement_type == EntanglementType.ANTI_CORRELATED:
            correlation_coefficient *= -1
        
        S = 2 * np.sqrt(2) * abs(correlation_coefficient)
        
        if abs(S) > 2:
            self.logger.info(f"Bell violation detected: S = {S:.3f}")
        
        return S
    
    def get_entanglement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the entanglement network."""
        if not self.graph.nodes():
            return {"total_tasks": 0}
        
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph) if num_nodes > 1 else 0.0
        
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        try:
            clustering = nx.average_clustering(self.graph)
        except:
            clustering = 0.0
        
        type_counts = {}
        for bond in self.entanglement_bonds.values():
            etype = bond.entanglement_type.value
            type_counts[etype] = type_counts.get(etype, 0) + 1
        
        strengths = [bond.correlation_strength for bond in self.entanglement_bonds.values()]
        avg_strength = np.mean(strengths) if strengths else 0.0
        
        return {
            "total_tasks": num_nodes,
            "total_entanglements": num_edges,
            "network_density": density,
            "average_degree": avg_degree,
            "max_degree": max_degree,
            "clustering_coefficient": clustering,
            "total_clusters": len(self.clusters),
            "entanglement_types": type_counts,
            "average_correlation_strength": avg_strength,
            "highly_entangled_tasks": sum(1 for d in degrees if d >= 3)
        }


def run_comprehensive_quantum_test():
    """Run comprehensive quantum system test."""
    print('🧪 COMPREHENSIVE QUANTUM SYSTEM TEST')
    print('=' * 50)
    
    try:
        # Initialize components
        planner = QuantumTaskPlanner()
        superposition_manager = SuperpositionTaskManager()
        entanglement_graph = EntanglementDependencyGraph()
        
        print('✅ Components initialized')
        
        # Create test tasks
        task1 = planner.create_task('Primary Analysis Task', priority=Priority.THIRD_EXCITED)
        task2 = planner.create_task('Secondary Processing Task', priority=Priority.SECOND_EXCITED)
        task3 = planner.create_task('Support Task', priority=Priority.FIRST_EXCITED)
        
        print(f'✅ Created {len(planner.tasks)} quantum tasks')
        
        # Test superposition
        initial_states = {
            TaskState.SUPERPOSITION: 0.6,
            TaskState.COLLAPSED: 0.3,
            TaskState.ENTANGLED: 0.1
        }
        
        for task_id in [task1.id, task2.id, task3.id]:
            superposition_manager.create_superposition(task_id, initial_states)
        
        entropy1 = superposition_manager.calculate_superposition_entropy(task1.id)
        purity1 = superposition_manager.get_superposition_purity(task1.id)
        
        print(f'✅ Superposition entropy: {entropy1:.3f}, purity: {purity1:.3f}')
        
        # Test entanglement network
        entanglements = [
            (task1.id, task2.id, EntanglementType.SPIN_CORRELATED, 0.85),
            (task2.id, task3.id, EntanglementType.ANTI_CORRELATED, 0.70),
            (task1.id, task3.id, EntanglementType.BELL_STATE, 0.95)
        ]
        
        for t1, t2, etype, strength in entanglements:
            success = entanglement_graph.create_entanglement(t1, t2, etype, strength)
            planner.entangle_tasks(t1, t2, strength)
            print(f'✅ Entanglement created: {success}')
        
        # Test quantum interference
        interference_scores = planner.calculate_task_interference()
        total_interference = sum(abs(score) for score in interference_scores.values())
        print(f'✅ Total quantum interference: {total_interference:.3f}')
        
        # Test task observation and collapse
        observed_states = []
        for task in [task1, task2, task3]:
            observed_task = planner.observe_task(task.id)
            collapsed_state = superposition_manager.measure_superposition(task.id)
            observed_states.append((observed_task.state, collapsed_state))
            print(f'✅ Task {task.title[:15]}... observed: {observed_task.state.value} -> {collapsed_state.value if collapsed_state else None}')
        
        # Test optimal sequence
        sequence = planner.get_optimal_task_sequence(timedelta(hours=6))
        print(f'✅ Optimal sequence: {len(sequence)} tasks selected')
        
        # Test Bell violations
        bell_violations = []
        for task_id1, task_id2, _, _ in entanglements:
            bond_key = tuple(sorted([task_id1, task_id2]))
            if bond_key in entanglement_graph.entanglement_bonds:
                entanglement_graph.entanglement_bonds[bond_key].measurement_count = 10
                bell_param = entanglement_graph.detect_bell_violations(task_id1, task_id2)
                if bell_param is not None:
                    bell_violations.append(bell_param)
                    print(f'✅ Bell parameter: {bell_param:.3f} (violation: {abs(bell_param) > 2.0})')
        
        # Test execution
        if sequence:
            execution_results = planner.execute_task_sequence(sequence)
            print(f'✅ Execution: {len(execution_results["tasks_executed"])} completed')
            print(f'✅ Quantum effects: {len(execution_results["quantum_effects_observed"])} observed')
        
        # Test system statistics
        quantum_summary = planner.get_quantum_state_summary()
        entanglement_stats = entanglement_graph.get_entanglement_statistics()
        
        print('\n📊 SYSTEM STATISTICS')
        print(f'  Total tasks: {quantum_summary["total_tasks"]}')
        print(f'  Coherent tasks: {quantum_summary["coherent_tasks"]}')
        print(f'  Entanglement pairs: {quantum_summary["entanglement_pairs"]}')
        print(f'  Network density: {entanglement_stats["network_density"]:.3f}')
        print(f'  Average degree: {entanglement_stats["average_degree"]:.3f}')
        print(f'  Clustering coefficient: {entanglement_stats["clustering_coefficient"]:.3f}')
        print(f'  Bell violations: {len([b for b in bell_violations if abs(b) > 2.0])}')
        
        # Performance validation
        print('\n🎯 PERFORMANCE VALIDATION')
        completed_tasks = sum(1 for t in planner.tasks.values() if t.state == TaskState.COMPLETED)
        success_rate = completed_tasks / len(planner.tasks) if planner.tasks else 0
        
        print(f'  Task completion rate: {success_rate:.1%}')
        print(f'  Quantum coherence ratio: {quantum_summary["quantum_coherence_ratio"]:.3f}')
        print(f'  System complexity: {entanglement_stats["total_entanglements"]} bonds')
        
        # Quality gates
        quality_checks = {
            'Basic functionality': len(planner.tasks) >= 3,
            'Superposition created': entropy1 > 0.0,
            'Entanglement network': entanglement_stats['total_entanglements'] >= 3,
            'Quantum interference': total_interference > 0.0,
            'State observation': len(observed_states) == 3,
            'Bell violations': len(bell_violations) > 0,
            'Task execution': completed_tasks > 0,
            'Performance metrics': success_rate > 0.5
        }
        
        print(f'\n🛡️ QUALITY GATES ({sum(quality_checks.values())}/{len(quality_checks)} passed)')
        for check, passed in quality_checks.items():
            status = '✅' if passed else '❌'
            print(f'  {status} {check}')
        
        all_passed = all(quality_checks.values())
        
        print(f'\n🎯 COMPREHENSIVE TEST: {"PASSED" if all_passed else "FAILED"}')
        
        return all_passed
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_quantum_test()
    exit(0 if success else 1)