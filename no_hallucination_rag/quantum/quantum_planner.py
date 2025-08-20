"""
Quantum-inspired task planner with superposition and entanglement concepts.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import json
import math
import random


class TaskState(Enum):
    """Task states inspired by quantum mechanics."""
    SUPERPOSITION = "superposition"  # Task exists in multiple possible states
    COLLAPSED = "collapsed"          # Task state has been observed/measured
    ENTANGLED = "entangled"         # Task depends on other tasks
    COMPLETED = "completed"         # Task is finished
    FAILED = "failed"              # Task failed


class Priority(Enum):
    """Task priority levels using quantum energy levels."""
    GROUND_STATE = 1    # Lowest energy/priority
    FIRST_EXCITED = 2   # Low priority
    SECOND_EXCITED = 3  # Medium priority  
    THIRD_EXCITED = 4   # High priority
    IONIZED = 5        # Critical priority


@dataclass
class QuantumTask:
    """A task that exists in quantum superposition until observed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    state: TaskState = TaskState.SUPERPOSITION
    priority: Priority = Priority.GROUND_STATE
    
    # Quantum properties
    probability_amplitude: float = 1.0  # Probability of task existing
    coherence_time: float = 3600.0     # How long task stays in superposition (seconds)
    entangled_tasks: Set[str] = field(default_factory=set)
    
    # Traditional properties
    created_at: datetime = field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    
    # Execution properties
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
    
    def quantum_interference(self, other_task: 'QuantumTask') -> float:
        """Calculate quantum interference with another task."""
        if not (self.is_coherent() and other_task.is_coherent()):
            return 0.0
        
        # Simplified interference calculation
        phase_difference = abs(hash(self.id) - hash(other_task.id)) % (2 * math.pi)
        return self.probability_amplitude * other_task.probability_amplitude * math.cos(phase_difference)


class QuantumTaskPlanner:
    """
    Quantum-inspired task planner using superposition, entanglement, and interference.
    
    Key quantum concepts applied:
    - Superposition: Tasks exist in multiple states until observed
    - Entanglement: Tasks become correlated and affect each other
    - Interference: Tasks can constructively or destructively interfere
    - Measurement: Observing tasks collapses their superposition
    """
    
    def __init__(self, max_coherence_time: float = 7200.0, max_superposition_tasks: Optional[int] = None):
        self.tasks: Dict[str, QuantumTask] = {}
        self.max_coherence_time = max_coherence_time
        self.max_superposition_tasks = max_superposition_tasks or 50  # Default limit
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Quantum state tracking
        self.interference_matrix: Dict[Tuple[str, str], float] = {}
        self.entanglement_graph: Dict[str, Set[str]] = {}
        
        self.logger.info("Quantum Task Planner initialized")
    
    def create_task(
        self,
        title: str,
        description: str = "",
        priority: Priority = Priority.GROUND_STATE,
        due_date: Optional[datetime] = None,
        dependencies: Optional[Set[str]] = None,
        tags: Optional[Set[str]] = None,
        estimated_duration: Optional[timedelta] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QuantumTask:
        """Create a new quantum task in superposition."""
        
        task = QuantumTask(
            title=title,
            description=description,
            priority=priority,
            due_date=due_date,
            dependencies=dependencies or set(),
            tags=tags or set(),
            estimated_duration=estimated_duration or timedelta(hours=1),
            context=context or {}
        )
        
        # Set coherence time based on priority (higher priority = longer coherence)
        task.coherence_time = min(
            self.max_coherence_time,
            3600.0 * priority.value
        )
        
        self.tasks[task.id] = task
        self.entanglement_graph[task.id] = set()
        
        self.logger.info(f"Created quantum task: {task.title} (ID: {task.id[:8]})")
        
        return task
    
    def add_task(self, task: QuantumTask) -> bool:
        """Add an existing quantum task to the planner."""
        if task.id in self.tasks:
            return False
        
        self.tasks[task.id] = task
        if task.id not in self.entanglement_graph:
            self.entanglement_graph[task.id] = set()
        self.logger.info(f"Added existing quantum task: {task.title} (ID: {task.id[:8]})")
        return True
    
    def observe_task(self, task_id: str) -> Optional[QuantumTask]:
        """Observe a task, collapsing its superposition."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        if task.state == TaskState.SUPERPOSITION:
            # Collapse to most probable state based on context
            if task.dependencies and not self._dependencies_satisfied(task_id):
                task.collapse_state(TaskState.ENTANGLED)
            else:
                # Use quantum probability to determine collapsed state
                probability = task.probability_amplitude ** 2
                if random.random() < probability:
                    task.collapse_state(TaskState.COLLAPSED)
                else:
                    task.collapse_state(TaskState.FAILED)
            
            self.logger.info(f"Task {task.title} collapsed to state: {task.state.value}")
        
        return task
    
    def entangle_tasks(self, task_id1: str, task_id2: str, correlation_strength: float = 1.0) -> bool:
        """Create quantum entanglement between two tasks."""
        task1 = self.tasks.get(task_id1)
        task2 = self.tasks.get(task_id2)
        
        if not (task1 and task2):
            return False
        
        # Create bidirectional entanglement
        task1.entangled_tasks.add(task_id2)
        task2.entangled_tasks.add(task_id1)
        
        self.entanglement_graph[task_id1].add(task_id2)
        self.entanglement_graph[task_id2].add(task_id1)
        
        # Update interference matrix
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
                
                interference = task.quantum_interference(other_task)
                
                # Apply entanglement correlation if exists
                correlation = self.interference_matrix.get((task_id, other_id), 0.0)
                total_interference += interference * correlation
            
            interference_scores[task_id] = total_interference
        
        return interference_scores
    
    def get_optimal_task_sequence(self, available_time: timedelta) -> List[QuantumTask]:
        """Use quantum optimization to find optimal task sequence."""
        
        # First, collapse any tasks that have exceeded coherence time
        self._decohere_expired_tasks()
        
        # Get observable tasks (collapsed or about to decohere)
        observable_tasks = []
        for task in self.tasks.values():
            if (task.state in [TaskState.COLLAPSED, TaskState.ENTANGLED] or 
                not task.is_coherent()):
                observable_tasks.append(task)
        
        if not observable_tasks:
            return []
        
        # Calculate interference effects
        interference_scores = self.calculate_task_interference()
        
        # Sort by quantum-inspired priority function
        def quantum_priority_score(task: QuantumTask) -> float:
            base_priority = task.priority.value
            interference_boost = interference_scores.get(task.id, 0.0)
            
            # Time pressure factor
            time_factor = 1.0
            if task.due_date:
                days_left = (task.due_date - datetime.utcnow()).days
                time_factor = max(0.1, 1.0 / max(1, days_left))
            
            # Entanglement factor (entangled tasks get priority boost)
            entanglement_factor = 1.0 + 0.1 * len(task.entangled_tasks)
            
            return base_priority * time_factor * entanglement_factor + interference_boost
        
        # Sort tasks by quantum priority
        prioritized_tasks = sorted(
            observable_tasks,
            key=quantum_priority_score,
            reverse=True
        )
        
        # Select tasks that fit within available time
        selected_tasks = []
        remaining_time = available_time
        
        for task in prioritized_tasks:
            if (task.estimated_duration <= remaining_time and 
                self._dependencies_satisfied(task.id)):
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
                # Simulate task execution
                self.logger.info(f"Executing task: {task.title}")
                
                # Check for quantum entanglement effects
                self._check_entanglement_effects(task, execution_results)
                
                # Simulate work (in real implementation, this would call actual task logic)
                execution_time = min(task.estimated_duration.total_seconds(), 10)  # Cap simulation time
                time.sleep(execution_time / 100)  # Accelerated simulation
                
                # Mark task as completed
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
                
                self.logger.info(f"Completed task: {task.title} in {task_duration}")
                
            except Exception as e:
                task.state = TaskState.FAILED
                execution_results['tasks_failed'].append({
                    'task_id': task.id,
                    'title': task.title,
                    'error': str(e)
                })
                
                self.logger.error(f"Task failed: {task.title} - {e}")
        
        execution_results['completed_at'] = datetime.utcnow()
        execution_results['total_wall_time'] = timedelta(seconds=time.time() - start_time)
        
        return execution_results
    
    def _check_entanglement_effects(self, task: QuantumTask, results: Dict[str, Any]) -> None:
        """Check and apply quantum entanglement effects."""
        
        for entangled_id in task.entangled_tasks:
            entangled_task = self.tasks.get(entangled_id)
            if not entangled_task:
                continue
            
            # Entangled tasks affect each other's probability amplitudes
            if entangled_task.state == TaskState.SUPERPOSITION:
                correlation = self.interference_matrix.get((task.id, entangled_id), 0.5)
                
                if task.state == TaskState.COMPLETED:
                    # Successful task completion boosts entangled task probability
                    entangled_task.probability_amplitude = min(1.0, 
                        entangled_task.probability_amplitude + 0.1 * correlation)
                elif task.state == TaskState.FAILED:
                    # Failed task reduces entangled task probability
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
                # Force measurement/collapse
                task.collapse_state(TaskState.COLLAPSED)
                self.logger.info(f"Task {task.title} decohered and collapsed")
    
    def get_quantum_state_summary(self) -> Dict[str, Any]:
        """Get summary of quantum states across all tasks."""
        
        state_counts = {}
        for state in TaskState:
            state_counts[state.value] = 0
        
        coherent_tasks = 0
        total_interference = 0.0
        
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
    
    def get_task_by_id(self, task_id: str) -> Optional[QuantumTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self, filter_state: Optional[TaskState] = None) -> List[QuantumTask]:
        """List all tasks, optionally filtered by state."""
        if filter_state:
            return [task for task in self.tasks.values() if task.state == filter_state]
        return list(self.tasks.values())
    
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
            
            # Clean up interference matrix
            self.interference_matrix.pop((task_id, entangled_id), None)
            self.interference_matrix.pop((entangled_id, task_id), None)
        
        # Remove from entanglement graph
        self.entanglement_graph.pop(task_id, None)
        for task_set in self.entanglement_graph.values():
            task_set.discard(task_id)
        
        # Remove task
        del self.tasks[task_id]
        
        self.logger.info(f"Deleted task: {task.title}")
        
        return True
    
    def export_quantum_state(self) -> Dict[str, Any]:
        """Export complete quantum state for analysis or persistence."""
        
        tasks_data = {}
        for task_id, task in self.tasks.items():
            tasks_data[task_id] = {
                'title': task.title,
                'description': task.description,
                'state': task.state.value,
                'priority': task.priority.value,
                'probability_amplitude': task.probability_amplitude,
                'coherence_time': task.coherence_time,
                'entangled_tasks': list(task.entangled_tasks),
                'created_at': task.created_at.isoformat(),
                'due_date': task.due_date.isoformat() if task.due_date else None,
                'dependencies': list(task.dependencies),
                'tags': list(task.tags),
                'estimated_duration': task.estimated_duration.total_seconds(),
                'progress': task.progress,
                'context': task.context
            }
        
        return {
            'tasks': tasks_data,
            'interference_matrix': {
                f"{k[0]},{k[1]}": v for k, v in self.interference_matrix.items()
            },
            'entanglement_graph': {
                k: list(v) for k, v in self.entanglement_graph.items()
            },
            'quantum_summary': self.get_quantum_state_summary(),
            'exported_at': datetime.utcnow().isoformat()
        }
    
    def shutdown(self) -> None:
        """Shutdown the quantum task planner."""
        
        self.logger.info("Shutting down Quantum Task Planner...")
        self.executor.shutdown(wait=True)
        self.logger.info("Quantum Task Planner shutdown complete")