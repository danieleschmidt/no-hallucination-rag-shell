"""
Superposition task manager for handling tasks in multiple states simultaneously.
"""

import logging
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta

from .quantum_planner import QuantumTask, TaskState, Priority


@dataclass
class TaskSuperposition:
    """Represents a task existing in multiple states with different probabilities."""
    task_id: str
    state_probabilities: Dict[TaskState, float]
    coherence_coefficient: float = 1.0
    last_measurement: Optional[datetime] = None


class SuperpositionTaskManager:
    """
    Manages tasks that exist in quantum superposition states.
    
    Allows tasks to exist in multiple states simultaneously until measured/observed.
    """
    
    def __init__(self):
        self.superpositions: Dict[str, TaskSuperposition] = {}
        self.logger = logging.getLogger(__name__)
        
    def create_superposition(
        self, 
        initial_states: List[str] = None,
        task_id: str = None, 
        state_probs: Dict[TaskState, float] = None
    ) -> TaskSuperposition:
        """Create a new task superposition with given state probabilities."""
        
        # Handle both call patterns
        if initial_states and isinstance(initial_states, list):
            # New API: create equal probability superposition from state names
            task_id = task_id or f"superposition_{len(self.superpositions)}"
            states_dict = {}
            for state_name in initial_states:
                # Convert string to TaskState enum
                try:
                    if hasattr(TaskState, state_name.upper()):
                        states_dict[getattr(TaskState, state_name.upper())] = 1.0/len(initial_states)
                    else:
                        # Try with exact string match
                        for task_state in TaskState:
                            if task_state.value == state_name.lower():
                                states_dict[task_state] = 1.0/len(initial_states)
                                break
                except:
                    # Fallback: use superposition state
                    states_dict[TaskState.SUPERPOSITION] = 1.0
        else:
            # Legacy API: use provided state probabilities
            states_dict = state_probs or initial_states or {TaskState.SUPERPOSITION: 1.0}
            task_id = task_id or f"task_{len(self.superpositions)}"
        
        # Ensure we have at least one state
        if not states_dict:
            states_dict = {TaskState.SUPERPOSITION: 1.0}
        
        # Normalize probabilities to sum to 1
        total_prob = sum(states_dict.values())
        if total_prob == 0:
            raise ValueError("Total probability cannot be zero")
        
        normalized_states = {
            state: prob / total_prob 
            for state, prob in states_dict.items()
        }
        
        superposition = TaskSuperposition(
            task_id=task_id,
            state_probabilities=normalized_states
        )
        
        self.superpositions[task_id] = superposition
        
        self.logger.info(f"Created superposition for task {task_id} with states: {normalized_states}")
        
        return superposition
    
    def measure_superposition(self, task_id: str) -> Optional[TaskState]:
        """Measure/observe a task superposition, collapsing it to a single state."""
        
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return None
        
        # Use quantum measurement - probabilistic collapse
        states = list(superposition.state_probabilities.keys())
        probabilities = list(superposition.state_probabilities.values())
        
        # Account for coherence decay
        coherence_factor = self._calculate_coherence_decay(superposition)
        adjusted_probabilities = [p * coherence_factor for p in probabilities]
        
        # Renormalize
        total_prob = sum(adjusted_probabilities)
        if total_prob > 0:
            adjusted_probabilities = [p / total_prob for p in adjusted_probabilities]
        else:
            # If completely decohered, collapse to most likely state
            max_prob_index = probabilities.index(max(probabilities))
            adjusted_probabilities = [0] * len(probabilities)
            adjusted_probabilities[max_prob_index] = 1.0
        
        # Collapse to single state
        collapsed_state = np.random.choice(states, p=adjusted_probabilities)
        
        # Update superposition record
        superposition.last_measurement = datetime.utcnow()
        superposition.state_probabilities = {collapsed_state: 1.0}
        
        self.logger.info(f"Task {task_id} superposition collapsed to: {collapsed_state.value}")
        
        return collapsed_state
    
    def measure(self, superposition_task: TaskSuperposition) -> Optional[TaskState]:
        """Convenience method for measuring superposition (alias for measure_superposition)."""
        return self.measure_superposition(superposition_task.task_id)
    
    def evolve_superposition(self, task_id: str, time_step: float = 1.0) -> bool:
        """Evolve task superposition over time using Schrödinger-like equation."""
        
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return False
        
        # Simple quantum evolution - phases evolve at different rates
        evolved_states = {}
        
        for state, amplitude in superposition.state_probabilities.items():
            if amplitude == 0:
                evolved_states[state] = 0.0
                continue
            
            # Each state has different "energy" affecting evolution rate
            energy = self._get_state_energy(state)
            phase_evolution = np.exp(-1j * energy * time_step)
            
            # In simplified model, just adjust amplitude slightly
            evolved_amplitude = amplitude * abs(phase_evolution)
            evolved_states[state] = evolved_amplitude
        
        # Renormalize
        total_amplitude = sum(evolved_states.values())
        if total_amplitude > 0:
            superposition.state_probabilities = {
                state: amp / total_amplitude 
                for state, amp in evolved_states.items()
            }
        
        return True
    
    def apply_quantum_gate(
        self, 
        task_id: str, 
        gate_operation: Callable[[Dict[TaskState, float]], Dict[TaskState, float]]
    ) -> bool:
        """Apply a quantum gate operation to transform the superposition."""
        
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return False
        
        try:
            new_state_probs = gate_operation(superposition.state_probabilities)
            
            # Ensure probabilities are normalized
            total_prob = sum(new_state_probs.values())
            if total_prob > 0:
                superposition.state_probabilities = {
                    state: prob / total_prob
                    for state, prob in new_state_probs.items()
                }
            
            self.logger.info(f"Applied quantum gate to task {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply quantum gate to task {task_id}: {e}")
            return False
    
    def create_bell_state(self, task_id1: str, task_id2: str) -> bool:
        """Create maximally entangled Bell state between two tasks."""
        
        # Bell state: (|00⟩ + |11⟩) / √2 
        # In task terms: both completed or both failed with equal probability
        
        bell_states = {
            TaskState.COMPLETED: 0.5,
            TaskState.FAILED: 0.5
        }
        
        success1 = self.create_superposition(task_id1, bell_states)
        success2 = self.create_superposition(task_id2, bell_states)
        
        if success1 and success2:
            self.logger.info(f"Created Bell state between tasks {task_id1} and {task_id2}")
            return True
        
        return False
    
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
    
    def get_most_probable_state(self, task_id: str) -> Optional[TaskState]:
        """Get the most probable state without collapsing superposition."""
        
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return None
        
        return max(
            superposition.state_probabilities.items(),
            key=lambda x: x[1]
        )[0]
    
    def get_superposition_purity(self, task_id: str) -> float:
        """Calculate purity of the superposition state (0=mixed, 1=pure)."""
        
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return 0.0
        
        # Purity = Tr(ρ²) where ρ is density matrix
        # For diagonal density matrix: Purity = Σ(p_i²)
        purity = sum(p**2 for p in superposition.state_probabilities.values())
        
        return purity
    
    def interfere_superpositions(
        self, 
        task_id1: str, 
        task_id2: str, 
        interference_strength: float = 0.1
    ) -> bool:
        """Create interference between two task superpositions."""
        
        superposition1 = self.superpositions.get(task_id1)
        superposition2 = self.superpositions.get(task_id2)
        
        if not (superposition1 and superposition2):
            return False
        
        # Apply interference - constructive/destructive based on phase relationships
        for state in TaskState:
            if state in superposition1.state_probabilities and state in superposition2.state_probabilities:
                
                prob1 = superposition1.state_probabilities[state]
                prob2 = superposition2.state_probabilities[state]
                
                # Simple interference model
                phase_factor = np.cos(hash(task_id1) - hash(task_id2))
                interference = interference_strength * phase_factor * np.sqrt(prob1 * prob2)
                
                # Update probabilities
                superposition1.state_probabilities[state] += interference
                superposition2.state_probabilities[state] += interference
        
        # Renormalize both superpositions
        self._renormalize_superposition(task_id1)
        self._renormalize_superposition(task_id2)
        
        self.logger.info(f"Applied interference between tasks {task_id1} and {task_id2}")
        
        return True
    
    def create_coherent_superposition(
        self,
        task_id: str,
        base_state: TaskState,
        coherence_strength: float = 0.8
    ) -> TaskSuperposition:
        """Create a coherent superposition centered around a base state."""
        
        # Create coherent state with high probability in base state
        # and distributed probability in adjacent states
        state_probs = {state: 0.0 for state in TaskState}
        state_probs[base_state] = coherence_strength
        
        # Distribute remaining probability
        remaining_prob = 1.0 - coherence_strength
        other_states = [s for s in TaskState if s != base_state]
        
        if other_states:
            prob_per_state = remaining_prob / len(other_states)
            for state in other_states:
                state_probs[state] = prob_per_state
        
        return self.create_superposition(task_id, state_probs)
    
    def _calculate_coherence_decay(self, superposition: TaskSuperposition) -> float:
        """Calculate coherence decay factor over time."""
        
        if not superposition.last_measurement:
            return superposition.coherence_coefficient
        
        elapsed = (datetime.utcnow() - superposition.last_measurement).total_seconds()
        decay_rate = 0.001  # Coherence decay rate per second
        
        return superposition.coherence_coefficient * np.exp(-decay_rate * elapsed)
    
    def _get_state_energy(self, state: TaskState) -> float:
        """Get energy level for quantum state evolution."""
        
        energy_levels = {
            TaskState.SUPERPOSITION: 0.0,
            TaskState.COLLAPSED: 1.0,
            TaskState.ENTANGLED: 0.5,
            TaskState.COMPLETED: 2.0,
            TaskState.FAILED: -1.0
        }
        
        return energy_levels.get(state, 0.0)
    
    def _renormalize_superposition(self, task_id: str) -> None:
        """Renormalize superposition probabilities to sum to 1."""
        
        superposition = self.superpositions.get(task_id)
        if not superposition:
            return
        
        # Ensure all probabilities are non-negative
        for state in superposition.state_probabilities:
            superposition.state_probabilities[state] = max(
                0.0, superposition.state_probabilities[state]
            )
        
        total_prob = sum(superposition.state_probabilities.values())
        
        if total_prob > 0:
            for state in superposition.state_probabilities:
                superposition.state_probabilities[state] /= total_prob
        else:
            # If all probabilities are zero, create uniform distribution
            num_states = len(superposition.state_probabilities)
            uniform_prob = 1.0 / num_states
            for state in superposition.state_probabilities:
                superposition.state_probabilities[state] = uniform_prob
    
    def get_all_superpositions(self) -> Dict[str, TaskSuperposition]:
        """Get all active superpositions."""
        return self.superpositions.copy()
    
    def clear_superposition(self, task_id: str) -> bool:
        """Remove a task superposition."""
        if task_id in self.superpositions:
            del self.superpositions[task_id]
            self.logger.info(f"Cleared superposition for task {task_id}")
            return True
        return False
    
    def get_superposition_statistics(self) -> Dict[str, Any]:
        """Get statistics about all superpositions."""
        
        if not self.superpositions:
            return {"total_superpositions": 0}
        
        total_entropy = 0.0
        total_purity = 0.0
        state_distribution = {state.value: 0 for state in TaskState}
        
        for task_id, superposition in self.superpositions.items():
            total_entropy += self.calculate_superposition_entropy(task_id)
            total_purity += self.get_superposition_purity(task_id)
            
            # Count most probable states
            most_probable = self.get_most_probable_state(task_id)
            if most_probable:
                state_distribution[most_probable.value] += 1
        
        num_superpositions = len(self.superpositions)
        
        return {
            "total_superpositions": num_superpositions,
            "average_entropy": total_entropy / num_superpositions,
            "average_purity": total_purity / num_superpositions,
            "state_distribution": state_distribution,
            "coherent_superpositions": sum(
                1 for s in self.superpositions.values() 
                if self._calculate_coherence_decay(s) > 0.5
            )
        }