"""
Quantum-inspired task planning commands for the interactive shell.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio

from ..quantum.quantum_planner import (
    QuantumTaskPlanner, QuantumTask, TaskState, Priority
)
from ..quantum.superposition_tasks import SuperpositionTaskManager
from ..quantum.entanglement_dependencies import (
    EntanglementDependencyGraph, EntanglementType
)


class QuantumCommands:
    """Quantum task planning commands for the shell."""
    
    def __init__(self):
        self.planner = QuantumTaskPlanner()
        self.superposition_manager = SuperpositionTaskManager()
        self.entanglement_graph = EntanglementDependencyGraph()
        self.logger = logging.getLogger(__name__)
        
    def create_task(
        self,
        title: str,
        description: str = "",
        priority: str = "ground_state",
        due_date: Optional[str] = None,
        dependencies: Optional[str] = None,
        tags: Optional[str] = None,
        duration_hours: float = 1.0
    ) -> Dict[str, Any]:
        """Create a new quantum task."""
        
        try:
            # Parse priority
            priority_map = {
                "ground_state": Priority.GROUND_STATE,
                "first_excited": Priority.FIRST_EXCITED,
                "second_excited": Priority.SECOND_EXCITED,
                "third_excited": Priority.THIRD_EXCITED,
                "ionized": Priority.IONIZED
            }
            task_priority = priority_map.get(priority.lower(), Priority.GROUND_STATE)
            
            # Parse due date
            task_due_date = None
            if due_date:
                try:
                    task_due_date = datetime.fromisoformat(due_date)
                except ValueError:
                    # Try relative parsing (e.g., "3 days")
                    parts = due_date.split()
                    if len(parts) == 2 and parts[1] in ["days", "hours", "weeks"]:
                        amount = int(parts[0])
                        if parts[1] == "days":
                            task_due_date = datetime.utcnow() + timedelta(days=amount)
                        elif parts[1] == "hours":
                            task_due_date = datetime.utcnow() + timedelta(hours=amount)
                        elif parts[1] == "weeks":
                            task_due_date = datetime.utcnow() + timedelta(weeks=amount)
            
            # Parse dependencies and tags
            task_dependencies = set(dependencies.split(",")) if dependencies else set()
            task_tags = set(tags.split(",")) if tags else set()
            
            # Create task
            task = self.planner.create_task(
                title=title,
                description=description,
                priority=task_priority,
                due_date=task_due_date,
                dependencies=task_dependencies,
                tags=task_tags,
                estimated_duration=timedelta(hours=duration_hours)
            )
            
            # Create superposition state
            initial_states = {
                TaskState.SUPERPOSITION: 0.7,
                TaskState.COLLAPSED: 0.2,
                TaskState.ENTANGLED: 0.1
            }
            self.superposition_manager.create_superposition(task.id, initial_states)
            
            return {
                "success": True,
                "task_id": task.id,
                "title": task.title,
                "state": task.state.value,
                "quantum_properties": {
                    "probability_amplitude": task.probability_amplitude,
                    "coherence_time": task.coherence_time,
                    "superposition_entropy": self.superposition_manager.calculate_superposition_entropy(task.id)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create task: {e}")
            return {"success": False, "error": str(e)}
    
    def observe_task(self, task_id: str) -> Dict[str, Any]:
        """Observe a task, collapsing its quantum superposition."""
        
        try:
            # Observe task in planner
            task = self.planner.observe_task(task_id)
            if not task:
                return {"success": False, "error": "Task not found"}
            
            # Measure superposition
            collapsed_state = self.superposition_manager.measure_superposition(task_id)
            
            # Get entanglement correlations
            correlations = []
            if collapsed_state:
                correlations = self.entanglement_graph.measure_entangled_state(
                    task_id, collapsed_state
                )
            
            return {
                "success": True,
                "task_id": task_id,
                "title": task.title,
                "observed_state": task.state.value,
                "collapsed_superposition": collapsed_state.value if collapsed_state else None,
                "quantum_correlations": [
                    {
                        "entangled_task": corr[0],
                        "predicted_state": corr[1].value,
                        "probability": corr[2]
                    }
                    for corr in correlations
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to observe task: {e}")
            return {"success": False, "error": str(e)}
    
    def entangle_tasks(
        self,
        task_id1: str,
        task_id2: str,
        entanglement_type: str = "spin_correlated",
        strength: float = 0.8
    ) -> Dict[str, Any]:
        """Create quantum entanglement between two tasks."""
        
        try:
            # Map entanglement type
            type_map = {
                "bell_state": EntanglementType.BELL_STATE,
                "spin_correlated": EntanglementType.SPIN_CORRELATED,
                "anti_correlated": EntanglementType.ANTI_CORRELATED,
                "ghz_state": EntanglementType.GHZ_STATE,
                "cluster_state": EntanglementType.CLUSTER_STATE
            }
            ent_type = type_map.get(entanglement_type.lower(), EntanglementType.SPIN_CORRELATED)
            
            # Create entanglement in planner
            success_planner = self.planner.entangle_tasks(task_id1, task_id2, strength)
            
            # Create entanglement in graph
            success_graph = self.entanglement_graph.create_entanglement(
                task_id1, task_id2, ent_type, strength
            )
            
            return {
                "success": success_planner and success_graph,
                "task_id1": task_id1,
                "task_id2": task_id2,
                "entanglement_type": entanglement_type,
                "correlation_strength": strength,
                "bell_violation": self.entanglement_graph.detect_bell_violations(task_id1, task_id2)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to entangle tasks: {e}")
            return {"success": False, "error": str(e)}
    
    def get_optimal_sequence(self, available_hours: float = 8.0) -> Dict[str, Any]:
        """Get quantum-optimized task execution sequence."""
        
        try:
            available_time = timedelta(hours=available_hours)
            optimal_tasks = self.planner.get_optimal_task_sequence(available_time)
            
            sequence_info = []
            total_duration = timedelta()
            
            for task in optimal_tasks:
                # Get superposition info
                superposition = self.superposition_manager.superpositions.get(task.id)
                most_probable_state = self.superposition_manager.get_most_probable_state(task.id)
                
                # Get entanglement info
                entangled_tasks = self.entanglement_graph.get_entangled_tasks(task.id)
                
                task_info = {
                    "task_id": task.id,
                    "title": task.title,
                    "priority": task.priority.value,
                    "estimated_duration_hours": task.estimated_duration.total_seconds() / 3600,
                    "quantum_state": task.state.value,
                    "most_probable_state": most_probable_state.value if most_probable_state else None,
                    "entangled_with": len(entangled_tasks),
                    "superposition_purity": self.superposition_manager.get_superposition_purity(task.id)
                }
                
                sequence_info.append(task_info)
                total_duration += task.estimated_duration
            
            return {
                "success": True,
                "optimal_sequence": sequence_info,
                "total_tasks": len(optimal_tasks),
                "total_duration_hours": total_duration.total_seconds() / 3600,
                "quantum_interference_scores": self.planner.calculate_task_interference()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get optimal sequence: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_sequence(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute quantum-optimized task sequence."""
        
        try:
            if task_ids:
                # Execute specific tasks
                tasks = [self.planner.get_task_by_id(tid) for tid in task_ids]
                tasks = [t for t in tasks if t]  # Filter None values
            else:
                # Get optimal sequence
                optimal_tasks = self.planner.get_optimal_task_sequence(timedelta(hours=8))
                tasks = optimal_tasks[:5]  # Limit to 5 tasks for demo
            
            if not tasks:
                return {"success": False, "error": "No tasks to execute"}
            
            # Execute task sequence
            execution_results = self.planner.execute_task_sequence(tasks)
            
            # Update superpositions based on execution results
            for task_result in execution_results['tasks_executed']:
                task_id = task_result['task_id']
                
                # Collapse superposition to completed state
                self.superposition_manager.measure_superposition(task_id)
            
            return {
                "success": True,
                "execution_results": {
                    "started_at": execution_results['started_at'].isoformat(),
                    "completed_at": execution_results['completed_at'].isoformat(),
                    "total_duration_seconds": execution_results['total_duration'].total_seconds(),
                    "tasks_completed": len(execution_results['tasks_executed']),
                    "tasks_failed": len(execution_results['tasks_failed']),
                    "quantum_effects": execution_results['quantum_effects_observed']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute sequence: {e}")
            return {"success": False, "error": str(e)}
    
    def list_tasks(self, filter_state: Optional[str] = None) -> Dict[str, Any]:
        """List all quantum tasks with their states."""
        
        try:
            # Parse filter state
            state_filter = None
            if filter_state:
                state_map = {
                    "superposition": TaskState.SUPERPOSITION,
                    "collapsed": TaskState.COLLAPSED,
                    "entangled": TaskState.ENTANGLED,
                    "completed": TaskState.COMPLETED,
                    "failed": TaskState.FAILED
                }
                state_filter = state_map.get(filter_state.lower())
            
            # Get tasks
            tasks = self.planner.list_tasks(state_filter)
            
            task_list = []
            for task in tasks:
                # Get quantum properties
                superposition = self.superposition_manager.superpositions.get(task.id)
                most_probable = self.superposition_manager.get_most_probable_state(task.id)
                entangled_tasks = self.entanglement_graph.get_entangled_tasks(task.id)
                
                task_info = {
                    "task_id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "state": task.state.value,
                    "priority": task.priority.value,
                    "progress": task.progress,
                    "created_at": task.created_at.isoformat(),
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "quantum_properties": {
                        "probability_amplitude": task.probability_amplitude,
                        "is_coherent": task.is_coherent(),
                        "most_probable_state": most_probable.value if most_probable else None,
                        "superposition_entropy": self.superposition_manager.calculate_superposition_entropy(task.id),
                        "entanglement_degree": len(entangled_tasks)
                    }
                }
                
                task_list.append(task_info)
            
            return {
                "success": True,
                "tasks": task_list,
                "total_tasks": len(task_list),
                "quantum_summary": self.planner.get_quantum_state_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list tasks: {e}")
            return {"success": False, "error": str(e)}
    
    def show_quantum_state(self) -> Dict[str, Any]:
        """Show comprehensive quantum state of the system."""
        
        try:
            # Get quantum state summaries
            planner_summary = self.planner.get_quantum_state_summary()
            superposition_stats = self.superposition_manager.get_superposition_statistics()
            entanglement_stats = self.entanglement_graph.get_entanglement_statistics()
            
            # Get visualization data
            entanglement_viz = self.entanglement_graph.visualize_entanglement_graph()
            
            return {
                "success": True,
                "quantum_state": {
                    "task_planner": planner_summary,
                    "superposition_manager": superposition_stats,
                    "entanglement_graph": entanglement_stats,
                    "visualization": entanglement_viz
                },
                "system_coherence": {
                    "total_coherent_tasks": planner_summary.get('coherent_tasks', 0),
                    "coherence_ratio": planner_summary.get('quantum_coherence_ratio', 0.0),
                    "entanglement_density": entanglement_stats.get('network_density', 0.0),
                    "average_superposition_entropy": superposition_stats.get('average_entropy', 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to show quantum state: {e}")
            return {"success": False, "error": str(e)}
    
    def create_ghz_state(self, task_ids: List[str]) -> Dict[str, Any]:
        """Create GHZ (multi-task entangled) state."""
        
        try:
            if len(task_ids) < 3:
                return {"success": False, "error": "GHZ state requires at least 3 tasks"}
            
            success = self.entanglement_graph.create_ghz_state(task_ids)
            
            if success:
                # Get cluster info
                clusters = self.entanglement_graph.get_entanglement_clusters()
                ghz_cluster = None
                for cluster_id, task_set in clusters.items():
                    if set(task_ids).issubset(task_set):
                        ghz_cluster = cluster_id
                        break
                
                return {
                    "success": True,
                    "ghz_tasks": task_ids,
                    "cluster_id": ghz_cluster,
                    "cluster_coherence": self.entanglement_graph.calculate_cluster_coherence(ghz_cluster) if ghz_cluster else 0.0
                }
            else:
                return {"success": False, "error": "Failed to create GHZ state"}
                
        except Exception as e:
            self.logger.error(f"Failed to create GHZ state: {e}")
            return {"success": False, "error": str(e)}
    
    def apply_quantum_gate(self, task_id: str, gate_type: str) -> Dict[str, Any]:
        """Apply quantum gate operations to task superposition."""
        
        try:
            # Define quantum gate operations
            def hadamard_gate(state_probs):
                """Hadamard gate - creates equal superposition"""
                return {state: 1.0/len(state_probs) for state in state_probs.keys()}
            
            def pauli_x_gate(state_probs):
                """Pauli-X gate - flips states"""
                flipped_states = {}
                for state, prob in state_probs.items():
                    if state == TaskState.COMPLETED:
                        flipped_states[TaskState.FAILED] = prob
                    elif state == TaskState.FAILED:
                        flipped_states[TaskState.COMPLETED] = prob
                    else:
                        flipped_states[state] = prob
                return flipped_states
            
            def phase_gate(state_probs):
                """Phase gate - rotates quantum phases"""
                # In simplified model, just redistribute probabilities
                rotated_states = {}
                states = list(state_probs.keys())
                probs = list(state_probs.values())
                
                # Rotate probability distribution
                for i, state in enumerate(states):
                    rotated_states[state] = probs[(i + 1) % len(probs)]
                
                return rotated_states
            
            # Map gate types to operations
            gate_operations = {
                "hadamard": hadamard_gate,
                "h": hadamard_gate,
                "pauli_x": pauli_x_gate, 
                "x": pauli_x_gate,
                "phase": phase_gate,
                "p": phase_gate
            }
            
            gate_op = gate_operations.get(gate_type.lower())
            if not gate_op:
                return {"success": False, "error": f"Unknown gate type: {gate_type}"}
            
            # Apply quantum gate
            success = self.superposition_manager.apply_quantum_gate(task_id, gate_op)
            
            if success:
                # Get updated superposition
                superposition = self.superposition_manager.superpositions.get(task_id)
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "gate_applied": gate_type,
                    "new_superposition": {
                        state.value: prob 
                        for state, prob in superposition.state_probabilities.items()
                    } if superposition else {},
                    "new_entropy": self.superposition_manager.calculate_superposition_entropy(task_id),
                    "new_purity": self.superposition_manager.get_superposition_purity(task_id)
                }
            else:
                return {"success": False, "error": "Failed to apply quantum gate"}
                
        except Exception as e:
            self.logger.error(f"Failed to apply quantum gate: {e}")
            return {"success": False, "error": str(e)}
    
    def export_quantum_state(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Export complete quantum state to file."""
        
        try:
            # Get complete quantum state
            quantum_state = self.planner.export_quantum_state()
            
            # Add superposition and entanglement data
            quantum_state['superpositions'] = {
                task_id: {
                    'state_probabilities': {
                        state.value: prob 
                        for state, prob in sp.state_probabilities.items()
                    },
                    'coherence_coefficient': sp.coherence_coefficient,
                    'last_measurement': sp.last_measurement.isoformat() if sp.last_measurement else None
                }
                for task_id, sp in self.superposition_manager.get_all_superpositions().items()
            }
            
            quantum_state['entanglement_network'] = self.entanglement_graph.get_entanglement_statistics()
            quantum_state['entanglement_visualization'] = self.entanglement_graph.visualize_entanglement_graph()
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(quantum_state, f, indent=2, default=str)
                
                return {
                    "success": True,
                    "exported_to": filename,
                    "total_tasks": len(quantum_state['tasks']),
                    "superpositions": len(quantum_state['superpositions']),
                    "entanglements": quantum_state['entanglement_network']['total_entanglements']
                }
            else:
                return {
                    "success": True,
                    "quantum_state": quantum_state
                }
                
        except Exception as e:
            self.logger.error(f"Failed to export quantum state: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self) -> Dict[str, Any]:
        """Shutdown quantum components."""
        
        try:
            self.planner.shutdown()
            self.entanglement_graph.clear_all_entanglements()
            
            return {"success": True, "message": "Quantum components shutdown successfully"}
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown quantum components: {e}")
            return {"success": False, "error": str(e)}