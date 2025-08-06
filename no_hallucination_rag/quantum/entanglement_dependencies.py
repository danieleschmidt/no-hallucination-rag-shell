"""
Entanglement dependency graph for quantum-correlated task relationships.
"""

import logging
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

from .quantum_planner import QuantumTask, TaskState


class EntanglementType(Enum):
    """Types of quantum entanglement between tasks."""
    BELL_STATE = "bell_state"           # Maximally entangled
    SPIN_CORRELATED = "spin_correlated" # Same outcome correlation
    ANTI_CORRELATED = "anti_correlated" # Opposite outcome correlation
    GHZ_STATE = "ghz_state"            # Multi-task entanglement
    CLUSTER_STATE = "cluster_state"     # Network entanglement


@dataclass 
class EntanglementBond:
    """Represents an entanglement relationship between tasks."""
    task_id1: str
    task_id2: str
    entanglement_type: EntanglementType
    correlation_strength: float  # 0.0 to 1.0
    created_at: datetime
    last_interaction: Optional[datetime] = None
    measurement_count: int = 0
    
    def get_correlation_coefficient(self) -> float:
        """Get correlation coefficient based on entanglement type."""
        base_correlation = self.correlation_strength
        
        type_multipliers = {
            EntanglementType.BELL_STATE: 1.0,
            EntanglementType.SPIN_CORRELATED: 0.8,
            EntanglementType.ANTI_CORRELATED: -0.8,
            EntanglementType.GHZ_STATE: 0.9,
            EntanglementType.CLUSTER_STATE: 0.6
        }
        
        multiplier = type_multipliers.get(self.entanglement_type, 0.5)
        return base_correlation * multiplier


class EntanglementDependencyGraph:
    """
    Graph-based system for managing quantum entanglement between tasks.
    
    Uses NetworkX to represent complex entanglement relationships and
    their effects on task state correlations.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entanglement_bonds: Dict[Tuple[str, str], EntanglementBond] = {}
        self.logger = logging.getLogger(__name__)
        
        # Track entanglement clusters  
        self.clusters: Dict[str, Set[str]] = {}
        
    def create_entanglement(
        self,
        task_id1: str,
        task_id2: str,
        entanglement_type: EntanglementType = EntanglementType.SPIN_CORRELATED,
        correlation_strength: float = 0.8
    ) -> bool:
        """Create quantum entanglement between two tasks."""
        
        if task_id1 == task_id2:
            self.logger.error("Cannot entangle task with itself")
            return False
        
        # Ensure consistent ordering for bond key
        bond_key = tuple(sorted([task_id1, task_id2]))
        
        if bond_key in self.entanglement_bonds:
            self.logger.warning(f"Entanglement already exists between {task_id1} and {task_id2}")
            return False
        
        # Create entanglement bond
        bond = EntanglementBond(
            task_id1=task_id1,
            task_id2=task_id2,
            entanglement_type=entanglement_type,
            correlation_strength=correlation_strength,
            created_at=datetime.utcnow()
        )
        
        # Add to graph and bond tracking
        self.graph.add_edge(
            task_id1, 
            task_id2,
            bond=bond,
            weight=correlation_strength
        )
        self.entanglement_bonds[bond_key] = bond
        
        # Update clusters
        self._update_entanglement_clusters(task_id1, task_id2)
        
        self.logger.info(
            f"Created {entanglement_type.value} entanglement between {task_id1} and {task_id2} "
            f"(strength: {correlation_strength})"
        )
        
        return True
    
    def measure_entangled_state(self, measured_task_id: str, measured_state: TaskState) -> List[Tuple[str, TaskState, float]]:
        """
        Measure one task's state and determine correlated states of entangled tasks.
        
        Returns list of (task_id, predicted_state, probability) for entangled tasks.
        """
        
        if not self.graph.has_node(measured_task_id):
            return []
        
        correlated_predictions = []
        
        # Get all entangled tasks
        for neighbor_id in self.graph.neighbors(measured_task_id):
            bond_key = tuple(sorted([measured_task_id, neighbor_id]))
            bond = self.entanglement_bonds.get(bond_key)
            
            if not bond:
                continue
            
            # Update interaction tracking
            bond.last_interaction = datetime.utcnow()
            bond.measurement_count += 1
            
            # Determine correlated state based on entanglement type
            predicted_state, probability = self._calculate_correlated_state(
                measured_state, bond.entanglement_type, bond.correlation_strength
            )
            
            correlated_predictions.append((neighbor_id, predicted_state, probability))
            
            self.logger.info(
                f"Measurement correlation: {measured_task_id} -> {measured_state.value} "
                f"implies {neighbor_id} -> {predicted_state.value} (p={probability:.3f})"
            )
        
        return correlated_predictions
    
    def create_ghz_state(self, task_ids: List[str]) -> bool:
        """Create GHZ (Greenberger-Horne-Zeilinger) state for multi-task entanglement."""
        
        if len(task_ids) < 3:
            self.logger.error("GHZ state requires at least 3 tasks")
            return False
        
        # Create pairwise entanglements with GHZ correlation
        success_count = 0
        for i in range(len(task_ids)):
            for j in range(i + 1, len(task_ids)):
                if self.create_entanglement(
                    task_ids[i], 
                    task_ids[j],
                    EntanglementType.GHZ_STATE,
                    0.9
                ):
                    success_count += 1
        
        # Create cluster for GHZ state
        cluster_id = f"ghz_{hash(tuple(sorted(task_ids))) % 10000:04d}"
        self.clusters[cluster_id] = set(task_ids)
        
        self.logger.info(f"Created GHZ state with {len(task_ids)} tasks ({success_count} bonds)")
        
        return success_count > 0
    
    def create_cluster_state(self, task_graph: List[Tuple[str, str]]) -> bool:
        """Create cluster state from specified task connections."""
        
        success_count = 0
        task_set = set()
        
        for task_id1, task_id2 in task_graph:
            task_set.add(task_id1)
            task_set.add(task_id2)
            
            if self.create_entanglement(
                task_id1,
                task_id2, 
                EntanglementType.CLUSTER_STATE,
                0.6
            ):
                success_count += 1
        
        # Create cluster tracking
        cluster_id = f"cluster_{hash(tuple(sorted(task_set))) % 10000:04d}"
        self.clusters[cluster_id] = task_set
        
        self.logger.info(f"Created cluster state with {len(task_set)} tasks ({success_count} bonds)")
        
        return success_count > 0
    
    def break_entanglement(self, task_id1: str, task_id2: str) -> bool:
        """Break entanglement between two tasks."""
        
        bond_key = tuple(sorted([task_id1, task_id2]))
        
        if bond_key not in self.entanglement_bonds:
            return False
        
        # Remove from graph and tracking
        if self.graph.has_edge(task_id1, task_id2):
            self.graph.remove_edge(task_id1, task_id2)
        
        del self.entanglement_bonds[bond_key]
        
        # Update clusters
        self._remove_from_clusters(task_id1, task_id2)
        
        self.logger.info(f"Broke entanglement between {task_id1} and {task_id2}")
        
        return True
    
    def get_entanglement_degree(self, task_id: str) -> int:
        """Get number of entangled connections for a task."""
        return self.graph.degree(task_id) if self.graph.has_node(task_id) else 0
    
    def get_entangled_tasks(self, task_id: str) -> Set[str]:
        """Get all tasks entangled with the given task."""
        if not self.graph.has_node(task_id):
            return set()
        
        return set(self.graph.neighbors(task_id))
    
    def get_entanglement_strength(self, task_id1: str, task_id2: str) -> float:
        """Get entanglement strength between two tasks."""
        bond_key = tuple(sorted([task_id1, task_id2]))
        bond = self.entanglement_bonds.get(bond_key)
        
        return bond.correlation_strength if bond else 0.0
    
    def find_entanglement_path(self, start_task: str, end_task: str) -> Optional[List[str]]:
        """Find path of entangled connections between two tasks."""
        
        if not (self.graph.has_node(start_task) and self.graph.has_node(end_task)):
            return None
        
        try:
            path = nx.shortest_path(self.graph, start_task, end_task)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_entanglement_clusters(self) -> Dict[str, Set[str]]:
        """Get all entanglement clusters."""
        return self.clusters.copy()
    
    def calculate_cluster_coherence(self, cluster_id: str) -> float:
        """Calculate coherence measure for an entanglement cluster."""
        
        if cluster_id not in self.clusters:
            return 0.0
        
        task_set = self.clusters[cluster_id]
        
        if len(task_set) < 2:
            return 1.0
        
        # Calculate average correlation strength within cluster
        total_correlation = 0.0
        connection_count = 0
        
        for task_id1 in task_set:
            for task_id2 in task_set:
                if task_id1 < task_id2:  # Avoid double counting
                    strength = self.get_entanglement_strength(task_id1, task_id2)
                    if strength > 0:
                        total_correlation += strength
                        connection_count += 1
        
        if connection_count == 0:
            return 0.0
        
        return total_correlation / connection_count
    
    def detect_bell_violations(self, task_id1: str, task_id2: str) -> Optional[float]:
        """
        Detect violations of Bell inequalities (quantum non-locality).
        
        Returns Bell parameter S, where |S| > 2 indicates quantum entanglement.
        """
        
        bond_key = tuple(sorted([task_id1, task_id2]))
        bond = self.entanglement_bonds.get(bond_key)
        
        if not bond or bond.measurement_count < 4:
            return None
        
        # Simplified Bell test - in practice would require measurement statistics
        correlation_coefficient = bond.get_correlation_coefficient()
        
        # CHSH Bell parameter approximation
        S = 2 * np.sqrt(2) * abs(correlation_coefficient)
        
        if abs(S) > 2:
            self.logger.info(f"Bell violation detected: S = {S:.3f} for tasks {task_id1}, {task_id2}")
        
        return S
    
    def get_entanglement_entropy(self, task_set: Set[str]) -> float:
        """Calculate entanglement entropy for a set of tasks."""
        
        if len(task_set) < 2:
            return 0.0
        
        # Calculate connectivity matrix
        n = len(task_set)
        task_list = list(task_set)
        connectivity_matrix = np.zeros((n, n))
        
        for i, task_id1 in enumerate(task_list):
            for j, task_id2 in enumerate(task_list):
                if i != j:
                    strength = self.get_entanglement_strength(task_id1, task_id2)
                    connectivity_matrix[i, j] = strength
        
        # Calculate eigenvalues for entropy
        try:
            eigenvalues = np.linalg.eigvals(connectivity_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter near-zero values
            
            # Von Neumann entropy
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
            return entropy
            
        except Exception as e:
            self.logger.error(f"Failed to calculate entanglement entropy: {e}")
            return 0.0
    
    def visualize_entanglement_graph(self) -> Dict[str, Any]:
        """Generate visualization data for the entanglement graph."""
        
        # Node data
        nodes = []
        for node_id in self.graph.nodes():
            degree = self.graph.degree(node_id)
            nodes.append({
                'id': node_id,
                'label': node_id[:8] + '...' if len(node_id) > 8 else node_id,
                'degree': degree,
                'size': max(10, degree * 3)
            })
        
        # Edge data  
        edges = []
        for edge in self.graph.edges(data=True):
            task_id1, task_id2, data = edge
            bond = data.get('bond')
            
            if bond:
                edges.append({
                    'source': task_id1,
                    'target': task_id2,
                    'weight': bond.correlation_strength,
                    'type': bond.entanglement_type.value,
                    'color': self._get_entanglement_color(bond.entanglement_type)
                })
        
        # Cluster data
        clusters = []
        for cluster_id, task_set in self.clusters.items():
            coherence = self.calculate_cluster_coherence(cluster_id)
            clusters.append({
                'id': cluster_id,
                'tasks': list(task_set),
                'size': len(task_set),
                'coherence': coherence
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'clusters': clusters,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'total_clusters': len(clusters),
                'average_degree': np.mean([n['degree'] for n in nodes]) if nodes else 0,
                'max_degree': max([n['degree'] for n in nodes]) if nodes else 0
            }
        }
    
    def _calculate_correlated_state(
        self, 
        measured_state: TaskState, 
        entanglement_type: EntanglementType,
        correlation_strength: float
    ) -> Tuple[TaskState, float]:
        """Calculate correlated state based on entanglement type."""
        
        if entanglement_type == EntanglementType.SPIN_CORRELATED:
            # Same state with high probability
            return measured_state, correlation_strength
        
        elif entanglement_type == EntanglementType.ANTI_CORRELATED:
            # Opposite state mapping
            opposite_states = {
                TaskState.COMPLETED: TaskState.FAILED,
                TaskState.FAILED: TaskState.COMPLETED,
                TaskState.SUPERPOSITION: TaskState.COLLAPSED,
                TaskState.COLLAPSED: TaskState.SUPERPOSITION,
                TaskState.ENTANGLED: TaskState.ENTANGLED  # Self-referential
            }
            opposite_state = opposite_states.get(measured_state, measured_state)
            return opposite_state, correlation_strength
        
        elif entanglement_type == EntanglementType.BELL_STATE:
            # Maximal correlation - either same or opposite with equal probability
            if np.random.random() < 0.5:
                return measured_state, correlation_strength
            else:
                opposite_states = {
                    TaskState.COMPLETED: TaskState.FAILED,
                    TaskState.FAILED: TaskState.COMPLETED
                }
                opposite_state = opposite_states.get(measured_state, measured_state)
                return opposite_state, correlation_strength
        
        else:
            # Default: same state with reduced probability
            return measured_state, correlation_strength * 0.7
    
    def _update_entanglement_clusters(self, task_id1: str, task_id2: str) -> None:
        """Update entanglement clusters when new bond is created."""
        
        # Find existing clusters containing these tasks
        clusters_with_task1 = [cid for cid, tasks in self.clusters.items() if task_id1 in tasks]
        clusters_with_task2 = [cid for cid, tasks in self.clusters.items() if task_id2 in tasks]
        
        if not clusters_with_task1 and not clusters_with_task2:
            # Create new cluster
            cluster_id = f"cluster_{len(self.clusters):04d}"
            self.clusters[cluster_id] = {task_id1, task_id2}
            
        elif clusters_with_task1 and not clusters_with_task2:
            # Add task2 to existing cluster
            self.clusters[clusters_with_task1[0]].add(task_id2)
            
        elif not clusters_with_task1 and clusters_with_task2:
            # Add task1 to existing cluster  
            self.clusters[clusters_with_task2[0]].add(task_id1)
            
        elif clusters_with_task1 and clusters_with_task2:
            # Merge clusters if different
            cluster1_id = clusters_with_task1[0]
            cluster2_id = clusters_with_task2[0]
            
            if cluster1_id != cluster2_id:
                # Merge into first cluster and remove second
                self.clusters[cluster1_id].update(self.clusters[cluster2_id])
                del self.clusters[cluster2_id]
    
    def _remove_from_clusters(self, task_id1: str, task_id2: str) -> None:
        """Remove tasks from clusters when entanglement is broken."""
        
        # Check if breaking this bond disconnects the cluster
        clusters_to_check = []
        
        for cluster_id, task_set in self.clusters.items():
            if task_id1 in task_set or task_id2 in task_set:
                clusters_to_check.append(cluster_id)
        
        for cluster_id in clusters_to_check:
            task_set = self.clusters[cluster_id]
            
            # Check if cluster is still connected after removing this bond
            subgraph = self.graph.subgraph(task_set)
            
            if not nx.is_connected(subgraph):
                # Split cluster into connected components
                components = list(nx.connected_components(subgraph))
                
                # Remove original cluster
                del self.clusters[cluster_id]
                
                # Create new clusters for each component
                for i, component in enumerate(components):
                    if len(component) > 1:
                        new_cluster_id = f"{cluster_id}_split_{i}"
                        self.clusters[new_cluster_id] = component
    
    def _get_entanglement_color(self, entanglement_type: EntanglementType) -> str:
        """Get color for visualization based on entanglement type."""
        
        colors = {
            EntanglementType.BELL_STATE: "#ff0000",        # Red
            EntanglementType.SPIN_CORRELATED: "#00ff00",   # Green  
            EntanglementType.ANTI_CORRELATED: "#0000ff",   # Blue
            EntanglementType.GHZ_STATE: "#ff00ff",         # Magenta
            EntanglementType.CLUSTER_STATE: "#ffff00"      # Yellow
        }
        
        return colors.get(entanglement_type, "#666666")
    
    def get_entanglement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the entanglement network."""
        
        if not self.graph.nodes():
            return {"total_tasks": 0}
        
        # Basic graph metrics
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        
        # Degree statistics
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        avg_degree = np.mean(degrees)
        max_degree = max(degrees)
        
        # Clustering coefficient
        clustering = nx.average_clustering(self.graph)
        
        # Path lengths (if connected)
        try:
            avg_path_length = nx.average_shortest_path_length(self.graph)
        except:
            avg_path_length = float('inf')
        
        # Entanglement type distribution
        type_counts = {}
        for bond in self.entanglement_bonds.values():
            etype = bond.entanglement_type.value
            type_counts[etype] = type_counts.get(etype, 0) + 1
        
        # Correlation strength statistics
        strengths = [bond.correlation_strength for bond in self.entanglement_bonds.values()]
        avg_strength = np.mean(strengths) if strengths else 0.0
        
        return {
            "total_tasks": num_nodes,
            "total_entanglements": num_edges,
            "network_density": density,
            "average_degree": avg_degree,
            "max_degree": max_degree,
            "clustering_coefficient": clustering,
            "average_path_length": avg_path_length,
            "total_clusters": len(self.clusters),
            "entanglement_types": type_counts,
            "average_correlation_strength": avg_strength,
            "highly_entangled_tasks": sum(1 for d in degrees if d >= 3)
        }
    
    def clear_all_entanglements(self) -> None:
        """Clear all entanglements and reset the graph."""
        
        self.graph.clear()
        self.entanglement_bonds.clear()
        self.clusters.clear()
        
        self.logger.info("Cleared all entanglements")