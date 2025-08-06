"""
Quantum-inspired task planning module.
"""

from .quantum_planner import QuantumTaskPlanner
from .superposition_tasks import SuperpositionTaskManager
from .entanglement_dependencies import EntanglementDependencyGraph

__all__ = [
    'QuantumTaskPlanner',
    'SuperpositionTaskManager', 
    'EntanglementDependencyGraph'
]