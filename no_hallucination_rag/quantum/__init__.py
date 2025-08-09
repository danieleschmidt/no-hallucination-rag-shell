"""
Quantum-inspired task planning module.
Generation 1: Basic functionality with graceful imports.
"""

# Import core quantum planner
try:
    from .quantum_planner import QuantumTaskPlanner
except ImportError as e:
    print(f"Warning: Could not import QuantumTaskPlanner: {e}")
    QuantumTaskPlanner = None

# Import optional components  
try:
    from .superposition_tasks import SuperpositionTaskManager
except ImportError:
    SuperpositionTaskManager = None

try:
    from .entanglement_dependencies import EntanglementDependencyGraph  
except ImportError:
    EntanglementDependencyGraph = None

# Export available components
__all__ = []
for component in ["QuantumTaskPlanner", "SuperpositionTaskManager", "EntanglementDependencyGraph"]:
    if globals().get(component):
        __all__.append(component)