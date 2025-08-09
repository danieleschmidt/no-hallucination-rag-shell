"""
No-Hallucination RAG Shell: Retrieval-First CLI with Zero-Hallucination Guarantees & Quantum-Inspired Task Planning
Generation 1: Basic functionality with graceful imports
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

# Core components - import with error handling
try:
    from .core.factual_rag import FactualRAG
except ImportError as e:
    print(f"Warning: Could not import FactualRAG: {e}")
    FactualRAG = None

try:
    from .core.source_ranker import SourceRanker
except ImportError:
    SourceRanker = None

try:
    from .verification.factuality_detector import FactualityDetector
except ImportError:
    FactualityDetector = None

try:
    from .governance.compliance_checker import GovernanceChecker
except ImportError:
    GovernanceChecker = None

try:
    from .knowledge.knowledge_base import KnowledgeBase
except ImportError:
    KnowledgeBase = None

# Quantum components
try:
    from .quantum.quantum_planner import QuantumTaskPlanner
except ImportError:
    QuantumTaskPlanner = None

try:
    from .quantum.superposition_tasks import SuperpositionTaskManager
except ImportError:
    SuperpositionTaskManager = None

try:
    from .quantum.entanglement_dependencies import EntanglementDependencyGraph
except ImportError:
    EntanglementDependencyGraph = None

# Shell interface
try:
    from .shell.interactive_shell import InteractiveShell
except ImportError:
    InteractiveShell = None

# Export available components
__all__ = []
for component in [
    "FactualRAG", "SourceRanker", "FactualityDetector", "GovernanceChecker", 
    "KnowledgeBase", "QuantumTaskPlanner", "SuperpositionTaskManager", 
    "EntanglementDependencyGraph", "InteractiveShell"
]:
    if globals().get(component):
        __all__.append(component)