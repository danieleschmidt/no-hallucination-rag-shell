"""
No-Hallucination RAG Shell: Retrieval-First CLI with Zero-Hallucination Guarantees & Quantum-Inspired Task Planning
"""

from .core.factual_rag import FactualRAG
from .core.source_ranker import SourceRanker
from .verification.factuality_detector import FactualityDetector
from .governance.compliance_checker import GovernanceChecker
from .knowledge.knowledge_base import KnowledgeBase

# Quantum-inspired task planning components
from .quantum.quantum_planner import QuantumTaskPlanner
from .quantum.superposition_tasks import SuperpositionTaskManager
from .quantum.entanglement_dependencies import EntanglementDependencyGraph

# Optional shell import (requires rich)
try:
    from .shell.interactive_shell import InteractiveShell
except ImportError:
    InteractiveShell = None

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "FactualRAG",
    "SourceRanker", 
    "FactualityDetector",
    "GovernanceChecker",
    "KnowledgeBase",
    "QuantumTaskPlanner",
    "SuperpositionTaskManager",
    "EntanglementDependencyGraph"
] + (["InteractiveShell"] if InteractiveShell else [])