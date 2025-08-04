"""
No-Hallucination RAG Shell: Retrieval-First CLI with Zero-Hallucination Guarantees
"""

from .core.factual_rag import FactualRAG
from .core.source_ranker import SourceRanker
from .verification.factuality_detector import FactualityDetector
from .governance.compliance_checker import GovernanceChecker
from .knowledge.knowledge_base import KnowledgeBase

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
] + (["InteractiveShell"] if InteractiveShell else [])