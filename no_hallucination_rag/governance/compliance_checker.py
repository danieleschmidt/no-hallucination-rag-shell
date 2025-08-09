"""
Governance compliance checker for RAG responses.
Generation 1: Basic rule-based compliance checking.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ComplianceResult:
    """Result of compliance checking."""
    is_compliant: bool
    details: Dict[str, Any]
    violations: List[str]
    recommendations: List[str]


class GovernanceChecker:
    """Checks governance compliance for RAG responses."""
    
    def __init__(self, mode: str = "strict"):
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize compliance policies
        self.policies = self._init_policies()
        
        self.logger.info(f"GovernanceChecker initialized (Mode: {mode}, Generation 1)")
    
    def check_compliance(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> ComplianceResult:
        """Check compliance of RAG response."""
        
        violations = []
        details = {}
        
        # Basic compliance checks
        if not sources:
            violations.append("No sources provided for verification")
        
        if len(answer) < 10:
            violations.append("Answer too brief to assess compliance")
        
        # Check for potentially sensitive content
        sensitive_patterns = ['personal data', 'private information', 'confidential']
        for pattern in sensitive_patterns:
            if pattern.lower() in answer.lower():
                violations.append(f"Potential sensitive content: {pattern}")
        
        # Source authority check
        if sources:
            low_authority_sources = [s for s in sources if s.get("authority_score", 0.5) < 0.5]
            if len(low_authority_sources) > len(sources) / 2:
                violations.append("Majority of sources have low authority scores")
        
        is_compliant = len(violations) == 0
        
        details = {
            "mode": self.mode,
            "policies_checked": list(self.policies.keys()),
            "source_count": len(sources),
            "answer_length": len(answer)
        }
        
        recommendations = []
        if violations:
            recommendations.extend([
                "Review sources for authority and reliability",
                "Ensure answer provides sufficient context",
                "Verify no sensitive information is disclosed"
            ])
        
        return ComplianceResult(
            is_compliant=is_compliant,
            details=details,
            violations=violations,
            recommendations=recommendations
        )
    
    def _init_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize compliance policies."""
        return {
            "gdpr": {
                "name": "GDPR Data Protection",
                "enabled": True,
                "checks": ["no_personal_data", "data_minimization"]
            },
            "ai_governance": {
                "name": "AI Governance Framework",
                "enabled": True,
                "checks": ["source_transparency", "factual_grounding"]
            },
            "content_safety": {
                "name": "Content Safety",
                "enabled": True,
                "checks": ["no_harmful_content", "appropriate_language"]
            }
        }
    
    def get_compliance_stats(self) -> Dict[str, Any]:
        """Get compliance checker statistics."""
        return {
            "mode": self.mode,
            "policies": list(self.policies.keys()),
            "generation": 1,
            "method": "rule_based"
        }