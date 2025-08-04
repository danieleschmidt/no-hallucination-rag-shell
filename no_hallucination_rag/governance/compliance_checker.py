"""
Governance compliance checking for AI safety requirements.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ComplianceResult:
    """Result of governance compliance check."""
    is_compliant: bool
    details: Dict[str, Any]
    violations: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.recommendations is None:
            self.recommendations = []


class GovernanceChecker:
    """Checks AI governance compliance according to policy frameworks."""
    
    def __init__(
        self,
        mode: str = "strict",
        policies: Optional[List[str]] = None
    ):
        self.mode = mode
        self.policies = policies or ["whitehouse_2025", "nist_framework"]
        self.logger = logging.getLogger(__name__)
        
        # Load compliance rules
        self.compliance_rules = self._load_compliance_rules()
    
    def check_compliance(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> ComplianceResult:
        """
        Check governance compliance of RAG response.
        
        Args:
            question: User query
            answer: Generated answer
            sources: Source documents used
            
        Returns:
            ComplianceResult with compliance status and details
        """
        try:
            violations = []
            recommendations = []
            details = {
                "policies_checked": self.policies,
                "mode": self.mode,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check source attribution compliance
            source_compliance = self._check_source_attribution(answer, sources)
            if not source_compliance["compliant"]:
                violations.extend(source_compliance["violations"])
                recommendations.extend(source_compliance["recommendations"])
            
            # Check content safety compliance
            content_compliance = self._check_content_safety(question, answer)
            if not content_compliance["compliant"]:
                violations.extend(content_compliance["violations"])
                recommendations.extend(content_compliance["recommendations"])
            
            # Check transparency compliance
            transparency_compliance = self._check_transparency(answer, sources)
            if not transparency_compliance["compliant"]:
                violations.extend(transparency_compliance["violations"])
                recommendations.extend(transparency_compliance["recommendations"])
            
            # Check bias and fairness compliance
            bias_compliance = self._check_bias_fairness(question, answer)
            if not bias_compliance["compliant"]:
                violations.extend(bias_compliance["violations"])
                recommendations.extend(bias_compliance["recommendations"])
            
            # Determine overall compliance
            is_compliant = len(violations) == 0
            
            details.update({
                "source_attribution": source_compliance,
                "content_safety": content_compliance,
                "transparency": transparency_compliance,
                "bias_fairness": bias_compliance,
                "total_violations": len(violations)
            })
            
            self.logger.info(
                f"Compliance check completed: {'PASS' if is_compliant else 'FAIL'} "
                f"({len(violations)} violations)"
            )
            
            return ComplianceResult(
                is_compliant=is_compliant,
                details=details,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in compliance check: {e}")
            return ComplianceResult(
                is_compliant=False,
                details={"error": str(e)},
                violations=["System error during compliance check"],
                recommendations=["Review compliance checker configuration"]
            )
    
    def _check_source_attribution(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check compliance with source attribution requirements."""
        violations = []
        recommendations = []
        
        # Check if answer cites sources
        has_citations = any(marker in answer for marker in ['[', ']', '(', ')', 'Source:', 'According to'])
        
        if not has_citations and sources:
            violations.append("Missing source citations in response")
            recommendations.append("Include proper citations for all sources used")
        
        # Check source quality
        authoritative_sources = 0
        for source in sources:
            authority_score = source.get("authority_score", 0.0)
            if authority_score >= 0.8:
                authoritative_sources += 1
        
        if len(sources) > 0 and authoritative_sources / len(sources) < 0.5:
            violations.append("Insufficient authoritative sources")
            recommendations.append("Use more authoritative sources (gov, edu, established organizations)")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations,
            "metrics": {
                "has_citations": has_citations,
                "source_count": len(sources),
                "authoritative_ratio": authoritative_sources / max(len(sources), 1)
            }
        }
    
    def _check_content_safety(self, question: str, answer: str) -> Dict[str, Any]:
        """Check content safety compliance."""
        violations = []
        recommendations = []
        
        # Check for harmful content patterns (simplified for Generation 1)
        harmful_patterns = [
            'violence', 'weapon', 'harm', 'illegal', 'dangerous',
            'discriminat', 'hate', 'threat', 'attack'
        ]
        
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        for pattern in harmful_patterns:
            if pattern in answer_lower or pattern in question_lower:
                if not self._is_educational_context(answer):
                    violations.append(f"Potentially harmful content detected: {pattern}")
                    recommendations.append("Review content for harmful implications")
        
        # Check for misinformation risks
        speculation_patterns = ['might', 'could', 'possibly', 'potentially', 'may']
        speculation_count = sum(1 for pattern in speculation_patterns if pattern in answer_lower)
        
        if speculation_count > 3:
            violations.append("High speculation content without proper disclaimers")
            recommendations.append("Add uncertainty disclaimers for speculative content")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations,
            "metrics": {
                "harmful_pattern_count": len([p for p in harmful_patterns if p in answer_lower]),
                "speculation_count": speculation_count
            }
        }
    
    def _check_transparency(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check transparency and explainability compliance."""
        violations = []
        recommendations = []
        
        # Check for transparency indicators
        transparency_indicators = [
            'based on', 'according to', 'source', 'research shows',
            'study found', 'data indicates', 'evidence suggests'
        ]
        
        answer_lower = answer.lower()
        has_transparency = any(indicator in answer_lower for indicator in transparency_indicators)
        
        if not has_transparency and sources:
            violations.append("Insufficient transparency in reasoning process")
            recommendations.append("Include explicit references to reasoning and sources")
        
        # Check for confidence indicators
        confidence_indicators = ['uncertain', 'likely', 'probably', 'appears', 'suggests']
        has_confidence_indicators = any(indicator in answer_lower for indicator in confidence_indicators)
        
        # For uncertain topics, require confidence indicators
        uncertain_topics = ['future', 'prediction', 'will', 'expect', 'forecast']
        discusses_uncertainty = any(topic in answer_lower for topic in uncertain_topics)
        
        if discusses_uncertainty and not has_confidence_indicators:
            violations.append("Missing uncertainty indicators for speculative content")
            recommendations.append("Include confidence levels for uncertain predictions")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations,
            "metrics": {
                "has_transparency": has_transparency,
                "has_confidence_indicators": has_confidence_indicators,
                "discusses_uncertainty": discusses_uncertainty
            }
        }
    
    def _check_bias_fairness(self, question: str, answer: str) -> Dict[str, Any]:
        """Check bias and fairness compliance."""
        violations = []
        recommendations = []
        
        # Check for biased language (simplified for Generation 1)
        biased_terms = [
            'obviously', 'clearly', 'everyone knows', 'it is obvious',
            'common sense', 'naturally', 'of course'
        ]
        
        answer_lower = answer.lower()
        biased_language_count = sum(1 for term in biased_terms if term in answer_lower)
        
        if biased_language_count > 0:
            violations.append("Potentially biased language detected")
            recommendations.append("Use more neutral and objective language")
        
        # Check for balanced representation
        # Simple check for one-sided presentation
        balanced_indicators = [
            'however', 'although', 'while', 'on the other hand',
            'alternatively', 'different perspective', 'some argue'
        ]
        
        has_balance = any(indicator in answer_lower for indicator in balanced_indicators)
        discusses_controversial = any(term in (question + answer).lower() for term in [
            'debate', 'controversial', 'opinion', 'disagree', 'conflict'
        ])
        
        if discusses_controversial and not has_balance:
            violations.append("One-sided presentation of controversial topic")
            recommendations.append("Include multiple perspectives on debated topics")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations,
            "metrics": {
                "biased_language_count": biased_language_count,
                "has_balance": has_balance,
                "discusses_controversial": discusses_controversial
            }
        }
    
    def _is_educational_context(self, text: str) -> bool:
        """Check if content is in educational context."""
        educational_indicators = [
            'according to research', 'studies show', 'for educational purposes',
            'to understand', 'it is important to know', 'awareness of'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in educational_indicators)
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules for different policies."""
        # Simplified rules for Generation 1
        return {
            "whitehouse_2025": {
                "source_attribution_required": True,
                "transparency_required": True,
                "bias_checks_required": True,
                "harmful_content_forbidden": True
            },
            "nist_framework": {
                "risk_assessment_required": False,  # Not applicable for individual queries
                "documentation_required": True,
                "testing_required": False,  # Not applicable for individual queries
                "monitoring_required": True
            }
        }