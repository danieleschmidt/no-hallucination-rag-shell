"""
Compliance and regulatory framework for quantum task planning systems.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"                    # General Data Protection Regulation (EU)
    CCPA = "ccpa"                    # California Consumer Privacy Act (US)
    PDPA = "pdpa"                    # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"                    # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"                # Personal Information Protection (Canada)
    SOX = "sox"                      # Sarbanes-Oxley Act (US)
    ISO27001 = "iso27001"            # ISO 27001 Information Security
    NIST_CSF = "nist_csf"            # NIST Cybersecurity Framework
    PCI_DSS = "pci_dss"              # Payment Card Industry Data Security
    QUANTUM_SAFETY = "quantum_safety" # Quantum-specific safety regulations


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ComplianceStatus(Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    EXEMPT = "exempt"


@dataclass
class ComplianceRule:
    """Individual compliance rule definition."""
    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    data_types: Set[str] = field(default_factory=set)
    severity: str = "medium"  # low, medium, high, critical
    automated_check: bool = True
    remediation_guidance: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceCheck:
    """Result of a compliance check."""
    rule_id: str
    status: ComplianceStatus
    details: str
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    next_check_due: Optional[datetime] = None


class QuantumComplianceManager:
    """
    Comprehensive compliance manager for quantum task planning systems.
    
    Ensures adherence to global privacy regulations, data protection laws,
    and quantum-specific safety requirements.
    """
    
    def __init__(self, enabled_frameworks: Optional[List[ComplianceFramework]] = None):
        self.enabled_frameworks = enabled_frameworks or [
            ComplianceFramework.GDPR,
            ComplianceFramework.CCPA, 
            ComplianceFramework.ISO27001,
            ComplianceFramework.QUANTUM_SAFETY
        ]
        
        # Compliance rules database
        self.rules: Dict[str, ComplianceRule] = {}
        self.compliance_history: List[ComplianceCheck] = []
        
        # Data processing records
        self.data_processing_records: Dict[str, Dict[str, Any]] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_retention_policies: Dict[str, timedelta] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize compliance rules
        self._initialize_compliance_rules()
        
        self.logger.info(f"Compliance Manager initialized with frameworks: {[f.value for f in self.enabled_frameworks]}")
    
    def _initialize_compliance_rules(self) -> None:
        """Initialize compliance rules for enabled frameworks."""
        
        # GDPR Rules
        if ComplianceFramework.GDPR in self.enabled_frameworks:
            self._add_gdpr_rules()
        
        # CCPA Rules
        if ComplianceFramework.CCPA in self.enabled_frameworks:
            self._add_ccpa_rules()
        
        # ISO 27001 Rules
        if ComplianceFramework.ISO27001 in self.enabled_frameworks:
            self._add_iso27001_rules()
        
        # Quantum Safety Rules
        if ComplianceFramework.QUANTUM_SAFETY in self.enabled_frameworks:
            self._add_quantum_safety_rules()
    
    def _add_gdpr_rules(self) -> None:
        """Add GDPR compliance rules."""
        
        gdpr_rules = [
            ComplianceRule(
                rule_id="GDPR-001",
                framework=ComplianceFramework.GDPR,
                title="Data Processing Lawfulness",
                description="All personal data processing must have a legal basis",
                requirement="Article 6 - Lawfulness of processing",
                data_types={"personal_data", "user_id", "session_data"},
                severity="critical",
                remediation_guidance="Ensure valid consent or legitimate interest for all personal data processing"
            ),
            ComplianceRule(
                rule_id="GDPR-002", 
                framework=ComplianceFramework.GDPR,
                title="Data Subject Rights",
                description="Provide mechanisms for data subject rights exercise",
                requirement="Articles 15-22 - Rights of the data subject",
                data_types={"personal_data", "user_profiles"},
                severity="high",
                remediation_guidance="Implement access, rectification, erasure, and portability rights"
            ),
            ComplianceRule(
                rule_id="GDPR-003",
                framework=ComplianceFramework.GDPR,
                title="Data Minimization",
                description="Process only data necessary for the specific purpose",
                requirement="Article 5(1)(c) - Data minimization",
                data_types={"all_data_types"},
                severity="medium",
                remediation_guidance="Review data collection to ensure only necessary data is processed"
            ),
            ComplianceRule(
                rule_id="GDPR-004",
                framework=ComplianceFramework.GDPR,
                title="Data Retention Limits",
                description="Personal data must not be kept longer than necessary",
                requirement="Article 5(1)(e) - Storage limitation",
                data_types={"personal_data", "logs", "analytics"},
                severity="medium", 
                remediation_guidance="Implement automated data deletion based on retention policies"
            ),
            ComplianceRule(
                rule_id="GDPR-005",
                framework=ComplianceFramework.GDPR,
                title="Privacy by Design",
                description="Data protection must be built into system design",
                requirement="Article 25 - Data protection by design and by default",
                data_types={"system_architecture"},
                severity="high",
                remediation_guidance="Integrate privacy controls into system architecture and defaults"
            )
        ]
        
        for rule in gdpr_rules:
            self.rules[rule.rule_id] = rule
    
    def _add_ccpa_rules(self) -> None:
        """Add CCPA compliance rules."""
        
        ccpa_rules = [
            ComplianceRule(
                rule_id="CCPA-001",
                framework=ComplianceFramework.CCPA,
                title="Consumer Right to Know",
                description="Consumers have right to know what personal information is collected",
                requirement="CCPA Section 1798.100 - Right to Know",
                data_types={"personal_information", "user_data"},
                severity="high",
                remediation_guidance="Provide clear disclosure of data collection practices"
            ),
            ComplianceRule(
                rule_id="CCPA-002",
                framework=ComplianceFramework.CCPA, 
                title="Consumer Right to Delete",
                description="Consumers have right to request deletion of personal information",
                requirement="CCPA Section 1798.105 - Right to Delete",
                data_types={"personal_information"},
                severity="high",
                remediation_guidance="Implement verified deletion process for consumer requests"
            ),
            ComplianceRule(
                rule_id="CCPA-003",
                framework=ComplianceFramework.CCPA,
                title="Do Not Sell",
                description="Consumers have right to opt-out of sale of personal information", 
                requirement="CCPA Section 1798.120 - Right to Opt-Out",
                data_types={"personal_information"},
                severity="medium",
                remediation_guidance="Provide clear opt-out mechanism and honor requests"
            )
        ]
        
        for rule in ccpa_rules:
            self.rules[rule.rule_id] = rule
    
    def _add_iso27001_rules(self) -> None:
        """Add ISO 27001 compliance rules."""
        
        iso_rules = [
            ComplianceRule(
                rule_id="ISO27001-001",
                framework=ComplianceFramework.ISO27001,
                title="Information Security Management System",
                description="Establish and maintain ISMS",
                requirement="ISO 27001 Clause 4 - Context of the organization",
                data_types={"all_information_assets"},
                severity="critical",
                remediation_guidance="Document and implement comprehensive ISMS"
            ),
            ComplianceRule(
                rule_id="ISO27001-002",
                framework=ComplianceFramework.ISO27001,
                title="Risk Assessment",
                description="Conduct regular information security risk assessments",
                requirement="ISO 27001 Clause 6.1.2 - Information security risk assessment",
                data_types={"risk_data", "vulnerability_data"},
                severity="high",
                remediation_guidance="Implement systematic risk assessment methodology"
            ),
            ComplianceRule(
                rule_id="ISO27001-003",
                framework=ComplianceFramework.ISO27001,
                title="Access Control",
                description="Implement appropriate access controls",
                requirement="ISO 27001 Annex A.9 - Access control", 
                data_types={"access_logs", "user_credentials"},
                severity="high",
                remediation_guidance="Deploy role-based access controls and regular access reviews"
            )
        ]
        
        for rule in iso_rules:
            self.rules[rule.rule_id] = rule
    
    def _add_quantum_safety_rules(self) -> None:
        """Add quantum-specific safety and compliance rules."""
        
        quantum_rules = [
            ComplianceRule(
                rule_id="QS-001",
                framework=ComplianceFramework.QUANTUM_SAFETY,
                title="Quantum Coherence Protection",
                description="Protect quantum coherence from unauthorized observation",
                requirement="Quantum Safety Standard QSS-001",
                data_types={"quantum_states", "superposition_data"},
                severity="critical",
                remediation_guidance="Implement quantum state protection mechanisms"
            ),
            ComplianceRule(
                rule_id="QS-002",
                framework=ComplianceFramework.QUANTUM_SAFETY,
                title="Entanglement Security",
                description="Secure quantum entanglement channels",
                requirement="Quantum Safety Standard QSS-002",
                data_types={"entanglement_data", "quantum_correlations"},
                severity="high",
                remediation_guidance="Use quantum key distribution for entanglement security"
            ),
            ComplianceRule(
                rule_id="QS-003",
                framework=ComplianceFramework.QUANTUM_SAFETY,
                title="Measurement Audit",
                description="Audit all quantum measurements for unauthorized access",
                requirement="Quantum Safety Standard QSS-003",
                data_types={"measurement_logs", "observation_records"},
                severity="medium",
                remediation_guidance="Log and review all quantum state observations"
            ),
            ComplianceRule(
                rule_id="QS-004",
                framework=ComplianceFramework.QUANTUM_SAFETY,
                title="Decoherence Prevention",
                description="Prevent malicious decoherence attacks",
                requirement="Quantum Safety Standard QSS-004",
                data_types={"coherence_metrics", "decoherence_logs"},
                severity="high",
                remediation_guidance="Monitor and protect against decoherence attacks"
            ),
            ComplianceRule(
                rule_id="QS-005",
                framework=ComplianceFramework.QUANTUM_SAFETY,
                title="Bell Inequality Monitoring",
                description="Monitor Bell inequality violations for quantum integrity",
                requirement="Quantum Safety Standard QSS-005",
                data_types={"bell_test_results", "correlation_data"},
                severity="medium",
                automated_check=True,
                remediation_guidance="Track Bell violations to ensure quantum entanglement integrity"
            )
        ]
        
        for rule in quantum_rules:
            self.rules[rule.rule_id] = rule
    
    def check_compliance(self, 
                        data_context: Dict[str, Any], 
                        frameworks: Optional[List[ComplianceFramework]] = None) -> List[ComplianceCheck]:
        """Perform comprehensive compliance check."""
        
        frameworks = frameworks or self.enabled_frameworks
        compliance_results = []
        
        for rule_id, rule in self.rules.items():
            if rule.framework not in frameworks:
                continue
            
            # Perform rule-specific compliance check
            check_result = self._check_rule_compliance(rule, data_context)
            compliance_results.append(check_result)
            
            # Add to compliance history
            self.compliance_history.append(check_result)
        
        # Trim compliance history to last 1000 entries
        if len(self.compliance_history) > 1000:
            self.compliance_history = self.compliance_history[-1000:]
        
        self.logger.info(f"Compliance check completed: {len(compliance_results)} rules evaluated")
        
        return compliance_results
    
    def _check_rule_compliance(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Check compliance for a specific rule."""
        
        try:
            # Rule-specific compliance checks
            if rule.rule_id == "GDPR-001":
                return self._check_gdpr_lawful_basis(rule, context)
            elif rule.rule_id == "GDPR-002":
                return self._check_gdpr_data_subject_rights(rule, context)
            elif rule.rule_id == "GDPR-003":
                return self._check_gdpr_data_minimization(rule, context)
            elif rule.rule_id == "GDPR-004":
                return self._check_gdpr_retention_limits(rule, context)
            elif rule.rule_id == "CCPA-001":
                return self._check_ccpa_right_to_know(rule, context)
            elif rule.rule_id == "QS-001":
                return self._check_quantum_coherence_protection(rule, context)
            elif rule.rule_id == "QS-005":
                return self._check_bell_inequality_monitoring(rule, context)
            else:
                # Generic compliance check
                return self._generic_compliance_check(rule, context)
                
        except Exception as e:
            self.logger.error(f"Compliance check failed for rule {rule.rule_id}: {e}")
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.UNDER_REVIEW,
                details=f"Check failed due to error: {e}",
                recommendations=["Review compliance check implementation"]
            )
    
    def _check_gdpr_lawful_basis(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Check GDPR lawful basis requirement."""
        
        has_consent = context.get("user_consent", False)
        has_legitimate_interest = context.get("legitimate_interest", False)
        has_legal_obligation = context.get("legal_obligation", False)
        
        if has_consent or has_legitimate_interest or has_legal_obligation:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.COMPLIANT,
                details="Valid legal basis for processing identified",
                evidence=[f"Consent: {has_consent}", f"Legitimate interest: {has_legitimate_interest}"]
            )
        else:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.NON_COMPLIANT,
                details="No valid legal basis for personal data processing",
                recommendations=["Obtain user consent or establish legitimate interest"]
            )
    
    def _check_gdpr_data_subject_rights(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Check GDPR data subject rights implementation."""
        
        rights_implemented = context.get("data_subject_rights", [])
        required_rights = ["access", "rectification", "erasure", "portability"]
        
        implemented_count = sum(1 for right in required_rights if right in rights_implemented)
        
        if implemented_count >= len(required_rights):
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.COMPLIANT,
                details="All required data subject rights implemented",
                evidence=[f"Implemented rights: {', '.join(rights_implemented)}"]
            )
        else:
            missing_rights = [right for right in required_rights if right not in rights_implemented]
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                details=f"Missing data subject rights: {', '.join(missing_rights)}",
                recommendations=[f"Implement missing rights: {', '.join(missing_rights)}"]
            )
    
    def _check_gdpr_data_minimization(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Check GDPR data minimization principle."""
        
        data_collected = context.get("data_fields_collected", [])
        data_necessary = context.get("data_fields_necessary", [])
        
        if not data_collected:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.COMPLIANT,
                details="No data collection detected"
            )
        
        unnecessary_data = [field for field in data_collected if field not in data_necessary]
        
        if not unnecessary_data:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.COMPLIANT,
                details="Only necessary data is collected",
                evidence=[f"Necessary fields: {', '.join(data_necessary)}"]
            )
        else:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.NON_COMPLIANT,
                details=f"Unnecessary data collected: {', '.join(unnecessary_data)}",
                recommendations=["Remove collection of unnecessary data fields"]
            )
    
    def _check_gdpr_retention_limits(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Check GDPR data retention limits."""
        
        data_age = context.get("data_age_days", 0)
        retention_limit = context.get("retention_limit_days", 365)
        
        if data_age <= retention_limit:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.COMPLIANT,
                details=f"Data within retention limit ({data_age}/{retention_limit} days)"
            )
        else:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.NON_COMPLIANT,
                details=f"Data exceeds retention limit ({data_age}/{retention_limit} days)",
                recommendations=["Implement automated data deletion process"]
            )
    
    def _check_ccpa_right_to_know(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Check CCPA right to know implementation."""
        
        privacy_notice_provided = context.get("privacy_notice_provided", False)
        data_disclosure_available = context.get("data_disclosure_mechanism", False)
        
        if privacy_notice_provided and data_disclosure_available:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.COMPLIANT,
                details="Privacy notice and data disclosure mechanisms in place"
            )
        else:
            missing_elements = []
            if not privacy_notice_provided:
                missing_elements.append("privacy notice")
            if not data_disclosure_available:
                missing_elements.append("data disclosure mechanism")
            
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.NON_COMPLIANT,
                details=f"Missing: {', '.join(missing_elements)}",
                recommendations=["Provide clear privacy notice and data disclosure mechanism"]
            )
    
    def _check_quantum_coherence_protection(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Check quantum coherence protection measures."""
        
        coherence_protection_enabled = context.get("coherence_protection", False)
        unauthorized_observations = context.get("unauthorized_observations", 0)
        
        if coherence_protection_enabled and unauthorized_observations == 0:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.COMPLIANT,
                details="Quantum coherence protection active, no unauthorized observations"
            )
        elif unauthorized_observations > 0:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.NON_COMPLIANT,
                details=f"Unauthorized quantum observations detected: {unauthorized_observations}",
                recommendations=["Strengthen quantum state access controls"]
            )
        else:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                details="Coherence protection not fully enabled",
                recommendations=["Enable comprehensive quantum coherence protection"]
            )
    
    def _check_bell_inequality_monitoring(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Check Bell inequality violation monitoring."""
        
        bell_violations = context.get("bell_violations", [])
        total_entanglements = context.get("total_entanglements", 0)
        
        if total_entanglements == 0:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.EXEMPT,
                details="No quantum entanglements present"
            )
        
        violation_rate = len(bell_violations) / total_entanglements if total_entanglements > 0 else 0
        
        if violation_rate >= 0.1:  # At least 10% should show violations for true quantum entanglement
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.COMPLIANT,
                details=f"Bell violations detected in {violation_rate:.1%} of entanglements (indicates genuine quantum effects)",
                evidence=[f"Violations: {len(bell_violations)}/{total_entanglements}"]
            )
        else:
            return ComplianceCheck(
                rule_id=rule.rule_id,
                status=ComplianceStatus.UNDER_REVIEW,
                details=f"Low Bell violation rate ({violation_rate:.1%}) - verify quantum entanglement integrity",
                recommendations=["Review entanglement implementation and measurement procedures"]
            )
    
    def _generic_compliance_check(self, rule: ComplianceRule, context: Dict[str, Any]) -> ComplianceCheck:
        """Generic compliance check for rules without specific implementation."""
        
        return ComplianceCheck(
            rule_id=rule.rule_id,
            status=ComplianceStatus.UNDER_REVIEW,
            details="Manual review required - no automated check available",
            recommendations=["Conduct manual compliance review"]
        )
    
    def record_data_processing(self, 
                             processing_id: str,
                             purpose: str,
                             data_types: List[str],
                             legal_basis: str,
                             retention_period: timedelta,
                             data_subjects: Optional[List[str]] = None) -> None:
        """Record data processing activity for compliance tracking."""
        
        self.data_processing_records[processing_id] = {
            "purpose": purpose,
            "data_types": data_types,
            "legal_basis": legal_basis,
            "retention_period": retention_period,
            "data_subjects": data_subjects or [],
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow()
        }
        
        # Set retention policy
        self.data_retention_policies[processing_id] = retention_period
        
        self.logger.info(f"Recorded data processing activity: {processing_id}")
    
    def record_consent(self, 
                      user_id: str,
                      consent_types: List[str],
                      consent_given: bool,
                      consent_method: str = "explicit") -> None:
        """Record user consent for GDPR compliance."""
        
        consent_record = {
            "consent_types": consent_types,
            "consent_given": consent_given,
            "consent_method": consent_method,
            "timestamp": datetime.utcnow(),
            "user_agent": None,  # Could be populated from request context
            "ip_address": None   # Could be populated from request context
        }
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent_record)
        
        self.logger.info(f"Recorded consent for user {user_id}: {consent_given}")
    
    def get_compliance_report(self, 
                             frameworks: Optional[List[ComplianceFramework]] = None,
                             start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        frameworks = frameworks or self.enabled_frameworks
        start_date = start_date or (datetime.utcnow() - timedelta(days=30))
        
        # Filter compliance history
        relevant_checks = [
            check for check in self.compliance_history
            if check.checked_at >= start_date
        ]
        
        # Calculate compliance metrics
        total_checks = len(relevant_checks)
        compliant_checks = sum(1 for check in relevant_checks if check.status == ComplianceStatus.COMPLIANT)
        non_compliant_checks = sum(1 for check in relevant_checks if check.status == ComplianceStatus.NON_COMPLIANT)
        
        compliance_rate = compliant_checks / total_checks if total_checks > 0 else 0
        
        # Framework-specific metrics
        framework_metrics = {}
        for framework in frameworks:
            framework_checks = [check for check in relevant_checks 
                              if self.rules.get(check.rule_id, {}).framework == framework]
            
            if framework_checks:
                framework_compliant = sum(1 for check in framework_checks 
                                        if check.status == ComplianceStatus.COMPLIANT)
                framework_metrics[framework.value] = {
                    "total_checks": len(framework_checks),
                    "compliant": framework_compliant,
                    "compliance_rate": framework_compliant / len(framework_checks)
                }
        
        # Non-compliance issues
        non_compliance_issues = []
        for check in relevant_checks:
            if check.status == ComplianceStatus.NON_COMPLIANT:
                rule = self.rules.get(check.rule_id)
                non_compliance_issues.append({
                    "rule_id": check.rule_id,
                    "rule_title": rule.title if rule else "Unknown",
                    "framework": rule.framework.value if rule else "Unknown",
                    "severity": rule.severity if rule else "medium",
                    "details": check.details,
                    "recommendations": check.recommendations
                })
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": datetime.utcnow().isoformat()
            },
            "overall_metrics": {
                "total_checks": total_checks,
                "compliant_checks": compliant_checks,
                "non_compliant_checks": non_compliant_checks,
                "compliance_rate": compliance_rate,
                "frameworks_enabled": [f.value for f in frameworks]
            },
            "framework_metrics": framework_metrics,
            "non_compliance_issues": non_compliance_issues,
            "data_processing_activities": len(self.data_processing_records),
            "consent_records": len(self.consent_records),
            "recommendations": self._generate_compliance_recommendations(non_compliance_issues)
        }
    
    def _generate_compliance_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on compliance issues."""
        
        recommendations = []
        
        # High severity issues
        high_severity_count = sum(1 for issue in issues if issue.get("severity") == "high")
        if high_severity_count > 0:
            recommendations.append(f"Address {high_severity_count} high-severity compliance issues immediately")
        
        # Framework-specific recommendations
        gdpr_issues = sum(1 for issue in issues if issue.get("framework") == "gdpr")
        if gdpr_issues > 0:
            recommendations.append("Review and strengthen GDPR compliance measures")
        
        quantum_issues = sum(1 for issue in issues if issue.get("framework") == "quantum_safety")
        if quantum_issues > 0:
            recommendations.append("Enhance quantum safety and security controls")
        
        # General recommendations
        if len(issues) > 5:
            recommendations.append("Consider implementing automated compliance monitoring")
        
        return recommendations
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up data based on retention policies."""
        
        cleanup_stats = {"processing_records": 0, "consent_records": 0}
        
        # Clean up processing records
        current_time = datetime.utcnow()
        expired_processing = []
        
        for processing_id, record in self.data_processing_records.items():
            retention_period = self.data_retention_policies.get(processing_id, timedelta(days=365))
            if current_time - record["created_at"] > retention_period:
                expired_processing.append(processing_id)
        
        for processing_id in expired_processing:
            del self.data_processing_records[processing_id]
            self.data_retention_policies.pop(processing_id, None)
            cleanup_stats["processing_records"] += 1
        
        # Clean up old consent records (keep last 5 years)
        consent_retention = timedelta(days=365 * 5)
        for user_id, consents in self.consent_records.items():
            original_count = len(consents)
            self.consent_records[user_id] = [
                consent for consent in consents
                if current_time - consent["timestamp"] <= consent_retention
            ]
            cleanup_stats["consent_records"] += original_count - len(self.consent_records[user_id])
        
        self.logger.info(f"Compliance data cleanup: {cleanup_stats}")
        
        return cleanup_stats
    
    def export_compliance_data(self) -> Dict[str, Any]:
        """Export compliance data for auditing purposes."""
        
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "enabled_frameworks": [f.value for f in self.enabled_frameworks],
            "compliance_rules": {
                rule_id: {
                    "framework": rule.framework.value,
                    "title": rule.title,
                    "description": rule.description,
                    "severity": rule.severity,
                    "last_updated": rule.last_updated.isoformat()
                }
                for rule_id, rule in self.rules.items()
            },
            "recent_compliance_checks": [
                {
                    "rule_id": check.rule_id,
                    "status": check.status.value,
                    "details": check.details,
                    "checked_at": check.checked_at.isoformat()
                }
                for check in self.compliance_history[-100:]  # Last 100 checks
            ],
            "data_processing_summary": {
                "total_activities": len(self.data_processing_records),
                "activities": list(self.data_processing_records.keys())
            },
            "consent_summary": {
                "total_users": len(self.consent_records),
                "total_consent_records": sum(len(consents) for consents in self.consent_records.values())
            }
        }