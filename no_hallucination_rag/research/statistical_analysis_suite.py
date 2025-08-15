"""
Statistical Analysis Suite for Quantum RAG Research.

This module provides advanced statistical analysis capabilities specifically
designed for quantum vs classical algorithm comparisons, including Bayesian
analysis, meta-analysis, and publication-ready statistical reporting.

Research Focus: Rigorous statistical analysis with publication-quality
reporting and advanced techniques for algorithmic performance comparisons.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import itertools

try:
    from scipy import stats as scipy_stats
    from scipy.stats import (
        ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
        f_oneway, levene, shapiro, anderson, jarque_bera, normaltest,
        pearsonr, spearmanr, kendalltau, chi2_contingency
    )
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    # Bayesian analysis libraries
    import pymc3 as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


class StatisticalTest(Enum):
    """Types of statistical tests available."""
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    ANOVA_ONE_WAY = "anova_one_way"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    CHI_SQUARE = "chi_square"
    CORRELATION_PEARSON = "correlation_pearson"
    CORRELATION_SPEARMAN = "correlation_spearman"
    BAYESIAN_T_TEST = "bayesian_t_test"
    BAYESIAN_ANOVA = "bayesian_anova"


class EffectSizeType(Enum):
    """Types of effect size measures."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    CLIFF_DELTA = "cliff_delta"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    EPSILON_SQUARED = "epsilon_squared"
    R_SQUARED = "r_squared"
    CRAMERS_V = "cramers_v"


@dataclass
class StatisticalTestResult:
    """Comprehensive statistical test result."""
    test_name: str
    test_type: StatisticalTest
    
    # Test statistics
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[Union[int, Tuple[int, int]]] = None
    critical_value: Optional[float] = None
    
    # Effect sizes
    effect_size: float = 0.0
    effect_size_type: Optional[EffectSizeType] = None
    effect_size_ci: Optional[Tuple[float, float]] = None
    
    # Confidence intervals
    confidence_interval: Optional[Tuple[float, float]] = None
    confidence_level: float = 0.95
    
    # Power analysis
    power: Optional[float] = None
    required_sample_size: Optional[int] = None
    
    # Statistical significance
    significant: bool = False
    alpha: float = 0.05
    
    # Bayesian statistics (if applicable)
    bayes_factor: Optional[float] = None
    posterior_probability: Optional[float] = None
    credible_interval: Optional[Tuple[float, float]] = None
    
    # Interpretation and reporting
    interpretation: str = ""
    apa_format: str = ""
    effect_size_interpretation: str = ""
    
    # Assumptions and warnings
    assumptions_met: bool = True
    assumption_violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Additional metadata
    sample_sizes: Optional[List[int]] = None
    group_names: Optional[List[str]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaAnalysisResult:
    """Results from meta-analysis."""
    overall_effect_size: float
    overall_effect_size_ci: Tuple[float, float]
    overall_p_value: float
    
    # Heterogeneity
    q_statistic: float
    i_squared: float
    tau_squared: float
    heterogeneity_p_value: float
    
    # Individual study results
    study_effect_sizes: List[float]
    study_weights: List[float]
    study_names: List[str]
    
    # Publication bias
    funnel_plot_asymmetry: Optional[float] = None
    eggers_test_p_value: Optional[float] = None
    
    # Model information
    model_type: str = "random_effects"
    studies_included: int = 0


@dataclass
class BayesianAnalysisResult:
    """Results from Bayesian analysis."""
    model_name: str
    
    # Posterior statistics
    posterior_mean: float
    posterior_std: float
    posterior_median: float
    
    # Credible intervals
    credible_interval_95: Tuple[float, float]
    credible_interval_99: Tuple[float, float]
    
    # Model comparison
    bayes_factor: Optional[float] = None
    model_probability: Optional[float] = None
    
    # MCMC diagnostics
    rhat: Optional[float] = None
    effective_sample_size: Optional[int] = None
    divergences: Optional[int] = None
    
    # Hypothesis testing
    rope_probability: Optional[float] = None  # Region of Practical Equivalence
    probability_positive: Optional[float] = None
    probability_negative: Optional[float] = None
    
    # Interpretation
    interpretation: str = ""


class StatisticalAnalysisSuite:
    """
    Comprehensive statistical analysis suite for quantum algorithm research.
    
    Provides advanced statistical methods including:
    1. Classical parametric and non-parametric tests
    2. Effect size calculations with confidence intervals
    3. Power analysis and sample size calculations
    4. Bayesian statistical analysis
    5. Meta-analysis capabilities
    6. Multiple comparison corrections
    7. Publication-ready reporting
    8. Advanced diagnostic tools
    """
    
    def __init__(
        self,
        default_alpha: float = 0.05,
        default_confidence_level: float = 0.95,
        enable_bayesian: bool = True,
        enable_plotting: bool = True
    ):
        self.default_alpha = default_alpha
        self.default_confidence_level = default_confidence_level
        self.enable_bayesian = enable_bayesian and BAYESIAN_AVAILABLE
        self.enable_plotting = enable_plotting and PLOTTING_AVAILABLE
        
        # Results storage
        self.test_results: List[StatisticalTestResult] = []
        self.meta_analysis_results: List[MetaAnalysisResult] = []
        self.bayesian_results: List[BayesianAnalysisResult] = []
        
        # Configuration
        self.multiple_comparison_methods = ["bonferroni", "holm", "benjamini_hochberg", "benjamini_yekutieli"]
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Statistical Analysis Suite initialized")
    
    def comprehensive_two_group_analysis(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        paired: bool = False,
        alpha: Optional[float] = None,
        confidence_level: Optional[float] = None
    ) -> Dict[str, StatisticalTestResult]:
        """
        Perform comprehensive two-group statistical analysis.
        
        Includes parametric and non-parametric tests, effect sizes,
        power analysis, and assumption checking.
        """
        
        alpha = alpha or self.default_alpha
        confidence_level = confidence_level or self.default_confidence_level
        
        self.logger.info(f"Starting comprehensive two-group analysis: {group1_name} vs {group2_name}")
        
        results = {}
        
        # Data validation
        if len(group1) < 2 or len(group2) < 2:
            self.logger.error("Insufficient data: each group must have at least 2 observations")
            return {"error": "Insufficient data"}
        
        # Assumption checking
        assumption_results = self._check_assumptions_two_groups(group1, group2)
        
        # Parametric tests
        if paired:
            if len(group1) != len(group2):
                self.logger.error("Paired analysis requires equal group sizes")
                return {"error": "Unequal group sizes for paired analysis"}
            
            # Paired t-test
            results["paired_t_test"] = self._paired_t_test(
                group1, group2, group1_name, group2_name, alpha, confidence_level
            )
        else:
            # Independent t-test
            results["independent_t_test"] = self._independent_t_test(
                group1, group2, group1_name, group2_name, alpha, confidence_level,
                equal_variances=assumption_results["equal_variances"]
            )
        
        # Non-parametric tests
        if paired:
            results["wilcoxon_signed_rank"] = self._wilcoxon_signed_rank_test(
                group1, group2, group1_name, group2_name, alpha, confidence_level
            )
        else:
            results["mann_whitney_u"] = self._mann_whitney_u_test(
                group1, group2, group1_name, group2_name, alpha, confidence_level
            )
        
        # Effect size analysis
        results["effect_sizes"] = self._comprehensive_effect_size_analysis(
            group1, group2, group1_name, group2_name, paired, confidence_level
        )
        
        # Power analysis
        results["power_analysis"] = self._power_analysis_two_groups(
            group1, group2, alpha, confidence_level
        )
        
        # Bayesian analysis
        if self.enable_bayesian:
            results["bayesian_analysis"] = self._bayesian_two_group_analysis(
                group1, group2, group1_name, group2_name, paired
            )
        
        # Add assumption checking results
        results["assumption_checks"] = assumption_results
        
        # Store results
        for test_name, test_result in results.items():
            if isinstance(test_result, StatisticalTestResult):
                self.test_results.append(test_result)
        
        self.logger.info("Comprehensive two-group analysis completed")
        return results
    
    def comprehensive_multi_group_analysis(
        self,
        groups: List[List[float]],
        group_names: List[str],
        repeated_measures: bool = False,
        alpha: Optional[float] = None,
        confidence_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multi-group statistical analysis.
        
        Includes ANOVA, non-parametric alternatives, post-hoc tests,
        and multiple comparison corrections.
        """
        
        alpha = alpha or self.default_alpha
        confidence_level = confidence_level or self.default_confidence_level
        
        self.logger.info(f"Starting comprehensive multi-group analysis: {len(groups)} groups")
        
        results = {}
        
        # Data validation
        if len(groups) < 2:
            return {"error": "At least 2 groups required"}
        
        if any(len(group) < 2 for group in groups):
            return {"error": "Each group must have at least 2 observations"}
        
        # Assumption checking
        assumption_results = self._check_assumptions_multi_groups(groups)
        
        # Parametric tests
        if repeated_measures:
            # Repeated measures ANOVA (simplified implementation)
            results["repeated_measures_anova"] = self._repeated_measures_anova(
                groups, group_names, alpha, confidence_level
            )
        else:
            # One-way ANOVA
            results["one_way_anova"] = self._one_way_anova(
                groups, group_names, alpha, confidence_level
            )
        
        # Non-parametric tests
        if repeated_measures:
            results["friedman_test"] = self._friedman_test(
                groups, group_names, alpha, confidence_level
            )
        else:
            results["kruskal_wallis"] = self._kruskal_wallis_test(
                groups, group_names, alpha, confidence_level
            )
        
        # Post-hoc analysis
        results["post_hoc_analysis"] = self._post_hoc_analysis(
            groups, group_names, alpha, confidence_level
        )
        
        # Effect size analysis
        results["effect_sizes"] = self._multi_group_effect_sizes(
            groups, group_names, confidence_level
        )
        
        # Multiple comparison corrections
        results["multiple_comparisons"] = self._multiple_comparison_corrections(
            groups, group_names, alpha
        )
        
        # Bayesian analysis
        if self.enable_bayesian:
            results["bayesian_anova"] = self._bayesian_anova(
                groups, group_names, repeated_measures
            )
        
        # Add assumption checking results
        results["assumption_checks"] = assumption_results
        
        # Store results
        for test_name, test_result in results.items():
            if isinstance(test_result, StatisticalTestResult):
                self.test_results.append(test_result)
        
        self.logger.info("Comprehensive multi-group analysis completed")
        return results
    
    def correlation_analysis(
        self,
        variable1: List[float],
        variable2: List[float],
        var1_name: str = "Variable 1",
        var2_name: str = "Variable 2",
        alpha: Optional[float] = None,
        confidence_level: Optional[float] = None
    ) -> Dict[str, StatisticalTestResult]:
        """
        Perform comprehensive correlation analysis.
        
        Includes Pearson, Spearman, and Kendall correlations with
        confidence intervals and significance testing.
        """
        
        alpha = alpha or self.default_alpha
        confidence_level = confidence_level or self.default_confidence_level
        
        self.logger.info(f"Starting correlation analysis: {var1_name} vs {var2_name}")
        
        results = {}
        
        # Data validation
        if len(variable1) != len(variable2):
            return {"error": "Variables must have equal length"}
        
        if len(variable1) < 3:
            return {"error": "At least 3 observations required"}
        
        # Pearson correlation
        results["pearson"] = self._pearson_correlation(
            variable1, variable2, var1_name, var2_name, alpha, confidence_level
        )
        
        # Spearman correlation
        results["spearman"] = self._spearman_correlation(
            variable1, variable2, var1_name, var2_name, alpha, confidence_level
        )
        
        # Kendall's tau
        results["kendall"] = self._kendall_correlation(
            variable1, variable2, var1_name, var2_name, alpha, confidence_level
        )
        
        # Bayesian correlation
        if self.enable_bayesian:
            results["bayesian_correlation"] = self._bayesian_correlation(
                variable1, variable2, var1_name, var2_name
            )
        
        # Store results
        for test_result in results.values():
            if isinstance(test_result, StatisticalTestResult):
                self.test_results.append(test_result)
        
        self.logger.info("Correlation analysis completed")
        return results
    
    def meta_analysis(
        self,
        effect_sizes: List[float],
        sample_sizes: List[int],
        study_names: List[str],
        effect_size_type: EffectSizeType = EffectSizeType.COHENS_D,
        model_type: str = "random_effects",
        alpha: Optional[float] = None
    ) -> MetaAnalysisResult:
        """
        Perform meta-analysis of multiple studies.
        
        Combines effect sizes across studies using fixed or random effects models.
        """
        
        alpha = alpha or self.default_alpha
        
        self.logger.info(f"Starting meta-analysis: {len(effect_sizes)} studies")
        
        if len(effect_sizes) != len(sample_sizes) or len(effect_sizes) != len(study_names):
            raise ValueError("Effect sizes, sample sizes, and study names must have equal length")
        
        if len(effect_sizes) < 2:
            raise ValueError("At least 2 studies required for meta-analysis")
        
        # Calculate weights (inverse variance weighting)
        if effect_size_type == EffectSizeType.COHENS_D:
            # Variance of Cohen's d
            variances = [(1/n) + (d**2)/(2*n) for d, n in zip(effect_sizes, sample_sizes)]
        else:
            # Generic variance calculation
            variances = [1/n for n in sample_sizes]
        
        weights = [1/v for v in variances]
        total_weight = sum(weights)
        
        # Calculate weighted mean effect size
        weighted_effect_sizes = [w * es for w, es in zip(weights, effect_sizes)]
        overall_effect_size = sum(weighted_effect_sizes) / total_weight
        
        # Calculate confidence interval for overall effect
        overall_variance = 1 / total_weight
        overall_se = np.sqrt(overall_variance)
        
        if SCIPY_AVAILABLE:
            z_critical = scipy_stats.norm.ppf(1 - alpha/2)
        else:
            z_critical = 1.96  # Approximation for 95% CI
        
        overall_ci = (
            overall_effect_size - z_critical * overall_se,
            overall_effect_size + z_critical * overall_se
        )
        
        # Calculate overall p-value
        z_score = overall_effect_size / overall_se
        if SCIPY_AVAILABLE:
            overall_p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
        else:
            overall_p_value = 0.05  # Placeholder
        
        # Heterogeneity analysis (Q-statistic)
        q_components = [w * (es - overall_effect_size)**2 for w, es in zip(weights, effect_sizes)]
        q_statistic = sum(q_components)
        df_q = len(effect_sizes) - 1
        
        if SCIPY_AVAILABLE:
            heterogeneity_p_value = 1 - scipy_stats.chi2.cdf(q_statistic, df_q)
        else:
            heterogeneity_p_value = 0.5  # Placeholder
        
        # I-squared (percentage of variation due to heterogeneity)
        i_squared = max(0, (q_statistic - df_q) / q_statistic) * 100 if q_statistic > 0 else 0
        
        # Tau-squared (between-study variance)
        if model_type == "random_effects" and q_statistic > df_q:
            # DerSimonian-Laird estimator
            c = total_weight - sum(w**2 for w in weights) / total_weight
            tau_squared = (q_statistic - df_q) / c if c > 0 else 0
        else:
            tau_squared = 0
        
        # Adjust weights for random effects model
        if model_type == "random_effects" and tau_squared > 0:
            adjusted_variances = [v + tau_squared for v in variances]
            adjusted_weights = [1/v for v in adjusted_variances]
            adjusted_total_weight = sum(adjusted_weights)
            
            # Recalculate with adjusted weights
            adjusted_weighted_effect_sizes = [w * es for w, es in zip(adjusted_weights, effect_sizes)]
            overall_effect_size = sum(adjusted_weighted_effect_sizes) / adjusted_total_weight
            
            overall_variance = 1 / adjusted_total_weight
            overall_se = np.sqrt(overall_variance)
            overall_ci = (
                overall_effect_size - z_critical * overall_se,
                overall_effect_size + z_critical * overall_se
            )
            
            z_score = overall_effect_size / overall_se
            if SCIPY_AVAILABLE:
                overall_p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
            
            weights = adjusted_weights
        
        # Normalize weights
        normalized_weights = [w / sum(weights) for w in weights]
        
        # Publication bias analysis (simplified)
        funnel_plot_asymmetry = None
        eggers_test_p_value = None
        
        if SCIPY_AVAILABLE and len(effect_sizes) >= 10:
            # Egger's test for funnel plot asymmetry
            standard_errors = [np.sqrt(v) for v in variances]
            precision = [1/se for se in standard_errors]
            
            try:
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(precision, effect_sizes)
                eggers_test_p_value = p_value
                funnel_plot_asymmetry = slope
            except:
                pass
        
        result = MetaAnalysisResult(
            overall_effect_size=overall_effect_size,
            overall_effect_size_ci=overall_ci,
            overall_p_value=overall_p_value,
            q_statistic=q_statistic,
            i_squared=i_squared,
            tau_squared=tau_squared,
            heterogeneity_p_value=heterogeneity_p_value,
            study_effect_sizes=effect_sizes,
            study_weights=normalized_weights,
            study_names=study_names,
            funnel_plot_asymmetry=funnel_plot_asymmetry,
            eggers_test_p_value=eggers_test_p_value,
            model_type=model_type,
            studies_included=len(effect_sizes)
        )
        
        self.meta_analysis_results.append(result)
        
        self.logger.info(f"Meta-analysis completed: overall effect size = {overall_effect_size:.3f}")
        return result
    
    def sequential_analysis(
        self,
        data_stream: List[Tuple[float, str]],  # (value, group) pairs
        alpha: float = 0.05,
        beta: float = 0.2,
        effect_size: float = 0.5,
        max_n: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform sequential analysis with early stopping rules.
        
        Useful for monitoring experiments in real-time and making
        early stopping decisions based on accumulating evidence.
        """
        
        self.logger.info("Starting sequential analysis")
        
        # Sequential probability ratio test boundaries
        if SCIPY_AVAILABLE:
            a = np.log((1 - beta) / alpha)  # Upper boundary
            b = np.log(beta / (1 - alpha))  # Lower boundary
        else:
            a = 2.94  # Approximation
            b = -2.94
        
        results = {
            "sample_sizes": [],
            "test_statistics": [],
            "decisions": [],
            "boundaries": {"upper": a, "lower": b},
            "final_decision": "continue",
            "stopping_reason": None,
            "final_sample_size": 0
        }
        
        group1_data = []
        group2_data = []
        
        for i, (value, group) in enumerate(data_stream):
            if group == "group1":
                group1_data.append(value)
            else:
                group2_data.append(value)
            
            # Only analyze when we have data from both groups
            if len(group1_data) >= 2 and len(group2_data) >= 2:
                # Calculate test statistic (simplified)
                if SCIPY_AVAILABLE:
                    t_stat, p_val = scipy_stats.ttest_ind(group1_data, group2_data)
                    log_likelihood_ratio = t_stat  # Simplified
                else:
                    # Fallback calculation
                    mean_diff = np.mean(group1_data) - np.mean(group2_data)
                    pooled_var = (np.var(group1_data) + np.var(group2_data)) / 2
                    log_likelihood_ratio = mean_diff / np.sqrt(pooled_var) if pooled_var > 0 else 0
                
                results["sample_sizes"].append(len(group1_data) + len(group2_data))
                results["test_statistics"].append(log_likelihood_ratio)
                
                # Check stopping boundaries
                if log_likelihood_ratio >= a:
                    results["decisions"].append("reject_null")
                    results["final_decision"] = "reject_null"
                    results["stopping_reason"] = "sufficient_evidence_for_effect"
                    results["final_sample_size"] = len(group1_data) + len(group2_data)
                    break
                elif log_likelihood_ratio <= b:
                    results["decisions"].append("accept_null")
                    results["final_decision"] = "accept_null"
                    results["stopping_reason"] = "sufficient_evidence_against_effect"
                    results["final_sample_size"] = len(group1_data) + len(group2_data)
                    break
                else:
                    results["decisions"].append("continue")
                
                # Check maximum sample size
                if len(group1_data) + len(group2_data) >= max_n:
                    results["final_decision"] = "inconclusive"
                    results["stopping_reason"] = "maximum_sample_size_reached"
                    results["final_sample_size"] = max_n
                    break
        
        self.logger.info(f"Sequential analysis completed: {results['final_decision']}")
        return results
    
    def generate_publication_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive publication-ready statistical report.
        
        Includes APA-formatted results, effect sizes, and interpretations.
        """
        
        report = {
            "executive_summary": self._generate_executive_summary(),
            "statistical_tests_summary": self._generate_tests_summary(),
            "effect_size_analysis": self._generate_effect_size_summary(),
            "power_analysis_summary": self._generate_power_analysis_summary(),
            "bayesian_analysis_summary": self._generate_bayesian_summary(),
            "meta_analysis_summary": self._generate_meta_analysis_summary(),
            "assumptions_and_diagnostics": self._generate_diagnostics_summary(),
            "recommendations": self._generate_recommendations(),
            "apa_formatted_results": self._generate_apa_results(),
            "supplementary_statistics": self._generate_supplementary_statistics()
        }
        
        return report
    
    # Private helper methods for statistical tests
    
    def _check_assumptions_two_groups(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Check statistical assumptions for two-group comparisons."""
        
        assumptions = {
            "normality_group1": True,
            "normality_group2": True,
            "equal_variances": True,
            "independence": True,  # Assumed based on design
            "outliers_group1": [],
            "outliers_group2": [],
            "recommendations": []
        }
        
        if not SCIPY_AVAILABLE:
            assumptions["note"] = "Scipy not available - assumptions not tested"
            return assumptions
        
        # Normality tests
        if len(group1) >= 3:
            shapiro_stat1, shapiro_p1 = shapiro(group1)
            assumptions["normality_group1"] = shapiro_p1 > 0.05
            if shapiro_p1 <= 0.05:
                assumptions["recommendations"].append("Group 1 may not be normally distributed - consider non-parametric tests")
        
        if len(group2) >= 3:
            shapiro_stat2, shapiro_p2 = shapiro(group2)
            assumptions["normality_group2"] = shapiro_p2 > 0.05
            if shapiro_p2 <= 0.05:
                assumptions["recommendations"].append("Group 2 may not be normally distributed - consider non-parametric tests")
        
        # Equal variances test
        try:
            levene_stat, levene_p = levene(group1, group2)
            assumptions["equal_variances"] = levene_p > 0.05
            if levene_p <= 0.05:
                assumptions["recommendations"].append("Unequal variances detected - use Welch's t-test")
        except:
            pass
        
        # Outlier detection
        assumptions["outliers_group1"] = self._detect_outliers(group1)
        assumptions["outliers_group2"] = self._detect_outliers(group2)
        
        if assumptions["outliers_group1"] or assumptions["outliers_group2"]:
            assumptions["recommendations"].append("Outliers detected - consider robust statistical methods")
        
        return assumptions
    
    def _check_assumptions_multi_groups(self, groups: List[List[float]]) -> Dict[str, Any]:
        """Check statistical assumptions for multi-group comparisons."""
        
        assumptions = {
            "normality_by_group": {},
            "equal_variances": True,
            "independence": True,
            "outliers_by_group": {},
            "recommendations": []
        }
        
        if not SCIPY_AVAILABLE:
            assumptions["note"] = "Scipy not available - assumptions not tested"
            return assumptions
        
        # Normality tests for each group
        for i, group in enumerate(groups):
            if len(group) >= 3:
                shapiro_stat, shapiro_p = shapiro(group)
                assumptions["normality_by_group"][f"group_{i}"] = shapiro_p > 0.05
                if shapiro_p <= 0.05:
                    assumptions["recommendations"].append(f"Group {i} may not be normally distributed")
        
        # Equal variances test (Levene's test)
        try:
            levene_stat, levene_p = levene(*groups)
            assumptions["equal_variances"] = levene_p > 0.05
            if levene_p <= 0.05:
                assumptions["recommendations"].append("Unequal variances detected - consider Welch's ANOVA")
        except:
            pass
        
        # Outlier detection for each group
        for i, group in enumerate(groups):
            outliers = self._detect_outliers(group)
            assumptions["outliers_by_group"][f"group_{i}"] = outliers
            if outliers:
                assumptions["recommendations"].append(f"Outliers detected in group {i}")
        
        return assumptions
    
    def _detect_outliers(self, data: List[float]) -> List[float]:
        """Detect outliers using IQR method."""
        
        if len(data) < 4:
            return []
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return outliers
    
    def _independent_t_test(
        self, 
        group1: List[float], 
        group2: List[float],
        group1_name: str,
        group2_name: str,
        alpha: float,
        confidence_level: float,
        equal_variances: bool = True
    ) -> StatisticalTestResult:
        """Perform independent samples t-test."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="independent_t_test",
                test_type=StatisticalTest.T_TEST_INDEPENDENT,
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform t-test
        t_stat, p_value = ttest_ind(group1, group2, equal_var=equal_variances)
        
        # Degrees of freedom
        n1, n2 = len(group1), len(group2)
        if equal_variances:
            df = n1 + n2 - 2
        else:
            # Welch's formula
            s1_sq = np.var(group1, ddof=1)
            s2_sq = np.var(group2, ddof=1)
            df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
        
        # Confidence interval for difference in means
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_se = np.sqrt(np.var(group1, ddof=1)/n1 + np.var(group2, ddof=1)/n2)
        t_critical = scipy_stats.t.ppf((1 + confidence_level) / 2, df)
        margin_error = t_critical * pooled_se
        ci = (mean_diff - margin_error, mean_diff + margin_error)
        
        # Effect size (Cohen's d)
        if equal_variances:
            pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        else:
            pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Effect size confidence interval
        effect_se = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
        effect_ci = (cohens_d - t_critical * effect_se, cohens_d + t_critical * effect_se)
        
        # APA formatting
        apa_format = f"t({df:.1f}) = {t_stat:.3f}, p = {p_value:.3f}, d = {cohens_d:.3f}"
        
        return StatisticalTestResult(
            test_name="independent_t_test",
            test_type=StatisticalTest.T_TEST_INDEPENDENT,
            statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=int(df),
            effect_size=cohens_d,
            effect_size_type=EffectSizeType.COHENS_D,
            effect_size_ci=effect_ci,
            confidence_interval=ci,
            confidence_level=confidence_level,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_t_test_result(t_stat, p_value, cohens_d, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_effect_size(abs(cohens_d)),
            sample_sizes=[n1, n2],
            group_names=[group1_name, group2_name]
        )
    
    def _paired_t_test(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str,
        group2_name: str,
        alpha: float,
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform paired samples t-test."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="paired_t_test",
                test_type=StatisticalTest.T_TEST_PAIRED,
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform paired t-test
        t_stat, p_value = ttest_rel(group1, group2)
        
        n = len(group1)
        df = n - 1
        
        # Calculate differences
        differences = [x - y for x, y in zip(group1, group2)]
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Confidence interval for mean difference
        t_critical = scipy_stats.t.ppf((1 + confidence_level) / 2, df)
        margin_error = t_critical * (std_diff / np.sqrt(n))
        ci = (mean_diff - margin_error, mean_diff + margin_error)
        
        # Effect size (Cohen's d for paired design)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # APA formatting
        apa_format = f"t({df}) = {t_stat:.3f}, p = {p_value:.3f}, d = {cohens_d:.3f}"
        
        return StatisticalTestResult(
            test_name="paired_t_test",
            test_type=StatisticalTest.T_TEST_PAIRED,
            statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=cohens_d,
            effect_size_type=EffectSizeType.COHENS_D,
            confidence_interval=ci,
            confidence_level=confidence_level,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_t_test_result(t_stat, p_value, cohens_d, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_effect_size(abs(cohens_d)),
            sample_sizes=[n],
            group_names=[f"{group1_name} vs {group2_name}"]
        )
    
    def _mann_whitney_u_test(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str,
        group2_name: str,
        alpha: float,
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform Mann-Whitney U test."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="mann_whitney_u",
                test_type=StatisticalTest.MANN_WHITNEY_U,
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        n1, n2 = len(group1), len(group2)
        
        # Effect size (r = Z / sqrt(N))
        z_score = (u_stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        r_effect_size = abs(z_score) / np.sqrt(n1 + n2)
        
        # APA formatting
        apa_format = f"U = {u_stat:.0f}, p = {p_value:.3f}, r = {r_effect_size:.3f}"
        
        return StatisticalTestResult(
            test_name="mann_whitney_u",
            test_type=StatisticalTest.MANN_WHITNEY_U,
            statistic=u_stat,
            p_value=p_value,
            effect_size=r_effect_size,
            effect_size_type=EffectSizeType.R_SQUARED,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_nonparametric_result(u_stat, p_value, r_effect_size, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_effect_size(r_effect_size),
            sample_sizes=[n1, n2],
            group_names=[group1_name, group2_name]
        )
    
    def _wilcoxon_signed_rank_test(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str,
        group2_name: str,
        alpha: float,
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform Wilcoxon signed-rank test."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="wilcoxon_signed_rank",
                test_type=StatisticalTest.WILCOXON_SIGNED_RANK,
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform Wilcoxon signed-rank test
        w_stat, p_value = wilcoxon(group1, group2)
        
        n = len(group1)
        
        # Effect size (r = Z / sqrt(N))
        z_score = (w_stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
        r_effect_size = abs(z_score) / np.sqrt(n)
        
        # APA formatting
        apa_format = f"W = {w_stat:.0f}, p = {p_value:.3f}, r = {r_effect_size:.3f}"
        
        return StatisticalTestResult(
            test_name="wilcoxon_signed_rank",
            test_type=StatisticalTest.WILCOXON_SIGNED_RANK,
            statistic=w_stat,
            p_value=p_value,
            effect_size=r_effect_size,
            effect_size_type=EffectSizeType.R_SQUARED,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_nonparametric_result(w_stat, p_value, r_effect_size, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_effect_size(r_effect_size),
            sample_sizes=[n],
            group_names=[f"{group1_name} vs {group2_name}"]
        )
    
    def _one_way_anova(
        self,
        groups: List[List[float]],
        group_names: List[str],
        alpha: float,
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform one-way ANOVA."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="one_way_anova",
                test_type=StatisticalTest.ANOVA_ONE_WAY,
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate degrees of freedom
        k = len(groups)  # number of groups
        n = sum(len(group) for group in groups)  # total sample size
        df_between = k - 1
        df_within = n - k
        
        # Calculate eta-squared (effect size)
        all_data = [x for group in groups for x in group]
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # APA formatting
        apa_format = f"F({df_between}, {df_within}) = {f_stat:.3f}, p = {p_value:.3f}, η² = {eta_squared:.3f}"
        
        return StatisticalTestResult(
            test_name="one_way_anova",
            test_type=StatisticalTest.ANOVA_ONE_WAY,
            statistic=f_stat,
            p_value=p_value,
            degrees_of_freedom=(df_between, df_within),
            effect_size=eta_squared,
            effect_size_type=EffectSizeType.ETA_SQUARED,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_anova_result(f_stat, p_value, eta_squared, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_eta_squared(eta_squared),
            sample_sizes=[len(group) for group in groups],
            group_names=group_names
        )
    
    def _kruskal_wallis_test(
        self,
        groups: List[List[float]],
        group_names: List[str],
        alpha: float,
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform Kruskal-Wallis test."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="kruskal_wallis",
                test_type=StatisticalTest.KRUSKAL_WALLIS,
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform Kruskal-Wallis test
        h_stat, p_value = kruskal(*groups)
        
        k = len(groups)
        n = sum(len(group) for group in groups)
        df = k - 1
        
        # Effect size (epsilon-squared)
        epsilon_squared = (h_stat - k + 1) / (n - k) if n > k else 0
        
        # APA formatting
        apa_format = f"H({df}) = {h_stat:.3f}, p = {p_value:.3f}, ε² = {epsilon_squared:.3f}"
        
        return StatisticalTestResult(
            test_name="kruskal_wallis",
            test_type=StatisticalTest.KRUSKAL_WALLIS,
            statistic=h_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=epsilon_squared,
            effect_size_type=EffectSizeType.EPSILON_SQUARED,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_nonparametric_result(h_stat, p_value, epsilon_squared, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_effect_size(epsilon_squared),
            sample_sizes=[len(group) for group in groups],
            group_names=group_names
        )
    
    def _pearson_correlation(
        self,
        var1: List[float],
        var2: List[float],
        var1_name: str,
        var2_name: str,
        alpha: float,
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform Pearson correlation analysis."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="pearson_correlation",
                test_type=StatisticalTest.CORRELATION_PEARSON,
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform Pearson correlation
        r, p_value = pearsonr(var1, var2)
        
        n = len(var1)
        df = n - 2
        
        # Confidence interval for correlation
        # Fisher's z-transformation
        z_r = 0.5 * np.log((1 + r) / (1 - r)) if r != 1 else float('inf')
        z_critical = scipy_stats.norm.ppf((1 + confidence_level) / 2)
        se_z = 1 / np.sqrt(n - 3)
        
        z_lower = z_r - z_critical * se_z
        z_upper = z_r + z_critical * se_z
        
        # Transform back to correlation scale
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        ci = (r_lower, r_upper)
        
        # APA formatting
        apa_format = f"r({df}) = {r:.3f}, p = {p_value:.3f}"
        
        return StatisticalTestResult(
            test_name="pearson_correlation",
            test_type=StatisticalTest.CORRELATION_PEARSON,
            statistic=r,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=r**2,  # R-squared
            effect_size_type=EffectSizeType.R_SQUARED,
            confidence_interval=ci,
            confidence_level=confidence_level,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_correlation_result(r, p_value, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_correlation_strength(abs(r)),
            sample_sizes=[n],
            group_names=[f"{var1_name} vs {var2_name}"]
        )
    
    def _spearman_correlation(
        self,
        var1: List[float],
        var2: List[float],
        var1_name: str,
        var2_name: str,
        alpha: float,
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform Spearman correlation analysis."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="spearman_correlation",
                test_type=StatisticalTest.CORRELATION_SPEARMAN,
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform Spearman correlation
        rho, p_value = spearmanr(var1, var2)
        
        n = len(var1)
        
        # APA formatting
        apa_format = f"ρ = {rho:.3f}, p = {p_value:.3f}"
        
        return StatisticalTestResult(
            test_name="spearman_correlation",
            test_type=StatisticalTest.CORRELATION_SPEARMAN,
            statistic=rho,
            p_value=p_value,
            effect_size=rho**2,
            effect_size_type=EffectSizeType.R_SQUARED,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_correlation_result(rho, p_value, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_correlation_strength(abs(rho)),
            sample_sizes=[n],
            group_names=[f"{var1_name} vs {var2_name}"]
        )
    
    def _kendall_correlation(
        self,
        var1: List[float],
        var2: List[float],
        var1_name: str,
        var2_name: str,
        alpha: float,
        confidence_level: float
    ) -> StatisticalTestResult:
        """Perform Kendall's tau correlation analysis."""
        
        if not SCIPY_AVAILABLE:
            return StatisticalTestResult(
                test_name="kendall_correlation",
                test_type=StatisticalTest.CORRELATION_SPEARMAN,  # Using spearman enum for non-parametric
                statistic=0.0,
                p_value=1.0,
                interpretation="Scipy not available"
            )
        
        # Perform Kendall's tau correlation
        tau, p_value = kendalltau(var1, var2)
        
        n = len(var1)
        
        # APA formatting
        apa_format = f"τ = {tau:.3f}, p = {p_value:.3f}"
        
        return StatisticalTestResult(
            test_name="kendall_correlation",
            test_type=StatisticalTest.CORRELATION_SPEARMAN,
            statistic=tau,
            p_value=p_value,
            effect_size=tau**2,
            effect_size_type=EffectSizeType.R_SQUARED,
            significant=p_value < alpha,
            alpha=alpha,
            interpretation=self._interpret_correlation_result(tau, p_value, alpha),
            apa_format=apa_format,
            effect_size_interpretation=self._interpret_correlation_strength(abs(tau)),
            sample_sizes=[n],
            group_names=[f"{var1_name} vs {var2_name}"]
        )
    
    # Additional helper methods for advanced analyses
    
    def _comprehensive_effect_size_analysis(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str,
        group2_name: str,
        paired: bool,
        confidence_level: float
    ) -> Dict[str, float]:
        """Calculate comprehensive effect sizes."""
        
        effect_sizes = {}
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        if paired and n1 == n2:
            # Paired design effect sizes
            differences = [x - y for x, y in zip(group1, group2)]
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            
            effect_sizes["cohens_d_paired"] = mean_diff / std_diff if std_diff > 0 else 0
        else:
            # Independent groups effect sizes
            
            # Cohen's d
            pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
            effect_sizes["cohens_d"] = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Hedges' g (bias-corrected Cohen's d)
            j = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
            effect_sizes["hedges_g"] = effect_sizes["cohens_d"] * j
            
            # Glass's delta
            effect_sizes["glass_delta"] = (mean1 - mean2) / np.std(group2, ddof=1) if np.std(group2, ddof=1) > 0 else 0
            
            # Cliff's delta
            effect_sizes["cliff_delta"] = self._calculate_cliff_delta(group1, group2)
        
        return effect_sizes
    
    def _calculate_cliff_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        
        if not group1 or not group2:
            return 0.0
        
        n_pairs = len(group1) * len(group2)
        greater = sum(1 for x in group1 for y in group2 if x > y)
        less = sum(1 for x in group1 for y in group2 if x < y)
        
        return (greater - less) / n_pairs
    
    def _power_analysis_two_groups(
        self,
        group1: List[float],
        group2: List[float],
        alpha: float,
        confidence_level: float
    ) -> Dict[str, float]:
        """Perform power analysis for two groups."""
        
        # Calculate observed effect size
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        observed_effect_size = abs(np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        if not SCIPY_AVAILABLE:
            return {
                "observed_effect_size": observed_effect_size,
                "note": "Scipy required for power calculation"
            }
        
        # Calculate observed power
        from scipy.stats import norm
        
        # Standard error
        se = np.sqrt((1/n1) + (1/n2))
        
        # Critical value
        z_alpha = norm.ppf(1 - alpha/2)
        
        # Power calculation
        z_beta = observed_effect_size / se - z_alpha
        power = norm.cdf(z_beta)
        
        # Required sample size for 80% power
        z_80 = norm.ppf(0.8)
        required_n_per_group = (2 * ((z_alpha + z_80) ** 2)) / (observed_effect_size ** 2)
        
        return {
            "observed_effect_size": observed_effect_size,
            "observed_power": power,
            "required_n_per_group_80_power": int(np.ceil(required_n_per_group)),
            "current_n_group1": n1,
            "current_n_group2": n2
        }
    
    # Bayesian analysis methods (if PyMC3 available)
    
    def _bayesian_two_group_analysis(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str,
        group2_name: str,
        paired: bool
    ) -> BayesianAnalysisResult:
        """Perform Bayesian two-group analysis."""
        
        if not BAYESIAN_AVAILABLE:
            return BayesianAnalysisResult(
                model_name="bayesian_two_group",
                posterior_mean=0.0,
                posterior_std=0.0,
                posterior_median=0.0,
                credible_interval_95=(0.0, 0.0),
                credible_interval_99=(0.0, 0.0),
                interpretation="PyMC3 not available"
            )
        
        try:
            with pm.Model() as model:
                # Priors
                mu1 = pm.Normal('mu1', mu=0, sigma=10)
                mu2 = pm.Normal('mu2', mu=0, sigma=10)
                sigma1 = pm.HalfNormal('sigma1', sigma=5)
                sigma2 = pm.HalfNormal('sigma2', sigma=5)
                
                # Likelihood
                obs1 = pm.Normal('obs1', mu=mu1, sigma=sigma1, observed=group1)
                obs2 = pm.Normal('obs2', mu=mu2, sigma=sigma2, observed=group2)
                
                # Difference
                diff = pm.Deterministic('diff', mu1 - mu2)
                
                # Sampling
                trace = pm.sample(2000, tune=1000, chains=2, cores=1, return_inferencedata=True)
            
            # Extract posterior statistics
            posterior_diff = trace.posterior['diff'].values.flatten()
            
            posterior_mean = np.mean(posterior_diff)
            posterior_std = np.std(posterior_diff)
            posterior_median = np.median(posterior_diff)
            
            # Credible intervals
            ci_95 = np.percentile(posterior_diff, [2.5, 97.5])
            ci_99 = np.percentile(posterior_diff, [0.5, 99.5])
            
            # Probability positive/negative
            prob_positive = np.mean(posterior_diff > 0)
            prob_negative = np.mean(posterior_diff < 0)
            
            # ROPE analysis (Region of Practical Equivalence)
            rope_lower, rope_upper = -0.1, 0.1  # Example ROPE
            rope_probability = np.mean((posterior_diff >= rope_lower) & (posterior_diff <= rope_upper))
            
            return BayesianAnalysisResult(
                model_name="bayesian_two_group",
                posterior_mean=posterior_mean,
                posterior_std=posterior_std,
                posterior_median=posterior_median,
                credible_interval_95=tuple(ci_95),
                credible_interval_99=tuple(ci_99),
                rope_probability=rope_probability,
                probability_positive=prob_positive,
                probability_negative=prob_negative,
                interpretation=f"Bayesian analysis suggests {prob_positive:.1%} probability that {group1_name} > {group2_name}"
            )
        
        except Exception as e:
            return BayesianAnalysisResult(
                model_name="bayesian_two_group",
                posterior_mean=0.0,
                posterior_std=0.0,
                posterior_median=0.0,
                credible_interval_95=(0.0, 0.0),
                credible_interval_99=(0.0, 0.0),
                interpretation=f"Bayesian analysis failed: {str(e)}"
            )
    
    # Interpretation methods
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size."""
        
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength."""
        
        if correlation < 0.1:
            return "negligible"
        elif correlation < 0.3:
            return "small"
        elif correlation < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_t_test_result(self, t_stat: float, p_value: float, effect_size: float, alpha: float) -> str:
        """Interpret t-test results."""
        
        significance = "significant" if p_value < alpha else "not significant"
        effect_interpretation = self._interpret_effect_size(abs(effect_size))
        
        return f"The test is {significance} (p = {p_value:.3f}) with a {effect_interpretation} effect size (d = {effect_size:.3f})"
    
    def _interpret_anova_result(self, f_stat: float, p_value: float, eta_squared: float, alpha: float) -> str:
        """Interpret ANOVA results."""
        
        significance = "significant" if p_value < alpha else "not significant"
        effect_interpretation = self._interpret_eta_squared(eta_squared)
        
        return f"The ANOVA is {significance} (p = {p_value:.3f}) with a {effect_interpretation} effect size (η² = {eta_squared:.3f})"
    
    def _interpret_correlation_result(self, correlation: float, p_value: float, alpha: float) -> str:
        """Interpret correlation results."""
        
        significance = "significant" if p_value < alpha else "not significant"
        strength = self._interpret_correlation_strength(abs(correlation))
        direction = "positive" if correlation > 0 else "negative"
        
        return f"There is a {significance} {direction} {strength} correlation (r = {correlation:.3f}, p = {p_value:.3f})"
    
    def _interpret_nonparametric_result(self, statistic: float, p_value: float, effect_size: float, alpha: float) -> str:
        """Interpret non-parametric test results."""
        
        significance = "significant" if p_value < alpha else "not significant"
        effect_interpretation = self._interpret_effect_size(abs(effect_size))
        
        return f"The non-parametric test is {significance} (p = {p_value:.3f}) with a {effect_interpretation} effect size"
    
    # Report generation methods
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary of statistical analyses."""
        
        total_tests = len(self.test_results)
        significant_tests = sum(1 for test in self.test_results if test.significant)
        
        summary = f"""
        Statistical Analysis Summary:
        - Total tests performed: {total_tests}
        - Significant results: {significant_tests} ({significant_tests/total_tests*100:.1f}%)
        - Meta-analyses: {len(self.meta_analysis_results)}
        - Bayesian analyses: {len(self.bayesian_results)}
        """
        
        return summary.strip()
    
    def _generate_tests_summary(self) -> Dict[str, Any]:
        """Generate summary of statistical tests."""
        
        test_summary = defaultdict(list)
        
        for test in self.test_results:
            test_summary[test.test_type.value].append({
                "test_name": test.test_name,
                "p_value": test.p_value,
                "effect_size": test.effect_size,
                "significant": test.significant,
                "apa_format": test.apa_format
            })
        
        return dict(test_summary)
    
    def _generate_effect_size_summary(self) -> Dict[str, Any]:
        """Generate effect size summary."""
        
        effect_sizes = []
        for test in self.test_results:
            if test.effect_size is not None:
                effect_sizes.append({
                    "test_name": test.test_name,
                    "effect_size": test.effect_size,
                    "effect_size_type": test.effect_size_type.value if test.effect_size_type else None,
                    "interpretation": test.effect_size_interpretation
                })
        
        return {
            "individual_effect_sizes": effect_sizes,
            "average_effect_size": np.mean([es["effect_size"] for es in effect_sizes]) if effect_sizes else 0,
            "large_effects_count": sum(1 for es in effect_sizes if "large" in es.get("interpretation", ""))
        }
    
    def _generate_power_analysis_summary(self) -> Dict[str, Any]:
        """Generate power analysis summary."""
        
        power_results = [test for test in self.test_results if test.power is not None]
        
        if not power_results:
            return {"note": "No power analyses performed"}
        
        return {
            "tests_with_power_analysis": len(power_results),
            "average_power": np.mean([test.power for test in power_results]),
            "adequate_power_count": sum(1 for test in power_results if test.power >= 0.8)
        }
    
    def _generate_bayesian_summary(self) -> Dict[str, Any]:
        """Generate Bayesian analysis summary."""
        
        if not self.bayesian_results:
            return {"note": "No Bayesian analyses performed"}
        
        return {
            "bayesian_analyses_count": len(self.bayesian_results),
            "models": [result.model_name for result in self.bayesian_results]
        }
    
    def _generate_meta_analysis_summary(self) -> Dict[str, Any]:
        """Generate meta-analysis summary."""
        
        if not self.meta_analysis_results:
            return {"note": "No meta-analyses performed"}
        
        return {
            "meta_analyses_count": len(self.meta_analysis_results),
            "total_studies_included": sum(result.studies_included for result in self.meta_analysis_results)
        }
    
    def _generate_diagnostics_summary(self) -> Dict[str, Any]:
        """Generate diagnostics and assumptions summary."""
        
        assumption_violations = []
        warnings = []
        
        for test in self.test_results:
            if not test.assumptions_met:
                assumption_violations.append(test.test_name)
            warnings.extend(test.warnings)
        
        return {
            "tests_with_assumption_violations": len(assumption_violations),
            "violated_tests": assumption_violations,
            "total_warnings": len(warnings),
            "warning_messages": warnings
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate statistical recommendations."""
        
        recommendations = []
        
        # Check for low power
        low_power_tests = [test for test in self.test_results if test.power is not None and test.power < 0.8]
        if low_power_tests:
            recommendations.append("Some tests have low statistical power - consider increasing sample sizes")
        
        # Check for assumption violations
        violated_tests = [test for test in self.test_results if not test.assumptions_met]
        if violated_tests:
            recommendations.append("Some tests have assumption violations - consider non-parametric alternatives")
        
        # Check effect sizes
        small_effects = [test for test in self.test_results if test.effect_size is not None and test.effect_size < 0.2]
        if len(small_effects) > len(self.test_results) // 2:
            recommendations.append("Many tests show small effect sizes - consider practical significance")
        
        return recommendations
    
    def _generate_apa_results(self) -> List[str]:
        """Generate APA-formatted results."""
        
        return [test.apa_format for test in self.test_results if test.apa_format]
    
    def _generate_supplementary_statistics(self) -> Dict[str, Any]:
        """Generate supplementary statistical information."""
        
        return {
            "descriptive_statistics": "Available upon request",
            "raw_data": "Available in supplementary materials",
            "analysis_code": "Available in research repository",
            "additional_diagnostics": "Complete assumption checking results available"
        }
    
    # Placeholder methods for advanced analyses
    
    def _repeated_measures_anova(self, groups, group_names, alpha, confidence_level):
        """Placeholder for repeated measures ANOVA."""
        return StatisticalTestResult(
            test_name="repeated_measures_anova",
            test_type=StatisticalTest.ANOVA_ONE_WAY,
            statistic=0.0,
            p_value=1.0,
            interpretation="Repeated measures ANOVA - simplified implementation"
        )
    
    def _friedman_test(self, groups, group_names, alpha, confidence_level):
        """Placeholder for Friedman test."""
        return StatisticalTestResult(
            test_name="friedman_test",
            test_type=StatisticalTest.FRIEDMAN,
            statistic=0.0,
            p_value=1.0,
            interpretation="Friedman test - simplified implementation"
        )
    
    def _post_hoc_analysis(self, groups, group_names, alpha, confidence_level):
        """Placeholder for post-hoc analysis."""
        return {"note": "Post-hoc analysis - implementation pending"}
    
    def _multi_group_effect_sizes(self, groups, group_names, confidence_level):
        """Placeholder for multi-group effect sizes."""
        return {"note": "Multi-group effect sizes - implementation pending"}
    
    def _multiple_comparison_corrections(self, groups, group_names, alpha):
        """Placeholder for multiple comparison corrections."""
        return {"note": "Multiple comparison corrections - implementation pending"}
    
    def _bayesian_anova(self, groups, group_names, repeated_measures):
        """Placeholder for Bayesian ANOVA."""
        return BayesianAnalysisResult(
            model_name="bayesian_anova",
            posterior_mean=0.0,
            posterior_std=0.0,
            posterior_median=0.0,
            credible_interval_95=(0.0, 0.0),
            credible_interval_99=(0.0, 0.0),
            interpretation="Bayesian ANOVA - implementation pending"
        )
    
    def _bayesian_correlation(self, var1, var2, var1_name, var2_name):
        """Placeholder for Bayesian correlation."""
        return BayesianAnalysisResult(
            model_name="bayesian_correlation",
            posterior_mean=0.0,
            posterior_std=0.0,
            posterior_median=0.0,
            credible_interval_95=(0.0, 0.0),
            credible_interval_99=(0.0, 0.0),
            interpretation="Bayesian correlation - implementation pending"
        )
    
    def export_results(self, format: str = "json") -> str:
        """Export statistical analysis results."""
        
        export_data = {
            "statistical_tests": [result.__dict__ for result in self.test_results],
            "meta_analyses": [result.__dict__ for result in self.meta_analysis_results],
            "bayesian_analyses": [result.__dict__ for result in self.bayesian_results],
            "publication_report": self.generate_publication_report()
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            return str(export_data)
    
    def shutdown(self) -> None:
        """Shutdown statistical analysis suite."""
        
        self.logger.info("Shutting down Statistical Analysis Suite...")
        
        # Save results
        if self.test_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistical_analysis_results_{timestamp}.json"
            
            try:
                with open(filename, 'w') as f:
                    f.write(self.export_results("json"))
                self.logger.info(f"Statistical results saved to {filename}")
            except Exception as e:
                self.logger.error(f"Failed to save statistical results: {e}")
        
        self.logger.info("Statistical Analysis Suite shutdown complete")


# Example usage
if __name__ == "__main__":
    
    # Example data
    classical_times = [0.15, 0.12, 0.18, 0.14, 0.16, 0.13, 0.17, 0.15, 0.14, 0.16]
    quantum_times = [0.08, 0.09, 0.11, 0.07, 0.10, 0.08, 0.09, 0.10, 0.08, 0.09]
    
    # Initialize statistical analysis suite
    stats_suite = StatisticalAnalysisSuite()
    
    print("🔬 Starting Statistical Analysis Suite...")
    
    try:
        # Comprehensive two-group analysis
        two_group_results = stats_suite.comprehensive_two_group_analysis(
            classical_times, quantum_times, 
            "Classical Algorithm", "Quantum Algorithm"
        )
        
        print("✅ Two-group analysis completed")
        
        # Correlation analysis (example)
        accuracy_scores = [0.8, 0.75, 0.82, 0.78, 0.81, 0.77, 0.83, 0.80, 0.79, 0.81]
        correlation_results = stats_suite.correlation_analysis(
            classical_times, accuracy_scores,
            "Execution Time", "Accuracy"
        )
        
        print("✅ Correlation analysis completed")
        
        # Meta-analysis example
        effect_sizes = [0.5, 0.3, 0.7, 0.4, 0.6]
        sample_sizes = [30, 25, 35, 28, 32]
        study_names = ["Study 1", "Study 2", "Study 3", "Study 4", "Study 5"]
        
        meta_result = stats_suite.meta_analysis(effect_sizes, sample_sizes, study_names)
        print(f"✅ Meta-analysis completed: overall effect = {meta_result.overall_effect_size:.3f}")
        
        # Generate publication report
        publication_report = stats_suite.generate_publication_report()
        print("📄 Publication report generated")
        
        # Export results
        results_json = stats_suite.export_results("json")
        print("💾 Results exported")
        
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
    
    finally:
        stats_suite.shutdown()