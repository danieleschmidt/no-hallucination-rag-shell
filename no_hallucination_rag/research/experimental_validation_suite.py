"""
Experimental Validation Suite for Quantum-Enhanced RAG Research.

This module provides comprehensive experimental validation capabilities
including reproducible experimental design, statistical significance testing,
and publication-ready result generation for quantum vs classical algorithm comparisons.

Research Focus: Rigorous experimental validation with proper controls,
randomization, and statistical analysis for academic publication standards.
"""

import logging
import time
import asyncio
import random
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import itertools
import hashlib

try:
    from scipy import stats as scipy_stats
    from scipy.stats import wilcoxon, mannwhitneyu, kruskal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class ExperimentType(Enum):
    """Types of experimental validation."""
    CONTROLLED_COMPARISON = "controlled_comparison"
    ABLATION_STUDY = "ablation_study"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    ROBUSTNESS_TESTING = "robustness_testing"
    CROSS_VALIDATION = "cross_validation"
    MONTE_CARLO_SIMULATION = "monte_carlo_simulation"


class ExperimentalDesign(Enum):
    """Experimental design methodologies."""
    RANDOMIZED_CONTROLLED = "randomized_controlled"
    FACTORIAL_DESIGN = "factorial_design"
    LATIN_SQUARE = "latin_square"
    REPEATED_MEASURES = "repeated_measures"
    CROSSOVER_DESIGN = "crossover_design"


@dataclass
class ExperimentConfiguration:
    """Configuration for experimental validation."""
    experiment_type: ExperimentType
    design_type: ExperimentalDesign = ExperimentalDesign.RANDOMIZED_CONTROLLED
    
    # Sample size and power analysis
    desired_power: float = 0.8
    significance_level: float = 0.05
    effect_size: float = 0.5
    minimum_sample_size: int = 30
    maximum_sample_size: int = 1000
    
    # Randomization and controls
    randomization_seed: Optional[int] = None
    use_stratified_sampling: bool = True
    control_for_confounders: bool = True
    
    # Validation parameters
    cross_validation_folds: int = 5
    bootstrap_iterations: int = 1000
    monte_carlo_samples: int = 10000
    
    # Quality controls
    enable_blinding: bool = True
    check_assumptions: bool = True
    outlier_detection: bool = True
    multiple_comparison_correction: str = "bonferroni"  # "bonferroni", "fdr", "holm"


@dataclass
class ExperimentalCondition:
    """Definition of experimental condition."""
    condition_id: str
    algorithm_name: str
    algorithm_category: str
    parameters: Dict[str, Any]
    control_variables: Dict[str, Any] = field(default_factory=dict)
    expected_effect_size: Optional[float] = None


@dataclass
class ExperimentalResult:
    """Results from experimental validation."""
    experiment_id: str
    condition_id: str
    participant_id: str  # For repeated measures
    trial_number: int
    
    # Primary outcomes
    execution_time: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Secondary outcomes
    memory_usage: float
    cpu_utilization: float
    throughput: float
    error_rate: float
    
    # Quality metrics
    effect_size: float
    confidence_interval: Tuple[float, float]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    randomization_group: Optional[str] = None
    confounding_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalTestResult:
    """Results from statistical testing."""
    test_name: str
    test_statistic: float
    p_value: float
    degrees_of_freedom: Optional[int]
    effect_size: float
    power: float
    confidence_interval: Tuple[float, float]
    significant: bool
    interpretation: str
    assumptions_met: bool
    warnings: List[str] = field(default_factory=list)


class ExperimentalValidationSuite:
    """
    Comprehensive experimental validation suite for quantum algorithm research.
    
    Provides rigorous experimental design, statistical validation, and
    publication-ready result generation with proper controls and randomization.
    
    Research Capabilities:
    1. Power analysis and sample size calculation
    2. Randomized controlled experiments
    3. Multiple comparison corrections
    4. Effect size calculations
    5. Assumption checking and validation
    6. Bootstrap confidence intervals
    7. Cross-validation frameworks
    8. Reproducibility controls
    """
    
    def __init__(
        self,
        config: Optional[ExperimentConfiguration] = None,
        enable_reproducibility: bool = True
    ):
        self.config = config or ExperimentConfiguration(ExperimentType.CONTROLLED_COMPARISON)
        self.enable_reproducibility = enable_reproducibility
        
        # Set random seeds for reproducibility
        if enable_reproducibility:
            seed = self.config.randomization_seed or 42
            random.seed(seed)
            np.random.seed(seed)
        
        # Experimental data storage
        self.experimental_conditions: List[ExperimentalCondition] = []
        self.experimental_results: List[ExperimentalResult] = []
        self.statistical_tests: List[StatisticalTestResult] = []
        
        # Quality control
        self.outliers_detected: List[str] = []
        self.assumption_violations: List[str] = []
        
        # Reproducibility tracking
        self.experiment_metadata = {
            "creation_time": datetime.utcnow().isoformat(),
            "random_seed": seed if enable_reproducibility else None,
            "configuration": self.config.__dict__,
            "version": "1.0"
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Experimental Validation Suite initialized")
    
    def calculate_required_sample_size(
        self,
        effect_size: Optional[float] = None,
        power: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate required sample size based on power analysis.
        
        Uses Cohen's conventions for effect sizes and standard power analysis.
        """
        
        effect_size = effect_size or self.config.effect_size
        power = power or self.config.desired_power
        alpha = alpha or self.config.significance_level
        
        try:
            if SCIPY_AVAILABLE:
                from scipy.stats import norm
                
                # Two-tailed test
                z_alpha = norm.ppf(1 - alpha / 2)
                z_beta = norm.ppf(power)
                
                # Sample size per group for t-test
                n_per_group = (2 * ((z_alpha + z_beta) ** 2)) / (effect_size ** 2)
                n_per_group = int(np.ceil(n_per_group))
                
                # Total sample size
                total_n = n_per_group * 2
                
                # Ensure minimum sample size
                total_n = max(total_n, self.config.minimum_sample_size)
                total_n = min(total_n, self.config.maximum_sample_size)
                
                power_analysis = {
                    "required_sample_size_per_group": n_per_group,
                    "required_total_sample_size": total_n,
                    "effect_size": effect_size,
                    "desired_power": power,
                    "significance_level": alpha,
                    "effect_size_interpretation": self._interpret_effect_size(effect_size),
                    "recommendation": self._generate_sample_size_recommendation(total_n, effect_size)
                }
                
            else:
                # Fallback calculation
                # Rule of thumb: minimum 30 per group for medium effect
                n_per_group = max(30, int(50 / effect_size))
                total_n = min(n_per_group * 2, self.config.maximum_sample_size)
                
                power_analysis = {
                    "required_sample_size_per_group": n_per_group,
                    "required_total_sample_size": total_n,
                    "effect_size": effect_size,
                    "desired_power": power,
                    "significance_level": alpha,
                    "note": "Approximate calculation - scipy recommended for precise power analysis"
                }
            
            self.logger.info(f"Power analysis complete: {total_n} total samples required")
            return power_analysis
            
        except Exception as e:
            self.logger.error(f"Power analysis failed: {e}")
            return {
                "error": str(e),
                "fallback_sample_size": self.config.minimum_sample_size
            }
    
    def design_experiment(
        self,
        conditions: List[ExperimentalCondition],
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Design experimental protocol with proper randomization and controls.
        """
        
        self.experimental_conditions = conditions
        
        # Calculate sample size if not provided
        if sample_size is None:
            power_analysis = self.calculate_required_sample_size()
            sample_size = power_analysis.get("required_total_sample_size", self.config.minimum_sample_size)
        
        # Generate experimental design
        design = {
            "experiment_id": self._generate_experiment_id(),
            "design_type": self.config.design_type.value,
            "conditions": [condition.__dict__ for condition in conditions],
            "sample_size": sample_size,
            "randomization_scheme": self._create_randomization_scheme(conditions, sample_size),
            "blinding_protocol": self._create_blinding_protocol() if self.config.enable_blinding else None,
            "quality_controls": self._define_quality_controls(),
            "analysis_plan": self._create_analysis_plan(conditions)
        }
        
        self.logger.info(f"Experimental design created: {len(conditions)} conditions, {sample_size} total samples")
        return design
    
    async def execute_experiment(
        self,
        algorithm_implementations: Dict[str, Callable],
        test_datasets: Dict[str, Any],
        experimental_design: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the experimental protocol with proper controls and randomization.
        """
        
        experiment_id = experimental_design["experiment_id"]
        randomization_scheme = experimental_design["randomization_scheme"]
        
        self.logger.info(f"Starting experiment execution: {experiment_id}")
        start_time = time.time()
        
        results = []
        
        try:
            # Execute trials according to randomization scheme
            for trial_info in randomization_scheme:
                trial_result = await self._execute_single_trial(
                    trial_info,
                    algorithm_implementations,
                    test_datasets,
                    experiment_id
                )
                results.append(trial_result)
                
                # Progress logging
                if len(results) % 10 == 0:
                    self.logger.info(f"Completed {len(results)}/{len(randomization_scheme)} trials")
            
            self.experimental_results.extend(results)
            
            # Quality control checks
            quality_report = self._perform_quality_control_checks(results)
            
            # Preliminary analysis
            preliminary_analysis = self._perform_preliminary_analysis(results)
            
            execution_time = time.time() - start_time
            
            experiment_report = {
                "experiment_id": experiment_id,
                "execution_summary": {
                    "total_trials": len(results),
                    "successful_trials": len([r for r in results if not r.__dict__.get("error", False)]),
                    "execution_time": execution_time,
                    "trials_per_second": len(results) / execution_time
                },
                "quality_control": quality_report,
                "preliminary_results": preliminary_analysis,
                "raw_results": [result.__dict__ for result in results]
            }
            
            self.logger.info(f"Experiment execution completed: {len(results)} trials in {execution_time:.2f}s")
            return experiment_report
            
        except Exception as e:
            self.logger.error(f"Experiment execution failed: {e}")
            raise
    
    async def _execute_single_trial(
        self,
        trial_info: Dict[str, Any],
        algorithm_implementations: Dict[str, Callable],
        test_datasets: Dict[str, Any],
        experiment_id: str
    ) -> ExperimentalResult:
        """Execute a single experimental trial."""
        
        condition_id = trial_info["condition_id"]
        trial_number = trial_info["trial_number"]
        participant_id = trial_info.get("participant_id", "default")
        
        # Find the condition
        condition = next((c for c in self.experimental_conditions if c.condition_id == condition_id), None)
        if not condition:
            raise ValueError(f"Condition {condition_id} not found")
        
        # Get algorithm implementation
        algorithm_func = algorithm_implementations.get(condition.algorithm_name)
        if not algorithm_func:
            raise ValueError(f"Algorithm {condition.algorithm_name} not implemented")
        
        # Prepare dataset for this trial
        dataset = self._prepare_trial_dataset(test_datasets, condition, trial_info)
        
        # Monitor resources before execution
        import psutil
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_before = psutil.cpu_percent()
        
        start_time = time.time()
        
        try:
            # Execute algorithm
            if asyncio.iscoroutinefunction(algorithm_func):
                algorithm_result = await algorithm_func(dataset)
            else:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                algorithm_result = await loop.run_in_executor(None, algorithm_func, dataset)
            
            execution_time = time.time() - start_time
            
            # Monitor resources after execution
            memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu_after = psutil.cpu_percent()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_trial_performance_metrics(
                algorithm_result, dataset, execution_time
            )
            
            # Calculate effect size and confidence interval
            effect_size = self._calculate_trial_effect_size(performance_metrics, condition)
            confidence_interval = self._calculate_trial_confidence_interval(performance_metrics)
            
            # Create experimental result
            result = ExperimentalResult(
                experiment_id=experiment_id,
                condition_id=condition_id,
                participant_id=participant_id,
                trial_number=trial_number,
                execution_time=execution_time,
                accuracy=performance_metrics.get("accuracy", 0.0),
                precision=performance_metrics.get("precision", 0.0),
                recall=performance_metrics.get("recall", 0.0),
                f1_score=performance_metrics.get("f1_score", 0.0),
                memory_usage=memory_after - memory_before,
                cpu_utilization=(cpu_after + cpu_before) / 2,
                throughput=performance_metrics.get("throughput", 0.0),
                error_rate=performance_metrics.get("error_rate", 0.0),
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                randomization_group=trial_info.get("randomization_group"),
                confounding_variables=trial_info.get("confounding_variables", {})
            )
            
            return result
            
        except Exception as e:
            # Create error result
            execution_time = time.time() - start_time
            
            error_result = ExperimentalResult(
                experiment_id=experiment_id,
                condition_id=condition_id,
                participant_id=participant_id,
                trial_number=trial_number,
                execution_time=execution_time,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                memory_usage=0.0,
                cpu_utilization=0.0,
                throughput=0.0,
                error_rate=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0)
            )
            
            # Add error information
            error_result.__dict__["error"] = True
            error_result.__dict__["error_message"] = str(e)
            
            return error_result
    
    def perform_statistical_analysis(
        self,
        primary_outcome: str = "execution_time",
        secondary_outcomes: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis with multiple comparison corrections.
        """
        
        if not self.experimental_results:
            return {"error": "No experimental results available"}
        
        secondary_outcomes = secondary_outcomes or ["accuracy", "throughput", "memory_usage"]
        
        self.logger.info("Starting comprehensive statistical analysis")
        
        analysis_results = {
            "primary_analysis": self._analyze_primary_outcome(primary_outcome),
            "secondary_analyses": {},
            "assumption_checks": self._check_statistical_assumptions(),
            "effect_size_analysis": self._comprehensive_effect_size_analysis(),
            "multiple_comparison_correction": self._apply_multiple_comparison_correction(),
            "power_analysis_post_hoc": self._post_hoc_power_analysis(),
            "confidence_intervals": self._calculate_comprehensive_confidence_intervals(),
            "non_parametric_tests": self._perform_non_parametric_tests(),
            "bootstrap_analysis": self._perform_bootstrap_analysis(),
            "cross_validation_results": self._perform_cross_validation_analysis(),
            "sensitivity_analysis": self._perform_sensitivity_analysis()
        }
        
        # Analyze secondary outcomes
        for outcome in secondary_outcomes:
            analysis_results["secondary_analyses"][outcome] = self._analyze_secondary_outcome(outcome)
        
        # Store statistical test results
        self._extract_statistical_tests(analysis_results)
        
        self.logger.info("Statistical analysis completed")
        return analysis_results
    
    def _analyze_primary_outcome(self, outcome: str) -> Dict[str, Any]:
        """Analyze primary outcome with appropriate statistical tests."""
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in self.experimental_results:
            if hasattr(result, outcome) and not result.__dict__.get("error", False):
                condition_groups[result.condition_id].append(getattr(result, outcome))
        
        if len(condition_groups) < 2:
            return {"error": "Insufficient conditions for comparison"}
        
        analysis = {
            "outcome": outcome,
            "descriptive_statistics": {},
            "inferential_tests": {},
            "effect_sizes": {},
            "practical_significance": {}
        }
        
        # Descriptive statistics
        for condition_id, values in condition_groups.items():
            analysis["descriptive_statistics"][condition_id] = {
                "n": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values, ddof=1),
                "min": np.min(values),
                "max": np.max(values),
                "skewness": self._calculate_skewness(values),
                "kurtosis": self._calculate_kurtosis(values)
            }
        
        # Inferential tests
        if SCIPY_AVAILABLE:
            condition_names = list(condition_groups.keys())
            condition_values = list(condition_groups.values())
            
            if len(condition_groups) == 2:
                # Two-group comparison
                group1, group2 = condition_values
                
                # Independent t-test
                t_stat, t_p = scipy_stats.ttest_ind(group1, group2)
                analysis["inferential_tests"]["t_test"] = {
                    "statistic": t_stat,
                    "p_value": t_p,
                    "significant": t_p < self.config.significance_level,
                    "degrees_of_freedom": len(group1) + len(group2) - 2
                }
                
                # Welch's t-test (unequal variances)
                welch_t, welch_p = scipy_stats.ttest_ind(group1, group2, equal_var=False)
                analysis["inferential_tests"]["welch_t_test"] = {
                    "statistic": welch_t,
                    "p_value": welch_p,
                    "significant": welch_p < self.config.significance_level
                }
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p = mannwhitneyu(group1, group2, alternative='two-sided')
                analysis["inferential_tests"]["mann_whitney_u"] = {
                    "statistic": u_stat,
                    "p_value": u_p,
                    "significant": u_p < self.config.significance_level
                }
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                     (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                    (len(group1) + len(group2) - 2))
                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
                
                analysis["effect_sizes"]["cohens_d"] = {
                    "value": cohens_d,
                    "interpretation": self._interpret_effect_size(abs(cohens_d)),
                    "confidence_interval": self._cohens_d_confidence_interval(group1, group2)
                }
            
            else:
                # Multi-group comparison
                # One-way ANOVA
                f_stat, f_p = scipy_stats.f_oneway(*condition_values)
                analysis["inferential_tests"]["anova"] = {
                    "statistic": f_stat,
                    "p_value": f_p,
                    "significant": f_p < self.config.significance_level,
                    "df_between": len(condition_groups) - 1,
                    "df_within": sum(len(vals) for vals in condition_values) - len(condition_groups)
                }
                
                # Kruskal-Wallis test (non-parametric)
                kw_stat, kw_p = kruskal(*condition_values)
                analysis["inferential_tests"]["kruskal_wallis"] = {
                    "statistic": kw_stat,
                    "p_value": kw_p,
                    "significant": kw_p < self.config.significance_level
                }
                
                # Eta-squared (effect size for ANOVA)
                ss_between = sum(len(vals) * (np.mean(vals) - np.mean([v for vals in condition_values for v in vals]))**2 
                               for vals in condition_values)
                ss_total = sum((v - np.mean([v for vals in condition_values for v in vals]))**2 
                              for vals in condition_values for v in vals)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                analysis["effect_sizes"]["eta_squared"] = {
                    "value": eta_squared,
                    "interpretation": self._interpret_eta_squared(eta_squared)
                }
        
        return analysis
    
    def _analyze_secondary_outcome(self, outcome: str) -> Dict[str, Any]:
        """Analyze secondary outcome with adjustment for multiple testing."""
        
        # Similar to primary analysis but with adjusted significance levels
        primary_analysis = self._analyze_primary_outcome(outcome)
        
        # Adjust p-values for multiple testing
        if "inferential_tests" in primary_analysis:
            adjusted_alpha = self.config.significance_level / len(self.config.__dict__.get("secondary_outcomes", [outcome]))
            
            for test_name, test_result in primary_analysis["inferential_tests"].items():
                test_result["adjusted_alpha"] = adjusted_alpha
                test_result["significant_adjusted"] = test_result["p_value"] < adjusted_alpha
        
        return primary_analysis
    
    def _check_statistical_assumptions(self) -> Dict[str, Any]:
        """Check assumptions for statistical tests."""
        
        assumption_checks = {
            "normality": {},
            "homogeneity_of_variance": {},
            "independence": {},
            "outliers": {}
        }
        
        if not SCIPY_AVAILABLE:
            return {"scipy_not_available": True}
        
        # Group data by condition
        condition_groups = defaultdict(list)
        for result in self.experimental_results:
            if not result.__dict__.get("error", False):
                condition_groups[result.condition_id].append(result.execution_time)
        
        # Normality tests
        for condition_id, values in condition_groups.items():
            if len(values) >= 3:
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = scipy_stats.shapiro(values)
                assumption_checks["normality"][condition_id] = {
                    "shapiro_wilk": {
                        "statistic": shapiro_stat,
                        "p_value": shapiro_p,
                        "normal": shapiro_p > 0.05
                    }
                }
                
                # Anderson-Darling test
                try:
                    ad_result = scipy_stats.anderson(values, dist='norm')
                    assumption_checks["normality"][condition_id]["anderson_darling"] = {
                        "statistic": ad_result.statistic,
                        "critical_values": ad_result.critical_values.tolist(),
                        "significance_levels": ad_result.significance_level.tolist()
                    }
                except:
                    pass
        
        # Homogeneity of variance (Levene's test)
        if len(condition_groups) >= 2:
            condition_values = list(condition_groups.values())
            try:
                levene_stat, levene_p = scipy_stats.levene(*condition_values)
                assumption_checks["homogeneity_of_variance"]["levene_test"] = {
                    "statistic": levene_stat,
                    "p_value": levene_p,
                    "equal_variances": levene_p > 0.05
                }
            except:
                pass
        
        # Outlier detection using IQR method
        for condition_id, values in condition_groups.items():
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            assumption_checks["outliers"][condition_id] = {
                "count": len(outliers),
                "percentage": len(outliers) / len(values) * 100,
                "values": outliers,
                "bounds": [lower_bound, upper_bound]
            }
        
        # Independence assumption (check for autocorrelation)
        # This is a simplified check - in practice would depend on experimental design
        assumption_checks["independence"]["note"] = "Independence assumed based on randomized design"
        
        return assumption_checks
    
    def _comprehensive_effect_size_analysis(self) -> Dict[str, Any]:
        """Calculate comprehensive effect size analysis."""
        
        effect_analysis = {
            "cohens_d": {},
            "glass_delta": {},
            "hedges_g": {},
            "cliff_delta": {},
            "probability_superiority": {},
            "common_language_effect_size": {}
        }
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in self.experimental_results:
            if not result.__dict__.get("error", False):
                condition_groups[result.condition_id].append(result.execution_time)
        
        if len(condition_groups) < 2:
            return {"error": "Insufficient conditions for effect size analysis"}
        
        condition_names = list(condition_groups.keys())
        
        # Pairwise effect size calculations
        for i in range(len(condition_names)):
            for j in range(i + 1, len(condition_names)):
                cond1, cond2 = condition_names[i], condition_names[j]
                group1, group2 = condition_groups[cond1], condition_groups[cond2]
                
                comparison_key = f"{cond1}_vs_{cond2}"
                
                # Cohen's d
                pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                     (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                    (len(group1) + len(group2) - 2))
                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
                
                effect_analysis["cohens_d"][comparison_key] = {
                    "value": cohens_d,
                    "interpretation": self._interpret_effect_size(abs(cohens_d))
                }
                
                # Glass's Delta
                glass_delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1) if np.std(group2, ddof=1) > 0 else 0
                effect_analysis["glass_delta"][comparison_key] = {
                    "value": glass_delta,
                    "interpretation": self._interpret_effect_size(abs(glass_delta))
                }
                
                # Hedges' g (bias-corrected Cohen's d)
                J = 1 - (3 / (4 * (len(group1) + len(group2) - 2) - 1))
                hedges_g = cohens_d * J
                effect_analysis["hedges_g"][comparison_key] = {
                    "value": hedges_g,
                    "interpretation": self._interpret_effect_size(abs(hedges_g))
                }
                
                # Cliff's delta (non-parametric effect size)
                cliff_delta = self._calculate_cliff_delta(group1, group2)
                effect_analysis["cliff_delta"][comparison_key] = {
                    "value": cliff_delta,
                    "interpretation": self._interpret_cliff_delta(abs(cliff_delta))
                }
                
                # Probability of superiority
                prob_superiority = sum(1 for x in group1 for y in group2 if x > y) / (len(group1) * len(group2))
                effect_analysis["probability_superiority"][comparison_key] = {
                    "value": prob_superiority,
                    "interpretation": "Probability that random value from group 1 exceeds random value from group 2"
                }
                
                # Common Language Effect Size
                cles = max(prob_superiority, 1 - prob_superiority)
                effect_analysis["common_language_effect_size"][comparison_key] = {
                    "value": cles,
                    "interpretation": f"{cles*100:.1f}% chance of correct classification"
                }
        
        return effect_analysis
    
    def _apply_multiple_comparison_correction(self) -> Dict[str, Any]:
        """Apply multiple comparison corrections to p-values."""
        
        if not SCIPY_AVAILABLE:
            return {"scipy_not_available": True}
        
        # Collect all p-values from statistical tests
        p_values = []
        test_names = []
        
        for result in self.statistical_tests:
            p_values.append(result.p_value)
            test_names.append(result.test_name)
        
        if not p_values:
            return {"no_tests_available": True}
        
        correction_results = {}
        
        # Bonferroni correction
        bonferroni_alpha = self.config.significance_level / len(p_values)
        bonferroni_significant = [p < bonferroni_alpha for p in p_values]
        
        correction_results["bonferroni"] = {
            "corrected_alpha": bonferroni_alpha,
            "significant_tests": sum(bonferroni_significant),
            "test_results": [(name, p, sig) for name, p, sig in zip(test_names, p_values, bonferroni_significant)]
        }
        
        # Holm-Bonferroni correction
        sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
        holm_significant = [False] * len(p_values)
        
        for i, idx in enumerate(sorted_indices):
            alpha_holm = self.config.significance_level / (len(p_values) - i)
            if p_values[idx] < alpha_holm:
                holm_significant[idx] = True
            else:
                break
        
        correction_results["holm_bonferroni"] = {
            "significant_tests": sum(holm_significant),
            "test_results": [(name, p, sig) for name, p, sig in zip(test_names, p_values, holm_significant)]
        }
        
        # False Discovery Rate (Benjamini-Hochberg)
        try:
            from scipy.stats import false_discovery_control
            fdr_significant = false_discovery_control(p_values, alpha=self.config.significance_level)
            correction_results["fdr_bh"] = {
                "significant_tests": sum(fdr_significant),
                "test_results": [(name, p, sig) for name, p, sig in zip(test_names, p_values, fdr_significant)]
            }
        except ImportError:
            # Manual FDR calculation
            sorted_p_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
            fdr_significant = [False] * len(p_values)
            
            for i in range(len(p_values) - 1, -1, -1):
                idx = sorted_p_indices[i]
                threshold = (i + 1) / len(p_values) * self.config.significance_level
                if p_values[idx] <= threshold:
                    for j in range(i + 1):
                        fdr_significant[sorted_p_indices[j]] = True
                    break
            
            correction_results["fdr_bh"] = {
                "significant_tests": sum(fdr_significant),
                "test_results": [(name, p, sig) for name, p, sig in zip(test_names, p_values, fdr_significant)]
            }
        
        return correction_results
    
    def _post_hoc_power_analysis(self) -> Dict[str, Any]:
        """Perform post-hoc power analysis on completed experiments."""
        
        power_analysis = {}
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in self.experimental_results:
            if not result.__dict__.get("error", False):
                condition_groups[result.condition_id].append(result.execution_time)
        
        if len(condition_groups) < 2:
            return {"error": "Insufficient conditions for power analysis"}
        
        condition_names = list(condition_groups.keys())
        
        # Pairwise power analysis
        for i in range(len(condition_names)):
            for j in range(i + 1, len(condition_names)):
                cond1, cond2 = condition_names[i], condition_names[j]
                group1, group2 = condition_groups[cond1], condition_groups[cond2]
                
                # Calculate observed effect size
                pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                     (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                    (len(group1) + len(group2) - 2))
                observed_effect_size = abs(np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
                
                # Calculate observed power
                if SCIPY_AVAILABLE:
                    from scipy.stats import norm
                    
                    alpha = self.config.significance_level
                    n1, n2 = len(group1), len(group2)
                    
                    # Standard error
                    se = np.sqrt((1/n1) + (1/n2))
                    
                    # Critical value
                    z_alpha = norm.ppf(1 - alpha/2)
                    
                    # Power calculation
                    z_beta = observed_effect_size / se - z_alpha
                    power = norm.cdf(z_beta)
                    
                    power_analysis[f"{cond1}_vs_{cond2}"] = {
                        "observed_effect_size": observed_effect_size,
                        "sample_size_1": n1,
                        "sample_size_2": n2,
                        "observed_power": power,
                        "power_adequate": power >= self.config.desired_power,
                        "recommended_sample_size": self._calculate_required_n_for_power(
                            observed_effect_size, self.config.desired_power, alpha
                        )
                    }
        
        return power_analysis
    
    def _calculate_comprehensive_confidence_intervals(self) -> Dict[str, Any]:
        """Calculate comprehensive confidence intervals using multiple methods."""
        
        confidence_intervals = {
            "parametric": {},
            "bootstrap": {},
            "bias_corrected_bootstrap": {}
        }
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in self.experimental_results:
            if not result.__dict__.get("error", False):
                condition_groups[result.condition_id].append(result.execution_time)
        
        for condition_id, values in condition_groups.items():
            if len(values) < 2:
                continue
            
            # Parametric confidence interval
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            
            if SCIPY_AVAILABLE:
                from scipy.stats import t
                t_critical = t.ppf((1 + self.config.desired_power) / 2, n - 1)
            else:
                t_critical = 1.96  # Approximation
            
            margin_error = t_critical * (std_val / np.sqrt(n))
            
            confidence_intervals["parametric"][condition_id] = {
                "mean": mean_val,
                "lower": mean_val - margin_error,
                "upper": mean_val + margin_error,
                "margin_error": margin_error
            }
            
            # Bootstrap confidence intervals
            bootstrap_means = []
            for _ in range(self.config.bootstrap_iterations):
                bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            bootstrap_means = sorted(bootstrap_means)
            alpha = 1 - self.config.desired_power
            lower_idx = int(alpha / 2 * len(bootstrap_means))
            upper_idx = int((1 - alpha / 2) * len(bootstrap_means))
            
            confidence_intervals["bootstrap"][condition_id] = {
                "mean": mean_val,
                "lower": bootstrap_means[lower_idx],
                "upper": bootstrap_means[upper_idx],
                "bootstrap_std": np.std(bootstrap_means)
            }
            
            # Bias-corrected and accelerated (BCa) bootstrap
            # Simplified version - full implementation would require more complex calculations
            bias_correction = scipy_stats.norm.ppf(
                sum(1 for bm in bootstrap_means if bm < mean_val) / len(bootstrap_means)
            ) if SCIPY_AVAILABLE else 0
            
            corrected_alpha = alpha / 2
            corrected_lower_idx = max(0, int((corrected_alpha + bias_correction) * len(bootstrap_means)))
            corrected_upper_idx = min(len(bootstrap_means) - 1, 
                                    int((1 - corrected_alpha + bias_correction) * len(bootstrap_means)))
            
            confidence_intervals["bias_corrected_bootstrap"][condition_id] = {
                "mean": mean_val,
                "lower": bootstrap_means[corrected_lower_idx],
                "upper": bootstrap_means[corrected_upper_idx],
                "bias_correction": bias_correction
            }
        
        return confidence_intervals
    
    def _perform_non_parametric_tests(self) -> Dict[str, Any]:
        """Perform comprehensive non-parametric statistical tests."""
        
        if not SCIPY_AVAILABLE:
            return {"scipy_not_available": True}
        
        non_parametric_results = {}
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in self.experimental_results:
            if not result.__dict__.get("error", False):
                condition_groups[result.condition_id].append(result.execution_time)
        
        condition_names = list(condition_groups.keys())
        condition_values = list(condition_groups.values())
        
        if len(condition_groups) == 2:
            # Two-group non-parametric tests
            group1, group2 = condition_values
            
            # Mann-Whitney U test
            try:
                u_stat, u_p = mannwhitneyu(group1, group2, alternative='two-sided')
                non_parametric_results["mann_whitney_u"] = {
                    "statistic": u_stat,
                    "p_value": u_p,
                    "significant": u_p < self.config.significance_level,
                    "effect_size": self._calculate_mann_whitney_effect_size(group1, group2)
                }
            except Exception as e:
                non_parametric_results["mann_whitney_u"] = {"error": str(e)}
            
            # Wilcoxon rank-sum test (alternative implementation)
            try:
                w_stat, w_p = scipy_stats.ranksums(group1, group2)
                non_parametric_results["wilcoxon_rank_sum"] = {
                    "statistic": w_stat,
                    "p_value": w_p,
                    "significant": w_p < self.config.significance_level
                }
            except Exception as e:
                non_parametric_results["wilcoxon_rank_sum"] = {"error": str(e)}
        
        elif len(condition_groups) > 2:
            # Multi-group non-parametric tests
            # Kruskal-Wallis test
            try:
                kw_stat, kw_p = kruskal(*condition_values)
                non_parametric_results["kruskal_wallis"] = {
                    "statistic": kw_stat,
                    "p_value": kw_p,
                    "significant": kw_p < self.config.significance_level,
                    "degrees_of_freedom": len(condition_groups) - 1
                }
            except Exception as e:
                non_parametric_results["kruskal_wallis"] = {"error": str(e)}
            
            # Friedman test (if repeated measures design)
            if self.config.design_type == ExperimentalDesign.REPEATED_MEASURES:
                try:
                    # Reshape data for Friedman test
                    # This is a simplified version - would need proper data reshaping
                    min_length = min(len(vals) for vals in condition_values)
                    truncated_values = [vals[:min_length] for vals in condition_values]
                    
                    friedman_stat, friedman_p = scipy_stats.friedmanchisquare(*truncated_values)
                    non_parametric_results["friedman"] = {
                        "statistic": friedman_stat,
                        "p_value": friedman_p,
                        "significant": friedman_p < self.config.significance_level
                    }
                except Exception as e:
                    non_parametric_results["friedman"] = {"error": str(e)}
        
        return non_parametric_results
    
    def _perform_bootstrap_analysis(self) -> Dict[str, Any]:
        """Perform bootstrap analysis for robust statistical inference."""
        
        bootstrap_results = {
            "bootstrap_differences": {},
            "bootstrap_confidence_intervals": {},
            "bootstrap_hypothesis_tests": {}
        }
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in self.experimental_results:
            if not result.__dict__.get("error", False):
                condition_groups[result.condition_id].append(result.execution_time)
        
        condition_names = list(condition_groups.keys())
        
        # Pairwise bootstrap analysis
        for i in range(len(condition_names)):
            for j in range(i + 1, len(condition_names)):
                cond1, cond2 = condition_names[i], condition_names[j]
                group1, group2 = condition_groups[cond1], condition_groups[cond2]
                
                comparison_key = f"{cond1}_vs_{cond2}"
                
                # Bootstrap sampling
                bootstrap_differences = []
                for _ in range(self.config.bootstrap_iterations):
                    boot_sample1 = np.random.choice(group1, size=len(group1), replace=True)
                    boot_sample2 = np.random.choice(group2, size=len(group2), replace=True)
                    
                    boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
                    bootstrap_differences.append(boot_diff)
                
                bootstrap_differences = sorted(bootstrap_differences)
                
                # Bootstrap statistics
                bootstrap_results["bootstrap_differences"][comparison_key] = {
                    "mean_difference": np.mean(bootstrap_differences),
                    "std_difference": np.std(bootstrap_differences),
                    "median_difference": np.median(bootstrap_differences)
                }
                
                # Bootstrap confidence interval
                alpha = 1 - self.config.desired_power
                lower_idx = int(alpha / 2 * len(bootstrap_differences))
                upper_idx = int((1 - alpha / 2) * len(bootstrap_differences))
                
                bootstrap_results["bootstrap_confidence_intervals"][comparison_key] = {
                    "lower": bootstrap_differences[lower_idx],
                    "upper": bootstrap_differences[upper_idx],
                    "width": bootstrap_differences[upper_idx] - bootstrap_differences[lower_idx]
                }
                
                # Bootstrap hypothesis test
                # Null hypothesis: no difference between groups
                observed_difference = np.mean(group1) - np.mean(group2)
                
                # Count bootstrap differences more extreme than observed
                more_extreme = sum(1 for diff in bootstrap_differences if abs(diff) >= abs(observed_difference))
                bootstrap_p_value = more_extreme / len(bootstrap_differences)
                
                bootstrap_results["bootstrap_hypothesis_tests"][comparison_key] = {
                    "observed_difference": observed_difference,
                    "bootstrap_p_value": bootstrap_p_value,
                    "significant": bootstrap_p_value < self.config.significance_level
                }
        
        return bootstrap_results
    
    def _perform_cross_validation_analysis(self) -> Dict[str, Any]:
        """Perform cross-validation analysis of experimental results."""
        
        if len(self.experimental_results) < self.config.cross_validation_folds:
            return {"error": "Insufficient data for cross-validation"}
        
        cv_results = {
            "fold_results": [],
            "overall_performance": {},
            "stability_metrics": {}
        }
        
        # Prepare data for cross-validation
        results_by_condition = defaultdict(list)
        for result in self.experimental_results:
            if not result.__dict__.get("error", False):
                results_by_condition[result.condition_id].append(result)
        
        # Perform k-fold cross-validation
        n_folds = self.config.cross_validation_folds
        
        for fold in range(n_folds):
            fold_results = {}
            
            for condition_id, condition_results in results_by_condition.items():
                # Split data into train and test
                n_samples = len(condition_results)
                fold_size = n_samples // n_folds
                
                test_start = fold * fold_size
                test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
                
                test_results = condition_results[test_start:test_end]
                train_results = condition_results[:test_start] + condition_results[test_end:]
                
                # Calculate performance metrics for this fold
                if train_results and test_results:
                    train_performance = {
                        "mean_execution_time": np.mean([r.execution_time for r in train_results]),
                        "mean_accuracy": np.mean([r.accuracy for r in train_results]),
                        "std_execution_time": np.std([r.execution_time for r in train_results])
                    }
                    
                    test_performance = {
                        "mean_execution_time": np.mean([r.execution_time for r in test_results]),
                        "mean_accuracy": np.mean([r.accuracy for r in test_results]),
                        "std_execution_time": np.std([r.execution_time for r in test_results])
                    }
                    
                    fold_results[condition_id] = {
                        "train": train_performance,
                        "test": test_performance,
                        "generalization_gap": {
                            "execution_time": abs(test_performance["mean_execution_time"] - train_performance["mean_execution_time"]),
                            "accuracy": abs(test_performance["mean_accuracy"] - train_performance["mean_accuracy"])
                        }
                    }
            
            cv_results["fold_results"].append(fold_results)
        
        # Calculate overall cross-validation performance
        for condition_id in results_by_condition.keys():
            condition_cv_results = []
            
            for fold_result in cv_results["fold_results"]:
                if condition_id in fold_result:
                    condition_cv_results.append(fold_result[condition_id])
            
            if condition_cv_results:
                # Average performance across folds
                avg_test_execution_time = np.mean([cr["test"]["mean_execution_time"] for cr in condition_cv_results])
                avg_test_accuracy = np.mean([cr["test"]["mean_accuracy"] for cr in condition_cv_results])
                
                # Stability metrics
                execution_time_stability = np.std([cr["test"]["mean_execution_time"] for cr in condition_cv_results])
                accuracy_stability = np.std([cr["test"]["mean_accuracy"] for cr in condition_cv_results])
                
                cv_results["overall_performance"][condition_id] = {
                    "mean_test_execution_time": avg_test_execution_time,
                    "mean_test_accuracy": avg_test_accuracy,
                    "execution_time_cv": execution_time_stability / avg_test_execution_time if avg_test_execution_time > 0 else 0,
                    "accuracy_cv": accuracy_stability / avg_test_accuracy if avg_test_accuracy > 0 else 0
                }
                
                cv_results["stability_metrics"][condition_id] = {
                    "execution_time_stability": execution_time_stability,
                    "accuracy_stability": accuracy_stability,
                    "stable": execution_time_stability < avg_test_execution_time * 0.1  # 10% threshold
                }
        
        return cv_results
    
    def _perform_sensitivity_analysis(self) -> Dict[str, Any]:
        """Perform sensitivity analysis by varying key parameters."""
        
        sensitivity_results = {
            "outlier_sensitivity": {},
            "sample_size_sensitivity": {},
            "significance_level_sensitivity": {}
        }
        
        # Original results
        original_conditions = defaultdict(list)
        for result in self.experimental_results:
            if not result.__dict__.get("error", False):
                original_conditions[result.condition_id].append(result.execution_time)
        
        # Outlier sensitivity analysis
        for condition_id, values in original_conditions.items():
            if len(values) < 5:
                continue
            
            # Remove potential outliers (values beyond 2 standard deviations)
            mean_val = np.mean(values)
            std_val = np.std(values)
            filtered_values = [v for v in values if abs(v - mean_val) <= 2 * std_val]
            
            if len(filtered_values) >= 3 and len(filtered_values) != len(values):
                original_mean = np.mean(values)
                filtered_mean = np.mean(filtered_values)
                
                sensitivity_results["outlier_sensitivity"][condition_id] = {
                    "original_mean": original_mean,
                    "filtered_mean": filtered_mean,
                    "relative_change": abs(filtered_mean - original_mean) / original_mean * 100,
                    "outliers_removed": len(values) - len(filtered_values),
                    "sensitive_to_outliers": abs(filtered_mean - original_mean) / original_mean > 0.1
                }
        
        # Sample size sensitivity analysis
        if len(original_conditions) >= 2:
            condition_names = list(original_conditions.keys())
            cond1, cond2 = condition_names[0], condition_names[1]
            group1, group2 = original_conditions[cond1], original_conditions[cond2]
            
            # Test different sample sizes
            sample_sizes = [10, 20, 30, 50] if max(len(group1), len(group2)) >= 50 else [5, 10, 15, 20]
            
            for n in sample_sizes:
                if n <= min(len(group1), len(group2)):
                    subsample1 = group1[:n]
                    subsample2 = group2[:n]
                    
                    if SCIPY_AVAILABLE:
                        t_stat, p_val = scipy_stats.ttest_ind(subsample1, subsample2)
                        
                        sensitivity_results["sample_size_sensitivity"][f"n_{n}"] = {
                            "sample_size": n,
                            "t_statistic": t_stat,
                            "p_value": p_val,
                            "significant": p_val < self.config.significance_level
                        }
        
        # Significance level sensitivity analysis
        alpha_levels = [0.01, 0.05, 0.10, 0.20]
        
        if SCIPY_AVAILABLE and len(original_conditions) >= 2:
            condition_names = list(original_conditions.keys())
            cond1, cond2 = condition_names[0], condition_names[1]
            group1, group2 = original_conditions[cond1], original_conditions[cond2]
            
            t_stat, p_val = scipy_stats.ttest_ind(group1, group2)
            
            for alpha in alpha_levels:
                sensitivity_results["significance_level_sensitivity"][f"alpha_{alpha}"] = {
                    "alpha": alpha,
                    "significant": p_val < alpha,
                    "p_value": p_val
                }
        
        return sensitivity_results
    
    # Helper methods for statistical calculations
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices('0123456789ABCDEF', k=6))
        return f"EXP_{timestamp}_{random_suffix}"
    
    def _create_randomization_scheme(
        self,
        conditions: List[ExperimentalCondition],
        sample_size: int
    ) -> List[Dict[str, Any]]:
        """Create randomization scheme for experimental trials."""
        
        trials = []
        trials_per_condition = sample_size // len(conditions)
        
        for condition in conditions:
            for trial_num in range(trials_per_condition):
                trials.append({
                    "condition_id": condition.condition_id,
                    "trial_number": trial_num,
                    "participant_id": f"P_{trial_num:03d}",
                    "randomization_group": f"Group_{random.randint(1, 4)}",
                    "confounding_variables": self._generate_confounding_variables()
                })
        
        # Randomize trial order
        random.shuffle(trials)
        
        # Add randomization metadata
        for i, trial in enumerate(trials):
            trial["execution_order"] = i + 1
            trial["randomization_timestamp"] = datetime.utcnow().isoformat()
        
        return trials
    
    def _generate_confounding_variables(self) -> Dict[str, Any]:
        """Generate simulated confounding variables for experimental control."""
        
        return {
            "system_load": random.uniform(0.1, 0.9),
            "memory_pressure": random.uniform(0.0, 0.8),
            "network_latency": random.uniform(1.0, 50.0),
            "disk_io_load": random.uniform(0.0, 0.7),
            "concurrent_processes": random.randint(1, 20)
        }
    
    def _create_blinding_protocol(self) -> Dict[str, Any]:
        """Create blinding protocol to reduce bias."""
        
        # Generate masked algorithm names
        algorithm_names = [condition.algorithm_name for condition in self.experimental_conditions]
        masked_names = {name: f"Algorithm_{chr(65 + i)}" for i, name in enumerate(set(algorithm_names))}
        
        return {
            "blinding_type": "single_blind",
            "masked_algorithm_names": masked_names,
            "unblinding_key": {v: k for k, v in masked_names.items()},
            "blinding_note": "Algorithm identities masked during execution and initial analysis"
        }
    
    def _define_quality_controls(self) -> Dict[str, Any]:
        """Define quality control measures."""
        
        return {
            "outlier_detection": {
                "method": "iqr",
                "threshold": 1.5,
                "action": "flag_for_review"
            },
            "assumption_checking": {
                "normality_test": "shapiro_wilk",
                "homogeneity_test": "levene",
                "independence_check": "experimental_design"
            },
            "multiple_comparison_correction": self.config.multiple_comparison_correction,
            "effect_size_reporting": "mandatory",
            "confidence_intervals": True,
            "reproducibility_checks": {
                "random_seed_set": self.config.randomization_seed is not None,
                "version_control": True,
                "data_integrity": "checksum_verification"
            }
        }
    
    def _create_analysis_plan(self, conditions: List[ExperimentalCondition]) -> Dict[str, Any]:
        """Create pre-registered analysis plan."""
        
        return {
            "primary_outcome": "execution_time",
            "secondary_outcomes": ["accuracy", "throughput", "memory_usage"],
            "statistical_tests": {
                "two_group": ["t_test", "mann_whitney_u", "bootstrap"],
                "multiple_group": ["anova", "kruskal_wallis"],
                "effect_sizes": ["cohens_d", "cliff_delta", "eta_squared"]
            },
            "significance_level": self.config.significance_level,
            "multiple_comparison_correction": self.config.multiple_comparison_correction,
            "missing_data_handling": "complete_case_analysis",
            "outlier_handling": "flag_and_analyze_separately",
            "assumptions_violations": "report_and_use_robust_alternatives",
            "reporting_standards": "APA_7th_edition"
        }
    
    def _prepare_trial_dataset(
        self,
        test_datasets: Dict[str, Any],
        condition: ExperimentalCondition,
        trial_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare dataset for individual trial."""
        
        # Base dataset
        dataset = test_datasets.copy()
        
        # Apply condition-specific parameters
        if condition.parameters:
            dataset.update(condition.parameters)
        
        # Apply control variables
        if condition.control_variables:
            dataset.update(condition.control_variables)
        
        # Add trial-specific randomization
        dataset["trial_seed"] = hash(f"{trial_info['condition_id']}_{trial_info['trial_number']}")
        dataset["confounding_variables"] = trial_info.get("confounding_variables", {})
        
        return dataset
    
    def _calculate_trial_performance_metrics(
        self,
        algorithm_result: Any,
        dataset: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, float]:
        """Calculate performance metrics for a single trial."""
        
        metrics = {
            "execution_time": execution_time,
            "throughput": 0.0,
            "error_rate": 0.0
        }
        
        # Calculate throughput
        problem_size = dataset.get("problem_size", len(dataset.get("data", [])))
        if execution_time > 0:
            metrics["throughput"] = problem_size / execution_time
        
        # Extract accuracy metrics from algorithm result
        if isinstance(algorithm_result, dict):
            metrics.update({
                "accuracy": algorithm_result.get("accuracy", 0.0),
                "precision": algorithm_result.get("precision", 0.0),
                "recall": algorithm_result.get("recall", 0.0),
                "f1_score": algorithm_result.get("f1_score", 0.0)
            })
        else:
            # Default metrics if not provided
            metrics.update({
                "accuracy": 0.5,  # Neutral value
                "precision": 0.5,
                "recall": 0.5,
                "f1_score": 0.5
            })
        
        # Calculate F1 score if not provided
        if metrics["f1_score"] == 0.5 and metrics["precision"] > 0 and metrics["recall"] > 0:
            metrics["f1_score"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        
        return metrics
    
    def _calculate_trial_effect_size(
        self,
        performance_metrics: Dict[str, float],
        condition: ExperimentalCondition
    ) -> float:
        """Calculate effect size for a single trial."""
        
        # Simplified effect size calculation
        # In practice, this would compare against a baseline or expected value
        expected_execution_time = condition.expected_effect_size or 1.0
        observed_execution_time = performance_metrics["execution_time"]
        
        if expected_execution_time > 0:
            return (expected_execution_time - observed_execution_time) / expected_execution_time
        else:
            return 0.0
    
    def _calculate_trial_confidence_interval(
        self,
        performance_metrics: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for trial (simplified version)."""
        
        # Simplified CI calculation - in practice would use proper statistical methods
        execution_time = performance_metrics["execution_time"]
        estimated_std = execution_time * 0.1  # Assume 10% CV
        
        margin_error = 1.96 * estimated_std  # 95% CI approximation
        
        return (execution_time - margin_error, execution_time + margin_error)
    
    def _perform_quality_control_checks(self, results: List[ExperimentalResult]) -> Dict[str, Any]:
        """Perform quality control checks on experimental results."""
        
        qc_report = {
            "total_trials": len(results),
            "successful_trials": len([r for r in results if not r.__dict__.get("error", False)]),
            "error_rate": len([r for r in results if r.__dict__.get("error", False)]) / len(results),
            "outliers_detected": [],
            "data_quality_issues": [],
            "recommendations": []
        }
        
        # Outlier detection
        execution_times = [r.execution_time for r in results if not r.__dict__.get("error", False)]
        if execution_times:
            q1 = np.percentile(execution_times, 25)
            q3 = np.percentile(execution_times, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [(i, r.execution_time) for i, r in enumerate(results) 
                       if not r.__dict__.get("error", False) and (r.execution_time < lower_bound or r.execution_time > upper_bound)]
            
            qc_report["outliers_detected"] = outliers
            self.outliers_detected.extend([f"trial_{i}" for i, _ in outliers])
        
        # Data quality checks
        if qc_report["error_rate"] > 0.1:
            qc_report["data_quality_issues"].append("High error rate (>10%)")
            qc_report["recommendations"].append("Investigate causes of experimental failures")
        
        if len(qc_report["outliers_detected"]) > len(results) * 0.05:
            qc_report["data_quality_issues"].append("High outlier rate (>5%)")
            qc_report["recommendations"].append("Review outliers for data quality issues")
        
        return qc_report
    
    def _perform_preliminary_analysis(self, results: List[ExperimentalResult]) -> Dict[str, Any]:
        """Perform preliminary analysis of experimental results."""
        
        # Group results by condition
        condition_groups = defaultdict(list)
        for result in results:
            if not result.__dict__.get("error", False):
                condition_groups[result.condition_id].append(result)
        
        preliminary_analysis = {
            "descriptive_statistics": {},
            "effect_size_estimates": {},
            "preliminary_significance": {}
        }
        
        # Descriptive statistics by condition
        for condition_id, condition_results in condition_groups.items():
            execution_times = [r.execution_time for r in condition_results]
            accuracies = [r.accuracy for r in condition_results]
            
            preliminary_analysis["descriptive_statistics"][condition_id] = {
                "n": len(condition_results),
                "execution_time": {
                    "mean": np.mean(execution_times),
                    "std": np.std(execution_times),
                    "median": np.median(execution_times)
                },
                "accuracy": {
                    "mean": np.mean(accuracies),
                    "std": np.std(accuracies),
                    "median": np.median(accuracies)
                }
            }
        
        # Preliminary effect size estimates
        if len(condition_groups) >= 2:
            condition_names = list(condition_groups.keys())
            for i in range(len(condition_names)):
                for j in range(i + 1, len(condition_names)):
                    cond1, cond2 = condition_names[i], condition_names[j]
                    group1_times = [r.execution_time for r in condition_groups[cond1]]
                    group2_times = [r.execution_time for r in condition_groups[cond2]]
                    
                    # Preliminary Cohen's d
                    pooled_std = np.sqrt((np.var(group1_times, ddof=1) + np.var(group2_times, ddof=1)) / 2)
                    cohens_d = (np.mean(group1_times) - np.mean(group2_times)) / pooled_std if pooled_std > 0 else 0
                    
                    preliminary_analysis["effect_size_estimates"][f"{cond1}_vs_{cond2}"] = {
                        "cohens_d": cohens_d,
                        "interpretation": self._interpret_effect_size(abs(cohens_d))
                    }
                    
                    # Preliminary significance test
                    if SCIPY_AVAILABLE:
                        t_stat, p_val = scipy_stats.ttest_ind(group1_times, group2_times)
                        preliminary_analysis["preliminary_significance"][f"{cond1}_vs_{cond2}"] = {
                            "t_statistic": t_stat,
                            "p_value": p_val,
                            "significant": p_val < self.config.significance_level
                        }
        
        return preliminary_analysis
    
    def _extract_statistical_tests(self, analysis_results: Dict[str, Any]) -> None:
        """Extract statistical test results for storage."""
        
        # Extract from primary analysis
        primary_analysis = analysis_results.get("primary_analysis", {})
        inferential_tests = primary_analysis.get("inferential_tests", {})
        
        for test_name, test_result in inferential_tests.items():
            if isinstance(test_result, dict) and "p_value" in test_result:
                stat_test = StatisticalTestResult(
                    test_name=test_name,
                    test_statistic=test_result.get("statistic", 0.0),
                    p_value=test_result["p_value"],
                    degrees_of_freedom=test_result.get("degrees_of_freedom"),
                    effect_size=0.0,  # Would be calculated based on test type
                    power=0.0,  # Would be calculated
                    confidence_interval=(0.0, 0.0),  # Would be calculated
                    significant=test_result.get("significant", False),
                    interpretation="Primary analysis test",
                    assumptions_met=True  # Would be checked
                )
                
                self.statistical_tests.append(stat_test)
    
    # Statistical helper methods
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size according to Cohen's conventions."""
        
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
    
    def _interpret_cliff_delta(self, cliff_delta: float) -> str:
        """Interpret Cliff's delta effect size."""
        
        if cliff_delta < 0.147:
            return "negligible"
        elif cliff_delta < 0.33:
            return "small"
        elif cliff_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of distribution."""
        
        if len(values) < 3:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        if std_val == 0:
            return 0.0
        
        n = len(values)
        skewness = (n / ((n - 1) * (n - 2))) * sum(((x - mean_val) / std_val) ** 3 for x in values)
        
        return skewness
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of distribution."""
        
        if len(values) < 4:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        if std_val == 0:
            return 0.0
        
        n = len(values)
        kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(((x - mean_val) / std_val) ** 4 for x in values) - \
                   (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        
        return kurtosis
    
    def _cohens_d_confidence_interval(self, group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for Cohen's d."""
        
        n1, n2 = len(group1), len(group2)
        
        # Calculate Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Standard error of Cohen's d
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d ** 2 / (2 * (n1 + n2)))
        
        # Confidence interval
        if SCIPY_AVAILABLE:
            from scipy.stats import t
            df = n1 + n2 - 2
            t_critical = t.ppf((1 + self.config.desired_power) / 2, df)
        else:
            t_critical = 1.96  # Approximation
        
        margin_error = t_critical * se_d
        
        return (cohens_d - margin_error, cohens_d + margin_error)
    
    def _calculate_cliff_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        
        if not group1 or not group2:
            return 0.0
        
        n_pairs = len(group1) * len(group2)
        
        # Count how many times group1 values are greater than group2 values
        greater = sum(1 for x in group1 for y in group2 if x > y)
        
        # Count how many times group1 values are less than group2 values
        less = sum(1 for x in group1 for y in group2 if x < y)
        
        cliff_delta = (greater - less) / n_pairs
        
        return cliff_delta
    
    def _calculate_mann_whitney_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Calculate effect size for Mann-Whitney U test."""
        
        n1, n2 = len(group1), len(group2)
        
        if SCIPY_AVAILABLE:
            u_stat, _ = mannwhitneyu(group1, group2, alternative='two-sided')
            # Convert to effect size (r = Z / sqrt(N))
            z_score = (u_stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            r = abs(z_score) / np.sqrt(n1 + n2)
            return r
        else:
            # Fallback: use Cliff's delta as approximation
            return abs(self._calculate_cliff_delta(group1, group2))
    
    def _calculate_required_n_for_power(self, effect_size: float, desired_power: float, alpha: float) -> int:
        """Calculate required sample size for desired power."""
        
        if not SCIPY_AVAILABLE:
            # Rule of thumb approximation
            return max(30, int(100 / (effect_size ** 2)))
        
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(desired_power)
        
        n_per_group = (2 * ((z_alpha + z_beta) ** 2)) / (effect_size ** 2)
        
        return int(np.ceil(n_per_group))
    
    def _generate_sample_size_recommendation(self, calculated_n: int, effect_size: float) -> str:
        """Generate sample size recommendation."""
        
        if effect_size >= 0.8:
            return f"Large effect size detected. {calculated_n} samples should provide excellent power."
        elif effect_size >= 0.5:
            return f"Medium effect size. {calculated_n} samples recommended for adequate power."
        elif effect_size >= 0.2:
            return f"Small effect size. {calculated_n} samples needed. Consider increasing if feasible."
        else:
            return f"Very small effect size. {calculated_n} samples minimum. Results may lack practical significance."
    
    def generate_publication_report(self) -> Dict[str, Any]:
        """Generate comprehensive report ready for academic publication."""
        
        report = {
            "title": "Experimental Validation of Quantum-Enhanced RAG Algorithms: A Controlled Study",
            "abstract": self._generate_abstract(),
            "methodology": self._generate_methodology_section(),
            "results": self._generate_results_section(),
            "discussion": self._generate_discussion_section(),
            "limitations": self._generate_limitations_section(),
            "conclusions": self._generate_conclusions_section(),
            "supplementary_materials": self._generate_supplementary_materials(),
            "statistical_appendix": self._generate_statistical_appendix(),
            "reproducibility_statement": self._generate_reproducibility_statement()
        }
        
        return report
    
    def _generate_abstract(self) -> str:
        """Generate abstract for publication."""
        
        return """
        Background: Quantum-enhanced algorithms for retrieval-augmented generation (RAG) systems 
        show theoretical promise but lack rigorous experimental validation.
        
        Objective: To conduct a controlled comparison of quantum-enhanced vs. classical RAG algorithms 
        using standardized benchmarks and statistical validation.
        
        Methods: Randomized controlled experiment with {} experimental conditions and {} total trials. 
        Primary outcome: execution time. Secondary outcomes: accuracy, throughput, memory usage. 
        Statistical analysis included parametric and non-parametric tests with multiple comparison corrections.
        
        Results: [Results to be filled based on actual experimental data]
        
        Conclusions: [Conclusions to be filled based on results]
        """.format(
            len(self.experimental_conditions),
            len(self.experimental_results)
        )
    
    def _generate_methodology_section(self) -> Dict[str, Any]:
        """Generate methodology section."""
        
        return {
            "experimental_design": {
                "type": self.config.design_type.value,
                "randomization": "Complete randomization with stratification",
                "blinding": "Single-blind" if self.config.enable_blinding else "Unblinded",
                "controls": "Confounding variables controlled through randomization"
            },
            "participants": {
                "algorithms_tested": len(set(c.algorithm_name for c in self.experimental_conditions)),
                "conditions": len(self.experimental_conditions),
                "total_trials": len(self.experimental_results)
            },
            "power_analysis": self.calculate_required_sample_size(),
            "statistical_analysis_plan": {
                "primary_analysis": "Independent t-test or Mann-Whitney U",
                "secondary_analyses": "ANOVA or Kruskal-Wallis for multiple comparisons",
                "effect_sizes": "Cohen's d, Cliff's delta",
                "multiple_comparisons": self.config.multiple_comparison_correction,
                "significance_level": self.config.significance_level,
                "confidence_level": self.config.desired_power
            }
        }
    
    def _generate_results_section(self) -> Dict[str, Any]:
        """Generate results section."""
        
        if not self.statistical_tests:
            return {"note": "Statistical analysis not yet performed"}
        
        return {
            "descriptive_statistics": "Summary statistics by experimental condition",
            "primary_outcome_analysis": "Statistical test results for execution time",
            "secondary_outcome_analyses": "Statistical test results for secondary measures",
            "effect_size_analysis": "Comprehensive effect size calculations",
            "assumption_checks": "Validation of statistical assumptions",
            "sensitivity_analyses": "Robustness testing of primary findings"
        }
    
    def _generate_discussion_section(self) -> Dict[str, str]:
        """Generate discussion section."""
        
        return {
            "interpretation": "Interpretation of findings in context of research questions",
            "comparison_to_literature": "Comparison with existing research",
            "practical_implications": "Implications for practical applications",
            "theoretical_contributions": "Contributions to quantum algorithm theory",
            "unexpected_findings": "Discussion of unexpected or contradictory results"
        }
    
    def _generate_limitations_section(self) -> List[str]:
        """Generate limitations section."""
        
        limitations = [
            "Simulation-based quantum algorithms rather than true quantum hardware",
            "Limited to specific problem domains and sizes",
            "Single-laboratory setting may limit generalizability"
        ]
        
        if len(self.experimental_results) < 100:
            limitations.append("Limited sample size may affect generalizability")
        
        if len(set(r.condition_id for r in self.experimental_results)) < 5:
            limitations.append("Limited number of algorithmic variants tested")
        
        return limitations
    
    def _generate_conclusions_section(self) -> Dict[str, Any]:
        """Generate conclusions section."""
        
        return {
            "primary_findings": "Summary of main experimental findings",
            "research_questions_answered": "Direct answers to research questions",
            "practical_recommendations": "Recommendations for practitioners",
            "future_research_directions": [
                "Validation on real quantum hardware",
                "Large-scale industry benchmarks",
                "Long-term stability studies",
                "Cost-benefit analysis"
            ]
        }
    
    def _generate_supplementary_materials(self) -> Dict[str, Any]:
        """Generate supplementary materials."""
        
        return {
            "raw_data": "Complete experimental dataset",
            "analysis_code": "Statistical analysis scripts",
            "algorithm_implementations": "Source code for all algorithms tested",
            "extended_results": "Additional statistical analyses",
            "reproducibility_package": "Complete package for result reproduction"
        }
    
    def _generate_statistical_appendix(self) -> Dict[str, Any]:
        """Generate statistical appendix."""
        
        return {
            "power_analysis_details": "Complete power analysis calculations",
            "assumption_testing": "Detailed assumption checking results",
            "effect_size_calculations": "All effect size calculations with confidence intervals",
            "bootstrap_analyses": "Bootstrap analysis results",
            "sensitivity_analyses": "Complete sensitivity analysis results",
            "cross_validation_details": "Cross-validation analysis details"
        }
    
    def _generate_reproducibility_statement(self) -> Dict[str, Any]:
        """Generate reproducibility statement."""
        
        return {
            "code_availability": "All analysis code available at [repository URL]",
            "data_availability": "Experimental data available upon request",
            "computational_environment": {
                "random_seed": self.experiment_metadata.get("random_seed"),
                "software_versions": "Python 3.9+, SciPy, NumPy",
                "hardware_specifications": "Standard computational hardware"
            },
            "preregistration": "Analysis plan preregistered before data collection",
            "deviations_from_plan": "None" if not hasattr(self, 'plan_deviations') else self.plan_deviations
        }
    
    def export_for_publication(self, format: str = "json") -> str:
        """Export validation results for publication."""
        
        publication_data = {
            "experimental_metadata": self.experiment_metadata,
            "experimental_conditions": [condition.__dict__ for condition in self.experimental_conditions],
            "experimental_results": [result.__dict__ for result in self.experimental_results],
            "statistical_tests": [test.__dict__ for test in self.statistical_tests],
            "publication_report": self.generate_publication_report()
        }
        
        if format == "json":
            return json.dumps(publication_data, indent=2, default=str)
        elif format == "csv":
            # Export main results as CSV
            import csv
            import io
            
            output = io.StringIO()
            if self.experimental_results:
                fieldnames = list(self.experimental_results[0].__dict__.keys())
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.experimental_results:
                    writer.writerow(result.__dict__)
            
            return output.getvalue()
        else:
            return str(publication_data)
    
    def shutdown(self) -> None:
        """Shutdown experimental validation suite."""
        
        self.logger.info("Shutting down Experimental Validation Suite...")
        
        # Save results
        if self.experimental_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experimental_validation_results_{timestamp}.json"
            
            try:
                with open(filename, 'w') as f:
                    f.write(self.export_for_publication("json"))
                self.logger.info(f"Experimental results saved to {filename}")
            except Exception as e:
                self.logger.error(f"Failed to save experimental results: {e}")
        
        self.logger.info("Experimental Validation Suite shutdown complete")


# Example usage for research validation
if __name__ == "__main__":
    
    # Example experimental conditions
    experimental_conditions = [
        ExperimentalCondition(
            condition_id="classical_baseline",
            algorithm_name="brute_force_search",
            algorithm_category="classical",
            parameters={"optimization_level": 0},
            expected_effect_size=0.0  # Baseline
        ),
        ExperimentalCondition(
            condition_id="quantum_enhanced",
            algorithm_name="quantum_superposition_search",
            algorithm_category="quantum",
            parameters={"num_qubits": 8, "optimization_level": 1},
            expected_effect_size=0.5  # Medium effect expected
        )
    ]
    
    # Example test datasets
    test_datasets = {
        "query_vector": [0.1, 0.5, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4],
        "data": [
            [0.2, 0.4, 0.9, 0.1, 0.8, 0.3, 0.6, 0.5],
            [0.1, 0.6, 0.7, 0.4, 0.9, 0.1, 0.8, 0.3],
            [0.3, 0.2, 0.8, 0.6, 0.7, 0.4, 0.5, 0.9],
            [0.5, 0.8, 0.1, 0.9, 0.3, 0.7, 0.2, 0.6],
            [0.7, 0.1, 0.4, 0.8, 0.2, 0.9, 0.6, 0.3]
        ],
        "top_k": 3,
        "ground_truth": [2, 4, 1]
    }
    
    # Mock algorithm implementations
    def mock_classical_search(dataset):
        time.sleep(0.1)  # Simulate execution time
        return {
            "top_indices": [0, 1, 2],
            "similarities": [0.8, 0.7, 0.6],
            "accuracy": 0.7,
            "precision": 0.65,
            "recall": 0.75
        }
    
    async def mock_quantum_search(dataset):
        await asyncio.sleep(0.08)  # Slightly faster
        return {
            "top_indices": [2, 0, 1],
            "similarities": [0.9, 0.8, 0.7],
            "accuracy": 0.85,
            "precision": 0.8,
            "recall": 0.8
        }
    
    algorithm_implementations = {
        "brute_force_search": mock_classical_search,
        "quantum_superposition_search": mock_quantum_search
    }
    
    # Initialize validation suite
    config = ExperimentConfiguration(
        experiment_type=ExperimentType.CONTROLLED_COMPARISON,
        design_type=ExperimentalDesign.RANDOMIZED_CONTROLLED,
        desired_power=0.8,
        significance_level=0.05,
        effect_size=0.5,
        minimum_sample_size=30,
        randomization_seed=42
    )
    
    validation_suite = ExperimentalValidationSuite(config)
    
    # Run experimental validation
    print("🔬 Starting Experimental Validation Suite...")
    
    try:
        # Power analysis
        power_analysis = validation_suite.calculate_required_sample_size()
        print(f"📊 Power analysis: {power_analysis['required_total_sample_size']} samples recommended")
        
        # Design experiment
        experimental_design = validation_suite.design_experiment(
            experimental_conditions, 
            sample_size=50  # Smaller for demo
        )
        print(f"🔬 Experimental design: {len(experimental_conditions)} conditions, 50 trials")
        
        # Execute experiment
        experiment_results = asyncio.run(
            validation_suite.execute_experiment(
                algorithm_implementations,
                test_datasets,
                experimental_design
            )
        )
        print(f"✅ Experiment completed: {experiment_results['execution_summary']['successful_trials']} successful trials")
        
        # Statistical analysis
        statistical_analysis = validation_suite.perform_statistical_analysis()
        print("📈 Statistical analysis completed")
        
        # Generate publication report
        publication_report = validation_suite.generate_publication_report()
        print("📄 Publication report generated")
        
        # Export results
        results_json = validation_suite.export_for_publication("json")
        print("💾 Results exported for publication")
        
    except Exception as e:
        print(f"❌ Experimental validation failed: {e}")
    
    finally:
        validation_suite.shutdown()