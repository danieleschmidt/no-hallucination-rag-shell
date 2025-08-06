"""
Integration layer between quantum task planning and RAG system.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .quantum_planner import QuantumTaskPlanner, QuantumTask, TaskState, Priority
from .superposition_tasks import SuperpositionTaskManager
from .entanglement_dependencies import EntanglementDependencyGraph, EntanglementType
from .quantum_validator import QuantumValidator
from .quantum_security import QuantumSecurityManager, QuantumSecurityLevel
from .quantum_logging import QuantumLogger
from .quantum_optimizer import QuantumOptimizer, OptimizationStrategy
from .quantum_commands import QuantumCommands


@dataclass
class QuantumRAGConfig:
    """Configuration for quantum-enhanced RAG system."""
    enable_quantum_planning: bool = True
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    security_level: QuantumSecurityLevel = QuantumSecurityLevel.PROTECTED
    cache_size: int = 10000
    max_quantum_tasks: int = 1000
    coherence_time: float = 3600.0
    enable_auto_optimization: bool = True
    enable_parallel_execution: bool = True
    log_quantum_operations: bool = True
    quantum_interference_threshold: float = 0.1


class QuantumEnhancedRAG:
    """
    Quantum-enhanced RAG system that integrates quantum task planning
    with traditional RAG retrieval and generation.
    
    Provides quantum superposition-based query optimization, entangled
    task execution, and quantum-inspired performance optimization.
    """
    
    def __init__(
        self,
        config: Optional[QuantumRAGConfig] = None,
        base_rag_system: Optional[Any] = None
    ):
        self.config = config or QuantumRAGConfig()
        self.base_rag_system = base_rag_system
        
        # Initialize quantum components
        self.quantum_planner = QuantumTaskPlanner(max_coherence_time=self.config.coherence_time)
        self.superposition_manager = SuperpositionTaskManager()
        self.entanglement_graph = EntanglementDependencyGraph()
        self.validator = QuantumValidator(max_tasks=self.config.max_quantum_tasks)
        self.security_manager = QuantumSecurityManager(self.config.security_level)
        
        # Initialize optimization and logging
        self.optimizer = QuantumOptimizer(
            strategy=self.config.optimization_strategy,
            enable_auto_optimization=self.config.enable_auto_optimization,
            cache_size=self.config.cache_size
        )
        
        self.logger = QuantumLogger() if self.config.log_quantum_operations else None
        self.commands = QuantumCommands()
        
        # Active quantum queries
        self.active_quantum_queries: Dict[str, QuantumTask] = {}
        self.query_entanglements: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.quantum_performance_metrics = {
            "queries_processed": 0,
            "quantum_speedup": 0.0,
            "cache_hit_rate": 0.0,
            "parallel_execution_factor": 0.0
        }
        
        self.logger_instance = logging.getLogger(__name__)
        self.logger_instance.info("Quantum-Enhanced RAG system initialized")
    
    async def query_with_quantum_enhancement(
        self,
        question: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        quantum_hints: Optional[Dict[str, Any]] = None,
        require_citations: bool = True,
        min_sources: int = 2,
        min_factuality_score: float = 0.95,
        client_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process query with quantum enhancement.
        
        Creates quantum tasks for different aspects of the query,
        uses superposition for parallel exploration, and entanglement
        for correlated information retrieval.
        """
        start_time = time.time()
        quantum_hints = quantum_hints or {}
        client_context = client_context or {}
        
        try:
            # Create quantum task for this query
            quantum_task = self._create_quantum_query_task(
                question, user_id, session_id, quantum_hints
            )
            
            if self.logger:
                self.logger.log_task_created(quantum_task, user_id, session_id)
            
            # Security validation
            if not self._validate_quantum_query_security(quantum_task, user_id, client_context):
                return self._create_security_error_response(question)
            
            # Create superposition for multiple query interpretations
            query_superposition = await self._create_query_superposition(quantum_task, question)
            
            # Execute quantum-enhanced retrieval
            quantum_results = await self._execute_quantum_retrieval(
                quantum_task, question, query_superposition, quantum_hints
            )
            
            # Apply quantum optimization to results
            optimized_results = await self._optimize_quantum_results(quantum_results)
            
            # Integrate with base RAG system if available
            if self.base_rag_system:
                base_results = await self._integrate_with_base_rag(
                    question, optimized_results, require_citations, min_sources, min_factuality_score
                )
            else:
                base_results = self._create_quantum_only_response(optimized_results)
            
            # Apply quantum post-processing
            final_response = await self._apply_quantum_postprocessing(
                base_results, quantum_task, optimized_results
            )
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self._update_quantum_performance_metrics(execution_time, quantum_task)
            
            # Log completion
            if self.logger:
                self.logger.log_task_observed(quantum_task, TaskState.COMPLETED, user_id, session_id)
            
            return final_response
            
        except Exception as e:
            self.logger_instance.error(f"Quantum-enhanced query failed: {e}")
            
            if self.logger:
                self.logger.log_system_error("quantum_query", type(e).__name__, str(e), 
                                           quantum_task.id if 'quantum_task' in locals() else None, user_id)
            
            # Fallback to base RAG system
            if self.base_rag_system:
                return await self._fallback_to_base_rag(question, require_citations, min_sources, min_factuality_score)
            else:
                return self._create_error_response(question, str(e))
    
    def _create_quantum_query_task(
        self,
        question: str,
        user_id: Optional[str],
        session_id: Optional[str], 
        quantum_hints: Dict[str, Any]
    ) -> QuantumTask:
        """Create quantum task for query processing."""
        
        # Determine priority based on query characteristics
        priority = Priority.SECOND_EXCITED
        if quantum_hints.get("urgent", False):
            priority = Priority.IONIZED
        elif quantum_hints.get("complex", False):
            priority = Priority.THIRD_EXCITED
        elif quantum_hints.get("simple", False):
            priority = Priority.FIRST_EXCITED
        
        # Estimate duration based on query complexity
        estimated_duration = timedelta(seconds=5)  # Base time
        if len(question.split()) > 20:
            estimated_duration += timedelta(seconds=10)
        if quantum_hints.get("requires_deep_analysis", False):
            estimated_duration += timedelta(seconds=30)
        
        # Create quantum task
        task = self.quantum_planner.create_task(
            title=f"Quantum Query: {question[:50]}...",
            description=f"Process query with quantum enhancement: {question}",
            priority=priority,
            estimated_duration=estimated_duration,
            context={
                "query": question,
                "user_id": user_id,
                "session_id": session_id,
                "quantum_hints": quantum_hints
            }
        )
        
        self.active_quantum_queries[task.id] = task
        return task
    
    def _validate_quantum_query_security(
        self,
        quantum_task: QuantumTask,
        user_id: Optional[str],
        client_context: Dict[str, Any]
    ) -> bool:
        """Validate quantum query security."""
        
        try:
            user_roles = client_context.get("user_roles", ["user"])
            
            success, message, _ = self.security_manager.secure_task_creation(
                quantum_task, user_id or "anonymous", user_roles, client_context
            )
            
            return success
            
        except Exception as e:
            self.logger_instance.error(f"Quantum query security validation failed: {e}")
            return False
    
    async def _create_query_superposition(
        self,
        quantum_task: QuantumTask,
        question: str
    ) -> Dict[str, float]:
        """Create superposition of different query interpretations."""
        
        try:
            # Analyze query to determine possible interpretations
            interpretations = self._analyze_query_interpretations(question)
            
            # Create superposition state
            superposition_states = {}
            total_weight = sum(interpretations.values())
            
            for interpretation, weight in interpretations.items():
                normalized_weight = weight / total_weight
                
                # Map interpretations to task states
                if "factual" in interpretation:
                    superposition_states[TaskState.COLLAPSED] = normalized_weight
                elif "analytical" in interpretation:
                    superposition_states[TaskState.SUPERPOSITION] = normalized_weight
                elif "complex" in interpretation:
                    superposition_states[TaskState.ENTANGLED] = normalized_weight
                else:
                    superposition_states[TaskState.COLLAPSED] = superposition_states.get(TaskState.COLLAPSED, 0) + normalized_weight
            
            # Create superposition
            self.superposition_manager.create_superposition(quantum_task.id, superposition_states)
            
            return interpretations
            
        except Exception as e:
            self.logger_instance.error(f"Query superposition creation failed: {e}")
            return {"default": 1.0}
    
    def _analyze_query_interpretations(self, question: str) -> Dict[str, float]:
        """Analyze query to determine possible interpretations and their weights."""
        
        interpretations = {}
        
        # Factual question patterns
        factual_keywords = ["what", "where", "when", "who", "how many", "define"]
        if any(keyword in question.lower() for keyword in factual_keywords):
            interpretations["factual_retrieval"] = 0.8
        
        # Analytical question patterns
        analytical_keywords = ["why", "how", "analyze", "compare", "evaluate", "explain"]
        if any(keyword in question.lower() for keyword in analytical_keywords):
            interpretations["analytical_reasoning"] = 0.7
        
        # Complex/multi-step patterns
        complex_indicators = ["and", "also", "furthermore", "in addition", "moreover"]
        if any(indicator in question.lower() for indicator in complex_indicators) or len(question.split()) > 15:
            interpretations["complex_processing"] = 0.6
        
        # Opinion/subjective patterns
        opinion_keywords = ["should", "better", "prefer", "opinion", "recommend"]
        if any(keyword in question.lower() for keyword in opinion_keywords):
            interpretations["subjective_analysis"] = 0.5
        
        # Default if no specific patterns found
        if not interpretations:
            interpretations["general_query"] = 1.0
        
        return interpretations
    
    async def _execute_quantum_retrieval(
        self,
        quantum_task: QuantumTask,
        question: str,
        query_superposition: Dict[str, float],
        quantum_hints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute retrieval with quantum enhancement."""
        
        try:
            # Observe the quantum task to collapse superposition
            observed_task = self.quantum_planner.observe_task(quantum_task.id)
            collapsed_state = self.superposition_manager.measure_superposition(quantum_task.id)
            
            # Create retrieval subtasks based on collapsed state and interpretations
            retrieval_tasks = await self._create_retrieval_subtasks(
                observed_task, question, query_superposition, quantum_hints
            )
            
            # Execute retrieval tasks in parallel if enabled
            if self.config.enable_parallel_execution and len(retrieval_tasks) > 1:
                retrieval_results = await self._execute_parallel_retrieval(retrieval_tasks)
            else:
                retrieval_results = await self._execute_sequential_retrieval(retrieval_tasks)
            
            # Apply quantum interference effects
            interfered_results = self._apply_quantum_interference(retrieval_results, quantum_task)
            
            return {
                "primary_results": interfered_results,
                "quantum_task_id": quantum_task.id,
                "collapsed_state": collapsed_state.value if collapsed_state else None,
                "retrieval_task_count": len(retrieval_tasks),
                "quantum_interference_applied": True
            }
            
        except Exception as e:
            self.logger_instance.error(f"Quantum retrieval execution failed: {e}")
            raise
    
    async def _create_retrieval_subtasks(
        self,
        parent_task: QuantumTask,
        question: str,
        interpretations: Dict[str, float],
        quantum_hints: Dict[str, Any]
    ) -> List[QuantumTask]:
        """Create retrieval subtasks based on query interpretations."""
        
        subtasks = []
        
        for interpretation, weight in interpretations.items():
            if weight < 0.3:  # Skip low-weight interpretations
                continue
            
            # Create subtask for this interpretation
            subtask = self.quantum_planner.create_task(
                title=f"Retrieval: {interpretation}",
                description=f"Retrieve information for {interpretation} aspect of query",
                priority=Priority.FIRST_EXCITED,
                dependencies={parent_task.id},
                estimated_duration=timedelta(seconds=2),
                context={
                    "interpretation": interpretation,
                    "weight": weight,
                    "query": question,
                    "parent_task": parent_task.id
                }
            )
            
            # Entangle subtask with parent
            self.quantum_planner.entangle_tasks(parent_task.id, subtask.id, weight)
            self.entanglement_graph.create_entanglement(
                parent_task.id, subtask.id, 
                EntanglementType.SPIN_CORRELATED, weight
            )
            
            subtasks.append(subtask)
        
        return subtasks
    
    async def _execute_parallel_retrieval(self, retrieval_tasks: List[QuantumTask]) -> List[Dict[str, Any]]:
        """Execute retrieval tasks in parallel."""
        
        try:
            results = await self.optimizer.optimize_parallel_execution(
                self.quantum_planner, retrieval_tasks, max_concurrent=4
            )
            
            return [task_result for task_result in results["tasks_executed"]]
            
        except Exception as e:
            self.logger_instance.error(f"Parallel retrieval execution failed: {e}")
            # Fallback to sequential
            return await self._execute_sequential_retrieval(retrieval_tasks)
    
    async def _execute_sequential_retrieval(self, retrieval_tasks: List[QuantumTask]) -> List[Dict[str, Any]]:
        """Execute retrieval tasks sequentially."""
        
        results = []
        
        for task in retrieval_tasks:
            try:
                # Simulate retrieval execution
                task_result = {
                    "task_id": task.id,
                    "interpretation": task.context.get("interpretation", "unknown"),
                    "weight": task.context.get("weight", 1.0),
                    "retrieval_data": f"Retrieved data for {task.context.get('interpretation', 'query')}",
                    "execution_time": 0.1
                }
                
                results.append(task_result)
                
            except Exception as e:
                self.logger_instance.error(f"Sequential retrieval task failed: {e}")
        
        return results
    
    def _apply_quantum_interference(
        self, 
        retrieval_results: List[Dict[str, Any]], 
        quantum_task: QuantumTask
    ) -> List[Dict[str, Any]]:
        """Apply quantum interference effects to retrieval results."""
        
        try:
            # Calculate interference scores
            interference_scores = self.quantum_planner.calculate_task_interference()
            
            # Apply interference to results
            interfered_results = []
            
            for result in retrieval_results:
                task_id = result.get("task_id", "")
                interference_score = interference_scores.get(task_id, 0.0)
                
                # Modify result weight based on interference
                original_weight = result.get("weight", 1.0)
                
                if abs(interference_score) > self.config.quantum_interference_threshold:
                    # Constructive or destructive interference
                    if interference_score > 0:
                        # Constructive interference - boost result
                        modified_weight = original_weight * (1 + interference_score)
                    else:
                        # Destructive interference - reduce result
                        modified_weight = original_weight * (1 + interference_score)
                        modified_weight = max(0.1, modified_weight)  # Minimum weight
                    
                    result["weight"] = modified_weight
                    result["quantum_interference"] = interference_score
                
                interfered_results.append(result)
            
            return interfered_results
            
        except Exception as e:
            self.logger_instance.error(f"Quantum interference application failed: {e}")
            return retrieval_results
    
    async def _optimize_quantum_results(self, quantum_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quantum optimization to results."""
        
        try:
            # Use optimizer to enhance results
            primary_results = quantum_results["primary_results"]
            
            # Sort results by quantum-enhanced weight
            optimized_results = sorted(
                primary_results,
                key=lambda x: x.get("weight", 0.0) * (1 + x.get("quantum_interference", 0.0)),
                reverse=True
            )
            
            # Apply quantum filtering
            filtered_results = self._apply_quantum_filtering(optimized_results)
            
            quantum_results["primary_results"] = filtered_results
            quantum_results["optimization_applied"] = True
            quantum_results["optimization_strategy"] = self.optimizer.strategy.value
            
            return quantum_results
            
        except Exception as e:
            self.logger_instance.error(f"Quantum results optimization failed: {e}")
            return quantum_results
    
    def _apply_quantum_filtering(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply quantum-inspired filtering to results."""
        
        # Filter based on quantum coherence and entanglement
        filtered_results = []
        
        for result in results:
            # Check quantum coherence (relevance threshold)
            weight = result.get("weight", 0.0)
            interference = result.get("quantum_interference", 0.0)
            
            # Quantum coherence score
            coherence_score = weight * (1 + abs(interference))
            
            if coherence_score > 0.3:  # Coherence threshold
                result["quantum_coherence_score"] = coherence_score
                filtered_results.append(result)
        
        return filtered_results
    
    async def _integrate_with_base_rag(
        self,
        question: str,
        quantum_results: Dict[str, Any],
        require_citations: bool,
        min_sources: int,
        min_factuality_score: float
    ) -> Dict[str, Any]:
        """Integrate quantum results with base RAG system."""
        
        try:
            # Extract quantum-enhanced context for base RAG
            quantum_context = self._extract_quantum_context(quantum_results)
            
            # Call base RAG system with quantum context
            if hasattr(self.base_rag_system, 'query'):
                base_response = self.base_rag_system.query(
                    question,
                    require_citations=require_citations,
                    min_sources=min_sources,
                    min_factuality_score=min_factuality_score,
                    client_context={"quantum_context": quantum_context}
                )
            else:
                base_response = self._create_quantum_only_response(quantum_results)
            
            # Enhance base response with quantum metadata
            base_response["quantum_enhancement"] = {
                "applied": True,
                "quantum_task_id": quantum_results.get("quantum_task_id"),
                "collapsed_state": quantum_results.get("collapsed_state"),
                "interference_effects": len([r for r in quantum_results["primary_results"] if r.get("quantum_interference", 0) != 0]),
                "optimization_strategy": self.optimizer.strategy.value
            }
            
            return base_response
            
        except Exception as e:
            self.logger_instance.error(f"Base RAG integration failed: {e}")
            return self._create_quantum_only_response(quantum_results)
    
    def _extract_quantum_context(self, quantum_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quantum context for base RAG system."""
        
        primary_results = quantum_results.get("primary_results", [])
        
        return {
            "quantum_interpretations": [r.get("interpretation", "") for r in primary_results],
            "quantum_weights": [r.get("weight", 0.0) for r in primary_results],
            "interference_scores": [r.get("quantum_interference", 0.0) for r in primary_results],
            "coherence_scores": [r.get("quantum_coherence_score", 0.0) for r in primary_results],
            "optimization_applied": quantum_results.get("optimization_applied", False)
        }
    
    def _create_quantum_only_response(self, quantum_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create response using only quantum results."""
        
        primary_results = quantum_results.get("primary_results", [])
        
        # Synthesize answer from quantum results
        answer_parts = []
        for result in primary_results[:3]:  # Top 3 results
            interpretation = result.get("interpretation", "unknown")
            weight = result.get("weight", 0.0)
            data = result.get("retrieval_data", "")
            
            answer_parts.append(f"[{interpretation.title()} - Weight: {weight:.2f}] {data}")
        
        answer = "\n\n".join(answer_parts)
        
        return {
            "answer": answer,
            "sources": [{"title": r.get("interpretation", ""), "weight": r.get("weight", 0.0)} for r in primary_results],
            "factuality_score": sum(r.get("weight", 0.0) for r in primary_results) / len(primary_results) if primary_results else 0.0,
            "governance_compliant": True,
            "quantum_enhanced": True,
            "quantum_metadata": quantum_results
        }
    
    async def _apply_quantum_postprocessing(
        self,
        base_response: Dict[str, Any],
        quantum_task: QuantumTask,
        quantum_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply quantum post-processing to final response."""
        
        try:
            # Apply quantum coherence scoring to final answer
            coherence_score = self._calculate_response_coherence(base_response, quantum_results)
            
            # Add quantum analytics
            base_response["quantum_analytics"] = {
                "coherence_score": coherence_score,
                "quantum_task_id": quantum_task.id,
                "task_state": quantum_task.state.value,
                "entanglement_count": len(quantum_task.entangled_tasks),
                "processing_time": (datetime.utcnow() - quantum_task.created_at).total_seconds(),
                "optimization_benefits": self._calculate_optimization_benefits(quantum_results)
            }
            
            # Mark task as completed
            quantum_task.state = TaskState.COMPLETED
            quantum_task.progress = 1.0
            
            return base_response
            
        except Exception as e:
            self.logger_instance.error(f"Quantum post-processing failed: {e}")
            return base_response
    
    def _calculate_response_coherence(
        self, 
        response: Dict[str, Any], 
        quantum_results: Dict[str, Any]
    ) -> float:
        """Calculate quantum coherence score for response."""
        
        try:
            # Base factuality score
            base_score = response.get("factuality_score", 0.0)
            
            # Quantum enhancement factors
            primary_results = quantum_results.get("primary_results", [])
            avg_quantum_weight = sum(r.get("weight", 0.0) for r in primary_results) / len(primary_results) if primary_results else 0.0
            avg_interference = sum(abs(r.get("quantum_interference", 0.0)) for r in primary_results) / len(primary_results) if primary_results else 0.0
            
            # Coherence formula: base * (1 + quantum_enhancement)
            quantum_enhancement = (avg_quantum_weight * 0.1) + (avg_interference * 0.05)
            coherence_score = base_score * (1 + quantum_enhancement)
            
            return min(1.0, coherence_score)
            
        except Exception as e:
            self.logger_instance.error(f"Coherence calculation failed: {e}")
            return response.get("factuality_score", 0.0)
    
    def _calculate_optimization_benefits(self, quantum_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimization benefits from quantum processing."""
        
        try:
            primary_results = quantum_results.get("primary_results", [])
            
            # Parallelization benefit
            parallelization_benefit = len(primary_results) * 0.1 if len(primary_results) > 1 else 0.0
            
            # Interference benefit
            interference_effects = sum(abs(r.get("quantum_interference", 0.0)) for r in primary_results)
            interference_benefit = min(0.5, interference_effects * 0.1)
            
            # Cache benefit (estimated)
            cache_benefit = 0.1 if quantum_results.get("optimization_applied", False) else 0.0
            
            return {
                "parallelization": parallelization_benefit,
                "interference": interference_benefit,
                "caching": cache_benefit,
                "total": parallelization_benefit + interference_benefit + cache_benefit
            }
            
        except Exception as e:
            self.logger_instance.error(f"Optimization benefits calculation failed: {e}")
            return {"parallelization": 0.0, "interference": 0.0, "caching": 0.0, "total": 0.0}
    
    async def _fallback_to_base_rag(
        self,
        question: str,
        require_citations: bool,
        min_sources: int,
        min_factuality_score: float
    ) -> Dict[str, Any]:
        """Fallback to base RAG system without quantum enhancement."""
        
        if self.base_rag_system and hasattr(self.base_rag_system, 'query'):
            response = self.base_rag_system.query(
                question,
                require_citations=require_citations,
                min_sources=min_sources,
                min_factuality_score=min_factuality_score
            )
            response["quantum_enhanced"] = False
            response["fallback_reason"] = "Quantum processing failed"
            return response
        else:
            return self._create_error_response(question, "Both quantum and base systems unavailable")
    
    def _create_security_error_response(self, question: str) -> Dict[str, Any]:
        """Create response for security validation failure."""
        
        return {
            "answer": "Access denied due to security policy restrictions.",
            "sources": [],
            "factuality_score": 0.0,
            "governance_compliant": False,
            "quantum_enhanced": False,
            "error": "security_validation_failed"
        }
    
    def _create_error_response(self, question: str, error: str) -> Dict[str, Any]:
        """Create error response."""
        
        return {
            "answer": f"Unable to process query due to error: {error}",
            "sources": [],
            "factuality_score": 0.0,
            "governance_compliant": True,
            "quantum_enhanced": False,
            "error": error
        }
    
    def _update_quantum_performance_metrics(self, execution_time: float, quantum_task: QuantumTask) -> None:
        """Update quantum performance metrics."""
        
        self.quantum_performance_metrics["queries_processed"] += 1
        
        # Calculate quantum speedup (simplified estimation)
        baseline_time = 1.0  # Assumed baseline without quantum enhancement
        speedup = baseline_time / execution_time if execution_time > 0 else 1.0
        
        # Running average
        current_speedup = self.quantum_performance_metrics["quantum_speedup"]
        queries_processed = self.quantum_performance_metrics["queries_processed"]
        
        self.quantum_performance_metrics["quantum_speedup"] = (
            (current_speedup * (queries_processed - 1) + speedup) / queries_processed
        )
        
        # Update cache hit rate from optimizer
        optimizer_report = self.optimizer.get_optimization_report()
        self.quantum_performance_metrics["cache_hit_rate"] = optimizer_report.get("overall_cache_hit_rate", 0.0)
        
        # Estimate parallel execution factor
        entanglement_count = len(quantum_task.entangled_tasks)
        parallel_factor = min(4.0, 1.0 + entanglement_count * 0.1)
        
        current_parallel = self.quantum_performance_metrics["parallel_execution_factor"]
        self.quantum_performance_metrics["parallel_execution_factor"] = (
            (current_parallel * (queries_processed - 1) + parallel_factor) / queries_processed
        )
    
    def get_quantum_system_status(self) -> Dict[str, Any]:
        """Get comprehensive quantum system status."""
        
        # Quantum planner status
        planner_summary = self.quantum_planner.get_quantum_state_summary()
        
        # Optimizer status
        optimizer_report = self.optimizer.get_optimization_report()
        
        # Security status
        security_report = self.security_manager.get_security_report()
        
        # Performance status
        performance_stats = self.quantum_performance_metrics.copy()
        
        return {
            "system_status": "operational",
            "configuration": {
                "optimization_strategy": self.config.optimization_strategy.value,
                "security_level": self.config.security_level.value,
                "parallel_execution": self.config.enable_parallel_execution,
                "auto_optimization": self.config.enable_auto_optimization
            },
            "quantum_planner": planner_summary,
            "optimization": optimizer_report,
            "security": security_report,
            "performance": performance_stats,
            "active_queries": len(self.active_quantum_queries),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def cleanup_completed_queries(self, max_age_hours: int = 24) -> int:
        """Clean up completed quantum queries."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        completed_queries = []
        for task_id, task in self.active_quantum_queries.items():
            if task.state == TaskState.COMPLETED and task.created_at < cutoff_time:
                completed_queries.append(task_id)
        
        # Remove from active queries
        for task_id in completed_queries:
            self.active_quantum_queries.pop(task_id, None)
            self.query_entanglements.pop(task_id, None)
            
            # Delete from planner
            self.quantum_planner.delete_task(task_id)
        
        return len(completed_queries)
    
    async def shutdown(self) -> None:
        """Shutdown quantum-enhanced RAG system."""
        
        self.logger_instance.info("Shutting down Quantum-Enhanced RAG system...")
        
        try:
            # Stop optimization
            self.optimizer.shutdown()
            
            # Clean up quantum components
            self.quantum_planner.shutdown()
            self.entanglement_graph.clear_all_entanglements()
            
            # Clear active queries
            self.active_quantum_queries.clear()
            self.query_entanglements.clear()
            
            self.logger_instance.info("Quantum-Enhanced RAG system shutdown complete")
            
        except Exception as e:
            self.logger_instance.error(f"Error during shutdown: {e}")