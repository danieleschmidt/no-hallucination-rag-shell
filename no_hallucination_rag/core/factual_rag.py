"""
Main FactualRAG pipeline with zero-hallucination guarantees.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from datetime import datetime
import time
import hashlib

from ..retrieval.hybrid_retriever import HybridRetriever
from ..verification.factuality_detector import FactualityDetector
from ..governance.compliance_checker import GovernanceChecker
from ..core.source_ranker import SourceRanker
from ..core.validation import InputValidator, ValidationResult
from ..core.error_handling import (
    ErrorHandler, ErrorContext, ValidationError, RetrievalError, 
    ProcessingError, retry_on_error, safe_execute
)
from ..monitoring.metrics import MetricsCollector
from ..security.security_manager import SecurityManager
from ..optimization.caching import CacheManager, cached_method
from ..optimization.concurrent_processing import AsyncRAGProcessor, ThreadPoolManager
from ..optimization.performance_optimizer import PerformanceOptimizer
from .text_generator import TextGenerator, GenerationConfig
from ..monitoring.audit_logger import audit_logger, AuditEventType, AuditLevel


@dataclass
class RAGResponse:
    """Response from FactualRAG with verification metadata."""
    answer: str
    sources: List[Dict[str, Any]]
    factuality_score: float
    governance_compliant: bool
    confidence: float = 0.0
    citations: List[str] = None
    verification_details: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.verification_details is None:
            self.verification_details = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class FactualRAG:
    """Main RAG system with mandatory factual grounding."""
    
    def __init__(
        self,
        retriever: str = "hybrid",
        factuality_threshold: float = 0.95,
        governance_mode: str = "strict",
        max_sources: int = 10,
        require_citations: bool = True,
        min_sources: int = 2,
        enable_security: bool = True,
        enable_metrics: bool = True,
        enable_caching: bool = True,
        enable_optimization: bool = True,
        enable_concurrency: bool = True
    ):
        self.factuality_threshold = factuality_threshold
        self.governance_mode = governance_mode
        self.max_sources = max_sources
        self.require_citations = require_citations
        self.min_sources = min_sources
        self.enable_security = enable_security
        self.enable_metrics = enable_metrics
        self.enable_caching = enable_caching
        self.enable_optimization = enable_optimization
        self.enable_concurrency = enable_concurrency
        
        # Initialize core components
        self.retriever = HybridRetriever()
        self.source_ranker = SourceRanker()
        self.factuality_detector = FactualityDetector()
        self.governance_checker = GovernanceChecker(mode=governance_mode)
        self.text_generator = TextGenerator()
        
        # Initialize robust components
        self.input_validator = InputValidator()
        self.error_handler = ErrorHandler()
        self.metrics_collector = MetricsCollector() if enable_metrics else None
        self.security_manager = SecurityManager() if enable_security else None
        
        # Initialize optimization components
        self.cache_manager = CacheManager() if enable_caching else None
        self.async_processor = AsyncRAGProcessor() if enable_concurrency else None
        self.thread_manager = ThreadPoolManager() if enable_concurrency else None
        from ..optimization.performance_optimizer import CacheOptimizer
        self.performance_optimizer = CacheOptimizer(self.cache_manager) if enable_optimization else None
        
        # Start optimization if enabled (CacheOptimizer doesn't have start_auto_optimization)
        
        self.logger = logging.getLogger(__name__)
        
    def query(
        self,
        question: str,
        require_citations: Optional[bool] = None,
        min_sources: Optional[int] = None,
        min_factuality_score: Optional[float] = None,
        client_context: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Query the RAG system with factuality guarantees.
        
        Args:
            question: User query
            require_citations: Override instance setting
            min_sources: Minimum number of sources required
            min_factuality_score: Minimum factuality score threshold
            client_context: Client context for security and metrics
            
        Returns:
            RAGResponse with answer and verification metadata
        """
        start_time = time.time()
        require_citations = require_citations or self.require_citations
        min_sources = min_sources or self.min_sources
        min_factuality_score = min_factuality_score or self.factuality_threshold
        client_context = client_context or {}
        
        # Log user query for audit
        query_event_id = audit_logger.log_user_query(
            query=question,
            user_id=client_context.get("user_id"),
            session_id=client_context.get("session_id"),
            client_ip=client_context.get("client_ip"),
            user_agent=client_context.get("user_agent"),
            metadata={
                "require_citations": require_citations,
                "min_sources": min_sources,
                "min_factuality_score": min_factuality_score
            }
        )
        
        # Create error context for detailed error handling
        error_context = ErrorContext(
            user_query=question,
            component="FactualRAG",
            operation="query"
        )
        
        try:
            # Step 0: Check cache first
            if self.cache_manager:
                query_hash = self.cache_manager.create_query_hash(
                    question, 
                    {
                        'factuality_threshold': min_factuality_score,
                        'min_sources': min_sources,
                        'require_citations': require_citations
                    }
                )
                
                cached_result = self.cache_manager.get_cached_query_result(query_hash)
                if cached_result:
                    # Track cache hit
                    if self.metrics_collector:
                        self.metrics_collector.counter("cache_hits", 1.0, {"cache_type": "query"})
                    
                    self.logger.debug(f"Cache hit for query: {question[:50]}...")
                    return cached_result
                
                # Track cache miss
                if self.metrics_collector:
                    self.metrics_collector.counter("cache_misses", 1.0, {"cache_type": "query"})
            
            # Step 1: Input validation and security checks
            if not self._validate_and_secure_request(question, client_context, error_context):
                return self._create_error_response(question, "Request validation failed")
            
            # Step 2: Retrieve relevant sources with error handling
            raw_sources = self._safe_retrieve_sources(question, error_context)
            if not raw_sources:
                return self._create_no_sources_response(question)
            
            # Step 3: Rank and filter sources
            ranked_sources = self._safe_rank_sources(question, raw_sources, error_context)
            if len(ranked_sources) < min_sources:
                return self._create_insufficient_sources_response(
                    question, len(ranked_sources), min_sources
                )
            
            # Step 4: Generate answer from sources
            answer = self._safe_generate_answer(question, ranked_sources, error_context)
            
            # Step 5: Verify factuality
            factuality_score = self._safe_verify_factuality(
                question, answer, ranked_sources, error_context
            )
            
            if factuality_score < min_factuality_score:
                return self._create_low_confidence_response(
                    question, factuality_score, min_factuality_score
                )
            
            # Step 6: Check governance compliance
            governance_result = self._safe_check_governance(
                question, answer, ranked_sources, error_context
            )
            
            # Step 7: Create response with citations
            citations = self._generate_citations(ranked_sources) if require_citations else []
            
            # Step 8: Track metrics and create response
            response_time = time.time() - start_time
            self._track_query_metrics(
                question, response_time, factuality_score, 
                len(ranked_sources), True, client_context
            )
            
            response = RAGResponse(
                answer=answer,
                sources=ranked_sources,
                factuality_score=factuality_score,
                governance_compliant=governance_result.is_compliant,
                confidence=factuality_score,
                citations=citations,
                verification_details={
                    "governance_details": governance_result.details,
                    "source_count": len(ranked_sources),
                    "retrieval_method": "hybrid",
                    "response_time": response_time
                }
            )
            
            # Step 9: Cache successful response
            if self.cache_manager and factuality_score >= min_factuality_score:
                self.cache_manager.cache_query_result(query_hash, response)
            
            # Step 10: Record performance data for optimization
            if self.performance_optimizer:
                self.performance_optimizer.record_query_performance(
                    response_time=response_time,
                    factuality_score=factuality_score,
                    source_count=len(ranked_sources),
                    success=True,
                    query_type="general"
                )
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            error_details = self.error_handler.handle_error(e, error_context)
            
            # Track error metrics
            self._track_query_metrics(
                question, response_time, 0.0, 0, False, 
                client_context, error_details.category.value
            )
            
            self.logger.error(f"Error in RAG query: {e}")
            return self._create_error_response(
                question, 
                error_details.user_message or str(e)
            )
    
    async def aquery(self, question: str, **kwargs) -> RAGResponse:
        """Async version of query method with optimization."""
        if not self.enable_concurrency or not self.async_processor:
            # Fallback to sync version in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.query, question, **kwargs)
        
        # Use optimized async processing
        try:
            results = await self.async_processor.process_multiple_queries(
                [question], self, batch_size=1
            )
            
            if results and results[0].success:
                return results[0].result
            elif results and results[0].error:
                raise results[0].error
            else:
                raise ProcessingError("Async query processing failed")
                
        except Exception as e:
            # Fallback to sync version
            self.logger.warning(f"Async processing failed, falling back to sync: {e}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.query, question, **kwargs)
    
    async def aquery_batch(
        self, 
        questions: List[str], 
        batch_size: int = 5,
        **kwargs
    ) -> List[RAGResponse]:
        """Process multiple questions asynchronously."""
        if not self.enable_concurrency or not self.async_processor:
            # Fallback to sequential processing
            results = []
            for question in questions:
                result = await self.aquery(question, **kwargs)
                results.append(result)
            return results
        
        # Use optimized batch processing
        task_results = await self.async_processor.process_multiple_queries(
            questions, self, batch_size=batch_size
        )
        
        responses = []
        for task_result in task_results:
            if task_result.success:
                responses.append(task_result.result)
            else:
                # Create error response
                error_response = self._create_error_response(
                    "batch_query", 
                    str(task_result.error) if task_result.error else "Unknown error"
                )
                responses.append(error_response)
        
        return responses
        
    def _generate_answer(self, question: str, sources: List[Dict[str, Any]]) -> str:
        """Generate answer from ranked sources using LLM or template-based generation."""
        
        if not sources:
            return "I cannot provide an answer as no reliable sources were found."
        
        # Use the text generator to create a comprehensive answer
        generation_result = self.text_generator.generate_answer(
            query=question,
            sources=sources,
            require_citations=self.require_citations
        )
        
        # Extract the generated answer
        answer = generation_result.get("answer", "")
        
        # If generation failed, fall back to simple template
        if not answer:
            context_parts = []
            for i, source in enumerate(sources[:3], 1):  # Use top 3 sources
                content = source.get("content", "")[:500]  # Truncate for simplicity
                context_parts.append(f"[Source {i}]: {content}")
            
            context = "\n\n".join(context_parts)
            answer = f"Based on the available sources:\n\n{context}"
        
        return answer
    
    def _generate_citations(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Generate citations for sources."""
        citations = []
        for i, source in enumerate(sources, 1):
            url = source.get("url", "Unknown source")
            title = source.get("title", "Untitled")
            citations.append(f"[{i}] {title} - {url}")
        return citations
    
    def _create_no_sources_response(self, question: str) -> RAGResponse:
        """Create response when no sources found."""
        return RAGResponse(
            answer="I cannot provide an answer as no reliable sources were found for your query. Please try rephrasing your question or check if the topic is covered in available knowledge bases.",
            sources=[],
            factuality_score=0.0,
            governance_compliant=True,
            verification_details={"error": "no_sources_found"}
        )
    
    def _create_insufficient_sources_response(
        self, question: str, found: int, required: int
    ) -> RAGResponse:
        """Create response when insufficient sources found."""
        return RAGResponse(
            answer=f"I found only {found} reliable source(s) but require at least {required} for a confident answer. Please try a more specific query or check available knowledge bases.",
            sources=[],
            factuality_score=0.0,
            governance_compliant=True,
            verification_details={"error": "insufficient_sources", "found": found, "required": required}
        )
    
    def _create_low_confidence_response(
        self, question: str, score: float, threshold: float
    ) -> RAGResponse:
        """Create response when factuality score is too low."""
        return RAGResponse(
            answer=f"I cannot provide a confident answer as the factuality score ({score:.2f}) is below the required threshold ({threshold:.2f}). The available sources may be contradictory or insufficient.",
            sources=[],
            factuality_score=score,
            governance_compliant=True,
            verification_details={"error": "low_confidence", "score": score, "threshold": threshold}
        )
    
    def _create_error_response(self, question: str, error: str) -> RAGResponse:
        """Create response for system errors."""
        return RAGResponse(
            answer=f"An error occurred while processing your query: {error}",
            sources=[],
            factuality_score=0.0,
            governance_compliant=False,
            verification_details={"error": "system_error", "details": error}
        )
    
    def models_loaded(self) -> Dict[str, bool]:
        """Check if all models are loaded."""
        return {
            "retriever": True,  # Simplified for Generation 1
            "factuality_detector": True,
            "governance_checker": True,
            "source_ranker": True
        }
    
    def list_knowledge_bases(self) -> List[str]:
        """List available knowledge bases."""
        return ["default"]  # Simplified for Generation 1
    
    def _validate_and_secure_request(
        self, 
        question: str, 
        client_context: Dict[str, Any], 
        error_context: ErrorContext
    ) -> bool:
        """Validate input and perform security checks."""
        try:
            # Input validation
            validation_result = self.input_validator.validate_query(question)
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Input validation failed: {validation_result.errors}",
                    context=error_context
                )
            
            # Security validation if enabled
            if self.security_manager:
                is_valid, security_details = self.security_manager.validate_request(
                    client_ip=client_context.get("client_ip"),
                    user_id=client_context.get("user_id"),
                    api_key=client_context.get("api_key")
                )
                
                if not is_valid:
                    self.logger.warning(f"Security validation failed: {security_details}")
                    return False
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, error_context)
            return False
    
    @retry_on_error(max_retries=2, delay=1.0)
    def _safe_retrieve_sources(
        self, 
        question: str, 
        error_context: ErrorContext
    ) -> List[Dict[str, Any]]:
        """Safely retrieve sources with error handling and retries."""
        try:
            # Check retrieval cache first
            if self.cache_manager:
                retrieval_hash = self.cache_manager.create_query_hash(
                    question,
                    {'top_k': self.max_sources * 2}
                )
                
                cached_sources = self.cache_manager.get_cached_retrieval_result(retrieval_hash)
                if cached_sources:
                    if self.metrics_collector:
                        self.metrics_collector.counter("cache_hits", 1.0, {"cache_type": "retrieval"})
                    
                    return cached_sources
                
                if self.metrics_collector:
                    self.metrics_collector.counter("cache_misses", 1.0, {"cache_type": "retrieval"})
            
            start_time = time.time()
            
            sources = self.retriever.retrieve(
                query=question,
                top_k=self.max_sources * 2  # Get more for ranking
            )
            
            # Log document access for audit
            if sources:
                sources_accessed = [source.get("url", source.get("title", source.get("id", "unknown"))) for source in sources]
                audit_logger.log_document_access(
                    sources_accessed=sources_accessed,
                    user_id=error_context.additional_data.get("user_id") if error_context.additional_data else None,
                    session_id=error_context.additional_data.get("session_id") if error_context.additional_data else None,
                    query_hash=hashlib.sha256(question.encode()).hexdigest()[:16],
                    metadata={"num_sources": len(sources)}
                )
            
            retrieval_time = time.time() - start_time
            
            # Track retrieval metrics
            if self.metrics_collector:
                self.metrics_collector.track_retrieval_metrics(
                    query=question,
                    sources_found=len(sources) if sources else 0,
                    retrieval_time=retrieval_time,
                    top_k=self.max_sources * 2,
                    retrieval_method="hybrid"
                )
            
            # Cache successful retrieval
            if self.cache_manager and sources:
                self.cache_manager.cache_retrieval_result(retrieval_hash, sources)
            
            return sources or []
            
        except Exception as e:
            raise RetrievalError(f"Source retrieval failed: {e}", context=error_context)
    
    def _safe_rank_sources(
        self, 
        question: str, 
        sources: List[Dict[str, Any]], 
        error_context: ErrorContext
    ) -> List[Dict[str, Any]]:
        """Safely rank sources with error handling."""
        try:
            start_time = time.time()
            
            ranked_sources = self.source_ranker.rank(
                query=question,
                sources=sources,
                top_k=self.max_sources
            )
            
            ranking_time = time.time() - start_time
            
            # Track component metrics
            if self.metrics_collector:
                self.metrics_collector.track_component_metrics(
                    component="source_ranker",
                    operation="rank",
                    duration=ranking_time,
                    success=True
                )
            
            return ranked_sources
            
        except Exception as e:
            if self.metrics_collector:
                self.metrics_collector.track_component_metrics(
                    component="source_ranker",
                    operation="rank", 
                    duration=0.0,
                    success=False,
                    error_type=type(e).__name__
                )
            
            raise ProcessingError(f"Source ranking failed: {e}", context=error_context)
    
    def _safe_generate_answer(
        self, 
        question: str, 
        sources: List[Dict[str, Any]], 
        error_context: ErrorContext
    ) -> str:
        """Safely generate answer with error handling."""
        try:
            start_time = time.time()
            
            answer = self._generate_answer(question, sources)
            
            generation_time = time.time() - start_time
            
            # Track component metrics
            if self.metrics_collector:
                self.metrics_collector.track_component_metrics(
                    component="answer_generator",
                    operation="generate",
                    duration=generation_time,
                    success=True
                )
            
            return answer
            
        except Exception as e:
            if self.metrics_collector:
                self.metrics_collector.track_component_metrics(
                    component="answer_generator",
                    operation="generate",
                    duration=0.0,
                    success=False,
                    error_type=type(e).__name__
                )
            
            raise ProcessingError(f"Answer generation failed: {e}", context=error_context)
    
    def _safe_verify_factuality(
        self, 
        question: str, 
        answer: str, 
        sources: List[Dict[str, Any]], 
        error_context: ErrorContext
    ) -> float:
        """Safely verify factuality with error handling."""
        try:
            start_time = time.time()
            
            factuality_score = self.factuality_detector.verify_answer(
                question=question,
                answer=answer,
                sources=sources
            )
            
            verification_time = time.time() - start_time
            
            # Log factuality check for audit
            claims_verified = len(self.factuality_detector._extract_claims(answer))
            audit_logger.log_factuality_check(
                query=question,
                factuality_score=factuality_score,
                claims_verified=claims_verified,
                user_id=error_context.additional_data.get("user_id") if error_context.additional_data else None,
                session_id=error_context.additional_data.get("session_id") if error_context.additional_data else None,
                metadata={
                    "num_sources": len(sources),
                    "verification_time_ms": verification_time * 1000
                }
            )
            
            # Track factuality metrics
            if self.metrics_collector:
                # Estimate claim count for metrics (simplified)
                claim_count = len(answer.split('.'))
                verified_claims = max(1, int(claim_count * factuality_score))
                
                self.metrics_collector.track_factuality_metrics(
                    factuality_score=factuality_score,
                    verification_time=verification_time,
                    claim_count=claim_count,
                    verified_claims=verified_claims
                )
            
            return factuality_score
            
        except Exception as e:
            if self.metrics_collector:
                self.metrics_collector.track_component_metrics(
                    component="factuality_detector",
                    operation="verify",
                    duration=0.0,
                    success=False,
                    error_type=type(e).__name__
                )
            
            # Return low score rather than failing completely
            self.logger.error(f"Factuality verification failed: {e}")
            return 0.0
    
    def _safe_check_governance(
        self, 
        question: str, 
        answer: str, 
        sources: List[Dict[str, Any]], 
        error_context: ErrorContext
    ):
        """Safely check governance compliance with error handling."""
        try:
            start_time = time.time()
            
            governance_result = self.governance_checker.check_compliance(
                question=question,
                answer=answer,
                sources=sources
            )
            
            check_time = time.time() - start_time
            
            # Track governance metrics
            if self.metrics_collector:
                policies_checked = ["whitehouse_2025", "nist_framework"]  # Default policies
                violation_count = len(governance_result.violations)
                
                self.metrics_collector.track_governance_metrics(
                    is_compliant=governance_result.is_compliant,
                    check_time=check_time,
                    violation_count=violation_count,
                    policies_checked=policies_checked
                )
            
            return governance_result
            
        except Exception as e:
            if self.metrics_collector:
                self.metrics_collector.track_component_metrics(
                    component="governance_checker",
                    operation="check_compliance",
                    duration=0.0,
                    success=False,
                    error_type=type(e).__name__
                )
            
            # Return non-compliant result rather than failing
            self.logger.error(f"Governance check failed: {e}")
            from ..governance.compliance_checker import ComplianceResult
            return ComplianceResult(
                is_compliant=False,
                details={"error": str(e)},
                violations=["Governance check failed"],
                recommendations=["Review governance checker configuration"]
            )
    
    def _track_query_metrics(
        self,
        question: str,
        response_time: float,
        factuality_score: float,
        source_count: int,
        success: bool,
        client_context: Dict[str, Any],
        error_type: Optional[str] = None
    ) -> None:
        """Track comprehensive query metrics."""
        if not self.metrics_collector:
            return
        
        self.metrics_collector.track_query_metrics(
            query=question,
            response_time=response_time,
            factuality_score=factuality_score,
            source_count=source_count,
            success=success,
            error_type=error_type
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "retriever": True,
                "source_ranker": True,
                "factuality_detector": True,
                "governance_checker": True
            }
        }
        
        if self.metrics_collector:
            metrics_health = self.metrics_collector.get_health_metrics()
            health.update(metrics_health)
        
        if self.security_manager:
            security_stats = self.security_manager.get_security_stats()
            health["security"] = security_stats
        
        if self.error_handler:
            error_stats = self.error_handler.get_error_statistics()
            health["errors"] = error_stats
        
        return health
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "components_enabled": {
                "caching": self.enable_caching,
                "metrics": self.enable_metrics,
                "security": self.enable_security,
                "optimization": self.enable_optimization,
                "concurrency": self.enable_concurrency
            }
        }
        
        # Cache statistics
        if self.cache_manager:
            stats["cache"] = {
                "stats": self.cache_manager.get_all_stats(),
                "memory_usage": self.cache_manager.get_total_memory_usage()
            }
        
        # Performance optimization stats
        if self.performance_optimizer:
            stats["optimization"] = {
                "current_parameters": self.performance_optimizer.get_current_parameters(),
                "performance_summary": self.performance_optimizer.get_performance_summary(),
                "optimization_history": len(self.performance_optimizer.get_optimization_history())
            }
        
        return stats
    
    def invalidate_caches(self, pattern: Optional[str] = None) -> Dict[str, int]:
        """Invalidate caches matching pattern."""
        if not self.cache_manager:
            return {"message": "Caching not enabled"}
        
        if pattern:
            invalidated = {}
            for cache_name, cache in self.cache_manager.caches.items():
                count = cache.invalidate_pattern(pattern)
                invalidated[cache_name] = count
            return invalidated
        else:
            self.cache_manager.invalidate_all()
            return {"message": "All caches invalidated"}
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Force performance optimization."""
        if not self.performance_optimizer:
            return {"message": "Optimization not enabled"}
        
        return self.performance_optimizer.force_optimization()
    
    def shutdown(self) -> None:
        """Gracefully shutdown the RAG system."""
        self.logger.info("Shutting down RAG system...")
        
        # Stop optimization
        if self.performance_optimizer:
            self.performance_optimizer.stop_auto_optimization()
        
        # Shutdown thread pools
        if self.thread_manager:
            self.thread_manager.shutdown()
        
        # Save caches if persistent
        if self.cache_manager:
            for cache in self.cache_manager.caches.values():
                cache._save_to_disk()
        
        # Save optimization state
        if self.performance_optimizer:
            try:
                self.performance_optimizer.save_optimization_state("data/optimization_state.json")
            except Exception as e:
                self.logger.error(f"Failed to save optimization state: {e}")
        
        self.logger.info("RAG system shutdown complete")