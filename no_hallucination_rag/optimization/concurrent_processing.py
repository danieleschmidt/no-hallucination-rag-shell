"""
Concurrent processing and async operations for improved performance.
"""

import logging
import asyncio
import concurrent.futures
import threading
import time
from typing import List, Dict, Any, Optional, Callable, Tuple, Awaitable
from dataclasses import dataclass
import queue
from functools import wraps


@dataclass
class ProcessingTask:
    """Task for concurrent processing."""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass 
class TaskResult:
    """Result of processed task."""
    task_id: str
    result: Any = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    success: bool = True


class AsyncRAGProcessor:
    """Asynchronous processor for RAG operations."""
    
    def __init__(
        self,
        max_concurrent_retrievals: int = 5,
        max_concurrent_verifications: int = 3,
        max_concurrent_generations: int = 2
    ):
        self.max_concurrent_retrievals = max_concurrent_retrievals
        self.max_concurrent_verifications = max_concurrent_verifications
        self.max_concurrent_generations = max_concurrent_generations
        
        # Semaphores for controlling concurrency
        self.retrieval_semaphore = asyncio.Semaphore(max_concurrent_retrievals)
        self.verification_semaphore = asyncio.Semaphore(max_concurrent_verifications)
        self.generation_semaphore = asyncio.Semaphore(max_concurrent_generations)
        
        self.logger = logging.getLogger(__name__)
    
    async def process_multiple_queries(
        self,
        queries: List[str],
        rag_system: Any,
        batch_size: int = 5
    ) -> List[TaskResult]:
        """Process multiple queries concurrently."""
        results = []
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            # Create tasks for batch
            tasks = [
                self._process_single_query(query, rag_system, f"query_{i + j}")
                for j, query in enumerate(batch_queries)
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                task_id = f"query_{i + j}"
                if isinstance(result, Exception):
                    results.append(TaskResult(
                        task_id=task_id,
                        error=result,
                        success=False
                    ))
                else:
                    results.append(TaskResult(
                        task_id=task_id,
                        result=result,
                        success=True
                    ))
        
        return results
    
    async def _process_single_query(
        self,
        query: str,
        rag_system: Any,
        task_id: str
    ) -> Any:
        """Process single query with controlled concurrency."""
        start_time = time.time()
        
        try:
            # Use async version if available
            if hasattr(rag_system, 'aquery'):
                result = await rag_system.aquery(query)
            else:
                # Fallback to sync version in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, rag_system.query, query
                )
            
            processing_time = time.time() - start_time
            self.logger.debug(f"Query {task_id} processed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query {task_id}: {e}")
            raise
    
    async def parallel_source_retrieval(
        self,
        query: str,
        retrievers: List[Callable],
        merge_strategy: str = "union"
    ) -> List[Dict[str, Any]]:
        """Retrieve sources from multiple retrievers in parallel."""
        async with self.retrieval_semaphore:
            # Create tasks for each retriever
            tasks = [
                self._safe_retrieve(retriever, query, f"retriever_{i}")
                for i, retriever in enumerate(retrievers)
            ]
            
            # Execute retrievals concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge results
            all_sources = []
            for result in results:
                if isinstance(result, list):
                    all_sources.extend(result)
                elif isinstance(result, Exception):
                    self.logger.warning(f"Retriever failed: {result}")
            
            # Apply merge strategy
            if merge_strategy == "union":
                return self._merge_sources_union(all_sources)
            elif merge_strategy == "intersection":
                return self._merge_sources_intersection(all_sources)
            else:
                return all_sources
    
    async def _safe_retrieve(
        self,
        retriever: Callable,
        query: str,
        retriever_id: str
    ) -> List[Dict[str, Any]]:
        """Safely execute retrieval with error handling."""
        try:
            if asyncio.iscoroutinefunction(retriever):
                return await retriever(query)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, retriever, query)
        except Exception as e:
            self.logger.error(f"Retrieval failed for {retriever_id}: {e}")
            return []
    
    async def parallel_factuality_verification(
        self,
        claims: List[str],
        sources: List[Dict[str, Any]],
        verifiers: List[Callable]
    ) -> List[float]:
        """Verify multiple claims in parallel."""
        async with self.verification_semaphore:
            verification_tasks = []
            
            for claim in claims:
                for verifier in verifiers:
                    task = self._safe_verify_claim(verifier, claim, sources)
                    verification_tasks.append(task)
            
            # Execute verifications concurrently
            results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            # Process results and calculate scores
            scores = []
            for i in range(0, len(results), len(verifiers)):
                verifier_scores = results[i:i + len(verifiers)]
                valid_scores = [
                    score for score in verifier_scores 
                    if isinstance(score, (int, float))
                ]
                
                if valid_scores:
                    # Average scores from multiple verifiers
                    avg_score = sum(valid_scores) / len(valid_scores)
                    scores.append(avg_score)
                else:
                    scores.append(0.0)
            
            return scores
    
    async def _safe_verify_claim(
        self,
        verifier: Callable,
        claim: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """Safely verify claim with error handling."""
        try:
            if asyncio.iscoroutinefunction(verifier):
                return await verifier(claim, sources)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, verifier, claim, sources)
        except Exception as e:
            self.logger.error(f"Claim verification failed: {e}")
            return 0.0
    
    def _merge_sources_union(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge sources using union strategy (remove duplicates)."""
        seen_urls = set()
        merged_sources = []
        
        for source in sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged_sources.append(source)
            elif not url:  # Keep sources without URLs
                merged_sources.append(source)
        
        return merged_sources
    
    def _merge_sources_intersection(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge sources using intersection strategy (keep common ones)."""
        # Group sources by URL
        url_groups = {}
        for source in sources:
            url = source.get('url', '')
            if url:
                if url not in url_groups:
                    url_groups[url] = []
                url_groups[url].append(source)
        
        # Keep sources that appear in multiple retrievers
        common_sources = []
        for url, source_list in url_groups.items():
            if len(source_list) > 1:  # Found in multiple retrievers
                # Take the source with highest score
                best_source = max(
                    source_list, 
                    key=lambda s: s.get('relevance_score', 0)
                )
                common_sources.append(best_source)
        
        return common_sources


class ThreadPoolManager:
    """Manages thread pools for different types of operations."""
    
    def __init__(
        self,
        io_pool_size: int = 10,
        cpu_pool_size: int = 4,
        gpu_pool_size: int = 2
    ):
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=io_pool_size,
            thread_name_prefix="RAG-IO"
        )
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=cpu_pool_size,
            thread_name_prefix="RAG-CPU"
        )
        self.gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=gpu_pool_size,
            thread_name_prefix="RAG-GPU"
        )
        
        self.logger = logging.getLogger(__name__)
    
    def submit_io_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit I/O bound task."""
        return self.io_executor.submit(func, *args, **kwargs)
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit CPU bound task."""
        return self.cpu_executor.submit(func, *args, **kwargs)
    
    def submit_gpu_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit GPU bound task."""
        return self.gpu_executor.submit(func, *args, **kwargs)
    
    def process_batch_io(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """Process batch of I/O tasks."""
        futures = []
        for i, (func, args, kwargs) in enumerate(tasks):
            future = self.submit_io_task(func, *args, **kwargs)
            futures.append((f"task_{i}", future))
        
        results = []
        for task_id, future in futures:
            try:
                start_time = time.time()
                result = future.result(timeout=timeout)
                processing_time = time.time() - start_time
                
                results.append(TaskResult(
                    task_id=task_id,
                    result=result,
                    processing_time=processing_time,
                    success=True
                ))
            except Exception as e:
                results.append(TaskResult(
                    task_id=task_id,
                    error=e,
                    success=False
                ))
        
        return results
    
    def shutdown(self) -> None:
        """Shutdown all thread pools."""
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)
        self.gpu_executor.shutdown(wait=True)


class TaskQueue:
    """Priority task queue for managing concurrent operations."""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.results: Dict[str, TaskResult] = {}
        self.workers: List[threading.Thread] = []
        self.running = False
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def start_workers(self, num_workers: int = 4) -> None:
        """Start worker threads."""
        self.running = True
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {num_workers} task workers")
    
    def submit_task(
        self,
        task_id: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: int = 0
    ) -> None:
        """Submit task to queue."""
        task = ProcessingTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority
        )
        
        try:
            # Priority queue uses tuple for ordering (priority, task)
            self.queue.put((-priority, task), timeout=1.0)
        except queue.Full:
            self.logger.warning(f"Task queue full, dropping task {task_id}")
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result for task."""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            with self.lock:
                if task_id in self.results:
                    return self.results.pop(task_id)
            
            time.sleep(0.1)
        
        return None
    
    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self.running:
            try:
                # Get task from queue
                priority, task = self.queue.get(timeout=1.0)
                
                # Execute task
                start_time = time.time()
                try:
                    result = task.function(*task.args, **task.kwargs)
                    processing_time = time.time() - start_time
                    
                    task_result = TaskResult(
                        task_id=task.task_id,
                        result=result,
                        processing_time=processing_time,
                        success=True
                    )
                except Exception as e:
                    processing_time = time.time() - start_time
                    task_result = TaskResult(
                        task_id=task.task_id,
                        error=e,
                        processing_time=processing_time,
                        success=False
                    )
                
                # Store result
                with self.lock:
                    self.results[task.task_id] = task_result
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def stop_workers(self) -> None:
        """Stop all worker threads."""
        self.running = False
        
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        self.logger.info("Stopped all task workers")


def async_cached(cache_manager, cache_name: str, ttl: Optional[float] = None):
    """Decorator for async method caching."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash((args, tuple(kwargs.items())))}"
            
            # Try to get from cache
            cache = cache_manager.get_cache(cache_name)
            if cache:
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if cache:
                cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def rate_limited_async(calls_per_second: float):
    """Rate limiting decorator for async functions."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator