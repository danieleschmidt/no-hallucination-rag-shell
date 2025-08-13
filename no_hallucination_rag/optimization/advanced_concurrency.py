
"""
Advanced concurrent processing with connection pooling and resource management.
"""

import asyncio
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue, Empty
from collections import defaultdict
import multiprocessing
import weakref


@dataclass
class Task:
    """Task for concurrent execution."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    completed_at: datetime = field(default_factory=datetime.utcnow)


class ConnectionPool:
    """Generic connection pool for resource management."""
    
    def __init__(
        self,
        factory: Callable,
        min_connections: int = 5,
        max_connections: int = 20,
        max_idle_time: float = 300.0,  # 5 minutes
        validation_query: Optional[Callable] = None
    ):
        self.factory = factory
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.validation_query = validation_query
        
        self.pool: Queue = Queue(maxsize=max_connections)
        self.active_connections = set()
        self.connection_times = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Pre-populate pool with minimum connections
        self._initialize_pool()
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
    
    def get_connection(self, timeout: float = 10.0):
        """Get connection from pool."""
        with self.lock:
            try:
                # Try to get existing connection
                if not self.pool.empty():
                    conn = self.pool.get_nowait()
                    
                    # Validate connection
                    if self._validate_connection(conn):
                        self.active_connections.add(conn)
                        self.connection_times[conn] = time.time()
                        return conn
                    else:
                        # Connection invalid, discard it
                        self._close_connection(conn)
                
                # Create new connection if under limit
                if len(self.active_connections) < self.max_connections:
                    conn = self._create_connection()
                    if conn:
                        self.active_connections.add(conn)
                        self.connection_times[conn] = time.time()
                        return conn
                
                # Wait for connection to be returned
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        conn = self.pool.get(timeout=0.1)
                        if self._validate_connection(conn):
                            self.active_connections.add(conn)
                            self.connection_times[conn] = time.time()
                            return conn
                        else:
                            self._close_connection(conn)
                    except Empty:
                        continue
                
                raise Exception("Connection pool timeout")
                
            except Exception as e:
                self.logger.error(f"Failed to get connection: {e}")
                raise
    
    def return_connection(self, conn):
        """Return connection to pool."""
        with self.lock:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                self.connection_times.pop(conn, None)
                
                # Return to pool if valid and pool not full
                if self._validate_connection(conn) and not self.pool.full():
                    self.pool.put(conn)
                else:
                    self._close_connection(conn)
    
    def _create_connection(self):
        """Create new connection using factory."""
        try:
            conn = self.factory()
            self.logger.debug("Created new connection")
            return conn
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return None
    
    def _validate_connection(self, conn) -> bool:
        """Validate connection is still usable."""
        if not self.validation_query:
            return True
        
        try:
            return self.validation_query(conn)
        except Exception as e:
            self.logger.warning(f"Connection validation failed: {e}")
            return False
    
    def _close_connection(self, conn):
        """Close connection and clean up."""
        try:
            if hasattr(conn, 'close'):
                conn.close()
        except Exception as e:
            self.logger.warning(f"Error closing connection: {e}")
    
    def _initialize_pool(self):
        """Initialize pool with minimum connections."""
        for _ in range(self.min_connections):
            conn = self._create_connection()
            if conn:
                self.pool.put(conn)
    
    def _maintenance_loop(self):
        """Background maintenance for connection pool."""
        while True:
            try:
                with self.lock:
                    now = time.time()
                    
                    # Close idle connections
                    idle_connections = []
                    for conn in list(self.active_connections):
                        if now - self.connection_times.get(conn, now) > self.max_idle_time:
                            idle_connections.append(conn)
                    
                    for conn in idle_connections:
                        self.active_connections.remove(conn)
                        self.connection_times.pop(conn, None)
                        self._close_connection(conn)
                    
                    # Ensure minimum connections
                    current_total = len(self.active_connections) + self.pool.qsize()
                    if current_total < self.min_connections:
                        needed = self.min_connections - current_total
                        for _ in range(needed):
                            conn = self._create_connection()
                            if conn:
                                self.pool.put(conn)
                
                time.sleep(60)  # Run maintenance every minute
                
            except Exception as e:
                self.logger.error(f"Pool maintenance error: {e}")
                time.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self.lock:
            return {
                "active_connections": len(self.active_connections),
                "available_connections": self.pool.qsize(),
                "total_connections": len(self.active_connections) + self.pool.qsize(),
                "max_connections": self.max_connections,
                "min_connections": self.min_connections
            }


class AsyncTaskQueue:
    """Asynchronous task queue with priority and retry logic."""
    
    def __init__(self, max_workers: int = 10, max_queue_size: int = 1000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        self.result_callbacks: Dict[str, Callable] = {}
        self.workers: List[threading.Thread] = []
        self.running = False
        self.logger = logging.getLogger(__name__)
        self.stats = defaultdict(int)
    
    def start(self):
        """Start worker threads."""
        if self.running:
            return
        
        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} async workers")
    
    def stop(self):
        """Stop worker threads."""
        self.running = False
        
        # Add poison pills for workers
        for _ in range(self.max_workers):
            try:
                self.task_queue.put((999999, None), timeout=1)
            except:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.logger.info("Stopped async workers")
    
    def submit_task(
        self,
        task: Task,
        callback: Optional[Callable[[TaskResult], None]] = None
    ) -> bool:
        """Submit task for asynchronous execution."""
        try:
            # Priority queue uses negative priority for max-heap behavior
            priority_score = -task.priority
            self.task_queue.put((priority_score, task), timeout=1)
            
            if callback:
                self.result_callbacks[task.id] = callback
            
            self.stats['tasks_submitted'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit task {task.id}: {e}")
            self.stats['tasks_failed_submit'] += 1
            return False
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        self.logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue
                try:
                    priority, task = self.task_queue.get(timeout=1)
                    if task is None:  # Poison pill
                        break
                except Empty:
                    continue
                
                # Execute task
                result = self._execute_task(task)
                
                # Call result callback if registered
                if task.id in self.result_callbacks:
                    try:
                        self.result_callbacks[task.id](result)
                        del self.result_callbacks[task.id]
                    except Exception as e:
                        self.logger.error(f"Callback error for task {task.id}: {e}")
                
                self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Apply timeout if specified
            if task.timeout:
                # Simple timeout using threading
                result_container = [None]
                exception_container = [None]
                
                def target():
                    try:
                        result_container[0] = task.func(*task.args, **task.kwargs)
                    except Exception as e:
                        exception_container[0] = e
                
                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout=task.timeout)
                
                if thread.is_alive():
                    # Timeout occurred
                    execution_time = time.time() - start_time
                    self.stats['tasks_timeout'] += 1
                    
                    return TaskResult(
                        task_id=task.id,
                        success=False,
                        error=TimeoutError(f"Task timed out after {task.timeout}s"),
                        execution_time=execution_time
                    )
                
                if exception_container[0]:
                    raise exception_container[0]
                
                result = result_container[0]
            else:
                # Execute without timeout
                result = task.func(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            self.stats['tasks_completed'] += 1
            
            return TaskResult(
                task_id=task.id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats['tasks_failed'] += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
                
                # Re-queue task with lower priority
                retry_priority = -max(1, task.priority - task.retry_count)
                self.task_queue.put((retry_priority, task))
                self.stats['tasks_retried'] += 1
                
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error=e,
                    execution_time=execution_time
                )
            
            return TaskResult(
                task_id=task.id,
                success=False,
                error=e,
                execution_time=execution_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_workers": len([w for w in self.workers if w.is_alive()]),
            "max_workers": self.max_workers,
            "stats": dict(self.stats)
        }


class ProcessPoolManager:
    """Process pool manager for CPU-intensive tasks."""
    
    def __init__(self, max_processes: Optional[int] = None):
        self.max_processes = max_processes or multiprocessing.cpu_count()
        self.executor: Optional[ProcessPoolExecutor] = None
        self.logger = logging.getLogger(__name__)
        self.active_futures = weakref.WeakSet()
    
    def start(self):
        """Start process pool."""
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.max_processes)
            self.logger.info(f"Started process pool with {self.max_processes} processes")
    
    def stop(self):
        """Stop process pool."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            self.logger.info("Stopped process pool")
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit task to process pool."""
        if not self.executor:
            self.start()
        
        future = self.executor.submit(func, *args, **kwargs)
        self.active_futures.add(future)
        return future
    
    def map(self, func: Callable, iterable, timeout: Optional[float] = None):
        """Map function over iterable using process pool."""
        if not self.executor:
            self.start()
        
        return self.executor.map(func, iterable, timeout=timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get process pool statistics."""
        return {
            "max_processes": self.max_processes,
            "active": self.executor is not None,
            "active_futures": len(self.active_futures)
        }


class ConcurrencyManager:
    """Central manager for all concurrent processing."""
    
    def __init__(self):
        self.thread_pools: Dict[str, ThreadPoolExecutor] = {}
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.task_queues: Dict[str, AsyncTaskQueue] = {}
        self.process_pool = ProcessPoolManager()
        self.logger = logging.getLogger(__name__)
    
    def get_thread_pool(self, name: str, max_workers: int = 10) -> ThreadPoolExecutor:
        """Get or create named thread pool."""
        if name not in self.thread_pools:
            self.thread_pools[name] = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"pool-{name}"
            )
            self.logger.info(f"Created thread pool: {name} with {max_workers} workers")
        
        return self.thread_pools[name]
    
    def get_connection_pool(self, name: str, factory: Callable, **kwargs) -> ConnectionPool:
        """Get or create named connection pool."""
        if name not in self.connection_pools:
            self.connection_pools[name] = ConnectionPool(factory, **kwargs)
            self.logger.info(f"Created connection pool: {name}")
        
        return self.connection_pools[name]
    
    def get_task_queue(self, name: str, **kwargs) -> AsyncTaskQueue:
        """Get or create named task queue."""
        if name not in self.task_queues:
            queue = AsyncTaskQueue(**kwargs)
            queue.start()
            self.task_queues[name] = queue
            self.logger.info(f"Created task queue: {name}")
        
        return self.task_queues[name]
    
    def get_process_pool(self) -> ProcessPoolManager:
        """Get process pool manager."""
        return self.process_pool
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive concurrency statistics."""
        return {
            "thread_pools": {
                name: {
                    "max_workers": pool._max_workers,
                    "active_threads": len(pool._threads)
                }
                for name, pool in self.thread_pools.items()
            },
            "connection_pools": {
                name: pool.get_stats()
                for name, pool in self.connection_pools.items()
            },
            "task_queues": {
                name: queue.get_stats()
                for name, queue in self.task_queues.items()
            },
            "process_pool": self.process_pool.get_stats()
        }
    
    def shutdown(self):
        """Shutdown all concurrency resources."""
        # Shutdown task queues
        for queue in self.task_queues.values():
            queue.stop()
        
        # Shutdown thread pools
        for pool in self.thread_pools.values():
            pool.shutdown(wait=True)
        
        # Shutdown process pool
        self.process_pool.stop()
        
        self.logger.info("All concurrency resources shut down")


# Global concurrency manager
global_concurrency_manager = ConcurrencyManager()
