"""
Advanced concurrency management for high-performance RAG systems.
"""

import asyncio
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import time


class ConcurrencyManager:
    """Advanced concurrency manager with adaptive thread/process pools."""
    
    def __init__(self, max_threads: int = 10, max_processes: int = 4):
        self.logger = logging.getLogger(__name__)
        self.max_threads = max_threads
        self.max_processes = max_processes
        
        # Thread and process pools
        self.thread_executor = ThreadPoolExecutor(max_workers=max_threads)
        self.process_executor = ProcessPoolExecutor(max_workers=max_processes)
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = []
        self.task_counter = 0
        
        # Performance metrics
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0.0
        }
        
    async def submit_async_task(self, function: Callable, task_id: str = None, 
                                priority: int = 1, *args, **kwargs) -> Any:
        """Submit an async task for execution."""
        if task_id is None:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"
        
        # Create a simple async wrapper
        try:
            start_time = time.time()
            
            # Execute function in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_executor, 
                function, 
                *args
            )
            
            execution_time = time.time() - start_time
            
            # Update metrics
            self.metrics["tasks_completed"] += 1
            self._update_avg_execution_time(execution_time)
            
            return result
            
        except Exception as e:
            self.metrics["tasks_failed"] += 1
            self.logger.error(f"Task {task_id} failed: {e}")
            raise
    
    def submit_cpu_intensive_task(self, function: Callable, *args, **kwargs):
        """Submit CPU-intensive task to process pool."""
        return self.process_executor.submit(function, *args, **kwargs)
    
    def submit_io_task(self, function: Callable, *args, **kwargs):
        """Submit I/O task to thread pool."""
        return self.thread_executor.submit(function, *args, **kwargs)
    
    def _update_avg_execution_time(self, execution_time: float):
        """Update average execution time metric."""
        completed = self.metrics["tasks_completed"]
        if completed == 1:
            self.metrics["avg_execution_time"] = execution_time
        else:
            # Running average
            current_avg = self.metrics["avg_execution_time"]
            self.metrics["avg_execution_time"] = (
                (current_avg * (completed - 1) + execution_time) / completed
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get concurrency metrics."""
        return self.metrics.copy()
    
    def get_active_task_count(self) -> int:
        """Get number of active tasks."""
        return len(self.active_tasks)
    
    def shutdown(self, wait: bool = True):
        """Shutdown executors."""
        self.thread_executor.shutdown(wait=wait)
        self.process_executor.shutdown(wait=wait)