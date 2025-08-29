"""
Detailed Logger for TE-AI System
Creates comprehensive logs with function tracing, timing, and detailed state tracking
"""

import logging
import logging.handlers
import os
import sys
import time
import traceback
import functools
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import json
import torch
import numpy as np
import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

class DetailedLogger:
    """
    Comprehensive logging system with:
    - Function entry/exit tracking
    - Execution timing
    - Parameter logging
    - Return value logging
    - Exception tracking
    - Memory usage monitoring
    - GPU utilization tracking
    """
    
    def __init__(self, run_name: Optional[str] = None, log_dir: str = "logs", 
                 max_bytes: int = 15 * 1024 * 1024, backup_count: int = 100,
                 async_enabled: bool = True):
        self.start_time = time.time()
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"te_ai_run_{self.run_name}.log"
        
        unique_logger_name = f"TE_AI_{self.run_name}"
        self.logger = logging.getLogger(unique_logger_name)


        # Respect env var; default to WARNING to keep console quiet unless explicitly enabled
        env_level = os.getenv('TEAI_LOG_LEVEL', 'WARNING').upper()
        log_level = getattr(logging, env_level, logging.WARNING)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        self.logger.handlers.clear()
        
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file, 
            mode='a',
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        console_handler = logging.StreamHandler(sys.stdout)
        # Allow silencing console independently
        console_env_level = os.getenv('TEAI_CONSOLE_LOG_LEVEL', env_level).upper()
        console_level = getattr(logging, console_env_level, log_level)
        console_handler.setLevel(console_level)
        
        file_formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(funcName)-30s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.call_stack = []
        self.function_timings = {}
        self.function_calls = {}
        self.silent = False
        
        # --- ASYNC LOGGING SETUP ---
        self.async_enabled = async_enabled
        if self.async_enabled:
            self.log_queue = queue.Queue(maxsize=10000)
            self.log_thread = threading.Thread(target=self._async_log_worker, daemon=True)
            self.log_thread.start()
            self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TEAILogger")
        
        self.logger.info("="*80)
        self.logger.info(f"TE-AI RUN INITIALIZED: {self.run_name}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Max file size: {max_bytes / 1024 / 1024:.1f} MB")
        self.logger.info(f"Backup files: {backup_count}")
        self.logger.info(f"Async logging: {'ENABLED' if self.async_enabled else 'DISABLED'}")
        self.logger.info("="*80)
        
        self._log_system_info()
    
    def _async_log_worker(self):
        """Background thread worker for async logging"""
        while True:
            try:
                # Block until item available or timeout
                level, msg = self.log_queue.get(timeout=1.0)
                if level == 'STOP':
                    break
                
                # Log the message synchronously in this thread
                if level == 'INFO':
                    self.logger.info(msg)
                elif level == 'WARNING':
                    self.logger.warning(msg)
                elif level == 'ERROR':
                    self.logger.error(msg)
                elif level == 'DEBUG':
                    self.logger.debug(msg)
                    
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Emergency fallback - log directly
                self.logger.error(f"Async logging error: {e}")
    
    def _log_system_info(self):
        self.logger.info("SYSTEM INFORMATION:")
        self.logger.info(f"  Python version: {sys.version}")
        self.logger.info(f"  PyTorch version: {torch.__version__}")
        cuda_available = False
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        self.logger.info(f"  CUDA available: {cuda_available}")
        try:
            device_count = torch.cuda.device_count() if cuda_available else 0
        except Exception:
            device_count = 0
        self.logger.info(f"  CUDA devices: {device_count}")
        if cuda_available and device_count > 0:
            try:
                for i in range(device_count):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    self.logger.info(f"  CUDA device {i}: {name}")
                    self.logger.info(f"    memory: {props.total_memory / 1e9:.2f} GB")
            except Exception as e:
                self.logger.warning(f"  CUDA device query failed: {e}")
        self.logger.info("-"*80)
    
    def _format_value(self, value: Any, max_length: int = 100) -> str:
        try:
            if isinstance(value, torch.Tensor):
                return f"Tensor(shape={list(value.shape)}, dtype={value.dtype}, device={value.device})"
            elif isinstance(value, np.ndarray):
                return f"Array(shape={value.shape}, dtype={value.dtype})"
            elif isinstance(value, (list, tuple)) and len(value) > 5:
                return f"{type(value).__name__}(len={len(value)}, first={self._format_value(value[0] if value else None)})"
            elif isinstance(value, dict) and len(value) > 5:
                return f"Dict(keys={len(value)}, sample_keys={list(value.keys())[:3]})"
            elif isinstance(value, str) and len(value) > max_length:
                return f"{value[:max_length]}..."
            elif hasattr(value, '__class__'):
                return f"<{value.__class__.__module__}.{value.__class__.__name__} object>"
            else:
                return str(value)
        except Exception as e:
            return f"<{type(value).__name__} object (formatting error)>"
    
    def trace(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            call_id = f"{func_name}_{time.time()}"
            
            self.call_stack.append(func_name)
            indent = "  " * (len(self.call_stack) - 1)
            
            args_str = ", ".join([self._format_value(arg) for arg in args[:5]])
            kwargs_str = ", ".join([f"{k}={self._format_value(v)}" for k, v in list(kwargs.items())[:5]])
            
            self.logger.debug(f"{indent}-> ENTER {func_name}")
            if args_str:
                self.logger.debug(f"{indent}  args: {args_str}")
            if kwargs_str:
                self.logger.debug(f"{indent}  kwargs: {kwargs_str}")
            
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            try:
                result = func(*args, **kwargs)
                
                elapsed = time.time() - start_time
                memory_delta = (torch.cuda.memory_allocated() - start_memory) if torch.cuda.is_available() else 0
                
                self.logger.debug(f"{indent}<- EXIT {func_name} [OK] ({elapsed:.3f}s)")
                if memory_delta != 0:
                    self.logger.debug(f"{indent}  Memory Δ: {memory_delta/1e6:.1f} MB")
                
                if result is not None:
                    self.logger.debug(f"{indent}  returns: {self._format_value(result)}")
                
                if func_name not in self.function_timings:
                    self.function_timings[func_name] = []
                    self.function_calls[func_name] = 0
                self.function_timings[func_name].append(elapsed)
                self.function_calls[func_name] += 1
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"{indent}X EXCEPTION in {func_name} after {elapsed:.3f}s")
                self.logger.error(f"{indent}  {type(e).__name__}: {str(e)}")
                self.logger.debug(f"{indent}  Traceback:\n{traceback.format_exc()}")
                raise
                
            finally:
                self.call_stack.pop()
        
        return wrapper
    
    def log_generation(self, generation: int, metrics: Dict[str, Any]):
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"GENERATION {generation}")
        self.logger.info(f"{'='*60}")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
            else:
                self.logger.info(f"  {key}: {self._format_value(value)}")
    
    def log_phase_transition(self, from_phase: str, to_phase: str, indicators: Dict[str, float]):
        self.logger.warning(f"\n{'!'*60}")
        self.logger.warning(f"PHASE TRANSITION: {from_phase} -> {to_phase}")
        self.logger.warning(f"{'!'*60}")
        
        for key, value in indicators.items():
            self.logger.warning(f"  {key}: {value:.4f}")
    
    def log_intervention(self, intervention_type: str, details: Dict[str, Any]):
        self.logger.warning(f"\n{'*'*60}")
        self.logger.warning(f"INTERVENTION: {intervention_type}")
        self.logger.warning(f"{'*'*60}")
        
        for key, value in details.items():
            self.logger.warning(f"  {key}: {self._format_value(value)}")
    
    def log_checkpoint(self, checkpoint_path: str, generation: int, metrics: Dict[str, Any]):
        self.logger.info(f"\n💾 CHECKPOINT SAVED: {checkpoint_path}")
        self.logger.info(f"  Generation: {generation}")
        self.logger.info(f"  Best fitness: {metrics.get('best_fitness', 'N/A')}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        summary = {
            'total_runtime': time.time() - self.start_time,
            'function_statistics': {}
        }
        
        for func_name, timings in self.function_timings.items():
            summary['function_statistics'][func_name] = {
                'calls': self.function_calls[func_name],
                'total_time': sum(timings),
                'avg_time': np.mean(timings),
                'min_time': min(timings),
                'max_time': max(timings)
            }
        
        return summary
    
    def debug(self, message: str, async_log: bool = None):
        """Log debug message, optionally async"""
        if async_log is None:
            async_log = self.async_enabled
            
        if async_log and hasattr(self, 'log_queue'):
            try:
                self.log_queue.put_nowait(('DEBUG', message))
            except queue.Full:
                # Fallback to sync logging if queue is full
                self.logger.debug(message)
        else:
            self.logger.debug(message)
    
    def info(self, message: str, async_log: bool = None):
        """Log info message, optionally async"""
        if self.silent:
            return
        if async_log is None:
            async_log = self.async_enabled
            
        if async_log and hasattr(self, 'log_queue'):
            try:
                self.log_queue.put_nowait(('INFO', message))
            except queue.Full:
                # Fallback to sync logging if queue is full
                self.logger.info(message)
        else:
            self.logger.info(message)

    def set_silent(self, silent: bool = True):
        """Silence info-level messages (useful for fast demo mode)."""
        self.silent = bool(silent)
    
    def warning(self, message: str, async_log: bool = None):
        """Log warning message, optionally async"""
        if async_log is None:
            async_log = self.async_enabled
            
        if async_log and hasattr(self, 'log_queue'):
            try:
                self.log_queue.put_nowait(('WARNING', message))
            except queue.Full:
                # Fallback to sync logging if queue is full
                self.logger.warning(message)
        else:
            self.logger.warning(message)
    
    def error(self, message: str, async_log: bool = None):
        """Log error message, optionally async"""
        if async_log is None:
            async_log = self.async_enabled
            
        if async_log and hasattr(self, 'log_queue'):
            try:
                self.log_queue.put_nowait(('ERROR', message))
            except queue.Full:
                # Fallback to sync logging if queue is full
                self.logger.error(message)
        else:
            self.logger.error(message)
    
    def critical(self, message: str):
        """Critical messages are always logged synchronously"""
        self.logger.critical(message)
    
    def finalize(self):
        """Finalize logging and shutdown async workers if enabled"""
        # Flush any pending async logs
        if self.async_enabled and hasattr(self, 'log_queue'):
            # Wait for queue to empty
            self.log_queue.join()
            # Send stop signal
            self.log_queue.put(('STOP', None))
            # Wait for thread to finish
            if hasattr(self, 'log_thread'):
                self.log_thread.join(timeout=5.0)
            # Shutdown executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("RUN COMPLETED")
        self.logger.info("="*80)
        
        summary = self.get_performance_summary()
        self.logger.info(f"Total runtime: {summary['total_runtime']:.2f}s")
        
        self.logger.info("\nTOP 10 TIME-CONSUMING FUNCTIONS:")
        sorted_funcs = sorted(
            summary['function_statistics'].items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]
        
        for func_name, stats in sorted_funcs:
            self.logger.info(
                f"  {func_name}: {stats['calls']} calls, "
                f"{stats['total_time']:.3f}s total, "
                f"{stats['avg_time']:.3f}s avg"
            )
        
        summary_file = self.log_file.with_suffix('.summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\nSummary saved to: {summary_file}")
        self.logger.info("="*80)



# Global logger instance

def get_logger(run_name: Optional[str] = None, log_dir: str = "logs",
               max_bytes: int = 50 * 1024 * 1024,
               backup_count: int = 10) -> DetailedLogger:
    """Get or create the global logger instance
    
    Args:
        run_name: Name for this run (defaults to timestamp)
        log_dir: Directory to store log files (default "logs")
        max_bytes: Maximum size per log file in bytes (default 50MB)
        backup_count: Number of backup files to keep (default 10)
    """
    return DetailedLogger(run_name=run_name, log_dir=log_dir, 
                                         max_bytes=max_bytes, backup_count=backup_count)


def trace(func: Callable) -> Callable:
    """Convenience decorator for function tracing"""
    logger = get_logger()
    return logger.trace(func)


logger = get_logger()