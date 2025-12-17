"""
Performance metrics collection module for Product Hunt RAG Analyzer.

This module provides the MetricsCollector class for tracking system metrics
including operation timing, memory usage, and performance statistics.
"""

import json
import time
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Collects and manages system performance metrics.
    
    Tracks operation timing, memory usage, and provides aggregated statistics
    for monitoring and optimization purposes.
    """
    
    def __init__(self):
        """Initialize MetricsCollector with empty metric storage."""
        self.operation_timings: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Dict[str, float]] = []
        self.start_time = time.time()
        self.metrics_count = 0
        
        logger.info("MetricsCollector initialized")
    
    def track_operation(self, operation_name: str, duration: float) -> None:
        """
        Record operation timing.
        
        Args:
            operation_name: Name of the operation being tracked
            duration: Duration in milliseconds
            
        Example:
            >>> collector = MetricsCollector()
            >>> collector.track_operation("embedding_generation", 125.5)
        """
        self.operation_timings[operation_name].append(duration)
        self.metrics_count += 1
        
        logger.debug(
            f"Tracked operation '{operation_name}': {duration:.2f}ms "
            f"(total samples: {len(self.operation_timings[operation_name])})"
        )
    
    def track_memory_usage(self) -> Dict[str, float]:
        """
        Record current memory usage using psutil.
        
        Returns:
            Dictionary with memory metrics in MB:
                - current_mb: Current process memory usage
                - available_mb: Available system memory
                - percent: Memory usage percentage
                - peak_mb: Peak memory usage (if available)
                
        Example:
            >>> collector = MetricsCollector()
            >>> memory = collector.track_memory_usage()
            >>> print(f"Current memory: {memory['current_mb']:.2f}MB")
        """
        try:
            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get system memory info
            virtual_memory = psutil.virtual_memory()
            
            # Calculate metrics in MB
            current_mb = memory_info.rss / (1024 * 1024)
            available_mb = virtual_memory.available / (1024 * 1024)
            percent = virtual_memory.percent
            
            # Try to get peak memory (platform-dependent)
            try:
                peak_mb = memory_info.peak_wset / (1024 * 1024) if hasattr(memory_info, 'peak_wset') else current_mb
            except AttributeError:
                peak_mb = current_mb
            
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "current_mb": current_mb,
                "available_mb": available_mb,
                "percent": percent,
                "peak_mb": peak_mb
            }
            
            self.memory_snapshots.append(snapshot)
            
            logger.debug(
                f"Memory snapshot: {current_mb:.2f}MB current, "
                f"{available_mb:.2f}MB available, {percent:.1f}% used"
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to track memory usage: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "current_mb": 0.0,
                "available_mb": 0.0,
                "percent": 0.0,
                "peak_mb": 0.0,
                "error": str(e)
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Return aggregated metrics (average, min, max, percentiles).
        
        Returns:
            Dictionary with aggregated statistics for all tracked operations:
                - operations: Dict of operation stats (avg, min, max, p50, p95, p99, count)
                - memory: Memory usage statistics
                - uptime_seconds: Time since collector initialization
                - total_metrics: Total number of metrics collected
                
        Example:
            >>> collector = MetricsCollector()
            >>> collector.track_operation("test", 100)
            >>> summary = collector.get_metrics_summary()
            >>> print(summary["operations"]["test"]["average"])
        """
        summary = {
            "operations": {},
            "memory": {},
            "uptime_seconds": time.time() - self.start_time,
            "total_metrics": self.metrics_count,
            "generated_at": datetime.now().isoformat()
        }
        
        # Aggregate operation timings
        for operation_name, timings in self.operation_timings.items():
            if not timings:
                continue
            
            timings_array = np.array(timings)
            
            summary["operations"][operation_name] = {
                "count": len(timings),
                "average": float(np.mean(timings_array)),
                "min": float(np.min(timings_array)),
                "max": float(np.max(timings_array)),
                "median": float(np.median(timings_array)),
                "p50": float(np.percentile(timings_array, 50)),
                "p95": float(np.percentile(timings_array, 95)),
                "p99": float(np.percentile(timings_array, 99)),
                "std_dev": float(np.std(timings_array)),
                "total_time": float(np.sum(timings_array))
            }
        
        # Aggregate memory snapshots
        if self.memory_snapshots:
            current_memory = [s["current_mb"] for s in self.memory_snapshots if "current_mb" in s]
            available_memory = [s["available_mb"] for s in self.memory_snapshots if "available_mb" in s]
            percent_memory = [s["percent"] for s in self.memory_snapshots if "percent" in s]
            peak_memory = [s["peak_mb"] for s in self.memory_snapshots if "peak_mb" in s]
            
            if current_memory:
                summary["memory"]["current"] = {
                    "average": float(np.mean(current_memory)),
                    "min": float(np.min(current_memory)),
                    "max": float(np.max(current_memory)),
                    "latest": current_memory[-1]
                }
            
            if available_memory:
                summary["memory"]["available"] = {
                    "average": float(np.mean(available_memory)),
                    "min": float(np.min(available_memory)),
                    "max": float(np.max(available_memory)),
                    "latest": available_memory[-1]
                }
            
            if percent_memory:
                summary["memory"]["percent"] = {
                    "average": float(np.mean(percent_memory)),
                    "min": float(np.min(percent_memory)),
                    "max": float(np.max(percent_memory)),
                    "latest": percent_memory[-1]
                }
            
            if peak_memory:
                summary["memory"]["peak"] = {
                    "max": float(np.max(peak_memory)),
                    "latest": peak_memory[-1]
                }
            
            summary["memory"]["snapshots_count"] = len(self.memory_snapshots)
        
        logger.debug(f"Generated metrics summary with {len(summary['operations'])} operations")
        
        return summary
    
    def export_metrics_json(self, path: Optional[str] = None) -> str:
        """
        Export metrics in JSON format.
        
        Args:
            path: Optional file path to save metrics. If None, returns JSON string.
            
        Returns:
            JSON string of metrics summary
            
        Raises:
            IOError: If file write fails
            
        Example:
            >>> collector = MetricsCollector()
            >>> json_str = collector.export_metrics_json()
            >>> collector.export_metrics_json("./logs/metrics.json")
        """
        try:
            summary = self.get_metrics_summary()
            json_str = json.dumps(summary, indent=2)
            
            if path:
                # Create directory if it doesn't exist
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w') as f:
                    f.write(json_str)
                
                logger.info(f"Metrics exported to {path}")
            
            return json_str
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}", exc_info=True)
            raise IOError(f"Failed to export metrics: {e}")
    
    def reset(self) -> None:
        """
        Reset all collected metrics.
        
        Example:
            >>> collector = MetricsCollector()
            >>> collector.track_operation("test", 100)
            >>> collector.reset()
            >>> collector.get_metrics_summary()["total_metrics"]
            0
        """
        self.operation_timings.clear()
        self.memory_snapshots.clear()
        self.start_time = time.time()
        self.metrics_count = 0
        
        logger.info("Metrics collector reset")
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with operation statistics or None if not found
            
        Example:
            >>> collector = MetricsCollector()
            >>> collector.track_operation("test", 100)
            >>> stats = collector.get_operation_stats("test")
            >>> stats["average"]
            100.0
        """
        if operation_name not in self.operation_timings:
            return None
        
        timings = self.operation_timings[operation_name]
        if not timings:
            return None
        
        timings_array = np.array(timings)
        
        return {
            "count": len(timings),
            "average": float(np.mean(timings_array)),
            "min": float(np.min(timings_array)),
            "max": float(np.max(timings_array)),
            "median": float(np.median(timings_array)),
            "p95": float(np.percentile(timings_array, 95)),
            "p99": float(np.percentile(timings_array, 99))
        }
    
    def __repr__(self) -> str:
        """String representation of MetricsCollector."""
        return (
            f"MetricsCollector(operations={len(self.operation_timings)}, "
            f"metrics={self.metrics_count}, "
            f"memory_snapshots={len(self.memory_snapshots)})"
        )


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get or create the global metrics collector instance.
    
    Returns:
        Global MetricsCollector instance
        
    Example:
        >>> collector = get_metrics_collector()
        >>> collector.track_operation("test", 100)
    """
    global _global_collector
    
    if _global_collector is None:
        _global_collector = MetricsCollector()
        logger.info("Created global metrics collector")
    
    return _global_collector


def reset_metrics_collector() -> None:
    """
    Reset the global metrics collector.
    
    Example:
        >>> reset_metrics_collector()
    """
    global _global_collector
    
    if _global_collector is not None:
        _global_collector.reset()
    else:
        _global_collector = MetricsCollector()


def log_performance_bottlenecks(threshold_ms: float = 1000.0) -> None:
    """
    Log performance bottlenecks and optimization suggestions.
    
    Analyzes collected metrics and logs warnings for operations that exceed
    the specified threshold, along with optimization suggestions.
    
    Args:
        threshold_ms: Threshold in milliseconds for considering an operation slow
        
    Example:
        >>> log_performance_bottlenecks(threshold_ms=500.0)
    """
    collector = get_metrics_collector()
    summary = collector.get_metrics_summary()
    
    operations = summary.get("operations", {})
    
    if not operations:
        logger.info("No operations tracked yet for bottleneck analysis")
        return
    
    logger.info("=" * 60)
    logger.info("Performance Bottleneck Analysis")
    logger.info("=" * 60)
    
    bottlenecks_found = False
    
    for operation_name, stats in operations.items():
        avg_time = stats.get("average", 0)
        max_time = stats.get("max", 0)
        p95_time = stats.get("p95", 0)
        
        if avg_time > threshold_ms or p95_time > threshold_ms:
            bottlenecks_found = True
            logger.warning(
                f"BOTTLENECK: {operation_name} - "
                f"avg={avg_time:.2f}ms, p95={p95_time:.2f}ms, max={max_time:.2f}ms"
            )
            
            # Provide optimization suggestions
            if "embedding" in operation_name.lower():
                logger.info(
                    f"  → Suggestion: Consider increasing batch size or using GPU acceleration"
                )
            elif "faiss" in operation_name.lower():
                logger.info(
                    f"  → Suggestion: Consider using IVF or HNSW index type for faster search"
                )
            elif "llm" in operation_name.lower():
                logger.info(
                    f"  → Suggestion: LLM generation is inherently slow. Consider using a smaller model or caching results"
                )
            elif "preprocessing" in operation_name.lower():
                logger.info(
                    f"  → Suggestion: Preprocessing is CPU-bound. Consider increasing batch size or parallel workers"
                )
            elif "index_build" in operation_name.lower():
                logger.info(
                    f"  → Suggestion: Index building is a one-time operation. Consider pre-building indices"
                )
    
    if not bottlenecks_found:
        logger.info(f"No bottlenecks detected (threshold: {threshold_ms}ms)")
    
    # Memory analysis
    memory = summary.get("memory", {})
    if memory:
        current_mem = memory.get("current", {})
        peak_mem = memory.get("peak", {})
        
        if current_mem:
            latest_mb = current_mem.get("latest", 0)
            max_mb = current_mem.get("max", 0)
            
            logger.info(f"Memory usage: current={latest_mb:.2f}MB, peak={max_mb:.2f}MB")
            
            if max_mb > 2048:  # 2GB threshold
                logger.warning(
                    f"High memory usage detected: {max_mb:.2f}MB"
                )
                logger.info(
                    "  → Suggestion: Consider reducing batch sizes or using memory-efficient index types"
                )
    
    logger.info("=" * 60)
