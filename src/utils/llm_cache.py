"""
Caching system for LLM responses to reduce API calls.

Implements disk-based caching with TTL and size limits.
"""

import json
import hashlib
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMCache:
    """
    Disk-based cache for LLM responses with TTL and size management.
    
    Caches responses based on hash of (product_idea + competitors_context)
    to avoid redundant API calls for identical queries.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/llm",
        ttl_hours: int = 24,
        max_cache_size_mb: int = 100
    ):
        """
        Initialize LLM cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries in hours
            max_cache_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_hours * 3600
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        logger.info(
            f"LLMCache initialized: dir={cache_dir}, ttl={ttl_hours}h, "
            f"max_size={max_cache_size_mb}MB"
        )
    
    def _get_cache_key(self, product_idea: str, competitors_context: str) -> str:
        """
        Generate cache key from inputs.
        
        Args:
            product_idea: User's product idea
            competitors_context: Competitor data context
            
        Returns:
            SHA256 hash as cache key
        """
        combined = f"{product_idea}||{competitors_context}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(
        self,
        product_idea: str,
        competitors_context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if available and not expired.
        
        Args:
            product_idea: User's product idea
            competitors_context: Competitor data context
            
        Returns:
            Cached response dict or None if not found/expired
        """
        cache_key = self._get_cache_key(product_idea, competitors_context)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            self._misses += 1
            logger.debug(f"Cache miss: {cache_key[:16]}...")
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if expired
            cached_time = cached_data.get('timestamp', 0)
            age_seconds = time.time() - cached_time
            
            if age_seconds > self.ttl_seconds:
                logger.debug(
                    f"Cache expired: {cache_key[:16]}... "
                    f"(age: {age_seconds/3600:.1f}h)"
                )
                cache_path.unlink()  # Delete expired cache
                self._misses += 1
                return None
            
            self._hits += 1
            logger.info(
                f"Cache hit: {cache_key[:16]}... "
                f"(age: {age_seconds/3600:.1f}h, saved API call)"
            )
            
            return cached_data.get('response')
            
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            self._misses += 1
            return None
    
    def set(
        self,
        product_idea: str,
        competitors_context: str,
        response: Dict[str, Any]
    ) -> None:
        """
        Store response in cache.
        
        Args:
            product_idea: User's product idea
            competitors_context: Competitor data context
            response: LLM response to cache
        """
        cache_key = self._get_cache_key(product_idea, competitors_context)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cached_data = {
                'timestamp': time.time(),
                'product_idea': product_idea[:200],  # Store truncated for reference
                'response': response
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, indent=2)
            
            logger.info(f"Response cached: {cache_key[:16]}...")
            
            # Check cache size and clean if needed
            self._cleanup_if_needed()
            
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def _get_cache_size(self) -> int:
        """Get total size of cache directory in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.json"):
            total_size += cache_file.stat().st_size
        return total_size
    
    def _cleanup_if_needed(self) -> None:
        """Remove oldest cache entries if size limit exceeded."""
        current_size = self._get_cache_size()
        
        if current_size <= self.max_cache_size_bytes:
            return
        
        logger.info(
            f"Cache size ({current_size/1024/1024:.1f}MB) exceeds limit "
            f"({self.max_cache_size_bytes/1024/1024:.1f}MB), cleaning up..."
        )
        
        # Get all cache files sorted by modification time (oldest first)
        cache_files = sorted(
            self.cache_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime
        )
        
        # Remove oldest files until under limit
        for cache_file in cache_files:
            if current_size <= self.max_cache_size_bytes * 0.8:  # 80% of limit
                break
            
            file_size = cache_file.stat().st_size
            cache_file.unlink()
            current_size -= file_size
            logger.debug(f"Removed old cache: {cache_file.name}")
        
        logger.info(f"Cache cleanup complete. New size: {current_size/1024/1024:.1f}MB")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Cache cleared: {count} entries removed")
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        cache_size = self._get_cache_size()
        
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_entries": len(cache_files),
            "cache_size_mb": cache_size / 1024 / 1024,
            "cache_limit_mb": self.max_cache_size_bytes / 1024 / 1024,
            "ttl_hours": self.ttl_seconds / 3600,
        }
    
    def __repr__(self) -> str:
        """String representation of cache."""
        stats = self.get_stats()
        return (
            f"LLMCache(entries={stats['cache_entries']}, "
            f"hit_rate={stats['hit_rate']:.1f}%, "
            f"size={stats['cache_size_mb']:.1f}MB)"
        )
