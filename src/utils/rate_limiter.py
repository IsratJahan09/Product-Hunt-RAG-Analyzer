"""
Rate limiter for Groq API to respect free tier limits.

Free Tier Limits:
- 30 requests per minute (RPM)
- 6000 requests per day (RPD)
- 6000 tokens per minute (TPM)
"""

import time
import threading
from collections import deque
from typing import Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class RateLimiter:
    """
    Thread-safe rate limiter for Groq API requests.
    
    Tracks requests per minute (RPM) and requests per day (RPD)
    to stay within free tier limits.
    """
    
    def __init__(
        self,
        rpm_limit: int = 10,
        rpd_limit: int = 250,
        tpm_limit: int = 250000
    ):
        """
        Initialize rate limiter with Groq free tier limits.
        
        Args:
            rpm_limit: Requests per minute limit (default: 30)
            rpd_limit: Requests per day limit (default: 6000)
            tpm_limit: Tokens per minute limit (default: 6000)
        """
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.tpm_limit = tpm_limit
        
        # Thread-safe request tracking
        self._lock = threading.Lock()
        self._minute_requests = deque()  # Timestamps of requests in last minute
        self._daily_requests = deque()   # Timestamps of requests in last day
        self._minute_tokens = deque()    # (timestamp, token_count) tuples
        
        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._blocked_requests = 0
        
        logger.info(
            f"RateLimiter initialized: RPM={rpm_limit}, RPD={rpd_limit}, TPM={tpm_limit}"
        )
    
    def _clean_old_entries(self):
        """Remove expired entries from tracking queues."""
        now = time.time()
        minute_ago = now - 60
        day_ago = now - 86400
        
        # Clean minute requests
        while self._minute_requests and self._minute_requests[0] < minute_ago:
            self._minute_requests.popleft()
        
        # Clean daily requests
        while self._daily_requests and self._daily_requests[0] < day_ago:
            self._daily_requests.popleft()
        
        # Clean minute tokens
        while self._minute_tokens and self._minute_tokens[0][0] < minute_ago:
            self._minute_tokens.popleft()
    
    def check_rate_limit(self, estimated_tokens: int = 0) -> bool:
        """
        Check if request can proceed without exceeding limits.
        
        Args:
            estimated_tokens: Estimated token count for this request
            
        Returns:
            True if request can proceed, False otherwise
        """
        with self._lock:
            self._clean_old_entries()
            
            # Check RPM limit
            if len(self._minute_requests) >= self.rpm_limit:
                return False
            
            # Check RPD limit
            if len(self._daily_requests) >= self.rpd_limit:
                return False
            
            # Check TPM limit if tokens provided
            if estimated_tokens > 0:
                current_tpm = sum(tokens for _, tokens in self._minute_tokens)
                if current_tpm + estimated_tokens > self.tpm_limit:
                    return False
            
            return True
    
    def wait_if_needed(self, estimated_tokens: int = 0, timeout: float = 60.0) -> None:
        """
        Wait until request can proceed within rate limits.
        
        Args:
            estimated_tokens: Estimated token count for this request
            timeout: Maximum time to wait in seconds
            
        Raises:
            RateLimitExceeded: If timeout is reached
        """
        start_time = time.time()
        
        while not self.check_rate_limit(estimated_tokens):
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self._blocked_requests += 1
                raise RateLimitExceeded(
                    f"Rate limit exceeded. RPM: {len(self._minute_requests)}/{self.rpm_limit}, "
                    f"RPD: {len(self._daily_requests)}/{self.rpd_limit}"
                )
            
            # Wait a bit before checking again
            time.sleep(0.5)
            
            # Log waiting status every 5 seconds
            if int(elapsed) % 5 == 0 and elapsed > 0:
                logger.info(
                    f"Waiting for rate limit... ({elapsed:.1f}s elapsed, "
                    f"RPM: {len(self._minute_requests)}/{self.rpm_limit})"
                )
    
    def record_request(self, token_count: int = 0) -> None:
        """
        Record a successful request.
        
        Args:
            token_count: Number of tokens used in this request
        """
        with self._lock:
            now = time.time()
            
            self._minute_requests.append(now)
            self._daily_requests.append(now)
            
            if token_count > 0:
                self._minute_tokens.append((now, token_count))
                self._total_tokens += token_count
            
            self._total_requests += 1
            
            logger.debug(
                f"Request recorded. RPM: {len(self._minute_requests)}/{self.rpm_limit}, "
                f"RPD: {len(self._daily_requests)}/{self.rpd_limit}"
            )
    
    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            self._clean_old_entries()
            
            current_tpm = sum(tokens for _, tokens in self._minute_tokens)
            
            return {
                "rpm_current": len(self._minute_requests),
                "rpm_limit": self.rpm_limit,
                "rpm_remaining": max(0, self.rpm_limit - len(self._minute_requests)),
                "rpd_current": len(self._daily_requests),
                "rpd_limit": self.rpd_limit,
                "rpd_remaining": max(0, self.rpd_limit - len(self._daily_requests)),
                "tpm_current": current_tpm,
                "tpm_limit": self.tpm_limit,
                "tpm_remaining": max(0, self.tpm_limit - current_tpm),
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "blocked_requests": self._blocked_requests,
                "rpm_utilization": len(self._minute_requests) / self.rpm_limit * 100,
                "rpd_utilization": len(self._daily_requests) / self.rpd_limit * 100,
            }
    
    def reset_daily_counter(self) -> None:
        """Reset daily request counter (for testing or manual reset)."""
        with self._lock:
            self._daily_requests.clear()
            logger.info("Daily request counter reset")
    
    def __repr__(self) -> str:
        """String representation of rate limiter."""
        stats = self.get_stats()
        return (
            f"RateLimiter(RPM: {stats['rpm_current']}/{self.rpm_limit}, "
            f"RPD: {stats['rpd_current']}/{self.rpd_limit})"
        )
