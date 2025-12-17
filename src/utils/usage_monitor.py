"""
Usage monitoring and reporting for Groq API.

Tracks daily usage, costs, and provides alerts when approaching limits.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UsageMonitor:
    """
    Monitor and track Groq API usage over time.
    
    Tracks requests, tokens, and provides usage reports and alerts.
    """
    
    def __init__(
        self,
        storage_path: str = ".cache/usage_stats.json",
        alert_threshold: float = 0.8  # Alert at 80% of daily limit
    ):
        """
        Initialize usage monitor.
        
        Args:
            storage_path: Path to store usage statistics
            alert_threshold: Fraction of limit to trigger alert (0.0-1.0)
        """
        self.storage_path = Path(storage_path)
        self.alert_threshold = alert_threshold
        
        # Ensure parent directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing stats or initialize
        self.stats = self._load_stats()
        
        logger.info(f"UsageMonitor initialized: storage={storage_path}")
    
    def _load_stats(self) -> Dict:
        """Load usage statistics from disk."""
        if not self.storage_path.exists():
            return {
                "daily_usage": {},
                "total_requests": 0,
                "total_tokens": 0,
                "total_cached_hits": 0,
                "first_request": None,
                "last_request": None,
            }
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading usage stats: {e}")
            return self._load_stats()  # Return empty stats
    
    def _save_stats(self) -> None:
        """Save usage statistics to disk."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving usage stats: {e}")
    
    def _get_today_key(self) -> str:
        """Get date key for today (YYYY-MM-DD)."""
        return datetime.now().strftime("%Y-%m-%d")
    
    def record_request(
        self,
        tokens_used: int = 0,
        cached: bool = False,
        response_time_ms: float = 0
    ) -> None:
        """
        Record an API request or cache hit.
        
        Args:
            tokens_used: Number of tokens used
            cached: Whether this was a cache hit
            response_time_ms: Response time in milliseconds
        """
        today = self._get_today_key()
        now = datetime.now().isoformat()
        
        # Initialize today's stats if needed
        if today not in self.stats["daily_usage"]:
            self.stats["daily_usage"][today] = {
                "requests": 0,
                "tokens": 0,
                "cached_hits": 0,
                "response_times": [],
            }
        
        # Update daily stats
        if cached:
            self.stats["daily_usage"][today]["cached_hits"] += 1
            self.stats["total_cached_hits"] += 1
        else:
            self.stats["daily_usage"][today]["requests"] += 1
            self.stats["daily_usage"][today]["tokens"] += tokens_used
            self.stats["total_requests"] += 1
            self.stats["total_tokens"] += tokens_used
        
        if response_time_ms > 0:
            self.stats["daily_usage"][today]["response_times"].append(response_time_ms)
        
        # Update timestamps
        if not self.stats["first_request"]:
            self.stats["first_request"] = now
        self.stats["last_request"] = now
        
        # Save to disk
        self._save_stats()
        
        # Check for alerts
        self._check_alerts(today)
    
    def _check_alerts(self, date_key: str) -> None:
        """Check if usage is approaching limits and log alerts."""
        daily_stats = self.stats["daily_usage"].get(date_key, {})
        requests_today = daily_stats.get("requests", 0)
        
        # Free tier limits
        RPD_LIMIT = 250
        
        usage_ratio = requests_today / RPD_LIMIT
        
        if usage_ratio >= 1.0:
            logger.error(
                f"⚠️ DAILY LIMIT REACHED: {requests_today}/{RPD_LIMIT} requests used today"
            )
        elif usage_ratio >= self.alert_threshold:
            logger.warning(
                f"⚠️ Approaching daily limit: {requests_today}/{RPD_LIMIT} requests "
                f"({usage_ratio*100:.1f}% used)"
            )
    
    def get_today_usage(self) -> Dict:
        """
        Get usage statistics for today.
        
        Returns:
            Dictionary with today's usage stats
        """
        today = self._get_today_key()
        daily_stats = self.stats["daily_usage"].get(today, {
            "requests": 0,
            "tokens": 0,
            "cached_hits": 0,
            "response_times": [],
        })
        
        # Calculate averages
        response_times = daily_stats.get("response_times", [])
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Free tier limits
        RPD_LIMIT = 250
        TPM_LIMIT = 250000
        
        requests = daily_stats.get("requests", 0)
        tokens = daily_stats.get("tokens", 0)
        cached = daily_stats.get("cached_hits", 0)
        
        return {
            "date": today,
            "requests": requests,
            "requests_limit": RPD_LIMIT,
            "requests_remaining": max(0, RPD_LIMIT - requests),
            "requests_percentage": requests / RPD_LIMIT * 100,
            "tokens": tokens,
            "tokens_limit": TPM_LIMIT,
            "cached_hits": cached,
            "total_queries": requests + cached,
            "cache_hit_rate": cached / (requests + cached) * 100 if (requests + cached) > 0 else 0,
            "avg_response_time_ms": avg_response_time,
        }
    
    def get_weekly_summary(self) -> Dict:
        """
        Get usage summary for the past 7 days.
        
        Returns:
            Dictionary with weekly usage stats
        """
        today = datetime.now()
        weekly_requests = 0
        weekly_tokens = 0
        weekly_cached = 0
        daily_breakdown = []
        
        for i in range(7):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_stats = self.stats["daily_usage"].get(date, {})
            
            requests = daily_stats.get("requests", 0)
            tokens = daily_stats.get("tokens", 0)
            cached = daily_stats.get("cached_hits", 0)
            
            weekly_requests += requests
            weekly_tokens += tokens
            weekly_cached += cached
            
            daily_breakdown.append({
                "date": date,
                "requests": requests,
                "tokens": tokens,
                "cached_hits": cached,
            })
        
        return {
            "period": "7 days",
            "total_requests": weekly_requests,
            "total_tokens": weekly_tokens,
            "total_cached_hits": weekly_cached,
            "avg_requests_per_day": weekly_requests / 7,
            "daily_breakdown": daily_breakdown,
        }
    
    def get_lifetime_stats(self) -> Dict:
        """
        Get lifetime usage statistics.
        
        Returns:
            Dictionary with all-time stats
        """
        return {
            "total_requests": self.stats.get("total_requests", 0),
            "total_tokens": self.stats.get("total_tokens", 0),
            "total_cached_hits": self.stats.get("total_cached_hits", 0),
            "first_request": self.stats.get("first_request"),
            "last_request": self.stats.get("last_request"),
            "days_active": len(self.stats.get("daily_usage", {})),
        }
    
    def print_usage_report(self) -> None:
        """Print a formatted usage report to console."""
        today_stats = self.get_today_usage()
        weekly_stats = self.get_weekly_summary()
        lifetime_stats = self.get_lifetime_stats()
        
        print("\n" + "="*60)
        print("📊 GROQ API USAGE REPORT")
        print("="*60)
        
        print(f"\n📅 TODAY ({today_stats['date']}):")
        print(f"  Requests: {today_stats['requests']}/{today_stats['requests_limit']} "
              f"({today_stats['requests_percentage']:.1f}%)")
        print(f"  Remaining: {today_stats['requests_remaining']} requests")
        print(f"  Tokens: {today_stats['tokens']:,}")
        print(f"  Cache Hits: {today_stats['cached_hits']} "
              f"({today_stats['cache_hit_rate']:.1f}% hit rate)")
        print(f"  Avg Response: {today_stats['avg_response_time_ms']:.0f}ms")
        
        print(f"\n📈 PAST 7 DAYS:")
        print(f"  Total Requests: {weekly_stats['total_requests']}")
        print(f"  Total Tokens: {weekly_stats['total_tokens']:,}")
        print(f"  Cache Hits: {weekly_stats['total_cached_hits']}")
        print(f"  Avg/Day: {weekly_stats['avg_requests_per_day']:.1f} requests")
        
        print(f"\n🌍 LIFETIME:")
        print(f"  Total Requests: {lifetime_stats['total_requests']}")
        print(f"  Total Tokens: {lifetime_stats['total_tokens']:,}")
        print(f"  Cache Hits: {lifetime_stats['total_cached_hits']}")
        print(f"  Days Active: {lifetime_stats['days_active']}")
        
        # Warnings
        if today_stats['requests_percentage'] >= 80:
            print(f"\n⚠️  WARNING: You've used {today_stats['requests_percentage']:.1f}% "
                  f"of your daily limit!")
        
        print("="*60 + "\n")
    
    def reset_today(self) -> None:
        """Reset today's usage (for testing)."""
        today = self._get_today_key()
        if today in self.stats["daily_usage"]:
            del self.stats["daily_usage"][today]
            self._save_stats()
            logger.info("Today's usage reset")
    
    def __repr__(self) -> str:
        """String representation of usage monitor."""
        today_stats = self.get_today_usage()
        return (
            f"UsageMonitor(today: {today_stats['requests']}/250 requests, "
            f"{today_stats['cached_hits']} cached)"
        )
