"""
LLM generation module for Product Hunt RAG Analyzer.

This module provides integration with Groq API 
for generating competitive intelligence insights from retrieved competitor data.
"""

import json
import os
import time
from typing import Dict, Any, Optional, Iterator
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_collector
from src.utils.rate_limiter import RateLimiter, RateLimitExceeded
from src.utils.llm_cache import LLMCache
from src.utils.usage_monitor import UsageMonitor

logger = get_logger(__name__)
metrics_collector = get_metrics_collector()


class LLMConnectionError(Exception):
    """Exception raised when connection to LLM provider fails."""
    pass


class LLMGenerationError(Exception):
    """Exception raised when LLM generation fails."""
    pass


class LLMResponseValidationError(Exception):
    """Exception raised when LLM response validation fails."""
    pass


class LLMGenerator:
    """
    Manages LLM API interactions for competitive intelligence generation.
    
    Uses Groq API and provides methods to connect,
    generate insights from competitor data, and validate responses with proper 
    error handling.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 3000,
        timeout: int = 120,
        enable_cache: bool = True,
        enable_rate_limiting: bool = True,
        enable_usage_monitoring: bool = True
    ):
        """
        Initialize LLMGenerator with Groq configuration.
        
        Args:
            config: LLM configuration dictionary with api_key and model
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            max_tokens: Maximum tokens to generate (default: 3000)
            timeout: Request timeout in seconds (default: 120)
            enable_cache: Enable response caching (default: True)
            enable_rate_limiting: Enable rate limiting (default: True)
            enable_usage_monitoring: Enable usage tracking (default: True)
        
        Example:
            >>> generator = LLMGenerator(
            ...     config={"api_key": "key", "model": "llama-3.3-70b-versatile"}
            ... )
        """
        self.config = config or {}
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize rate limiting, caching, and monitoring
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.cache = LLMCache() if enable_cache else None
        self.usage_monitor = UsageMonitor() if enable_usage_monitoring else None
        
        # Initialize Groq provider
        self.provider = "groq"
        self._init_groq()
        
        logger.info(
            f"LLMGenerator initialized with Groq API, "
            f"model={self.model_name}, temperature={temperature}, "
            f"top_p={top_p}, max_tokens={max_tokens}, "
            f"cache={'enabled' if enable_cache else 'disabled'}, "
            f"rate_limiting={'enabled' if enable_rate_limiting else 'disabled'}"
        )
    
    def _init_groq(self):
        """Initialize Groq API configuration."""
        # Get API key from config or environment variable
        api_key = self.config.get("api_key", "").strip()
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY", "").strip()
        
        if not api_key:
            raise ValueError(
                "Groq API key not found. Set it in config or GROQ_API_KEY environment variable"
            )
        
        self.api_key = api_key
        self.model_name = self.config.get("model", "llama-3.3-70b-versatile")
        
        # Initialize Groq SDK
        try:
            from groq import Groq
            
            self.groq_client = Groq(
                api_key=self.api_key,
                timeout=self.timeout
            )
            
            # System message for Groq
            self.system_message = {
                "role": "system",
                "content": (
                    "You are a professional business analyst providing competitive intelligence "
                    "for product development. Your analysis is factual, objective, and based on "
                    "market data. You provide structured business insights in JSON format."
                )
            }
            
            logger.info(f"Groq API initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "groq package not installed. Install it with: pip install groq"
            )
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize Groq SDK: {e}")

    def connect(self) -> bool:
        """
        Verify connection to Groq API service.
        
        Returns:
            True if connection successful
            
        Raises:
            LLMConnectionError: If connection fails
            
        Example:
            >>> generator = LLMGenerator()
            >>> generator.connect()
            True
        """
        try:
            logger.info("Verifying Groq API connection")
            
            # Verify we have a valid client
            if not hasattr(self, 'groq_client'):
                raise LLMConnectionError("Groq SDK not properly initialized")
            
            # Test connection with a simple request
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            if response and response.choices:
                logger.info(f"Groq API connection verified for model: {self.model_name}")
                return True
            
            raise LLMConnectionError("Groq API returned empty response")
            
        except Exception as e:
            error_msg = f"Failed to verify Groq connection: {e}"
            logger.error(error_msg)
            raise LLMConnectionError(error_msg)

    def build_analysis_prompt(
        self,
        product_idea: str,
        competitors_context: str
    ) -> str:
        """
        Build structured prompt using the exact template from design document.
        
        Args:
            product_idea: User's product idea description
            competitors_context: Formatted competitor data and reviews
            
        Returns:
            Complete prompt string for LLM
            
        Example:
            >>> generator = LLMGenerator()
            >>> prompt = generator.build_analysis_prompt(
            ...     "A task management app",
            ...     "Competitor: Todoist\\nReviews: ..."
            ... )
        """
        prompt = f"""You are a product analyst specializing in competitive intelligence for Product Hunt launches.

The user has the following product idea:
"{product_idea}"

Based on competitor reviews and launch data from Product Hunt, provide comprehensive competitive intelligence.

Competitor Data:
{competitors_context}

Analyze the data and provide a structured response with:

1. Market Positioning Insights
   - How are competitors currently positioned?
   - What positioning strategies are they using?
   - Where are the differentiation opportunities for this product idea?
   - Is the market saturated or are there gaps?

2. Feature Gap Analysis
   - What features do competitors have that users love?
   - What features are users requesting but competitors lack?
   - What are the must-have features based on competitor analysis?
   - What unique features could differentiate this product idea?

3. Sentiment Analysis Summary
   - What do users love most about competitor products?
   - What are the common pain points and complaints?
   - What aspects have the most negative sentiment?
   - What opportunities exist to address unmet needs?

4. Actionable Recommendations
   - Provide 5-7 specific, prioritized recommendations for the product idea
   - Each recommendation should include: description, priority (high/medium/low), and rationale
   - Focus on actionable insights that can guide product development

5. Confidence Assessment
   - Rate your confidence in this analysis (0.0-1.0)
   - Explain any data gaps or limitations

Format your response as valid JSON matching this EXACT structure:
{{
  "market_positioning": {{
    "summary": "A comprehensive overview of the market positioning landscape",
    "differentiation_opportunities": ["opportunity 1", "opportunity 2", "..."],
    "competitor_positions": [
      {{"name": "Competitor Name", "position": "Their positioning strategy"}},
      "..."
    ]
  }},
  "feature_gaps": {{
    "high_priority": [
      {{"name": "Feature name", "description": "Why it's important"}},
      "..."
    ],
    "medium_priority": ["feature 1", "feature 2", "..."],
    "low_priority": ["feature 1", "feature 2", "..."]
  }},
  "sentiment_summary": {{
    "overall_trends": {{"positive_aspects": "...", "negative_aspects": "..."}},
    "pain_points": ["pain point 1", "pain point 2", "..."],
    "loved_features": ["loved feature 1", "loved feature 2", "..."]
  }},
  "recommendations": [
    {{
      "title": "Recommendation title",
      "description": "Detailed description",
      "priority": "high|medium|low",
      "rationale": "Why this recommendation matters"
    }},
    "..."
  ],
  "confidence_score": 0.85
}}"""
        
        return prompt

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Check response is valid JSON with all required sections.
        
        Args:
            response: Parsed JSON response from LLM
            
        Returns:
            True if response is valid
            
        Raises:
            LLMResponseValidationError: If validation fails
            
        Example:
            >>> generator = LLMGenerator()
            >>> response = {"market_positioning": {}, ...}
            >>> generator.validate_response(response)
            True
        """
        required_sections = [
            "market_positioning",
            "feature_gaps",
            "sentiment_summary",
            "recommendations",
            "confidence_score"
        ]
        
        missing_sections = [
            section for section in required_sections
            if section not in response
        ]
        
        if missing_sections:
            error_msg = f"Response missing required sections: {', '.join(missing_sections)}"
            logger.error(error_msg)
            raise LLMResponseValidationError(error_msg)
        
        # Validate confidence_score is a float between 0 and 1
        confidence = response.get("confidence_score")
        if not isinstance(confidence, (int, float)):
            error_msg = f"confidence_score must be a number, got {type(confidence)}"
            logger.error(error_msg)
            raise LLMResponseValidationError(error_msg)
        
        if not (0.0 <= confidence <= 1.0):
            logger.warning(
                f"confidence_score {confidence} is outside expected range [0.0, 1.0]"
            )
        
        # Validate recommendations is a list
        if not isinstance(response.get("recommendations"), list):
            error_msg = "recommendations must be a list"
            logger.error(error_msg)
            raise LLMResponseValidationError(error_msg)
        
        logger.info("Response validation successful")
        return True

    def _generate(self, prompt: str) -> str:
        """
        Generate response using Groq API.
        
        Args:
            prompt: Complete prompt string
            
        Returns:
            Generated text response
            
        Raises:
            LLMGenerationError: If generation fails
        """
        return self._generate_groq(prompt)
    
    def _generate_groq(self, prompt: str) -> str:
        """Generate response using Groq API."""
        try:
            logger.info(f"Sending request to Groq API (model: {self.model_name})")
            
            # Generate content using Groq
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    self.system_message,
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            
            if not response or not response.choices:
                error_msg = "Groq API returned empty response"
                logger.error(error_msg)
                raise LLMGenerationError(error_msg)
            
            # Extract text from response
            generated_text = response.choices[0].message.content
            
            if not generated_text:
                error_msg = "Groq API returned empty content"
                logger.error(error_msg)
                raise LLMGenerationError(error_msg)
            
            # Check finish reason
            finish_reason = response.choices[0].finish_reason
            if finish_reason != "stop":
                logger.warning(f"Generation stopped with reason: {finish_reason}")
                if finish_reason == "length":
                    logger.warning("Response may be truncated due to max_tokens limit")
            
            logger.info(f"Groq response received (length: {len(generated_text)} chars)")
            logger.debug(f"First 200 chars: {generated_text[:200]}")
            
            return generated_text
            
        except LLMGenerationError:
            raise
        except Exception as e:
            error_msg = f"Groq API generation failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise LLMGenerationError(error_msg)

    def generate_competitive_intelligence(
        self,
        product_idea: str,
        competitors_context: str
    ) -> Dict[str, Any]:
        """
        Generate competitive intelligence insights from competitor data.
        
        Builds structured prompt with product idea and competitor data,
        requests LLM to generate insights, and parses JSON response.
        Includes caching, rate limiting, and usage monitoring.
        
        Args:
            product_idea: User's product idea description
            competitors_context: Formatted competitor data and reviews
            
        Returns:
            Dictionary with sections: market_positioning, feature_gaps,
            sentiment_summary, recommendations, confidence_score
            
        Raises:
            LLMGenerationError: If generation fails
            LLMResponseValidationError: If response validation fails
            RateLimitExceeded: If rate limit is exceeded
            
        Example:
            >>> generator = LLMGenerator()
            >>> result = generator.generate_competitive_intelligence(
            ...     "A task management app with AI",
            ...     "Competitor: Todoist\\nReviews: Great app..."
            ... )
            >>> result["confidence_score"]
            0.85
        """
        try:
            start_time = time.time()
            
            # Check cache first
            if self.cache:
                cached_response = self.cache.get(product_idea, competitors_context)
                if cached_response:
                    logger.info("Returning cached response (saved API call)")
                    if self.usage_monitor:
                        self.usage_monitor.record_request(cached=True)
                    return cached_response
            
            logger.info("Starting competitive intelligence generation with Groq API")
            
            # Build prompt
            prompt = self.build_analysis_prompt(product_idea, competitors_context)
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Estimate tokens (rough estimate: 1 token ≈ 4 characters)
            estimated_tokens = len(prompt) // 4 + self.max_tokens
            
            # Check rate limits and wait if needed
            if self.rate_limiter:
                try:
                    self.rate_limiter.wait_if_needed(estimated_tokens, timeout=60.0)
                except RateLimitExceeded as e:
                    logger.error(f"Rate limit exceeded: {e}")
                    # Log current stats
                    if self.usage_monitor:
                        today_usage = self.usage_monitor.get_today_usage()
                        logger.error(
                            f"Today's usage: {today_usage['requests']}/{today_usage['requests_limit']} requests"
                        )
                    raise
            
            # Generate with Groq
            generated_text = self._generate(prompt)
            
            generation_time = (time.time() - start_time) * 1000
            logger.info(f"LLM generation completed in {generation_time:.2f}ms")
            
            # Record successful request
            if self.rate_limiter:
                self.rate_limiter.record_request(estimated_tokens)
            
            # Track LLM generation time
            metrics_collector.track_operation("llm_generation", generation_time)
            metrics_collector.track_memory_usage()
            
            # Parse JSON from response
            try:
                # Try to find JSON in the response
                # Sometimes LLMs wrap JSON in code blocks
                json_start = generated_text.find("{")
                json_end = generated_text.rfind("}") + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON object found in response")
                
                json_text = generated_text[json_start:json_end]
                parsed_response = json.loads(json_text)
                
            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"Failed to parse JSON from LLM response: {e}"
                logger.error(error_msg)
                logger.debug(f"Raw response: {generated_text[:500]}...")
                raise LLMResponseValidationError(error_msg)
            
            # Validate response structure
            self.validate_response(parsed_response)
            
            # Cache the response
            if self.cache:
                self.cache.set(product_idea, competitors_context, parsed_response)
            
            # Record usage
            if self.usage_monitor:
                self.usage_monitor.record_request(
                    tokens_used=estimated_tokens,
                    cached=False,
                    response_time_ms=generation_time
                )
            
            logger.info("Competitive intelligence generation successful")
            return parsed_response
            
        except RateLimitExceeded:
            raise
        except LLMGenerationError:
            raise
        except LLMResponseValidationError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error during LLM generation: {e}"
            logger.error(error_msg, exc_info=True)
            raise LLMGenerationError(error_msg)

    def stream_response(
        self,
        context: str,
        query: str
    ) -> Iterator[str]:
        """
        Stream response tokens for real-time output.
        
        Args:
            context: Context information for generation
            query: User query or product idea
            
        Yields:
            Response tokens as they are generated
            
        Raises:
            LLMGenerationError: If streaming fails
            
        Example:
            >>> generator = LLMGenerator()
            >>> for token in generator.stream_response("Context...", "Query..."):
            ...     print(token, end="", flush=True)
        """
        try:
            logger.info("Starting streaming response generation with Groq API")
            start_time = time.time()
            
            # Build prompt
            prompt = self.build_analysis_prompt(query, context)
            
            # Stream with Groq
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    self.system_message,
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            token_count = 0
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token_count += 1
                    yield chunk.choices[0].delta.content
            
            generation_time = (time.time() - start_time) * 1000
            logger.info(
                f"Streaming completed: {token_count} chunks in "
                f"{generation_time:.2f}ms"
            )
            
        except Exception as e:
            error_msg = f"Streaming error: {e}"
            logger.error(error_msg, exc_info=True)
            raise LLMGenerationError(error_msg)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics.
        
        Returns:
            Dictionary with rate limiter, cache, and usage monitor stats
        """
        stats = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        
        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        if self.usage_monitor:
            stats["today"] = self.usage_monitor.get_today_usage()
            stats["weekly"] = self.usage_monitor.get_weekly_summary()
            stats["lifetime"] = self.usage_monitor.get_lifetime_stats()
        
        return stats
    
    def print_usage_report(self) -> None:
        """Print comprehensive usage report."""
        if self.usage_monitor:
            self.usage_monitor.print_usage_report()
        
        if self.rate_limiter:
            rate_stats = self.rate_limiter.get_stats()
            print(f"⏱️  RATE LIMITER:")
            print(f"  RPM: {rate_stats['rpm_current']}/{rate_stats['rpm_limit']} "
                  f"({rate_stats['rpm_utilization']:.1f}%)")
            print(f"  RPD: {rate_stats['rpd_current']}/{rate_stats['rpd_limit']} "
                  f"({rate_stats['rpd_utilization']:.1f}%)")
            print(f"  Blocked: {rate_stats['blocked_requests']} requests\n")
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            print(f"💾 CACHE:")
            print(f"  Hit Rate: {cache_stats['hit_rate']:.1f}%")
            print(f"  Entries: {cache_stats['cache_entries']}")
            print(f"  Size: {cache_stats['cache_size_mb']:.1f}MB / "
                  f"{cache_stats['cache_limit_mb']:.1f}MB")
            print(f"  Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}\n")
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics.
        
        Returns:
            Dictionary with rate limiter, cache, and usage monitor stats
        """
        stats = {
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
        }
        
        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        if self.usage_monitor:
            stats["today"] = self.usage_monitor.get_today_usage()
            stats["weekly"] = self.usage_monitor.get_weekly_summary()
            stats["lifetime"] = self.usage_monitor.get_lifetime_stats()
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of LLMGenerator."""
        return (
            f"LLMGenerator(provider='{self.provider}', model='{self.model_name}', "
            f"temperature={self.temperature}, top_p={self.top_p})"
        )
