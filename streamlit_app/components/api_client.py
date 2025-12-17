"""
API Client for Product Hunt RAG Analyzer Backend.

This module provides a client for communicating with the FastAPI backend,
handling all HTTP requests, error handling, and response parsing.

Performance optimizations:
- Connection pooling via requests.Session
- Configurable timeouts
- Efficient error handling
"""

import os
from typing import Dict, Any, Optional
import requests
from requests.exceptions import (
    RequestException,
    Timeout,
    ConnectionError,
    HTTPError
)


class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class APIConnectionError(APIClientError):
    """Raised when unable to connect to the backend."""
    pass


class APITimeoutError(APIClientError):
    """Raised when a request times out."""
    pass


class APIResponseError(APIClientError):
    """Raised when the API returns an error response."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 error_details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_details = error_details or {}


class APIClient:
    """
    Client for communicating with the Product Hunt RAG Analyzer backend.
    
    Handles all HTTP communication, error handling, and response parsing
    for the Streamlit frontend.
    
    Attributes:
        base_url: Base URL of the backend API
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None, 
        timeout: int = 300
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the backend API. If None, reads from
                     BACKEND_URL environment variable or defaults to
                     http://localhost:8000
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = (
            base_url or 
            os.getenv("BACKEND_URL", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = timeout
        
        # Create session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the backend API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (without base URL)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            APIConnectionError: If unable to connect to backend
            APITimeoutError: If request times out
            APIResponseError: If API returns an error response
        """
        url = f"{self.base_url}{endpoint}"
        
        # Set timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            # Raise HTTPError for bad status codes
            response.raise_for_status()
            
            # Parse JSON response
            return response.json()
            
        except Timeout as e:
            raise APITimeoutError(
                f"Request to {endpoint} timed out after {self.timeout} seconds"
            ) from e
            
        except ConnectionError as e:
            raise APIConnectionError(
                f"Unable to connect to backend at {self.base_url}. "
                f"Please ensure the backend is running."
            ) from e
            
        except HTTPError as e:
            # Try to extract error details from response
            error_details = {}
            try:
                error_data = e.response.json()
                error_details = error_data.get("detail", {})
                if isinstance(error_details, str):
                    error_message = error_details
                else:
                    error_message = error_details.get(
                        "message", 
                        f"HTTP {e.response.status_code} error"
                    )
            except Exception:
                error_message = f"HTTP {e.response.status_code} error"
            
            raise APIResponseError(
                error_message,
                status_code=e.response.status_code,
                error_details=error_details
            ) from e
            
        except RequestException as e:
            raise APIClientError(
                f"Request failed: {str(e)}"
            ) from e
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health status of the backend API.
        
        Queries the /api/v1/health endpoint to verify backend connectivity
        and check the status of dependencies (Ollama, FAISS indices).
        
        Returns:
            Dictionary containing:
                - status: "healthy", "degraded", or "unhealthy"
                - timestamp: ISO timestamp of health check
                - version: API version
                - ollama_connected: Whether Ollama LLM is connected
                - indices_loaded: Whether FAISS indices are loaded
                
        Raises:
            APIConnectionError: If unable to connect to backend
            APITimeoutError: If request times out
            APIResponseError: If API returns an error
        """
        return self._make_request("GET", "/api/v1/health")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.
        
        Queries the /api/v1/dataset/stats endpoint to retrieve information
        about the number of products and reviews in the loaded indices.
        
        Returns:
            Dictionary containing:
                - total_products: Number of products in index
                - total_reviews: Number of reviews in index
                - avg_reviews_per_product: Average reviews per product
                - indices_loaded: Whether indices are loaded
                
        Raises:
            APIConnectionError: If unable to connect to backend
            APITimeoutError: If request times out
            APIResponseError: If API returns an error
        """
        return self._make_request("GET", "/api/v1/dataset/stats")
    
    def submit_analysis(
        self,
        product_idea: str,
        max_competitors: int = 5,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Submit a product idea for competitive analysis.
        
        Posts to the /api/v1/analyze endpoint to execute the full
        analysis pipeline and retrieve comprehensive results.
        
        Args:
            product_idea: Description of the product idea to analyze
            max_competitors: Maximum number of competitors to identify (1-10)
            output_format: Desired output format ("json", "markdown", or "pdf")
            
        Returns:
            Dictionary containing:
                - analysis_id: Unique identifier for this analysis
                - status: Analysis status ("completed" or "failed")
                - product_idea: Original product idea
                - competitors_identified: List of competitor names
                - results: Comprehensive analysis results including:
                    - market_positioning: Market positioning insights
                    - feature_gaps: Feature gap analysis by priority
                    - sentiment_summary: Sentiment analysis results
                    - recommendations: Strategic recommendations
                - confidence_score: Confidence score (0.0 to 1.0)
                - generated_at: ISO timestamp of analysis generation
                - processing_time_ms: Processing time in milliseconds
                - warnings: Optional list of warnings
                
        Raises:
            APIConnectionError: If unable to connect to backend
            APITimeoutError: If request times out
            APIResponseError: If API returns an error or analysis fails
        """
        payload = {
            "product_idea": product_idea,
            "max_competitors": max_competitors,
            "output_format": output_format
        }
        
        return self._make_request(
            "POST",
            "/api/v1/analyze",
            json=payload
        )
    
    def is_backend_available(self) -> bool:
        """
        Check if the backend is available and responding.
        
        Attempts to connect to the backend health endpoint to verify
        connectivity. This is a convenience method that returns a boolean
        instead of raising exceptions.
        
        Returns:
            True if backend is reachable and responding, False otherwise
        """
        try:
            self.check_health()
            return True
        except (APIConnectionError, APITimeoutError, APIClientError):
            return False
    
    def analyze_feature_gaps(
        self,
        product_name: str,
        product_description: str,
        product_id: Optional[str] = None,
        existing_features: Optional[list] = None,
        include_llm_suggestions: bool = True,
        min_gaps_threshold: int = 3
    ) -> Dict[str, Any]:
        """
        Perform feature gap analysis for a product.
        
        Posts to the /api/v1/feature-gaps/analyze endpoint to identify
        missing features from reviews and generate AI suggestions.
        
        The analysis follows a two-step process:
        1. Analyzes product reviews to identify missing features
        2. Falls back to LLM-generated suggestions when insufficient gaps found
        
        Args:
            product_name: Name of the product to analyze
            product_description: Description of the product
            product_id: Optional product ID for retrieving reviews from index
            existing_features: Optional list of known product features
            include_llm_suggestions: Whether to include LLM suggestions (default: True)
            min_gaps_threshold: Minimum gaps before triggering LLM fallback (default: 3)
            
        Returns:
            Dictionary containing:
                - product_name: Name of the analyzed product
                - product_description: Description of the product
                - total_reviews_analyzed: Number of reviews analyzed
                - source: Source of gaps ("review_analysis", "llm_generated", "hybrid")
                - summary: Summary of the analysis
                - confidence_score: Confidence score (0.0 to 1.0)
                - gaps_from_reviews: List of gaps identified from reviews
                - llm_generated_suggestions: List of AI-generated suggestions
                - recommendations: Actionable recommendations
                - metadata: Additional metadata
                
        Raises:
            APIConnectionError: If unable to connect to backend
            APITimeoutError: If request times out
            APIResponseError: If API returns an error
        """
        payload = {
            "product_name": product_name,
            "product_description": product_description,
            "include_llm_suggestions": include_llm_suggestions,
            "min_gaps_threshold": min_gaps_threshold
        }
        
        if product_id:
            payload["product_id"] = product_id
        
        if existing_features:
            payload["existing_features"] = existing_features
        
        return self._make_request(
            "POST",
            "/api/v1/feature-gaps/analyze",
            json=payload
        )
    
    def close(self):
        """
        Close the API client and cleanup resources.
        
        Closes the underlying requests session to free up connections.
        """
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
