"""
Product Hunt API client module for data ingestion.

This module provides the ProductHuntAPIClient class for fetching product data,
reviews, and metadata from the Product Hunt API with robust error handling,
retry logic, and comprehensive logging.
"""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils.logger import get_logger, log_error, log_performance, LogContext


class ProductHuntAPIError(Exception):
    """Base exception for Product Hunt API errors."""
    pass


class RateLimitError(ProductHuntAPIError):
    """Exception raised when API rate limit is exceeded."""
    pass


class AuthenticationError(ProductHuntAPIError):
    """Exception raised when authentication fails."""
    pass


class InvalidResponseError(ProductHuntAPIError):
    """Exception raised when API returns invalid response."""
    pass


class ProductHuntAPIClient:
    """
    Client for interacting with Product Hunt API.
    
    Provides methods to fetch products, product details, and reviews with
    automatic retry logic, rate limiting handling, and comprehensive error
    handling.
    
    Attributes:
        base_url: Base URL for Product Hunt API
        timeout: Request timeout in seconds
        retry_attempts: Maximum number of retry attempts
        retry_delays: List of delays for exponential backoff
        logger: Logger instance for this client
    """
    
    def __init__(
        self,
        base_url: str = "https://api.producthunt.com/v2",
        api_token: Optional[str] = None,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delays: Optional[List[int]] = None
    ):
        """
        Initialize Product Hunt API client.
        
        Args:
            base_url: Base URL for Product Hunt API
            api_token: API authentication token (optional)
            timeout: Request timeout in seconds
            retry_attempts: Maximum number of retry attempts
            retry_delays: List of delays for exponential backoff (default: [1, 2, 4, 8, 16])
        """
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delays = retry_delays or [1, 2, 4, 8, 16]
        self.logger = get_logger(__name__)
        
        # Setup session with headers
        self.session = requests.Session()
        if self.api_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
        
        self.logger.info(
            f"ProductHuntAPIClient initialized with base_url={base_url}, "
            f"timeout={timeout}s, retry_attempts={retry_attempts}"
        )
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            retry_count: Current retry attempt number
            
        Returns:
            Response data as dictionary
            
        Raises:
            RateLimitError: When rate limit is exceeded
            AuthenticationError: When authentication fails
            InvalidResponseError: When response is invalid
            ProductHuntAPIError: For other API errors
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            start_time = time.time()
            
            self.logger.debug(
                f"Making {method} request to {url} "
                f"(attempt {retry_count + 1}/{self.retry_attempts + 1})"
            )
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request details
            self.logger.info(
                f"{method} {endpoint} - Status: {response.status_code}, "
                f"Duration: {duration_ms:.2f}ms"
            )
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                if retry_count < self.retry_attempts:
                    delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                    self.logger.warning(
                        f"Rate limit exceeded. Retrying in {delay}s "
                        f"(attempt {retry_count + 1}/{self.retry_attempts})"
                    )
                    time.sleep(delay)
                    return self._make_request(method, endpoint, params, data, retry_count + 1)
                else:
                    raise RateLimitError(
                        f"Rate limit exceeded after {self.retry_attempts} retries"
                    )
            
            # Handle authentication errors (401, 403)
            if response.status_code in [401, 403]:
                raise AuthenticationError(
                    f"Authentication failed: {response.status_code} - {response.text}"
                )
            
            # Handle client errors (4xx)
            if 400 <= response.status_code < 500:
                error_msg = f"Client error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise ProductHuntAPIError(error_msg)
            
            # Handle server errors (5xx) with retry
            if response.status_code >= 500:
                if retry_count < self.retry_attempts:
                    delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                    self.logger.warning(
                        f"Server error {response.status_code}. Retrying in {delay}s "
                        f"(attempt {retry_count + 1}/{self.retry_attempts})"
                    )
                    time.sleep(delay)
                    return self._make_request(method, endpoint, params, data, retry_count + 1)
                else:
                    raise ProductHuntAPIError(
                        f"Server error after {self.retry_attempts} retries: "
                        f"{response.status_code} - {response.text}"
                    )
            
            # Parse JSON response
            try:
                response_data = response.json()
                return response_data
            except ValueError as e:
                raise InvalidResponseError(f"Invalid JSON response: {str(e)}")
        
        except requests.exceptions.Timeout as e:
            if retry_count < self.retry_attempts:
                delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                self.logger.warning(
                    f"Request timeout. Retrying in {delay}s "
                    f"(attempt {retry_count + 1}/{self.retry_attempts})"
                )
                time.sleep(delay)
                return self._make_request(method, endpoint, params, data, retry_count + 1)
            else:
                log_error(self.logger, e, f"Request timeout after {self.retry_attempts} retries")
                raise ProductHuntAPIError(f"Request timeout: {str(e)}")
        
        except requests.exceptions.ConnectionError as e:
            if retry_count < self.retry_attempts:
                delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                self.logger.warning(
                    f"Connection error. Retrying in {delay}s "
                    f"(attempt {retry_count + 1}/{self.retry_attempts})"
                )
                time.sleep(delay)
                return self._make_request(method, endpoint, params, data, retry_count + 1)
            else:
                log_error(self.logger, e, f"Connection error after {self.retry_attempts} retries")
                raise ProductHuntAPIError(f"Connection error: {str(e)}")
        
        except requests.exceptions.RequestException as e:
            log_error(self.logger, e, "Request failed")
            raise ProductHuntAPIError(f"Request failed: {str(e)}")
    
    def fetch_products(
        self,
        query: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Fetch products from Product Hunt API with pagination support.
        
        Args:
            query: Search query for products (optional)
            limit: Maximum number of products to fetch
            offset: Pagination offset
            
        Returns:
            List of product dictionaries with metadata
            
        Raises:
            ProductHuntAPIError: When API request fails
            
        Example:
            >>> client = ProductHuntAPIClient()
            >>> products = client.fetch_products(query="task management", limit=10)
        """
        with LogContext(self.logger, f"fetch_products(query={query}, limit={limit})"):
            try:
                params = {
                    'limit': limit,
                    'offset': offset
                }
                
                if query:
                    params['search'] = query
                
                # Note: This is a simplified implementation
                # Real Product Hunt API uses GraphQL, not REST
                # For this implementation, we'll structure it for REST-like access
                response = self._make_request('GET', '/posts', params=params)
                
                # Extract products from response
                products = response.get('posts', [])
                
                # Validate and normalize product data
                validated_products = []
                for product in products:
                    try:
                        validated_product = self._validate_product_data(product)
                        validated_products.append(validated_product)
                    except Exception as e:
                        self.logger.warning(
                            f"Skipping invalid product data: {str(e)}"
                        )
                        continue
                
                self.logger.info(
                    f"Successfully fetched {len(validated_products)} products "
                    f"(requested: {limit})"
                )
                
                return validated_products
            
            except Exception as e:
                log_error(self.logger, e, "Failed to fetch products")
                raise
    
    def fetch_product_details(self, product_id: str) -> Dict[str, Any]:
        """
        Fetch detailed product information including metadata.
        
        Args:
            product_id: Product ID to fetch details for
            
        Returns:
            Dictionary with detailed product information including:
            - id, name, tagline, description
            - votesCount, commentsCount, createdAt
            - url, thumbnail, topics, makers
            
        Raises:
            ProductHuntAPIError: When API request fails
            
        Example:
            >>> client = ProductHuntAPIClient()
            >>> details = client.fetch_product_details("12345")
        """
        with LogContext(self.logger, f"fetch_product_details(product_id={product_id})"):
            try:
                endpoint = f'/posts/{product_id}'
                response = self._make_request('GET', endpoint)
                
                # Extract product details
                product = response.get('post', {})
                
                if not product:
                    raise InvalidResponseError(
                        f"No product data found for ID: {product_id}"
                    )
                
                # Validate and normalize product details
                validated_product = self._validate_product_data(product)
                
                self.logger.info(
                    f"Successfully fetched details for product: {product_id}"
                )
                
                return validated_product
            
            except Exception as e:
                log_error(
                    self.logger,
                    e,
                    f"Failed to fetch product details for ID: {product_id}"
                )
                raise
    
    def fetch_reviews(
        self,
        product_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch reviews and comments for a product with user information.
        
        Args:
            product_id: Product ID to fetch reviews for
            limit: Maximum number of reviews to fetch
            
        Returns:
            List of review dictionaries with:
            - id, body, rating (if available)
            - userId, userName, userHeadline
            - createdAt, votesCount
            
        Raises:
            ProductHuntAPIError: When API request fails
            
        Example:
            >>> client = ProductHuntAPIClient()
            >>> reviews = client.fetch_reviews("12345", limit=20)
        """
        with LogContext(
            self.logger,
            f"fetch_reviews(product_id={product_id}, limit={limit})"
        ):
            try:
                endpoint = f'/posts/{product_id}/comments'
                params = {'limit': limit}
                
                response = self._make_request('GET', endpoint, params=params)
                
                # Extract comments/reviews from response
                comments = response.get('comments', [])
                
                # Validate and normalize review data
                validated_reviews = []
                for comment in comments:
                    try:
                        validated_review = self._validate_review_data(
                            comment,
                            product_id
                        )
                        validated_reviews.append(validated_review)
                    except Exception as e:
                        self.logger.warning(
                            f"Skipping invalid review data: {str(e)}"
                        )
                        continue
                
                self.logger.info(
                    f"Successfully fetched {len(validated_reviews)} reviews "
                    f"for product {product_id} (requested: {limit})"
                )
                
                return validated_reviews
            
            except Exception as e:
                log_error(
                    self.logger,
                    e,
                    f"Failed to fetch reviews for product ID: {product_id}"
                )
                raise
    
    def _validate_product_data(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize product data.
        
        Args:
            product: Raw product data from API
            
        Returns:
            Validated and normalized product dictionary
            
        Raises:
            InvalidResponseError: When required fields are missing
        """
        required_fields = ['id', 'name']
        
        # Check required fields
        for field in required_fields:
            if field not in product:
                raise InvalidResponseError(
                    f"Missing required field in product data: {field}"
                )
        
        # Normalize product data
        normalized = {
            'id': str(product['id']),
            'name': product['name'],
            'tagline': product.get('tagline', ''),
            'description': product.get('description', ''),
            'votesCount': product.get('votesCount', 0),
            'commentsCount': product.get('commentsCount', 0),
            'createdAt': product.get('createdAt', ''),
            'url': product.get('url', ''),
            'thumbnail': product.get('thumbnail', {}).get('url', ''),
            'topics': [
                topic.get('name', '') for topic in product.get('topics', [])
            ],
            'makers': [
                {
                    'id': maker.get('id', ''),
                    'name': maker.get('name', ''),
                    'headline': maker.get('headline', '')
                }
                for maker in product.get('makers', [])
            ]
        }
        
        return normalized
    
    def _validate_review_data(
        self,
        review: Dict[str, Any],
        product_id: str
    ) -> Dict[str, Any]:
        """
        Validate and normalize review/comment data.
        
        Args:
            review: Raw review data from API
            product_id: Associated product ID
            
        Returns:
            Validated and normalized review dictionary
            
        Raises:
            InvalidResponseError: When required fields are missing
        """
        required_fields = ['id', 'body']
        
        # Check required fields
        for field in required_fields:
            if field not in review:
                raise InvalidResponseError(
                    f"Missing required field in review data: {field}"
                )
        
        # Extract user information
        user = review.get('user', {})
        
        # Normalize review data
        normalized = {
            'id': str(review['id']),
            'productId': product_id,
            'body': review['body'],
            'rating': review.get('rating'),  # May not be available
            'userId': str(user.get('id', '')),
            'userName': user.get('name', ''),
            'userHeadline': user.get('headline', ''),
            'createdAt': review.get('createdAt', ''),
            'votesCount': review.get('votesCount', 0)
        }
        
        return normalized
    
    def close(self):
        """Close the session and cleanup resources."""
        self.session.close()
        self.logger.info("ProductHuntAPIClient session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
