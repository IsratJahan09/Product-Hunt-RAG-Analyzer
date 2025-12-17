"""
Dataset loading and validation module for Product Hunt RAG Analyzer.

This module provides the DatasetLoader class for loading, validating,
and extracting data from Product Hunt datasets in JSON/JSONL format.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from src.utils.logger import get_logger


logger = get_logger(__name__)


class DatasetValidationError(Exception):
    """Exception raised when dataset validation fails."""
    pass


class DatasetLoader:
    """
    Loads and validates Product Hunt datasets.
    
    Provides methods to load JSON/JSONL datasets, validate structure,
    compute statistics, and extract product and review data for embeddings.
    """
    
    # Required fields for product data
    REQUIRED_PRODUCT_FIELDS = [
        "id", "name", "tagline", "description", 
        "votesCount", "commentsCount", "createdAt"
    ]
    
    # Required fields for review data
    REQUIRED_REVIEW_FIELDS = [
        "id", "body", "createdAt", "product_id"
    ]
    
    def __init__(self):
        """Initialize DatasetLoader."""
        self.products_data: List[Dict[str, Any]] = []
        self.reviews_data: List[Dict[str, Any]] = []
        self.dataset_loaded = False
    
    def load_dataset(
        self,
        dataset_path: str,
        dataset_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Load dataset from JSON or JSONL file.
        
        Args:
            dataset_path: Path to dataset file (JSON or JSONL)
            dataset_type: Type of dataset - "products", "reviews", or "auto" to detect
            
        Returns:
            Dictionary with loaded data
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            DatasetValidationError: If dataset format is invalid
            
        Example:
            >>> loader = DatasetLoader()
            >>> data = loader.load_dataset("dataset/products.jsonl")
        """
        logger.info(f"Loading dataset from: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            error_msg = f"Dataset file not found: {dataset_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Determine file format
            file_ext = Path(dataset_path).suffix.lower()
            
            if file_ext == ".jsonl":
                data = self._load_jsonl(dataset_path)
            elif file_ext == ".json":
                data = self._load_json(dataset_path)
            else:
                raise DatasetValidationError(
                    f"Unsupported file format: {file_ext}. Use .json or .jsonl"
                )
            
            # Auto-detect dataset type if needed
            if dataset_type == "auto":
                dataset_type = self._detect_dataset_type(data)
                logger.info(f"Auto-detected dataset type: {dataset_type}")
            
            # Store data based on type
            if dataset_type == "products":
                self.products_data = data if isinstance(data, list) else [data]
            elif dataset_type == "reviews":
                self.reviews_data = data if isinstance(data, list) else [data]
            else:
                raise DatasetValidationError(
                    f"Invalid dataset type: {dataset_type}. Use 'products' or 'reviews'"
                )
            
            self.dataset_loaded = True
            
            logger.info(
                f"Successfully loaded {len(data)} records from {dataset_path}"
            )
            
            return {
                "type": dataset_type,
                "count": len(data),
                "path": dataset_path
            }
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON from {dataset_path}: {e}"
            logger.error(error_msg)
            raise DatasetValidationError(error_msg)
        except Exception as e:
            error_msg = f"Error loading dataset: {e}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise DatasetValidationError(
                f"Invalid JSON structure. Expected list or dict, got {type(data)}"
            )
    
    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file (one JSON object per line)."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping invalid JSON at line {line_num}: {e}"
                    )
                    continue
        
        return data
    
    def _detect_dataset_type(self, data: List[Dict[str, Any]]) -> str:
        """
        Auto-detect dataset type based on fields.
        
        Args:
            data: List of records
            
        Returns:
            "products" or "reviews"
        """
        if not data:
            raise DatasetValidationError("Cannot detect type of empty dataset")
        
        sample = data[0]
        
        # Check for product-specific fields
        product_indicators = ["tagline", "votesCount", "thumbnail"]
        product_score = sum(1 for field in product_indicators if field in sample)
        
        # Check for review-specific fields
        review_indicators = ["body", "product_id", "user"]
        review_score = sum(1 for field in review_indicators if field in sample)
        
        if product_score > review_score:
            return "products"
        elif review_score > product_score:
            return "reviews"
        else:
            # Default to products if ambiguous
            logger.warning("Ambiguous dataset type, defaulting to 'products'")
            return "products"
    
    def validate_dataset_structure(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        dataset_type: str = "products"
    ) -> Dict[str, Any]:
        """
        Validate dataset structure and check for required fields.
        
        Args:
            data: Dataset to validate. If None, uses loaded data.
            dataset_type: Type of dataset - "products" or "reviews"
            
        Returns:
            Dictionary with validation results
            
        Raises:
            DatasetValidationError: If validation fails critically
            
        Example:
            >>> loader = DatasetLoader()
            >>> loader.load_dataset("dataset/products.jsonl")
            >>> results = loader.validate_dataset_structure()
        """
        logger.info(f"Validating {dataset_type} dataset structure")
        
        # Use loaded data if not provided
        if data is None:
            data = self.products_data if dataset_type == "products" else self.reviews_data
        
        if not data:
            raise DatasetValidationError("No data to validate")
        
        # Select required fields based on type
        required_fields = (
            self.REQUIRED_PRODUCT_FIELDS if dataset_type == "products"
            else self.REQUIRED_REVIEW_FIELDS
        )
        
        validation_results = {
            "total_records": len(data),
            "valid_records": 0,
            "invalid_records": 0,
            "missing_fields": defaultdict(int),
            "incomplete_records": [],
            "field_coverage": {}
        }
        
        for idx, record in enumerate(data):
            missing = [field for field in required_fields if field not in record]
            
            if missing:
                validation_results["invalid_records"] += 1
                validation_results["incomplete_records"].append({
                    "index": idx,
                    "id": record.get("id", "unknown"),
                    "missing_fields": missing
                })
                
                for field in missing:
                    validation_results["missing_fields"][field] += 1
            else:
                validation_results["valid_records"] += 1
        
        # Calculate field coverage
        for field in required_fields:
            present_count = sum(
                1 for record in data if field in record and record[field] is not None
            )
            validation_results["field_coverage"][field] = {
                "present": present_count,
                "missing": len(data) - present_count,
                "coverage_pct": (present_count / len(data)) * 100
            }
        
        # Log validation summary
        logger.info(
            f"Validation complete: {validation_results['valid_records']}/{len(data)} "
            f"records valid"
        )
        
        if validation_results["invalid_records"] > 0:
            logger.warning(
                f"Found {validation_results['invalid_records']} records with missing fields"
            )
            for field, count in validation_results["missing_fields"].items():
                logger.warning(f"  - {field}: missing in {count} records")
        
        return validation_results

    
    def get_dataset_statistics(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        dataset_type: str = "products"
    ) -> Dict[str, Any]:
        """
        Compute statistics for the dataset.
        
        Args:
            data: Dataset to analyze. If None, uses loaded data.
            dataset_type: Type of dataset - "products" or "reviews"
            
        Returns:
            Dictionary with dataset statistics
            
        Example:
            >>> loader = DatasetLoader()
            >>> loader.load_dataset("dataset/products.jsonl")
            >>> stats = loader.get_dataset_statistics()
        """
        logger.info(f"Computing statistics for {dataset_type} dataset")
        
        # Use loaded data if not provided
        if data is None:
            data = self.products_data if dataset_type == "products" else self.reviews_data
        
        if not data:
            logger.warning("No data available for statistics")
            return {}
        
        stats = {
            "total_records": len(data),
            "dataset_type": dataset_type
        }
        
        if dataset_type == "products":
            stats.update(self._compute_product_statistics(data))
        elif dataset_type == "reviews":
            stats.update(self._compute_review_statistics(data))
        
        logger.info(f"Statistics computed: {stats['total_records']} records analyzed")
        
        return stats
    
    def _compute_product_statistics(
        self,
        products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics specific to product data."""
        stats = {
            "total_products": len(products),
            "total_votes": 0,
            "total_comments": 0,
            "total_reviews": 0,
            "avg_votes_per_product": 0.0,
            "avg_comments_per_product": 0.0,
            "avg_reviews_per_product": 0.0,
            "date_range": {},
            "topics": defaultdict(int)
        }
        
        dates = []
        
        for product in products:
            # Aggregate counts
            stats["total_votes"] += product.get("votesCount", 0)
            stats["total_comments"] += product.get("commentsCount", 0)
            stats["total_reviews"] += product.get("reviewsCount", 0)
            
            # Collect dates
            created_at = product.get("createdAt")
            if created_at:
                try:
                    date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    dates.append(date)
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid date format for product {product.get('id')}")
            
            # Count topics
            topics = product.get("topics", {})
            if isinstance(topics, dict) and "edges" in topics:
                for edge in topics["edges"]:
                    topic_name = edge.get("node", {}).get("name")
                    if topic_name:
                        stats["topics"][topic_name] += 1
        
        # Calculate averages
        if len(products) > 0:
            stats["avg_votes_per_product"] = stats["total_votes"] / len(products)
            stats["avg_comments_per_product"] = stats["total_comments"] / len(products)
            stats["avg_reviews_per_product"] = stats["total_reviews"] / len(products)
        
        # Date range
        if dates:
            stats["date_range"] = {
                "earliest": min(dates).isoformat(),
                "latest": max(dates).isoformat(),
                "span_days": (max(dates) - min(dates)).days
            }
        
        # Convert topics to regular dict and get top topics
        stats["topics"] = dict(stats["topics"])
        stats["top_topics"] = sorted(
            stats["topics"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return stats
    
    def _compute_review_statistics(
        self,
        reviews: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics specific to review data."""
        stats = {
            "total_reviews": len(reviews),
            "total_votes": 0,
            "avg_votes_per_review": 0.0,
            "avg_review_length": 0.0,
            "date_range": {},
            "reviews_by_product": defaultdict(int),
            "unique_products": 0,
            "unique_users": 0
        }
        
        dates = []
        review_lengths = []
        unique_products = set()
        unique_users = set()
        
        for review in reviews:
            # Aggregate votes
            stats["total_votes"] += review.get("votesCount", 0)
            
            # Track review length
            body = review.get("body", "")
            if body:
                review_lengths.append(len(body))
            
            # Collect dates
            created_at = review.get("createdAt")
            if created_at:
                try:
                    date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    dates.append(date)
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid date format for review {review.get('id')}")
            
            # Track products and users
            product_id = review.get("product_id")
            if product_id:
                unique_products.add(product_id)
                stats["reviews_by_product"][product_id] += 1
            
            user_id = review.get("user", {}).get("id")
            if user_id:
                unique_users.add(user_id)
        
        # Calculate averages
        if len(reviews) > 0:
            stats["avg_votes_per_review"] = stats["total_votes"] / len(reviews)
        
        if review_lengths:
            stats["avg_review_length"] = sum(review_lengths) / len(review_lengths)
        
        # Date range
        if dates:
            stats["date_range"] = {
                "earliest": min(dates).isoformat(),
                "latest": max(dates).isoformat(),
                "span_days": (max(dates) - min(dates)).days
            }
        
        # Unique counts
        stats["unique_products"] = len(unique_products)
        stats["unique_users"] = len(unique_users)
        
        # Convert reviews_by_product to regular dict
        stats["reviews_by_product"] = dict(stats["reviews_by_product"])
        
        return stats
    
    def extract_products(
        self,
        data: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract product-level data for product embeddings.
        
        Combines name, tagline, and description into a single text field
        for embedding generation.
        
        Args:
            data: Product dataset. If None, uses loaded products_data.
            
        Returns:
            List of dictionaries with product data formatted for embeddings
            
        Example:
            >>> loader = DatasetLoader()
            >>> loader.load_dataset("dataset/products.jsonl", "products")
            >>> products = loader.extract_products()
        """
        logger.info("Extracting product-level data for embeddings")
        
        # Use loaded data if not provided
        if data is None:
            data = self.products_data
        
        if not data:
            logger.warning("No product data available for extraction")
            return []
        
        extracted_products = []
        
        for product in data:
            try:
                # Combine text fields for embedding
                text_parts = []
                
                name = product.get("name", "").strip()
                if name:
                    text_parts.append(name)
                
                tagline = product.get("tagline", "").strip()
                if tagline:
                    text_parts.append(tagline)
                
                description = product.get("description", "").strip()
                if description:
                    text_parts.append(description)
                
                combined_text = " | ".join(text_parts)
                
                if not combined_text:
                    logger.warning(
                        f"Product {product.get('id')} has no text content, skipping"
                    )
                    continue
                
                # Extract topics
                topics = []
                topics_data = product.get("topics", {})
                if isinstance(topics_data, dict) and "edges" in topics_data:
                    for edge in topics_data["edges"]:
                        topic_name = edge.get("node", {}).get("name")
                        if topic_name:
                            topics.append(topic_name)
                
                extracted_product = {
                    "product_id": product.get("id"),
                    "name": name,
                    "tagline": tagline,
                    "description": description,
                    "combined_text": combined_text,
                    "votesCount": product.get("votesCount", 0),
                    "commentsCount": product.get("commentsCount", 0),
                    "reviewsCount": product.get("reviewsCount", 0),
                    "createdAt": product.get("createdAt"),
                    "url": product.get("url"),
                    "topics": topics,
                    "thumbnail_url": product.get("thumbnail", {}).get("url")
                }
                
                extracted_products.append(extracted_product)
                
            except Exception as e:
                logger.error(
                    f"Error extracting product {product.get('id')}: {e}",
                    exc_info=True
                )
                continue
        
        logger.info(
            f"Extracted {len(extracted_products)} products from {len(data)} records"
        )
        
        return extracted_products
    
    def extract_reviews(
        self,
        data: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract review-level data for review embeddings.
        
        Args:
            data: Review dataset or product dataset with nested reviews. 
                  If None, uses loaded reviews_data or extracts from products_data.
            
        Returns:
            List of dictionaries with review data formatted for embeddings
            
        Example:
            >>> loader = DatasetLoader()
            >>> loader.load_dataset("dataset/reviews.jsonl", "reviews")
            >>> reviews = loader.extract_reviews()
        """
        logger.info("Extracting review-level data for embeddings")
        
        # Use loaded data if not provided
        if data is None:
            data = self.reviews_data
            
            # If no reviews_data, try to extract from products_data
            if not data and self.products_data:
                logger.info("No reviews_data found, extracting reviews from products_data")
                return self._extract_reviews_from_products(self.products_data)
        
        if not data:
            logger.warning("No review data available for extraction")
            return []
        
        # Check if data contains nested reviews (products with reviews)
        if data and isinstance(data[0], dict) and "reviews" in data[0]:
            logger.info("Detected nested reviews in product data, extracting...")
            return self._extract_reviews_from_products(data)
        
        extracted_reviews = []
        
        for review in data:
            try:
                body = review.get("body", "").strip()
                
                if not body:
                    logger.warning(
                        f"Review {review.get('id')} has no body text, skipping"
                    )
                    continue
                
                # Extract user information
                user = review.get("user", {})
                user_id = user.get("id", "0")
                user_name = user.get("name", "[REDACTED]")
                user_username = user.get("username", "[REDACTED]")
                
                extracted_review = {
                    "review_id": review.get("id"),
                    "product_id": review.get("product_id"),
                    "body": body,
                    "votesCount": review.get("votesCount", 0),
                    "createdAt": review.get("createdAt"),
                    "user_id": user_id,
                    "user_name": user_name,
                    "user_username": user_username
                }
                
                extracted_reviews.append(extracted_review)
                
            except Exception as e:
                logger.error(
                    f"Error extracting review {review.get('id')}: {e}",
                    exc_info=True
                )
                continue
        
        logger.info(
            f"Extracted {len(extracted_reviews)} reviews from {len(data)} records"
        )
        
        return extracted_reviews
    
    def _extract_reviews_from_products(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract reviews from products with nested review data.
        
        Args:
            products: List of product dictionaries with nested reviews
            
        Returns:
            List of flattened review dictionaries
        """
        extracted_reviews = []
        
        for product in products:
            product_id = product.get("id")
            reviews = product.get("reviews", [])
            
            for review in reviews:
                try:
                    body = review.get("body", "").strip()
                    
                    if not body:
                        logger.warning(
                            f"Review {review.get('id')} has no body text, skipping"
                        )
                        continue
                    
                    # Extract user information if available
                    user = review.get("user", {})
                    user_id = user.get("id", "0") if user else "0"
                    user_name = user.get("name", "[REDACTED]") if user else "[REDACTED]"
                    user_username = user.get("username", "[REDACTED]") if user else "[REDACTED]"
                    
                    extracted_review = {
                        "review_id": review.get("id"),
                        "product_id": product_id,
                        "body": body,
                        "votesCount": review.get("votesCount", 0),
                        "createdAt": review.get("createdAt"),
                        "user_id": user_id,
                        "user_name": user_name,
                        "user_username": user_username
                    }
                    
                    extracted_reviews.append(extracted_review)
                    
                except Exception as e:
                    logger.error(
                        f"Error extracting review {review.get('id')}: {e}",
                        exc_info=True
                    )
                    continue
        
        logger.info(
            f"Extracted {len(extracted_reviews)} reviews from {len(products)} products"
        )
        
        return extracted_reviews
    
    def load_multiple_datasets(
        self,
        products_path: Optional[str] = None,
        reviews_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load both products and reviews datasets.
        
        Args:
            products_path: Path to products dataset
            reviews_path: Path to reviews dataset
            
        Returns:
            Dictionary with loading results for both datasets
            
        Example:
            >>> loader = DatasetLoader()
            >>> results = loader.load_multiple_datasets(
            ...     "dataset/products.jsonl",
            ...     "dataset/reviews.jsonl"
            ... )
        """
        results = {}
        
        if products_path:
            try:
                results["products"] = self.load_dataset(products_path, "products")
            except Exception as e:
                logger.error(f"Failed to load products dataset: {e}")
                results["products"] = {"error": str(e)}
        
        if reviews_path:
            try:
                results["reviews"] = self.load_dataset(reviews_path, "reviews")
            except Exception as e:
                logger.error(f"Failed to load reviews dataset: {e}")
                results["reviews"] = {"error": str(e)}
        
        return results
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific product by ID.
        
        Args:
            product_id: Product ID to retrieve
            
        Returns:
            Product dictionary or None if not found
        """
        for product in self.products_data:
            if product.get("id") == product_id:
                return product
        return None
    
    def get_reviews_by_product_id(self, product_id: str) -> List[Dict[str, Any]]:
        """
        Get all reviews for a specific product.
        
        Args:
            product_id: Product ID
            
        Returns:
            List of review dictionaries
        """
        return [
            review for review in self.reviews_data
            if review.get("product_id") == product_id
        ]
    
    def clear(self) -> None:
        """Clear all loaded data."""
        self.products_data = []
        self.reviews_data = []
        self.dataset_loaded = False
        logger.info("Dataset loader cleared")
