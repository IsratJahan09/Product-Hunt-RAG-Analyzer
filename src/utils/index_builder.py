"""
Index building module for Product Hunt RAG Analyzer.

This module provides the IndexBuilder class for creating FAISS indices
from Product Hunt datasets, including both product and review embeddings.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.modules.preprocessing import TextPreprocessor
from src.modules.embeddings import EmbeddingGenerator
from src.modules.vector_storage import FAISSIndexManager
from src.utils.dataset_loader import DatasetLoader
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_collector


logger = get_logger(__name__)
metrics_collector = get_metrics_collector()


class IndexBuilder:
    """
    Orchestrates building FAISS indices from Product Hunt datasets.
    
    Provides methods to build product and review indices with preprocessing,
    embedding generation, and progress tracking.
    """
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        lazy_load: bool = False
    ):
        """
        Initialize IndexBuilder.
        
        Args:
            embedding_generator: EmbeddingGenerator instance. If None, creates default.
            preprocessor: TextPreprocessor instance. If None, creates default.
            lazy_load: If True, delay model loading until first use (default: False)
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator(lazy_load=lazy_load)
        self.preprocessor = preprocessor or TextPreprocessor()
        self.dataset_loader = DatasetLoader()
        
        logger.info(
            f"IndexBuilder initialized with embedding dimension: "
            f"{self.embedding_generator.get_embedding_dimension()}"
        )
    
    def build_product_index(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str,
        index_type: str = "flat",
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Build FAISS index for product embeddings.
        
        Extracts product-level data (name, tagline, description combined),
        preprocesses texts, generates embeddings, creates FAISS index,
        and saves index with metadata mapping.
        
        Args:
            dataset: List of product dictionaries
            output_path: Path to save index (without extension)
            index_type: FAISS index type ("flat", "ivf", "hnsw")
            batch_size: Batch size for embedding generation
            
        Returns:
            Dictionary with build statistics and metadata
            
        Raises:
            ValueError: If dataset is empty or invalid
            RuntimeError: If index building fails
            
        Example:
            >>> builder = IndexBuilder()
            >>> loader = DatasetLoader()
            >>> loader.load_dataset("dataset/products.jsonl", "products")
            >>> products = loader.extract_products()
            >>> result = builder.build_product_index(
            ...     products,
            ...     "data/indices/products",
            ...     index_type="flat"
            ... )
        """
        logger.info(
            f"Starting product index build with {len(dataset)} products, "
            f"index_type={index_type}, batch_size={batch_size}"
        )
        
        if not dataset:
            raise ValueError("Dataset is empty")
        
        start_time = time.time()
        
        try:
            # Step 1: Extract and preprocess product texts
            logger.info("Step 1/4: Extracting and preprocessing product texts")
            texts = []
            metadata_list = []
            
            for idx, product in enumerate(dataset):
                # Get combined text
                combined_text = product.get("combined_text", "")
                
                if not combined_text:
                    logger.warning(
                        f"Product {product.get('product_id')} has no text, skipping"
                    )
                    continue
                
                # Preprocess text
                preprocessed_text = self.preprocessor.preprocess(combined_text)
                
                if not preprocessed_text:
                    logger.warning(
                        f"Product {product.get('product_id')} preprocessing resulted "
                        f"in empty text, skipping"
                    )
                    continue
                
                texts.append(preprocessed_text)
                
                # Create metadata
                metadata = {
                    "vector_id": len(metadata_list),
                    "product_id": product.get("product_id"),
                    "review_id": None,  # Not applicable for products
                    "original_text": combined_text,
                    "name": product.get("name"),
                    "tagline": product.get("tagline"),
                    "description": product.get("description"),
                    "votesCount": product.get("votesCount", 0),
                    "commentsCount": product.get("commentsCount", 0),
                    "reviewsCount": product.get("reviewsCount", 0),
                    "createdAt": product.get("createdAt"),
                    "url": product.get("url"),
                    "topics": product.get("topics", []),
                    "thumbnail_url": product.get("thumbnail_url"),
                    "source": "product",
                    "timestamp": datetime.now().isoformat()
                }
                metadata_list.append(metadata)
                
                # Progress logging
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(dataset)} products")
            
            logger.info(
                f"Preprocessing complete: {len(texts)} products ready for embedding"
            )
            
            if not texts:
                raise ValueError("No valid texts after preprocessing")
            
            # Step 2: Generate embeddings
            logger.info(f"Step 2/4: Generating embeddings for {len(texts)} products")
            embeddings = self.embedding_generator.batch_generate(
                texts,
                batch_size=batch_size,
                show_progress=True
            )
            
            logger.info(
                f"Generated embeddings with shape: {embeddings.shape}"
            )
            
            # Step 3: Create FAISS index
            logger.info(f"Step 3/4: Creating FAISS index (type: {index_type})")
            embedding_dim = self.embedding_generator.get_embedding_dimension()
            index_manager = FAISSIndexManager(
                dimension=embedding_dim,
                index_type=index_type
            )
            
            # Create index with embeddings
            index_manager.create_index(embeddings, index_type=index_type)
            
            # Add metadata
            index_manager.metadata = metadata_list
            index_manager.vector_count = len(embeddings)
            
            logger.info(
                f"FAISS index created with {index_manager.get_index_size()} vectors"
            )
            
            # Step 4: Save index
            logger.info(f"Step 4/4: Saving index to {output_path}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            index_manager.save_index(output_path)
            
            # Validate index
            logger.info("Validating saved index...")
            validation_result = self._validate_index(output_path, len(texts))
            
            if not validation_result["valid"]:
                raise RuntimeError(
                    f"Index validation failed: {validation_result['error']}"
                )
            
            # Calculate statistics
            build_time = time.time() - start_time
            build_time_ms = build_time * 1000
            
            # Track index building time
            metrics_collector.track_operation("index_build_product", build_time_ms)
            metrics_collector.track_memory_usage()
            
            result = {
                "success": True,
                "index_type": index_type,
                "faiss_index_type": index_type,
                "total_products": len(dataset),
                "products_indexed": len(texts),
                "indexed_products": len(texts),
                "skipped_products": len(dataset) - len(texts),
                "embedding_dimension": embedding_dim,
                "index_size": index_manager.get_index_size(),
                "output_path": output_path,
                "build_time_seconds": build_time,
                "build_time_ms": build_time_ms,
                "build_time_formatted": f"{build_time:.2f}s",
                "validation": validation_result,
                "metadata": metadata_list,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                f"Product index build complete: {len(texts)} products indexed "
                f"in {build_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to build product index: {e}", exc_info=True)
            raise RuntimeError(f"Product index build failed: {e}")

    def build_review_index(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str,
        index_type: str = "flat",
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Build FAISS index for review embeddings.
        
        Extracts review-level data, preprocesses texts, generates embeddings,
        creates FAISS index, and saves index with metadata mapping including
        product_id associations.
        
        Args:
            dataset: List of review dictionaries
            output_path: Path to save index (without extension)
            index_type: FAISS index type ("flat", "ivf", "hnsw")
            batch_size: Batch size for embedding generation
            
        Returns:
            Dictionary with build statistics and metadata
            
        Raises:
            ValueError: If dataset is empty or invalid
            RuntimeError: If index building fails
            
        Example:
            >>> builder = IndexBuilder()
            >>> loader = DatasetLoader()
            >>> loader.load_dataset("dataset/reviews.jsonl", "reviews")
            >>> reviews = loader.extract_reviews()
            >>> result = builder.build_review_index(
            ...     reviews,
            ...     "data/indices/reviews",
            ...     index_type="flat"
            ... )
        """
        logger.info(
            f"Starting review index build with {len(dataset)} reviews, "
            f"index_type={index_type}, batch_size={batch_size}"
        )
        
        if not dataset:
            raise ValueError("Dataset is empty")
        
        start_time = time.time()
        
        try:
            # Step 1: Extract and preprocess review texts
            logger.info("Step 1/4: Extracting and preprocessing review texts")
            texts = []
            metadata_list = []
            
            for idx, review in enumerate(dataset):
                # Get review body
                body = review.get("body", "")
                
                if not body:
                    logger.warning(
                        f"Review {review.get('review_id')} has no body, skipping"
                    )
                    continue
                
                # Preprocess text
                preprocessed_text = self.preprocessor.preprocess(body)
                
                if not preprocessed_text:
                    logger.warning(
                        f"Review {review.get('review_id')} preprocessing resulted "
                        f"in empty text, skipping"
                    )
                    continue
                
                texts.append(preprocessed_text)
                
                # Create metadata
                metadata = {
                    "vector_id": len(metadata_list),
                    "product_id": review.get("product_id"),
                    "review_id": review.get("review_id"),
                    "original_text": body,
                    "votesCount": review.get("votesCount", 0),
                    "createdAt": review.get("createdAt"),
                    "user_id": review.get("user_id"),
                    "user_name": review.get("user_name"),
                    "user_username": review.get("user_username"),
                    "source": "review",
                    "timestamp": datetime.now().isoformat()
                }
                metadata_list.append(metadata)
                
                # Progress logging
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(dataset)} reviews")
            
            logger.info(
                f"Preprocessing complete: {len(texts)} reviews ready for embedding"
            )
            
            if not texts:
                raise ValueError("No valid texts after preprocessing")
            
            # Step 2: Generate embeddings
            logger.info(f"Step 2/4: Generating embeddings for {len(texts)} reviews")
            embeddings = self.embedding_generator.batch_generate(
                texts,
                batch_size=batch_size,
                show_progress=True
            )
            
            logger.info(
                f"Generated embeddings with shape: {embeddings.shape}"
            )
            
            # Step 3: Create FAISS index
            logger.info(f"Step 3/4: Creating FAISS index (type: {index_type})")
            embedding_dim = self.embedding_generator.get_embedding_dimension()
            index_manager = FAISSIndexManager(
                dimension=embedding_dim,
                index_type=index_type
            )
            
            # Create index with embeddings
            index_manager.create_index(embeddings, index_type=index_type)
            
            # Add metadata
            index_manager.metadata = metadata_list
            index_manager.vector_count = len(embeddings)
            
            logger.info(
                f"FAISS index created with {index_manager.get_index_size()} vectors"
            )
            
            # Step 4: Save index
            logger.info(f"Step 4/4: Saving index to {output_path}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            index_manager.save_index(output_path)
            
            # Validate index
            logger.info("Validating saved index...")
            validation_result = self._validate_index(output_path, len(texts))
            
            if not validation_result["valid"]:
                raise RuntimeError(
                    f"Index validation failed: {validation_result['error']}"
                )
            
            # Calculate statistics
            build_time = time.time() - start_time
            build_time_ms = build_time * 1000
            
            # Track index building time
            metrics_collector.track_operation("index_build_review", build_time_ms)
            metrics_collector.track_memory_usage()
            
            # Count unique products
            unique_products = len(set(
                m["product_id"] for m in metadata_list if m.get("product_id")
            ))
            
            result = {
                "success": True,
                "index_type": index_type,
                "faiss_index_type": index_type,
                "total_reviews": len(dataset),
                "reviews_indexed": len(texts),
                "indexed_reviews": len(texts),
                "skipped_reviews": len(dataset) - len(texts),
                "unique_products": unique_products,
                "embedding_dimension": embedding_dim,
                "index_size": index_manager.get_index_size(),
                "output_path": output_path,
                "build_time_seconds": build_time,
                "build_time_ms": build_time_ms,
                "build_time_formatted": f"{build_time:.2f}s",
                "validation": validation_result,
                "metadata": metadata_list,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                f"Review index build complete: {len(texts)} reviews indexed "
                f"in {build_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to build review index: {e}", exc_info=True)
            raise RuntimeError(f"Review index build failed: {e}")
    
    def build_all_indices(
        self,
        dataset_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate building both product and review indices from dataset.
        
        Loads dataset, extracts products and reviews, builds both indices,
        and returns comprehensive statistics.
        
        Args:
            dataset_path: Path to dataset directory or file
            config: Optional configuration dictionary with keys:
                - products_file: Path to products dataset (default: "products.jsonl")
                - reviews_file: Path to reviews dataset (default: "reviews.jsonl")
                - output_dir: Output directory for indices (default: "data/indices")
                - product_index_type: FAISS index type for products (default: "flat")
                - review_index_type: FAISS index type for reviews (default: "flat")
                - batch_size: Batch size for embeddings (default: 32)
            
        Returns:
            Dictionary with build results for both indices
            
        Raises:
            FileNotFoundError: If dataset files don't exist
            RuntimeError: If index building fails
            
        Example:
            >>> builder = IndexBuilder()
            >>> config = {
            ...     "products_file": "dataset/products.jsonl",
            ...     "reviews_file": "dataset/reviews.jsonl",
            ...     "output_dir": "data/indices",
            ...     "product_index_type": "flat",
            ...     "review_index_type": "flat",
            ...     "batch_size": 32
            ... }
            >>> results = builder.build_all_indices("dataset", config)
        """
        logger.info(f"Starting build_all_indices from dataset_path: {dataset_path}")
        
        # Set default config
        if config is None:
            config = {}
        
        # Determine dataset paths
        dataset_path_obj = Path(dataset_path)
        
        if dataset_path_obj.is_dir():
            # Dataset path is a directory
            products_file = config.get("products_file", "products.jsonl")
            reviews_file = config.get("reviews_file", "reviews.jsonl")
            
            products_path = str(dataset_path_obj / products_file)
            reviews_path = str(dataset_path_obj / reviews_file)
        else:
            # Assume separate paths provided in config
            products_path = config.get("products_file")
            reviews_path = config.get("reviews_file")
            
            if not products_path or not reviews_path:
                raise ValueError(
                    "If dataset_path is not a directory, must provide "
                    "products_file and reviews_file in config"
                )
        
        # Get other config parameters
        output_dir = config.get("output_dir", "data/indices")
        product_index_type = config.get("product_index_type", "flat")
        review_index_type = config.get("review_index_type", "flat")
        batch_size = config.get("batch_size", 32)
        
        logger.info(f"Configuration:")
        logger.info(f"  Products file: {products_path}")
        logger.info(f"  Reviews file: {reviews_path}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Product index type: {product_index_type}")
        logger.info(f"  Review index type: {review_index_type}")
        logger.info(f"  Batch size: {batch_size}")
        
        start_time = time.time()
        results = {
            "success": True,
            "product_index": None,
            "review_index": None,
            "products": None,
            "reviews": None,
            "total_time_seconds": 0,
            "total_build_time_ms": 0,
            "errors": []
        }
        
        try:
            # Load datasets
            logger.info("Loading datasets...")
            load_results = self.dataset_loader.load_multiple_datasets(
                products_path=products_path,
                reviews_path=reviews_path
            )
            
            # Check for loading errors
            if "products" in load_results and "error" in load_results["products"]:
                error_msg = f"Failed to load products: {load_results['products']['error']}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            
            if "reviews" in load_results and "error" in load_results["reviews"]:
                error_msg = f"Failed to load reviews: {load_results['reviews']['error']}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            
            # Build product index
            if self.dataset_loader.products_data:
                logger.info("\n" + "="*60)
                logger.info("Building product index...")
                logger.info("="*60)
                
                try:
                    products = self.dataset_loader.extract_products()
                    
                    if products:
                        product_output_path = str(Path(output_dir) / "products")
                        
                        product_result = self.build_product_index(
                            dataset=products,
                            output_path=product_output_path,
                            index_type=product_index_type,
                            batch_size=batch_size
                        )
                        
                        results["products"] = product_result
                        results["product_index"] = product_result
                        logger.info("Product index build successful")
                    else:
                        error_msg = "No products extracted from dataset"
                        logger.warning(error_msg)
                        results["errors"].append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Product index build failed: {e}"
                    logger.error(error_msg, exc_info=True)
                    results["errors"].append(error_msg)
            else:
                error_msg = "No product data loaded"
                logger.warning(error_msg)
                results["errors"].append(error_msg)
            
            # Build review index
            # Try to extract reviews from either reviews_data or products_data
            if self.dataset_loader.reviews_data or self.dataset_loader.products_data:
                logger.info("\n" + "="*60)
                logger.info("Building review index...")
                logger.info("="*60)
                
                try:
                    reviews = self.dataset_loader.extract_reviews()
                    
                    if reviews:
                        review_output_path = str(Path(output_dir) / "reviews")
                        
                        review_result = self.build_review_index(
                            dataset=reviews,
                            output_path=review_output_path,
                            index_type=review_index_type,
                            batch_size=batch_size
                        )
                        
                        results["reviews"] = review_result
                        results["review_index"] = review_result
                        logger.info("Review index build successful")
                    else:
                        error_msg = "No reviews extracted from dataset"
                        logger.warning(error_msg)
                        results["errors"].append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Review index build failed: {e}"
                    logger.error(error_msg, exc_info=True)
                    results["errors"].append(error_msg)
            else:
                error_msg = "No review data loaded"
                logger.warning(error_msg)
                results["errors"].append(error_msg)
            
            # Calculate total time
            total_time = time.time() - start_time
            results["total_time_seconds"] = total_time
            results["total_build_time_ms"] = total_time * 1000
            results["total_time_formatted"] = f"{total_time:.2f}s"
            
            # Summary
            logger.info("\n" + "="*60)
            logger.info("Index building complete!")
            logger.info("="*60)
            logger.info(f"Total time: {total_time:.2f}s")
            
            if results["products"]:
                logger.info(
                    f"Products indexed: {results['products']['indexed_products']} "
                    f"in {results['products']['build_time_formatted']}"
                )
            
            if results["reviews"]:
                logger.info(
                    f"Reviews indexed: {results['reviews']['indexed_reviews']} "
                    f"in {results['reviews']['build_time_formatted']}"
                )
            
            if results["errors"]:
                logger.warning(f"Encountered {len(results['errors'])} errors:")
                for error in results["errors"]:
                    logger.warning(f"  - {error}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to build indices: {e}", exc_info=True)
            raise RuntimeError(f"Index building failed: {e}")
    
    def _validate_index(
        self,
        index_path: str,
        expected_size: int
    ) -> Dict[str, Any]:
        """
        Validate that index was correctly built and is searchable.
        
        Args:
            index_path: Path to index (without extension)
            expected_size: Expected number of vectors
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Try to load the index
            test_manager = FAISSIndexManager(
                dimension=self.embedding_generator.get_embedding_dimension(),
                index_type="flat"  # Type doesn't matter for loading
            )
            
            test_manager.load_index(index_path)
            
            # Check size
            actual_size = test_manager.get_index_size()
            
            if actual_size != expected_size:
                return {
                    "valid": False,
                    "error": f"Size mismatch: expected {expected_size}, got {actual_size}",
                    "actual_size": actual_size,
                    "expected_size": expected_size
                }
            
            # Try a test search if index is not empty
            if actual_size > 0:
                # Create a random query vector
                import numpy as np
                query_vector = np.random.randn(
                    self.embedding_generator.get_embedding_dimension()
                ).astype(np.float32)
                
                # Normalize
                query_vector = query_vector / np.linalg.norm(query_vector)
                
                # Search for validation to get raw arrays
                distances, indices = test_manager.search(query_vector, k=min(5, actual_size))
                
                # Check that we got results
                if len(indices) == 0:
                    return {
                        "valid": False,
                        "error": "Search returned no results",
                        "actual_size": actual_size,
                        "expected_size": expected_size
                    }
            
            return {
                "valid": True,
                "actual_size": actual_size,
                "expected_size": expected_size,
                "searchable": True
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {e}",
                "actual_size": None,
                "expected_size": expected_size
            }
