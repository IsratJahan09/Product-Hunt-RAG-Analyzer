"""
RAG retrieval module for Product Hunt RAG Analyzer.

This module provides the RAGRetriever class for two-stage retrieval:
1. Identify competitor products based on product idea similarity
2. Retrieve relevant reviews and metadata for identified competitors

Supports filtering, ranking, and formatting context for LLM consumption.
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import numpy as np

from src.modules.embeddings import EmbeddingGenerator
from src.modules.vector_storage import FAISSIndexManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """
    Manages two-stage RAG retrieval for competitor analysis.
    
    Stage 1: Identify competitor products using product embeddings
    Stage 2: Retrieve relevant reviews for each competitor
    
    Provides methods for ranking, filtering, and formatting results for LLM input.
    """
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        product_index_manager: FAISSIndexManager,
        review_index_manager: FAISSIndexManager
    ):
        """
        Initialize RAGRetriever with embedding generator and FAISS indices.
        
        Args:
            embedding_generator: EmbeddingGenerator instance for text-to-vector conversion
            product_index_manager: FAISSIndexManager for product embeddings
            review_index_manager: FAISSIndexManager for review embeddings
            
        Raises:
            ValueError: If any component is None or invalid
            
        Example:
            >>> embedding_gen = EmbeddingGenerator()
            >>> product_index = FAISSIndexManager(dimension=384)
            >>> review_index = FAISSIndexManager(dimension=384)
            >>> retriever = RAGRetriever(embedding_gen, product_index, review_index)
        """
        if embedding_generator is None:
            raise ValueError("embedding_generator cannot be None")
        if product_index_manager is None:
            raise ValueError("product_index_manager cannot be None")
        if review_index_manager is None:
            raise ValueError("review_index_manager cannot be None")
        
        self.embedding_generator = embedding_generator
        self.product_index = product_index_manager
        self.review_index = review_index_manager
        
        logger.info(
            f"RAGRetriever initialized with "
            f"product_index_size={product_index_manager.get_index_size()}, "
            f"review_index_size={review_index_manager.get_index_size()}"
        )
    
    def identify_competitors(
        self,
        product_idea: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify top-K competitor products based on product idea similarity.
        
        Converts product idea to embedding and searches product FAISS index
        for most similar products.
        
        Args:
            product_idea: Text description of the product idea
            k: Number of competitors to identify (default: 5)
            
        Returns:
            List of competitor dictionaries with:
                - product_id: Product identifier
                - similarity_score: Similarity score (lower is more similar for L2)
                - metadata: Product metadata from index
            
        Raises:
            ValueError: If product_idea is empty or k is invalid
            RuntimeError: If FAISS search fails
            
        Example:
            >>> retriever = RAGRetriever(embedding_gen, product_idx, review_idx)
            >>> competitors = retriever.identify_competitors(
            ...     "A task management app with AI prioritization",
            ...     k=5
            ... )
            >>> len(competitors)
            5
        """
        if not product_idea or not product_idea.strip():
            raise ValueError("product_idea cannot be empty")
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        try:
            start_time = time.time()
            logger.info(
                f"Identifying top-{k} competitors for product idea: "
                f"'{product_idea[:100]}...'"
            )
            
            # Convert product idea to embedding
            product_embedding = self.embedding_generator.generate_embeddings(
                product_idea,
                normalize=True,
                show_progress=False
            )
            
            # Search product index for similar products
            distances, indices = self.product_index.search(
                product_embedding[0],
                k=k
            )
            
            # Flatten indices if needed (search returns 2D array)
            if len(indices.shape) > 1:
                indices = indices[0]
                distances = distances[0]
            
            # Retrieve metadata for identified competitors
            competitors_metadata = self.product_index.get_metadata(indices)
            
            # Build result list
            competitors = []
            for i, (distance, idx, metadata) in enumerate(
                zip(distances, indices, competitors_metadata)
            ):
                competitor = {
                    "rank": i + 1,
                    "product_id": metadata.get("product_id"),
                    "similarity_score": float(distance),
                    "metadata": metadata
                }
                competitors.append(competitor)
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Identified {len(competitors)} competitors in {duration_ms:.2f}ms"
            )
            
            # Log top competitors
            for comp in competitors[:3]:
                logger.debug(
                    f"  Rank {comp['rank']}: {comp['product_id']} "
                    f"(score: {comp['similarity_score']:.4f})"
                )
            
            return competitors
            
        except ValueError as e:
            logger.error(f"Invalid input for competitor identification: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to identify competitors: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Competitor identification failed: {e}")
    
    def retrieve_competitor_reviews(
        self,
        competitor_ids: List[str],
        k_per_competitor: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve top-K relevant reviews for each competitor.
        
        For each competitor, searches review FAISS index and filters
        reviews by product_id to ensure they belong to the competitor.
        
        Args:
            competitor_ids: List of competitor product IDs
            k_per_competitor: Number of reviews to retrieve per competitor (default: 10)
            
        Returns:
            Dictionary mapping product_id to list of review dictionaries with:
                - review_id: Review identifier
                - similarity_score: Similarity score
                - metadata: Review metadata including original_text, sentiment, etc.
            
        Raises:
            ValueError: If competitor_ids is empty or k_per_competitor is invalid
            RuntimeError: If review retrieval fails
            
        Example:
            >>> competitor_ids = ["product_1", "product_2"]
            >>> reviews = retriever.retrieve_competitor_reviews(
            ...     competitor_ids,
            ...     k_per_competitor=10
            ... )
            >>> len(reviews["product_1"])
            10
        """
        if not competitor_ids:
            raise ValueError("competitor_ids cannot be empty")
        
        if k_per_competitor <= 0:
            raise ValueError(f"k_per_competitor must be positive, got {k_per_competitor}")
        
        try:
            start_time = time.time()
            logger.info(
                f"Retrieving top-{k_per_competitor} reviews for "
                f"{len(competitor_ids)} competitors"
            )
            
            competitor_reviews = {}
            total_reviews_retrieved = 0
            
            for product_id in competitor_ids:
                # Get all metadata to filter by product_id
                # This is a simplified approach - in production, you'd want
                # a more efficient filtering mechanism
                
                # For now, we'll retrieve more reviews and filter
                # Retrieve k_per_competitor * 5 to account for filtering
                search_k = min(k_per_competitor * 5, self.review_index.get_index_size())
                
                if search_k == 0:
                    logger.warning(f"No reviews available for product {product_id}")
                    competitor_reviews[product_id] = []
                    continue
                
                # Create a dummy query vector (we'll improve this in production)
                # For now, use a random vector from the index
                # In a real implementation, you'd use the product description as query
                try:
                    # Get product metadata to create query
                    product_metadata = self._get_product_metadata_by_id(product_id)
                    
                    if product_metadata and product_metadata.get("original_text"):
                        # Use product text as query for relevant reviews
                        query_embedding = self.embedding_generator.generate_embeddings(
                            product_metadata["original_text"],
                            normalize=True,
                            show_progress=False
                        )
                        
                        # Search review index
                        distances, indices = self.review_index.search(
                            query_embedding[0],
                            k=search_k
                        )
                        
                        # Get metadata and filter by product_id
                        all_reviews_metadata = self.review_index.get_metadata(indices)
                        
                        filtered_reviews = []
                        for distance, idx, metadata in zip(
                            distances, indices, all_reviews_metadata
                        ):
                            if metadata.get("product_id") == product_id:
                                review = {
                                    "review_id": metadata.get("review_id"),
                                    "similarity_score": float(distance),
                                    "metadata": metadata
                                }
                                filtered_reviews.append(review)
                                
                                if len(filtered_reviews) >= k_per_competitor:
                                    break
                        
                        competitor_reviews[product_id] = filtered_reviews
                        total_reviews_retrieved += len(filtered_reviews)
                        
                        logger.debug(
                            f"Retrieved {len(filtered_reviews)} reviews for {product_id}"
                        )
                    else:
                        logger.warning(
                            f"No product metadata found for {product_id}, "
                            f"skipping review retrieval"
                        )
                        competitor_reviews[product_id] = []
                        
                except Exception as e:
                    logger.error(
                        f"Failed to retrieve reviews for {product_id}: {e}",
                        exc_info=True
                    )
                    competitor_reviews[product_id] = []
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Retrieved {total_reviews_retrieved} total reviews for "
                f"{len(competitor_ids)} competitors in {duration_ms:.2f}ms"
            )
            
            return competitor_reviews
            
        except ValueError as e:
            logger.error(f"Invalid input for review retrieval: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to retrieve competitor reviews: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Review retrieval failed: {e}")
    
    def _get_product_metadata_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Helper method to get product metadata by product_id.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product metadata dictionary or None if not found
        """
        # Search through product index metadata
        for metadata in self.product_index.metadata:
            if metadata.get("product_id") == product_id:
                return metadata
        return None
    
    def retrieve_competitor_metadata(
        self,
        competitor_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve product launch data for competitors.
        
        Gets metadata including name, tagline, description, votes,
        comments, created_at from product index.
        
        Args:
            competitor_ids: List of competitor product IDs
            
        Returns:
            Dictionary mapping product_id to metadata dictionary
            
        Raises:
            ValueError: If competitor_ids is empty
            
        Example:
            >>> competitor_ids = ["product_1", "product_2"]
            >>> metadata = retriever.retrieve_competitor_metadata(competitor_ids)
            >>> metadata["product_1"]["name"]
            'Competitor Product Name'
        """
        if not competitor_ids:
            raise ValueError("competitor_ids cannot be empty")
        
        try:
            logger.info(f"Retrieving metadata for {len(competitor_ids)} competitors")
            
            competitor_metadata = {}
            found_count = 0
            
            for product_id in competitor_ids:
                metadata = self._get_product_metadata_by_id(product_id)
                
                if metadata:
                    competitor_metadata[product_id] = metadata
                    found_count += 1
                else:
                    logger.warning(f"No metadata found for product {product_id}")
                    competitor_metadata[product_id] = {
                        "product_id": product_id,
                        "error": "Metadata not found"
                    }
            
            logger.info(
                f"Retrieved metadata for {found_count}/{len(competitor_ids)} competitors"
            )
            
            return competitor_metadata
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve competitor metadata: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Metadata retrieval failed: {e}")

    def rank_results(
        self,
        results: List[Dict[str, Any]],
        scores: Optional[List[float]] = None,
        recency_weight: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rank retrieved data by relevance score with optional recency weighting.
        
        Args:
            results: List of result dictionaries with metadata
            scores: Optional list of similarity scores (if not in results)
            recency_weight: Weight for recency in ranking (0.0 = no recency, 1.0 = only recency)
            
        Returns:
            Sorted list of results by combined score
            
        Raises:
            ValueError: If results is empty or recency_weight is invalid
            
        Example:
            >>> results = [{"similarity_score": 0.5, "metadata": {...}}, ...]
            >>> ranked = retriever.rank_results(results, recency_weight=0.2)
        """
        if not results:
            logger.warning("Empty results provided for ranking")
            return []
        
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError(
                f"recency_weight must be between 0.0 and 1.0, got {recency_weight}"
            )
        
        try:
            logger.debug(
                f"Ranking {len(results)} results with recency_weight={recency_weight}"
            )
            
            # If scores provided separately, add them to results
            if scores is not None:
                if len(scores) != len(results):
                    raise ValueError(
                        f"Length mismatch: {len(scores)} scores for {len(results)} results"
                    )
                for result, score in zip(results, scores):
                    if "similarity_score" not in result:
                        result["similarity_score"] = score
            
            # Calculate combined scores
            for result in results:
                similarity_score = result.get("similarity_score", 0.0)
                
                # Calculate recency score if timestamp available
                recency_score = 0.0
                if recency_weight > 0.0:
                    metadata = result.get("metadata", {})
                    timestamp_str = metadata.get("timestamp") or metadata.get("created_at")
                    
                    if timestamp_str:
                        try:
                            # Parse timestamp and calculate recency
                            # More recent = higher score
                            timestamp = datetime.fromisoformat(
                                timestamp_str.replace('Z', '+00:00')
                            )
                            now = datetime.now(timestamp.tzinfo)
                            days_old = (now - timestamp).days
                            
                            # Exponential decay: score decreases with age
                            # Half-life of 30 days
                            recency_score = np.exp(-days_old / 30.0)
                        except Exception as e:
                            logger.debug(f"Failed to parse timestamp: {e}")
                            recency_score = 0.0
                
                # Combined score (lower similarity score is better for L2 distance)
                # Normalize similarity to 0-1 range and invert
                normalized_similarity = 1.0 / (1.0 + similarity_score)
                
                combined_score = (
                    (1.0 - recency_weight) * normalized_similarity +
                    recency_weight * recency_score
                )
                
                result["combined_score"] = combined_score
            
            # Sort by combined score (higher is better)
            ranked_results = sorted(
                results,
                key=lambda x: x.get("combined_score", 0.0),
                reverse=True
            )
            
            logger.debug(f"Ranked {len(ranked_results)} results")
            
            return ranked_results
            
        except ValueError as e:
            logger.error(f"Invalid input for ranking: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to rank results: {e}", exc_info=True)
            raise RuntimeError(f"Ranking failed: {e}")
    
    def filter_by_sentiment(
        self,
        results: List[Dict[str, Any]],
        sentiment: str
    ) -> List[Dict[str, Any]]:
        """
        Filter results by sentiment type.
        
        Args:
            results: List of result dictionaries with sentiment in metadata
            sentiment: Sentiment to filter by ("positive", "negative", "neutral")
            
        Returns:
            Filtered list of results matching the sentiment
            
        Raises:
            ValueError: If sentiment is invalid
            
        Example:
            >>> results = [{"metadata": {"sentiment": "positive"}}, ...]
            >>> positive_results = retriever.filter_by_sentiment(results, "positive")
        """
        valid_sentiments = ["positive", "negative", "neutral"]
        if sentiment not in valid_sentiments:
            raise ValueError(
                f"Invalid sentiment '{sentiment}'. Must be one of {valid_sentiments}"
            )
        
        if not results:
            logger.warning("Empty results provided for sentiment filtering")
            return []
        
        try:
            logger.debug(
                f"Filtering {len(results)} results by sentiment: {sentiment}"
            )
            
            filtered_results = []
            for result in results:
                metadata = result.get("metadata", {})
                result_sentiment = metadata.get("sentiment", "").lower()
                
                if result_sentiment == sentiment.lower():
                    filtered_results.append(result)
            
            logger.info(
                f"Filtered to {len(filtered_results)}/{len(results)} results "
                f"with sentiment={sentiment}"
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(
                f"Failed to filter by sentiment: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Sentiment filtering failed: {e}")
    
    def filter_by_confidence(
        self,
        results: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Filter results by minimum confidence threshold.
        
        Args:
            results: List of result dictionaries with confidence in metadata
            threshold: Minimum confidence threshold (0.0 to 1.0)
            
        Returns:
            Filtered list of results meeting the confidence threshold
            
        Raises:
            ValueError: If threshold is invalid
            
        Example:
            >>> results = [{"metadata": {"confidence": 0.85}}, ...]
            >>> high_conf = retriever.filter_by_confidence(results, threshold=0.7)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be between 0.0 and 1.0, got {threshold}"
            )
        
        if not results:
            logger.warning("Empty results provided for confidence filtering")
            return []
        
        try:
            logger.debug(
                f"Filtering {len(results)} results by confidence >= {threshold}"
            )
            
            filtered_results = []
            for result in results:
                metadata = result.get("metadata", {})
                confidence = metadata.get("confidence", 0.0)
                
                if confidence >= threshold:
                    filtered_results.append(result)
            
            logger.info(
                f"Filtered to {len(filtered_results)}/{len(results)} results "
                f"with confidence >= {threshold}"
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(
                f"Failed to filter by confidence: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Confidence filtering failed: {e}")
    
    def format_context_for_llm(
        self,
        product_idea: str,
        competitors_data: List[Dict[str, Any]]
    ) -> str:
        """
        Prepare structured context string for LLM input.
        
        Combines product idea with competitor metadata and reviews into
        a formatted string suitable for LLM consumption.
        
        Args:
            product_idea: Text description of the product idea
            competitors_data: List of competitor dictionaries with metadata and reviews
            
        Returns:
            Formatted context string for LLM
            
        Raises:
            ValueError: If product_idea is empty
            
        Example:
            >>> context = retriever.format_context_for_llm(
            ...     "A task management app",
            ...     competitors_data
            ... )
            >>> print(context[:100])
            Product Idea: A task management app...
        """
        if not product_idea or not product_idea.strip():
            raise ValueError("product_idea cannot be empty")
        
        try:
            logger.info(
                f"Formatting context for LLM with {len(competitors_data)} competitors"
            )
            
            # Build context string
            context_parts = []
            
            # Add product idea
            context_parts.append("=== PRODUCT IDEA ===")
            context_parts.append(product_idea.strip())
            context_parts.append("")
            
            # Add competitor data
            context_parts.append("=== COMPETITOR ANALYSIS ===")
            context_parts.append(f"Total Competitors Analyzed: {len(competitors_data)}")
            context_parts.append("")
            
            for i, competitor in enumerate(competitors_data, 1):
                context_parts.append(f"--- Competitor {i} ---")
                
                # Add competitor metadata
                metadata = competitor.get("metadata", {})
                product_id = competitor.get("product_id", "Unknown")
                similarity_score = competitor.get("similarity_score", 0.0)
                
                context_parts.append(f"Product ID: {product_id}")
                context_parts.append(f"Similarity Score: {similarity_score:.4f}")
                
                # Add product details if available
                if "name" in metadata:
                    context_parts.append(f"Name: {metadata['name']}")
                if "tagline" in metadata:
                    context_parts.append(f"Tagline: {metadata['tagline']}")
                if "description" in metadata:
                    context_parts.append(f"Description: {metadata['description']}")
                if "votes_count" in metadata or "votesCount" in metadata:
                    votes = metadata.get("votes_count") or metadata.get("votesCount")
                    context_parts.append(f"Votes: {votes}")
                if "comments_count" in metadata or "commentsCount" in metadata:
                    comments = metadata.get("comments_count") or metadata.get("commentsCount")
                    context_parts.append(f"Comments: {comments}")
                
                # Add reviews if available
                reviews = competitor.get("reviews", [])
                if reviews:
                    context_parts.append(f"\nReviews ({len(reviews)}):")
                    
                    for j, review in enumerate(reviews[:10], 1):  # Limit to 10 reviews
                        review_metadata = review.get("metadata", {})
                        review_text = review_metadata.get("original_text", "")
                        sentiment = review_metadata.get("sentiment", "unknown")
                        confidence = review_metadata.get("confidence", 0.0)
                        
                        if review_text:
                            # Truncate long reviews
                            if len(review_text) > 200:
                                review_text = review_text[:200] + "..."
                            
                            context_parts.append(
                                f"  {j}. [{sentiment.upper()}, conf={confidence:.2f}] "
                                f"{review_text}"
                            )
                else:
                    context_parts.append("\nNo reviews available")
                
                context_parts.append("")
            
            # Join all parts
            context = "\n".join(context_parts)
            
            logger.info(
                f"Formatted context: {len(context)} characters, "
                f"{len(competitors_data)} competitors"
            )
            
            return context
            
        except ValueError as e:
            logger.error(f"Invalid input for context formatting: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to format context for LLM: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Context formatting failed: {e}")
    
    def __repr__(self) -> str:
        """String representation of RAGRetriever."""
        return (
            f"RAGRetriever("
            f"product_index_size={self.product_index.get_index_size()}, "
            f"review_index_size={self.review_index.get_index_size()})"
        )
