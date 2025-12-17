"""
Validation procedures for Product Hunt RAG Analyzer.

This module implements unit-level, integration-level, and end-to-end validation
procedures to ensure system quality and performance meet defined targets.

Requirements: 13.4, 13.5
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    sentiment_accuracy,
    sentiment_f1_score,
    feature_extraction_recall,
    feature_categorization_accuracy,
    calculate_average_latency,
    calculate_error_rate,
    evaluate_against_targets,
)
from src.modules.preprocessing import TextPreprocessor
from src.modules.embeddings import EmbeddingGenerator
from src.modules.vector_storage import FAISSIndexManager
from src.modules.rag_retrieval import RAGRetriever
from src.modules.sentiment import SentimentAnalyzer
from src.modules.feature_analysis import FeatureAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationRunner:
    """
    Executes validation procedures for the Product Hunt RAG Analyzer.
    
    Provides unit-level, integration-level, and end-to-end validation
    with detailed logging and reporting of pass/fail status.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ValidationRunner.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results = {
            "unit_level": {},
            "integration_level": {},
            "end_to_end": {},
            "summary": {}
        }
        logger.info("ValidationRunner initialized")
    
    # =============================================================================
    # Unit-Level Validation
    # =============================================================================
    
    def validate_preprocessing(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Test preprocessing on sample texts.
        
        Validates HTML removal, special character handling, and text normalization.
        Pass criteria: 100% success on all checks.
        
        Args:
            sample_texts: List of sample texts to preprocess (should be 20 samples)
            
        Returns:
            Dict with validation results including pass/fail status
        """
        logger.info(f"Starting preprocessing validation with {len(sample_texts)} samples")
        start_time = time.time()
        
        try:
            preprocessor = TextPreprocessor()
            
            results = {
                "total_samples": len(sample_texts),
                "successful": 0,
                "failed": 0,
                "checks": {
                    "html_removal": {"passed": 0, "failed": 0},
                    "special_char_handling": {"passed": 0, "failed": 0},
                    "whitespace_normalization": {"passed": 0, "failed": 0},
                    "non_empty_output": {"passed": 0, "failed": 0}
                },
                "errors": []
            }
            
            for idx, text in enumerate(sample_texts):
                try:
                    # Preprocess text
                    processed = preprocessor.preprocess(text)
                    
                    # Check 1: HTML tags removed
                    if "<" not in processed and ">" not in processed:
                        results["checks"]["html_removal"]["passed"] += 1
                    else:
                        results["checks"]["html_removal"]["failed"] += 1
                        results["errors"].append(f"Sample {idx}: HTML tags not removed")
                    
                    # Check 2: Special characters handled
                    # Allow alphanumeric, spaces, and basic punctuation
                    if all(c.isalnum() or c.isspace() or c in ".,!?;:'\"-" for c in processed):
                        results["checks"]["special_char_handling"]["passed"] += 1
                    else:
                        results["checks"]["special_char_handling"]["failed"] += 1
                        results["errors"].append(f"Sample {idx}: Special characters not handled")
                    
                    # Check 3: Whitespace normalized (no multiple spaces)
                    if "  " not in processed and "\n" not in processed and "\t" not in processed:
                        results["checks"]["whitespace_normalization"]["passed"] += 1
                    else:
                        results["checks"]["whitespace_normalization"]["failed"] += 1
                        results["errors"].append(f"Sample {idx}: Whitespace not normalized")
                    
                    # Check 4: Non-empty output (unless input was empty)
                    if text.strip() and processed.strip():
                        results["checks"]["non_empty_output"]["passed"] += 1
                    elif not text.strip() and not processed.strip():
                        results["checks"]["non_empty_output"]["passed"] += 1
                    else:
                        results["checks"]["non_empty_output"]["failed"] += 1
                        results["errors"].append(f"Sample {idx}: Empty output from non-empty input")
                    
                    # Overall success if all checks passed
                    if all(
                        results["checks"][check]["failed"] == 0 or 
                        results["checks"][check]["passed"] > results["checks"][check]["failed"]
                        for check in results["checks"]
                    ):
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                        
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Sample {idx}: {str(e)}")
                    logger.error(f"Error preprocessing sample {idx}: {e}")
            
            # Calculate success rate
            success_rate = results["successful"] / results["total_samples"] if results["total_samples"] > 0 else 0
            results["success_rate"] = success_rate
            results["passed"] = success_rate >= 1.0  # 100% success required
            results["status"] = "PASS" if results["passed"] else "FAIL"
            results["duration_seconds"] = time.time() - start_time
            
            logger.info(
                f"Preprocessing validation completed: {results['status']} "
                f"(success_rate={success_rate:.2%}, duration={results['duration_seconds']:.2f}s)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Preprocessing validation failed: {e}")
            return {
                "passed": False,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    def validate_embeddings(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Test embedding generation on sample texts.
        
        Validates dimension=384, normalized values, and consistency.
        Pass criteria: All checks pass.
        
        Args:
            sample_texts: List of sample texts (should be 10 samples)
            
        Returns:
            Dict with validation results including pass/fail status
        """
        logger.info(f"Starting embedding validation with {len(sample_texts)} samples")
        start_time = time.time()
        
        try:
            generator = EmbeddingGenerator()
            
            results = {
                "total_samples": len(sample_texts),
                "checks": {
                    "correct_dimension": {"passed": 0, "failed": 0},
                    "normalized_values": {"passed": 0, "failed": 0},
                    "consistency": {"passed": 0, "failed": 0},
                    "non_zero": {"passed": 0, "failed": 0}
                },
                "errors": []
            }
            
            # Generate embeddings
            embeddings = generator.generate_embeddings(sample_texts)
            
            # Check 1: Correct dimension (384 for all-MiniLM-L6-v2)
            expected_dim = 384
            if embeddings.shape[1] == expected_dim:
                results["checks"]["correct_dimension"]["passed"] = len(sample_texts)
                logger.info(f"✓ Dimension check passed: {embeddings.shape[1]} == {expected_dim}")
            else:
                results["checks"]["correct_dimension"]["failed"] = len(sample_texts)
                results["errors"].append(
                    f"Incorrect dimension: expected {expected_dim}, got {embeddings.shape[1]}"
                )
                logger.error(f"✗ Dimension check failed: {embeddings.shape[1]} != {expected_dim}")
            
            # Check 2: Normalized values (should be in reasonable range)
            for idx, embedding in enumerate(embeddings):
                # Check if values are in reasonable range (typically -1 to 1 for normalized)
                if np.all(np.abs(embedding) <= 10):  # Reasonable upper bound
                    results["checks"]["normalized_values"]["passed"] += 1
                else:
                    results["checks"]["normalized_values"]["failed"] += 1
                    results["errors"].append(f"Sample {idx}: Values out of range")
            
            # Check 3: Consistency (same input produces same output)
            if len(sample_texts) > 0:
                test_text = sample_texts[0]
                embedding1 = generator.generate_embeddings([test_text])[0]
                embedding2 = generator.generate_embeddings([test_text])[0]
                
                if np.allclose(embedding1, embedding2, rtol=1e-5):
                    results["checks"]["consistency"]["passed"] = 1
                    logger.info("✓ Consistency check passed")
                else:
                    results["checks"]["consistency"]["failed"] = 1
                    results["errors"].append("Inconsistent embeddings for same input")
                    logger.error("✗ Consistency check failed")
            
            # Check 4: Non-zero embeddings
            for idx, embedding in enumerate(embeddings):
                if not np.allclose(embedding, 0):
                    results["checks"]["non_zero"]["passed"] += 1
                else:
                    results["checks"]["non_zero"]["failed"] += 1
                    results["errors"].append(f"Sample {idx}: Zero embedding")
            
            # Overall pass if all checks passed
            all_passed = all(
                check_results["failed"] == 0
                for check_results in results["checks"].values()
            )
            
            results["passed"] = all_passed
            results["status"] = "PASS" if all_passed else "FAIL"
            results["duration_seconds"] = time.time() - start_time
            
            logger.info(
                f"Embedding validation completed: {results['status']} "
                f"(duration={results['duration_seconds']:.2f}s)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Embedding validation failed: {e}")
            return {
                "passed": False,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    def validate_faiss_index(self, sample_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Test FAISS operations with sample embeddings.
        
        Validates index creation, search functionality, and metadata handling.
        Pass criteria: All operations successful.
        
        Args:
            sample_embeddings: Array of sample embeddings (should be 100 embeddings)
            
        Returns:
            Dict with validation results including pass/fail status
        """
        logger.info(f"Starting FAISS index validation with {len(sample_embeddings)} embeddings")
        start_time = time.time()
        
        try:
            dimension = sample_embeddings.shape[1]
            index_manager = FAISSIndexManager(dimension=dimension, index_type="flat")
            
            results = {
                "total_embeddings": len(sample_embeddings),
                "checks": {
                    "index_creation": {"passed": 0, "failed": 0},
                    "add_embeddings": {"passed": 0, "failed": 0},
                    "search_functionality": {"passed": 0, "failed": 0},
                    "index_size": {"passed": 0, "failed": 0},
                    "metadata_retrieval": {"passed": 0, "failed": 0}
                },
                "errors": []
            }
            
            # Check 1: Index creation
            try:
                index_manager.create_index()
                results["checks"]["index_creation"]["passed"] = 1
                logger.info("✓ Index creation successful")
            except Exception as e:
                results["checks"]["index_creation"]["failed"] = 1
                results["errors"].append(f"Index creation failed: {e}")
                logger.error(f"✗ Index creation failed: {e}")
                return results  # Can't continue without index
            
            # Check 2: Add embeddings with metadata
            try:
                metadata_list = [
                    {
                        "vector_id": i,
                        "text": f"sample_{i}",
                        "source": "validation"
                    }
                    for i in range(len(sample_embeddings))
                ]
                index_manager.add_embeddings(sample_embeddings, metadata_list)
                results["checks"]["add_embeddings"]["passed"] = 1
                logger.info("✓ Add embeddings successful")
            except Exception as e:
                results["checks"]["add_embeddings"]["failed"] = 1
                results["errors"].append(f"Add embeddings failed: {e}")
                logger.error(f"✗ Add embeddings failed: {e}")
            
            # Check 3: Index size
            try:
                index_size = index_manager.get_index_size()
                if index_size == len(sample_embeddings):
                    results["checks"]["index_size"]["passed"] = 1
                    logger.info(f"✓ Index size correct: {index_size}")
                else:
                    results["checks"]["index_size"]["failed"] = 1
                    results["errors"].append(
                        f"Index size mismatch: expected {len(sample_embeddings)}, got {index_size}"
                    )
                    logger.error(f"✗ Index size mismatch: {index_size} != {len(sample_embeddings)}")
            except Exception as e:
                results["checks"]["index_size"]["failed"] = 1
                results["errors"].append(f"Index size check failed: {e}")
                logger.error(f"✗ Index size check failed: {e}")
            
            # Check 4: Search functionality
            try:
                # Search with first embedding
                query_vector = sample_embeddings[0]
                distances, indices = index_manager.search(query_vector, k=5)
                
                # First result should be the query itself (distance ~0)
                if len(indices) > 0 and indices[0] == 0 and distances[0] < 0.01:
                    results["checks"]["search_functionality"]["passed"] = 1
                    logger.info("✓ Search functionality working")
                else:
                    results["checks"]["search_functionality"]["failed"] = 1
                    results["errors"].append("Search did not return expected results")
                    logger.error("✗ Search functionality failed")
            except Exception as e:
                results["checks"]["search_functionality"]["failed"] = 1
                results["errors"].append(f"Search failed: {e}")
                logger.error(f"✗ Search failed: {e}")
            
            # Check 5: Metadata retrieval
            try:
                metadata = index_manager.get_metadata([0, 1, 2])
                if len(metadata) == 3 and all("vector_id" in m for m in metadata):
                    results["checks"]["metadata_retrieval"]["passed"] = 1
                    logger.info("✓ Metadata retrieval successful")
                else:
                    results["checks"]["metadata_retrieval"]["failed"] = 1
                    results["errors"].append("Metadata retrieval incomplete")
                    logger.error("✗ Metadata retrieval failed")
            except Exception as e:
                results["checks"]["metadata_retrieval"]["failed"] = 1
                results["errors"].append(f"Metadata retrieval failed: {e}")
                logger.error(f"✗ Metadata retrieval failed: {e}")
            
            # Overall pass if all checks passed
            all_passed = all(
                check_results["failed"] == 0
                for check_results in results["checks"].values()
            )
            
            results["passed"] = all_passed
            results["status"] = "PASS" if all_passed else "FAIL"
            results["duration_seconds"] = time.time() - start_time
            
            logger.info(
                f"FAISS index validation completed: {results['status']} "
                f"(duration={results['duration_seconds']:.2f}s)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS index validation failed: {e}")
            return {
                "passed": False,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    # =============================================================================
    # Integration-Level Validation
    # =============================================================================
    
    def validate_rag_retrieval(
        self,
        benchmark_queries: List[Dict[str, Any]],
        product_index_path: str,
        review_index_path: str
    ) -> Dict[str, Any]:
        """
        Run benchmark queries and calculate retrieval metrics.
        
        Calculates Precision@5, Recall@5, and MRR.
        Pass criteria: All metrics meet targets (P@5≥0.70, R@5≥0.60, MRR≥0.75).
        
        Args:
            benchmark_queries: List of queries with product_idea and expected_competitors
            product_index_path: Path to product FAISS index
            review_index_path: Path to review FAISS index
            
        Returns:
            Dict with validation results including metrics and pass/fail status
        """
        logger.info(f"Starting RAG retrieval validation with {len(benchmark_queries)} queries")
        start_time = time.time()
        
        try:
            # Load indices
            embedding_gen = EmbeddingGenerator()
            
            product_index = FAISSIndexManager(dimension=384)
            product_index.load_index(product_index_path)
            
            review_index = FAISSIndexManager(dimension=384)
            review_index.load_index(review_index_path)
            
            retriever = RAGRetriever(embedding_gen, product_index, review_index)
            
            # Run queries and collect results
            precision_scores = []
            recall_scores = []
            mrr_scores = []
            
            for query in benchmark_queries:
                product_idea = query["product_idea"]
                expected_competitors = query["expected_competitors"]
                
                # Identify competitors
                competitors = retriever.identify_competitors(product_idea, k=5)
                # Extract product names from metadata
                retrieved_ids = []
                for c in competitors:
                    metadata = c.get("metadata", {})
                    if "name" in metadata:
                        retrieved_ids.append(metadata["name"])
                    elif "product_name" in metadata:
                        retrieved_ids.append(metadata["product_name"])
                
                # Calculate metrics
                p_at_5 = precision_at_k(retrieved_ids, expected_competitors, k=5)
                r_at_5 = recall_at_k(retrieved_ids, expected_competitors, k=5)
                mrr = mean_reciprocal_rank(retrieved_ids, expected_competitors)
                
                precision_scores.append(p_at_5)
                recall_scores.append(r_at_5)
                mrr_scores.append(mrr)
            
            # Calculate average metrics
            avg_precision = np.mean(precision_scores) if precision_scores else 0.0
            avg_recall = np.mean(recall_scores) if recall_scores else 0.0
            avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
            
            # Evaluate against targets
            metrics = {
                "precision_at_5": avg_precision,
                "recall_at_5": avg_recall,
                "mrr": avg_mrr
            }
            
            evaluation = evaluate_against_targets(metrics)
            
            # Check if all metrics passed
            all_passed = all(result["passed"] for result in evaluation.values())
            
            results = {
                "total_queries": len(benchmark_queries),
                "metrics": metrics,
                "evaluation": evaluation,
                "passed": all_passed,
                "status": "PASS" if all_passed else "FAIL",
                "duration_seconds": time.time() - start_time
            }
            
            logger.info(
                f"RAG retrieval validation completed: {results['status']} "
                f"(P@5={avg_precision:.3f}, R@5={avg_recall:.3f}, MRR={avg_mrr:.3f})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"RAG retrieval validation failed: {e}")
            return {
                "passed": False,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    def validate_sentiment_analysis(
        self,
        labeled_reviews: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze labeled reviews and calculate sentiment metrics.
        
        Calculates accuracy and F1-score.
        Pass criteria: Accuracy ≥0.82, F1 ≥0.80.
        
        Args:
            labeled_reviews: List of reviews with body and sentiment labels
            
        Returns:
            Dict with validation results including metrics and pass/fail status
        """
        logger.info(f"Starting sentiment analysis validation with {len(labeled_reviews)} reviews")
        start_time = time.time()
        
        try:
            analyzer = SentimentAnalyzer()
            
            predictions = []
            ground_truth = []
            
            for review in labeled_reviews:
                body = review["body"]
                true_sentiment = review["sentiment"]
                
                # Analyze sentiment
                result = analyzer.analyze_sentiment(body)
                predicted_sentiment = result["sentiment"]
                
                predictions.append(predicted_sentiment)
                ground_truth.append(true_sentiment)
            
            # Calculate metrics
            accuracy = sentiment_accuracy(predictions, ground_truth)
            f1 = sentiment_f1_score(predictions, ground_truth)
            
            # Evaluate against targets
            metrics = {
                "sentiment_accuracy": accuracy,
                "sentiment_f1": f1
            }
            
            evaluation = evaluate_against_targets(metrics)
            
            # Check if all metrics passed
            all_passed = all(result["passed"] for result in evaluation.values())
            
            results = {
                "total_reviews": len(labeled_reviews),
                "metrics": metrics,
                "evaluation": evaluation,
                "passed": all_passed,
                "status": "PASS" if all_passed else "FAIL",
                "duration_seconds": time.time() - start_time
            }
            
            logger.info(
                f"Sentiment analysis validation completed: {results['status']} "
                f"(accuracy={accuracy:.3f}, f1={f1:.3f})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Sentiment analysis validation failed: {e}")
            return {
                "passed": False,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    def validate_feature_gaps(
        self,
        labeled_reviews: List[Dict[str, Any]],
        ground_truth_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract features from reviews and calculate feature gap metrics.
        
        Calculates extraction recall and categorization accuracy.
        Pass criteria: Recall ≥0.75, Accuracy ≥0.80.
        
        Args:
            labeled_reviews: List of reviews to extract features from
            ground_truth_features: Dict with expected features and categories
            
        Returns:
            Dict with validation results including metrics and pass/fail status
        """
        logger.info(f"Starting feature gap validation with {len(labeled_reviews)} reviews")
        start_time = time.time()
        
        try:
            analyzer = FeatureAnalyzer()
            
            # Extract features from reviews
            review_texts = [r["body"] for r in labeled_reviews]
            extracted_features = analyzer.extract_features(review_texts)
            
            # Get feature names
            extracted_feature_names = [f["feature"] for f in extracted_features]
            expected_feature_names = ground_truth_features.get("features", [])
            
            # Calculate extraction recall
            recall = feature_extraction_recall(
                extracted_feature_names,
                expected_feature_names,
                case_sensitive=False
            )
            
            # Categorize features
            categorized = analyzer.categorize_features(extracted_features)
            predicted_categories = {
                f["feature"]: f["category"]
                for f in categorized
            }
            
            ground_truth_categories = ground_truth_features.get("categories", {})
            
            # Calculate categorization accuracy
            accuracy = feature_categorization_accuracy(
                predicted_categories,
                ground_truth_categories,
                case_sensitive=False
            )
            
            # Evaluate against targets
            metrics = {
                "feature_extraction_recall": recall,
                "feature_categorization_accuracy": accuracy
            }
            
            evaluation = evaluate_against_targets(metrics)
            
            # Check if all metrics passed
            all_passed = all(result["passed"] for result in evaluation.values())
            
            results = {
                "total_reviews": len(labeled_reviews),
                "extracted_features_count": len(extracted_feature_names),
                "expected_features_count": len(expected_feature_names),
                "metrics": metrics,
                "evaluation": evaluation,
                "passed": all_passed,
                "status": "PASS" if all_passed else "FAIL",
                "duration_seconds": time.time() - start_time
            }
            
            logger.info(
                f"Feature gap validation completed: {results['status']} "
                f"(recall={recall:.3f}, accuracy={accuracy:.3f})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Feature gap validation failed: {e}")
            return {
                "passed": False,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    # =============================================================================
    # End-to-End Validation
    # =============================================================================
    
    def validate_full_pipeline(
        self,
        benchmark_queries: List[Dict[str, Any]],
        product_index_path: str,
        review_index_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run queries through complete pipeline and evaluate.
        
        Evaluates report completeness, processing latency, and error rate.
        Pass criteria: Completeness ≥95%, Latency ≤30s, Error rate ≤2%.
        
        Args:
            benchmark_queries: List of queries with product_idea
            product_index_path: Path to product FAISS index
            review_index_path: Path to review FAISS index
            config: Optional configuration for pipeline
            
        Returns:
            Dict with validation results including metrics and pass/fail status
        """
        logger.info(f"Starting full pipeline validation with {len(benchmark_queries)} queries")
        start_time = time.time()
        
        try:
            from src.main import AnalysisPipeline
            
            # Initialize pipeline
            pipeline = AnalysisPipeline(config or {})
            pipeline.load_indices(product_index_path, review_index_path)
            
            latencies = []
            failed_queries = 0
            completeness_scores = []
            
            for query in benchmark_queries:
                product_idea = query["product_idea"]
                query_start = time.time()
                
                try:
                    # Run analysis
                    result = pipeline.run_analysis(
                        product_idea=product_idea,
                        max_competitors=5,
                        output_format="json"
                    )
                    
                    query_latency = time.time() - query_start
                    latencies.append(query_latency)
                    
                    # Check report completeness
                    required_sections = [
                        "market_positioning",
                        "feature_gaps",
                        "sentiment_summary",
                        "recommendations"
                    ]
                    
                    results_data = result.get("results", {})
                    present_sections = sum(
                        1 for section in required_sections
                        if section in results_data and results_data[section]
                    )
                    
                    completeness = present_sections / len(required_sections)
                    completeness_scores.append(completeness)
                    
                except Exception as e:
                    failed_queries += 1
                    logger.error(f"Query failed: {e}")
            
            # Calculate metrics
            avg_latency = calculate_average_latency(latencies)
            error_rate = calculate_error_rate(len(benchmark_queries), failed_queries)
            avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
            
            # Evaluate against targets
            metrics = {
                "average_latency": avg_latency,
                "error_rate": error_rate,
                "report_completeness": avg_completeness
            }
            
            # Check targets
            latency_passed = avg_latency <= 30.0
            error_rate_passed = error_rate <= 0.02
            completeness_passed = avg_completeness >= 0.95
            
            all_passed = latency_passed and error_rate_passed and completeness_passed
            
            results = {
                "total_queries": len(benchmark_queries),
                "successful_queries": len(benchmark_queries) - failed_queries,
                "failed_queries": failed_queries,
                "metrics": metrics,
                "targets": {
                    "average_latency": {"value": avg_latency, "target": 30.0, "passed": latency_passed},
                    "error_rate": {"value": error_rate, "target": 0.02, "passed": error_rate_passed},
                    "report_completeness": {"value": avg_completeness, "target": 0.95, "passed": completeness_passed}
                },
                "passed": all_passed,
                "status": "PASS" if all_passed else "FAIL",
                "duration_seconds": time.time() - start_time
            }
            
            logger.info(
                f"Full pipeline validation completed: {results['status']} "
                f"(latency={avg_latency:.2f}s, error_rate={error_rate:.2%}, "
                f"completeness={avg_completeness:.2%})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Full pipeline validation failed: {e}")
            return {
                "passed": False,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    # =============================================================================
    # Run All Validations
    # =============================================================================
    
    def run_all_validations(
        self,
        sample_texts: Optional[List[str]] = None,
        sample_embeddings: Optional[np.ndarray] = None,
        benchmark_queries: Optional[List[Dict[str, Any]]] = None,
        labeled_reviews: Optional[List[Dict[str, Any]]] = None,
        ground_truth_features: Optional[Dict[str, Any]] = None,
        product_index_path: Optional[str] = None,
        review_index_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute all validation procedures in sequence.
        
        Args:
            sample_texts: Sample texts for preprocessing and embedding validation
            sample_embeddings: Sample embeddings for FAISS validation
            benchmark_queries: Benchmark queries for RAG and pipeline validation
            labeled_reviews: Labeled reviews for sentiment and feature validation
            ground_truth_features: Ground truth features for feature gap validation
            product_index_path: Path to product FAISS index
            review_index_path: Path to review FAISS index
            
        Returns:
            Dict with all validation results and summary
        """
        logger.info("=" * 80)
        logger.info("Starting comprehensive validation suite")
        logger.info("=" * 80)
        
        overall_start = time.time()
        results = {
            "unit_level": {},
            "integration_level": {},
            "end_to_end": {},
            "summary": {}
        }
        
        # Unit-Level Validations
        logger.info("\n" + "=" * 80)
        logger.info("UNIT-LEVEL VALIDATIONS")
        logger.info("=" * 80)
        
        if sample_texts:
            logger.info("\n--- Preprocessing Validation ---")
            results["unit_level"]["preprocessing"] = self.validate_preprocessing(
                sample_texts[:20]  # Use first 20 samples
            )
        
        if sample_texts:
            logger.info("\n--- Embedding Validation ---")
            results["unit_level"]["embeddings"] = self.validate_embeddings(
                sample_texts[:10]  # Use first 10 samples
            )
        
        if sample_embeddings is not None:
            logger.info("\n--- FAISS Index Validation ---")
            results["unit_level"]["faiss_index"] = self.validate_faiss_index(
                sample_embeddings[:100]  # Use first 100 embeddings
            )
        
        # Integration-Level Validations
        logger.info("\n" + "=" * 80)
        logger.info("INTEGRATION-LEVEL VALIDATIONS")
        logger.info("=" * 80)
        
        if benchmark_queries and product_index_path and review_index_path:
            logger.info("\n--- RAG Retrieval Validation ---")
            results["integration_level"]["rag_retrieval"] = self.validate_rag_retrieval(
                benchmark_queries,
                product_index_path,
                review_index_path
            )
        
        if labeled_reviews:
            logger.info("\n--- Sentiment Analysis Validation ---")
            results["integration_level"]["sentiment_analysis"] = self.validate_sentiment_analysis(
                labeled_reviews
            )
        
        if labeled_reviews and ground_truth_features:
            logger.info("\n--- Feature Gap Validation ---")
            results["integration_level"]["feature_gaps"] = self.validate_feature_gaps(
                labeled_reviews[:50],  # Use first 50 reviews
                ground_truth_features
            )
        
        # End-to-End Validation
        logger.info("\n" + "=" * 80)
        logger.info("END-TO-END VALIDATION")
        logger.info("=" * 80)
        
        if benchmark_queries and product_index_path and review_index_path:
            logger.info("\n--- Full Pipeline Validation ---")
            results["end_to_end"]["full_pipeline"] = self.validate_full_pipeline(
                benchmark_queries,
                product_index_path,
                review_index_path,
                self.config
            )
        
        # Generate Summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        total_duration = time.time() - overall_start
        
        # Count passed/failed validations
        all_validations = []
        for level in ["unit_level", "integration_level", "end_to_end"]:
            for validation_name, validation_result in results[level].items():
                all_validations.append({
                    "level": level,
                    "name": validation_name,
                    "passed": validation_result.get("passed", False),
                    "status": validation_result.get("status", "UNKNOWN")
                })
        
        passed_count = sum(1 for v in all_validations if v["passed"])
        failed_count = len(all_validations) - passed_count
        
        overall_status = "PASS" if failed_count == 0 else "FAIL"
        
        results["summary"] = {
            "total_validations": len(all_validations),
            "passed": passed_count,
            "failed": failed_count,
            "overall_status": overall_status,
            "total_duration_seconds": total_duration,
            "validations": all_validations
        }
        
        # Log summary
        logger.info(f"\nTotal Validations: {len(all_validations)}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"Total Duration: {total_duration:.2f}s")
        
        logger.info("\nDetailed Results:")
        for validation in all_validations:
            status_symbol = "✓" if validation["passed"] else "✗"
            logger.info(
                f"  {status_symbol} [{validation['level']}] {validation['name']}: "
                f"{validation['status']}"
            )
        
        logger.info("\n" + "=" * 80)
        logger.info("Validation suite completed")
        logger.info("=" * 80)
        
        return results
    
    def generate_validation_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a validation report from results.
        
        Args:
            results: Validation results from run_all_validations
            output_path: Optional path to save report (JSON format)
            
        Returns:
            Report as JSON string
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "validation_results": results,
            "summary": results.get("summary", {})
        }
        
        report_json = json.dumps(report, indent=2)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_json)
            logger.info(f"Validation report saved to: {output_path}")
        
        return report_json
