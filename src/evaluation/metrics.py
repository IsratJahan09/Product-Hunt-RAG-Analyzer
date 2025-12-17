"""
Evaluation metrics for Product Hunt RAG Analyzer.

This module implements evaluation metrics for:
- Retrieval quality (Precision@K, Recall@K, MRR)
- Sentiment analysis (Accuracy, F1-Score)
- Feature gap analysis (Extraction Recall, Categorization Accuracy)
- System performance (Average Latency, Error Rate)

Requirements: 13.2, 13.3, 13.4
"""

from datetime import datetime
from typing import List, Dict, Set, Union, Optional
from collections import Counter

from src.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# =============================================================================
# Retrieval Quality Metrics (Requirements: 13.2)
# =============================================================================

def precision_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int = 5
) -> float:
    """
    Calculate Precision@K for retrieval evaluation.
    
    Precision@K measures the proportion of retrieved items in the top-K
    that are relevant (present in ground truth).
    
    Target: ≥0.70 for K=5
    
    Args:
        retrieved_ids: List of retrieved item IDs in ranked order
        ground_truth_ids: List of relevant/ground truth item IDs
        k: Number of top results to consider
        
    Returns:
        Precision@K score between 0.0 and 1.0
        
    Example:
        >>> retrieved = ["a", "b", "c", "d", "e"]
        >>> ground_truth = ["a", "c", "f"]
        >>> precision_at_k(retrieved, ground_truth, k=5)
        0.4  # 2 relevant out of 5 retrieved
    """
    timestamp = datetime.now().isoformat()
    
    if k <= 0:
        logger.warning(f"[{timestamp}] precision_at_k: Invalid k={k}, returning 0.0")
        return 0.0
    
    if not retrieved_ids:
        logger.info(f"[{timestamp}] precision_at_k: Empty retrieved_ids, returning 0.0")
        return 0.0
    
    # Take top-K retrieved items
    top_k_retrieved = retrieved_ids[:k]
    ground_truth_set = set(ground_truth_ids)
    
    # Count relevant items in top-K
    relevant_count = sum(1 for item in top_k_retrieved if item in ground_truth_set)
    
    # Calculate precision
    precision = relevant_count / len(top_k_retrieved)
    
    logger.info(
        f"[{timestamp}] precision_at_k: k={k}, "
        f"retrieved={len(top_k_retrieved)}, relevant={relevant_count}, "
        f"precision={precision:.4f}"
    )
    
    return precision



def recall_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int = 5
) -> float:
    """
    Calculate Recall@K for retrieval evaluation.
    
    Recall@K measures the proportion of relevant items that appear
    in the top-K retrieved results.
    
    Target: ≥0.60 for K=5
    
    Args:
        retrieved_ids: List of retrieved item IDs in ranked order
        ground_truth_ids: List of relevant/ground truth item IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score between 0.0 and 1.0
        
    Example:
        >>> retrieved = ["a", "b", "c", "d", "e"]
        >>> ground_truth = ["a", "c", "f"]
        >>> recall_at_k(retrieved, ground_truth, k=5)
        0.667  # 2 out of 3 ground truth items found
    """
    timestamp = datetime.now().isoformat()
    
    if k <= 0:
        logger.warning(f"[{timestamp}] recall_at_k: Invalid k={k}, returning 0.0")
        return 0.0
    
    if not ground_truth_ids:
        logger.info(f"[{timestamp}] recall_at_k: Empty ground_truth_ids, returning 0.0")
        return 0.0
    
    # Take top-K retrieved items
    top_k_retrieved = set(retrieved_ids[:k])
    ground_truth_set = set(ground_truth_ids)
    
    # Count relevant items found in top-K
    relevant_found = len(top_k_retrieved & ground_truth_set)
    
    # Calculate recall
    recall = relevant_found / len(ground_truth_set)
    
    logger.info(
        f"[{timestamp}] recall_at_k: k={k}, "
        f"ground_truth_size={len(ground_truth_set)}, relevant_found={relevant_found}, "
        f"recall={recall:.4f}"
    )
    
    return recall


def mean_reciprocal_rank(
    retrieved_ids: List[str],
    ground_truth_ids: List[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for retrieval evaluation.
    
    MRR is the average of reciprocal ranks of the first relevant result.
    For a single query, it's 1/rank where rank is the position of the
    first relevant item (1-indexed).
    
    Target: ≥0.75
    
    Args:
        retrieved_ids: List of retrieved item IDs in ranked order
        ground_truth_ids: List of relevant/ground truth item IDs
        
    Returns:
        MRR score between 0.0 and 1.0
        
    Example:
        >>> retrieved = ["x", "a", "b", "c"]
        >>> ground_truth = ["a", "c"]
        >>> mean_reciprocal_rank(retrieved, ground_truth)
        0.5  # First relevant item "a" is at position 2, so 1/2 = 0.5
    """
    timestamp = datetime.now().isoformat()
    
    if not retrieved_ids:
        logger.info(f"[{timestamp}] mean_reciprocal_rank: Empty retrieved_ids, returning 0.0")
        return 0.0
    
    if not ground_truth_ids:
        logger.info(f"[{timestamp}] mean_reciprocal_rank: Empty ground_truth_ids, returning 0.0")
        return 0.0
    
    ground_truth_set = set(ground_truth_ids)
    
    # Find the rank of the first relevant item (1-indexed)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in ground_truth_set:
            mrr = 1.0 / rank
            logger.info(
                f"[{timestamp}] mean_reciprocal_rank: "
                f"first_relevant_rank={rank}, mrr={mrr:.4f}"
            )
            return mrr
    
    # No relevant item found
    logger.info(f"[{timestamp}] mean_reciprocal_rank: No relevant item found, returning 0.0")
    return 0.0


def mean_reciprocal_rank_batch(
    queries_results: List[Dict[str, List[str]]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.
    
    Args:
        queries_results: List of dicts with 'retrieved_ids' and 'ground_truth_ids'
        
    Returns:
        Average MRR across all queries
        
    Example:
        >>> results = [
        ...     {"retrieved_ids": ["a", "b"], "ground_truth_ids": ["a"]},
        ...     {"retrieved_ids": ["x", "y", "a"], "ground_truth_ids": ["a"]}
        ... ]
        >>> mean_reciprocal_rank_batch(results)
        0.667  # (1/1 + 1/3) / 2
    """
    timestamp = datetime.now().isoformat()
    
    if not queries_results:
        logger.info(f"[{timestamp}] mean_reciprocal_rank_batch: Empty queries, returning 0.0")
        return 0.0
    
    total_rr = 0.0
    for query in queries_results:
        rr = mean_reciprocal_rank(
            query.get("retrieved_ids", []),
            query.get("ground_truth_ids", [])
        )
        total_rr += rr
    
    avg_mrr = total_rr / len(queries_results)
    
    logger.info(
        f"[{timestamp}] mean_reciprocal_rank_batch: "
        f"num_queries={len(queries_results)}, avg_mrr={avg_mrr:.4f}"
    )
    
    return avg_mrr



# =============================================================================
# Sentiment Analysis Metrics (Requirements: 13.3)
# =============================================================================

def sentiment_accuracy(
    predictions: List[str],
    ground_truth: List[str]
) -> float:
    """
    Calculate accuracy for sentiment classification.
    
    Accuracy is the proportion of correct predictions out of total predictions.
    
    Target: ≥0.82
    
    Args:
        predictions: List of predicted sentiment labels
        ground_truth: List of true sentiment labels
        
    Returns:
        Accuracy score between 0.0 and 1.0
        
    Example:
        >>> predictions = ["positive", "negative", "neutral", "positive"]
        >>> ground_truth = ["positive", "negative", "positive", "positive"]
        >>> sentiment_accuracy(predictions, ground_truth)
        0.75  # 3 out of 4 correct
    """
    timestamp = datetime.now().isoformat()
    
    if not predictions or not ground_truth:
        logger.warning(
            f"[{timestamp}] sentiment_accuracy: Empty input, returning 0.0"
        )
        return 0.0
    
    if len(predictions) != len(ground_truth):
        logger.warning(
            f"[{timestamp}] sentiment_accuracy: Length mismatch "
            f"(predictions={len(predictions)}, ground_truth={len(ground_truth)}), "
            f"using minimum length"
        )
    
    # Use minimum length if mismatched
    min_len = min(len(predictions), len(ground_truth))
    
    # Count correct predictions
    correct = sum(
        1 for pred, truth in zip(predictions[:min_len], ground_truth[:min_len])
        if pred == truth
    )
    
    accuracy = correct / min_len
    
    logger.info(
        f"[{timestamp}] sentiment_accuracy: "
        f"total={min_len}, correct={correct}, accuracy={accuracy:.4f}"
    )
    
    return accuracy


def sentiment_f1_score(
    predictions: List[str],
    ground_truth: List[str],
    labels: Optional[List[str]] = None
) -> float:
    """
    Calculate macro-averaged F1 score for sentiment classification.
    
    F1 score is the harmonic mean of precision and recall.
    Macro-averaging computes F1 for each class and averages them.
    
    Target: ≥0.80
    
    Args:
        predictions: List of predicted sentiment labels
        ground_truth: List of true sentiment labels
        labels: Optional list of label classes (auto-detected if not provided)
        
    Returns:
        Macro-averaged F1 score between 0.0 and 1.0
        
    Example:
        >>> predictions = ["positive", "negative", "neutral", "positive", "negative"]
        >>> ground_truth = ["positive", "negative", "positive", "positive", "neutral"]
        >>> sentiment_f1_score(predictions, ground_truth)
        0.556  # Macro-averaged F1
    """
    timestamp = datetime.now().isoformat()
    
    if not predictions or not ground_truth:
        logger.warning(
            f"[{timestamp}] sentiment_f1_score: Empty input, returning 0.0"
        )
        return 0.0
    
    if len(predictions) != len(ground_truth):
        logger.warning(
            f"[{timestamp}] sentiment_f1_score: Length mismatch "
            f"(predictions={len(predictions)}, ground_truth={len(ground_truth)}), "
            f"using minimum length"
        )
    
    # Use minimum length if mismatched
    min_len = min(len(predictions), len(ground_truth))
    predictions = predictions[:min_len]
    ground_truth = ground_truth[:min_len]
    
    # Auto-detect labels if not provided
    if labels is None:
        labels = list(set(predictions) | set(ground_truth))
    
    f1_scores = []
    
    for label in labels:
        # Calculate true positives, false positives, false negatives
        tp = sum(1 for p, t in zip(predictions, ground_truth) if p == label and t == label)
        fp = sum(1 for p, t in zip(predictions, ground_truth) if p == label and t != label)
        fn = sum(1 for p, t in zip(predictions, ground_truth) if p != label and t == label)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
        
        logger.debug(
            f"[{timestamp}] sentiment_f1_score: label={label}, "
            f"tp={tp}, fp={fp}, fn={fn}, precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f}"
        )
    
    # Macro-average F1
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    logger.info(
        f"[{timestamp}] sentiment_f1_score: "
        f"num_labels={len(labels)}, macro_f1={macro_f1:.4f}"
    )
    
    return macro_f1



# =============================================================================
# Feature Gap Analysis Metrics (Requirements: 13.4)
# =============================================================================

def feature_extraction_recall(
    extracted_features: List[str],
    ground_truth_features: List[str],
    case_sensitive: bool = False
) -> float:
    """
    Calculate recall for feature extraction.
    
    Recall measures the proportion of ground truth features that were
    successfully extracted.
    
    Target: ≥0.75
    
    Args:
        extracted_features: List of extracted feature names
        ground_truth_features: List of expected/ground truth feature names
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        Recall score between 0.0 and 1.0
        
    Example:
        >>> extracted = ["dark mode", "export", "collaboration"]
        >>> ground_truth = ["dark mode", "export", "api", "mobile app"]
        >>> feature_extraction_recall(extracted, ground_truth)
        0.5  # 2 out of 4 ground truth features extracted
    """
    timestamp = datetime.now().isoformat()
    
    if not ground_truth_features:
        logger.info(
            f"[{timestamp}] feature_extraction_recall: "
            f"Empty ground_truth_features, returning 0.0"
        )
        return 0.0
    
    if not extracted_features:
        logger.info(
            f"[{timestamp}] feature_extraction_recall: "
            f"Empty extracted_features, returning 0.0"
        )
        return 0.0
    
    # Normalize for comparison
    if case_sensitive:
        extracted_set = set(extracted_features)
        ground_truth_set = set(ground_truth_features)
    else:
        extracted_set = set(f.lower().strip() for f in extracted_features)
        ground_truth_set = set(f.lower().strip() for f in ground_truth_features)
    
    # Count ground truth features that were extracted
    found = len(extracted_set & ground_truth_set)
    
    recall = found / len(ground_truth_set)
    
    logger.info(
        f"[{timestamp}] feature_extraction_recall: "
        f"extracted={len(extracted_set)}, ground_truth={len(ground_truth_set)}, "
        f"found={found}, recall={recall:.4f}"
    )
    
    return recall


def feature_categorization_accuracy(
    predicted_categories: Dict[str, str],
    ground_truth_categories: Dict[str, str],
    case_sensitive: bool = False
) -> float:
    """
    Calculate accuracy for feature categorization.
    
    Accuracy measures the proportion of features that were assigned
    the correct category.
    
    Target: ≥0.80
    
    Args:
        predicted_categories: Dict mapping feature names to predicted categories
        ground_truth_categories: Dict mapping feature names to true categories
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        Accuracy score between 0.0 and 1.0
        
    Example:
        >>> predicted = {"dark mode": "UI/UX", "api": "integrations", "speed": "performance"}
        >>> ground_truth = {"dark mode": "UI/UX", "api": "integrations", "speed": "UI/UX"}
        >>> feature_categorization_accuracy(predicted, ground_truth)
        0.667  # 2 out of 3 correct
    """
    timestamp = datetime.now().isoformat()
    
    if not ground_truth_categories:
        logger.info(
            f"[{timestamp}] feature_categorization_accuracy: "
            f"Empty ground_truth_categories, returning 0.0"
        )
        return 0.0
    
    if not predicted_categories:
        logger.info(
            f"[{timestamp}] feature_categorization_accuracy: "
            f"Empty predicted_categories, returning 0.0"
        )
        return 0.0
    
    # Normalize keys for comparison
    if case_sensitive:
        pred_norm = predicted_categories
        truth_norm = ground_truth_categories
    else:
        pred_norm = {k.lower().strip(): v.lower().strip() for k, v in predicted_categories.items()}
        truth_norm = {k.lower().strip(): v.lower().strip() for k, v in ground_truth_categories.items()}
    
    # Find common features
    common_features = set(pred_norm.keys()) & set(truth_norm.keys())
    
    if not common_features:
        logger.warning(
            f"[{timestamp}] feature_categorization_accuracy: "
            f"No common features between predicted and ground truth, returning 0.0"
        )
        return 0.0
    
    # Count correct categorizations
    correct = sum(
        1 for feature in common_features
        if pred_norm[feature] == truth_norm[feature]
    )
    
    accuracy = correct / len(common_features)
    
    logger.info(
        f"[{timestamp}] feature_categorization_accuracy: "
        f"common_features={len(common_features)}, correct={correct}, "
        f"accuracy={accuracy:.4f}"
    )
    
    return accuracy



# =============================================================================
# System Performance Metrics (Requirements: 13.5)
# =============================================================================

def calculate_average_latency(
    latencies: List[float]
) -> float:
    """
    Calculate average latency from a list of latency measurements.
    
    Target: ≤30 seconds
    
    Args:
        latencies: List of latency values in seconds
        
    Returns:
        Average latency in seconds
        
    Example:
        >>> latencies = [1.5, 2.0, 1.8, 2.2, 1.9]
        >>> calculate_average_latency(latencies)
        1.88
    """
    timestamp = datetime.now().isoformat()
    
    if not latencies:
        logger.info(
            f"[{timestamp}] calculate_average_latency: "
            f"Empty latencies list, returning 0.0"
        )
        return 0.0
    
    # Filter out invalid values
    valid_latencies = [l for l in latencies if isinstance(l, (int, float)) and l >= 0]
    
    if not valid_latencies:
        logger.warning(
            f"[{timestamp}] calculate_average_latency: "
            f"No valid latency values, returning 0.0"
        )
        return 0.0
    
    avg_latency = sum(valid_latencies) / len(valid_latencies)
    min_latency = min(valid_latencies)
    max_latency = max(valid_latencies)
    
    logger.info(
        f"[{timestamp}] calculate_average_latency: "
        f"count={len(valid_latencies)}, avg={avg_latency:.4f}s, "
        f"min={min_latency:.4f}s, max={max_latency:.4f}s"
    )
    
    return avg_latency


def calculate_error_rate(
    total_queries: int,
    failed_queries: int
) -> float:
    """
    Calculate error rate from total and failed query counts.
    
    Target: ≤2% (0.02)
    
    Args:
        total_queries: Total number of queries processed
        failed_queries: Number of queries that failed
        
    Returns:
        Error rate as a decimal (0.0 to 1.0)
        
    Example:
        >>> calculate_error_rate(total_queries=100, failed_queries=2)
        0.02  # 2% error rate
    """
    timestamp = datetime.now().isoformat()
    
    if total_queries <= 0:
        logger.warning(
            f"[{timestamp}] calculate_error_rate: "
            f"Invalid total_queries={total_queries}, returning 0.0"
        )
        return 0.0
    
    if failed_queries < 0:
        logger.warning(
            f"[{timestamp}] calculate_error_rate: "
            f"Invalid failed_queries={failed_queries}, using 0"
        )
        failed_queries = 0
    
    if failed_queries > total_queries:
        logger.warning(
            f"[{timestamp}] calculate_error_rate: "
            f"failed_queries ({failed_queries}) > total_queries ({total_queries}), "
            f"capping to total_queries"
        )
        failed_queries = total_queries
    
    error_rate = failed_queries / total_queries
    
    logger.info(
        f"[{timestamp}] calculate_error_rate: "
        f"total={total_queries}, failed={failed_queries}, "
        f"error_rate={error_rate:.4f} ({error_rate * 100:.2f}%)"
    )
    
    return error_rate


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_latency_percentiles(
    latencies: List[float],
    percentiles: List[int] = None
) -> Dict[str, float]:
    """
    Calculate latency percentiles (p50, p90, p95, p99).
    
    Args:
        latencies: List of latency values in seconds
        percentiles: List of percentiles to calculate (default: [50, 90, 95, 99])
        
    Returns:
        Dict mapping percentile names to values
        
    Example:
        >>> latencies = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0]
        >>> calculate_latency_percentiles(latencies)
        {"p50": 2.75, "p90": 5.5, "p95": 7.75, "p99": 9.55}
    """
    timestamp = datetime.now().isoformat()
    
    if percentiles is None:
        percentiles = [50, 90, 95, 99]
    
    if not latencies:
        logger.info(
            f"[{timestamp}] calculate_latency_percentiles: "
            f"Empty latencies list, returning zeros"
        )
        return {f"p{p}": 0.0 for p in percentiles}
    
    # Filter and sort valid latencies
    valid_latencies = sorted([l for l in latencies if isinstance(l, (int, float)) and l >= 0])
    
    if not valid_latencies:
        logger.warning(
            f"[{timestamp}] calculate_latency_percentiles: "
            f"No valid latency values, returning zeros"
        )
        return {f"p{p}": 0.0 for p in percentiles}
    
    result = {}
    n = len(valid_latencies)
    
    for p in percentiles:
        # Calculate percentile index
        idx = (p / 100) * (n - 1)
        lower_idx = int(idx)
        upper_idx = min(lower_idx + 1, n - 1)
        
        # Linear interpolation
        weight = idx - lower_idx
        percentile_value = (
            valid_latencies[lower_idx] * (1 - weight) +
            valid_latencies[upper_idx] * weight
        )
        result[f"p{p}"] = percentile_value
    
    logger.info(
        f"[{timestamp}] calculate_latency_percentiles: "
        f"count={n}, percentiles={result}"
    )
    
    return result


def evaluate_against_targets(
    metrics: Dict[str, float]
) -> Dict[str, Dict[str, Union[float, bool, str]]]:
    """
    Evaluate metrics against predefined targets.
    
    Targets:
    - precision_at_5: ≥0.70
    - recall_at_5: ≥0.60
    - mrr: ≥0.75
    - sentiment_accuracy: ≥0.82
    - sentiment_f1: ≥0.80
    - feature_extraction_recall: ≥0.75
    - feature_categorization_accuracy: ≥0.80
    - average_latency: ≤30.0 seconds
    - error_rate: ≤0.02 (2%)
    
    Args:
        metrics: Dict of metric names to values
        
    Returns:
        Dict with evaluation results for each metric
        
    Example:
        >>> metrics = {"precision_at_5": 0.75, "recall_at_5": 0.55}
        >>> evaluate_against_targets(metrics)
        {
            "precision_at_5": {"value": 0.75, "target": 0.70, "passed": True},
            "recall_at_5": {"value": 0.55, "target": 0.60, "passed": False}
        }
    """
    timestamp = datetime.now().isoformat()
    
    # Define targets (metric_name: (target_value, comparison_type))
    # comparison_type: "gte" = greater than or equal, "lte" = less than or equal
    targets = {
        "precision_at_5": (0.70, "gte"),
        "recall_at_5": (0.60, "gte"),
        "mrr": (0.75, "gte"),
        "sentiment_accuracy": (0.82, "gte"),
        "sentiment_f1": (0.80, "gte"),
        "feature_extraction_recall": (0.75, "gte"),
        "feature_categorization_accuracy": (0.80, "gte"),
        "average_latency": (30.0, "lte"),
        "error_rate": (0.02, "lte"),
    }
    
    results = {}
    passed_count = 0
    total_evaluated = 0
    
    for metric_name, value in metrics.items():
        if metric_name in targets:
            target_value, comparison = targets[metric_name]
            
            if comparison == "gte":
                passed = value >= target_value
            else:  # lte
                passed = value <= target_value
            
            results[metric_name] = {
                "value": value,
                "target": target_value,
                "comparison": comparison,
                "passed": passed,
                "status": "PASS" if passed else "FAIL"
            }
            
            if passed:
                passed_count += 1
            total_evaluated += 1
    
    logger.info(
        f"[{timestamp}] evaluate_against_targets: "
        f"evaluated={total_evaluated}, passed={passed_count}, "
        f"failed={total_evaluated - passed_count}"
    )
    
    return results
