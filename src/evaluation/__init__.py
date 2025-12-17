"""
Evaluation framework for Product Hunt RAG Analyzer.

This module provides evaluation metrics, validation procedures, and reporting
for assessing the quality and performance of the RAG analyzer system.

Metrics include:
- Retrieval quality: Precision@K, Recall@K, MRR
- Sentiment analysis: Accuracy, F1-Score
- Feature gap analysis: Extraction Recall, Categorization Accuracy
- System performance: Average Latency, Error Rate

Validation procedures include:
- Unit-level: Preprocessing, Embeddings, FAISS Index
- Integration-level: RAG Retrieval, Sentiment Analysis, Feature Gaps
- End-to-end: Full Pipeline

Requirements: 13.2, 13.3, 13.4, 13.5
"""

from src.evaluation.metrics import (
    # Retrieval quality metrics
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    mean_reciprocal_rank_batch,
    # Sentiment analysis metrics
    sentiment_accuracy,
    sentiment_f1_score,
    # Feature gap analysis metrics
    feature_extraction_recall,
    feature_categorization_accuracy,
    # System performance metrics
    calculate_average_latency,
    calculate_error_rate,
    # Utility functions
    calculate_latency_percentiles,
    evaluate_against_targets,
)

from src.evaluation.validation import ValidationRunner
from src.evaluation.runner import EvaluationRunner
from src.evaluation.report_generator import EvaluationReportGenerator

__all__ = [
    # Retrieval quality metrics
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "mean_reciprocal_rank_batch",
    # Sentiment analysis metrics
    "sentiment_accuracy",
    "sentiment_f1_score",
    # Feature gap analysis metrics
    "feature_extraction_recall",
    "feature_categorization_accuracy",
    # System performance metrics
    "calculate_average_latency",
    "calculate_error_rate",
    # Utility functions
    "calculate_latency_percentiles",
    "evaluate_against_targets",
    # Validation
    "ValidationRunner",
    "EvaluationRunner",
    "EvaluationReportGenerator",
]
