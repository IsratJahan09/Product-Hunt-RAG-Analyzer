"""
Comprehensive evaluation metrics for fine-tuning.

This module provides evaluation metrics for:
1. Embedding models (retrieval metrics)
2. Sentiment models (classification metrics)
3. LLM models (generation metrics)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from collections import defaultdict


class RetrievalMetrics:
    """Metrics for evaluating retrieval/ranking performance."""
    
    @staticmethod
    def precision_at_k(relevant: List[int], retrieved: List[int], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            relevant: List of relevant item indices
            retrieved: List of retrieved item indices (ranked)
            k: Number of top items to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        num_relevant_retrieved = len(retrieved_at_k.intersection(relevant_set))
        return num_relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(relevant: List[int], retrieved: List[int], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            relevant: List of relevant item indices
            retrieved: List of retrieved item indices (ranked)
            k: Number of top items to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        num_relevant_retrieved = len(retrieved_at_k.intersection(relevant_set))
        return num_relevant_retrieved / len(relevant)
    
    @staticmethod
    def average_precision(relevant: List[int], retrieved: List[int]) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            relevant: List of relevant item indices
            retrieved: List of retrieved item indices (ranked)
            
        Returns:
            Average Precision score
        """
        if len(relevant) == 0:
            return 0.0
        
        relevant_set = set(relevant)
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(retrieved):
            if item in relevant_set:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                score += precision_at_i
        
        return score / len(relevant)
    
    @staticmethod
    def mean_average_precision(
        relevant_lists: List[List[int]],
        retrieved_lists: List[List[int]]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).
        
        Args:
            relevant_lists: List of relevant item lists for each query
            retrieved_lists: List of retrieved item lists for each query
            
        Returns:
            MAP score
        """
        if len(relevant_lists) == 0:
            return 0.0
        
        ap_scores = []
        for relevant, retrieved in zip(relevant_lists, retrieved_lists):
            ap = RetrievalMetrics.average_precision(relevant, retrieved)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    @staticmethod
    def reciprocal_rank(relevant: List[int], retrieved: List[int]) -> float:
        """
        Calculate Reciprocal Rank (RR).
        
        Args:
            relevant: List of relevant item indices
            retrieved: List of retrieved item indices (ranked)
            
        Returns:
            Reciprocal Rank score
        """
        relevant_set = set(relevant)
        
        for i, item in enumerate(retrieved):
            if item in relevant_set:
                return 1.0 / (i + 1.0)
        
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(
        relevant_lists: List[List[int]],
        retrieved_lists: List[List[int]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            relevant_lists: List of relevant item lists for each query
            retrieved_lists: List of retrieved item lists for each query
            
        Returns:
            MRR score
        """
        if len(relevant_lists) == 0:
            return 0.0
        
        rr_scores = []
        for relevant, retrieved in zip(relevant_lists, retrieved_lists):
            rr = RetrievalMetrics.reciprocal_rank(relevant, retrieved)
            rr_scores.append(rr)
        
        return np.mean(rr_scores)
    
    @staticmethod
    def ndcg_at_k(
        relevant: List[int],
        retrieved: List[int],
        k: int,
        relevance_scores: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            relevant: List of relevant item indices
            retrieved: List of retrieved item indices (ranked)
            k: Number of top items to consider
            relevance_scores: Optional dict mapping item indices to relevance scores
                            (default: binary relevance)
            
        Returns:
            NDCG@K score
        """
        if k == 0 or len(relevant) == 0:
            return 0.0
        
        # Default to binary relevance
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(retrieved[:k]):
            if item in relevance_scores:
                rel = relevance_scores[item]
                dcg += rel / np.log2(i + 2)  # i+2 because i starts at 0
        
        # Calculate IDCG (ideal DCG)
        ideal_retrieved = sorted(
            relevant,
            key=lambda x: relevance_scores.get(x, 0.0),
            reverse=True
        )[:k]
        
        idcg = 0.0
        for i, item in enumerate(ideal_retrieved):
            rel = relevance_scores.get(item, 0.0)
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def evaluate_retrieval(
        relevant_lists: List[List[int]],
        retrieved_lists: List[List[int]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Comprehensive retrieval evaluation.
        
        Args:
            relevant_lists: List of relevant item lists for each query
            retrieved_lists: List of retrieved item lists for each query
            k_values: List of K values for Precision@K, Recall@K, NDCG@K
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Precision@K and Recall@K
        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            
            for relevant, retrieved in zip(relevant_lists, retrieved_lists):
                precisions.append(
                    RetrievalMetrics.precision_at_k(relevant, retrieved, k)
                )
                recalls.append(
                    RetrievalMetrics.recall_at_k(relevant, retrieved, k)
                )
                ndcgs.append(
                    RetrievalMetrics.ndcg_at_k(relevant, retrieved, k)
                )
            
            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'recall@{k}'] = np.mean(recalls)
            metrics[f'ndcg@{k}'] = np.mean(ndcgs)
        
        # MAP and MRR
        metrics['map'] = RetrievalMetrics.mean_average_precision(
            relevant_lists, retrieved_lists
        )
        metrics['mrr'] = RetrievalMetrics.mean_reciprocal_rank(
            relevant_lists, retrieved_lists
        )
        
        return metrics


class ClassificationMetrics:
    """Metrics for evaluating classification performance."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional label names
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
        }
        
        # Per-class metrics
        if labels is not None:
            for i, label in enumerate(labels):
                metrics[f'precision_{label}'] = precision[i]
                metrics[f'recall_{label}'] = recall[i]
                metrics[f'f1_{label}'] = f1[i]
                metrics[f'support_{label}'] = support[i]
        
        return metrics
    
    @staticmethod
    def confusion_matrix_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute confusion matrix and derived metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional label names
            
        Returns:
            Tuple of (confusion_matrix, metrics_dict)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class accuracy from confusion matrix
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        metrics = {
            'confusion_matrix': cm.tolist(),
        }
        
        if labels is not None:
            for i, label in enumerate(labels):
                metrics[f'accuracy_{label}'] = per_class_accuracy[i]
        
        return cm, metrics
    
    @staticmethod
    def classification_report_dict(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate classification report as dictionary.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional label names
            
        Returns:
            Classification report dictionary
        """
        return classification_report(
            y_true, y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0
        )


class EmbeddingEvaluator:
    """Evaluator for embedding models with baseline comparison."""
    
    def __init__(self, base_model, finetuned_model):
        """
        Initialize evaluator.
        
        Args:
            base_model: Base embedding model
            finetuned_model: Fine-tuned embedding model
        """
        self.base_model = base_model
        self.finetuned_model = finetuned_model
    
    def evaluate_similarity_improvement(
        self,
        text_pairs: List[Tuple[str, str]],
        expected_similarities: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate similarity score improvements.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            expected_similarities: Expected similarity scores (0-1)
            
        Returns:
            Dictionary with improvement metrics
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        base_errors = []
        ft_errors = []
        
        for (text1, text2), expected in zip(text_pairs, expected_similarities):
            # Base model
            base_emb1 = self.base_model.generate_embeddings(text1)
            base_emb2 = self.base_model.generate_embeddings(text2)
            base_sim = cosine_similarity(base_emb1, base_emb2)[0][0]
            base_errors.append(abs(base_sim - expected))
            
            # Fine-tuned model
            ft_emb1 = self.finetuned_model.generate_embeddings(text1)
            ft_emb2 = self.finetuned_model.generate_embeddings(text2)
            ft_sim = cosine_similarity(ft_emb1, ft_emb2)[0][0]
            ft_errors.append(abs(ft_sim - expected))
        
        base_mae = np.mean(base_errors)
        ft_mae = np.mean(ft_errors)
        
        return {
            'base_mae': base_mae,
            'finetuned_mae': ft_mae,
            'improvement': base_mae - ft_mae,
            'improvement_pct': ((base_mae - ft_mae) / base_mae) * 100 if base_mae > 0 else 0
        }
    
    def evaluate_ranking_improvement(
        self,
        queries: List[str],
        candidates: List[List[str]],
        relevant_indices: List[List[int]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ranking improvements.
        
        Args:
            queries: List of query texts
            candidates: List of candidate lists for each query
            relevant_indices: List of relevant candidate indices for each query
            k_values: K values for metrics
            
        Returns:
            Dictionary with base and fine-tuned metrics
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        base_retrieved = []
        ft_retrieved = []
        
        for query, cands in zip(queries, candidates):
            # Base model
            query_emb_base = self.base_model.generate_embeddings(query)
            cands_emb_base = self.base_model.generate_embeddings(cands)
            base_sims = cosine_similarity(query_emb_base, cands_emb_base)[0]
            base_ranking = np.argsort(base_sims)[::-1].tolist()
            base_retrieved.append(base_ranking)
            
            # Fine-tuned model
            query_emb_ft = self.finetuned_model.generate_embeddings(query)
            cands_emb_ft = self.finetuned_model.generate_embeddings(cands)
            ft_sims = cosine_similarity(query_emb_ft, cands_emb_ft)[0]
            ft_ranking = np.argsort(ft_sims)[::-1].tolist()
            ft_retrieved.append(ft_ranking)
        
        # Evaluate both
        base_metrics = RetrievalMetrics.evaluate_retrieval(
            relevant_indices, base_retrieved, k_values
        )
        ft_metrics = RetrievalMetrics.evaluate_retrieval(
            relevant_indices, ft_retrieved, k_values
        )
        
        # Calculate improvements
        improvements = {}
        for key in base_metrics:
            improvements[key] = ft_metrics[key] - base_metrics[key]
            improvements[f'{key}_pct'] = (
                (improvements[key] / base_metrics[key]) * 100
                if base_metrics[key] > 0 else 0
            )
        
        return {
            'base': base_metrics,
            'finetuned': ft_metrics,
            'improvement': improvements
        }


def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Table title
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:30s}: {value:8.4f}")
        else:
            print(f"{metric:30s}: {value}")
    
    print(f"{'='*60}\n")


def print_comparison_table(
    base_metrics: Dict[str, float],
    finetuned_metrics: Dict[str, float],
    title: str = "Model Comparison"
) -> None:
    """
    Print comparison table for base vs fine-tuned models.
    
    Args:
        base_metrics: Base model metrics
        finetuned_metrics: Fine-tuned model metrics
        title: Table title
    """
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Base':>12} {'Fine-tuned':>12} {'Improvement':>12} {'%':>8}")
    print(f"{'-'*80}")
    
    for metric in base_metrics:
        if metric in finetuned_metrics:
            base_val = base_metrics[metric]
            ft_val = finetuned_metrics[metric]
            
            if isinstance(base_val, (int, float)) and isinstance(ft_val, (int, float)):
                improvement = ft_val - base_val
                pct = (improvement / base_val * 100) if base_val != 0 else 0
                
                # Color coding
                symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                
                print(f"{metric:<30} {base_val:>12.4f} {ft_val:>12.4f} "
                      f"{improvement:>12.4f} {pct:>7.1f}% {symbol}")
    
    print(f"{'='*80}\n")
