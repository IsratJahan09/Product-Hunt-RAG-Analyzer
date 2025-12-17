"""
Evaluation runner for Product Hunt RAG Analyzer.

This module provides the main evaluation execution logic with commands for
running different types of validations (unit, integration, end-to-end).

Requirements: 13.5
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import numpy as np

from src.evaluation.validation import ValidationRunner
from src.evaluation.report_generator import EvaluationReportGenerator
from src.utils.logger import get_logger
from src.utils.dataset_loader import DatasetLoader

logger = get_logger(__name__)


class EvaluationRunner:
    """
    Main evaluation runner that orchestrates validation procedures.
    
    Provides commands for running different types of evaluations:
    - Full: All validations (unit, integration, end-to-end)
    - Retrieval: Only RAG retrieval validation
    - Sentiment: Only sentiment analysis validation
    - Feature gaps: Only feature gap analysis validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EvaluationRunner.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.validator = ValidationRunner(config)
        self.report_generator = EvaluationReportGenerator()
        logger.info("EvaluationRunner initialized")
    
    def load_evaluation_data(
        self,
        eval_dir: str = "dataset/evaluation"
    ) -> Dict[str, Any]:
        """
        Load evaluation datasets from directory.
        
        Args:
            eval_dir: Path to evaluation directory
            
        Returns:
            Dict with loaded evaluation data
        """
        logger.info(f"Loading evaluation data from: {eval_dir}")
        eval_path = Path(eval_dir)
        
        data = {
            "benchmark_queries": [],
            "labeled_reviews": [],
            "ground_truth_features": {},
            "sample_texts": [],
            "sample_embeddings": None
        }
        
        # Load benchmark queries
        benchmark_file = eval_path / "benchmark_queries.jsonl"
        if benchmark_file.exists():
            logger.info(f"Loading benchmark queries from: {benchmark_file}")
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data["benchmark_queries"].append(json.loads(line))
            logger.info(f"Loaded {len(data['benchmark_queries'])} benchmark queries")
        else:
            logger.warning(f"Benchmark queries file not found: {benchmark_file}")
        
        # Load labeled reviews
        labeled_file = eval_path / "labeled_reviews.jsonl"
        if labeled_file.exists():
            logger.info(f"Loading labeled reviews from: {labeled_file}")
            with open(labeled_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data["labeled_reviews"].append(json.loads(line))
            logger.info(f"Loaded {len(data['labeled_reviews'])} labeled reviews")
        else:
            logger.warning(f"Labeled reviews file not found: {labeled_file}")
        
        # Load feature gaps
        features_file = eval_path / "feature_gaps.json"
        if features_file.exists():
            logger.info(f"Loading feature gaps from: {features_file}")
            with open(features_file, 'r', encoding='utf-8') as f:
                data["ground_truth_features"] = json.load(f)
            logger.info(f"Loaded ground truth features")
        else:
            logger.warning(f"Feature gaps file not found: {features_file}")
        
        # Extract sample texts from labeled reviews for unit tests
        if data["labeled_reviews"]:
            data["sample_texts"] = [
                review["body"] for review in data["labeled_reviews"][:20]
            ]
            logger.info(f"Extracted {len(data['sample_texts'])} sample texts")
        
        return data
    
    def run_full_evaluation(
        self,
        eval_dir: str = "dataset/evaluation",
        indices_dir: str = "data/indices",
        output_format: str = "json",
        output_dir: str = "reports/evaluation"
    ) -> Dict[str, Any]:
        """
        Run all validations (unit, integration, end-to-end).
        
        Args:
            eval_dir: Path to evaluation data directory
            indices_dir: Path to FAISS indices directory
            output_format: Report output format (json, html, markdown)
            output_dir: Directory to save reports
            
        Returns:
            Dict with evaluation results
        """
        logger.info("=" * 80)
        logger.info("Running FULL evaluation suite")
        logger.info("=" * 80)
        
        # Load evaluation data
        eval_data = self.load_evaluation_data(eval_dir)
        
        # Generate sample embeddings for FAISS validation
        sample_embeddings = None
        if eval_data["sample_texts"]:
            try:
                from src.modules.embeddings import EmbeddingGenerator
                generator = EmbeddingGenerator()
                sample_embeddings = generator.generate_embeddings(
                    eval_data["sample_texts"][:100]
                )
                logger.info(f"Generated {len(sample_embeddings)} sample embeddings")
            except Exception as e:
                logger.warning(f"Failed to generate sample embeddings: {e}")
        
        # Set up index paths
        indices_path = Path(indices_dir)
        product_index_path = str(indices_path / "products")
        review_index_path = str(indices_path / "reviews")
        
        # Run all validations
        results = self.validator.run_all_validations(
            sample_texts=eval_data["sample_texts"],
            sample_embeddings=sample_embeddings,
            benchmark_queries=eval_data["benchmark_queries"],
            labeled_reviews=eval_data["labeled_reviews"],
            ground_truth_features=eval_data["ground_truth_features"],
            product_index_path=product_index_path,
            review_index_path=review_index_path
        )
        
        # Generate and save report
        self._save_report(results, output_format, output_dir)
        
        return results
    
    def run_retrieval_evaluation(
        self,
        eval_dir: str = "dataset/evaluation",
        indices_dir: str = "data/indices",
        output_format: str = "json",
        output_dir: str = "reports/evaluation"
    ) -> Dict[str, Any]:
        """
        Run only RAG retrieval validation.
        
        Args:
            eval_dir: Path to evaluation data directory
            indices_dir: Path to FAISS indices directory
            output_format: Report output format (json, html, markdown)
            output_dir: Directory to save reports
            
        Returns:
            Dict with evaluation results
        """
        logger.info("=" * 80)
        logger.info("Running RETRIEVAL evaluation")
        logger.info("=" * 80)
        
        # Load evaluation data
        eval_data = self.load_evaluation_data(eval_dir)
        
        if not eval_data["benchmark_queries"]:
            logger.error("No benchmark queries found for retrieval evaluation")
            return {
                "passed": False,
                "status": "ERROR",
                "error": "No benchmark queries available"
            }
        
        # Set up index paths
        indices_path = Path(indices_dir)
        product_index_path = str(indices_path / "products")
        review_index_path = str(indices_path / "reviews")
        
        # Run retrieval validation
        results = self.validator.validate_rag_retrieval(
            benchmark_queries=eval_data["benchmark_queries"],
            product_index_path=product_index_path,
            review_index_path=review_index_path
        )
        
        # Wrap in standard format
        wrapped_results = {
            "integration_level": {
                "rag_retrieval": results
            },
            "summary": {
                "total_validations": 1,
                "passed": 1 if results.get("passed") else 0,
                "failed": 0 if results.get("passed") else 1,
                "overall_status": results.get("status", "UNKNOWN")
            }
        }
        
        # Generate and save report
        self._save_report(wrapped_results, output_format, output_dir)
        
        return wrapped_results
    
    def run_sentiment_evaluation(
        self,
        eval_dir: str = "dataset/evaluation",
        output_format: str = "json",
        output_dir: str = "reports/evaluation"
    ) -> Dict[str, Any]:
        """
        Run only sentiment analysis validation.
        
        Args:
            eval_dir: Path to evaluation data directory
            output_format: Report output format (json, html, markdown)
            output_dir: Directory to save reports
            
        Returns:
            Dict with evaluation results
        """
        logger.info("=" * 80)
        logger.info("Running SENTIMENT ANALYSIS evaluation")
        logger.info("=" * 80)
        
        # Load evaluation data
        eval_data = self.load_evaluation_data(eval_dir)
        
        if not eval_data["labeled_reviews"]:
            logger.error("No labeled reviews found for sentiment evaluation")
            return {
                "passed": False,
                "status": "ERROR",
                "error": "No labeled reviews available"
            }
        
        # Run sentiment validation
        results = self.validator.validate_sentiment_analysis(
            labeled_reviews=eval_data["labeled_reviews"]
        )
        
        # Wrap in standard format
        wrapped_results = {
            "integration_level": {
                "sentiment_analysis": results
            },
            "summary": {
                "total_validations": 1,
                "passed": 1 if results.get("passed") else 0,
                "failed": 0 if results.get("passed") else 1,
                "overall_status": results.get("status", "UNKNOWN")
            }
        }
        
        # Generate and save report
        self._save_report(wrapped_results, output_format, output_dir)
        
        return wrapped_results
    
    def run_feature_gaps_evaluation(
        self,
        eval_dir: str = "dataset/evaluation",
        output_format: str = "json",
        output_dir: str = "reports/evaluation"
    ) -> Dict[str, Any]:
        """
        Run only feature gap analysis validation.
        
        Args:
            eval_dir: Path to evaluation data directory
            output_format: Report output format (json, html, markdown)
            output_dir: Directory to save reports
            
        Returns:
            Dict with evaluation results
        """
        logger.info("=" * 80)
        logger.info("Running FEATURE GAP ANALYSIS evaluation")
        logger.info("=" * 80)
        
        # Load evaluation data
        eval_data = self.load_evaluation_data(eval_dir)
        
        if not eval_data["labeled_reviews"]:
            logger.error("No labeled reviews found for feature gap evaluation")
            return {
                "passed": False,
                "status": "ERROR",
                "error": "No labeled reviews available"
            }
        
        if not eval_data["ground_truth_features"]:
            logger.error("No ground truth features found for feature gap evaluation")
            return {
                "passed": False,
                "status": "ERROR",
                "error": "No ground truth features available"
            }
        
        # Run feature gap validation
        results = self.validator.validate_feature_gaps(
            labeled_reviews=eval_data["labeled_reviews"][:50],
            ground_truth_features=eval_data["ground_truth_features"]
        )
        
        # Wrap in standard format
        wrapped_results = {
            "integration_level": {
                "feature_gaps": results
            },
            "summary": {
                "total_validations": 1,
                "passed": 1 if results.get("passed") else 0,
                "failed": 0 if results.get("passed") else 1,
                "overall_status": results.get("status", "UNKNOWN")
            }
        }
        
        # Generate and save report
        self._save_report(wrapped_results, output_format, output_dir)
        
        return wrapped_results
    
    def _save_report(
        self,
        results: Dict[str, Any],
        output_format: str,
        output_dir: str
    ) -> None:
        """
        Generate and save evaluation report.
        
        Args:
            results: Evaluation results
            output_format: Report format (json, html, markdown)
            output_dir: Directory to save report
        """
        from datetime import datetime
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate report based on format
        if output_format == "json":
            filename = f"evaluation_report_{timestamp}.json"
            filepath = output_path / filename
            self.report_generator.generate_json_report(results, str(filepath))
            logger.info(f"JSON report saved to: {filepath}")
        
        elif output_format == "html":
            filename = f"evaluation_report_{timestamp}.html"
            filepath = output_path / filename
            self.report_generator.generate_html_report(results, str(filepath))
            logger.info(f"HTML report saved to: {filepath}")
        
        elif output_format == "markdown":
            filename = f"evaluation_report_{timestamp}.md"
            filepath = output_path / filename
            self.report_generator.generate_markdown_report(results, str(filepath))
            logger.info(f"Markdown report saved to: {filepath}")
        
        else:
            logger.warning(f"Unknown output format: {output_format}, defaulting to JSON")
            filename = f"evaluation_report_{timestamp}.json"
            filepath = output_path / filename
            self.report_generator.generate_json_report(results, str(filepath))
            logger.info(f"JSON report saved to: {filepath}")


def main():
    """
    Main entry point for evaluation runner CLI.
    """
    parser = argparse.ArgumentParser(
        description="Run evaluation procedures for Product Hunt RAG Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all evaluations
  python -m src.evaluation.runner --full
  
  # Run only retrieval evaluation
  python -m src.evaluation.runner --retrieval
  
  # Run only sentiment evaluation
  python -m src.evaluation.runner --sentiment
  
  # Run only feature gap evaluation
  python -m src.evaluation.runner --feature_gaps
  
  # Specify custom paths and output format
  python -m src.evaluation.runner --full --eval-dir dataset/evaluation --output-format html
        """
    )
    
    # Evaluation type arguments (mutually exclusive)
    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument(
        "--full",
        action="store_true",
        help="Run all validations (unit, integration, end-to-end)"
    )
    eval_group.add_argument(
        "--retrieval",
        action="store_true",
        help="Run only retrieval validation"
    )
    eval_group.add_argument(
        "--sentiment",
        action="store_true",
        help="Run only sentiment validation"
    )
    eval_group.add_argument(
        "--feature_gaps",
        action="store_true",
        help="Run only feature gap validation"
    )
    
    # Path arguments
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="dataset/evaluation",
        help="Path to evaluation data directory (default: dataset/evaluation)"
    )
    parser.add_argument(
        "--indices-dir",
        type=str,
        default="data/indices",
        help="Path to FAISS indices directory (default: data/indices)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation",
        help="Directory to save evaluation reports (default: reports/evaluation)"
    )
    
    # Output format
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "html", "markdown"],
        help="Report output format (default: json)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner()
    
    # Run appropriate evaluation
    try:
        if args.full:
            results = runner.run_full_evaluation(
                eval_dir=args.eval_dir,
                indices_dir=args.indices_dir,
                output_format=args.output_format,
                output_dir=args.output_dir
            )
        elif args.retrieval:
            results = runner.run_retrieval_evaluation(
                eval_dir=args.eval_dir,
                indices_dir=args.indices_dir,
                output_format=args.output_format,
                output_dir=args.output_dir
            )
        elif args.sentiment:
            results = runner.run_sentiment_evaluation(
                eval_dir=args.eval_dir,
                output_format=args.output_format,
                output_dir=args.output_dir
            )
        elif args.feature_gaps:
            results = runner.run_feature_gaps_evaluation(
                eval_dir=args.eval_dir,
                output_format=args.output_format,
                output_dir=args.output_dir
            )
        
        # Print summary
        summary = results.get("summary", {})
        overall_status = summary.get("overall_status", "UNKNOWN")
        
        print("\n" + "=" * 80)
        print(f"EVALUATION COMPLETE: {overall_status}")
        print("=" * 80)
        print(f"Total Validations: {summary.get('total_validations', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print("=" * 80)
        
        # Exit with appropriate code
        sys.exit(0 if overall_status == "PASS" else 1)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
