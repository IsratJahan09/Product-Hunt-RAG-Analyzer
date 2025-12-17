"""
Main orchestration layer for Product Hunt RAG Analyzer.

This module provides the AnalysisPipeline class that coordinates all modules
to execute the complete competitive intelligence analysis workflow.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from src.utils.config import ConfigManager
from src.utils.logger import Logger, get_logger
from src.utils.dataset_loader import DatasetLoader
from src.modules.preprocessing import TextPreprocessor
from src.modules.embeddings import EmbeddingGenerator
from src.modules.sentiment import SentimentAnalyzer
from src.modules.vector_storage import FAISSIndexManager
from src.modules.rag_retrieval import RAGRetriever
from src.modules.feature_analysis import FeatureAnalyzer
from src.modules.positioning_analysis import PositioningAnalyzer
from src.modules.llm_generation import LLMGenerator, LLMConnectionError, LLMGenerationError
from src.modules.report_generation import ReportGenerator
from src.modules.feature_gap_service import FeatureGapService


logger = get_logger(__name__)


class AnalysisPipelineError(Exception):
    """Exception raised when pipeline execution fails."""
    pass


class AnalysisPipeline:
    """
    Main orchestration class that coordinates all modules for competitive analysis.
    
    Executes the complete pipeline from product idea to comprehensive report:
    1. Competitor Identification
    2. Review Retrieval and Analysis
    3. LLM Generation
    4. Report Generation
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize AnalysisPipeline with all module instances.
        
        Args:
            config: ConfigManager instance. If None, loads default config.
            
        Example:
            >>> pipeline = AnalysisPipeline()
            >>> pipeline = AnalysisPipeline(config=ConfigManager("custom_config.yaml"))
        """
        logger.info("Initializing AnalysisPipeline")
        start_time = time.time()
        
        # Initialize configuration
        if config is None:
            logger.info("No config provided, loading default configuration")
            self.config = ConfigManager()
        else:
            self.config = config
        
        # Setup logging from config
        Logger.setup_from_config(self.config)
        
        # Initialize dataset loader
        self.dataset_loader = DatasetLoader()
        
        # Initialize preprocessing
        max_length = self.config.get("processing.max_text_length", 512)
        self.text_preprocessor = TextPreprocessor(max_length=max_length)
        
        # Initialize embedding generator
        embedding_model = self.config.get("models.embedding.name", "all-MiniLM-L6-v2")
        embedding_device = self.config.get("models.embedding.device", "cpu")
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            device=embedding_device
        )
        
        # Initialize sentiment analyzer
        sentiment_model = self.config.get(
            "models.sentiment.name",
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        sentiment_device = self.config.get("models.sentiment.device", "cpu")
        self.sentiment_analyzer = SentimentAnalyzer(
            model_name=sentiment_model,
            device=sentiment_device
        )
        
        # Initialize FAISS index managers (will be loaded later)
        embedding_dim = self.embedding_generator.get_embedding_dimension()
        self.product_index = FAISSIndexManager(dimension=embedding_dim)
        self.review_index = FAISSIndexManager(dimension=embedding_dim)
        
        # Initialize RAG retriever
        self.rag_retriever = RAGRetriever(
            embedding_generator=self.embedding_generator,
            product_index_manager=self.product_index,
            review_index_manager=self.review_index
        )
        
        # Initialize analysis modules
        self.feature_analyzer = FeatureAnalyzer()
        self.positioning_analyzer = PositioningAnalyzer()
        
        # Initialize LLM generator with Groq API (will be set after LLM init)
        self.feature_gap_service = None  # Will be initialized after LLM generator
        
        # Initialize LLM generator with Groq API
        llm_config = {
            "api_key": self.config.get("models.llm.api_key", ""),
            "model": self.config.get("models.llm.model", "llama-3.3-70b-versatile")
        }
        llm_temperature = self.config.get("models.llm.temperature", 0.7)
        llm_top_p = self.config.get("models.llm.top_p", 0.9)
        llm_max_tokens = self.config.get("models.llm.max_tokens", 3000)
        llm_timeout = self.config.get("models.llm.timeout", 120)
        
        self.llm_generator = LLMGenerator(
            config=llm_config,
            temperature=llm_temperature,
            top_p=llm_top_p,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout
        )
        
        # Initialize report generator
        self.report_generator = ReportGenerator()
        
        # Initialize feature gap service with LLM generator
        self.feature_gap_service = FeatureGapService(
            feature_analyzer=self.feature_analyzer,
            llm_generator=self.llm_generator,
            min_gaps_threshold=3
        )
        
        # Pipeline state
        self.indices_loaded = False
        self.product_index_path = None
        self.review_index_path = None
        
        init_time = (time.time() - start_time) * 1000
        logger.info(f"AnalysisPipeline initialized successfully in {init_time:.2f}ms")
    
    def load_indices(
        self,
        product_index_path: str,
        review_index_path: str
    ) -> Dict[str, Any]:
        """
        Load pre-built FAISS indices from disk.
        
        Args:
            product_index_path: Path to product index (without extension)
            review_index_path: Path to review index (without extension)
            
        Returns:
            Dictionary with loading results and statistics
            
        Raises:
            FileNotFoundError: If index files don't exist
            RuntimeError: If loading fails
            
        Example:
            >>> pipeline = AnalysisPipeline()
            >>> pipeline.load_indices(
            ...     "data/indices/products",
            ...     "data/indices/reviews"
            ... )
        """
        logger.info("Loading FAISS indices")
        start_time = time.time()
        
        try:
            # Load product index
            logger.info(f"Loading product index from: {product_index_path}")
            self.product_index.load_index(product_index_path)
            
            # Load review index
            logger.info(f"Loading review index from: {review_index_path}")
            self.review_index.load_index(review_index_path)
            
            # Update state
            self.indices_loaded = True
            self.product_index_path = product_index_path
            self.review_index_path = review_index_path
            
            load_time = (time.time() - start_time) * 1000
            
            result = {
                "success": True,
                "product_index_size": self.product_index.get_index_size(),
                "review_index_size": self.review_index.get_index_size(),
                "load_time_ms": load_time
            }
            
            logger.info(
                f"Indices loaded successfully in {load_time:.2f}ms. "
                f"Products: {result['product_index_size']}, "
                f"Reviews: {result['review_index_size']}"
            )
            
            return result
            
        except FileNotFoundError as e:
            logger.error(f"Index files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load indices: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load indices: {e}")

    
    def run_analysis(
        self,
        product_idea: str,
        max_competitors: int = 5,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Execute full analysis pipeline from product idea to report.
        
        Pipeline stages:
        1. Competitor Identification - Find similar products
        2. Review Retrieval and Analysis - Get reviews and analyze sentiment/features
        3. LLM Generation - Generate competitive intelligence insights
        4. Report Generation - Create comprehensive report
        
        Args:
            product_idea: Description of the product idea to analyze
            max_competitors: Number of competitors to analyze (default: 5)
            output_format: Report format - "json", "markdown", or "pdf" (default: "json")
            
        Returns:
            Dictionary with analysis results including:
                - analysis_id: Unique identifier
                - status: "completed" or "failed"
                - product_idea: Original product idea
                - competitors_identified: List of competitor names
                - results: Analysis results with market_positioning, feature_gaps, etc.
                - confidence_score: Overall confidence (0.0-1.0)
                - processing_time_ms: Total processing time
                
        Raises:
            AnalysisPipelineError: If pipeline execution fails
            ValueError: If inputs are invalid
            
        Example:
            >>> pipeline = AnalysisPipeline()
            >>> pipeline.load_indices("data/indices/products", "data/indices/reviews")
            >>> result = pipeline.run_analysis(
            ...     "A task management app with AI-powered prioritization",
            ...     max_competitors=5,
            ...     output_format="json"
            ... )
        """
        # Validate inputs
        if not product_idea or not product_idea.strip():
            raise ValueError("product_idea cannot be empty")
        
        if max_competitors <= 0:
            raise ValueError(f"max_competitors must be positive, got {max_competitors}")
        
        if output_format not in ["json", "markdown", "pdf"]:
            raise ValueError(
                f"Invalid output_format: {output_format}. "
                f"Must be 'json', 'markdown', or 'pdf'"
            )
        
        if not self.indices_loaded:
            raise AnalysisPipelineError(
                "Indices not loaded. Call load_indices() first."
            )
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        logger.info(
            f"Starting analysis pipeline (ID: {analysis_id}) for product idea: "
            f"'{product_idea[:100]}...'"
        )
        logger.info(
            f"Parameters: max_competitors={max_competitors}, "
            f"output_format={output_format}"
        )
        
        pipeline_start_time = time.time()
        stage_times = {}
        
        try:
            # ===== STAGE 1: Competitor Identification =====
            logger.info("=" * 60)
            logger.info("STAGE 1: Competitor Identification")
            logger.info("=" * 60)
            
            stage_start = time.time()
            
            # Convert product idea to embedding
            logger.info("Converting product idea to embedding")
            product_embedding = self.embedding_generator.generate_embeddings(
                product_idea,
                normalize=True,
                show_progress=False
            )
            
            # Identify top-K competitor products
            logger.info(f"Identifying top-{max_competitors} competitors")
            competitors = self.rag_retriever.identify_competitors(
                product_idea=product_idea,
                k=max_competitors
            )
            
            # Retrieve competitor metadata
            competitor_ids = [comp["product_id"] for comp in competitors]
            logger.info(f"Retrieving metadata for {len(competitor_ids)} competitors")
            competitor_metadata = self.rag_retriever.retrieve_competitor_metadata(
                competitor_ids
            )
            
            stage_times["stage_1_competitor_identification"] = (
                (time.time() - stage_start) * 1000
            )
            
            logger.info(
                f"Stage 1 complete in {stage_times['stage_1_competitor_identification']:.2f}ms. "
                f"Identified {len(competitors)} competitors"
            )
            
            # ===== STAGE 2: Review Retrieval and Analysis =====
            logger.info("=" * 60)
            logger.info("STAGE 2: Review Retrieval and Analysis")
            logger.info("=" * 60)
            
            stage_start = time.time()
            
            # Retrieve top reviews for each competitor
            k_per_competitor = self.config.get("retrieval.reviews_per_competitor", 10)
            logger.info(
                f"Retrieving top-{k_per_competitor} reviews per competitor"
            )
            competitor_reviews = self.rag_retriever.retrieve_competitor_reviews(
                competitor_ids=competitor_ids,
                k_per_competitor=k_per_competitor
            )
            
            # Perform sentiment analysis on retrieved reviews
            logger.info("Performing sentiment analysis on reviews")
            all_reviews_for_sentiment = []
            review_to_competitor_map = {}
            
            for product_id, reviews in competitor_reviews.items():
                for review in reviews:
                    review_text = review.get("metadata", {}).get("original_text", "")
                    if review_text:
                        all_reviews_for_sentiment.append(review_text)
                        review_to_competitor_map[len(all_reviews_for_sentiment) - 1] = product_id
            
            sentiment_results = []
            if all_reviews_for_sentiment:
                sentiment_batch_size = self.config.get("processing.sentiment_batch_size", 16)
                sentiment_results = self.sentiment_analyzer.batch_analyze(
                    all_reviews_for_sentiment,
                    batch_size=sentiment_batch_size,
                    show_progress=True,
                    extract_aspects=True
                )
            
            # Merge sentiment results back into reviews
            sentiment_idx = 0
            for product_id, reviews in competitor_reviews.items():
                for review in reviews:
                    if review.get("metadata", {}).get("original_text"):
                        if sentiment_idx < len(sentiment_results):
                            sentiment_data = sentiment_results[sentiment_idx]
                            review["sentiment"] = sentiment_data.get("sentiment", "neutral")
                            review["confidence"] = sentiment_data.get("confidence", 0.0)
                            review["aspects"] = sentiment_data.get("aspects", [])
                            sentiment_idx += 1
            
            # Extract features using FeatureAnalyzer
            logger.info("Extracting features from reviews")
            all_reviews_for_features = []
            for product_id, reviews in competitor_reviews.items():
                for review in reviews:
                    review_dict = {
                        "text": review.get("metadata", {}).get("original_text", ""),
                        "sentiment": review.get("sentiment", "neutral"),
                        "confidence": review.get("confidence", 0.0)
                    }
                    if review_dict["text"]:
                        all_reviews_for_features.append(review_dict)
            
            feature_analysis = {}
            feature_gap_report = None
            
            if all_reviews_for_features:
                extracted_features = self.feature_analyzer.extract_features(
                    all_reviews_for_features
                )
                aggregated_features = self.feature_analyzer.aggregate_feature_mentions(
                    extracted_features
                )
                prioritized_features = self.feature_analyzer.identify_high_priority_gaps(
                    aggregated_features,
                    sentiment_threshold=0.3,
                    frequency_threshold=2
                )
                
                feature_analysis = {
                    "extracted_features": extracted_features,
                    "aggregated_features": aggregated_features,
                    "prioritized_features": prioritized_features
                }
                
                # Enhanced Feature Gap Analysis using FeatureGapService
                # This analyzes reviews for missing features and falls back to LLM suggestions
                logger.info("Performing enhanced feature gap analysis")
                try:
                    feature_gap_report = self.feature_gap_service.analyze_feature_gaps(
                        product_name=product_idea[:50],  # Use first 50 chars as name
                        product_description=product_idea,
                        reviews=all_reviews_for_features,
                        existing_features=None,  # Could be provided by user in future
                        competitor_features=None
                    )
                    
                    # Add feature gap report to feature_analysis
                    feature_analysis["feature_gap_report"] = self.feature_gap_service.to_dict(feature_gap_report)
                    
                    logger.info(
                        f"Feature gap analysis complete: {len(feature_gap_report.gaps_from_reviews)} gaps from reviews, "
                        f"{len(feature_gap_report.llm_generated_suggestions)} LLM suggestions"
                    )
                except Exception as e:
                    logger.warning(f"Feature gap analysis failed: {e}. Continuing with basic analysis.")
                    feature_analysis["feature_gap_report"] = None
            
            # Analyze positioning using PositioningAnalyzer
            logger.info("Analyzing market positioning")
            competitors_data_for_positioning = []
            for comp in competitors:
                product_id = comp["product_id"]
                metadata = competitor_metadata.get(product_id, {})
                reviews = competitor_reviews.get(product_id, [])
                
                competitors_data_for_positioning.append({
                    "name": metadata.get("name", product_id),
                    "description": metadata.get("description", ""),
                    "reviews": [
                        {
                            "text": r.get("metadata", {}).get("original_text", ""),
                            "sentiment": r.get("sentiment", "neutral")
                        }
                        for r in reviews
                    ]
                })
            
            positioning_analysis = {}
            if competitors_data_for_positioning:
                positioning_analysis = self.positioning_analyzer.analyze_competitive_landscape(
                    competitors_data_for_positioning
                )
                
                # Identify market saturation
                saturation_data = self.positioning_analyzer.identify_market_saturation(
                    positioning_analysis
                )
                positioning_analysis["saturation"] = saturation_data
                
                # Generate differentiation recommendations
                differentiation = self.positioning_analyzer.generate_differentiation_recommendations(
                    positioning_analysis,
                    product_idea
                )
                positioning_analysis["differentiation"] = differentiation
            
            stage_times["stage_2_review_analysis"] = (
                (time.time() - stage_start) * 1000
            )
            
            logger.info(
                f"Stage 2 complete in {stage_times['stage_2_review_analysis']:.2f}ms. "
                f"Analyzed {len(all_reviews_for_sentiment)} reviews"
            )

            
            # ===== STAGE 3: LLM Generation =====
            logger.info("=" * 60)
            logger.info("STAGE 3: LLM Generation")
            logger.info("=" * 60)
            
            stage_start = time.time()
            
            # Format context for LLM
            logger.info("Formatting context for LLM")
            competitors_data_for_llm = []
            for comp in competitors:
                product_id = comp["product_id"]
                metadata = competitor_metadata.get(product_id, {})
                reviews = competitor_reviews.get(product_id, [])
                
                competitors_data_for_llm.append({
                    "product_id": product_id,
                    "similarity_score": comp.get("similarity_score", 0.0),
                    "metadata": metadata,
                    "reviews": reviews
                })
            
            context = self.rag_retriever.format_context_for_llm(
                product_idea=product_idea,
                competitors_data=competitors_data_for_llm
            )
            
            # Generate competitive intelligence using LLM
            logger.info("Generating competitive intelligence with LLM")
            llm_insights = {}
            llm_error = None
            
            try:
                # Try to connect to LLM service (Groq)
                if not self.llm_generator.connect():
                    raise LLMConnectionError("Failed to connect to LLM service")
                
                # Generate insights
                llm_insights = self.llm_generator.generate_competitive_intelligence(
                    product_idea=product_idea,
                    competitors_context=context
                )
                
                logger.info("LLM generation successful")
                
            except (LLMConnectionError, LLMGenerationError) as e:
                logger.warning(
                    f"LLM generation failed: {e}. "
                    f"Continuing with available analysis data."
                )
                llm_error = str(e)
                
                # Provide fallback insights structure
                llm_insights = {
                    "market_positioning": {
                        "summary": "LLM unavailable - analysis based on retrieved data only"
                    },
                    "feature_gaps": {
                        "summary": "See structured feature analysis below"
                    },
                    "sentiment_summary": {
                        "summary": "See sentiment distribution below"
                    },
                    "recommendations": [],
                    "confidence_score": 0.5,
                    "error": llm_error
                }
            
            stage_times["stage_3_llm_generation"] = (
                (time.time() - stage_start) * 1000
            )
            
            logger.info(
                f"Stage 3 complete in {stage_times['stage_3_llm_generation']:.2f}ms"
            )
            
            # ===== STAGE 4: Report Generation =====
            logger.info("=" * 60)
            logger.info("STAGE 4: Report Generation")
            logger.info("=" * 60)
            
            stage_start = time.time()
            
            # Calculate total processing time
            total_processing_time = (time.time() - pipeline_start_time) * 1000
            
            # Generate comprehensive report
            logger.info("Generating comprehensive report")
            report = self.report_generator.generate_comprehensive_report(
                product_idea=product_idea,
                competitors_data=competitors_data_for_llm,
                llm_insights=llm_insights,
                feature_analysis=feature_analysis,
                positioning_analysis=positioning_analysis,
                processing_time_ms=int(total_processing_time)
            )
            
            stage_times["stage_4_report_generation"] = (
                (time.time() - stage_start) * 1000
            )
            
            logger.info(
                f"Stage 4 complete in {stage_times['stage_4_report_generation']:.2f}ms"
            )
            
            # ===== Pipeline Complete =====
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            
            # Log stage timing breakdown
            logger.info("Stage timing breakdown:")
            for stage_name, duration in stage_times.items():
                logger.info(f"  {stage_name}: {duration:.2f}ms")
            logger.info(f"  TOTAL: {total_processing_time:.2f}ms")
            
            # Prepare response
            response = {
                "analysis_id": analysis_id,
                "status": "completed",
                "product_idea": product_idea,
                "competitors_identified": [
                    comp.get("metadata", {}).get("name", comp.get("product_id", "Unknown"))
                    for comp in competitors_data_for_llm
                ],
                "results": report.get("results", {}),
                "confidence_score": llm_insights.get("confidence_score", 0.5),
                "generated_at": datetime.now().isoformat(),
                "processing_time_ms": int(total_processing_time),
                "stage_times": stage_times,
                "full_report": report
            }
            
            # Add warning if LLM failed
            if llm_error:
                response["warnings"] = [
                    f"LLM generation failed: {llm_error}. "
                    f"Report contains only structured analysis."
                ]
            
            logger.info(
                f"Analysis complete (ID: {analysis_id}). "
                f"Total time: {total_processing_time:.2f}ms"
            )
            
            return response
            
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            raise
        except AnalysisPipelineError as e:
            logger.error(f"Pipeline error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during pipeline execution: {e}",
                exc_info=True
            )
            
            # Return error response
            total_time = (time.time() - pipeline_start_time) * 1000
            
            return {
                "analysis_id": analysis_id,
                "status": "failed",
                "product_idea": product_idea,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": int(total_time),
                "stage_times": stage_times
            }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Retrieve and return dataset statistics from loaded indices.
        
        Returns:
            Dictionary with dataset statistics including:
                - product_index_size: Number of products in index
                - review_index_size: Number of reviews in index
                - indices_loaded: Whether indices are loaded
                - product_index_path: Path to product index
                - review_index_path: Path to review index
                
        Example:
            >>> pipeline = AnalysisPipeline()
            >>> pipeline.load_indices("data/indices/products", "data/indices/reviews")
            >>> stats = pipeline.get_dataset_stats()
            >>> stats["product_index_size"]
            1000
        """
        logger.info("Retrieving dataset statistics")
        
        stats = {
            "indices_loaded": self.indices_loaded,
            "product_index_path": self.product_index_path,
            "review_index_path": self.review_index_path
        }
        
        if self.indices_loaded:
            stats["product_index_size"] = self.product_index.get_index_size()
            stats["review_index_size"] = self.review_index.get_index_size()
            
            logger.info(
                f"Dataset stats: {stats['product_index_size']} products, "
                f"{stats['review_index_size']} reviews"
            )
        else:
            stats["product_index_size"] = 0
            stats["review_index_size"] = 0
            logger.warning("Indices not loaded, returning zero counts")
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of AnalysisPipeline."""
        return (
            f"AnalysisPipeline("
            f"indices_loaded={self.indices_loaded}, "
            f"products={self.product_index.get_index_size() if self.indices_loaded else 0}, "
            f"reviews={self.review_index.get_index_size() if self.indices_loaded else 0})"
        )


def main():
    """
    Main entry point for command-line execution.
    
    Example usage:
        python -m src.main
    """
    import sys
    
    # Setup basic logging
    Logger.setup(log_level="INFO")
    
    logger.info("Product Hunt RAG Analyzer - Main Orchestration Layer")
    logger.info("=" * 60)
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = AnalysisPipeline()
        
        # Load indices
        product_index_path = "data/indices/products"
        review_index_path = "data/indices/reviews"
        
        logger.info("Loading indices...")
        pipeline.load_indices(product_index_path, review_index_path)
        
        # Get dataset stats
        stats = pipeline.get_dataset_stats()
        logger.info(f"Dataset loaded: {stats}")
        
        # Example analysis
        product_idea = (
            "A task management app with AI-powered prioritization "
            "and natural language input"
        )
        
        logger.info(f"Running analysis for: {product_idea}")
        result = pipeline.run_analysis(
            product_idea=product_idea,
            max_competitors=5,
            output_format="json"
        )
        
        logger.info(f"Analysis complete: {result['status']}")
        logger.info(f"Competitors identified: {result['competitors_identified']}")
        logger.info(f"Confidence score: {result['confidence_score']:.2f}")
        logger.info(f"Processing time: {result['processing_time_ms']}ms")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
