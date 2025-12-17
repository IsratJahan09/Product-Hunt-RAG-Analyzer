"""
Feature Gap Analysis router for Product Hunt RAG Analyzer.

This module provides endpoints for comprehensive feature gap analysis that:
1. Analyzes product reviews to identify missing features
2. Falls back to LLM-generated suggestions when needed
3. Returns a comprehensive feature gap analysis report
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from src.api.models import (
    FeatureGapAnalysisRequest,
    FeatureGapAnalysisResponse,
    FeatureGapItem,
    FeatureGapRecommendation,
    GapSourceType,
    ErrorResponse
)
from src.main import AnalysisPipeline
from src.modules.feature_gap_service import FeatureGapService, GapSource
from src.utils.logger import get_logger


logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/feature-gaps",
    tags=["feature-gaps"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)

# Global instances (will be set by app startup)
_pipeline: AnalysisPipeline = None
_feature_gap_service: FeatureGapService = None


def set_pipeline(pipeline: AnalysisPipeline):
    """Set the global pipeline instance."""
    global _pipeline, _feature_gap_service
    _pipeline = pipeline
    
    # Initialize feature gap service with pipeline's components
    if pipeline:
        _feature_gap_service = FeatureGapService(
            feature_analyzer=pipeline.feature_analyzer,
            llm_generator=pipeline.llm_generator,
            min_gaps_threshold=3
        )
        logger.info("FeatureGapService initialized with pipeline components")


def get_feature_gap_service() -> FeatureGapService:
    """Get the feature gap service instance."""
    if _feature_gap_service is None:
        logger.error("FeatureGapService not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "SERVICE_UNAVAILABLE",
                "message": "Feature gap analysis service not initialized."
            }
        )
    return _feature_gap_service


@router.post(
    "/analyze",
    response_model=FeatureGapAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Feature Gaps",
    description="""
    Perform comprehensive feature gap analysis for a product.
    
    The analysis follows a two-step process:
    1. **Review Analysis**: Analyzes product reviews to identify missing features,
       user complaints, and areas where the product falls short.
    2. **LLM Fallback**: If insufficient gaps are found from reviews, generates
       creative feature suggestions using AI based on product context.
    
    The response includes:
    - Gaps identified from actual user feedback (prioritized by frequency and sentiment)
    - AI-generated feature suggestions (when applicable)
    - Actionable recommendations for product improvement
    - Confidence score indicating reliability of the analysis
    """,
    response_description="Comprehensive feature gap analysis report"
)
async def analyze_feature_gaps(
    request: FeatureGapAnalysisRequest
) -> FeatureGapAnalysisResponse:
    """
    Analyze feature gaps for a product.
    
    Args:
        request: Feature gap analysis request with product details
        
    Returns:
        FeatureGapAnalysisResponse with identified gaps and recommendations
    """
    logger.info(
        f"Received feature gap analysis request for '{request.product_name}'"
    )
    
    service = get_feature_gap_service()
    
    try:
        # Get reviews for analysis
        reviews = []
        
        if _pipeline and _pipeline.indices_loaded and request.product_id:
            # Try to retrieve reviews from index
            try:
                reviews = _get_reviews_for_product(request.product_id)
                logger.info(f"Retrieved {len(reviews)} reviews from index")
            except Exception as e:
                logger.warning(f"Failed to retrieve reviews from index: {e}")
        
        # If no reviews from index, use sample reviews for demo
        if not reviews:
            logger.info("No reviews from index, using provided context for analysis")
            # Create synthetic review context from product description
            reviews = _create_context_reviews(
                request.product_name,
                request.product_description
            )
        
        # Update service threshold if specified
        if request.min_gaps_threshold != service.min_gaps_threshold:
            service.min_gaps_threshold = request.min_gaps_threshold
        
        # Perform feature gap analysis
        report = service.analyze_feature_gaps(
            product_name=request.product_name,
            product_description=request.product_description,
            reviews=reviews,
            existing_features=request.existing_features,
            competitor_features=None  # Could be extended to accept this
        )
        
        # Convert to response model
        response = _convert_report_to_response(report)
        
        logger.info(
            f"Feature gap analysis completed for '{request.product_name}'. "
            f"Source: {response.source.value}, "
            f"Gaps from reviews: {len(response.gaps_from_reviews)}, "
            f"LLM suggestions: {len(response.llm_generated_suggestions)}, "
            f"Confidence: {response.confidence_score:.2f}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature gap analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "ANALYSIS_FAILED",
                "message": f"Feature gap analysis failed: {str(e)}"
            }
        )


@router.post(
    "/analyze-with-reviews",
    response_model=FeatureGapAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Feature Gaps with Custom Reviews",
    description="""
    Perform feature gap analysis using provided reviews.
    
    This endpoint allows you to provide your own review data for analysis,
    useful when you have reviews from external sources or want to analyze
    specific feedback.
    """,
    response_description="Feature gap analysis report based on provided reviews"
)
async def analyze_feature_gaps_with_reviews(
    request: FeatureGapAnalysisRequest,
    reviews: List[Dict[str, Any]]
) -> FeatureGapAnalysisResponse:
    """
    Analyze feature gaps using provided reviews.
    
    Args:
        request: Feature gap analysis request
        reviews: List of review dictionaries with 'text' and optional 'sentiment'
        
    Returns:
        FeatureGapAnalysisResponse with analysis results
    """
    logger.info(
        f"Received feature gap analysis with {len(reviews)} custom reviews "
        f"for '{request.product_name}'"
    )
    
    service = get_feature_gap_service()
    
    try:
        # Validate reviews
        if not reviews:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "INVALID_REVIEWS",
                    "message": "At least one review is required for analysis"
                }
            )
        
        # Update service threshold
        service.min_gaps_threshold = request.min_gaps_threshold
        
        # Perform analysis
        report = service.analyze_feature_gaps(
            product_name=request.product_name,
            product_description=request.product_description,
            reviews=reviews,
            existing_features=request.existing_features
        )
        
        response = _convert_report_to_response(report)
        
        logger.info(
            f"Feature gap analysis with custom reviews completed. "
            f"Analyzed {len(reviews)} reviews, found {len(response.gaps_from_reviews)} gaps"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature gap analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "ANALYSIS_FAILED",
                "message": f"Feature gap analysis failed: {str(e)}"
            }
        )


def _get_reviews_for_product(product_id: str) -> List[Dict[str, Any]]:
    """Retrieve reviews for a product from the index."""
    if not _pipeline or not _pipeline.indices_loaded:
        return []
    
    try:
        # Use RAG retriever to get reviews
        reviews_data = _pipeline.rag_retriever.retrieve_competitor_reviews(
            competitor_ids=[product_id],
            k_per_competitor=50  # Get more reviews for better analysis
        )
        
        reviews = []
        for product_reviews in reviews_data.values():
            for review in product_reviews:
                review_text = review.get("metadata", {}).get("original_text", "")
                if review_text:
                    reviews.append({
                        "text": review_text,
                        "sentiment": review.get("sentiment", "neutral"),
                        "confidence": review.get("confidence", 0.5)
                    })
        
        return reviews
        
    except Exception as e:
        logger.error(f"Failed to retrieve reviews: {e}")
        return []


def _create_context_reviews(
    product_name: str,
    product_description: str
) -> List[Dict[str, Any]]:
    """Create context-based synthetic reviews for analysis when no real reviews available."""
    # This provides minimal context for LLM to generate suggestions
    # In production, you'd want real reviews
    return [
        {
            "text": f"I've been using {product_name} and it's good for {product_description}",
            "sentiment": "positive",
            "confidence": 0.5
        }
    ]


def _convert_report_to_response(report) -> FeatureGapAnalysisResponse:
    """Convert FeatureGapReport to FeatureGapAnalysisResponse."""
    
    # Convert source enum
    source_map = {
        GapSource.REVIEW_ANALYSIS: GapSourceType.REVIEW_ANALYSIS,
        GapSource.LLM_GENERATED: GapSourceType.LLM_GENERATED,
        GapSource.HYBRID: GapSourceType.HYBRID
    }
    
    # Convert gaps from reviews
    gaps_from_reviews = [
        FeatureGapItem(
            name=g.name,
            description=g.description,
            category=g.category,
            priority=g.priority,
            source=source_map.get(g.source, GapSourceType.REVIEW_ANALYSIS),
            evidence=g.evidence,
            frequency=g.frequency,
            sentiment_score=g.sentiment_score,
            confidence=g.confidence
        )
        for g in report.gaps_from_reviews
    ]
    
    # Convert LLM suggestions
    llm_suggestions = [
        FeatureGapItem(
            name=g.name,
            description=g.description,
            category=g.category,
            priority=g.priority,
            source=source_map.get(g.source, GapSourceType.LLM_GENERATED),
            evidence=g.evidence,
            frequency=g.frequency,
            sentiment_score=g.sentiment_score,
            confidence=g.confidence
        )
        for g in report.llm_generated_suggestions
    ]
    
    # Convert recommendations
    recommendations = [
        FeatureGapRecommendation(
            id=r["id"],
            title=r["title"],
            description=r["description"],
            priority=r["priority"],
            category=r["category"],
            source=r["source"],
            rationale=r["rationale"],
            evidence=r.get("evidence", [])
        )
        for r in report.recommendations
    ]
    
    return FeatureGapAnalysisResponse(
        product_name=report.product_name,
        product_description=report.product_description,
        total_reviews_analyzed=report.total_reviews_analyzed,
        source=source_map.get(report.source, GapSourceType.REVIEW_ANALYSIS),
        summary=report.summary,
        confidence_score=report.confidence_score,
        gaps_from_reviews=gaps_from_reviews,
        llm_generated_suggestions=llm_suggestions,
        recommendations=recommendations,
        metadata=report.metadata
    )
