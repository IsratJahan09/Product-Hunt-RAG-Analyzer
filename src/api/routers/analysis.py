"""
Analysis router for competitive intelligence endpoints.

This module provides endpoints for submitting product ideas for analysis
and retrieving analysis results.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime

from src.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    ErrorResponse
)
from src.main import AnalysisPipeline, AnalysisPipelineError
from src.utils.logger import get_logger


logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/analyze",
    tags=["analysis"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)

# Global pipeline instance (will be set by app startup)
_pipeline: AnalysisPipeline = None


def set_pipeline(pipeline: AnalysisPipeline):
    """Set the global pipeline instance."""
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> AnalysisPipeline:
    """
    Dependency to get the pipeline instance.
    
    Raises:
        HTTPException: If pipeline is not initialized
    """
    if _pipeline is None:
        logger.error("Pipeline not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "SERVICE_UNAVAILABLE",
                "message": "Analysis pipeline not initialized. Please try again later."
            }
        )
    return _pipeline


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Product Idea",
    description="Submit a product idea for competitive analysis and receive insights about competitors, feature gaps, market positioning, and recommendations.",
    response_description="Comprehensive competitive analysis results"
)
async def analyze_product_idea(
    request: AnalysisRequest
) -> AnalysisResponse:
    """
    Analyze a product idea and generate competitive intelligence.
    
    This endpoint executes the full analysis pipeline:
    1. Identifies relevant competitor products
    2. Retrieves and analyzes competitor reviews
    3. Generates insights using LLM
    4. Creates comprehensive report
    
    Args:
        request: Analysis request with product idea and parameters
        
    Returns:
        AnalysisResponse with complete analysis results
        
    Raises:
        HTTPException 400: Invalid request parameters
        HTTPException 503: Service unavailable (indices not loaded)
        HTTPException 500: Internal server error
    """
    logger.info(
        f"Received analysis request: product_idea='{request.product_idea[:50]}...', "
        f"max_competitors={request.max_competitors}, "
        f"output_format={request.output_format}"
    )
    
    # Get pipeline after request validation
    pipeline = get_pipeline()
    
    try:
        # Check if indices are loaded
        if not pipeline.indices_loaded:
            logger.error("Indices not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "INDICES_NOT_LOADED",
                    "message": "FAISS indices are not loaded. Please contact administrator."
                }
            )
        
        # Execute analysis pipeline
        logger.info("Executing analysis pipeline")
        result = pipeline.run_analysis(
            product_idea=request.product_idea,
            max_competitors=request.max_competitors,
            output_format=request.output_format.value
        )
        
        # Check if analysis failed
        if result.get("status") == "failed":
            logger.error(f"Analysis failed: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "ANALYSIS_FAILED",
                    "message": f"Analysis failed: {result.get('error')}",
                    "details": {
                        "error_type": result.get("error_type"),
                        "analysis_id": result.get("analysis_id")
                    }
                }
            )
        
        # Convert result to response model
        response = AnalysisResponse(
            analysis_id=result["analysis_id"],
            status=AnalysisStatus.COMPLETED,
            product_idea=result["product_idea"],
            competitors_identified=result.get("competitors_identified", []),
            results=result.get("results", {}),
            confidence_score=result.get("confidence_score", 0.5),
            generated_at=datetime.fromisoformat(result["generated_at"]),
            processing_time_ms=result["processing_time_ms"],
            warnings=result.get("warnings")
        )
        
        logger.info(
            f"Analysis completed successfully: analysis_id={response.analysis_id}, "
            f"competitors={len(response.competitors_identified)}, "
            f"confidence={response.confidence_score:.2f}, "
            f"time={response.processing_time_ms}ms"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except AnalysisPipelineError as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "PIPELINE_ERROR",
                "message": f"Analysis pipeline error: {str(e)}"
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "VALIDATION_ERROR",
                "message": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred during analysis",
                "details": {"error": str(e)}
            }
        )


# Optional: Cache for storing analysis results
# This is a simple in-memory cache - in production, use Redis or similar
_analysis_cache: Dict[str, Dict[str, Any]] = {}


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Analysis Results",
    description="Retrieve cached analysis results by analysis ID (optional caching feature).",
    response_description="Cached analysis results"
)
async def get_analysis_results(analysis_id: str) -> AnalysisResponse:
    """
    Retrieve cached analysis results by ID.
    
    This is an optional caching feature that allows retrieving
    previously completed analyses without re-running the pipeline.
    
    Args:
        analysis_id: Unique identifier of the analysis
        
    Returns:
        AnalysisResponse with cached results
        
    Raises:
        HTTPException 404: Analysis not found in cache
    """
    logger.info(f"Retrieving cached analysis: {analysis_id}")
    
    # Check cache
    if analysis_id not in _analysis_cache:
        logger.warning(f"Analysis not found in cache: {analysis_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "ANALYSIS_NOT_FOUND",
                "message": f"Analysis with ID '{analysis_id}' not found in cache",
                "details": {
                    "analysis_id": analysis_id,
                    "note": "Caching is optional and results may expire"
                }
            }
        )
    
    # Retrieve from cache
    cached_result = _analysis_cache[analysis_id]
    
    response = AnalysisResponse(
        analysis_id=cached_result["analysis_id"],
        status=AnalysisStatus.COMPLETED,
        product_idea=cached_result["product_idea"],
        competitors_identified=cached_result.get("competitors_identified", []),
        results=cached_result.get("results", {}),
        confidence_score=cached_result.get("confidence_score", 0.5),
        generated_at=datetime.fromisoformat(cached_result["generated_at"]),
        processing_time_ms=cached_result["processing_time_ms"],
        warnings=cached_result.get("warnings")
    )
    
    logger.info(f"Retrieved cached analysis: {analysis_id}")
    return response


def cache_analysis_result(result: Dict[str, Any]):
    """
    Cache an analysis result for later retrieval.
    
    Args:
        result: Analysis result dictionary
    """
    analysis_id = result.get("analysis_id")
    if analysis_id and result.get("status") == "completed":
        _analysis_cache[analysis_id] = result
        logger.debug(f"Cached analysis result: {analysis_id}")
