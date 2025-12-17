"""
Admin router for system management endpoints.

This module provides endpoints for health checks, dataset statistics,
index rebuilding, and system metrics.
"""

import time
import psutil
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends

from src.api.models import (
    HealthResponse,
    DatasetStatsResponse,
    IndexRebuildRequest,
    IndexRebuildResponse,
    MetricsResponse,
    ErrorResponse
)
from src.main import AnalysisPipeline
from src.utils.index_builder import IndexBuilder
from src.utils.dataset_loader import DatasetLoader
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_collector
from src.modules.llm_generation import LLMConnectionError


logger = get_logger(__name__)
metrics_collector = get_metrics_collector()

# Create router
router = APIRouter(
    prefix="",
    tags=["admin"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)

# Global pipeline instance and metrics
_pipeline: AnalysisPipeline = None
_server_start_time = time.time()
_metrics = {
    "api_calls": 0,
    "total_latency_ms": 0.0,
    "error_count": 0
}


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
                "message": "Analysis pipeline not initialized"
            }
        )
    return _pipeline


def track_api_call(latency_ms: float, is_error: bool = False):
    """
    Track API call metrics.
    
    Args:
        latency_ms: Request latency in milliseconds
        is_error: Whether the request resulted in an error
    """
    global _metrics
    _metrics["api_calls"] += 1
    _metrics["total_latency_ms"] += latency_ms
    if is_error:
        _metrics["error_count"] += 1


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the API and its dependencies (Groq LLM API, FAISS indices).",
    response_description="System health status"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Checks:
    - API is running
    - Groq LLM service is connected
    - FAISS indices are loaded
    
    Returns:
        HealthResponse with system status
    """
    logger.debug("Health check requested")
    
    # Check if pipeline is initialized
    if _pipeline is None:
        logger.warning("Pipeline not initialized")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            ollama_connected=False,
            indices_loaded=False
        )
    
    # Check LLM connection (Groq API)
    llm_connected = False
    try:
        _pipeline.llm_generator.connect()
        llm_connected = True
        logger.debug("LLM connection: OK")
    except LLMConnectionError as e:
        logger.warning(f"LLM connection failed: {e}")
        llm_connected = False
    except Exception as e:
        logger.error(f"Unexpected error checking LLM: {e}")
        llm_connected = False
    
    # Check indices
    indices_loaded = _pipeline.indices_loaded
    logger.debug(f"Indices loaded: {indices_loaded}")
    
    # Determine overall status
    if indices_loaded and llm_connected:
        overall_status = "healthy"
    elif indices_loaded:
        overall_status = "degraded"  # Can work without LLM
    else:
        overall_status = "unhealthy"
    
    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        ollama_connected=llm_connected,
        indices_loaded=indices_loaded
    )
    
    logger.info(
        f"Health check: status={overall_status}, "
        f"llm={llm_connected}, indices={indices_loaded}"
    )
    
    return response


@router.get(
    "/dataset/stats",
    response_model=DatasetStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Dataset Statistics",
    description="Retrieve statistics about the loaded dataset including product and review counts.",
    response_description="Dataset statistics"
)
async def get_dataset_stats() -> DatasetStatsResponse:
    """
    Get dataset statistics from loaded indices.
    
    Returns:
        DatasetStatsResponse with dataset statistics
        
    Raises:
        HTTPException 503: If indices are not loaded
    """
    logger.info("Dataset statistics requested")
    
    # Get pipeline
    pipeline = get_pipeline()
    
    try:
        # Get stats from pipeline
        stats = pipeline.get_dataset_stats()
        
        # Calculate average reviews per product
        total_products = stats.get("product_index_size", 0)
        total_reviews = stats.get("review_index_size", 0)
        
        avg_reviews = 0.0
        if total_products > 0:
            avg_reviews = total_reviews / total_products
        
        response = DatasetStatsResponse(
            total_products=total_products,
            total_reviews=total_reviews,
            avg_reviews_per_product=round(avg_reviews, 2),
            date_range=None,  # Could be extracted from metadata if available
            indices_loaded=stats.get("indices_loaded", False)
        )
        
        logger.info(
            f"Dataset stats: products={total_products}, "
            f"reviews={total_reviews}, avg={avg_reviews:.2f}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving dataset stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "STATS_ERROR",
                "message": f"Failed to retrieve dataset statistics: {str(e)}"
            }
        )


@router.post(
    "/index/rebuild",
    response_model=IndexRebuildResponse,
    status_code=status.HTTP_200_OK,
    summary="Rebuild FAISS Indices",
    description="Rebuild FAISS indices from a provided dataset file. This operation may take several minutes.",
    response_description="Index rebuild results"
)
async def rebuild_indices(
    request: IndexRebuildRequest
) -> IndexRebuildResponse:
    """
    Rebuild FAISS indices from dataset.
    
    This endpoint rebuilds both product and review indices from
    a provided dataset file. The operation may take several minutes
    depending on dataset size.
    
    Args:
        request: Index rebuild request with dataset path and index type
        
    Returns:
        IndexRebuildResponse with rebuild results
        
    Raises:
        HTTPException 400: Invalid dataset path or file not found
        HTTPException 500: Index building failed
    """
    logger.info(
        f"Index rebuild requested: dataset={request.dataset_path}, "
        f"type={request.index_type}"
    )
    
    # Get pipeline after request validation
    pipeline = get_pipeline()
    
    start_time = time.time()
    
    try:
        # Validate dataset path
        dataset_path = Path(request.dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {request.dataset_path}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "FILE_NOT_FOUND",
                    "message": f"Dataset file not found: {request.dataset_path}"
                }
            )
        
        # Load dataset
        logger.info("Loading dataset")
        dataset_loader = DatasetLoader()
        dataset = dataset_loader.load_dataset(str(dataset_path))
        
        # Validate dataset
        is_valid, errors = dataset_loader.validate_dataset_structure(dataset)
        if not is_valid:
            logger.error(f"Invalid dataset structure: {errors}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "INVALID_DATASET",
                    "message": "Dataset validation failed",
                    "details": {"errors": errors}
                }
            )
        
        # Build indices
        logger.info("Building indices")
        index_builder = IndexBuilder(
            embedding_generator=pipeline.embedding_generator,
            text_preprocessor=pipeline.text_preprocessor
        )
        
        # Determine output paths
        output_dir = Path(pipeline.config.get("storage.indices_dir", "./data/indices"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        product_index_path = str(output_dir / "products")
        review_index_path = str(output_dir / "reviews")
        
        # Build product index
        logger.info("Building product index")
        product_result = index_builder.build_product_index(
            dataset=dataset,
            output_path=product_index_path,
            index_type=request.index_type.value
        )
        
        # Build review index
        logger.info("Building review index")
        review_result = index_builder.build_review_index(
            dataset=dataset,
            output_path=review_index_path,
            index_type=request.index_type.value
        )
        
        # Reload indices in pipeline
        logger.info("Reloading indices in pipeline")
        pipeline.load_indices(product_index_path, review_index_path)
        
        build_time_ms = int((time.time() - start_time) * 1000)
        
        response = IndexRebuildResponse(
            success=True,
            message="Indices rebuilt successfully",
            product_index_size=product_result.get("index_size", 0),
            review_index_size=review_result.get("index_size", 0),
            build_time_ms=build_time_ms
        )
        
        logger.info(
            f"Index rebuild complete: products={response.product_index_size}, "
            f"reviews={response.review_index_size}, time={build_time_ms}ms"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "REBUILD_FAILED",
                "message": f"Failed to rebuild indices: {str(e)}"
            }
        )


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Get System Metrics",
    description="Retrieve comprehensive system metrics including operation timings, memory usage, and performance statistics.",
    response_description="System metrics"
)
async def get_metrics() -> Dict[str, Any]:
    """
    Get comprehensive system metrics.
    
    Returns detailed metrics about:
    - Operation timings (embedding generation, FAISS search, LLM generation, etc.)
    - Memory usage (current, peak, available)
    - API call statistics
    - Server uptime
    
    Returns:
        Dictionary with comprehensive metrics summary
    """
    logger.debug("Metrics requested")
    
    try:
        # Get comprehensive metrics from collector
        metrics_summary = metrics_collector.get_metrics_summary()
        
        # Add API-specific metrics
        avg_latency = 0.0
        if _metrics["api_calls"] > 0:
            avg_latency = _metrics["total_latency_ms"] / _metrics["api_calls"]
        
        metrics_summary["api"] = {
            "total_calls": _metrics["api_calls"],
            "avg_latency_ms": round(avg_latency, 2),
            "error_count": _metrics["error_count"],
            "error_rate": round(_metrics["error_count"] / _metrics["api_calls"] * 100, 2) if _metrics["api_calls"] > 0 else 0.0
        }
        
        # Add server uptime
        uptime_seconds = int(time.time() - _server_start_time)
        metrics_summary["server"] = {
            "uptime_seconds": uptime_seconds,
            "uptime_hours": round(uptime_seconds / 3600, 2)
        }
        
        logger.debug(
            f"Metrics summary: operations={len(metrics_summary.get('operations', {}))}, "
            f"api_calls={_metrics['api_calls']}, "
            f"uptime={uptime_seconds}s"
        )
        
        return metrics_summary
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "METRICS_ERROR",
                "message": f"Failed to retrieve metrics: {str(e)}"
            }
        )
