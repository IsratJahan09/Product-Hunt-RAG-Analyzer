"""
FastAPI application for Product Hunt RAG Analyzer.

This module initializes the FastAPI application, registers routers,
configures middleware, and handles application lifecycle events.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from dotenv import load_dotenv

from src.api.routers import analysis, admin, feature_gaps
from src.main import AnalysisPipeline
from src.utils.config import ConfigManager
from src.utils.logger import get_logger, Logger

# Load environment variables from .env file
load_dotenv()

logger = get_logger(__name__)

# Global pipeline instance
_pipeline: AnalysisPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize pipeline and load indices
    - Shutdown: Cleanup resources
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Product Hunt RAG Analyzer API")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = ConfigManager()
        
        # Setup logging from config
        Logger.setup_from_config(config)
        
        # Initialize pipeline
        logger.info("Initializing analysis pipeline")
        global _pipeline
        _pipeline = AnalysisPipeline(config=config)
        
        # Set pipeline in routers
        analysis.set_pipeline(_pipeline)
        admin.set_pipeline(_pipeline)
        feature_gaps.set_pipeline(_pipeline)
        
        # Load FAISS indices
        indices_dir = Path(config.get("storage.indices_dir", "./data/indices"))
        product_index_path = str(indices_dir / "products")
        review_index_path = str(indices_dir / "reviews")
        
        logger.info("Loading FAISS indices")
        logger.info(f"  Product index: {product_index_path}")
        logger.info(f"  Review index: {review_index_path}")
        
        try:
            load_result = _pipeline.load_indices(
                product_index_path=product_index_path,
                review_index_path=review_index_path
            )
            
            logger.info(
                f"Indices loaded successfully: "
                f"products={load_result['product_index_size']}, "
                f"reviews={load_result['review_index_size']}, "
                f"time={load_result['load_time_ms']:.2f}ms"
            )
            
        except FileNotFoundError as e:
            logger.warning(
                f"Index files not found: {e}. "
                f"API will start but analysis endpoints will be unavailable. "
                f"Use /api/v1/index/rebuild to build indices."
            )
        except Exception as e:
            logger.error(f"Failed to load indices: {e}", exc_info=True)
            logger.warning(
                "API will start but analysis endpoints may not work properly. "
                "Check logs and rebuild indices if needed."
            )
        
        logger.info("=" * 60)
        logger.info("API startup complete")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Product Hunt RAG Analyzer API")
    logger.info("=" * 60)
    
    try:
        # Cleanup resources
        logger.info("Cleaning up resources")
        
        # Clear pipeline reference
        _pipeline = None
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI application
def create_app(config: ConfigManager = None) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config: Optional configuration manager. If None, loads default config.
        
    Returns:
        Configured FastAPI application instance
    """
    # Load config if not provided
    if config is None:
        config = ConfigManager()
    
    # Get FastAPI settings from config
    title = config.get("fastapi.title", "Product Hunt RAG Analyzer")
    description = config.get(
        "fastapi.description",
        "AI-powered competitive analysis for Product Hunt products"
    )
    version = config.get("fastapi.version", "1.0.0")
    
    # Create app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Configure CORS
    cors_enabled = config.get("fastapi.cors.enabled", True)
    if cors_enabled:
        origins = config.get(
            "fastapi.cors.origins",
            ["http://localhost:3000", "http://localhost:8000"]
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=config.get("fastapi.cors.allow_credentials", True),
            allow_methods=config.get("fastapi.cors.allow_methods", ["*"]),
            allow_headers=config.get("fastapi.cors.allow_headers", ["*"])
        )
        
        logger.info(f"CORS enabled for origins: {origins}")
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Middleware to log all requests and responses.
        
        Adds request_id to each request and logs timing information.
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Status: {response.status_code}, Duration: {duration_ms:.2f}ms"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Track metrics
            admin.track_api_call(
                latency_ms=duration_ms,
                is_error=(response.status_code >= 400)
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Error: {e}, Duration: {duration_ms:.2f}ms",
                exc_info=True
            )
            
            # Track error
            admin.track_api_call(latency_ms=duration_ms, is_error=True)
            
            raise
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors.
        
        Returns a structured error response with field-level details.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.warning(
            f"[{request_id}] Validation error: {exc.errors()}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error_code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "errors": exc.errors(),
                    "body": exc.body if hasattr(exc, "body") else None
                }
            }
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(
        request: Request,
        exc: ValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic model validation errors.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.warning(
            f"[{request_id}] Pydantic validation error: {exc.errors()}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error_code": "VALIDATION_ERROR",
                "message": "Data validation failed",
                "details": {"errors": exc.errors()}
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """
        Handle all uncaught exceptions.
        
        Returns a generic error response and logs the full exception.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.error(
            f"[{request_id}] Unhandled exception: {exc}",
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {
                    "error": str(exc),
                    "type": type(exc).__name__
                }
            }
        )
    
    # Register routers
    app.include_router(
        analysis.router,
        prefix="/api/v1"
    )
    
    app.include_router(
        admin.router,
        prefix="/api/v1"
    )
    
    app.include_router(
        feature_gaps.router,
        prefix="/api/v1"
    )
    
    # Root endpoint
    @app.get(
        "/",
        tags=["root"],
        summary="API Root",
        description="Get basic API information"
    )
    async def root() -> Dict[str, Any]:
        """
        Root endpoint with API information.
        
        Returns:
            Dictionary with API name, version, and documentation links
        """
        return {
            "name": title,
            "version": version,
            "description": description,
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    
    logger.info(f"FastAPI application created: {title} v{version}")
    
    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    """
    Run the application with uvicorn.
    
    For development:
        python -m src.api.app
        
    For production:
        uvicorn src.api.app:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    # Load config
    config = ConfigManager()
    
    # Get server settings
    host = config.get("fastapi.host", "0.0.0.0")
    port = config.get("fastapi.port", 8000)
    reload = config.get("fastapi.reload", False)
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
