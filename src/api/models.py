"""
Pydantic models for FastAPI request/response validation.

This module defines all data models used in the API endpoints for
request validation, response serialization, and documentation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class OutputFormat(str, Enum):
    """Supported output formats for analysis reports."""
    JSON = "json"
    MARKDOWN = "markdown"
    PDF = "pdf"


class AnalysisStatus(str, Enum):
    """Status of an analysis request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexType(str, Enum):
    """Supported FAISS index types."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"


class AnalysisRequest(BaseModel):
    """
    Request model for competitive analysis.
    
    Attributes:
        product_idea: Description of the product idea to analyze (min 10 chars)
        max_competitors: Number of competitors to analyze (1-20, default: 5)
        output_format: Report format (json/markdown/pdf, default: json)
    """
    product_idea: str = Field(
        ...,
        min_length=10,
        description="Product idea description to analyze",
        examples=["A task management app with AI-powered prioritization"]
    )
    max_competitors: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of competitors to analyze"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Output format for the analysis report"
    )
    
    @field_validator("product_idea")
    @classmethod
    def validate_product_idea(cls, v: str) -> str:
        """Validate product idea is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("product_idea cannot be empty or whitespace")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_idea": "A task management app with AI-powered prioritization and natural language input",
                    "max_competitors": 5,
                    "output_format": "json"
                }
            ]
        }
    }


class AnalysisResponse(BaseModel):
    """
    Response model for competitive analysis results.
    
    Attributes:
        analysis_id: Unique identifier for this analysis
        status: Current status of the analysis
        product_idea: Original product idea that was analyzed
        competitors_identified: List of competitor product names
        results: Analysis results with market positioning, feature gaps, etc.
        confidence_score: Overall confidence in the analysis (0.0-1.0)
        generated_at: Timestamp when analysis was generated
        processing_time_ms: Total processing time in milliseconds
    """
    analysis_id: str = Field(
        ...,
        description="Unique identifier for this analysis"
    )
    status: AnalysisStatus = Field(
        ...,
        description="Current status of the analysis"
    )
    product_idea: str = Field(
        ...,
        description="Original product idea that was analyzed"
    )
    competitors_identified: List[str] = Field(
        default_factory=list,
        description="List of identified competitor product names"
    )
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis results including market positioning, feature gaps, sentiment summary, and recommendations"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the analysis"
    )
    generated_at: datetime = Field(
        ...,
        description="Timestamp when analysis was generated"
    )
    processing_time_ms: int = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Any warnings generated during analysis"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "completed",
                    "product_idea": "A task management app with AI-powered prioritization",
                    "competitors_identified": ["Todoist", "TickTick", "Any.do"],
                    "results": {
                        "market_positioning": {},
                        "feature_gaps": {},
                        "sentiment_summary": {},
                        "recommendations": []
                    },
                    "confidence_score": 0.85,
                    "generated_at": "2024-01-15T10:30:00Z",
                    "processing_time_ms": 2500
                }
            ]
        }
    }


class IndexRebuildRequest(BaseModel):
    """
    Request model for rebuilding FAISS indices.
    
    Attributes:
        dataset_path: Path to the dataset file
        index_type: Type of FAISS index to create (flat/ivf/hnsw)
    """
    dataset_path: str = Field(
        ...,
        description="Path to the Product Hunt dataset file",
        examples=["./dataset/products.jsonl"]
    )
    index_type: IndexType = Field(
        default=IndexType.FLAT,
        description="Type of FAISS index to create"
    )
    
    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, v: str) -> str:
        """Validate dataset path is not empty."""
        if not v or not v.strip():
            raise ValueError("dataset_path cannot be empty")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_path": "./dataset/products.jsonl",
                    "index_type": "flat"
                }
            ]
        }
    }


class IndexRebuildResponse(BaseModel):
    """
    Response model for index rebuild operation.
    
    Attributes:
        success: Whether the rebuild was successful
        message: Status message
        product_index_size: Number of products indexed
        review_index_size: Number of reviews indexed
        build_time_ms: Time taken to build indices in milliseconds
    """
    success: bool = Field(..., description="Whether the rebuild was successful")
    message: str = Field(..., description="Status message")
    product_index_size: int = Field(
        default=0,
        description="Number of products indexed"
    )
    review_index_size: int = Field(
        default=0,
        description="Number of reviews indexed"
    )
    build_time_ms: int = Field(
        default=0,
        description="Time taken to build indices in milliseconds"
    )


class DatasetStatsResponse(BaseModel):
    """
    Response model for dataset statistics.
    
    Attributes:
        total_products: Total number of products in the index
        total_reviews: Total number of reviews in the index
        avg_reviews_per_product: Average reviews per product
        date_range: Date range of the dataset
        indices_loaded: Whether indices are currently loaded
    """
    total_products: int = Field(
        ...,
        description="Total number of products in the index"
    )
    total_reviews: int = Field(
        ...,
        description="Total number of reviews in the index"
    )
    avg_reviews_per_product: float = Field(
        default=0.0,
        description="Average number of reviews per product"
    )
    date_range: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Date range of the dataset"
    )
    indices_loaded: bool = Field(
        ...,
        description="Whether indices are currently loaded"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total_products": 1000,
                    "total_reviews": 5000,
                    "avg_reviews_per_product": 5.0,
                    "date_range": {
                        "earliest": "2023-01-01",
                        "latest": "2024-01-15"
                    },
                    "indices_loaded": True
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status: Overall system status
        timestamp: Current server timestamp
        version: API version
        ollama_connected: Whether Ollama LLM service is connected
        indices_loaded: Whether FAISS indices are loaded
    """
    status: str = Field(
        ...,
        description="Overall system status (healthy/degraded/unhealthy)"
    )
    timestamp: datetime = Field(
        ...,
        description="Current server timestamp"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    ollama_connected: bool = Field(
        ...,
        description="Whether Ollama LLM service is connected"
    )
    indices_loaded: bool = Field(
        ...,
        description="Whether FAISS indices are loaded"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "version": "1.0.0",
                    "ollama_connected": True,
                    "indices_loaded": True
                }
            ]
        }
    }


class MetricsResponse(BaseModel):
    """
    Response model for system metrics.
    
    Attributes:
        api_calls: Total number of API calls
        avg_latency_ms: Average API latency in milliseconds
        error_count: Total number of errors
        memory_usage_mb: Current memory usage in MB
        uptime_seconds: Server uptime in seconds
    """
    api_calls: int = Field(
        default=0,
        description="Total number of API calls"
    )
    avg_latency_ms: float = Field(
        default=0.0,
        description="Average API latency in milliseconds"
    )
    error_count: int = Field(
        default=0,
        description="Total number of errors"
    )
    memory_usage_mb: float = Field(
        default=0.0,
        description="Current memory usage in MB"
    )
    uptime_seconds: int = Field(
        default=0,
        description="Server uptime in seconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "api_calls": 150,
                    "avg_latency_ms": 2500.0,
                    "error_count": 2,
                    "memory_usage_mb": 512.5,
                    "uptime_seconds": 3600
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """
    Response model for error responses.
    
    Attributes:
        error_code: Error code identifier
        message: Human-readable error message
        details: Optional additional error details
    """
    error_code: str = Field(
        ...,
        description="Error code identifier"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional additional error details"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error_code": "VALIDATION_ERROR",
                    "message": "Invalid product_idea: must be at least 10 characters",
                    "details": {
                        "field": "product_idea",
                        "value_length": 5
                    }
                }
            ]
        }
    }


class GapSourceType(str, Enum):
    """Source of the identified feature gap."""
    REVIEW_ANALYSIS = "review_analysis"
    LLM_GENERATED = "llm_generated"
    HYBRID = "hybrid"


class FeatureGapItem(BaseModel):
    """
    Model for a single feature gap item.
    
    Attributes:
        name: Name of the feature gap
        description: Detailed description of the gap
        category: Feature category (ui_ux, performance, etc.)
        priority: Priority level (high, medium, low)
        source: Source of the gap identification
        evidence: Supporting evidence/quotes from reviews
        frequency: How often mentioned in reviews
        sentiment_score: Average sentiment when mentioned
        confidence: Confidence in this gap identification
    """
    name: str = Field(..., description="Name of the feature gap")
    description: str = Field(..., description="Detailed description of the gap")
    category: str = Field(..., description="Feature category")
    priority: str = Field(..., description="Priority level (high, medium, low)")
    source: GapSourceType = Field(..., description="Source of gap identification")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    frequency: int = Field(default=0, description="Mention frequency in reviews")
    sentiment_score: float = Field(default=0.0, description="Average sentiment score")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")


class FeatureGapRecommendation(BaseModel):
    """
    Model for a feature gap recommendation.
    
    Attributes:
        id: Recommendation ID
        title: Recommendation title
        description: Detailed description
        priority: Priority level
        category: Feature category
        source: Source of the recommendation
        rationale: Why this recommendation matters
        evidence: Supporting evidence
    """
    id: int = Field(..., description="Recommendation ID")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    priority: str = Field(..., description="Priority level")
    category: str = Field(..., description="Feature category")
    source: str = Field(..., description="Source of recommendation")
    rationale: str = Field(..., description="Why this recommendation matters")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")


class FeatureGapAnalysisRequest(BaseModel):
    """
    Request model for feature gap analysis.
    
    Attributes:
        product_name: Name of the product to analyze
        product_description: Description of the product
        product_id: Optional product ID for retrieving reviews
        existing_features: Optional list of known product features
        include_llm_suggestions: Whether to include LLM suggestions if gaps are insufficient
        min_gaps_threshold: Minimum gaps before triggering LLM fallback
    """
    product_name: str = Field(
        ...,
        min_length=2,
        description="Name of the product to analyze",
        examples=["TaskMaster Pro"]
    )
    product_description: str = Field(
        ...,
        min_length=10,
        description="Description of the product",
        examples=["A task management app with AI-powered prioritization"]
    )
    product_id: Optional[str] = Field(
        default=None,
        description="Optional product ID for retrieving reviews from index"
    )
    existing_features: Optional[List[str]] = Field(
        default=None,
        description="Optional list of known product features"
    )
    include_llm_suggestions: bool = Field(
        default=True,
        description="Whether to include LLM-generated suggestions when review gaps are insufficient"
    )
    min_gaps_threshold: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Minimum gaps from reviews before triggering LLM fallback"
    )
    
    @field_validator("product_name", "product_description")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate fields are not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_name": "TaskMaster Pro",
                    "product_description": "A task management app with AI-powered prioritization and team collaboration features",
                    "existing_features": ["Task creation", "Due dates", "Priority levels", "Team sharing"],
                    "include_llm_suggestions": True,
                    "min_gaps_threshold": 3
                }
            ]
        }
    }


class FeatureGapAnalysisResponse(BaseModel):
    """
    Response model for feature gap analysis.
    
    Attributes:
        product_name: Name of the analyzed product
        product_description: Description of the product
        total_reviews_analyzed: Number of reviews analyzed
        source: Source of the gaps (review_analysis, llm_generated, hybrid)
        summary: Summary of the analysis
        confidence_score: Overall confidence in the analysis
        gaps_from_reviews: Feature gaps identified from review analysis
        llm_generated_suggestions: AI-generated feature suggestions
        recommendations: Actionable recommendations
        metadata: Additional metadata about the analysis
    """
    product_name: str = Field(..., description="Name of the analyzed product")
    product_description: str = Field(..., description="Description of the product")
    total_reviews_analyzed: int = Field(..., description="Number of reviews analyzed")
    source: GapSourceType = Field(..., description="Source of the gaps")
    summary: str = Field(..., description="Summary of the analysis")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    gaps_from_reviews: List[FeatureGapItem] = Field(
        default_factory=list,
        description="Feature gaps identified from reviews"
    )
    llm_generated_suggestions: List[FeatureGapItem] = Field(
        default_factory=list,
        description="AI-generated feature suggestions"
    )
    recommendations: List[FeatureGapRecommendation] = Field(
        default_factory=list,
        description="Actionable recommendations"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_name": "TaskMaster Pro",
                    "product_description": "A task management app",
                    "total_reviews_analyzed": 150,
                    "source": "hybrid",
                    "summary": "Analysis identified 5 gaps from reviews and 3 AI suggestions",
                    "confidence_score": 0.85,
                    "gaps_from_reviews": [
                        {
                            "name": "Mobile App Enhancement",
                            "description": "Users want better mobile experience",
                            "category": "mobile",
                            "priority": "high",
                            "source": "review_analysis",
                            "evidence": ["Wish the mobile app was better"],
                            "frequency": 12,
                            "sentiment_score": -0.6,
                            "confidence": 0.9
                        }
                    ],
                    "llm_generated_suggestions": [],
                    "recommendations": [
                        {
                            "id": 1,
                            "title": "Implement Mobile App Enhancement",
                            "description": "Improve mobile experience",
                            "priority": "high",
                            "category": "mobile",
                            "source": "review_analysis",
                            "rationale": "Mentioned 12 times in reviews",
                            "evidence": []
                        }
                    ],
                    "metadata": {"min_gaps_threshold": 3}
                }
            ]
        }
    }
