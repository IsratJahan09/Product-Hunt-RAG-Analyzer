"""Core modules for Product Hunt RAG Analyzer."""

# Lazy imports to avoid slow module loading
# Import these directly when needed instead of from src.modules

__all__ = ["FAISSIndexManager", "SentimentAnalyzer", "FeatureGapService"]


def __getattr__(name):
    """Lazy import to avoid slow module loading."""
    if name == "FAISSIndexManager":
        from .vector_storage import FAISSIndexManager
        return FAISSIndexManager
    elif name == "SentimentAnalyzer":
        from .sentiment import SentimentAnalyzer
        return SentimentAnalyzer
    elif name == "FeatureGapService":
        from .feature_gap_service import FeatureGapService
        return FeatureGapService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
