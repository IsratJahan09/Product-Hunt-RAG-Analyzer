"""UI components and utilities for the Streamlit application."""

from .api_client import (
    APIClient,
    APIClientError,
    APIConnectionError,
    APITimeoutError,
    APIResponseError
)

from .state_manager import (
    initialize_state,
    reset_analysis_state,
    update_config_state,
    preserve_config_on_reset
)

from .ui_components import (
    render_header,
    render_status_badge,
    render_competitor_card,
    render_metric_row,
    render_feature_list,
    render_recommendation_card
)

__all__ = [
    # API Client
    "APIClient",
    "APIClientError",
    "APIConnectionError",
    "APITimeoutError",
    "APIResponseError",
    # State Manager
    "initialize_state",
    "reset_analysis_state",
    "update_config_state",
    "preserve_config_on_reset",
    # UI Components
    "render_header",
    "render_status_badge",
    "render_competitor_card",
    "render_metric_row",
    "render_feature_list",
    "render_recommendation_card",
]
