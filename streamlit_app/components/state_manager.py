"""
State Management Module for Streamlit Frontend.

This module provides functions for managing application state using Streamlit's
session_state mechanism. It handles initialization, updates, and resets of
state variables throughout the application lifecycle.
"""

from typing import Optional, Dict, Any
import streamlit as st


def initialize_state() -> None:
    """
    Initialize all session_state variables with default values.
    
    Sets up the complete application state on first load, including:
    - Analysis state (submission status, results, errors)
    - Configuration state (user preferences)
    - Backend state (connectivity, dataset stats)
    
    This function is idempotent - it only initializes variables that
    don't already exist in session_state, preserving existing values.
    
    Requirements: 10.2, 10.3, 10.4
    """
    # Analysis state variables
    if "analysis_submitted" not in st.session_state:
        st.session_state.analysis_submitted = False
    
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    if "analysis_error" not in st.session_state:
        st.session_state.analysis_error = None
    
    # Configuration state variables (user preferences)
    if "product_idea" not in st.session_state:
        st.session_state.product_idea = ""
    
    if "max_competitors" not in st.session_state:
        st.session_state.max_competitors = 5
    
    if "output_format" not in st.session_state:
        st.session_state.output_format = "json"
    
    # Backend state variables
    if "backend_connected" not in st.session_state:
        st.session_state.backend_connected = False
    
    if "dataset_stats" not in st.session_state:
        st.session_state.dataset_stats = None
    
    if "connection_error" not in st.session_state:
        st.session_state.connection_error = None


def reset_analysis_state() -> None:
    """
    Clear all analysis results and error messages.
    
    Resets the analysis-related state variables to their initial values,
    effectively clearing the interface of previous analysis results and
    preparing for a new analysis. This function preserves user configuration
    preferences (competitor count, export format).
    
    Clears:
    - Analysis submission status
    - Analysis running status
    - Analysis results
    - Error messages
    
    Preserves:
    - User configuration (max_competitors, output_format)
    - Backend connection state
    - Dataset statistics
    
    Requirements: 10.2, 10.3, 10.4
    """
    st.session_state.analysis_submitted = False
    st.session_state.analysis_running = False
    st.session_state.analysis_results = None
    st.session_state.analysis_error = None


def update_config_state(
    max_competitors: Optional[int] = None,
    output_format: Optional[str] = None
) -> None:
    """
    Update configuration parameters in session state.
    
    Updates user preference settings for analysis configuration. Only
    updates parameters that are explicitly provided (not None).
    
    Args:
        max_competitors: Number of competitors to analyze (1-10).
                        If None, existing value is preserved.
        output_format: Export format ("json", "markdown", or "pdf").
                      If None, existing value is preserved.
    
    Requirements: 10.4
    """
    if max_competitors is not None:
        st.session_state.max_competitors = max_competitors
    
    if output_format is not None:
        st.session_state.output_format = output_format


def preserve_config_on_reset() -> Dict[str, Any]:
    """
    Capture current configuration preferences before reset.
    
    Returns a dictionary containing the current user configuration
    preferences that should be preserved when resetting the interface.
    This is useful for implementing "Start New Analysis" functionality
    where we want to clear results but maintain user preferences.
    
    Returns:
        Dictionary containing:
            - max_competitors: Current competitor count setting
            - output_format: Current export format setting
    
    Requirements: 10.4
    """
    return {
        "max_competitors": st.session_state.get("max_competitors", 5),
        "output_format": st.session_state.get("output_format", "json")
    }
