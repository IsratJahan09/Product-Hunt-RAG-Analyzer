"""
Product Hunt RAG Analyzer - Streamlit Frontend

Main application entry point for the Streamlit web interface.
Provides a user-friendly interface for competitive analysis of product ideas.

Performance optimizations:
- Caching for API client initialization
- Lazy loading of results sections
- Optimized state management

Accessibility features:
- ARIA labels for interactive elements
- Keyboard navigation support
- Screen reader friendly content

Requirements: 1.1, 1.5, 2.1, 2.3, 2.5, 9.5
"""

import streamlit as st
from typing import Dict, Any
from components.state_manager import initialize_state, reset_analysis_state
from components.ui_components import render_header, render_status_badge, render_metric_row, render_competitor_card
from components.api_client import APIClient, APIConnectionError, APITimeoutError, APIClientError
from components.export_handler import generate_download_filename, prepare_download_data
from utils.validators import validate_product_idea
from utils.styling import apply_custom_css, add_spacing, render_section_header


def _render_feature_card(feature: dict, priority: str, st_module) -> None:
    """
    Render a single feature as a styled card (simple version).
    
    Args:
        feature: Feature dictionary with name, frequency, sentiment, etc.
        priority: Priority level (high, medium, low)
        st_module: Streamlit module reference
    """
    name = feature.get("name", "Unknown").title()
    frequency = feature.get("frequency", 0)
    sentiment = feature.get("sentiment", "neutral")
    
    # Sentiment indicators
    sentiment_emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😞"
    sentiment_color = "#4ade80" if sentiment == "positive" else "#fbbf24" if sentiment == "neutral" else "#f87171"
    
    # Priority colors
    priority_colors = {"high": "#ff6b6b", "medium": "#ffd93d", "low": "#4ade80"}
    border_color = priority_colors.get(priority, "#6b7280")
    
    # Create progress bar for frequency (max 10 for visualization)
    freq_pct = min(frequency / 10 * 100, 100)
    
    st_module.markdown(f"""
<div style="background: #1e1e2e; border-radius: 8px; padding: 12px; margin: 8px 0; border-left: 3px solid {border_color};">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600; font-size: 1em;">{name}</span>
        <span style="color: {sentiment_color}; font-size: 0.9em;">{sentiment_emoji} {sentiment.title()}</span>
    </div>
    <div style="margin-top: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #9ca3af; font-size: 0.85em;">Mentions:</span>
            <div style="flex: 1; background: #374151; border-radius: 4px; height: 8px; overflow: hidden;">
                <div style="width: {freq_pct}%; background: {border_color}; height: 100%;"></div>
            </div>
            <span style="color: #e5e7eb; font-size: 0.85em; font-weight: 500;">{frequency}x</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def _render_feature_opportunity_card(feature: dict, priority: str, category_info: dict, st_module) -> None:
    """
    Render a feature as a rich opportunity card with category info.
    
    Args:
        feature: Feature dictionary with name, frequency, sentiment, category, etc.
        priority: Priority level (high, medium, low)
        category_info: Dictionary mapping category names to icons/labels
        st_module: Streamlit module reference
    """
    name = feature.get("name", "Unknown").title()
    category = feature.get("category", "functionality")
    frequency = feature.get("frequency", 0)
    sentiment = feature.get("sentiment", "neutral")
    sentiment_score = feature.get("sentiment_score", 0)
    
    # Get category info
    cat_data = category_info.get(category, {"icon": "📦", "label": category.replace("_", " ").title()})
    
    # Sentiment indicators
    sentiment_emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😞"
    sentiment_color = "#4ade80" if sentiment == "positive" else "#fbbf24" if sentiment == "neutral" else "#f87171"
    
    # Priority colors and badges
    priority_config = {
        "high": {"color": "#ff6b6b", "bg": "#ff6b6b20", "badge": "🔥 HIGH PRIORITY"},
        "medium": {"color": "#ffd93d", "bg": "#ffd93d20", "badge": "⭐ MEDIUM"},
        "low": {"color": "#4ade80", "bg": "#4ade8020", "badge": "💡 LOW"}
    }
    p_config = priority_config.get(priority, priority_config["low"])
    
    # Calculate opportunity score
    opp_score = frequency * (1 + abs(sentiment_score))
    
    # Create progress bar for frequency (max 10 for visualization)
    freq_pct = min(frequency / 10 * 100, 100)
    
    st_module.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; padding: 16px; margin: 12px 0; border-left: 4px solid {p_config['color']}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
        <div>
            <span style="font-size: 1.15em; font-weight: 700; color: #f8fafc;">{cat_data['icon']} {name}</span>
            <div style="color: #94a3b8; font-size: 0.8em; margin-top: 2px;">{cat_data['label']}</div>
        </div>
        <span style="background: {p_config['color']}; color: #000; padding: 4px 10px; border-radius: 20px; font-size: 0.7em; font-weight: 700;">{p_config['badge']}</span>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px;">
        <div style="background: #1e293b; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="color: #64748b; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.5px;">Mentions</div>
            <div style="color: #f8fafc; font-size: 1.3em; font-weight: 700;">{frequency}x</div>
        </div>
        <div style="background: #1e293b; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="color: #64748b; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.5px;">Sentiment</div>
            <div style="color: {sentiment_color}; font-size: 1.1em; font-weight: 600;">{sentiment_emoji} {sentiment.title()}</div>
        </div>
        <div style="background: #1e293b; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="color: #64748b; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.5px;">Opportunity</div>
            <div style="color: #fbbf24; font-size: 1.3em; font-weight: 700;">{opp_score:.1f}</div>
        </div>
    </div>
    
    <div style="margin-top: 12px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #64748b; font-size: 0.75em;">Demand Level:</span>
            <div style="flex: 1; background: #374151; border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="width: {freq_pct}%; background: linear-gradient(90deg, {p_config['color']} 0%, {sentiment_color} 100%); height: 100%;"></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def _render_enhanced_gap_card(gap: dict, st_module) -> None:
    """
    Render an enhanced feature gap card from the FeatureGapService.
    
    Args:
        gap: Gap dictionary from enhanced analysis
        st_module: Streamlit module reference
    """
    name = gap.get("name", "Unknown Feature")
    description = gap.get("description", "")
    category = gap.get("category", "general").replace("_", " ").title()
    priority = gap.get("priority", "medium")
    frequency = gap.get("frequency", 0)
    confidence = gap.get("confidence", 0)
    evidence = gap.get("evidence", [])
    
    # Priority colors
    priority_colors = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}
    border_color = priority_colors.get(priority, "#6b7280")
    priority_badge = {"high": "🔥 HIGH", "medium": "⭐ MEDIUM", "low": "💡 LOW"}
    
    st_module.markdown(f"""
<div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%); border-radius: 12px; padding: 16px; margin: 12px 0; border-left: 4px solid {border_color};">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <span style="font-size: 1.1em; font-weight: 700; color: #f8fafc;">📝 {name}</span>
            <div style="color: #94a3b8; font-size: 0.85em; margin-top: 4px;">📁 {category}</div>
        </div>
        <span style="background: {border_color}; color: #000; padding: 4px 10px; border-radius: 20px; font-size: 0.7em; font-weight: 700;">{priority_badge.get(priority, priority.upper())}</span>
    </div>
    <p style="color: #cbd5e1; margin: 12px 0; font-size: 0.9em;">{description}</p>
    <div style="display: flex; gap: 16px; color: #94a3b8; font-size: 0.8em;">
        <span>📊 Mentions: {frequency}x</span>
        <span>🎯 Confidence: {confidence*100:.0f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)
    
    # Show evidence as collapsible details (no nested expander)
    if evidence:
        st_module.markdown("**Evidence from user feedback:**")
        for i, quote in enumerate(evidence[:3], 1):
            st_module.caption(f"{i}. *\"{str(quote)[:200]}{'...' if len(str(quote)) > 200 else ''}\"*")


def _render_llm_suggestion_card(suggestion: dict, st_module) -> None:
    """
    Render an LLM-generated feature suggestion card.
    
    Args:
        suggestion: Suggestion dictionary from LLM
        st_module: Streamlit module reference
    """
    name = suggestion.get("name", "Feature Suggestion")
    description = suggestion.get("description", "")
    category = suggestion.get("category", "general").replace("_", " ").title()
    priority = suggestion.get("priority", "medium")
    evidence = suggestion.get("evidence", [])
    
    # Priority styling
    priority_colors = {"high": "#3b82f6", "medium": "#8b5cf6", "low": "#06b6d4"}
    border_color = priority_colors.get(priority, "#6b7280")
    
    st_module.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; padding: 16px; margin: 12px 0; border: 1px solid {border_color};">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <span style="font-size: 1.1em; font-weight: 700; color: #f8fafc;">🤖 {name}</span>
            <div style="color: #94a3b8; font-size: 0.85em; margin-top: 4px;">📁 {category}</div>
        </div>
        <span style="background: {border_color}; color: #fff; padding: 4px 10px; border-radius: 20px; font-size: 0.7em; font-weight: 600;">AI SUGGESTION</span>
    </div>
    <p style="color: #cbd5e1; margin: 12px 0; font-size: 0.9em;">{description}</p>
</div>
""", unsafe_allow_html=True)
    
    # Show rationale if available (no nested expander)
    if evidence:
        st_module.markdown("**AI Rationale:**")
        for rationale in evidence:
            st_module.caption(f"*{rationale}*")


def _render_gap_recommendation(rec: dict, st_module) -> None:
    """
    Render a feature gap recommendation card.
    
    Args:
        rec: Recommendation dictionary
        st_module: Streamlit module reference
    """
    rec_id = rec.get("id", 0)
    title = rec.get("title", "Recommendation")
    description = rec.get("description", "")
    priority = rec.get("priority", "medium")
    source = rec.get("source", "unknown")
    rationale = rec.get("rationale", "")
    
    # Priority colors
    priority_colors = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}
    border_color = priority_colors.get(priority, "#6b7280")
    
    # Source icon
    source_icons = {"review_analysis": "📝", "llm_generated": "🤖", "hybrid": "🔄"}
    source_icon = source_icons.get(source, "📌")
    
    st_module.markdown(f"""
<div style="background: #1e293b; border-radius: 12px; padding: 16px; margin: 12px 0; border-left: 4px solid {border_color};">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-size: 1.05em; font-weight: 700; color: #f8fafc;">#{rec_id} {title}</span>
        <span style="background: {border_color}; color: #000; padding: 2px 8px; border-radius: 12px; font-size: 0.7em; font-weight: 600;">{priority.upper()}</span>
    </div>
    <p style="color: #cbd5e1; margin: 8px 0; font-size: 0.9em;">{description}</p>
    <div style="color: #94a3b8; font-size: 0.8em;">{source_icon} {source.replace('_', ' ').title()}</div>
</div>
""", unsafe_allow_html=True)
    
    if rationale:
        st_module.markdown("**Rationale:**")
        st_module.caption(f"*{rationale}*")


@st.cache_resource
def get_api_client() -> APIClient:
    """
    Get or create a cached API client instance.
    
    Uses Streamlit's cache_resource to maintain a single API client
    instance across reruns, improving performance through connection pooling.
    
    Returns:
        Cached APIClient instance
    """
    return APIClient()


def check_backend_health():
    """
    Check backend health and update session state.
    
    Queries the backend health endpoint and updates session state with
    connectivity status and dataset statistics. Handles connection failures
    gracefully by catching exceptions and updating state accordingly.
    
    Performance: Uses cached API client for connection pooling
    
    Requirements: 7.1, 7.5, 3.4
    """
    # Get cached API client for better performance
    api_client = get_api_client()
    
    # Store backend URL for error reporting
    st.session_state.backend_url = api_client.base_url
    
    try:
        # Check backend health
        health_response = api_client.check_health()
        
        # Update connection status
        st.session_state.backend_connected = True
        st.session_state.connection_error = None
        st.session_state.connection_error_type = None
        
        # Try to get dataset stats
        try:
            stats_response = api_client.get_dataset_stats()
            st.session_state.dataset_stats = stats_response
        except (APIConnectionError, APITimeoutError, APIClientError) as e:
            # Backend is connected but stats unavailable
            st.session_state.dataset_stats = None
            
    except APIConnectionError as e:
        # Backend is not reachable (network error boundary)
        st.session_state.backend_connected = False
        st.session_state.dataset_stats = None
        st.session_state.connection_error = str(e)
        st.session_state.connection_error_type = "connection"
        
    except APITimeoutError as e:
        # Backend timed out (transient error - retry recommended)
        st.session_state.backend_connected = False
        st.session_state.dataset_stats = None
        st.session_state.connection_error = f"Connection timeout: {str(e)}"
        st.session_state.connection_error_type = "timeout"
        
    except APIClientError as e:
        # Other API error
        st.session_state.backend_connected = False
        st.session_state.dataset_stats = None
        st.session_state.connection_error = f"API error: {str(e)}"
        st.session_state.connection_error_type = "api_error"


def render_backend_status():
    """
    Display professional SaaS-style backend connection status badge.
    
    DESIGN-ONLY UPDATE: Clean, standard, professional status badge with:
    - Very light gray background (#f3f4f6)
    - Calm blue text (#3b82f6) 
    - Rounded corners and subtle border
    - Compact padding, one-line layout
    - Good accessibility and mobile support
    
    Shows "✅ Connected – Backend online and ready" when available,
    or error status when unavailable. Includes retry functionality.
    
    Requirements: 7.1, 7.2, 7.3, 3.4, 9.1, 9.2
    """
    # Professional status badge - NO EXTRA BOXES - DESIGN-ONLY UPDATE
    if st.session_state.backend_connected:
        # Perfect professional status badge with light-gray background and blue font
        st.markdown("""
        <div style="
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.5rem 0.875rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 0.75rem;
            font-family: 'Inter', sans-serif;
            width: fit-content;
        ">
            <span style="color: #3b82f6; font-size: 0.875rem;">✅</span>
            <span style="color: #3b82f6; font-size: 0.875rem; font-weight: 500;">Connected – Backend online and ready</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Error status with retry button in same row
        error_msg = st.session_state.get("connection_error", "Unable to connect")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div style="
                background: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 8px;
                padding: 0.5rem 0.875rem;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                margin-bottom: 0.5rem;
                font-family: 'Inter', sans-serif;
            ">
                <span style="color: #dc2626; font-size: 0.875rem;">❌</span>
                <span style="color: #dc2626; font-size: 0.875rem; font-weight: 500;">Offline – {error_msg[:35]}{"..." if len(error_msg) > 35 else ""}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Retry button only when disconnected
            if st.button("🔄 Retry", key="retry_backend", help="Retry connection", use_container_width=True):
                if "health_checked" in st.session_state:
                    del st.session_state.health_checked
                st.rerun()
    
    # Only show detailed troubleshooting if disconnected
    if not st.session_state.backend_connected:
        with st.expander("🔍 Troubleshooting", expanded=False):
            st.markdown('<p style="color: #9ca3af; font-size: 0.75rem; margin-bottom: 0.5rem;"><strong>Common Solutions:</strong></p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #9ca3af; font-size: 0.75rem; margin: 0;">• Ensure backend API is running on http://localhost:8000</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #9ca3af; font-size: 0.75rem; margin: 0;">• Check network connectivity and firewall settings</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #9ca3af; font-size: 0.75rem; margin: 0;">• Verify backend logs for startup errors</p>', unsafe_allow_html=True)


def render_dataset_stats(stats: dict):
    """
    Display COMPACT dataset statistics in a formatted layout.
    
    DESIGN-ONLY CHANGES: More compact metrics, better alignment, lighter colors
    
    Shows the number of products and reviews available in the loaded
    FAISS indices, along with average reviews per product.
    
    Args:
        stats: Dictionary containing dataset statistics with keys:
            - total_products: Number of products in index
            - total_reviews: Number of reviews in index
            - avg_reviews_per_product: Average reviews per product
            - indices_loaded: Whether indices are loaded
    
    Requirements: 7.4, 9.1, 9.2
    """
    if not stats:
        st.markdown('<p style="color: #9ca3af; font-size: 0.75rem; margin: 0;">Stats unavailable</p>', unsafe_allow_html=True)
        return
    
    # Check if indices are loaded
    indices_loaded = stats.get("indices_loaded", False)
    
    if not indices_loaded:
        st.markdown('<p style="color: #f59e0b; font-size: 0.75rem; margin: 0;">⚠️ Indices not loaded</p>', unsafe_allow_html=True)
        return
    
    # Display statistics compactly
    total_products = stats.get("total_products", 0)
    total_reviews = stats.get("total_reviews", 0)
    avg_reviews = stats.get("avg_reviews_per_product", 0)
    
    # Compact metrics with better alignment
    st.markdown(f"""
    <div style="
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    ">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.5rem;">
            <div style="text-align: center;">
                <div style="color: #FF6B6B; font-size: 1.25rem; font-weight: 600; font-family: 'Inter', sans-serif;">{total_products:,}</div>
                <div style="color: #9ca3af; font-size: 0.7rem; font-weight: 500; text-transform: uppercase;">Products</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #FF6B6B; font-size: 1.25rem; font-weight: 600; font-family: 'Inter', sans-serif;">{total_reviews:,}</div>
                <div style="color: #9ca3af; font-size: 0.7rem; font-weight: 500; text-transform: uppercase;">Reviews</div>
            </div>
        </div>
        <div style="text-align: center; padding-top: 0.5rem; border-top: 1px solid #f1f5f9;">
            <span style="color: #6b7280; font-size: 0.7rem;">📊 Avg: {avg_reviews:.1f} reviews/product</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """
    Main application entry point.
    
    Orchestrates the overall application flow:
    1. Configure page settings
    2. Apply custom styling
    3. Initialize session state
    4. Check backend health
    5. Render header
    6. Create sidebar for configuration
    7. Create main content area for input and results
    
    Requirements: 9.1, 9.2
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="Product Hunt RAG Analyzer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS styling for responsive layout and enhanced visuals
    # Requirements: 9.1, 9.2
    apply_custom_css()
    
    # Initialize session state
    initialize_state()
    
    # Check backend health on application load
    # Only check if not already checked in this session
    if "health_checked" not in st.session_state:
        with st.spinner("Checking backend connectivity..."):
            check_backend_health()
        st.session_state.health_checked = True
    
    # Render application header
    render_header()
    
    # Create sidebar for configuration
    render_sidebar()
    
    # Create main content area
    render_main_content()


def render_sidebar():
    """
    Render the ULTRA-COMPACT sidebar with configuration controls.
    
    DESIGN-ONLY UPDATE: Ultra-compact spacing, professional typography, subtle export colors
    - Reduced line heights and spacing for no-scroll experience
    - Export format colors: subtle black with hover highlights
    - Better visual hierarchy with consistent font sizes
    - Mobile responsive and accessible design
    
    Displays:
    - Competitor count slider (1-10, default 5)
    - Export format radio buttons (JSON, Markdown, PDF)
    
    Updates session state when configuration changes.
    
    Requirements: 2.1, 2.3, 2.5
    """
    with st.sidebar:
        # Compact header with better typography - DESIGN-ONLY UPDATE
        st.markdown('<h3 style="color: #374151; font-size: 0.9375rem; font-weight: 600; margin: 0 0 0.5rem 0; line-height: 1.3; font-family: Inter, sans-serif;">⚙️ Configuration</h3>', unsafe_allow_html=True)
        
        # Analysis Settings section - with top gap
        st.markdown('<p style="color: #4b5563; font-size: 0.8125rem; font-weight: 500; margin: 1rem 0 0.375rem 0; line-height: 1.2; font-family: Inter, sans-serif;">Analysis Settings</p>', unsafe_allow_html=True)
        max_competitors = st.slider(
            "Competitors to analyze",
            min_value=1,
            max_value=10,
            value=st.session_state.max_competitors,
            help="Select how many competitor products to analyze (1-10)"
        )
        
        # Update session state if value changed
        if max_competitors != st.session_state.max_competitors:
            st.session_state.max_competitors = max_competitors
        
        # Compact caption with lighter color
        st.markdown(f'<p style="color: #9ca3af; font-size: 0.6875rem; margin: 0.125rem 0 0.625rem 0; line-height: 1.1; font-family: Inter, sans-serif;">Analyzing top {max_competitors} competitors</p>', unsafe_allow_html=True)
        
        # Subtle thin divider
        st.markdown('<hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #f3f4f6; opacity: 0.8;">', unsafe_allow_html=True)
        
        # Export Format section - compact spacing
        st.markdown('<p style="color: #4b5563; font-size: 0.8125rem; font-weight: 500; margin: 0 0 0.375rem 0; line-height: 1.2; font-family: Inter, sans-serif;">Export Format</p>', unsafe_allow_html=True)
        output_format = st.radio(
            "Format",
            options=["json", "markdown", "pdf"],
            index=["json", "markdown", "pdf"].index(st.session_state.output_format),
            format_func=lambda x: x.upper(),
            help="Select the format for downloading analysis reports",
            label_visibility="collapsed"
        )
        
        # Update session state if value changed
        if output_format != st.session_state.output_format:
            st.session_state.output_format = output_format
        
        # Subtle thin divider
        st.markdown('<hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #f3f4f6; opacity: 0.8;">', unsafe_allow_html=True)
        
        # About section - compact
        st.markdown('<p style="color: #4b5563; font-size: 0.8125rem; font-weight: 500; margin: 0 0 0.375rem 0; line-height: 1.2; font-family: Inter, sans-serif;">ℹ️ About</p>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: #6b7280; font-size: 0.6875rem; line-height: 1.3; margin: 0; font-family: Inter, sans-serif;">'
            "AI-powered competitive intelligence for Product Hunt products. "
            "Includes market positioning, feature gaps, sentiment analysis, and strategic recommendations."
            '</p>', unsafe_allow_html=True
        )


def handle_analysis_submission():
    """
    Handle the analysis submission process with progress tracking.
    
    Submits the product idea to the backend for analysis, displays
    progress indicators during processing, and handles the response
    (success or error). Stores error details and request parameters
    for troubleshooting.
    
    Updates session state with:
    - analysis_running: Set to True during processing, False on completion
    - analysis_results: Populated with results on success
    - analysis_error: Populated with error message on failure
    - analysis_error_type: Type of error for better troubleshooting
    - last_request_params: Request parameters for debugging
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    # Clear any previous errors and results
    st.session_state.analysis_error = None
    st.session_state.analysis_error_type = None
    st.session_state.analysis_results = None
    
    # Set analysis running state
    st.session_state.analysis_running = True
    st.session_state.analysis_submitted = True
    
    # Store request parameters for error reporting
    request_params = {
        "product_idea": st.session_state.product_idea,
        "max_competitors": st.session_state.max_competitors,
        "output_format": st.session_state.output_format
    }
    st.session_state.last_request_params = request_params
    
    # Get cached API client for better performance
    api_client = get_api_client()
    
    # Display progress indicator with spinner
    with st.spinner("🔍 Analyzing your product idea..."):
        try:
            # Show progress status messages
            status_placeholder = st.empty()
            
            status_placeholder.info("📊 Searching for similar products...")
            
            # Submit analysis request
            results = api_client.submit_analysis(
                product_idea=st.session_state.product_idea,
                max_competitors=st.session_state.max_competitors,
                output_format=st.session_state.output_format
            )
            
            status_placeholder.info("🤖 Generating competitive insights...")
            
            # Store results in session state (Requirement 3.3)
            st.session_state.analysis_results = results
            st.session_state.analysis_error = None
            st.session_state.analysis_error_type = None
            
            # Clear status message (hide progress indicator - Requirement 3.3)
            status_placeholder.empty()
            
        except APIConnectionError as e:
            # Handle connection errors - network error boundary (Requirement 3.4)
            st.session_state.analysis_error = f"Connection Error: {str(e)}"
            st.session_state.analysis_error_type = "connection"
            st.session_state.analysis_results = None
            
        except APITimeoutError as e:
            # Handle timeout errors - transient error (Requirement 3.4)
            st.session_state.analysis_error = f"Timeout Error: {str(e)}"
            st.session_state.analysis_error_type = "timeout"
            st.session_state.analysis_results = None
            
        except APIClientError as e:
            # Handle other API errors (Requirement 3.4)
            st.session_state.analysis_error = f"API Error: {str(e)}"
            st.session_state.analysis_error_type = "api_error"
            st.session_state.analysis_results = None
            
        except Exception as e:
            # Handle unexpected errors (Requirement 3.4)
            st.session_state.analysis_error = f"Unexpected Error: {str(e)}"
            st.session_state.analysis_error_type = "unexpected"
            st.session_state.analysis_results = None
        
        finally:
            # Always reset running state when done (hide progress indicator - Requirement 3.3)
            st.session_state.analysis_running = False


def render_product_idea_input():
    """
    Render the COMPACT product idea input section with modern design.
    
    DESIGN-ONLY CHANGES: Compact spacing, lighter colors, better alignment
    
    Requirements: 1.1, 1.2, 1.3, 1.5, 3.4, 9.1, 9.2
    """
    render_section_header("Product Idea Input", "📝")
    
    # Compact modern input container
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1rem;">💡</span>
            <h4 style="
                margin: 0;
                color: #6b7280;
                font-size: 0.8125rem;
                font-weight: 500;
                font-family: 'Inter', sans-serif;
            ">Describe Your Product Idea</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Text area for product idea input
    product_idea = st.text_area(
        "Product Description",
        value=st.session_state.product_idea,
        height=100,
        placeholder="Example: A mobile app that helps remote teams coordinate lunch orders with AI-powered restaurant recommendations...",
        help="Enter a detailed description of your product idea (minimum 10 characters)",
        key="product_idea_input",
        disabled=st.session_state.analysis_running,
        label_visibility="collapsed"
    )
    
    # Update session state with current input
    st.session_state.product_idea = product_idea
    
    # Compact character count and validation in parallel alignment
    char_count = len(product_idea)
    is_valid, error_message = validate_product_idea(product_idea)
    
    # Parallel alignment for character count and validation
    col1, col2 = st.columns([1, 2], gap="small")
    
    with col1:
        if char_count >= 10:
            st.markdown(f"""
            <div style="
                background: #ecfdf5;
                border: 1px solid #bbf7d0;
                border-radius: 4px;
                padding: 0.25rem 0.5rem;
                display: inline-block;
                margin-bottom: 0.25rem;
            ">
                <span style="color: #065f46; font-size: 0.7rem; font-weight: 500;">✅ {char_count}</span>
            </div>
            """, unsafe_allow_html=True)
        elif char_count > 0:
            st.markdown(f"""
            <div style="
                background: #fffbeb;
                border: 1px solid #fed7aa;
                border-radius: 4px;
                padding: 0.25rem 0.5rem;
                display: inline-block;
                margin-bottom: 0.25rem;
            ">
                <span style="color: #92400e; font-size: 0.7rem; font-weight: 500;">⚠️ {char_count}/10</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 0.25rem 0.5rem;
                display: inline-block;
                margin-bottom: 0.25rem;
            ">
                <span style="color: #6b7280; font-size: 0.7rem; font-weight: 500;">0 chars</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Validation error display (compact)
        if product_idea and not is_valid and error_message:
            st.markdown(f"""
            <div style="
                background: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 6px;
                padding: 0.375rem 0.625rem;
                display: flex;
                align-items: center;
                gap: 0.375rem;
                margin-bottom: 0.5rem;
            ">
                <span style="color: #ef4444; font-size: 0.875rem;">❌</span>
                <span style="color: #991b1b; font-size: 0.75rem; font-weight: 500;">{error_message}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Submit button state logic
    submit_disabled = (
        not is_valid or 
        not st.session_state.backend_connected or 
        st.session_state.analysis_running
    )
    
    # Compact status message (only show when needed)
    if st.session_state.analysis_running:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.375rem;
        ">
            <div style="
                background: #FF6B6B;
                color: white;
                border-radius: 50%;
                width: 16px;
                height: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 8px;
            ">⏳</div>
            <span style="color: #1e40af; font-size: 0.75rem; font-weight: 500;">Analysis running...</span>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.backend_connected:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 1px solid #fed7aa;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.375rem;
        ">
            <div style="
                background: #f59e0b;
                color: white;
                border-radius: 50%;
                width: 16px;
                height: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 8px;
            ">⚠</div>
            <span style="color: #92400e; font-size: 0.75rem; font-weight: 500;">Backend not connected</span>
        </div>
        """, unsafe_allow_html=True)
    elif not product_idea:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #bbf7d0;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.375rem;
        ">
            <div style="
                background: #10b981;
                color: white;
                border-radius: 50%;
                width: 16px;
                height: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 8px;
            ">💡</div>
            <span style="color: #065f46; font-size: 0.75rem; font-weight: 500;">Enter your product idea to get started</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Compact submit button
    submit_button = st.button(
        "🚀 Analyze Product Idea",
        disabled=submit_disabled,
        type="primary",
        use_container_width=True
    )
    
    # Handle submit button click
    if submit_button:
        handle_analysis_submission()


def render_competitor_display(results: Dict[str, Any]) -> None:
    """
    Display identified competitors in responsive button-like cards.
    
    DESIGN-ONLY UPDATE: Responsive columns with button-like competitor cards
    - Desktop: 3-5 competitors per row based on count
    - Mobile: Stacks vertically for better readability
    - Button-like cards with rounded corners and subtle shadows
    - Hover effects with smooth animations (0.25s transition)
    - Light background (#f9f9f9) with professional styling
    - Handles any number of competitors dynamically
    
    Args:
        results: Analysis results dictionary containing competitor information
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 9.1, 9.2
    """
    render_section_header("Identified Competitors", "🎯", "Products similar to your idea")
    
    # Get competitors from results
    competitors_raw = results.get("competitors_identified", [])
    
    # Convert to list of dicts if needed
    competitors = []
    for comp in competitors_raw:
        if isinstance(comp, str):
            competitors.append({"name": comp})
        elif isinstance(comp, dict):
            competitors.append(comp)
    
    # Modern competitor count display
    if competitors:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            text-align: center;
        ">
            <div style="
                color: #1e40af;
                font-size: 2rem;
                font-weight: 700;
                font-family: 'Inter', sans-serif;
                margin-bottom: 0.25rem;
            ">{len(competitors)}</div>
            <div style="
                color: #3b82f6;
                font-size: 0.875rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.025em;
            ">Competitors Found</div>
            <div style="
                color: #64748b;
                font-size: 0.75rem;
                margin-top: 0.5rem;
            ">Showing top {len(competitors)} most relevant competitors</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Modern empty state display
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 1px solid #fed7aa;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
            <h3 style="
                margin: 0 0 1rem 0;
                color: #92400e;
                font-size: 1.125rem;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
            ">No Competitors Found</h3>
            <div style="
                background: #fef3c7;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
            ">
                <p style="margin: 0; color: #78350f; font-size: 0.875rem; line-height: 1.5;">
                    This could mean your product idea is highly unique! Here are some possibilities:
                </p>
            </div>
            <div style="text-align: left; max-width: 400px; margin: 0 auto;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="color: #f59e0b;">✨</span>
                    <span style="color: #92400e; font-size: 0.875rem;">Your product idea is highly unique</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="color: #f59e0b;">🔍</span>
                    <span style="color: #92400e; font-size: 0.875rem;">The search criteria didn't match existing products</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #f59e0b;">💡</span>
                    <span style="color: #92400e; font-size: 0.875rem;">Try rephrasing your product description</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sort competitors by relevance score if available
    competitors_sorted = sorted(
        competitors,
        key=lambda x: x.get("relevance_score", 0.0),
        reverse=True
    )
    
    add_spacing("medium")
    
    # DESIGN-ONLY UPDATE: Responsive columns with button-like cards
    # Calculate number of columns based on competitor count (3-5 per row on desktop)
    num_competitors = len(competitors_sorted)
    if num_competitors <= 3:
        cols = num_competitors
    elif num_competitors <= 6:
        cols = 3
    elif num_competitors <= 10:
        cols = 4
    else:
        cols = 5
    
    # Create responsive columns
    columns = st.columns(cols)
    
    # Display competitors in button-like cards across columns
    for idx, competitor in enumerate(competitors_sorted):
        name = competitor.get("name", "Unknown Product")
        col_idx = idx % cols
        
        with columns[col_idx]:
            # Button-like competitor card with hover effects
            st.markdown(f"""
            <div class="competitor-card-button">
                🎯 {name}
            </div>
            """, unsafe_allow_html=True)


def render_analysis_metadata(results: Dict[str, Any]) -> None:
    """
    Display analysis metadata and performance metrics in modern cards.
    
    DESIGN-ONLY CHANGES: Modern SaaS-style metadata display with clean cards
    
    Args:
        results: Analysis results dictionary containing metadata fields
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2
    """
    add_spacing("large")
    render_section_header("Analysis Metadata", "ℹ️", "Performance and confidence metrics")
    
    # Extract metadata
    analysis_id = results.get("analysis_id", "N/A")
    confidence_score = results.get("confidence_score", 0.0)
    processing_time = results.get("processing_time_ms", 0)
    generated_at = results.get("generated_at", "N/A")
    
    # Modern metrics grid
    col1, col2, col3 = st.columns(3)
    
    # Analysis ID Card
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        ">
            <div style="color: #15803d; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em; margin-bottom: 0.5rem;">Analysis ID</div>
            <div style="color: #16a34a; font-size: 1.25rem; font-weight: 700; font-family: 'Inter', sans-serif; word-break: break-all;">{analysis_id}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence Score Card
    with col2:
        confidence_pct = f"{confidence_score * 100:.1f}%"
        
        # Determine confidence level and colors
        if confidence_score >= 0.7:
            bg_color = "#ecfdf5"
            border_color = "#a7f3d0"
            text_color = "#065f46"
            value_color = "#10b981"
            indicator = "High"
        elif confidence_score >= 0.5:
            bg_color = "#fffbeb"
            border_color = "#fed7aa"
            text_color = "#92400e"
            value_color = "#f59e0b"
            indicator = "Medium"
        else:
            bg_color = "#fef2f2"
            border_color = "#fecaca"
            text_color = "#991b1b"
            value_color = "#ef4444"
            indicator = "Low"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {bg_color} 0%, {bg_color} 100%);
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        ">
            <div style="color: {text_color}; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em; margin-bottom: 0.5rem;">Confidence Score</div>
            <div style="color: {value_color}; font-size: 1.75rem; font-weight: 700; font-family: 'Inter', sans-serif;">{confidence_pct}</div>
            <div style="color: {text_color}; font-size: 0.75rem; font-weight: 500; margin-top: 0.25rem;">{indicator}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Processing Time Card
    with col3:
        if processing_time >= 1000:
            time_display = f"{processing_time / 1000:.2f}s"
        else:
            time_display = f"{processing_time}ms"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        ">
            <div style="color: #1e40af; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em; margin-bottom: 0.5rem;">Processing Time</div>
            <div style="color: #3b82f6; font-size: 1.75rem; font-weight: 700; font-family: 'Inter', sans-serif;">{time_display}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Generation timestamp
    st.markdown(f"""
    <div style="
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        text-align: center;
    ">
        <span style="color: #64748b; font-size: 0.875rem; font-weight: 500;">📅 Generated at: {generated_at}</span>
    </div>
    """, unsafe_allow_html=True)
    
    add_spacing("medium")
    
    # Low confidence warning
    if confidence_score < 0.5:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1px solid #fecaca;
            border-left: 4px solid #ef4444;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                <div style="
                    background: #ef4444;
                    color: white;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    flex-shrink: 0;
                ">⚠</div>
                <div>
                    <h4 style="margin: 0 0 0.75rem 0; color: #991b1b; font-size: 1rem; font-weight: 600;">Low Confidence Score</h4>
                    <p style="margin: 0 0 1rem 0; color: #dc2626; font-size: 0.875rem; line-height: 1.5;">
                        The analysis confidence is below 50%. Results may be less reliable. Consider:
                    </p>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="color: #ef4444;">•</span>
                            <span style="color: #dc2626; font-size: 0.875rem;">Providing a more detailed product description</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="color: #ef4444;">•</span>
                            <span style="color: #dc2626; font-size: 0.875rem;">Adjusting the number of competitors</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="color: #ef4444;">•</span>
                            <span style="color: #dc2626; font-size: 0.875rem;">Verifying that similar products exist in the dataset</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional warnings
    warnings = results.get("warnings", [])
    if warnings:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 1px solid #fed7aa;
            border-left: 4px solid #f59e0b;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin: 0 0 0.75rem 0; color: #92400e; font-size: 1rem; font-weight: 600;">Additional Warnings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for warning in warnings:
            st.markdown(f"""
            <div style="
                background: #fef3c7;
                border-radius: 6px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">
                <span style="color: #f59e0b;">⚠️</span>
                <span style="color: #92400e; font-size: 0.875rem;">{warning}</span>
            </div>
            """, unsafe_allow_html=True)


def render_download_section(results: Dict[str, Any]) -> None:
    """
    Display modern download section with styled buttons and file info.
    
    DESIGN-ONLY CHANGES: Modern SaaS-style download section with clean cards
    
    Args:
        results: Analysis results dictionary to be downloaded
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 9.1, 9.2
    """
    add_spacing("large")
    render_section_header("Download Report", "📥", "Export your analysis results")
    
    # Get selected output format from session state
    output_format = st.session_state.output_format
    
    # Generate filename with timestamp
    filename = generate_download_filename(output_format)
    
    # Modern download card container
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    ">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                border-radius: 50%;
                width: 32px;
                height: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
            ">📥</div>
            <div>
                <h4 style="margin: 0; color: #1e293b; font-size: 1.125rem; font-weight: 600; font-family: 'Inter', sans-serif;">Export Analysis Report</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.875rem;">Download your complete analysis in your preferred format</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data in selected format
    try:
        download_data = prepare_download_data(results, output_format)
        
        # Determine MIME type based on format
        mime_types = {
            "json": "application/json",
            "markdown": "text/markdown",
            "pdf": "application/pdf"
        }
        mime_type = mime_types.get(output_format.lower(), "application/octet-stream")
        
        # Format icons
        format_icons = {
            "json": "📄",
            "markdown": "📝",
            "pdf": "📋"
        }
        icon = format_icons.get(output_format.lower(), "📄")
        
        # Create modern download button
        download_button = st.download_button(
            label=f"{icon} Download as {output_format.upper()}",
            data=download_data,
            file_name=filename,
            mime=mime_type,
            use_container_width=True,
            type="primary"
        )
        
        # Display success message after download
        if download_button:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                border: 1px solid #a7f3d0;
                border-radius: 12px;
                padding: 1.25rem;
                margin: 1rem 0;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            ">
                <div style="
                    background: #10b981;
                    color: white;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                ">✓</div>
                <span style="color: #065f46; font-size: 0.875rem; font-weight: 500;">Report downloaded successfully!</span>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()  # Add celebratory animation
        
        # Modern file information display
        file_size_kb = len(download_data) / 1024
        if file_size_kb < 1024:
            size_display = f"{file_size_kb:.1f} KB"
        else:
            size_display = f"{file_size_kb / 1024:.1f} MB"
        
        st.markdown(f"""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        ">
            <div style="text-align: center;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em;">Filename</div>
                <div style="color: #1e293b; font-size: 0.875rem; font-weight: 600; margin-top: 0.25rem; word-break: break-all;">{filename}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em;">Format</div>
                <div style="color: #1e293b; font-size: 0.875rem; font-weight: 600; margin-top: 0.25rem;">{output_format.upper()}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em;">Size</div>
                <div style="color: #1e293b; font-size: 0.875rem; font-weight: 600; margin-top: 0.25rem;">{size_display}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1px solid #fecaca;
            border-left: 4px solid #ef4444;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        ">
            <div style="
                background: #ef4444;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                flex-shrink: 0;
            ">✕</div>
            <div>
                <h4 style="margin: 0 0 0.5rem 0; color: #991b1b; font-size: 1rem; font-weight: 600;">Download Error</h4>
                <p style="margin: 0 0 0.75rem 0; color: #dc2626; font-size: 0.875rem;">{str(e)}</p>
                <p style="margin: 0; color: #dc2626; font-size: 0.875rem;">Please try a different export format or contact support if the issue persists.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_analysis_results():
    """
    Display analysis results or error messages.
    
    Shows comprehensive analysis results when available, including:
    - Identified competitors
    - Market positioning insights
    - Feature gap analysis
    - Sentiment analysis
    - Strategic recommendations
    - Download functionality
    - Clear/reset functionality
    
    If an error occurred, displays the error message with expandable details
    and retry functionality.
    
    Requirements: 3.3, 3.4, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 9.3, 10.1, 10.2, 10.3, 10.4, 10.5
    """
    # Check if there's an error to display
    if st.session_state.analysis_error:
        # Banner error for critical failure
        st.error(f"❌ **Analysis Failed:** {st.session_state.analysis_error}")
        
        # Determine error type for better guidance
        error_msg = st.session_state.analysis_error.lower()
        is_network_error = "connection" in error_msg or "timeout" in error_msg
        is_validation_error = "validation" in error_msg or "invalid" in error_msg
        
        # Expandable error details section
        with st.expander("🔍 Error Details & Troubleshooting", expanded=False):
            st.markdown("### Error Information")
            st.code(st.session_state.analysis_error)
            
            # Store error type in session state if available
            error_type = st.session_state.get("analysis_error_type", "unknown")
            st.markdown(f"**Error Type:** {error_type}")
            
            # Store request details if available
            if "last_request_params" in st.session_state:
                st.markdown("### Request Parameters")
                params = st.session_state.last_request_params
                st.json(params)
            
            st.markdown("### Troubleshooting Steps")
            
            if is_network_error:
                st.markdown(
                    """
                    **Network/Connection Error:**
                    1. Verify the backend service is running
                    2. Check network connectivity
                    3. Try the retry button below (may help with transient issues)
                    4. Increase timeout if analysis takes longer than expected
                    5. Check backend logs for errors
                    """
                )
            elif is_validation_error:
                st.markdown(
                    """
                    **Validation Error:**
                    1. Ensure your product idea is at least 10 characters
                    2. Verify competitor count is between 1-10
                    3. Check that export format is valid (json, markdown, pdf)
                    4. Try rephrasing your product description
                    """
                )
            else:
                st.markdown(
                    """
                    **General Troubleshooting:**
                    1. Check that the backend is running and connected
                    2. Verify your product idea is descriptive enough
                    3. Try reducing the number of competitors
                    4. Ensure FAISS indices are loaded in the backend
                    5. Check backend logs for detailed error messages
                    6. Verify Ollama LLM service is running
                    """
                )
        
        # Retry functionality for transient errors
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Retry button for transient errors
            if st.button("🔄 Retry Analysis", type="secondary", use_container_width=True):
                # Clear error and retry with same parameters
                st.session_state.analysis_error = None
                st.session_state.analysis_error_type = None
                # Trigger analysis submission
                handle_analysis_submission()
                st.rerun()
        
        with col2:
            # Start new analysis button (Requirement 10.1)
            if st.button("🆕 Start New Analysis", type="primary", use_container_width=True):
                # Clear all analysis results and error messages (Requirement 10.2, 10.3)
                reset_analysis_state()
                # Clear product idea input to reset interface to initial state (Requirement 10.3)
                st.session_state.product_idea = ""
                # Clear previous error messages (Requirement 10.5)
                st.session_state.analysis_error = None
                st.session_state.analysis_error_type = None
                # Configuration preferences are preserved automatically (Requirement 10.4)
                # Rerun to update UI
                st.rerun()
        
        with col3:
            if is_network_error:
                st.info("💡 **Retry** may help with temporary network issues. **Start New** to enter a different product idea.")
            else:
                st.info("💡 **Retry** to attempt the same analysis again. **Start New** to enter a different product idea.")
        
        return
    
    # Check if results are available
    if not st.session_state.analysis_results:
        # Show placeholder when no results yet
        st.info(
            """
            Analysis results will appear here after submission.
            
            The results will include:
            - Identified competitors
            - Market positioning insights
            - Feature gap analysis
            - Sentiment analysis
            - Strategic recommendations
            """
        )
        return
    
    # Parse results from session state
    results = st.session_state.analysis_results
    
    # Display success message if just completed
    if st.session_state.analysis_submitted and not st.session_state.analysis_running:
        st.success("✅ Analysis completed successfully!")
    
    # Display analysis status
    status = results.get("status", "unknown")
    if status == "completed":
        st.success(f"**Status:** {status.upper()}")
    elif status == "failed":
        st.error(f"**Status:** {status.upper()}")
        # Show any error details from the backend
        error_detail = results.get("error", "Unknown error occurred")
        st.error(f"**Error:** {error_detail}")
        return
    else:
        st.warning(f"**Status:** {status}")
    
    # Display basic analysis information
    st.markdown(f"**Product Idea:** {results.get('product_idea', 'N/A')}")
    
    st.divider()
    
    # Display competitors in dedicated section (Task 13)
    render_competitor_display(results)
    
    # Display detailed results sections (Task 14)
    # Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 9.3
    # Performance: Lazy loading via expanders reduces initial render time
    detailed_results = results.get("results", {})
    
    if detailed_results:
        add_spacing("large")
        render_section_header("Detailed Analysis", "📊", "In-depth competitive insights")
        
        # Add accessibility note
        st.caption("💡 Expand sections below to view detailed analysis. Use Tab key to navigate between sections.")
        
        # Market Positioning Section (Requirement 5.1, 5.2)
        # Create expandable section with summary and differentiation
        market_positioning = detailed_results.get("market_positioning", {})
        if market_positioning:
            with st.expander("🎯 Market Positioning", expanded=True):
                # Handle both LLM insights and structured analysis formats
                llm_insights = market_positioning.get("llm_insights", {})
                
                # Display summary (try LLM insights first, then direct field)
                summary = llm_insights.get("summary") or market_positioning.get("summary", "No summary available")
                st.markdown("### Summary")
                st.markdown(summary)
                
                st.markdown("")  # Add spacing
                
                # Display differentiation opportunities with lighter font color
                diff_opportunities = llm_insights.get("differentiation_opportunities") or market_positioning.get("differentiation_opportunities", [])
                if diff_opportunities:
                    st.markdown('<h3 style="color: #9ca3af; font-size: 1.125rem; font-weight: 500; margin: 1rem 0 0.5rem 0; font-family: Inter, sans-serif;">Differentiation Opportunities</h3>', unsafe_allow_html=True)
                    for idx, opp in enumerate(diff_opportunities, 1):
                        st.markdown(f'<p style="color: #9ca3af; font-size: 0.875rem; margin: 0.25rem 0; font-family: Inter, sans-serif;">{idx}. {opp}</p>', unsafe_allow_html=True)
                else:
                    st.info("No differentiation opportunities identified.")
                
                # Display competitor positions with lighter font color
                competitor_positions = llm_insights.get("competitor_positions") or market_positioning.get("competitor_positions", [])
                if competitor_positions:
                    st.markdown("")  # Add spacing
                    st.markdown('<h3 style="color: #9ca3af; font-size: 1.125rem; font-weight: 500; margin: 1rem 0 0.5rem 0; font-family: Inter, sans-serif;">Competitor Positions</h3>', unsafe_allow_html=True)
                    for pos in competitor_positions:
                        if isinstance(pos, dict):
                            name = pos.get("name", "Unknown")
                            position = pos.get("position", "N/A")
                            st.markdown(f'<p style="color: #9ca3af; font-size: 0.875rem; margin: 0.25rem 0; font-family: Inter, sans-serif;">- <strong>{name}:</strong> {position}</p>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p style="color: #9ca3af; font-size: 0.875rem; margin: 0.25rem 0; font-family: Inter, sans-serif;">- {pos}</p>', unsafe_allow_html=True)
        
        # Feature Gaps Section - Enhanced Analysis Only
        feature_gaps = detailed_results.get("feature_gaps", {})
        enhanced_analysis = feature_gaps.get("enhanced_analysis", {})
        
        if enhanced_analysis:
            with st.expander("🎯 Feature Gap Analysis", expanded=False):
                st.markdown("*Comprehensive analysis combining user feedback gaps and AI-generated suggestions*")
                st.markdown("")
                
                # Show source and confidence
                source = enhanced_analysis.get("source", "unknown")
                confidence = enhanced_analysis.get("confidence_score", 0)
                source_labels = {
                    "review_analysis": "📝 User Reviews Only",
                    "llm_generated": "🤖 AI Suggestions Only",
                    "hybrid": "🔄 Combined Analysis"
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Analysis Type", source_labels.get(source, source))
                with col2:
                    st.metric("Confidence", f"{confidence*100:.0f}%")
                with col3:
                    st.metric("Reviews Analyzed", enhanced_analysis.get("total_reviews_analyzed", 0))
                
                # Show summary
                summary = enhanced_analysis.get("summary", "")
                if summary:
                    st.info(f"📋 {summary}")
                
                st.markdown("")
                
                # Create tabs for gaps from reviews and LLM suggestions
                gaps_from_reviews = enhanced_analysis.get("gaps_from_reviews", [])
                llm_suggestions = enhanced_analysis.get("llm_generated_suggestions", [])
                recommendations = enhanced_analysis.get("recommendations", [])
                
                tab_reviews, tab_ai, tab_actions = st.tabs([
                    f"📝 User Feedback Gaps ({len(gaps_from_reviews)})",
                    f"🤖 AI Suggestions ({len(llm_suggestions)})",
                    f"📌 Recommendations ({len(recommendations)})"
                ])
                
                with tab_reviews:
                    if gaps_from_reviews:
                        st.markdown("**Real gaps identified from actual user feedback:**")
                        st.markdown("")
                        for gap in gaps_from_reviews[:10]:
                            _render_enhanced_gap_card(gap, st)
                    else:
                        st.info(
                            "No significant feature gaps were identified from user reviews. "
                            "This could mean the product meets user expectations well, or "
                            "the reviews don't contain specific feature requests."
                        )
                
                with tab_ai:
                    if llm_suggestions:
                        st.markdown("**AI-generated feature suggestions based on market analysis:**")
                        st.caption("*These are creative ideas generated by AI to help identify improvement opportunities.*")
                        st.markdown("")
                        for suggestion in llm_suggestions[:10]:
                            _render_llm_suggestion_card(suggestion, st)
                    else:
                        st.info("No AI suggestions were generated for this analysis.")
                
                with tab_actions:
                    if recommendations:
                        st.markdown("**Prioritized recommendations for product improvement:**")
                        st.markdown("")
                        for rec in recommendations[:10]:
                            _render_gap_recommendation(rec, st)
                    else:
                        st.info("No specific recommendations available.")
        
        # Sentiment Analysis Section (Requirement 5.4)
        # Create expandable section with pain points and loved features
        sentiment_summary = detailed_results.get("sentiment_summary", {})
        if sentiment_summary:
            with st.expander("💭 Sentiment Analysis", expanded=False):
                st.markdown("Insights from user reviews of competitor products.")
                st.markdown("")  # Add spacing
                
                # Display overall trends if available
                overall_trends = sentiment_summary.get("overall_trends", {})
                if overall_trends:
                    st.markdown("### 📊 Overall Trends")
                    for key, value in overall_trends.items():
                        st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")
                    st.markdown("")  # Add spacing
                
                # Pain points
                pain_points = sentiment_summary.get("pain_points", [])
                if pain_points:
                    st.markdown("### 😞 Pain Points")
                    st.caption("Common complaints and frustrations from users")
                    for idx, pain in enumerate(pain_points, 1):
                        if isinstance(pain, dict):
                            pain_text = pain.get("text", str(pain))
                        else:
                            pain_text = str(pain)
                        st.markdown(f"{idx}. {pain_text}")
                    st.markdown("")  # Add spacing
                else:
                    # Check if we have promotional content info
                    promotional_count = sentiment_summary.get("promotional_count", 0)
                    total_reviews = sentiment_summary.get("total_reviews", 0)
                    
                    if promotional_count > 0 and total_reviews > 0:
                        promo_percentage = round((promotional_count / total_reviews) * 100, 1)
                        st.warning(
                            f"⚠️ No pain points identified. Note: {promotional_count}/{total_reviews} "
                            f"({promo_percentage}%) of analyzed content appears to be promotional/announcement material "
                            f"rather than user feedback. This may limit pain point extraction."
                        )
                    else:
                        st.info("ℹ️ No pain points identified from user reviews. This could mean:")
                        st.caption(
                            "• The available reviews are mostly positive or neutral\n"
                            "• Reviews in the dataset may be limited or promotional\n"
                            "• Consider analyzing more reviews for better insights"
                        )
                
                # Loved features
                loved_features = sentiment_summary.get("loved_features", [])
                if loved_features:
                    st.markdown("### ❤️ Loved Features")
                    st.caption("Features that users appreciate and praise")
                    for idx, feature in enumerate(loved_features, 1):
                        if isinstance(feature, dict):
                            feature_text = feature.get("text", str(feature))
                        else:
                            feature_text = str(feature)
                        st.markdown(f"{idx}. {feature_text}")
                else:
                    # Check if we have promotional content info
                    promotional_count = sentiment_summary.get("promotional_count", 0)
                    total_reviews = sentiment_summary.get("total_reviews", 0)
                    
                    if promotional_count > 0 and total_reviews > 0:
                        promo_percentage = round((promotional_count / total_reviews) * 100, 1)
                        st.warning(
                            f"⚠️ No loved features identified. Note: {promotional_count}/{total_reviews} "
                            f"({promo_percentage}%) of analyzed content appears to be promotional/announcement material "
                            f"rather than user feedback. This may limit feature extraction."
                        )
                    else:
                        st.info("ℹ️ No loved features identified from user reviews. This could mean:")
                        st.caption(
                            "• The available reviews are mostly neutral or critical\n"
                            "• Reviews in the dataset may be limited or promotional\n"
                            "• Consider analyzing more reviews for better insights"
                        )
        
        # Recommendations Section (Requirement 5.5)
        # Create expandable section with priority indicators
        recommendations = detailed_results.get("recommendations", [])
        if recommendations:
            with st.expander("💡 Recommendations", expanded=False):
                st.markdown("Strategic recommendations based on competitive analysis.")
                st.markdown("")  # Add spacing
                
                for idx, rec in enumerate(recommendations, 1):
                    if isinstance(rec, dict):
                        # Extract recommendation details
                        priority = rec.get("priority", "medium").lower()
                        title = rec.get("title", rec.get("text", rec.get("recommendation", f"Recommendation {idx}")))
                        description = rec.get("description", "")
                        rationale = rec.get("rationale", "")
                        
                        # Add priority indicator emoji
                        priority_emoji = {
                            "high": "🔴",
                            "medium": "🟡",
                            "low": "🟢"
                        }
                        emoji = priority_emoji.get(priority, "⚪")
                        
                        # Display recommendation with priority
                        st.markdown(f"### {emoji} {idx}. {title}")
                        st.caption(f"Priority: {priority.capitalize()}")
                        
                        if description:
                            st.markdown(description)
                        
                        if rationale:
                            st.markdown(f"*Rationale:* {rationale}")
                        
                        st.markdown("")  # Add spacing
                    else:
                        # Simple string recommendation
                        st.markdown(f"{idx}. {rec}")
                        st.markdown("")  # Add spacing
        else:
            # Show message if no recommendations available
            st.info("No recommendations available for this analysis.")
    
    # Display metadata using dedicated function
    render_analysis_metadata(results)
    
    # Display download section (Task 16)
    # Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    render_download_section(results)
    
    # Display "Start New Analysis" button (Task 17)
    # Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 9.1, 9.2
    add_spacing("large")
    render_section_header("Start Over", "🔄", "Begin a new analysis")
    
    # Create two columns for button and info - responsive layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Add "Start New Analysis" button when results are displayed (Requirement 10.1)
        if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
            # Clear all analysis results and error messages (Requirement 10.2, 10.3)
            reset_analysis_state()
            # Clear product idea input to reset interface to initial state (Requirement 10.3)
            st.session_state.product_idea = ""
            # Clear previous error messages and warnings (Requirement 10.5)
            st.session_state.analysis_error = None
            # Configuration preferences (max_competitors, output_format) are preserved automatically (Requirement 10.4)
            # Rerun to update UI
            st.rerun()
    
    with col2:
        st.info(
            "Click to clear results and start a new analysis. "
            "Your configuration preferences will be preserved."
        )


def render_main_content():
    """
    Render the main content area for input and results - COMPACT LAYOUT.
    
    DESIGN-ONLY CHANGES: Compact spacing, better parallel alignment
    
    Displays:
    - Backend status and dataset statistics
    - Product idea input section
    - Results display area
    
    Requirements: 1.1, 1.5, 7.1, 7.2, 7.3, 7.4, 7.5, 3.3, 3.4, 9.1, 9.2
    """
    # Check backend connectivity and display status
    render_backend_status()
    
    add_spacing("small")
    
    # Create main content columns for better parallel alignment - responsive on mobile
    col1, col2 = st.columns([3, 1], gap="medium")
    
    with col1:
        # Render product idea input section
        render_product_idea_input()
    
    with col2:
        render_section_header("Quick Stats", "📊")
        
        # Display dataset statistics if available
        if st.session_state.backend_connected and st.session_state.dataset_stats:
            render_dataset_stats(st.session_state.dataset_stats)
        else:
            st.info("Dataset statistics unavailable")
    
    add_spacing("small")
    
    # Results area - display results or placeholder
    render_section_header("📈 Analysis Results", "", "Comprehensive competitive insights")
    render_analysis_results()


if __name__ == "__main__":
    main()
