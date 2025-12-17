"""
UI Components Module for Streamlit Frontend.

This module provides reusable UI component functions for consistent styling
and display throughout the application. Components include headers, status
badges, competitor cards, metric displays, feature lists, and recommendation
cards.

Accessibility features:
- ARIA labels for screen readers
- Semantic HTML structure
- Keyboard navigation support
- High contrast color schemes
"""

from typing import Dict, Any, List, Optional
import streamlit as st


def render_header() -> None:
    """
    Display the application header with title and description - COMPACT MODERN DESIGN.
    
    DESIGN-ONLY CHANGES: Compact spacing, soft coral accent, lighter typography
    
    Requirements: 9.5
    """
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FF6B6B 0%, #ff8a8a 100%);
        padding: 1.5rem 1.5rem 1.75rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.03);
    ">
        <h1 style="
            color: white;
            font-size: 2rem;
            font-weight: 600;
            margin: 0 0 0.375rem 0;
            font-family: 'Inter', sans-serif;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        ">🚀 Product Hunt RAG Analyzer</h1>
        <p style="
            color: rgba(255, 255, 255, 0.9);
            font-size: 1rem;
            margin: 0;
            font-weight: 400;
            line-height: 1.5;
            font-family: 'Inter', sans-serif;
        ">AI-powered competitive intelligence for Product Hunt products. Get insights on market positioning, feature gaps, sentiment analysis, and strategic recommendations.</p>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str, message: str) -> None:
    """
    Display a modern color-coded status badge with message.
    
    DESIGN-ONLY CHANGES: Modern SaaS-style status badges with clean design
    
    Args:
        status: Status type - "success", "error", "warning", or "info"
        message: Status message to display
    
    Requirements: 7.2, 7.3
    """
    colors = {
        "success": {"bg": "#ecfdf5", "border": "#a7f3d0", "text": "#065f46", "icon": "✅"},
        "error": {"bg": "#fef2f2", "border": "#fecaca", "text": "#991b1b", "icon": "❌"},
        "warning": {"bg": "#fffbeb", "border": "#fed7aa", "text": "#92400e", "icon": "⚠️"},
        "info": {"bg": "#eff6ff", "border": "#bfdbfe", "text": "#1e40af", "icon": "ℹ️"}
    }
    
    color = colors.get(status.lower(), colors["info"])
    
    st.markdown(f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: {color['bg']};
        border: 1px solid {color['border']};
        border-radius: 8px;
        color: {color['text']};
        font-weight: 500;
        font-size: 0.875rem;
        font-family: 'Inter', sans-serif;
        margin-bottom: 0.5rem;
    ">
        <span>{color['icon']}</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_competitor_card(competitor: Dict[str, Any]) -> None:
    """
    Display information about a single competitor in a COMPACT MODERN CARD format.
    
    DESIGN-ONLY CHANGES: Compact spacing, lighter colors, better alignment
    
    Args:
        competitor: Dictionary containing competitor information with keys:
            - name: Competitor product name (required)
            - relevance_score: Relevance score 0.0-1.0 (optional)
            - description: Product description (optional)
            - url: Product URL (optional)
    
    Requirements: 4.2
    """
    name = competitor.get("name", "Unknown Product")
    score = competitor.get("relevance_score", 0.0)
    description = competitor.get("description", "")
    url = competitor.get("url", "")
    
    # Compact modern card with soft coral accent
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease;
    " onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.03)';" 
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px 0 rgba(0, 0, 0, 0.03)';">
        
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div style="flex: 1;">
                <h4 style="margin: 0; font-size: 1rem; font-weight: 500; color: #2f2f2f; font-family: 'Inter', sans-serif;">
                    🎯 {name}
                </h4>
                {f'<p style="margin: 0.375rem 0 0 0; color: #6b7280; font-size: 0.8125rem; line-height: 1.4;">{description[:100]}{"..." if len(description) > 100 else ""}</p>' if description else ''}
            </div>
            {f'''<div style="margin-left: 0.75rem; text-align: center; min-width: 60px;">
                <div style="background: linear-gradient(135deg, #FF6B6B 0%, #ff8a8a 100%); color: white; padding: 0.375rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    {score * 100:.0f}%
                </div>
                <div style="color: #9ca3af; font-size: 0.7rem; margin-top: 0.25rem;">Match</div>
            </div>''' if score > 0 else ''}
        </div>
        
        {f'<div style="margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #f1f5f9;"><a href="{url}" target="_blank" style="color: #FF6B6B; text-decoration: none; font-size: 0.8125rem; font-weight: 500;">🔗 View Product</a></div>' if url else ''}
    </div>
    """, unsafe_allow_html=True)


def render_metric_row(label: str, value: Any) -> None:
    """
    Display a key-value metric in a compact modern card format.
    
    DESIGN-ONLY CHANGES: Compact spacing, soft coral accent, lighter colors
    
    Args:
        label: Metric label/name
        value: Metric value (can be string, number, or other displayable type)
    
    Requirements: 7.2, 7.3
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.875rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease;
    " onmouseover="this.style.boxShadow='0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.03)';" 
       onmouseout="this.style.boxShadow='0 1px 2px 0 rgba(0, 0, 0, 0.03)';">
        <div style="
            color: #9ca3af;
            font-size: 0.7rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.025em;
            margin-bottom: 0.375rem;
            font-family: 'Inter', sans-serif;
        ">{label}</div>
        <div style="
            color: #FF6B6B;
            font-size: 1.5rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        ">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_list(features: List[Dict[str, Any]], priority: str) -> None:
    """
    Display a list of features organized by priority level in modern cards.
    
    DESIGN-ONLY CHANGES: Modern SaaS-style feature cards with priority indicators
    
    Args:
        features: List of feature dictionaries, each containing:
            - name: Feature name (required)
            - description: Feature description (optional)
            - impact: Impact assessment (optional)
        priority: Priority level - "high", "medium", or "low"
    
    Requirements: 5.2, 5.3, 5.5
    """
    # Map priority to colors and styling
    priority_config = {
        "high": {"emoji": "🔴", "bg": "#fef2f2", "border": "#fecaca", "text": "#991b1b", "accent": "#ef4444"},
        "medium": {"emoji": "🟡", "bg": "#fffbeb", "border": "#fed7aa", "text": "#92400e", "accent": "#f59e0b"},
        "low": {"emoji": "�", "bgo": "#ecfdf5", "border": "#a7f3d0", "text": "#065f46", "accent": "#10b981"}
    }
    
    config = priority_config.get(priority.lower(), {
        "emoji": "⚪", "bg": "#f8fafc", "border": "#e2e8f0", "text": "#475569", "accent": "#64748b"
    })
    
    # Modern priority header
    st.markdown(f"""
    <div style="
        background: {config['bg']};
        border: 1px solid {config['border']};
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    ">
        <span style="font-size: 1.25rem;">{config['emoji']}</span>
        <h3 style="
            margin: 0;
            color: {config['text']};
            font-size: 1.125rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        ">{priority.capitalize()} Priority Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not features:
        st.markdown(f"""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            color: #64748b;
            font-style: italic;
        ">
            No {priority.lower()} priority features identified.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display each feature in modern cards
    for idx, feature in enumerate(features, 1):
        feature_name = feature.get("name", f"Feature {idx}")
        description = feature.get("description", "")
        impact = feature.get("impact", "")
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-left: 4px solid {config['accent']};
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        ">
            <div style="
                display: flex;
                align-items: flex-start;
                gap: 0.75rem;
                margin-bottom: {0.75 if description or impact else 0}rem;
            ">
                <div style="
                    background: {config['accent']};
                    color: white;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.75rem;
                    font-weight: 600;
                    flex-shrink: 0;
                ">{idx}</div>
                <div style="flex: 1;">
                    <h4 style="
                        margin: 0;
                        color: #1e293b;
                        font-size: 1rem;
                        font-weight: 600;
                        font-family: 'Inter', sans-serif;
                    ">{feature_name}</h4>
                </div>
            </div>
            
            {f'''<div style="margin-left: 2.25rem; margin-bottom: 0.5rem;">
                <p style="margin: 0; color: #64748b; font-size: 0.875rem; line-height: 1.5;">{description}</p>
            </div>''' if description else ''}
            
            {f'''<div style="margin-left: 2.25rem;">
                <div style="
                    background: #f1f5f9;
                    border-radius: 6px;
                    padding: 0.5rem 0.75rem;
                    display: inline-block;
                ">
                    <span style="color: #475569; font-size: 0.75rem; font-weight: 500;">Impact: {impact}</span>
                </div>
            </div>''' if impact else ''}
        </div>
        """, unsafe_allow_html=True)


def render_recommendation_card(recommendation: Dict[str, Any]) -> None:
    """
    Display a strategic recommendation in a modern card format.
    
    DESIGN-ONLY CHANGES: Modern SaaS-style recommendation cards with clean design
    
    Args:
        recommendation: Dictionary containing:
            - title: Recommendation title (required)
            - description: Detailed description (optional)
            - priority: Priority level - "high", "medium", "low" (optional)
            - rationale: Supporting rationale (optional)
            - action_items: List of action items (optional)
    
    Requirements: 5.5
    """
    # Extract recommendation data
    title = recommendation.get("title", "Recommendation")
    description = recommendation.get("description", "")
    priority = recommendation.get("priority", "medium").lower()
    rationale = recommendation.get("rationale", "")
    action_items = recommendation.get("action_items", [])
    
    # Map priority to colors and styling
    priority_config = {
        "high": {"emoji": "🔴", "bg": "#fef2f2", "border": "#fecaca", "text": "#991b1b", "accent": "#ef4444"},
        "medium": {"emoji": "🟡", "bg": "#fffbeb", "border": "#fed7aa", "text": "#92400e", "accent": "#f59e0b"},
        "low": {"emoji": "🟢", "bg": "#ecfdf5", "border": "#a7f3d0", "text": "#065f46", "accent": "#10b981"}
    }
    
    config = priority_config.get(priority, {
        "emoji": "⚪", "bg": "#f8fafc", "border": "#e2e8f0", "text": "#475569", "accent": "#64748b"
    })
    
    # Modern recommendation card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-left: 4px solid {config['accent']};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';" 
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)';">
        
        <!-- Header with priority badge -->
        <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1rem;">
            <div style="flex: 1;">
                <h3 style="
                    margin: 0 0 0.5rem 0;
                    color: #1e293b;
                    font-size: 1.25rem;
                    font-weight: 600;
                    font-family: 'Inter', sans-serif;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                ">
                    <span>{config['emoji']}</span>
                    <span>{title}</span>
                </h3>
            </div>
            <div style="
                background: {config['bg']};
                border: 1px solid {config['border']};
                color: {config['text']};
                padding: 0.25rem 0.75rem;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.025em;
            ">{priority} Priority</div>
        </div>
        
        <!-- Description -->
        {f'''<div style="margin-bottom: 1rem;">
            <p style="margin: 0; color: #475569; font-size: 0.875rem; line-height: 1.6;">{description}</p>
        </div>''' if description else ''}
        
        <!-- Rationale -->
        {f'''<div style="
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 0.875rem; font-weight: 600;">💡 Rationale</h4>
            <p style="margin: 0; color: #64748b; font-size: 0.875rem; line-height: 1.5;">{rationale}</p>
        </div>''' if rationale else ''}
        
        <!-- Action Items -->
        {f'''<div>
            <h4 style="margin: 0 0 0.75rem 0; color: #1e293b; font-size: 0.875rem; font-weight: 600;">🎯 Action Items</h4>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                {"".join([f'<div style="display: flex; align-items: flex-start; gap: 0.5rem;"><div style="background: {config["accent"]}; color: white; border-radius: 50%; width: 16px; height: 16px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; flex-shrink: 0; margin-top: 0.125rem;">•</div><span style="color: #475569; font-size: 0.875rem; line-height: 1.4;">{item}</span></div>' for item in action_items])}
            </div>
        </div>''' if action_items else ''}
    </div>
    """, unsafe_allow_html=True)


def render_error_details(
    error_message: str,
    error_type: Optional[str] = None,
    request_params: Optional[Dict[str, Any]] = None,
    show_retry: bool = True
) -> None:
    """
    Display comprehensive error information with modern troubleshooting guidance.
    
    DESIGN-ONLY CHANGES: Modern SaaS-style error display with clean cards and styling
    
    Args:
        error_message: The error message to display
        error_type: Type of error (connection, timeout, validation, etc.)
        request_params: Request parameters that caused the error
        show_retry: Whether to show retry button
    
    Requirements: 3.4, 7.3
    """
    # Modern error banner
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
            <h4 style="margin: 0 0 0.25rem 0; color: #991b1b; font-size: 1rem; font-weight: 600; font-family: 'Inter', sans-serif;">Error Occurred</h4>
            <p style="margin: 0; color: #dc2626; font-size: 0.875rem; line-height: 1.4;">{error_message}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern expandable troubleshooting section
    with st.expander("🔍 Detailed Troubleshooting Guide", expanded=False):
        # Error information card
        st.markdown("""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin: 0 0 0.75rem 0; color: #1e293b; font-size: 0.875rem; font-weight: 600;">📋 Error Information</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.code(error_message, language="text")
        
        if error_type:
            st.markdown(f"""
            <div style="
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 6px;
                padding: 0.75rem;
                margin: 0.75rem 0;
            ">
                <span style="color: #1e40af; font-size: 0.875rem; font-weight: 500;">Error Type: {error_type}</span>
            </div>
            """, unsafe_allow_html=True)
        
        if request_params:
            st.markdown("""
            <div style="
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            ">
                <h4 style="margin: 0 0 0.75rem 0; color: #1e293b; font-size: 0.875rem; font-weight: 600;">🔧 Request Parameters</h4>
            </div>
            """, unsafe_allow_html=True)
            st.json(request_params)
        
        # Troubleshooting steps
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        ">
            <h4 style="margin: 0 0 0.75rem 0; color: #15803d; font-size: 0.875rem; font-weight: 600;">🛠️ Troubleshooting Steps</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Context-specific troubleshooting based on error type
        if error_type == "connection":
            st.markdown("""
            **Network/Connection Error Solutions:**
            1. ✅ Verify the backend service is running
            2. 🌐 Check network connectivity
            3. 🔗 Confirm the backend URL is correct
            4. 🛡️ Check firewall settings
            5. 📋 Review backend logs for startup errors
            """)
        elif error_type == "timeout":
            st.markdown("""
            **Timeout Error Solutions:**
            1. ⏱️ The request took longer than expected
            2. 📊 Try reducing the number of competitors
            3. 🖥️ Check backend performance and resource usage
            4. 🤖 Verify Ollama LLM service is responding
            5. ⚙️ Consider increasing timeout settings
            """)
        elif error_type == "validation":
            st.markdown("""
            **Validation Error Solutions:**
            1. ✏️ Ensure product idea is at least 10 characters
            2. 🔢 Verify competitor count is between 1-10
            3. 📄 Check export format is valid (json, markdown, pdf)
            4. 📋 Review input requirements above
            """)
        else:
            st.markdown("""
            **General Troubleshooting Steps:**
            1. 🔍 Check backend service status
            2. ⚙️ Verify all required services are running
            3. 📋 Review backend logs for detailed errors
            4. 🗂️ Ensure FAISS indices are loaded
            5. 🤖 Confirm Ollama LLM service is available
            """)
    
    # Modern retry suggestion for transient errors
    if show_retry and error_type in ["connection", "timeout"]:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        ">
            <div style="
                background: #3b82f6;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
            ">💡</div>
            <div>
                <span style="color: #1e40af; font-size: 0.875rem; font-weight: 500;">This appears to be a transient error. Try using the retry button to attempt the operation again.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
