"""
Styling utilities for Streamlit Frontend.

This module provides custom CSS styling and responsive layout utilities
to enhance the visual appearance and mobile-friendliness of the application.

Requirements: 9.1, 9.2
"""

import streamlit as st


def apply_custom_css() -> None:
    """
    Apply custom CSS styling to the Streamlit application.
    
    DESIGN-ONLY CHANGES: Modern SaaS-style UI/UX with professional aesthetics
    - Clean color palette with soft, readable colors
    - Card-based design with subtle shadows
    - Consistent typography and spacing
    - Professional visual hierarchy
    
    Requirements: 9.1, 9.2
    """
    custom_css = """
    <style>
    /* Import Google Fonts for modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Modern SaaS Color Palette - DESIGN-ONLY CHANGES */
    :root {
        --primary-color: #FF6B6B;
        --primary-light: #ff8a8a;
        --primary-dark: #e55555;
        --secondary-color: #f1f5f9;
        --accent-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --text-primary: #2f2f2f;
        --text-secondary: #6b7280;
        --text-muted: #9ca3af;
        --text-light: #d1d5db;
        --background: #ffffff;
        --surface: #f8fafc;
        --border: #e5e7eb;
        --border-light: #f3f4f6;
        --divider-color: #f1f5f9;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        --shadow-md: 0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.03);
        --shadow-lg: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 0.75rem;
        --spacing-lg: 1rem;
        --spacing-xl: 1.5rem;
    }
    
    /* Global typography and spacing - COMPACT DESIGN */
    .main .block-container {
        padding: var(--spacing-lg) var(--spacing-xl);
        max-width: 1200px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--surface);
        line-height: 1.4;
    }
    
    /* Ultra-compact element spacing - DESIGN-ONLY UPDATE */
    .element-container {
        margin-bottom: 0.25rem !important;
    }
    
    .stMarkdown {
        margin-bottom: 0.125rem !important;
        color: var(--text-primary);
        line-height: 1.4;
    }
    
    /* Ultra-compact paragraph spacing */
    .stMarkdown p {
        margin-bottom: 0.25rem !important;
        line-height: 1.4;
    }
    
    /* Compact header spacing with better hierarchy */
    .stMarkdown h1 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.3;
        font-weight: 600;
    }
    
    .stMarkdown h2 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.3;
        font-weight: 600;
    }
    
    .stMarkdown h3 {
        margin-top: 0.375rem !important;
        margin-bottom: 0.125rem !important;
        line-height: 1.3;
        font-weight: 600;
    }
    
    .stMarkdown h4 {
        margin-top: 0.25rem !important;
        margin-bottom: 0.125rem !important;
        line-height: 1.3;
        font-weight: 500;
    }
    
    /* Mobile responsive adjustments - DESIGN-ONLY UPDATE */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.75rem;
        }
        
        .row-widget.stHorizontalBlock {
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .stButton button {
            width: 100%;
            font-size: 0.75rem !important;
            padding: 0.25rem 0.5rem !important;
        }
        
        [data-testid="stSidebar"] {
            padding: 0.75rem 0.75rem;
        }
        
        .stMarkdown h1 {
            font-size: 1.5rem !important;
        }
        
        .stMarkdown h2 {
            font-size: 1.25rem !important;
        }
        
        .stMarkdown h3 {
            font-size: 1.125rem !important;
        }
    }
    
    /* Tablet responsive adjustments */
    @media (max-width: 1024px) and (min-width: 769px) {
        .main .block-container {
            padding: 1rem 1.25rem;
        }
        
        [data-testid="stSidebar"] {
            padding: 1rem 0.875rem;
        }
    }
    
    /* Modern card styling */
    .competitor-card, .modern-card {
        background: var(--background);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .competitor-card:hover, .modern-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-light);
        transform: translateY(-1px);
    }
    
    /* Modern status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: var(--radius-md);
        font-weight: 500;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    .status-badge.success {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .status-badge.error {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .status-badge.warning {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fed7aa;
    }
    
    .status-badge.info {
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    
    /* Modern typography hierarchy - COMPACT & LIGHTER */
    .stMarkdown h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.75rem;
        color: var(--text-primary);
        margin: var(--spacing-md) 0 var(--spacing-xs) 0 !important;
        line-height: 1.3;
        border-bottom: 2px solid var(--divider-color);
        padding-bottom: var(--spacing-xs);
    }
    
    .stMarkdown h2 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.375rem;
        color: var(--text-secondary);
        margin: var(--spacing-lg) 0 var(--spacing-xs) 0 !important;
        line-height: 1.4;
    }
    
    .stMarkdown h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.125rem;
        color: var(--text-secondary);
        margin: var(--spacing-md) 0 var(--spacing-xs) 0 !important;
        line-height: 1.4;
    }
    
    .stMarkdown h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        color: var(--text-muted);
        margin: var(--spacing-sm) 0 var(--spacing-xs) 0 !important;
        line-height: 1.4;
    }
    
    .stMarkdown p {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-bottom: var(--spacing-sm) !important;
    }
    
    /* Modern expander styling */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        padding: 1rem 1.25rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--background);
        border-color: var(--primary-light);
        box-shadow: var(--shadow-sm);
    }
    
    .streamlit-expanderContent {
        padding: 1.25rem;
        background: var(--background);
        border: 1px solid var(--border-light);
        border-top: none;
        border-radius: 0 0 var(--radius-md) var(--radius-md);
        margin-top: -1px;
    }
    
    /* Modern metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    [data-testid="metric-container"] {
        background: var(--background);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }
    
    /* Modern button styling */
    .stButton button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        padding: 0.75rem 1.5rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
        background: var(--background);
        color: var(--text-primary);
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton button:hover {
        background: var(--surface);
        border-color: var(--primary-color);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .stButton button[kind="primary"] {
        background: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    .stButton button[kind="primary"]:hover {
        background: var(--primary-dark);
        border-color: var(--primary-dark);
    }
    
    .stDownloadButton button {
        background: var(--accent-color);
        color: white;
        border-color: var(--accent-color);
    }
    
    .stDownloadButton button:hover {
        background: #059669;
        border-color: #059669;
    }
    
    /* Modern input styling */
    .stTextArea textarea {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--background);
        color: var(--text-primary);
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stTextArea textarea:focus {
        outline: none;
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    .stTextArea textarea::placeholder {
        color: var(--text-muted);
        font-style: italic;
    }
    
    /* Modern slider styling */
    .stSlider {
        padding: 1.5rem 0;
    }
    
    .stSlider > div > div > div > div {
        background: var(--primary-color);
    }
    
    /* Modern divider styling - THIN & ELEGANT */
    hr {
        margin: var(--spacing-sm) 0 !important;
        border: none;
        border-top: 1px solid var(--divider-color);
        height: 1px;
        background: transparent;
    }
    
    /* Custom thin dividers */
    .thin-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--divider-color) 20%, var(--divider-color) 80%, transparent 100%);
        border: none;
        margin: var(--spacing-xs) 0;
    }
    
    /* Modern alert styling */
    .stAlert {
        font-family: 'Inter', sans-serif;
        border-radius: var(--radius-md);
        padding: 1rem !important;
        margin: 1rem 0 !important;
        border: 1px solid;
        box-shadow: var(--shadow-sm);
    }
    
    .stAlert[data-baseweb="notification"] {
        background: var(--surface);
        border-color: var(--border);
        color: var(--text-primary);
    }
    
    .stAlert[data-baseweb="notification"][kind="success"] {
        background: #ecfdf5;
        border-color: #a7f3d0;
        color: #065f46;
    }
    
    .stAlert[data-baseweb="notification"][kind="error"] {
        background: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
    }
    
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background: #fffbeb;
        border-color: #fed7aa;
        color: #92400e;
    }
    
    /* Ultra-compact sidebar styling - DESIGN-ONLY UPDATE */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--surface) 0%, var(--background) 100%);
        border-right: 1px solid var(--border-light);
        padding: 1rem 1rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #374151;
        font-size: 0.9375rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        padding: 0;
        line-height: 1.3;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-secondary);
        font-size: 0.75rem;
        line-height: 1.3;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Ultra-compact sidebar elements */
    [data-testid="stSidebar"] .stSlider {
        margin-bottom: 0.25rem !important;
        padding: 0.125rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio {
        margin-bottom: 0.25rem !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.125rem !important;
    }
    
    /* Export format colors - FORCE BLACK COLOR - DESIGN-ONLY UPDATE */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.75rem !important;
        line-height: 1.2 !important;
        margin: 0.0625rem 0 !important;
        color: #000000 !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #000000 !important;
        background-color: #f9fafb !important;
        border-radius: 4px !important;
        padding: 0.125rem 0.25rem !important;
    }
    
    /* COMPREHENSIVE BLACK COLOR OVERRIDE FOR RADIO BUTTONS */
    [data-testid="stSidebar"] .stRadio * {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] .stRadio div,
    [data-testid="stSidebar"] .stRadio div span,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span,
    [data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] span,
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] span {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Force override any Streamlit default colors */
    [data-testid="stSidebar"] .stRadio label > span,
    [data-testid="stSidebar"] .stRadio label > div > span,
    [data-testid="stSidebar"] .stRadio > div > label,
    [data-testid="stSidebar"] .stRadio > div > label > span {
        color: #000000 !important;
    }
    
    /* Focus states for accessibility */
    [data-testid="stSidebar"] .stRadio input:focus + div {
        outline: 2px solid #3b82f6 !important;
        outline-offset: 2px !important;
        border-radius: 4px !important;
    }
    
    /* Modern spinner styling */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
        border-width: 3px !important;
    }
    
    /* Loading states */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
        border-radius: var(--radius-sm);
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Modern caption styling - LIGHTER */
    .stCaption {
        font-family: 'Inter', sans-serif;
        color: var(--text-light);
        font-size: 0.75rem;
        font-weight: 400;
        margin-top: var(--spacing-xs);
        line-height: 1.4;
    }
    
    /* Container spacing - COMPACT */
    .element-container {
        margin-bottom: var(--spacing-sm) !important;
    }
    
    .stColumn > div {
        padding: var(--spacing-xs);
        background: var(--background);
        border-radius: var(--radius-md);
    }
    
    .streamlit-expanderContent > div {
        padding-top: 0 !important;
    }
    
    /* Compact column spacing */
    .row-widget.stHorizontalBlock {
        gap: var(--spacing-sm);
    }
    
    /* Reduced section spacing */
    .stContainer > div {
        margin-bottom: var(--spacing-xs) !important;
    }
    
    /* Responsive grid for metrics */
    @media (max-width: 768px) {
        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        
        .stMarkdown h1 {
            font-size: 1.8rem;
        }
        
        .stMarkdown h2 {
            font-size: 1.4rem;
        }
    }
    
    /* Modern priority indicators */
    .priority-high {
        color: var(--error-color);
        font-weight: 600;
        background: #fef2f2;
        padding: 0.25rem 0.5rem;
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
    }
    
    .priority-medium {
        color: var(--warning-color);
        font-weight: 600;
        background: #fffbeb;
        padding: 0.25rem 0.5rem;
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
    }
    
    .priority-low {
        color: var(--accent-color);
        font-weight: 600;
        background: #ecfdf5;
        padding: 0.25rem 0.5rem;
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
    }
    
    /* Modern card containers */
    .result-card, .section-card {
        background: var(--background);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: all 0.2s ease;
    }
    
    .result-card:hover, .section-card:hover {
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-light);
    }
    
    /* Smooth transitions */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
    
    /* Loading state styling */
    .stProgress > div > div {
        background-color: #FF6B6B;
    }
    
    /* Table styling if used */
    .stDataFrame {
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    /* Tooltip styling */
    [data-testid="stTooltipIcon"] {
        color: var(--text-muted);
    }
    
    /* ULTRA-COMPACT LAYOUT IMPROVEMENTS - DESIGN-ONLY UPDATE */
    
    /* Ultra-compact expander spacing */
    .streamlit-expanderHeader {
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.125rem;
        font-size: 0.875rem;
        line-height: 1.3;
    }
    
    .streamlit-expanderContent {
        padding: 0.75rem;
        margin-top: 0;
    }
    
    /* Ultra-compact button spacing */
    .stButton {
        margin-bottom: 0.125rem !important;
    }
    
    .stButton button {
        padding: 0.375rem 0.75rem !important;
        font-size: 0.8125rem !important;
        line-height: 1.3 !important;
    }
    
    /* Ultra-compact form elements */
    .stTextArea {
        margin-bottom: 0.25rem !important;
    }
    
    .stTextArea textarea {
        line-height: 1.4 !important;
        font-size: 0.875rem !important;
    }
    
    .stSlider {
        margin-bottom: 0.25rem !important;
        padding: 0.125rem 0 !important;
    }
    
    .stSlider > div > div > div > div {
        font-size: 0.75rem !important;
    }
    
    /* Ultra-compact radio buttons */
    .stRadio {
        margin-bottom: 0.25rem !important;
    }
    
    .stRadio > div {
        gap: 0.125rem !important;
    }
    
    /* Ultra-compact selectbox */
    .stSelectbox {
        margin-bottom: 0.25rem !important;
    }
    
    /* Better column alignment with compact spacing */
    .row-widget.stHorizontalBlock > div {
        padding: 0.125rem;
    }
    
    .row-widget.stHorizontalBlock {
        gap: 0.5rem;
    }
    
    /* Ultra-compact info/warning/error boxes */
    .stAlert {
        margin: 0.25rem 0 !important;
        padding: 0.5rem !important;
        font-size: 0.8125rem !important;
        line-height: 1.3 !important;
    }
    
    /* Better text hierarchy with soft colors */
    .stMarkdown strong {
        color: #374151;
        font-weight: 600;
    }
    
    .stMarkdown em {
        color: #6b7280;
        font-style: normal;
        font-weight: 400;
    }
    
    /* Ultra-compact code blocks */
    .stCodeBlock {
        margin: 0.25rem 0;
        border-radius: var(--radius-sm);
        font-size: 0.8125rem;
    }
    
    /* Compact metric styling */
    [data-testid="metric-container"] {
        padding: 0.5rem !important;
        margin-bottom: 0.25rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        line-height: 1.2 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        line-height: 1.2 !important;
        margin-bottom: 0.125rem !important;
    }
    
    /* Better focus states with soft coral */
    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible {
        outline: 2px solid #FF6B6B;
        outline-offset: 2px;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.2);
    }
    
    /* Success message styling */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Error message styling */
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Accessibility improvements */
    button:focus,
    input:focus,
    textarea:focus,
    select:focus {
        outline: 3px solid #FF6B6B;
        outline-offset: 3px;
    }
    
    /* Focus visible for keyboard navigation */
    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible {
        outline: 3px solid #FF6B6B;
        outline-offset: 3px;
        box-shadow: 0 0 0 4px rgba(255, 107, 107, 0.2);
    }
    
    /* Skip to main content link for screen readers */
    .skip-to-main {
        position: absolute;
        left: -9999px;
        z-index: 999;
        padding: 1rem;
        background-color: #FF6B6B;
        color: white;
        text-decoration: none;
    }
    
    .skip-to-main:focus {
        left: 50%;
        transform: translateX(-50%);
        top: 1rem;
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
        .competitor-card {
            border: 2px solid #000;
        }
        
        button {
            border: 2px solid #000;
        }
    }
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
        
        .stButton button:hover {
            transform: none;
        }
    }
    
    /* Screen reader only content */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border-width: 0;
    }
    
    /* Print styles */
    @media print {
        .stButton,
        [data-testid="stSidebar"],
        .stSpinner {
            display: none;
        }
        
        .main .block-container {
            max-width: 100%;
        }
    }
    
    /* Professional SaaS status badge styling - DESIGN-ONLY UPDATE */
    .status-badge {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.5rem 0.875rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .status-badge:hover {
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.08);
    }
    
    .status-badge.connected {
        background: #f3f4f6;
        border-color: #e5e7eb;
    }
    
    .status-badge.connected span {
        color: #3b82f6;
        font-size: 0.875rem;
        font-weight: 500;
        line-height: 1.4;
    }
    
    .status-badge.error {
        background: #fef2f2;
        border-color: #fecaca;
    }
    
    .status-badge.error span {
        color: #dc2626;
        font-size: 0.875rem;
        font-weight: 500;
        line-height: 1.4;
    }
    
    /* Mobile responsive status badge */
    @media (max-width: 768px) {
        .status-badge {
            padding: 0.375rem 0.75rem;
            gap: 0.375rem;
        }
        
        .status-badge span {
            font-size: 0.8125rem;
        }
    }
    
    /* High contrast mode support for accessibility */
    @media (prefers-contrast: high) {
        .status-badge.connected {
            border-color: #3b82f6;
            border-width: 2px;
        }
        
        .status-badge.connected span {
            color: #1d4ed8;
            font-weight: 600;
        }
        
        .status-badge.error {
            border-color: #dc2626;
            border-width: 2px;
        }
        
        .status-badge.error span {
            color: #b91c1c;
            font-weight: 600;
        }
    }
    
    /* HIGHEST PRIORITY - FORCE BLACK COLOR FOR EXPORT FORMAT OPTIONS */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label *,
    [data-testid="stSidebar"] .stRadio div *,
    [data-testid="stSidebar"] .stRadio span,
    [data-testid="stSidebar"] .stRadio * {
        color: #000000 !important;
    }
    
    /* Override any Streamlit default radio button colors */
    .stRadio label,
    .stRadio label span,
    .stRadio div[role="radiogroup"] label,
    .stRadio div[role="radiogroup"] label span,
    .stRadio div[data-baseweb="radio"] span {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Final override with maximum specificity */
    [data-testid="stSidebar"] .stRadio > div > label > span,
    [data-testid="stSidebar"] .stRadio > div > div > label > span,
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > span {
        color: #000000 !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* CLEAN COMPETITOR LIST STYLING - DESIGN-ONLY UPDATE */
    .stMarkdown p strong {
        color: #2f2f2f !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.4 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact spacing for competitor list */
    .stMarkdown p {
        margin-bottom: 0.5rem !important;
        line-height: 1.4 !important;
    }
    
    /* RESPONSIVE COMPETITOR CARDS - DESIGN-ONLY UPDATE */
    .competitor-card-button {
        background: #f9f9f9;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.875rem 1rem;
        margin-bottom: 0.75rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.9375rem;
        font-weight: 500;
        color: #2f2f2f;
        text-align: center;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        line-height: 1.4;
        min-height: 3rem;
        display: flex;
        align-items: center;
        justify-content: center;
        word-wrap: break-word;
        hyphens: auto;
    }
    
    .competitor-card-button:hover {
        background: #f3f4f6;
        border-color: #d1d5db;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        color: #1f2937;
    }
    
    /* Responsive competitor grid */
    @media (max-width: 768px) {
        .competitor-card-button {
            font-size: 0.875rem;
            padding: 0.75rem 0.875rem;
            min-height: 2.75rem;
        }
    }
    
    @media (max-width: 480px) {
        .competitor-card-button {
            font-size: 0.8125rem;
            padding: 0.625rem 0.75rem;
            min-height: 2.5rem;
            margin-bottom: 0.5rem;
        }
    }
    
    /* Ensure proper column spacing and responsive behavior */
    .row-widget.stHorizontalBlock > div {
        padding: 0 0.25rem;
        min-width: 0; /* Allows flex items to shrink */
    }
    
    @media (max-width: 768px) {
        .row-widget.stHorizontalBlock > div {
            padding: 0 0.125rem;
        }
        
        /* Stack columns on mobile for better readability */
        .row-widget.stHorizontalBlock {
            flex-direction: column;
        }
        
        .competitor-card-button {
            width: 100%;
            margin-left: 0;
            margin-right: 0;
        }
    }
    
    /* Accessibility improvements for competitor cards */
    .competitor-card-button:focus {
        outline: 2px solid #FF6B6B;
        outline-offset: 2px;
    }
    
    .competitor-card-button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)


def create_responsive_columns(num_columns: int, mobile_stack: bool = True) -> list:
    """
    Create responsive columns that stack on mobile devices.
    
    Args:
        num_columns: Number of columns to create
        mobile_stack: Whether columns should stack on mobile (default: True)
    
    Returns:
        List of Streamlit column objects
    
    Requirements: 9.1, 9.2
    """
    # Create columns with equal width
    columns = st.columns(num_columns)
    
    # Add mobile stacking class if enabled
    if mobile_stack:
        st.markdown(
            """
            <style>
            @media (max-width: 768px) {
                .row-widget.stHorizontalBlock {
                    flex-direction: column !important;
                }
                .row-widget.stHorizontalBlock > div {
                    width: 100% !important;
                }
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
    return columns


def add_spacing(size: str = "medium") -> None:
    """
    Add consistent vertical spacing between sections - COMPACT SaaS spacing.
    
    DESIGN-ONLY CHANGES: Reduced spacing for more compact layout
    
    Args:
        size: Spacing size - "small", "medium", or "large"
    
    Requirements: 9.1, 9.2
    """
    spacing_map = {
        "small": "0.5rem",
        "medium": "0.75rem", 
        "large": "1.25rem"
    }
    
    height = spacing_map.get(size, "0.75rem")
    st.markdown(f'<div style="height: {height};"></div>', unsafe_allow_html=True)


def render_section_header(title: str, icon: str = "", description: str = "") -> None:
    """
    Render a MODERN styled section header with compact spacing and lighter colors.
    
    DESIGN-ONLY CHANGES: Compact spacing, lighter font colors, elegant dividers
    
    Args:
        title: Section title
        icon: Optional emoji icon
        description: Optional description text
    
    Requirements: 9.1, 9.2
    """
    header_text = f"{icon} {title}" if icon else title
    
    st.markdown(f"""
    <div style="
        margin: 0.75rem 0 0.5rem 0;
        padding-bottom: 0.375rem;
        border-bottom: 1px solid #f1f5f9;
    ">
        <h3 style="
            color: #9ca3af;
            font-size: 1rem;
            font-weight: 500;
            margin: 0;
            font-family: 'Inter', sans-serif;
        ">{header_text}</h3>
        {f'<p style="color: #9ca3af; font-size: 0.75rem; margin: 0.25rem 0 0 0; font-family: Inter, sans-serif; font-weight: 400;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)


def create_card_container(content_func, border_color: str = "#e0e0e0") -> None:
    """
    Create a styled card container for content.
    
    Args:
        content_func: Function that renders the card content
        border_color: Border color for the card
    
    Requirements: 9.1, 9.2
    """
    st.markdown(
        f"""
        <style>
        .custom-card {{
            background-color: #ffffff;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid {border_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.container():
        content_func()
