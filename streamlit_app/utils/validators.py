"""
Input validation utilities for the Streamlit frontend.

This module provides validation functions for user inputs to ensure
they meet the requirements before being sent to the backend API.
"""

from typing import Tuple, Optional


def validate_product_idea(product_idea: str) -> Tuple[bool, Optional[str]]:
    """
    Validate product idea input.
    
    Requirements: 1.2, 1.3
    - Product idea must be at least 10 characters long
    
    Args:
        product_idea: The product idea string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passes, False otherwise
        - error_message: None if valid, error description if invalid
    """
    if not product_idea:
        return False, None  # Empty input, no error message yet
    
    if len(product_idea.strip()) < 10:
        return False, "Product idea must be at least 10 characters long."
    
    return True, None


def validate_competitor_count(count: int) -> Tuple[bool, Optional[str]]:
    """
    Validate competitor count parameter.
    
    Requirements: 2.2
    - Competitor count must be between 1 and 10 (inclusive)
    
    Args:
        count: The number of competitors to analyze
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passes, False otherwise
        - error_message: None if valid, error description if invalid
    """
    if not isinstance(count, int):
        return False, "Competitor count must be an integer."
    
    if count < 1 or count > 10:
        return False, "Competitor count must be between 1 and 10."
    
    return True, None


def validate_export_format(format_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate export format selection.
    
    Requirements: 2.3
    - Export format must be one of: json, markdown, pdf
    
    Args:
        format_type: The export format string
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passes, False otherwise
        - error_message: None if valid, error description if invalid
    """
    valid_formats = {"json", "markdown", "pdf"}
    
    if not format_type:
        return False, "Export format must be specified."
    
    if format_type.lower() not in valid_formats:
        return False, f"Export format must be one of: {', '.join(valid_formats)}."
    
    return True, None
