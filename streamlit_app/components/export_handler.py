"""
Export Handler Module for Streamlit Frontend.

This module provides functionality for generating and downloading analysis
reports in various formats (JSON, Markdown, PDF).
"""

import json
from datetime import datetime
from typing import Dict, Any


def generate_download_filename(format: str) -> str:
    """
    Generate a timestamped filename for download.
    
    Creates a filename with the current timestamp to ensure uniqueness
    and provide context about when the analysis was performed.
    
    Args:
        format: File format extension ("json", "markdown", or "pdf")
    
    Returns:
        Filename string with timestamp in format:
        "product_hunt_analysis_YYYYMMDD_HHMMSS.{format}"
    
    Example:
        >>> generate_download_filename("json")
        "product_hunt_analysis_20231124_143052.json"
    
    Requirements: 6.2, 6.3
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Map format to file extension
    extension_map = {
        "json": "json",
        "markdown": "md",
        "pdf": "pdf"
    }
    
    extension = extension_map.get(format.lower(), format.lower())
    
    # Generate filename
    filename = f"product_hunt_analysis_{timestamp}.{extension}"
    
    return filename


def prepare_download_data(results: Dict[str, Any], format: str) -> bytes:
    """
    Prepare analysis results data for download in the specified format.
    
    Converts the analysis results dictionary into the appropriate format
    for download (JSON, Markdown, or PDF).
    
    Args:
        results: Analysis results dictionary containing all analysis data
        format: Desired output format ("json", "markdown", or "pdf")
    
    Returns:
        Bytes object containing the formatted data ready for download
    
    Raises:
        ValueError: If format is not supported
    
    Requirements: 6.2, 6.3, 6.4
    """
    format_lower = format.lower()
    
    if format_lower == "json":
        return _prepare_json_data(results)
    elif format_lower == "markdown":
        return _prepare_markdown_data(results)
    elif format_lower == "pdf":
        return _prepare_pdf_data(results)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _prepare_json_data(results: Dict[str, Any]) -> bytes:
    """
    Prepare data in JSON format.
    
    Args:
        results: Analysis results dictionary
    
    Returns:
        JSON data as bytes
    """
    # Convert to pretty-printed JSON
    json_str = json.dumps(results, indent=2, ensure_ascii=False)
    return json_str.encode('utf-8')


def _prepare_markdown_data(results: Dict[str, Any]) -> bytes:
    """
    Prepare data in Markdown format.
    
    Converts the analysis results into a well-formatted Markdown document
    with sections for all analysis components.
    
    Args:
        results: Analysis results dictionary
    
    Returns:
        Markdown data as bytes
    """
    lines = []
    
    # Title and metadata
    lines.append("# Product Hunt Competitive Analysis Report")
    lines.append("")
    lines.append(f"**Analysis ID:** {results.get('analysis_id', 'N/A')}")
    lines.append(f"**Generated:** {results.get('generated_at', 'N/A')}")
    lines.append(f"**Confidence Score:** {results.get('confidence_score', 0) * 100:.1f}%")
    lines.append(f"**Processing Time:** {results.get('processing_time_ms', 0)}ms")
    lines.append("")
    
    # Product idea
    lines.append("## Product Idea")
    lines.append("")
    lines.append(results.get('product_idea', 'N/A'))
    lines.append("")
    
    # Competitors
    lines.append("## Identified Competitors")
    lines.append("")
    competitors = results.get('competitors_identified', [])
    if competitors:
        for idx, comp in enumerate(competitors, 1):
            if isinstance(comp, dict):
                name = comp.get('name', f'Competitor {idx}')
                lines.append(f"{idx}. **{name}**")
                if 'relevance_score' in comp:
                    lines.append(f"   - Relevance Score: {comp['relevance_score']:.2f}")
            else:
                lines.append(f"{idx}. {comp}")
    else:
        lines.append("No competitors identified.")
    lines.append("")
    
    # Detailed results
    detailed_results = results.get('results', {})
    
    # Market positioning
    market_positioning = detailed_results.get('market_positioning', {})
    if market_positioning:
        lines.append("## Market Positioning")
        lines.append("")
        
        summary = market_positioning.get('summary', '')
        if summary:
            lines.append("### Summary")
            lines.append("")
            lines.append(summary)
            lines.append("")
        
        diff_opportunities = market_positioning.get('differentiation_opportunities', [])
        if diff_opportunities:
            lines.append("### Differentiation Opportunities")
            lines.append("")
            for idx, opp in enumerate(diff_opportunities, 1):
                lines.append(f"{idx}. {opp}")
            lines.append("")
    
    # Feature gaps
    feature_gaps = detailed_results.get('feature_gaps', {})
    if feature_gaps:
        lines.append("## 🎯 Feature Gap Analysis")
        lines.append("")
        
        # Get structured analysis (where the actual data is)
        structured_analysis = feature_gaps.get('structured_analysis', {})
        summary = structured_analysis.get('summary', {})
        
        # Add summary stats
        high_count = summary.get('high_priority_count', len(structured_analysis.get('high_priority', [])))
        medium_count = summary.get('medium_priority_count', len(structured_analysis.get('medium_priority', [])))
        low_count = summary.get('low_priority_count', len(structured_analysis.get('low_priority', [])))
        total = high_count + medium_count + low_count
        
        if total > 0:
            lines.append("### Overview")
            lines.append("")
            lines.append(f"| Priority | Count | Description |")
            lines.append(f"|----------|-------|-------------|")
            lines.append(f"| 🔥 High | {high_count} | Must-have features with strong user demand |")
            lines.append(f"| ⭐ Medium | {medium_count} | Important features for competitive edge |")
            lines.append(f"| 💡 Low | {low_count} | Nice-to-have features for future |")
            lines.append(f"| **Total** | **{total}** | |")
            lines.append("")
        
        # Category icons for better readability
        category_icons = {
            "ui_ux": "🎨", "performance": "⚡", "pricing": "💰", "integrations": "🔗",
            "functionality": "⚙️", "mobile": "📱", "collaboration": "👥", "security": "🔒",
            "support": "💬", "customization": "🎛️", "automation": "🤖", "reporting": "📊"
        }
        
        priority_config = {
            'high_priority': {'label': '🔥 High Priority Features', 'desc': 'Critical features that should be prioritized in your MVP'},
            'medium_priority': {'label': '⭐ Medium Priority Features', 'desc': 'Important features to consider for competitive advantage'},
            'low_priority': {'label': '💡 Low Priority Features', 'desc': 'Nice-to-have features for future development'}
        }
        
        for priority in ['high_priority', 'medium_priority', 'low_priority']:
            features = structured_analysis.get(priority, [])
            if features:
                config = priority_config[priority]
                lines.append(f"### {config['label']}")
                lines.append(f"*{config['desc']}*")
                lines.append("")
                
                # Create a table for features
                lines.append("| Feature | Category | Mentions | Sentiment | Opportunity Score |")
                lines.append("|---------|----------|----------|-----------|-------------------|")
                
                for feature in features:
                    if isinstance(feature, dict):
                        name = feature.get('name', 'Unknown').title()
                        category = feature.get('category', 'functionality')
                        cat_icon = category_icons.get(category, '📦')
                        cat_label = category.replace('_', ' ').title()
                        frequency = feature.get('frequency', 0)
                        sentiment = feature.get('sentiment', 'neutral')
                        sentiment_score = feature.get('sentiment_score', 0)
                        
                        # Calculate opportunity score
                        opp_score = frequency * (1 + abs(sentiment_score))
                        
                        sentiment_emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😞"
                        
                        lines.append(f"| **{name}** | {cat_icon} {cat_label} | {frequency}x | {sentiment_emoji} {sentiment.title()} | {opp_score:.1f} |")
                    else:
                        lines.append(f"| {feature} | - | - | - | - |")
                
                lines.append("")
        
        # Add strategic insights
        if total > 0:
            lines.append("### 💡 Strategic Insights")
            lines.append("")
            if high_count > 3:
                lines.append(f"- You have **{high_count} high-priority features** to consider - focus on the top 3-5 for your MVP")
            elif high_count > 0:
                lines.append(f"- **{high_count} critical features** identified - these should be in your initial release")
            if medium_count > 0:
                lines.append(f"- **{medium_count} medium-priority features** can give you competitive advantage")
            lines.append("")
    
    # Sentiment analysis
    sentiment_summary = detailed_results.get('sentiment_summary', {})
    if sentiment_summary:
        lines.append("## Sentiment Analysis")
        lines.append("")
        
        pain_points = sentiment_summary.get('pain_points', [])
        if pain_points:
            lines.append("### Pain Points")
            lines.append("")
            for idx, pain in enumerate(pain_points, 1):
                lines.append(f"{idx}. {pain}")
            lines.append("")
        
        loved_features = sentiment_summary.get('loved_features', [])
        if loved_features:
            lines.append("### Loved Features")
            lines.append("")
            for idx, feature in enumerate(loved_features, 1):
                lines.append(f"{idx}. {feature}")
            lines.append("")
    
    # Recommendations
    recommendations = detailed_results.get('recommendations', [])
    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for idx, rec in enumerate(recommendations, 1):
            if isinstance(rec, dict):
                title = rec.get('title', rec.get('text', rec.get('recommendation', f'Recommendation {idx}')))
                priority = rec.get('priority', 'medium')
                lines.append(f"{idx}. **{title}** (Priority: {priority})")
                description = rec.get('description', '')
                if description:
                    lines.append(f"   {description}")
            else:
                lines.append(f"{idx}. {rec}")
        lines.append("")
    
    # Warnings
    warnings = results.get('warnings', [])
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    
    # Join all lines and convert to bytes
    markdown_str = "\n".join(lines)
    return markdown_str.encode('utf-8')


def _prepare_pdf_data(results: Dict[str, Any]) -> bytes:
    """
    Prepare data in PDF format.
    
    Note: This is a placeholder implementation that returns Markdown data.
    Full PDF generation would require additional libraries like reportlab
    or weasyprint. For now, we return Markdown as a fallback.
    
    Args:
        results: Analysis results dictionary
    
    Returns:
        PDF data as bytes (currently returns Markdown as fallback)
    """
    # TODO: Implement proper PDF generation with reportlab or weasyprint
    # For now, return Markdown as a fallback
    return _prepare_markdown_data(results)
