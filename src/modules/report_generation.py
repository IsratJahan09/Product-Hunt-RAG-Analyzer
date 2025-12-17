"""
Report generation module for Product Hunt RAG Analyzer.

This module provides comprehensive report generation functionality, combining
analyses from multiple modules into structured reports with support for
JSON, Markdown, and PDF export formats.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerationError(Exception):
    """Exception raised when report generation fails."""
    pass


class ReportExportError(Exception):
    """Exception raised when report export fails."""
    pass


class ReportGenerator:
    """
    Manages report generation and export for competitive intelligence analysis.
    
    Provides methods to combine analyses from multiple modules, format sections,
    and export reports in multiple formats (JSON, Markdown, PDF).
    """
    
    def __init__(self):
        """Initialize ReportGenerator."""
        logger.info("ReportGenerator initialized")
    
    def generate_comprehensive_report(
        self,
        product_idea: str,
        competitors_data: list,
        llm_insights: dict,
        feature_analysis: dict = None,
        positioning_analysis: dict = None,
        processing_time_ms: int = None
    ) -> dict:
        """Combine all analyses into full report."""
        try:
            logger.info("Generating comprehensive report")
            start_time = datetime.now()
            
            competitor_names = [comp.get('name', 'Unknown') for comp in competitors_data]
            
            metadata = self.format_report_metadata(
                query=product_idea,
                competitors_count=len(competitors_data),
                processing_time=processing_time_ms
            )
            
            feature_gaps = self.format_feature_gaps_section(
                feature_analysis=feature_analysis,
                llm_feature_gaps=llm_insights.get('feature_gaps', {})
            )
            
            sentiment_summary = self.format_sentiment_section(
                sentiment_data=self._extract_sentiment_data(competitors_data),
                llm_sentiment_summary=llm_insights.get('sentiment_summary', {})
            )
            
            market_positioning = self.format_positioning_section(
                positioning_analysis=positioning_analysis,
                llm_positioning=llm_insights.get('market_positioning', {})
            )
            
            recommendations = self.format_recommendations_section(
                llm_recommendations=llm_insights.get('recommendations', [])
            )
            
            report = {
                "metadata": metadata,
                "product_idea": product_idea,
                "competitors_identified": competitor_names,
                "results": {
                    "market_positioning": market_positioning,
                    "feature_gaps": feature_gaps,
                    "sentiment_summary": sentiment_summary,
                    "recommendations": recommendations
                },
                "confidence_score": llm_insights.get('confidence_score', 0.0),
                "competitors_data": competitors_data,
                "feature_analysis": feature_analysis,
                "positioning_analysis": positioning_analysis
            }
            
            generation_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Comprehensive report generated in {generation_time:.2f}ms")
            
            return report
            
        except Exception as e:
            error_msg = f"Failed to generate comprehensive report: {e}"
            logger.error(error_msg, exc_info=True)
            raise ReportGenerationError(error_msg)
    
    def format_report_metadata(self, query: str, competitors_count: int, processing_time: int = None) -> dict:
        """Create metadata section."""
        try:
            logger.debug("Formatting report metadata")
            return {
                "generated_at": datetime.now().isoformat(),
                "query": query,
                "products_analyzed": competitors_count,
                "processing_time_ms": processing_time if processing_time is not None else 0
            }
        except Exception as e:
            logger.error(f"Failed to format metadata: {e}", exc_info=True)
            return {
                "generated_at": datetime.now().isoformat(),
                "query": query,
                "products_analyzed": 0,
                "processing_time_ms": 0,
                "error": str(e)
            }
    
    def format_feature_gaps_section(self, feature_analysis: dict, llm_feature_gaps: dict) -> dict:
        """Merge feature analysis with LLM insights."""
        try:
            logger.debug("Formatting feature gaps section")
            feature_gaps = {"llm_insights": llm_feature_gaps, "structured_analysis": {}}
            
            if feature_analysis:
                high_priority, medium_priority, low_priority = [], [], []
                
                # Handle prioritized_features if available
                if 'prioritized_features' in feature_analysis:
                    for feature in feature_analysis['prioritized_features']:
                        priority = feature.get('priority', 'low')
                        feature_summary = {
                            "name": feature.get('name', ''),
                            "category": feature.get('category', ''),
                            "frequency": feature.get('frequency', 0),
                            "sentiment": feature.get('average_sentiment', 'neutral'),
                            "sentiment_score": feature.get('sentiment_score', 0.0),
                            "sentiment_distribution": feature.get('sentiment_distribution', {}),
                            "average_confidence": feature.get('average_confidence', 0.0)
                        }
                        
                        if priority == 'high':
                            high_priority.append(feature_summary)
                        elif priority == 'medium':
                            medium_priority.append(feature_summary)
                        else:
                            low_priority.append(feature_summary)
                
                # Include aggregated features if available
                aggregated = feature_analysis.get('aggregated_features', [])
                extracted = feature_analysis.get('extracted_features', [])
                
                # Format aggregated features for display
                formatted_aggregated = []
                for agg_feature in aggregated:
                    formatted_aggregated.append({
                        "name": agg_feature.get('name', ''),
                        "category": agg_feature.get('category', ''),
                        "frequency": agg_feature.get('frequency', 0),
                        "sentiment": agg_feature.get('average_sentiment', 'neutral'),
                        "sentiment_score": agg_feature.get('sentiment_score', 0.0),
                        "sentiment_distribution": agg_feature.get('sentiment_distribution', {})
                    })
                
                feature_gaps["structured_analysis"] = {
                    "high_priority": high_priority,
                    "medium_priority": medium_priority,
                    "low_priority": low_priority,
                    "total_features_analyzed": len(high_priority) + len(medium_priority) + len(low_priority),
                    "total_extracted": len(extracted),
                    "total_aggregated": len(aggregated),
                    "aggregated_features": formatted_aggregated[:20] if formatted_aggregated else [],  # Top 20
                    "extracted_features_sample": extracted[:10] if extracted else [],  # Sample of 10
                    "summary": {
                        "high_priority_count": len(high_priority),
                        "medium_priority_count": len(medium_priority),
                        "low_priority_count": len(low_priority),
                        "total_unique_features": len(high_priority) + len(medium_priority) + len(low_priority)
                    }
                }
                
                # Include enhanced feature gap report if available
                feature_gap_report = feature_analysis.get('feature_gap_report')
                if feature_gap_report:
                    feature_gaps["enhanced_analysis"] = {
                        "source": feature_gap_report.get("source", "review_analysis"),
                        "summary": feature_gap_report.get("summary", ""),
                        "confidence_score": feature_gap_report.get("confidence_score", 0.0),
                        "gaps_from_reviews": feature_gap_report.get("gaps_from_reviews", []),
                        "llm_generated_suggestions": feature_gap_report.get("llm_generated_suggestions", []),
                        "recommendations": feature_gap_report.get("recommendations", []),
                        "total_reviews_analyzed": feature_gap_report.get("total_reviews_analyzed", 0)
                    }
                    logger.info(
                        f"Enhanced feature gap analysis included: "
                        f"{len(feature_gap_report.get('gaps_from_reviews', []))} gaps from reviews, "
                        f"{len(feature_gap_report.get('llm_generated_suggestions', []))} LLM suggestions"
                    )
                
                logger.info(
                    f"Feature gaps formatted: {len(high_priority)} high, "
                    f"{len(medium_priority)} medium, {len(low_priority)} low priority"
                )
            else:
                logger.warning("No feature analysis data provided")
                feature_gaps["structured_analysis"] = {
                    "high_priority": [],
                    "medium_priority": [],
                    "low_priority": [],
                    "total_features_analyzed": 0,
                    "total_extracted": 0,
                    "total_aggregated": 0,
                    "aggregated_features": [],
                    "extracted_features_sample": [],
                    "summary": {
                        "high_priority_count": 0,
                        "medium_priority_count": 0,
                        "low_priority_count": 0,
                        "total_unique_features": 0
                    }
                }
            
            logger.debug("Feature gaps section formatted")
            return feature_gaps
            
        except Exception as e:
            logger.error(f"Failed to format feature gaps section: {e}", exc_info=True)
            return {"llm_insights": llm_feature_gaps, "structured_analysis": {}, "error": str(e)}
    
    def format_sentiment_section(self, sentiment_data: dict, llm_sentiment_summary: dict) -> dict:
        """Create sentiment summary with distribution, trends, and key quotes."""
        try:
            logger.debug("Formatting sentiment section")
            
            distribution = sentiment_data.get('distribution', {})
            key_quotes = sentiment_data.get('key_quotes', [])
            statistics = sentiment_data.get('statistics', {})
            
            # Ensure we have meaningful data
            if not distribution:
                distribution = {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0
                }
            
            # Calculate statistics if not provided
            if not statistics:
                total = sum(distribution.values())
                if total > 0:
                    statistics = {
                        "total_reviews": total,
                        "positive_percentage": round((distribution.get("positive", 0) / total * 100), 2),
                        "negative_percentage": round((distribution.get("negative", 0) / total * 100), 2),
                        "neutral_percentage": round((distribution.get("neutral", 0) / total * 100), 2),
                        "average_sentiment_score": self._calculate_average_sentiment_score(key_quotes)
                    }
                else:
                    statistics = {
                        "total_reviews": 0,
                        "positive_percentage": 0,
                        "negative_percentage": 0,
                        "neutral_percentage": 0,
                        "average_sentiment_score": 0
                    }
            
            # Extract pain points and loved features from quotes
            pain_points = []
            loved_features = []
            
            for quote in key_quotes:
                if quote.get('sentiment') == 'negative':
                    pain_points.append({
                        "text": quote.get('text', ''),
                        "competitor": quote.get('competitor', 'Unknown'),
                        "confidence": quote.get('confidence', 0.0)
                    })
                elif quote.get('sentiment') == 'positive':
                    loved_features.append({
                        "text": quote.get('text', ''),
                        "competitor": quote.get('competitor', 'Unknown'),
                        "confidence": quote.get('confidence', 0.0)
                    })
            
            sentiment_section = {
                "llm_insights": llm_sentiment_summary,
                "distribution": distribution,
                "statistics": statistics,
                "trends": sentiment_data.get('trends', []),
                "key_quotes": key_quotes[:10],  # Top 10 quotes
                "pain_points": pain_points[:5],  # Top 5 pain points
                "loved_features": loved_features[:5],  # Top 5 loved features
                "summary": {
                    "total_reviews_analyzed": statistics.get("total_reviews", 0),
                    "dominant_sentiment": self._get_dominant_sentiment(distribution),
                    "pain_points_count": len(pain_points),
                    "loved_features_count": len(loved_features)
                }
            }
            
            logger.info(
                f"Sentiment section formatted: {statistics.get('total_reviews', 0)} reviews, "
                f"{len(pain_points)} pain points, {len(loved_features)} loved features"
            )
            
            return sentiment_section
        except Exception as e:
            logger.error(f"Failed to format sentiment section: {e}", exc_info=True)
            return {
                "llm_insights": llm_sentiment_summary,
                "distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "statistics": {
                    "total_reviews": 0,
                    "positive_percentage": 0,
                    "negative_percentage": 0,
                    "neutral_percentage": 0
                },
                "trends": [],
                "key_quotes": [],
                "pain_points": [],
                "loved_features": [],
                "summary": {
                    "total_reviews_analyzed": 0,
                    "dominant_sentiment": "neutral",
                    "pain_points_count": 0,
                    "loved_features_count": 0
                },
                "error": str(e)
            }
    
    def format_positioning_section(self, positioning_analysis: dict, llm_positioning: dict) -> dict:
        """Create market positioning section."""
        try:
            logger.debug("Formatting positioning section")
            positioning = {"llm_insights": llm_positioning, "structured_analysis": {}}
            
            if positioning_analysis:
                structured = {}
                if 'positioning_keywords' in positioning_analysis:
                    keywords_data = positioning_analysis['positioning_keywords']
                    structured["themes"] = keywords_data.get('themes', {})
                    structured["top_keywords"] = keywords_data.get('keyword_frequencies', {})
                
                if 'target_audience' in positioning_analysis:
                    audience_data = positioning_analysis['target_audience']
                    structured["target_audiences"] = audience_data.get('audiences', {})
                    structured["use_cases"] = audience_data.get('use_cases', {})
                
                positioning["structured_analysis"] = structured
            
            logger.debug("Positioning section formatted")
            return positioning
            
        except Exception as e:
            logger.error(f"Failed to format positioning section: {e}", exc_info=True)
            return {"llm_insights": llm_positioning, "structured_analysis": {}, "error": str(e)}
    
    def format_recommendations_section(self, llm_recommendations: list) -> list:
        """Format actionable recommendations with priority and rationale."""
        try:
            logger.debug(f"Formatting {len(llm_recommendations)} recommendations")
            formatted_recommendations = []
            
            for idx, rec in enumerate(llm_recommendations):
                formatted_rec = {
                    "id": idx + 1,
                    "recommendation": rec.get('recommendation', rec.get('description', '')),
                    "priority": rec.get('priority', 'medium'),
                    "rationale": rec.get('rationale', rec.get('reason', '')),
                    "category": rec.get('category', 'general'),
                    "impact": rec.get('impact', 'medium'),
                    "effort": rec.get('effort', 'medium')
                }
                formatted_recommendations.append(formatted_rec)
            
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            formatted_recommendations.sort(key=lambda x: priority_order.get(x['priority'].lower(), 1))
            
            logger.debug(f"Formatted {len(formatted_recommendations)} recommendations")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Failed to format recommendations: {e}", exc_info=True)
            return llm_recommendations
    
    def export_json(self, report: dict, path: str) -> bool:
        """Export report as JSON format with proper indentation."""
        try:
            logger.info(f"Exporting report to JSON: {path}")
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Report exported to JSON successfully: {path}")
            return True
            
        except IOError as e:
            error_msg = f"Failed to write JSON file: {e}"
            logger.error(error_msg, exc_info=True)
            raise ReportExportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to export JSON report: {e}"
            logger.error(error_msg, exc_info=True)
            raise ReportExportError(error_msg)
    
    def export_markdown(self, report: dict, path: str) -> bool:
        """Export report as Markdown format with headers, sections, and tables."""
        try:
            logger.info(f"Exporting report to Markdown: {path}")
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            md_lines = ["# Product Hunt Competitive Intelligence Report", ""]
            
            metadata = report.get('metadata', {})
            md_lines.extend([
                "## Report Metadata", "",
                f"- **Generated At:** {metadata.get('generated_at', 'N/A')}",
                f"- **Query:** {metadata.get('query', 'N/A')}",
                f"- **Products Analyzed:** {metadata.get('products_analyzed', 0)}",
                f"- **Processing Time:** {metadata.get('processing_time_ms', 0)}ms",
                f"- **Confidence Score:** {report.get('confidence_score', 0.0):.2f}", ""
            ])
            
            md_lines.extend(["## Product Idea", "", report.get('product_idea', 'N/A'), ""])
            
            competitors = report.get('competitors_identified', [])
            md_lines.extend(["## Competitors Identified", ""])
            if competitors:
                for idx, competitor in enumerate(competitors, 1):
                    md_lines.append(f"{idx}. {competitor}")
            else:
                md_lines.append("No competitors identified.")
            md_lines.append("")
            
            results = report.get('results', {})
            
            md_lines.extend(["## Market Positioning", ""])
            positioning = results.get('market_positioning', {})
            llm_positioning = positioning.get('llm_insights', {})
            if isinstance(llm_positioning, dict):
                for key, value in llm_positioning.items():
                    md_lines.extend([f"### {key.replace('_', ' ').title()}", ""])
                    if isinstance(value, str):
                        md_lines.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            md_lines.append(f"- {item if isinstance(item, str) else json.dumps(item)}")
                    else:
                        md_lines.append(json.dumps(value, indent=2))
                    md_lines.append("")
            
            md_lines.extend(["## Recommendations", ""])
            recommendations = results.get('recommendations', [])
            if recommendations:
                md_lines.extend(["| # | Recommendation | Priority | Rationale |", "|---|----------------|----------|-----------|"])
                for rec in recommendations:
                    rec_text = rec.get('recommendation', '').replace('|', '\\|')
                    rationale = rec.get('rationale', '').replace('|', '\\|')
                    md_lines.append(f"| {rec.get('id', '')} | {rec_text} | {rec.get('priority', 'medium')} | {rationale} |")
                md_lines.append("")
            else:
                md_lines.extend(["No recommendations available.", ""])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_lines))
            
            logger.info(f"Report exported to Markdown successfully: {path}")
            return True
            
        except IOError as e:
            error_msg = f"Failed to write Markdown file: {e}"
            logger.error(error_msg, exc_info=True)
            raise ReportExportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to export Markdown report: {e}"
            logger.error(error_msg, exc_info=True)
            raise ReportExportError(error_msg)
    
    def export_pdf(self, report: dict, path: str) -> bool:
        """Export report as PDF format using reportlab with proper styling."""
        try:
            logger.info(f"Exporting report to PDF: {path}")
            
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            except ImportError as e:
                error_msg = "reportlab library not installed. Install with: pip install reportlab"
                logger.error(error_msg)
                raise ReportExportError(error_msg)
            
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
            story = []
            styles = getSampleStyleSheet()
            
            story.append(Paragraph("Product Hunt Competitive Intelligence Report", styles['Title']))
            story.append(Spacer(1, 0.2 * inch))
            
            metadata = report.get('metadata', {})
            story.append(Paragraph("Report Metadata", styles['Heading1']))
            story.append(Spacer(1, 0.1 * inch))
            
            metadata_data = [
                ["Generated At:", metadata.get('generated_at', 'N/A')],
                ["Query:", metadata.get('query', 'N/A')],
                ["Products Analyzed:", str(metadata.get('products_analyzed', 0))],
                ["Processing Time:", f"{metadata.get('processing_time_ms', 0)}ms"],
                ["Confidence Score:", f"{report.get('confidence_score', 0.0):.2f}"]
            ]
            
            metadata_table = Table(metadata_data, colWidths=[2 * inch, 4.5 * inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(metadata_table)
            story.append(Spacer(1, 0.3 * inch))
            
            story.append(Paragraph("Product Idea", styles['Heading1']))
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(report.get('product_idea', 'N/A'), styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))
            
            doc.build(story)
            
            logger.info(f"Report exported to PDF successfully: {path}")
            return True
            
        except ImportError as e:
            error_msg = f"Missing required library for PDF export: {e}"
            logger.error(error_msg)
            raise ReportExportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to export PDF report: {e}"
            logger.error(error_msg, exc_info=True)
            raise ReportExportError(error_msg)
    
    def _get_dominant_sentiment(self, distribution: dict) -> str:
        """Determine the dominant sentiment from distribution."""
        if not distribution:
            return "neutral"
        
        max_sentiment = max(distribution.items(), key=lambda x: x[1])
        return max_sentiment[0] if max_sentiment[1] > 0 else "neutral"
    
    def _calculate_average_sentiment_score(self, key_quotes: list) -> float:
        """Calculate average sentiment score from quotes."""
        if not key_quotes:
            return 0.0
        
        scores = []
        for quote in key_quotes:
            sentiment = quote.get('sentiment', 'neutral')
            if sentiment == 'positive':
                scores.append(1.0)
            elif sentiment == 'negative':
                scores.append(-1.0)
            else:
                scores.append(0.0)
        
        return round(sum(scores) / len(scores), 2) if scores else 0.0
    
    def _extract_sentiment_data(self, competitors_data: list) -> dict:
        """Extract sentiment data from competitors' reviews."""
        try:
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            key_quotes = []
            pain_points = []
            loved_features = []
            promotional_count = 0
            
            for competitor in competitors_data:
                reviews = competitor.get('reviews', [])
                competitor_name = competitor.get('metadata', {}).get('name', competitor.get('name', 'Unknown'))
                
                for review in reviews:
                    # Handle sentiment - it should be present from sentiment analysis
                    sentiment = review.get('sentiment', 'neutral')
                    if sentiment in sentiment_counts:
                        sentiment_counts[sentiment] += 1
                    
                    # Get confidence score
                    confidence = review.get('confidence', 0.0)
                    
                    # Check if this is promotional content
                    is_promotional = review.get('is_promotional', False)
                    if is_promotional:
                        promotional_count += 1
                    
                    # Extract review text - handle multiple possible locations
                    review_text = (
                        review.get('metadata', {}).get('original_text', '') or
                        review.get('text', '') or
                        review.get('body', '')
                    )
                    
                    if not review_text:
                        continue
                    
                    # Skip promotional content for pain points/loved features extraction
                    # but still count it in sentiment distribution
                    if is_promotional:
                        continue
                    
                    # Add high-confidence quotes
                    if confidence > 0.5:  # Lowered threshold to capture more data
                        quote_entry = {
                            'text': review_text[:200],
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'competitor': competitor_name
                        }
                        key_quotes.append(quote_entry)
                        
                        # Categorize as pain point or loved feature
                        if sentiment == 'negative':
                            pain_points.append(quote_entry)
                        elif sentiment == 'positive':
                            loved_features.append(quote_entry)
            
            # Sort by confidence descending
            key_quotes.sort(key=lambda x: x['confidence'], reverse=True)
            pain_points.sort(key=lambda x: x['confidence'], reverse=True)
            loved_features.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate statistics
            total_reviews = sum(sentiment_counts.values())
            statistics = {
                'total_reviews': total_reviews,
                'positive_percentage': round((sentiment_counts['positive'] / total_reviews * 100), 2) if total_reviews > 0 else 0,
                'negative_percentage': round((sentiment_counts['negative'] / total_reviews * 100), 2) if total_reviews > 0 else 0,
                'neutral_percentage': round((sentiment_counts['neutral'] / total_reviews * 100), 2) if total_reviews > 0 else 0
            }
            
            logger.info(
                f"Extracted sentiment data: {total_reviews} reviews "
                f"({promotional_count} promotional), "
                f"{len(key_quotes)} high-confidence quotes, "
                f"{len(pain_points)} pain points, {len(loved_features)} loved features"
            )
            
            return {
                'distribution': sentiment_counts,
                'key_quotes': key_quotes[:10],
                'pain_points': pain_points[:5],
                'loved_features': loved_features[:5],
                'statistics': statistics,
                'trends': [],
                'promotional_count': promotional_count,
                'total_reviews': total_reviews
            }
            
        except Exception as e:
            logger.error(f"Failed to extract sentiment data: {e}", exc_info=True)
            return {
                'distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'key_quotes': [],
                'pain_points': [],
                'loved_features': [],
                'statistics': {},
                'trends': [],
                'promotional_count': 0,
                'total_reviews': 0
            }
    
    def __repr__(self) -> str:
        """String representation of ReportGenerator."""
        return "ReportGenerator()"
