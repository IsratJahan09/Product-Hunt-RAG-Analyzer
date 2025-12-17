"""
Evaluation report generator for Product Hunt RAG Analyzer.

This module generates evaluation reports in multiple formats (JSON, HTML, Markdown)
with executive summaries and detailed metrics for each component.

Requirements: 13.5
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationReportGenerator:
    """
    Generates evaluation reports in multiple formats.
    
    Supports JSON, HTML, and Markdown formats with:
    - Executive summary with overall system health (PASS/WARN/FAIL)
    - Detailed metrics for each component (retrieval, sentiment, features, system)
    - Timestamps and metadata
    """
    
    def __init__(self):
        """Initialize EvaluationReportGenerator."""
        logger.info("EvaluationReportGenerator initialized")
    
    def generate_json_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Generate evaluation report in JSON format.
        
        Args:
            evaluation_results: Results from validation runner
            output_path: Path to save JSON report
        """
        logger.info(f"Generating JSON report: {output_path}")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "evaluation",
            "executive_summary": self._generate_executive_summary(evaluation_results),
            "validation_results": evaluation_results,
            "metadata": {
                "format": "json",
                "version": "1.0"
            }
        }
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to: {output_path}")
    
    def generate_html_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Generate evaluation report in HTML format.
        
        Args:
            evaluation_results: Results from validation runner
            output_path: Path to save HTML report
        """
        logger.info(f"Generating HTML report: {output_path}")
        
        executive_summary = self._generate_executive_summary(evaluation_results)
        
        # Build HTML content
        html_content = self._build_html_report(evaluation_results, executive_summary)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {output_path}")
    
    def generate_markdown_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Generate evaluation report in Markdown format.
        
        Args:
            evaluation_results: Results from validation runner
            output_path: Path to save Markdown report
        """
        logger.info(f"Generating Markdown report: {output_path}")
        
        executive_summary = self._generate_executive_summary(evaluation_results)
        
        # Build Markdown content
        markdown_content = self._build_markdown_report(evaluation_results, executive_summary)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved to: {output_path}")
    
    def _generate_executive_summary(
        self,
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate executive summary with overall system health.
        
        Args:
            evaluation_results: Results from validation runner
            
        Returns:
            Dict with executive summary including health status
        """
        summary = evaluation_results.get("summary", {})
        
        total_validations = summary.get("total_validations", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        overall_status = summary.get("overall_status", "UNKNOWN")
        
        # Determine health status
        if overall_status == "PASS":
            health_status = "PASS"
            health_description = "All validation checks passed successfully"
        elif failed == 0:
            health_status = "PASS"
            health_description = "System is healthy"
        elif failed <= total_validations * 0.2:  # Less than 20% failed
            health_status = "WARN"
            health_description = "Some validation checks failed, but system is mostly functional"
        else:
            health_status = "FAIL"
            health_description = "Multiple validation checks failed, system requires attention"
        
        # Calculate pass rate
        pass_rate = (passed / total_validations * 100) if total_validations > 0 else 0
        
        executive_summary = {
            "health_status": health_status,
            "health_description": health_description,
            "overall_status": overall_status,
            "total_validations": total_validations,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{pass_rate:.1f}%",
            "duration_seconds": summary.get("total_duration_seconds", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add component-level summaries
        executive_summary["components"] = self._summarize_components(evaluation_results)
        
        return executive_summary
    
    def _summarize_components(
        self,
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Summarize results for each component.
        
        Args:
            evaluation_results: Results from validation runner
            
        Returns:
            Dict with component summaries
        """
        components = {}
        
        # Unit-level components
        unit_level = evaluation_results.get("unit_level", {})
        if unit_level:
            components["preprocessing"] = self._summarize_validation(
                unit_level.get("preprocessing", {})
            )
            components["embeddings"] = self._summarize_validation(
                unit_level.get("embeddings", {})
            )
            components["faiss_index"] = self._summarize_validation(
                unit_level.get("faiss_index", {})
            )
        
        # Integration-level components
        integration_level = evaluation_results.get("integration_level", {})
        if integration_level:
            components["rag_retrieval"] = self._summarize_validation(
                integration_level.get("rag_retrieval", {})
            )
            components["sentiment_analysis"] = self._summarize_validation(
                integration_level.get("sentiment_analysis", {})
            )
            components["feature_gaps"] = self._summarize_validation(
                integration_level.get("feature_gaps", {})
            )
        
        # End-to-end components
        end_to_end = evaluation_results.get("end_to_end", {})
        if end_to_end:
            components["full_pipeline"] = self._summarize_validation(
                end_to_end.get("full_pipeline", {})
            )
        
        return components
    
    def _summarize_validation(
        self,
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Summarize a single validation result.
        
        Args:
            validation_result: Result from a validation procedure
            
        Returns:
            Dict with summary information
        """
        if not validation_result:
            return {
                "status": "NOT_RUN",
                "passed": False,
                "metrics": {}
            }
        
        return {
            "status": validation_result.get("status", "UNKNOWN"),
            "passed": validation_result.get("passed", False),
            "metrics": validation_result.get("metrics", {}),
            "duration_seconds": validation_result.get("duration_seconds", 0)
        }
    
    def _build_html_report(
        self,
        evaluation_results: Dict[str, Any],
        executive_summary: Dict[str, Any]
    ) -> str:
        """
        Build HTML report content.
        
        Args:
            evaluation_results: Results from validation runner
            executive_summary: Executive summary
            
        Returns:
            HTML content as string
        """
        health_status = executive_summary["health_status"]
        health_color = {
            "PASS": "#28a745",
            "WARN": "#ffc107",
            "FAIL": "#dc3545"
        }.get(health_status, "#6c757d")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report - Product Hunt RAG Analyzer</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
        }}
        .executive-summary {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .health-status {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 1.2em;
            color: white;
            background-color: {health_color};
            margin-bottom: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .component {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .component h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .status-pass {{ background-color: #28a745; color: white; }}
        .status-fail {{ background-color: #dc3545; color: white; }}
        .status-warn {{ background-color: #ffc107; color: #333; }}
        .status-error {{ background-color: #6c757d; color: white; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Evaluation Report</h1>
        <p>Product Hunt RAG Analyzer - System Validation Results</p>
        <p>Generated: {executive_summary['timestamp']}</p>
    </div>
    
    <div class="executive-summary">
        <h2>Executive Summary</h2>
        <div class="health-status">{health_status}</div>
        <p>{executive_summary['health_description']}</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h4>Total Validations</h4>
                <div class="value">{executive_summary['total_validations']}</div>
            </div>
            <div class="metric-card">
                <h4>Passed</h4>
                <div class="value" style="color: #28a745;">{executive_summary['passed']}</div>
            </div>
            <div class="metric-card">
                <h4>Failed</h4>
                <div class="value" style="color: #dc3545;">{executive_summary['failed']}</div>
            </div>
            <div class="metric-card">
                <h4>Pass Rate</h4>
                <div class="value">{executive_summary['pass_rate']}</div>
            </div>
        </div>
    </div>
"""
        
        # Add component details
        html += self._build_html_components(executive_summary.get("components", {}))
        
        # Add detailed results
        html += self._build_html_detailed_results(evaluation_results)
        
        html += """
    <div class="footer">
        <p>Product Hunt RAG Analyzer - Evaluation Framework v1.0</p>
        <p>For more information, see the evaluation documentation</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def _build_html_components(
        self,
        components: Dict[str, Dict[str, Any]]
    ) -> str:
        """Build HTML for component summaries."""
        if not components:
            return ""
        
        html = '<div class="section"><h2>Component Status</h2>'
        
        for component_name, component_data in components.items():
            status = component_data.get("status", "UNKNOWN")
            passed = component_data.get("passed", False)
            
            status_class = "status-pass" if passed else "status-fail"
            if status == "NOT_RUN":
                status_class = "status-warn"
            elif status == "ERROR":
                status_class = "status-error"
            
            html += f'''
            <div class="component">
                <h3>{component_name.replace("_", " ").title()}
                    <span class="status-badge {status_class}">{status}</span>
                </h3>
            '''
            
            # Add metrics if available
            metrics = component_data.get("metrics", {})
            if metrics:
                html += '<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
                for metric_name, metric_value in metrics.items():
                    formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                    html += f'<tr><td>{metric_name.replace("_", " ").title()}</td><td>{formatted_value}</td></tr>'
                html += '</tbody></table>'
            
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _build_html_detailed_results(
        self,
        evaluation_results: Dict[str, Any]
    ) -> str:
        """Build HTML for detailed results."""
        html = '<div class="section"><h2>Detailed Results</h2>'
        html += '<pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto;">'
        html += json.dumps(evaluation_results, indent=2, default=str)
        html += '</pre></div>'
        return html
    
    def _build_markdown_report(
        self,
        evaluation_results: Dict[str, Any],
        executive_summary: Dict[str, Any]
    ) -> str:
        """
        Build Markdown report content.
        
        Args:
            evaluation_results: Results from validation runner
            executive_summary: Executive summary
            
        Returns:
            Markdown content as string
        """
        health_status = executive_summary["health_status"]
        health_emoji = {
            "PASS": "✅",
            "WARN": "⚠️",
            "FAIL": "❌"
        }.get(health_status, "❓")
        
        markdown = f"""# Evaluation Report - Product Hunt RAG Analyzer

**Generated:** {executive_summary['timestamp']}

---

## Executive Summary

### System Health: {health_emoji} {health_status}

{executive_summary['health_description']}

### Overall Metrics

| Metric | Value |
|--------|-------|
| Total Validations | {executive_summary['total_validations']} |
| Passed | {executive_summary['passed']} ✅ |
| Failed | {executive_summary['failed']} ❌ |
| Pass Rate | {executive_summary['pass_rate']} |
| Duration | {executive_summary['duration_seconds']:.2f}s |

---

## Component Status

"""
        
        # Add component details
        components = executive_summary.get("components", {})
        for component_name, component_data in components.items():
            status = component_data.get("status", "UNKNOWN")
            passed = component_data.get("passed", False)
            
            status_emoji = "✅" if passed else "❌"
            if status == "NOT_RUN":
                status_emoji = "⏭️"
            elif status == "ERROR":
                status_emoji = "⚠️"
            
            markdown += f"### {status_emoji} {component_name.replace('_', ' ').title()}\n\n"
            markdown += f"**Status:** {status}\n\n"
            
            # Add metrics if available
            metrics = component_data.get("metrics", {})
            if metrics:
                markdown += "**Metrics:**\n\n"
                markdown += "| Metric | Value |\n"
                markdown += "|--------|-------|\n"
                for metric_name, metric_value in metrics.items():
                    formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                    markdown += f"| {metric_name.replace('_', ' ').title()} | {formatted_value} |\n"
                markdown += "\n"
            
            duration = component_data.get("duration_seconds", 0)
            if duration > 0:
                markdown += f"**Duration:** {duration:.2f}s\n\n"
            
            markdown += "---\n\n"
        
        # Add detailed results section
        markdown += "## Detailed Results\n\n"
        markdown += "```json\n"
        markdown += json.dumps(evaluation_results, indent=2, default=str)
        markdown += "\n```\n\n"
        
        markdown += "---\n\n"
        markdown += "*Product Hunt RAG Analyzer - Evaluation Framework v1.0*\n"
        
        return markdown
