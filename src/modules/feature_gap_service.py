"""
Feature Gap Analysis Service for Product Hunt RAG Analyzer.

This module provides a comprehensive feature gap analysis capability that:
1. Analyzes product reviews to identify missing features or areas where the product
   falls short compared to user expectations or competitor offerings.
2. Falls back to LLM-generated feature suggestions when no gaps are found from reviews.
3. Presents results as a comprehensive feature gap analysis report.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from src.utils.logger import get_logger
from src.modules.feature_analysis import FeatureAnalyzer
from src.modules.llm_generation import LLMGenerator, LLMGenerationError, LLMConnectionError

logger = get_logger(__name__)


class GapSource(str, Enum):
    """Source of the identified feature gap."""
    REVIEW_ANALYSIS = "review_analysis"
    LLM_GENERATED = "llm_generated"
    HYBRID = "hybrid"


@dataclass
class FeatureGap:
    """Represents an identified feature gap."""
    name: str
    description: str
    category: str
    priority: str  # high, medium, low
    source: GapSource
    evidence: List[str] = field(default_factory=list)  # Supporting quotes/data
    frequency: int = 0  # How often mentioned in reviews
    sentiment_score: float = 0.0  # Average sentiment when mentioned
    confidence: float = 0.0  # Confidence in this gap identification


@dataclass
class FeatureGapReport:
    """Comprehensive feature gap analysis report."""
    product_name: str
    product_description: str
    total_reviews_analyzed: int
    gaps_from_reviews: List[FeatureGap]
    llm_generated_suggestions: List[FeatureGap]
    source: GapSource
    summary: str
    recommendations: List[Dict[str, Any]]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureGapService:
    """
    Service for comprehensive feature gap analysis.
    
    Orchestrates the two-step process:
    1. Analyze reviews to find real feature gaps from user feedback
    2. If no gaps found, use LLM to generate creative feature suggestions
    """
    
    # Keywords indicating missing features or user requests
    GAP_INDICATORS = [
        "wish", "would be nice", "should have", "need", "missing",
        "lack", "doesn't have", "can't", "cannot", "unable to",
        "no way to", "would love", "hope they add", "please add",
        "feature request", "suggestion", "improvement", "better if",
        "compared to", "unlike", "competitor has", "other apps have",
        "expected", "disappointed", "frustrating", "annoying",
        "why doesn't", "why can't", "when will", "waiting for"
    ]
    
    # Negative sentiment patterns indicating pain points
    PAIN_POINT_PATTERNS = [
        "problem", "issue", "bug", "broken", "doesn't work",
        "slow", "crash", "error", "fail", "difficult", "confusing",
        "complicated", "hard to", "struggle", "frustrat", "disappoint"
    ]
    
    def __init__(
        self,
        feature_analyzer: Optional[FeatureAnalyzer] = None,
        llm_generator: Optional[LLMGenerator] = None,
        min_gaps_threshold: int = 3
    ):
        """
        Initialize FeatureGapService.
        
        Args:
            feature_analyzer: FeatureAnalyzer instance for review analysis
            llm_generator: LLMGenerator instance for AI suggestions
            min_gaps_threshold: Minimum gaps needed before skipping LLM fallback
        """
        self.feature_analyzer = feature_analyzer or FeatureAnalyzer()
        self.llm_generator = llm_generator
        self.min_gaps_threshold = min_gaps_threshold
        
        logger.info(
            f"FeatureGapService initialized with min_gaps_threshold={min_gaps_threshold}"
        )
    
    def analyze_feature_gaps(
        self,
        product_name: str,
        product_description: str,
        reviews: List[Dict[str, Any]],
        existing_features: Optional[List[str]] = None,
        competitor_features: Optional[Dict[str, List[str]]] = None,
        always_include_llm: bool = True
    ) -> FeatureGapReport:
        """
        Perform comprehensive feature gap analysis.
        
        Analyzes reviews to identify real gaps from user feedback AND
        generates AI-powered feature suggestions. Both are always provided
        to give a complete picture of improvement opportunities.
        
        Args:
            product_name: Name of the product being analyzed
            product_description: Description of the product
            reviews: List of review dictionaries with 'text' and optional 'sentiment'
            existing_features: Optional list of known product features
            competitor_features: Optional dict mapping competitor names to their features
            always_include_llm: Always generate LLM suggestions (default: True)
            
        Returns:
            FeatureGapReport with identified gaps and recommendations
        """
        logger.info(f"Starting feature gap analysis for '{product_name}'")
        logger.info(f"Analyzing {len(reviews)} reviews")
        
        # Step 1: Analyze reviews for feature gaps (real user feedback)
        review_gaps = self._extract_gaps_from_reviews(reviews)
        
        logger.info(f"Found {len(review_gaps)} gaps from review analysis")
        
        # Step 2: Always generate LLM suggestions for creative feature ideas
        llm_suggestions = []
        
        if always_include_llm or len(review_gaps) < self.min_gaps_threshold:
            logger.info("Generating AI-powered feature suggestions...")
            
            llm_suggestions = self._generate_llm_suggestions(
                product_name=product_name,
                product_description=product_description,
                existing_features=existing_features or [],
                competitor_features=competitor_features or {},
                existing_gaps=review_gaps
            )
            
            logger.info(f"Generated {len(llm_suggestions)} AI suggestions")
        
        # Determine source type
        if review_gaps and llm_suggestions:
            source = GapSource.HYBRID
        elif review_gaps:
            source = GapSource.REVIEW_ANALYSIS
        elif llm_suggestions:
            source = GapSource.LLM_GENERATED
        else:
            source = GapSource.REVIEW_ANALYSIS
        
        # Step 3: Generate summary and recommendations
        all_gaps = review_gaps + llm_suggestions
        summary = self._generate_summary(
            product_name=product_name,
            review_gaps=review_gaps,
            llm_suggestions=llm_suggestions,
            total_reviews=len(reviews)
        )
        
        recommendations = self._generate_recommendations(all_gaps)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            review_gaps=review_gaps,
            llm_suggestions=llm_suggestions,
            total_reviews=len(reviews)
        )
        
        report = FeatureGapReport(
            product_name=product_name,
            product_description=product_description,
            total_reviews_analyzed=len(reviews),
            gaps_from_reviews=review_gaps,
            llm_generated_suggestions=llm_suggestions,
            source=source,
            summary=summary,
            recommendations=recommendations,
            confidence_score=confidence,
            metadata={
                "min_gaps_threshold": self.min_gaps_threshold,
                "existing_features_count": len(existing_features) if existing_features else 0,
                "competitors_analyzed": len(competitor_features) if competitor_features else 0
            }
        )
        
        logger.info(
            f"Feature gap analysis complete. Source: {source.value}, "
            f"Total gaps: {len(all_gaps)}, Confidence: {confidence:.2f}"
        )
        
        return report

    def _extract_gaps_from_reviews(
        self,
        reviews: List[Dict[str, Any]]
    ) -> List[FeatureGap]:
        """
        Extract feature gaps from review text using keyword analysis.
        
        Identifies:
        - Explicit feature requests ("wish it had...", "please add...")
        - Comparisons to competitors ("unlike X which has...")
        - Pain points and frustrations
        - Missing functionality mentions
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            List of FeatureGap objects identified from reviews
        """
        if not reviews:
            return []
        
        gaps = {}  # Use dict to aggregate similar gaps
        
        for review in reviews:
            text = review.get('text', '') or review.get('body', '')
            if not text:
                continue
            
            text_lower = text.lower()
            sentiment = review.get('sentiment', 'neutral')
            confidence = review.get('confidence', 0.5)
            
            # Check for gap indicators
            for indicator in self.GAP_INDICATORS:
                if indicator in text_lower:
                    # Extract context around the indicator
                    gap_info = self._extract_gap_context(text, indicator)
                    if gap_info:
                        gap_key = gap_info['category']
                        
                        if gap_key not in gaps:
                            gaps[gap_key] = {
                                'name': gap_info['name'],
                                'description': gap_info['description'],
                                'category': gap_info['category'],
                                'evidence': [],
                                'frequency': 0,
                                'sentiment_scores': [],
                                'confidences': []
                            }
                        
                        gaps[gap_key]['evidence'].append(text[:200])
                        gaps[gap_key]['frequency'] += 1
                        gaps[gap_key]['sentiment_scores'].append(
                            -1.0 if sentiment == 'negative' else 
                            1.0 if sentiment == 'positive' else 0.0
                        )
                        gaps[gap_key]['confidences'].append(confidence)
            
            # Check for pain points (negative sentiment + pain patterns)
            if sentiment == 'negative':
                for pattern in self.PAIN_POINT_PATTERNS:
                    if pattern in text_lower:
                        gap_info = self._extract_pain_point(text, pattern)
                        if gap_info:
                            gap_key = f"pain_{gap_info['category']}"
                            
                            if gap_key not in gaps:
                                gaps[gap_key] = {
                                    'name': gap_info['name'],
                                    'description': gap_info['description'],
                                    'category': gap_info['category'],
                                    'evidence': [],
                                    'frequency': 0,
                                    'sentiment_scores': [],
                                    'confidences': []
                                }
                            
                            gaps[gap_key]['evidence'].append(text[:200])
                            gaps[gap_key]['frequency'] += 1
                            gaps[gap_key]['sentiment_scores'].append(-1.0)
                            gaps[gap_key]['confidences'].append(confidence)
        
        # Convert to FeatureGap objects
        feature_gaps = []
        for gap_key, gap_data in gaps.items():
            avg_sentiment = (
                sum(gap_data['sentiment_scores']) / len(gap_data['sentiment_scores'])
                if gap_data['sentiment_scores'] else 0.0
            )
            avg_confidence = (
                sum(gap_data['confidences']) / len(gap_data['confidences'])
                if gap_data['confidences'] else 0.5
            )
            
            # Determine priority based on frequency and sentiment
            priority = self._determine_priority(
                frequency=gap_data['frequency'],
                sentiment_score=avg_sentiment
            )
            
            feature_gaps.append(FeatureGap(
                name=gap_data['name'],
                description=gap_data['description'],
                category=gap_data['category'],
                priority=priority,
                source=GapSource.REVIEW_ANALYSIS,
                evidence=gap_data['evidence'][:5],  # Keep top 5 evidence
                frequency=gap_data['frequency'],
                sentiment_score=avg_sentiment,
                confidence=avg_confidence
            ))
        
        # Sort by priority and frequency
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        feature_gaps.sort(
            key=lambda x: (priority_order.get(x.priority, 2), -x.frequency)
        )
        
        return feature_gaps
    
    def _extract_gap_context(
        self,
        text: str,
        indicator: str
    ) -> Optional[Dict[str, str]]:
        """Extract context around a gap indicator to identify the feature."""
        import re
        
        text_lower = text.lower()
        idx = text_lower.find(indicator)
        if idx == -1:
            return None
        
        # Get surrounding context (50 chars before, 100 after)
        start = max(0, idx - 50)
        end = min(len(text), idx + len(indicator) + 100)
        context = text[start:end]
        
        # Try to identify the feature category
        category = self._categorize_gap(context)
        
        # Generate name and description
        name = f"{category.replace('_', ' ').title()} Enhancement"
        description = f"Users have expressed interest in improvements related to {category.replace('_', ' ')}"
        
        return {
            'name': name,
            'description': description,
            'category': category
        }
    
    def _extract_pain_point(
        self,
        text: str,
        pattern: str
    ) -> Optional[Dict[str, str]]:
        """Extract pain point information from negative review."""
        text_lower = text.lower()
        idx = text_lower.find(pattern)
        if idx == -1:
            return None
        
        # Get surrounding context
        start = max(0, idx - 30)
        end = min(len(text), idx + len(pattern) + 80)
        context = text[start:end]
        
        category = self._categorize_gap(context)
        
        name = f"{category.replace('_', ' ').title()} Issue"
        description = f"Users have reported issues with {category.replace('_', ' ')}"
        
        return {
            'name': name,
            'description': description,
            'category': category
        }
    
    def _categorize_gap(self, context: str) -> str:
        """Categorize a gap based on context keywords."""
        context_lower = context.lower()
        
        # Check against feature categories
        categories = self.feature_analyzer.FEATURE_CATEGORIES
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in context_lower:
                    return category
        
        return "general"
    
    def _determine_priority(
        self,
        frequency: int,
        sentiment_score: float
    ) -> str:
        """Determine priority based on frequency and sentiment."""
        # High priority: frequently mentioned OR very negative sentiment
        if frequency >= 5 or sentiment_score <= -0.7:
            return "high"
        # Medium priority: moderate frequency or negative sentiment
        elif frequency >= 2 or sentiment_score <= -0.3:
            return "medium"
        else:
            return "low"

    def _generate_llm_suggestions(
        self,
        product_name: str,
        product_description: str,
        existing_features: List[str],
        competitor_features: Dict[str, List[str]],
        existing_gaps: List[FeatureGap]
    ) -> List[FeatureGap]:
        """
        Generate feature suggestions using LLM when review analysis is insufficient.
        
        Args:
            product_name: Name of the product
            product_description: Description of the product
            existing_features: List of known product features
            competitor_features: Dict mapping competitor names to their features
            existing_gaps: Gaps already identified from reviews
            
        Returns:
            List of LLM-generated FeatureGap suggestions
        """
        if not self.llm_generator:
            logger.warning("LLM generator not available, skipping LLM suggestions")
            return []
        
        try:
            # Build prompt for feature suggestions
            prompt = self._build_feature_suggestion_prompt(
                product_name=product_name,
                product_description=product_description,
                existing_features=existing_features,
                competitor_features=competitor_features,
                existing_gaps=existing_gaps
            )
            
            # Generate suggestions using LLM
            response = self._call_llm_for_suggestions(prompt)
            
            # Parse response into FeatureGap objects
            suggestions = self._parse_llm_suggestions(response)
            
            logger.info(f"Generated {len(suggestions)} LLM feature suggestions")
            return suggestions
            
        except (LLMConnectionError, LLMGenerationError) as e:
            logger.error(f"LLM generation failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in LLM suggestion generation: {e}")
            return []
    
    def _build_feature_suggestion_prompt(
        self,
        product_name: str,
        product_description: str,
        existing_features: List[str],
        competitor_features: Dict[str, List[str]],
        existing_gaps: List[FeatureGap]
    ) -> str:
        """Build the prompt for LLM feature suggestions."""
        
        # Format existing features
        features_text = ""
        if existing_features:
            features_text = "Current Features:\n" + "\n".join(
                f"- {f}" for f in existing_features
            )
        else:
            features_text = "Current Features: Not specified"
        
        # Format competitor features
        competitor_text = ""
        if competitor_features:
            competitor_text = "Competitor Features:\n"
            for comp_name, features in competitor_features.items():
                competitor_text += f"\n{comp_name}:\n"
                competitor_text += "\n".join(f"  - {f}" for f in features[:10])
        
        # Format existing gaps
        gaps_text = ""
        if existing_gaps:
            gaps_text = "Already Identified Gaps (from user feedback):\n"
            gaps_text += "\n".join(
                f"- {gap.name}: {gap.description}" for gap in existing_gaps[:5]
            )
        
        prompt = f"""You are a product strategist analyzing feature opportunities for a product.

Product: {product_name}
Description: {product_description}

{features_text}

{competitor_text}

{gaps_text}

Based on this information, suggest 5-7 innovative feature ideas that could:
1. Address unmet user needs
2. Differentiate from competitors
3. Improve user experience
4. Add significant value to the product

For each suggestion, provide:
- name: A concise feature name
- description: What the feature does and why it's valuable
- category: One of (ui_ux, performance, pricing, integrations, functionality, mobile, collaboration, security, support, customization, automation, reporting)
- priority: high, medium, or low based on potential impact
- rationale: Why this feature would benefit users

Format your response as valid JSON:
{{
  "suggestions": [
    {{
      "name": "Feature Name",
      "description": "Detailed description of the feature",
      "category": "category_name",
      "priority": "high|medium|low",
      "rationale": "Why this feature is important"
    }}
  ],
  "market_context": "Brief analysis of the market opportunity"
}}"""
        
        return prompt
    
    def _call_llm_for_suggestions(self, prompt: str) -> Dict[str, Any]:
        """Call LLM to generate feature suggestions."""
        import json
        
        if not self.llm_generator:
            raise LLMConnectionError("LLM generator not configured")
        
        # Ensure connection
        if not self.llm_generator.connect():
            raise LLMConnectionError("Failed to connect to LLM service")
        
        # Generate response
        response_text = self.llm_generator._generate(prompt)
        
        # Parse JSON from response
        try:
            # Find JSON in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[json_start:json_end]
            return json.loads(json_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise LLMGenerationError(f"Invalid JSON response: {e}")
    
    def _parse_llm_suggestions(
        self,
        response: Dict[str, Any]
    ) -> List[FeatureGap]:
        """Parse LLM response into FeatureGap objects."""
        suggestions = []
        
        for item in response.get("suggestions", []):
            try:
                gap = FeatureGap(
                    name=item.get("name", "Unnamed Feature"),
                    description=item.get("description", ""),
                    category=item.get("category", "general"),
                    priority=item.get("priority", "medium"),
                    source=GapSource.LLM_GENERATED,
                    evidence=[item.get("rationale", "")],
                    frequency=0,
                    sentiment_score=0.0,
                    confidence=0.7  # Default confidence for LLM suggestions
                )
                suggestions.append(gap)
            except Exception as e:
                logger.warning(f"Failed to parse suggestion: {e}")
                continue
        
        return suggestions
    
    def _generate_summary(
        self,
        product_name: str,
        review_gaps: List[FeatureGap],
        llm_suggestions: List[FeatureGap],
        total_reviews: int
    ) -> str:
        """Generate a summary of the feature gap analysis."""
        
        # Build summary parts
        parts = []
        
        # Review gaps summary
        if review_gaps:
            high_count = sum(1 for g in review_gaps if g.priority == 'high')
            medium_count = sum(1 for g in review_gaps if g.priority == 'medium')
            low_count = sum(1 for g in review_gaps if g.priority == 'low')
            parts.append(
                f"Identified {len(review_gaps)} feature gaps from {total_reviews} user reviews "
                f"({high_count} high, {medium_count} medium, {low_count} low priority)"
            )
        else:
            parts.append(f"No significant feature gaps found in {total_reviews} user reviews")
        
        # LLM suggestions summary
        if llm_suggestions:
            parts.append(
                f"Generated {len(llm_suggestions)} AI-powered feature suggestions "
                f"based on market analysis and product positioning"
            )
        
        # Combine into final summary
        if review_gaps and llm_suggestions:
            return (
                f"Comprehensive feature gap analysis for {product_name}: {parts[0]}. "
                f"Additionally, {parts[1].lower()}. This provides both user-validated needs "
                f"and innovative improvement opportunities."
            )
        elif review_gaps:
            return f"Feature gap analysis for {product_name}: {parts[0]}."
        elif llm_suggestions:
            return (
                f"Feature gap analysis for {product_name}: {parts[0]}. "
                f"However, {parts[1].lower()} to help identify improvement opportunities."
            )
        else:
            return (
                f"Feature gap analysis for {product_name} completed. "
                f"No significant gaps identified from {total_reviews} reviews. "
                f"The product appears to meet current user expectations well."
            )
    
    def _generate_recommendations(
        self,
        gaps: List[FeatureGap]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from identified gaps."""
        recommendations = []
        
        # Group by priority
        high_priority = [g for g in gaps if g.priority == 'high']
        medium_priority = [g for g in gaps if g.priority == 'medium']
        
        # Generate recommendations for high priority gaps
        for i, gap in enumerate(high_priority[:3], 1):
            recommendations.append({
                "id": i,
                "title": f"Implement {gap.name}",
                "description": gap.description,
                "priority": "high",
                "category": gap.category,
                "source": gap.source.value,
                "rationale": (
                    f"Based on {'user feedback' if gap.source == GapSource.REVIEW_ANALYSIS else 'market analysis'}. "
                    f"{'Mentioned ' + str(gap.frequency) + ' times in reviews.' if gap.frequency > 0 else ''}"
                ),
                "evidence": gap.evidence[:2] if gap.evidence else []
            })
        
        # Add medium priority recommendations
        for i, gap in enumerate(medium_priority[:4], len(recommendations) + 1):
            recommendations.append({
                "id": i,
                "title": f"Consider {gap.name}",
                "description": gap.description,
                "priority": "medium",
                "category": gap.category,
                "source": gap.source.value,
                "rationale": (
                    f"Based on {'user feedback' if gap.source == GapSource.REVIEW_ANALYSIS else 'market analysis'}."
                ),
                "evidence": gap.evidence[:1] if gap.evidence else []
            })
        
        return recommendations
    
    def _calculate_confidence(
        self,
        review_gaps: List[FeatureGap],
        llm_suggestions: List[FeatureGap],
        total_reviews: int
    ) -> float:
        """Calculate overall confidence score for the analysis."""
        
        # Base confidence from review count
        review_confidence = min(0.3, total_reviews / 100 * 0.3)
        
        # Confidence from review gaps (higher if more gaps with evidence)
        gap_confidence = 0.0
        if review_gaps:
            avg_gap_confidence = sum(g.confidence for g in review_gaps) / len(review_gaps)
            gap_confidence = min(0.4, avg_gap_confidence * 0.4)
        
        # Confidence from LLM (lower than review-based)
        llm_confidence = 0.0
        if llm_suggestions:
            llm_confidence = 0.2  # Fixed lower confidence for LLM suggestions
        
        # Combine confidences
        if review_gaps and llm_suggestions:
            # Hybrid: weighted average
            total_confidence = review_confidence + gap_confidence * 0.7 + llm_confidence * 0.3
        elif review_gaps:
            # Review-only: higher confidence
            total_confidence = review_confidence + gap_confidence + 0.2
        elif llm_suggestions:
            # LLM-only: lower confidence
            total_confidence = 0.1 + llm_confidence
        else:
            total_confidence = review_confidence
        
        return min(1.0, max(0.0, total_confidence))
    
    def to_dict(self, report: FeatureGapReport) -> Dict[str, Any]:
        """Convert FeatureGapReport to dictionary for JSON serialization."""
        return {
            "product_name": report.product_name,
            "product_description": report.product_description,
            "total_reviews_analyzed": report.total_reviews_analyzed,
            "source": report.source.value,
            "summary": report.summary,
            "confidence_score": report.confidence_score,
            "gaps_from_reviews": [
                {
                    "name": g.name,
                    "description": g.description,
                    "category": g.category,
                    "priority": g.priority,
                    "source": g.source.value,
                    "evidence": g.evidence,
                    "frequency": g.frequency,
                    "sentiment_score": g.sentiment_score,
                    "confidence": g.confidence
                }
                for g in report.gaps_from_reviews
            ],
            "llm_generated_suggestions": [
                {
                    "name": g.name,
                    "description": g.description,
                    "category": g.category,
                    "priority": g.priority,
                    "source": g.source.value,
                    "evidence": g.evidence,
                    "confidence": g.confidence
                }
                for g in report.llm_generated_suggestions
            ],
            "recommendations": report.recommendations,
            "metadata": report.metadata
        }
    
    def __repr__(self) -> str:
        """String representation of FeatureGapService."""
        return (
            f"FeatureGapService(min_gaps_threshold={self.min_gaps_threshold}, "
            f"llm_available={self.llm_generator is not None})"
        )
