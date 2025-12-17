"""
Feature gap analysis module for Product Hunt RAG Analyzer.

This module provides feature extraction, categorization, and gap analysis
functionality to identify features mentioned in reviews and prioritize
development opportunities based on frequency and sentiment.
"""

import re
from typing import List, Dict, Optional, Union, Tuple
from collections import defaultdict, Counter
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureAnalyzer:
    """
    Manages feature extraction and gap analysis from product reviews.
    
    Provides methods to extract features from review text, categorize them,
    aggregate mentions with sentiment, identify high-priority gaps, and
    generate comparative analysis across competitors.
    """
    
    # Feature keywords organized by category
    FEATURE_CATEGORIES = {
        "ui_ux": [
            "interface", "ui", "ux", "design", "layout", "theme", "appearance",
            "visual", "look", "feel", "aesthetic", "navigation", "menu",
            "dashboard", "screen", "view", "display", "button", "icon"
        ],
        "performance": [
            "performance", "speed", "fast", "slow", "lag", "responsive",
            "loading", "load time", "quick", "efficient", "optimization",
            "latency", "delay", "smooth", "sluggish", "freeze", "crash"
        ],
        "pricing": [
            "price", "pricing", "cost", "expensive", "cheap", "affordable",
            "subscription", "plan", "tier", "free", "paid", "premium",
            "value", "worth", "money", "payment", "billing"
        ],
        "integrations": [
            "integration", "integrations", "integrate", "api", "connect", "sync", "plugin",
            "extension", "compatibility", "compatible", "works with",
            "third-party", "webhook", "import", "export", "zapier", "slack"
        ],
        "functionality": [
            "feature", "functionality", "capability", "function", "tool",
            "option", "setting", "ability", "can", "allows", "enables",
            "supports", "provides", "includes", "offers"
        ],
        "mobile": [
            "mobile", "app", "ios", "android", "phone", "tablet",
            "smartphone", "iphone", "ipad", "mobile app", "native"
        ],
        "collaboration": [
            "collaboration", "collaborate", "team", "share", "sharing",
            "multi-user", "workspace", "invite", "permission", "access",
            "real-time", "concurrent", "together"
        ],
        "security": [
            "security", "secure", "privacy", "safe", "encryption",
            "authentication", "auth", "login", "password", "2fa",
            "two-factor", "gdpr", "compliance", "backup"
        ],
        "support": [
            "support", "help", "customer service", "documentation", "docs",
            "tutorial", "guide", "onboarding", "training", "faq",
            "community", "forum", "chat", "email support"
        ],
        "customization": [
            "customization", "customize", "custom", "personalize",
            "configuration", "configure", "flexible", "adaptable",
            "tailor", "adjust", "modify", "settings"
        ],
        "automation": [
            "automation", "automate", "automatic", "auto", "workflow",
            "trigger", "rule", "schedule", "batch", "bulk", "ai", "ml",
            "machine learning", "intelligent", "smart"
        ],
        "reporting": [
            "report", "reporting", "analytics", "insights", "metrics",
            "statistics", "stats", "dashboard", "visualization", "chart",
            "graph", "export", "data", "tracking"
        ]
    }
    
    def __init__(self):
        """
        Initialize FeatureAnalyzer.
        
        Example:
            >>> analyzer = FeatureAnalyzer()
        """
        logger.info("FeatureAnalyzer initialized")
    
    def extract_features(
        self,
        reviews: List[Dict[str, Union[str, float, List]]]
    ) -> List[Dict[str, Union[str, str, int]]]:
        """
        Identify mentioned features and capabilities from review texts.
        
        Uses keyword extraction and NLP patterns to identify feature mentions
        in review text. Each feature is extracted with its category and the
        review context.
        
        Args:
            reviews: List of review dictionaries with 'text' or 'body' field
                    and optional 'sentiment' and 'confidence' fields
            
        Returns:
            List of extracted features with name, category, and review_id
            
        Raises:
            Exception: If extraction fails
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> reviews = [
            ...     {"text": "The UI is great and performance is excellent", "sentiment": "positive"}
            ... ]
            >>> features = analyzer.extract_features(reviews)
            >>> len(features) > 0
            True
        """
        if not reviews:
            logger.warning("Empty reviews list provided for feature extraction")
            return []
        
        try:
            logger.info(f"Starting feature extraction from {len(reviews)} reviews")
            extracted_features = []
            
            for idx, review in enumerate(reviews):
                # Get review text from either 'text' or 'body' field
                review_text = review.get('text') or review.get('body', '')
                
                if not review_text or not review_text.strip():
                    logger.debug(f"Skipping empty review at index {idx}")
                    continue
                
                review_text_lower = review_text.lower()
                
                # Extract features by matching keywords
                for category, keywords in self.FEATURE_CATEGORIES.items():
                    for keyword in keywords:
                        # Use word boundaries for accurate matching
                        pattern = r'\b' + re.escape(keyword) + r'\b'
                        if re.search(pattern, review_text_lower):
                            extracted_features.append({
                                "name": keyword,
                                "category": category,
                                "review_id": idx,
                                "review_text": review_text,
                                "sentiment": review.get('sentiment', 'neutral'),
                                "confidence": review.get('confidence', 0.0)
                            })
                            logger.debug(
                                f"Extracted feature '{keyword}' (category: {category}) "
                                f"from review {idx}"
                            )
            
            logger.info(
                f"Feature extraction complete: {len(extracted_features)} features "
                f"extracted from {len(reviews)} reviews"
            )
            
            return extracted_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            raise Exception(f"Feature extraction failed: {e}")
    
    def categorize_features(
        self,
        features: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, List[Dict[str, Union[str, int]]]]:
        """
        Categorize features by type.
        
        Groups extracted features into categories such as UI/UX, performance,
        pricing, integrations, functionality, etc.
        
        Args:
            features: List of extracted features with 'category' field
            
        Returns:
            Dictionary mapping category names to lists of features
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> features = [
            ...     {"name": "ui", "category": "ui_ux", "review_id": 0},
            ...     {"name": "speed", "category": "performance", "review_id": 1}
            ... ]
            >>> categorized = analyzer.categorize_features(features)
            >>> "ui_ux" in categorized
            True
        """
        if not features:
            logger.warning("Empty features list provided for categorization")
            return {}
        
        try:
            logger.info(f"Categorizing {len(features)} features")
            categorized = defaultdict(list)
            
            for feature in features:
                category = feature.get('category', 'uncategorized')
                categorized[category].append(feature)
            
            # Convert defaultdict to regular dict
            categorized = dict(categorized)
            
            logger.info(
                f"Categorization complete: {len(categorized)} categories, "
                f"distribution: {[(cat, len(feats)) for cat, feats in categorized.items()]}"
            )
            
            return categorized
            
        except Exception as e:
            logger.error(f"Feature categorization failed: {e}", exc_info=True)
            return {}

    
    def aggregate_feature_mentions(
        self,
        features: List[Dict[str, Union[str, int, float]]],
        sentiments: Optional[List[Dict[str, Union[str, float]]]] = None
    ) -> List[Dict[str, Union[str, int, float, List]]]:
        """
        Count feature frequency and associate with sentiment scores.
        
        Aggregates feature mentions across reviews, counting frequency and
        calculating average sentiment scores for each unique feature.
        
        Args:
            features: List of extracted features with 'name', 'category', 'sentiment', 'confidence'
            sentiments: Optional list of sentiment dictionaries (if not in features)
            
        Returns:
            List of aggregated features with name, category, frequency,
            average_sentiment, sentiment_distribution, and confidence
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> features = [
            ...     {"name": "ui", "category": "ui_ux", "sentiment": "positive", "confidence": 0.9},
            ...     {"name": "ui", "category": "ui_ux", "sentiment": "positive", "confidence": 0.8}
            ... ]
            >>> aggregated = analyzer.aggregate_feature_mentions(features)
            >>> aggregated[0]['frequency']
            2
        """
        if not features:
            logger.warning("Empty features list provided for aggregation")
            return []
        
        try:
            logger.info(f"Aggregating {len(features)} feature mentions")
            
            # Group features by (name, category) tuple
            feature_groups = defaultdict(list)
            
            for feature in features:
                key = (feature.get('name', ''), feature.get('category', 'uncategorized'))
                feature_groups[key].append(feature)
            
            # Aggregate each group
            aggregated = []
            
            for (name, category), group in feature_groups.items():
                # Count frequency
                frequency = len(group)
                
                # Calculate sentiment distribution
                sentiment_counts = Counter(f.get('sentiment', 'neutral') for f in group)
                
                # Calculate average confidence
                confidences = [f.get('confidence', 0.0) for f in group]
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                # Calculate sentiment score (positive=1, neutral=0, negative=-1)
                sentiment_scores = []
                for f in group:
                    sentiment = f.get('sentiment', 'neutral')
                    if sentiment == 'positive':
                        sentiment_scores.append(1.0)
                    elif sentiment == 'negative':
                        sentiment_scores.append(-1.0)
                    else:
                        sentiment_scores.append(0.0)
                
                avg_sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
                
                # Determine overall sentiment based on average score
                if avg_sentiment_score > 0.3:
                    overall_sentiment = 'positive'
                elif avg_sentiment_score < -0.3:
                    overall_sentiment = 'negative'
                else:
                    overall_sentiment = 'neutral'
                
                aggregated.append({
                    "name": name,
                    "category": category,
                    "frequency": frequency,
                    "average_sentiment": overall_sentiment,
                    "sentiment_score": float(avg_sentiment_score),
                    "sentiment_distribution": dict(sentiment_counts),
                    "average_confidence": float(avg_confidence),
                    "mentions": group  # Keep original mentions for reference
                })
                
                logger.debug(
                    f"Aggregated feature '{name}' (category: {category}): "
                    f"frequency={frequency}, sentiment={overall_sentiment}, "
                    f"score={avg_sentiment_score:.2f}"
                )
            
            # Sort by frequency (descending)
            aggregated.sort(key=lambda x: x['frequency'], reverse=True)
            
            logger.info(
                f"Aggregation complete: {len(aggregated)} unique features, "
                f"total mentions: {len(features)}"
            )
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Feature aggregation failed: {e}", exc_info=True)
            return []
    
    def identify_high_priority_gaps(
        self,
        features: List[Dict[str, Union[str, int, float]]],
        sentiment_threshold: float = 0.3,
        frequency_threshold: int = 2
    ) -> List[Dict[str, Union[str, int, float, str]]]:
        """
        Highlight features mentioned frequently with positive sentiment.
        
        Identifies high-priority feature gaps by filtering for features that
        appear frequently and have positive sentiment, indicating user demand.
        
        Args:
            features: List of aggregated features with frequency and sentiment
            sentiment_threshold: Minimum sentiment score for high priority (default: 0.3)
            frequency_threshold: Minimum frequency for high priority (default: 2)
            
        Returns:
            List of high-priority features with priority level assigned
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> features = [
            ...     {"name": "ui", "frequency": 5, "sentiment_score": 0.8, "category": "ui_ux"},
            ...     {"name": "price", "frequency": 1, "sentiment_score": -0.5, "category": "pricing"}
            ... ]
            >>> gaps = analyzer.identify_high_priority_gaps(features)
            >>> len(gaps) > 0
            True
        """
        if not features:
            logger.warning("Empty features list provided for gap identification")
            return []
        
        try:
            logger.info(
                f"Identifying high-priority gaps from {len(features)} features "
                f"(sentiment_threshold={sentiment_threshold}, "
                f"frequency_threshold={frequency_threshold})"
            )
            
            high_priority = []
            medium_priority = []
            low_priority = []
            
            for feature in features:
                frequency = feature.get('frequency', 0)
                sentiment_score = feature.get('sentiment_score', 0.0)
                
                # Create a copy with priority level
                feature_with_priority = feature.copy()
                
                # Determine priority based on frequency and sentiment
                if frequency >= frequency_threshold and sentiment_score >= sentiment_threshold:
                    feature_with_priority['priority'] = 'high'
                    high_priority.append(feature_with_priority)
                    logger.debug(
                        f"High priority: '{feature['name']}' "
                        f"(freq={frequency}, sentiment={sentiment_score:.2f})"
                    )
                elif frequency >= frequency_threshold or sentiment_score >= sentiment_threshold:
                    feature_with_priority['priority'] = 'medium'
                    medium_priority.append(feature_with_priority)
                    logger.debug(
                        f"Medium priority: '{feature['name']}' "
                        f"(freq={frequency}, sentiment={sentiment_score:.2f})"
                    )
                else:
                    feature_with_priority['priority'] = 'low'
                    low_priority.append(feature_with_priority)
            
            # Combine all priorities (high first, then medium, then low)
            prioritized_features = high_priority + medium_priority + low_priority
            
            logger.info(
                f"Gap identification complete: {len(high_priority)} high priority, "
                f"{len(medium_priority)} medium priority, {len(low_priority)} low priority"
            )
            
            return prioritized_features
            
        except Exception as e:
            logger.error(f"Gap identification failed: {e}", exc_info=True)
            return []
    
    def generate_feature_comparison(
        self,
        competitor_features: Dict[str, List[Dict[str, Union[str, int, float]]]]
    ) -> Dict[str, Union[List, Dict]]:
        """
        Create structured comparison across multiple competitors.
        
        Generates a comparative analysis showing which features are present
        in which competitors, along with frequency and sentiment data.
        
        Args:
            competitor_features: Dictionary mapping competitor names to their
                               aggregated feature lists
            
        Returns:
            Dictionary with comparison data including feature matrix,
            category distribution, and competitor summaries
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> competitor_features = {
            ...     "Competitor A": [{"name": "ui", "category": "ui_ux", "frequency": 5}],
            ...     "Competitor B": [{"name": "speed", "category": "performance", "frequency": 3}]
            ... }
            >>> comparison = analyzer.generate_feature_comparison(competitor_features)
            >>> "feature_matrix" in comparison
            True
        """
        if not competitor_features:
            logger.warning("Empty competitor features provided for comparison")
            return {
                "feature_matrix": [],
                "category_distribution": {},
                "competitor_summaries": {},
                "unique_features": [],
                "common_features": []
            }
        
        try:
            logger.info(
                f"Generating feature comparison for {len(competitor_features)} competitors"
            )
            
            # Collect all unique features across competitors
            all_features = set()
            for features in competitor_features.values():
                for feature in features:
                    all_features.add((feature.get('name', ''), feature.get('category', '')))
            
            # Build feature matrix
            feature_matrix = []
            
            for feature_name, category in sorted(all_features):
                feature_row = {
                    "feature": feature_name,
                    "category": category,
                    "competitors": {}
                }
                
                for competitor_name, features in competitor_features.items():
                    # Find this feature in competitor's features
                    matching_features = [
                        f for f in features
                        if f.get('name') == feature_name and f.get('category') == category
                    ]
                    
                    if matching_features:
                        feature_data = matching_features[0]
                        feature_row["competitors"][competitor_name] = {
                            "present": True,
                            "frequency": feature_data.get('frequency', 0),
                            "sentiment": feature_data.get('average_sentiment', 'neutral'),
                            "sentiment_score": feature_data.get('sentiment_score', 0.0)
                        }
                    else:
                        feature_row["competitors"][competitor_name] = {
                            "present": False,
                            "frequency": 0,
                            "sentiment": "neutral",
                            "sentiment_score": 0.0
                        }
                
                feature_matrix.append(feature_row)
            
            # Calculate category distribution per competitor
            category_distribution = {}
            for competitor_name, features in competitor_features.items():
                category_counts = Counter(f.get('category', 'uncategorized') for f in features)
                category_distribution[competitor_name] = dict(category_counts)
            
            # Generate competitor summaries
            competitor_summaries = {}
            for competitor_name, features in competitor_features.items():
                total_features = len(features)
                total_mentions = sum(f.get('frequency', 0) for f in features)
                
                # Calculate sentiment breakdown
                positive_count = sum(
                    1 for f in features if f.get('average_sentiment') == 'positive'
                )
                negative_count = sum(
                    1 for f in features if f.get('average_sentiment') == 'negative'
                )
                neutral_count = total_features - positive_count - negative_count
                
                competitor_summaries[competitor_name] = {
                    "total_features": total_features,
                    "total_mentions": total_mentions,
                    "sentiment_breakdown": {
                        "positive": positive_count,
                        "negative": negative_count,
                        "neutral": neutral_count
                    },
                    "top_features": sorted(
                        features,
                        key=lambda x: x.get('frequency', 0),
                        reverse=True
                    )[:5]  # Top 5 features
                }
            
            # Identify unique and common features
            competitor_names = list(competitor_features.keys())
            
            # Features present in all competitors
            common_features = []
            for feature_row in feature_matrix:
                if all(
                    feature_row["competitors"][comp]["present"]
                    for comp in competitor_names
                ):
                    common_features.append({
                        "feature": feature_row["feature"],
                        "category": feature_row["category"]
                    })
            
            # Features unique to single competitors
            unique_features = []
            for feature_row in feature_matrix:
                present_in = [
                    comp for comp in competitor_names
                    if feature_row["competitors"][comp]["present"]
                ]
                if len(present_in) == 1:
                    unique_features.append({
                        "feature": feature_row["feature"],
                        "category": feature_row["category"],
                        "competitor": present_in[0]
                    })
            
            comparison = {
                "feature_matrix": feature_matrix,
                "category_distribution": category_distribution,
                "competitor_summaries": competitor_summaries,
                "unique_features": unique_features,
                "common_features": common_features,
                "total_competitors": len(competitor_names),
                "total_unique_features": len(all_features)
            }
            
            logger.info(
                f"Feature comparison complete: {len(feature_matrix)} features analyzed, "
                f"{len(common_features)} common, {len(unique_features)} unique"
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Feature comparison generation failed: {e}", exc_info=True)
            return {
                "feature_matrix": [],
                "category_distribution": {},
                "competitor_summaries": {},
                "unique_features": [],
                "common_features": [],
                "error": str(e)
            }
    
    def __repr__(self) -> str:
        """String representation of FeatureAnalyzer."""
        return f"FeatureAnalyzer(categories={len(self.FEATURE_CATEGORIES)})"
