"""
Market positioning analysis module for Product Hunt RAG Analyzer.

This module provides positioning keyword extraction, target audience identification,
competitive landscape analysis, market saturation detection, and differentiation
recommendations to help understand market positioning opportunities.
"""

import re
from typing import List, Dict, Optional, Union, Tuple
from collections import defaultdict, Counter
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PositioningAnalyzer:
    """
    Manages market positioning analysis for competitive intelligence.
    
    Provides methods to extract positioning keywords, identify target audiences,
    analyze competitive landscape, detect market saturation, and generate
    differentiation recommendations.
    """
    
    # Positioning keyword categories
    POSITIONING_KEYWORDS = {
        "simplicity": [
            "simple", "easy", "straightforward", "intuitive", "user-friendly",
            "minimal", "clean", "effortless", "hassle-free", "no-code"
        ],
        "power": [
            "powerful", "advanced", "professional", "enterprise", "robust",
            "comprehensive", "full-featured", "complete", "all-in-one"
        ],
        "speed": [
            "fast", "quick", "instant", "real-time", "rapid", "immediate",
            "lightning", "speedy", "swift", "efficient"
        ],
        "affordability": [
            "affordable", "cheap", "free", "budget", "cost-effective",
            "economical", "inexpensive", "value", "low-cost"
        ],
        "premium": [
            "premium", "luxury", "high-end", "exclusive", "elite",
            "professional", "enterprise", "top-tier"
        ],
        "innovation": [
            "innovative", "cutting-edge", "revolutionary", "breakthrough",
            "next-generation", "modern", "advanced", "pioneering", "ai-powered"
        ],
        "reliability": [
            "reliable", "stable", "dependable", "trusted", "secure",
            "safe", "proven", "established", "consistent"
        ],
        "collaboration": [
            "collaborative", "team", "shared", "together", "cooperative",
            "multi-user", "workspace", "group"
        ],
        "automation": [
            "automated", "automatic", "smart", "intelligent", "ai",
            "machine learning", "workflow", "streamlined"
        ],
        "customization": [
            "customizable", "flexible", "adaptable", "configurable",
            "personalized", "tailored", "modular"
        ]
    }
    
    # Target audience indicators
    AUDIENCE_INDICATORS = {
        "developers": [
            "developer", "programmer", "coder", "engineer", "dev",
            "technical", "api", "code", "github", "open-source"
        ],
        "designers": [
            "designer", "design", "creative", "ui", "ux", "visual",
            "graphic", "artist", "figma", "sketch"
        ],
        "marketers": [
            "marketer", "marketing", "seo", "campaign", "analytics",
            "growth", "conversion", "lead", "funnel"
        ],
        "entrepreneurs": [
            "entrepreneur", "startup", "founder", "indie", "solopreneur",
            "small business", "bootstrapped", "side project"
        ],
        "enterprises": [
            "enterprise", "corporation", "large company", "organization",
            "team", "department", "scale", "compliance"
        ],
        "freelancers": [
            "freelancer", "consultant", "contractor", "independent",
            "self-employed", "gig", "remote worker"
        ],
        "students": [
            "student", "education", "learning", "academic", "university",
            "college", "school", "course"
        ],
        "content_creators": [
            "creator", "content", "blogger", "writer", "influencer",
            "youtuber", "podcaster", "streamer"
        ],
        "product_managers": [
            "product manager", "pm", "product", "roadmap", "feature",
            "backlog", "sprint", "agile"
        ],
        "sales_teams": [
            "sales", "crm", "pipeline", "deal", "prospect",
            "customer", "client", "account"
        ]
    }
    
    # Use case patterns
    USE_CASE_PATTERNS = {
        "project_management": [
            "project management", "task management", "todo", "planning",
            "organize", "workflow", "productivity"
        ],
        "communication": [
            "communication", "messaging", "chat", "email", "collaboration",
            "meeting", "video call", "conference"
        ],
        "data_analysis": [
            "data analysis", "analytics", "reporting", "dashboard",
            "visualization", "insights", "metrics"
        ],
        "content_creation": [
            "content creation", "writing", "editing", "publishing",
            "blog", "article", "post"
        ],
        "customer_support": [
            "customer support", "help desk", "ticketing", "support",
            "service", "helpdesk"
        ],
        "ecommerce": [
            "ecommerce", "online store", "shop", "cart", "checkout",
            "payment", "inventory"
        ],
        "automation": [
            "automation", "workflow", "integration", "zapier",
            "automate", "trigger", "action"
        ],
        "design": [
            "design", "prototype", "mockup", "wireframe",
            "ui design", "ux design", "visual"
        ]
    }
    
    def __init__(self):
        """
        Initialize PositioningAnalyzer.
        
        Example:
            >>> analyzer = PositioningAnalyzer()
        """
        logger.info("PositioningAnalyzer initialized")
    
    def extract_positioning_keywords(
        self,
        product_descriptions: List[str],
        reviews: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Union[List, Dict]]:
        """
        Identify positioning keywords and themes from product data.
        
        Analyzes product descriptions and reviews to extract positioning keywords
        that indicate how products position themselves in the market.
        
        Args:
            product_descriptions: List of product description texts
            reviews: Optional list of review dictionaries with 'text' or 'body' field
            
        Returns:
            Dictionary with positioning themes, keyword frequencies, and examples
            
        Raises:
            Exception: If extraction fails
            
        Example:
            >>> analyzer = PositioningAnalyzer()
            >>> descriptions = ["A simple and powerful task manager"]
            >>> result = analyzer.extract_positioning_keywords(descriptions)
            >>> "themes" in result
            True
        """
        if not product_descriptions and not reviews:
            logger.warning("No data provided for positioning keyword extraction")
            return {
                "themes": {},
                "keyword_frequencies": {},
                "examples": []
            }
        
        try:
            logger.info(
                f"Extracting positioning keywords from {len(product_descriptions)} "
                f"descriptions and {len(reviews) if reviews else 0} reviews"
            )
            
            # Combine all text
            all_texts = list(product_descriptions)
            if reviews:
                for review in reviews:
                    review_text = review.get('text') or review.get('body', '')
                    if review_text:
                        all_texts.append(review_text)
            
            # Extract keywords by theme
            theme_keywords = defaultdict(list)
            keyword_frequencies = Counter()
            examples = []
            
            for text in all_texts:
                if not text or not text.strip():
                    continue
                
                text_lower = text.lower()
                
                # Check each positioning theme
                for theme, keywords in self.POSITIONING_KEYWORDS.items():
                    for keyword in keywords:
                        # Use word boundaries for accurate matching
                        pattern = r'\b' + re.escape(keyword) + r'\b'
                        matches = re.finditer(pattern, text_lower)
                        
                        for match in matches:
                            theme_keywords[theme].append(keyword)
                            keyword_frequencies[keyword] += 1
                            
                            # Extract context (±30 chars)
                            start = max(0, match.start() - 30)
                            end = min(len(text), match.end() + 30)
                            context = text[start:end].strip()
                            
                            examples.append({
                                "keyword": keyword,
                                "theme": theme,
                                "context": context
                            })
                            
                            logger.debug(
                                f"Found positioning keyword '{keyword}' "
                                f"(theme: {theme})"
                            )
            
            # Calculate theme strengths
            theme_strengths = {}
            for theme, keywords_list in theme_keywords.items():
                theme_strengths[theme] = {
                    "count": len(keywords_list),
                    "unique_keywords": len(set(keywords_list)),
                    "keywords": list(set(keywords_list))
                }
            
            # Sort themes by strength
            sorted_themes = dict(
                sorted(
                    theme_strengths.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True
                )
            )
            
            result = {
                "themes": sorted_themes,
                "keyword_frequencies": dict(keyword_frequencies.most_common(20)),
                "examples": examples[:50],  # Limit examples
                "total_keywords_found": sum(keyword_frequencies.values()),
                "unique_keywords_found": len(keyword_frequencies)
            }
            
            logger.info(
                f"Positioning keyword extraction complete: "
                f"{result['total_keywords_found']} keywords found across "
                f"{len(sorted_themes)} themes"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Positioning keyword extraction failed: {e}", exc_info=True)
            raise Exception(f"Positioning keyword extraction failed: {e}")
    
    def extract_target_audience(
        self,
        texts: List[str]
    ) -> Dict[str, Union[List, Dict, int]]:
        """
        Identify target audience indicators and use case mentions.
        
        Analyzes text to identify mentions of target audiences and use cases,
        helping understand who the products are targeting.
        
        Args:
            texts: List of text strings to analyze (descriptions, reviews, etc.)
            
        Returns:
            Dictionary with audience segments, use cases, and frequencies
            
        Example:
            >>> analyzer = PositioningAnalyzer()
            >>> texts = ["Perfect for developers and designers"]
            >>> result = analyzer.extract_target_audience(texts)
            >>> "audiences" in result
            True
        """
        if not texts:
            logger.warning("Empty text list provided for target audience extraction")
            return {
                "audiences": {},
                "use_cases": {},
                "audience_use_case_mapping": {}
            }
        
        try:
            logger.info(f"Extracting target audience from {len(texts)} texts")
            
            audience_mentions = defaultdict(list)
            use_case_mentions = defaultdict(list)
            audience_use_case_pairs = defaultdict(lambda: defaultdict(int))
            
            for idx, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                
                text_lower = text.lower()
                
                # Extract audience indicators
                found_audiences = []
                for audience, keywords in self.AUDIENCE_INDICATORS.items():
                    for keyword in keywords:
                        # Use word boundary but allow for plurals with optional 's'
                        pattern = r'\b' + re.escape(keyword) + r's?\b'
                        if re.search(pattern, text_lower):
                            audience_mentions[audience].append({
                                "text_index": idx,
                                "keyword": keyword
                            })
                            found_audiences.append(audience)
                            logger.debug(
                                f"Found audience '{audience}' via keyword '{keyword}'"
                            )
                            break  # Only count once per audience per text
                
                # Extract use case indicators
                found_use_cases = []
                for use_case, keywords in self.USE_CASE_PATTERNS.items():
                    for keyword in keywords:
                        pattern = r'\b' + re.escape(keyword) + r'\b'
                        if re.search(pattern, text_lower):
                            use_case_mentions[use_case].append({
                                "text_index": idx,
                                "keyword": keyword
                            })
                            found_use_cases.append(use_case)
                            logger.debug(
                                f"Found use case '{use_case}' via keyword '{keyword}'"
                            )
                            break  # Only count once per use case per text
                
                # Map audiences to use cases when they co-occur
                for audience in found_audiences:
                    for use_case in found_use_cases:
                        audience_use_case_pairs[audience][use_case] += 1
            
            # Calculate audience frequencies and strengths
            audience_data = {}
            for audience, mentions in audience_mentions.items():
                audience_data[audience] = {
                    "frequency": len(mentions),
                    "keywords_found": list(set(m["keyword"] for m in mentions))
                }
            
            # Sort by frequency
            sorted_audiences = dict(
                sorted(
                    audience_data.items(),
                    key=lambda x: x[1]["frequency"],
                    reverse=True
                )
            )
            
            # Calculate use case frequencies
            use_case_data = {}
            for use_case, mentions in use_case_mentions.items():
                use_case_data[use_case] = {
                    "frequency": len(mentions),
                    "keywords_found": list(set(m["keyword"] for m in mentions))
                }
            
            # Sort by frequency
            sorted_use_cases = dict(
                sorted(
                    use_case_data.items(),
                    key=lambda x: x[1]["frequency"],
                    reverse=True
                )
            )
            
            # Convert audience-use case mapping to regular dict
            audience_use_case_mapping = {
                audience: dict(use_cases)
                for audience, use_cases in audience_use_case_pairs.items()
            }
            
            result = {
                "audiences": sorted_audiences,
                "use_cases": sorted_use_cases,
                "audience_use_case_mapping": audience_use_case_mapping,
                "total_audience_mentions": sum(
                    data["frequency"] for data in audience_data.values()
                ),
                "total_use_case_mentions": sum(
                    data["frequency"] for data in use_case_data.values()
                )
            }
            
            logger.info(
                f"Target audience extraction complete: "
                f"{len(sorted_audiences)} audiences, {len(sorted_use_cases)} use cases"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Target audience extraction failed: {e}", exc_info=True)
            return {
                "audiences": {},
                "use_cases": {},
                "audience_use_case_mapping": {},
                "error": str(e)
            }

    
    def analyze_competitive_landscape(
        self,
        competitors_data: List[Dict[str, Union[str, List, Dict]]]
    ) -> Dict[str, Union[List, Dict, int]]:
        """
        Generate competitive landscape summary with positioning clusters.
        
        Analyzes competitor data to identify positioning clusters, common themes,
        and competitive dynamics in the market.
        
        Args:
            competitors_data: List of competitor dictionaries with name, description,
                            reviews, and optional positioning/audience data
            
        Returns:
            Dictionary with positioning clusters, landscape summary, and insights
            
        Raises:
            Exception: If analysis fails
            
        Example:
            >>> analyzer = PositioningAnalyzer()
            >>> competitors = [
            ...     {"name": "Product A", "description": "Simple task manager"},
            ...     {"name": "Product B", "description": "Powerful project tool"}
            ... ]
            >>> result = analyzer.analyze_competitive_landscape(competitors)
            >>> "clusters" in result
            True
        """
        if not competitors_data:
            logger.warning("Empty competitors data provided for landscape analysis")
            return {
                "clusters": [],
                "landscape_summary": {},
                "competitive_dynamics": {}
            }
        
        try:
            logger.info(
                f"Analyzing competitive landscape for {len(competitors_data)} competitors"
            )
            
            # Extract positioning for each competitor
            competitor_positioning = []
            
            for competitor in competitors_data:
                name = competitor.get('name', 'Unknown')
                description = competitor.get('description', '')
                reviews = competitor.get('reviews', [])
                
                # Extract positioning keywords
                descriptions = [description] if description else []
                positioning = self.extract_positioning_keywords(descriptions, reviews)
                
                # Extract target audience
                texts_for_audience = descriptions.copy()
                if reviews:
                    texts_for_audience.extend([
                        r.get('text') or r.get('body', '') for r in reviews
                    ])
                audience = self.extract_target_audience(texts_for_audience)
                
                competitor_positioning.append({
                    "name": name,
                    "positioning_themes": positioning.get('themes', {}),
                    "audiences": audience.get('audiences', {}),
                    "use_cases": audience.get('use_cases', {})
                })
                
                logger.debug(
                    f"Analyzed positioning for '{name}': "
                    f"{len(positioning.get('themes', {}))} themes"
                )
            
            # Identify positioning clusters
            clusters = self._identify_positioning_clusters(competitor_positioning)
            
            # Generate landscape summary
            landscape_summary = self._generate_landscape_summary(
                competitor_positioning,
                clusters
            )
            
            # Analyze competitive dynamics
            competitive_dynamics = self._analyze_competitive_dynamics(
                competitor_positioning,
                clusters
            )
            
            result = {
                "clusters": clusters,
                "landscape_summary": landscape_summary,
                "competitive_dynamics": competitive_dynamics,
                "total_competitors": len(competitors_data),
                "competitor_positioning": competitor_positioning
            }
            
            logger.info(
                f"Competitive landscape analysis complete: "
                f"{len(clusters)} positioning clusters identified"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Competitive landscape analysis failed: {e}", exc_info=True)
            raise Exception(f"Competitive landscape analysis failed: {e}")
    
    def _identify_positioning_clusters(
        self,
        competitor_positioning: List[Dict]
    ) -> List[Dict[str, Union[str, List, int]]]:
        """
        Identify clusters of competitors with similar positioning.
        
        Args:
            competitor_positioning: List of competitor positioning data
            
        Returns:
            List of positioning clusters
        """
        # Group competitors by dominant positioning theme
        theme_groups = defaultdict(list)
        
        for comp in competitor_positioning:
            themes = comp.get('positioning_themes', {})
            
            if not themes:
                theme_groups['uncategorized'].append(comp['name'])
                continue
            
            # Get dominant theme (highest count)
            dominant_theme = max(
                themes.items(),
                key=lambda x: x[1].get('count', 0)
            )[0] if themes else 'uncategorized'
            
            theme_groups[dominant_theme].append(comp['name'])
        
        # Create clusters
        clusters = []
        for theme, competitors in theme_groups.items():
            if len(competitors) > 0:
                clusters.append({
                    "cluster_name": theme,
                    "competitors": competitors,
                    "size": len(competitors),
                    "description": f"Products positioning around {theme}"
                })
        
        # Sort by cluster size
        clusters.sort(key=lambda x: x['size'], reverse=True)
        
        return clusters
    
    def _generate_landscape_summary(
        self,
        competitor_positioning: List[Dict],
        clusters: List[Dict]
    ) -> Dict[str, Union[int, List, Dict]]:
        """
        Generate summary of competitive landscape.
        
        Args:
            competitor_positioning: List of competitor positioning data
            clusters: List of positioning clusters
            
        Returns:
            Dictionary with landscape summary statistics
        """
        # Count total themes and audiences
        all_themes = Counter()
        all_audiences = Counter()
        all_use_cases = Counter()
        
        for comp in competitor_positioning:
            for theme in comp.get('positioning_themes', {}).keys():
                all_themes[theme] += 1
            
            for audience in comp.get('audiences', {}).keys():
                all_audiences[audience] += 1
            
            for use_case in comp.get('use_cases', {}).keys():
                all_use_cases[use_case] += 1
        
        return {
            "total_clusters": len(clusters),
            "largest_cluster": clusters[0] if clusters else None,
            "most_common_themes": dict(all_themes.most_common(5)),
            "most_common_audiences": dict(all_audiences.most_common(5)),
            "most_common_use_cases": dict(all_use_cases.most_common(5)),
            "theme_diversity": len(all_themes),
            "audience_diversity": len(all_audiences)
        }
    
    def _analyze_competitive_dynamics(
        self,
        competitor_positioning: List[Dict],
        clusters: List[Dict]
    ) -> Dict[str, Union[str, List, float]]:
        """
        Analyze competitive dynamics and market concentration.
        
        Args:
            competitor_positioning: List of competitor positioning data
            clusters: List of positioning clusters
            
        Returns:
            Dictionary with competitive dynamics insights
        """
        total_competitors = len(competitor_positioning)
        
        if total_competitors == 0:
            return {
                "market_concentration": "unknown",
                "concentration_score": 0.0,
                "competitive_intensity": "low",
                "insights": []
            }
        
        # Calculate market concentration (Herfindahl index approximation)
        cluster_sizes = [c['size'] for c in clusters]
        concentration_score = sum(
            (size / total_competitors) ** 2 for size in cluster_sizes
        )
        
        # Determine market concentration level
        if concentration_score > 0.5:
            market_concentration = "high"
            competitive_intensity = "high"
        elif concentration_score > 0.25:
            market_concentration = "moderate"
            competitive_intensity = "moderate"
        else:
            market_concentration = "low"
            competitive_intensity = "low"
        
        # Generate insights
        insights = []
        
        if clusters:
            largest_cluster = clusters[0]
            if largest_cluster['size'] > total_competitors * 0.5:
                insights.append(
                    f"Market is dominated by {largest_cluster['cluster_name']} "
                    f"positioning ({largest_cluster['size']} competitors)"
                )
        
        if len(clusters) > 5:
            insights.append(
                f"Highly fragmented market with {len(clusters)} distinct "
                f"positioning strategies"
            )
        
        return {
            "market_concentration": market_concentration,
            "concentration_score": float(concentration_score),
            "competitive_intensity": competitive_intensity,
            "insights": insights
        }

    
    def identify_market_saturation(
        self,
        positioning_data: Dict[str, Union[List, Dict]]
    ) -> Dict[str, Union[List, Dict, str]]:
        """
        Detect areas where multiple products share similar positioning.
        
        Identifies saturated market segments based on positioning cluster analysis
        and provides saturation metrics.
        
        Args:
            positioning_data: Dictionary with competitive landscape data including
                            clusters and competitor positioning
            
        Returns:
            Dictionary with saturated segments, saturation metrics, and recommendations
            
        Example:
            >>> analyzer = PositioningAnalyzer()
            >>> positioning_data = {
            ...     "clusters": [
            ...         {"cluster_name": "simplicity", "size": 10},
            ...         {"cluster_name": "power", "size": 2}
            ...     ]
            ... }
            >>> result = analyzer.identify_market_saturation(positioning_data)
            >>> "saturated_segments" in result
            True
        """
        if not positioning_data:
            logger.warning("Empty positioning data provided for saturation analysis")
            return {
                "saturated_segments": [],
                "undersaturated_segments": [],
                "saturation_metrics": {}
            }
        
        try:
            logger.info("Identifying market saturation from positioning data")
            
            clusters = positioning_data.get('clusters', [])
            total_competitors = positioning_data.get('total_competitors', 0)
            
            if not clusters or total_competitors == 0:
                logger.warning("Insufficient data for saturation analysis")
                return {
                    "saturated_segments": [],
                    "undersaturated_segments": [],
                    "saturation_metrics": {},
                    "message": "Insufficient data for analysis"
                }
            
            # Calculate saturation thresholds
            avg_cluster_size = total_competitors / len(clusters)
            saturation_threshold = avg_cluster_size * 1.5  # 50% above average
            undersaturation_threshold = avg_cluster_size * 0.5  # 50% below average
            
            saturated_segments = []
            undersaturated_segments = []
            balanced_segments = []
            
            for cluster in clusters:
                cluster_size = cluster.get('size', 0)
                cluster_name = cluster.get('cluster_name', 'unknown')
                saturation_ratio = cluster_size / total_competitors
                
                segment_data = {
                    "segment": cluster_name,
                    "competitor_count": cluster_size,
                    "saturation_ratio": float(saturation_ratio),
                    "competitors": cluster.get('competitors', [])
                }
                
                if cluster_size >= saturation_threshold:
                    segment_data['saturation_level'] = 'high'
                    saturated_segments.append(segment_data)
                    logger.debug(
                        f"Saturated segment identified: {cluster_name} "
                        f"({cluster_size} competitors)"
                    )
                elif cluster_size <= undersaturation_threshold:
                    segment_data['saturation_level'] = 'low'
                    undersaturated_segments.append(segment_data)
                    logger.debug(
                        f"Undersaturated segment identified: {cluster_name} "
                        f"({cluster_size} competitors)"
                    )
                else:
                    segment_data['saturation_level'] = 'balanced'
                    balanced_segments.append(segment_data)
            
            # Sort by saturation ratio
            saturated_segments.sort(key=lambda x: x['saturation_ratio'], reverse=True)
            undersaturated_segments.sort(key=lambda x: x['saturation_ratio'])
            
            # Calculate overall saturation metrics
            saturation_metrics = {
                "total_segments": len(clusters),
                "saturated_count": len(saturated_segments),
                "undersaturated_count": len(undersaturated_segments),
                "balanced_count": len(balanced_segments),
                "average_cluster_size": float(avg_cluster_size),
                "saturation_threshold": float(saturation_threshold),
                "market_fragmentation": len(clusters) / total_competitors if total_competitors > 0 else 0
            }
            
            # Generate insights
            insights = []
            
            if saturated_segments:
                top_saturated = saturated_segments[0]
                insights.append(
                    f"'{top_saturated['segment']}' positioning is highly saturated "
                    f"with {top_saturated['competitor_count']} competitors "
                    f"({top_saturated['saturation_ratio']:.1%} of market)"
                )
            
            if undersaturated_segments:
                top_undersaturated = undersaturated_segments[0]
                insights.append(
                    f"'{top_undersaturated['segment']}' positioning is undersaturated "
                    f"with only {top_undersaturated['competitor_count']} competitors "
                    f"- potential opportunity"
                )
            
            if saturation_metrics['market_fragmentation'] > 0.3:
                insights.append(
                    "Market is highly fragmented with many small positioning niches"
                )
            
            result = {
                "saturated_segments": saturated_segments,
                "undersaturated_segments": undersaturated_segments,
                "balanced_segments": balanced_segments,
                "saturation_metrics": saturation_metrics,
                "insights": insights
            }
            
            logger.info(
                f"Market saturation analysis complete: "
                f"{len(saturated_segments)} saturated, "
                f"{len(undersaturated_segments)} undersaturated segments"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Market saturation identification failed: {e}", exc_info=True)
            return {
                "saturated_segments": [],
                "undersaturated_segments": [],
                "saturation_metrics": {},
                "error": str(e)
            }

    
    def generate_differentiation_recommendations(
        self,
        positioning_data: Dict[str, Union[List, Dict]],
        product_idea: str
    ) -> Dict[str, Union[List, str]]:
        """
        Suggest differentiation opportunities based on market analysis.
        
        Analyzes positioning data and product idea to generate specific
        recommendations for differentiation in the market.
        
        Args:
            positioning_data: Dictionary with competitive landscape and saturation data
            product_idea: Description of the product idea to position
            
        Returns:
            Dictionary with differentiation recommendations and strategies
            
        Example:
            >>> analyzer = PositioningAnalyzer()
            >>> positioning_data = {
            ...     "saturated_segments": [{"segment": "simplicity", "competitor_count": 10}],
            ...     "undersaturated_segments": [{"segment": "automation", "competitor_count": 2}]
            ... }
            >>> result = analyzer.generate_differentiation_recommendations(
            ...     positioning_data,
            ...     "A task management app"
            ... )
            >>> "recommendations" in result
            True
        """
        if not positioning_data or not product_idea:
            logger.warning(
                "Insufficient data provided for differentiation recommendations"
            )
            return {
                "recommendations": [],
                "strategies": [],
                "summary": "Insufficient data for recommendations"
            }
        
        try:
            logger.info(
                f"Generating differentiation recommendations for product idea: "
                f"{product_idea[:100]}..."
            )
            
            recommendations = []
            strategies = []
            
            # Analyze product idea for existing positioning signals
            product_idea_lower = product_idea.lower()
            product_positioning = self.extract_positioning_keywords([product_idea])
            product_audience = self.extract_target_audience([product_idea])
            
            # Get saturation data
            saturated_segments = positioning_data.get('saturated_segments', [])
            undersaturated_segments = positioning_data.get('undersaturated_segments', [])
            landscape_summary = positioning_data.get('landscape_summary', {})
            
            # Recommendation 1: Avoid saturated segments
            if saturated_segments:
                saturated_names = [s['segment'] for s in saturated_segments[:3]]
                recommendations.append({
                    "type": "avoid_saturation",
                    "priority": "high",
                    "title": "Avoid Saturated Positioning",
                    "description": (
                        f"The following positioning themes are highly saturated: "
                        f"{', '.join(saturated_names)}. Consider alternative positioning "
                        f"to stand out in the market."
                    ),
                    "saturated_segments": saturated_names
                })
                
                strategies.append(
                    f"Differentiate from the {saturated_segments[0]['competitor_count']} "
                    f"competitors in the '{saturated_segments[0]['segment']}' space"
                )
            
            # Recommendation 2: Explore undersaturated segments
            if undersaturated_segments:
                undersaturated_names = [s['segment'] for s in undersaturated_segments[:3]]
                recommendations.append({
                    "type": "explore_opportunity",
                    "priority": "high",
                    "title": "Explore Undersaturated Opportunities",
                    "description": (
                        f"These positioning themes are undersaturated: "
                        f"{', '.join(undersaturated_names)}. These represent potential "
                        f"opportunities for differentiation."
                    ),
                    "opportunity_segments": undersaturated_names
                })
                
                strategies.append(
                    f"Consider positioning around '{undersaturated_segments[0]['segment']}' "
                    f"which has only {undersaturated_segments[0]['competitor_count']} competitors"
                )
            
            # Recommendation 3: Combine positioning themes
            if len(product_positioning.get('themes', {})) > 1:
                themes = list(product_positioning['themes'].keys())[:2]
                recommendations.append({
                    "type": "hybrid_positioning",
                    "priority": "medium",
                    "title": "Hybrid Positioning Strategy",
                    "description": (
                        f"Your product idea suggests multiple positioning themes: "
                        f"{', '.join(themes)}. Consider a hybrid positioning that "
                        f"combines these strengths for unique differentiation."
                    ),
                    "themes": themes
                })
                
                strategies.append(
                    f"Combine {' and '.join(themes)} positioning for unique value proposition"
                )
            
            # Recommendation 4: Target underserved audiences
            common_audiences = landscape_summary.get('most_common_audiences', {})
            product_audiences = product_audience.get('audiences', {})
            
            if product_audiences:
                # Find audiences in product idea that are not common in market
                underserved = []
                for audience in product_audiences.keys():
                    if audience not in common_audiences or common_audiences[audience] < 3:
                        underserved.append(audience)
                
                if underserved:
                    recommendations.append({
                        "type": "target_audience",
                        "priority": "medium",
                        "title": "Target Underserved Audiences",
                        "description": (
                            f"Focus on these underserved audience segments: "
                            f"{', '.join(underserved)}. Few competitors are explicitly "
                            f"targeting these groups."
                        ),
                        "audiences": underserved
                    })
                    
                    strategies.append(
                        f"Explicitly target {underserved[0]} segment which is underserved"
                    )
            
            # Recommendation 5: Niche positioning
            if positioning_data.get('saturation_metrics', {}).get('market_fragmentation', 0) < 0.2:
                recommendations.append({
                    "type": "niche_positioning",
                    "priority": "medium",
                    "title": "Consider Niche Positioning",
                    "description": (
                        "The market shows low fragmentation, suggesting opportunities "
                        "for niche positioning. Focus on a specific use case or "
                        "audience segment to carve out a unique position."
                    )
                })
                
                strategies.append(
                    "Pursue niche positioning to avoid direct competition with established players"
                )
            
            # Recommendation 6: Feature-based differentiation
            if saturated_segments and undersaturated_segments:
                recommendations.append({
                    "type": "feature_differentiation",
                    "priority": "high",
                    "title": "Feature-Based Differentiation",
                    "description": (
                        "Combine features from saturated segments (proven demand) "
                        "with positioning from undersaturated segments (less competition) "
                        "to create a unique offering."
                    )
                })
                
                strategies.append(
                    "Blend proven features with unique positioning for competitive advantage"
                )
            
            # Generate summary
            summary = self._generate_differentiation_summary(
                recommendations,
                strategies,
                positioning_data
            )
            
            result = {
                "recommendations": recommendations,
                "strategies": strategies,
                "summary": summary,
                "total_recommendations": len(recommendations)
            }
            
            logger.info(
                f"Differentiation recommendations generated: "
                f"{len(recommendations)} recommendations, {len(strategies)} strategies"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Differentiation recommendation generation failed: {e}",
                exc_info=True
            )
            return {
                "recommendations": [],
                "strategies": [],
                "summary": f"Error generating recommendations: {e}",
                "error": str(e)
            }
    
    def _generate_differentiation_summary(
        self,
        recommendations: List[Dict],
        strategies: List[str],
        positioning_data: Dict
    ) -> str:
        """
        Generate a summary of differentiation recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            strategies: List of strategy strings
            positioning_data: Positioning analysis data
            
        Returns:
            Summary string
        """
        high_priority = sum(1 for r in recommendations if r.get('priority') == 'high')
        
        summary_parts = [
            f"Generated {len(recommendations)} differentiation recommendations "
            f"({high_priority} high priority)."
        ]
        
        saturated_count = len(positioning_data.get('saturated_segments', []))
        undersaturated_count = len(positioning_data.get('undersaturated_segments', []))
        
        if saturated_count > 0:
            summary_parts.append(
                f"Identified {saturated_count} saturated market segments to avoid."
            )
        
        if undersaturated_count > 0:
            summary_parts.append(
                f"Found {undersaturated_count} undersaturated opportunities to explore."
            )
        
        if strategies:
            summary_parts.append(
                f"Key strategy: {strategies[0]}"
            )
        
        return " ".join(summary_parts)
    
    def __repr__(self) -> str:
        """String representation of PositioningAnalyzer."""
        return (
            f"PositioningAnalyzer("
            f"positioning_themes={len(self.POSITIONING_KEYWORDS)}, "
            f"audience_segments={len(self.AUDIENCE_INDICATORS)}, "
            f"use_cases={len(self.USE_CASE_PATTERNS)})"
        )
