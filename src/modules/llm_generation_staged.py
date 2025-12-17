"""
Multi-stage LLM generation module for Product Hunt RAG Analyzer.

This module provides staged generation to avoid token limits by breaking
down the analysis into smaller, manageable chunks.
"""

import json
import time
from typing import Dict, Any, Optional, List
from src.modules.llm_generation import LLMGenerator, LLMGenerationError, LLMResponseValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StagedLLMGenerator(LLMGenerator):
    """
    Extended LLMGenerator that supports multi-stage generation to avoid token limits.
    
    Breaks down large analysis tasks into smaller stages:
    1. Market Positioning Analysis
    2. Feature Gap Analysis
    3. Sentiment Analysis
    4. Recommendations Generation
    5. Final Synthesis
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with parent class configuration."""
        super().__init__(*args, **kwargs)
        logger.info("StagedLLMGenerator initialized for multi-stage generation")
    
    def _generate_stage(
        self,
        stage_name: str,
        prompt: str,
        expected_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a single stage of analysis.
        
        Args:
            stage_name: Name of the stage for logging
            prompt: Prompt for this stage
            expected_keys: Optional list of expected keys in response
            
        Returns:
            Parsed JSON response for this stage
        """
        try:
            logger.info(f"Starting stage: {stage_name}")
            start_time = time.time()
            
            # Generate with parent class method
            generated_text = self._generate(prompt)
            
            # Parse JSON
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError(f"No JSON object found in {stage_name} response")
            
            json_text = generated_text[json_start:json_end]
            parsed_response = json.loads(json_text)
            
            # Validate expected keys if provided
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in parsed_response]
                if missing_keys:
                    logger.warning(f"{stage_name} missing keys: {missing_keys}")
            
            elapsed = time.time() - start_time
            logger.info(f"Stage '{stage_name}' completed in {elapsed:.2f}s")
            
            return parsed_response
            
        except Exception as e:
            error_msg = f"Stage '{stage_name}' failed: {e}"
            logger.error(error_msg)
            raise LLMGenerationError(error_msg)

    def _build_market_positioning_prompt(
        self,
        product_idea: str,
        competitors_context: str
    ) -> str:
        """Build prompt for market positioning stage."""
        return f"""You are a product analyst specializing in competitive intelligence.

Product Idea: "{product_idea}"

Competitor Data:
{competitors_context}

Analyze ONLY the market positioning aspects. Provide a focused analysis of:
1. How competitors are currently positioned
2. Their positioning strategies
3. Differentiation opportunities for this product idea
4. Market saturation assessment

Format your response as valid JSON:
{{
  "market_positioning": {{
    "summary": "Comprehensive overview of market positioning landscape",
    "differentiation_opportunities": ["opportunity 1", "opportunity 2"],
    "competitor_positions": [
      {{"name": "Competitor Name", "position": "Their positioning strategy"}}
    ],
    "market_saturation": "Assessment of market saturation and gaps"
  }}
}}"""

    def _build_feature_gaps_prompt(
        self,
        product_idea: str,
        competitors_context: str
    ) -> str:
        """Build prompt for feature gap analysis stage."""
        return f"""You are a product analyst specializing in feature analysis.

Product Idea: "{product_idea}"

Competitor Data:
{competitors_context}

Analyze ONLY the feature gaps. Focus on:
1. Features users love in competitor products
2. Features users are requesting but competitors lack
3. Must-have features based on competitor analysis
4. Unique features that could differentiate this product

Format your response as valid JSON:
{{
  "feature_gaps": {{
    "high_priority": [
      {{"name": "Feature name", "description": "Why it's important"}}
    ],
    "medium_priority": ["feature 1", "feature 2"],
    "low_priority": ["feature 1", "feature 2"],
    "loved_features": ["feature users love in competitors"],
    "requested_features": ["features users are requesting"]
  }}
}}"""

    def _build_sentiment_prompt(
        self,
        product_idea: str,
        competitors_context: str
    ) -> str:
        """Build prompt for sentiment analysis stage."""
        return f"""You are a product analyst specializing in user sentiment analysis.

Product Idea: "{product_idea}"

Competitor Data:
{competitors_context}

Analyze ONLY the sentiment aspects. Focus on:
1. What users love most about competitor products
2. Common pain points and complaints
3. Aspects with negative sentiment
4. Opportunities to address unmet needs

Format your response as valid JSON:
{{
  "sentiment_summary": {{
    "overall_trends": {{
      "positive_aspects": "What users love",
      "negative_aspects": "What users complain about"
    }},
    "pain_points": ["pain point 1", "pain point 2"],
    "loved_features": ["loved feature 1", "loved feature 2"],
    "opportunities": ["opportunity to address unmet need 1"]
  }}
}}"""

    def _build_recommendations_prompt(
        self,
        product_idea: str,
        market_positioning: Dict[str, Any],
        feature_gaps: Dict[str, Any],
        sentiment_summary: Dict[str, Any]
    ) -> str:
        """Build prompt for recommendations stage."""
        return f"""You are a product analyst providing actionable recommendations.

Product Idea: "{product_idea}"

Based on the following analysis:

Market Positioning:
{json.dumps(market_positioning, indent=2)}

Feature Gaps:
{json.dumps(feature_gaps, indent=2)}

Sentiment Summary:
{json.dumps(sentiment_summary, indent=2)}

Provide 5-7 specific, prioritized recommendations for the product idea.
Each recommendation should be actionable and guide product development.

Format your response as valid JSON:
{{
  "recommendations": [
    {{
      "title": "Recommendation title",
      "description": "Detailed description",
      "priority": "high|medium|low",
      "rationale": "Why this recommendation matters"
    }}
  ],
  "confidence_score": 0.85
}}"""

    def generate_competitive_intelligence_staged(
        self,
        product_idea: str,
        competitors_context: str,
        stages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate competitive intelligence using multi-stage approach.
        
        Breaks down analysis into smaller stages to avoid token limits:
        1. Market Positioning
        2. Feature Gaps
        3. Sentiment Analysis
        4. Recommendations
        
        Args:
            product_idea: User's product idea description
            competitors_context: Formatted competitor data and reviews
            stages: Optional list of stages to run (default: all stages)
            
        Returns:
            Complete analysis dictionary with all sections
            
        Raises:
            LLMGenerationError: If any stage fails
            
        Example:
            >>> generator = StagedLLMGenerator()
            >>> result = generator.generate_competitive_intelligence_staged(
            ...     "A task management app",
            ...     "Competitor data..."
            ... )
        """
        try:
            logger.info("Starting multi-stage competitive intelligence generation")
            start_time = time.time()
            
            # Default to all stages
            if stages is None:
                stages = ["market_positioning", "feature_gaps", "sentiment", "recommendations"]
            
            result = {}
            
            # Stage 1: Market Positioning
            if "market_positioning" in stages:
                logger.info("=" * 70)
                logger.info("STAGE 1/4: Market Positioning Analysis")
                logger.info("=" * 70)
                
                prompt = self._build_market_positioning_prompt(product_idea, competitors_context)
                stage_result = self._generate_stage(
                    "Market Positioning",
                    prompt,
                    expected_keys=["market_positioning"]
                )
                result.update(stage_result)
                
                # Small delay between stages to respect rate limits
                time.sleep(1)
            
            # Stage 2: Feature Gaps
            if "feature_gaps" in stages:
                logger.info("=" * 70)
                logger.info("STAGE 2/4: Feature Gap Analysis")
                logger.info("=" * 70)
                
                prompt = self._build_feature_gaps_prompt(product_idea, competitors_context)
                stage_result = self._generate_stage(
                    "Feature Gaps",
                    prompt,
                    expected_keys=["feature_gaps"]
                )
                result.update(stage_result)
                
                time.sleep(1)
            
            # Stage 3: Sentiment Analysis
            if "sentiment" in stages:
                logger.info("=" * 70)
                logger.info("STAGE 3/4: Sentiment Analysis")
                logger.info("=" * 70)
                
                prompt = self._build_sentiment_prompt(product_idea, competitors_context)
                stage_result = self._generate_stage(
                    "Sentiment Analysis",
                    prompt,
                    expected_keys=["sentiment_summary"]
                )
                result.update(stage_result)
                
                time.sleep(1)
            
            # Stage 4: Recommendations (synthesizes previous stages)
            if "recommendations" in stages:
                logger.info("=" * 70)
                logger.info("STAGE 4/4: Recommendations Generation")
                logger.info("=" * 70)
                
                prompt = self._build_recommendations_prompt(
                    product_idea,
                    result.get("market_positioning", {}),
                    result.get("feature_gaps", {}),
                    result.get("sentiment_summary", {})
                )
                stage_result = self._generate_stage(
                    "Recommendations",
                    prompt,
                    expected_keys=["recommendations", "confidence_score"]
                )
                result.update(stage_result)
            
            # Validate final result
            self.validate_response(result)
            
            total_time = time.time() - start_time
            logger.info("=" * 70)
            logger.info(f"✓ Multi-stage generation completed in {total_time:.2f}s")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            error_msg = f"Multi-stage generation failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise LLMGenerationError(error_msg)
    
    def generate_competitive_intelligence(
        self,
        product_idea: str,
        competitors_context: str,
        use_staged: bool = True
    ) -> Dict[str, Any]:
        """
        Generate competitive intelligence with optional staged approach.
        
        Args:
            product_idea: User's product idea description
            competitors_context: Formatted competitor data and reviews
            use_staged: If True, use multi-stage generation (default: True)
            
        Returns:
            Complete analysis dictionary
        """
        if use_staged:
            return self.generate_competitive_intelligence_staged(
                product_idea,
                competitors_context
            )
        else:
            # Fall back to parent class single-stage generation
            return super().generate_competitive_intelligence(
                product_idea,
                competitors_context
            )
