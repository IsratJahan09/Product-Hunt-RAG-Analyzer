"""
Sentiment analysis module for Product Hunt RAG Analyzer.

This module provides sentiment classification and aspect extraction using
transformer models with support for batch processing and confidence scoring.
"""

import time
import re
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Manages sentiment analysis using transformer models.
    
    Provides methods to classify sentiment (positive, negative, neutral),
    extract aspects from text, and perform batch analysis with confidence scoring.
    """
    
    # Common product feature keywords for aspect extraction
    ASPECT_KEYWORDS = {
        "ui": ["ui", "interface", "design", "layout", "theme", "appearance", "visual"],
        "ux": ["ux", "experience", "usability", "intuitive", "user-friendly", "navigation"],
        "performance": ["performance", "speed", "fast", "slow", "lag", "responsive", "loading"],
        "pricing": ["price", "pricing", "cost", "expensive", "cheap", "affordable", "subscription"],
        "features": ["feature", "functionality", "capability", "function", "tool"],
        "integration": ["integration", "integrate", "api", "connect", "sync", "compatibility"],
        "support": ["support", "help", "customer service", "documentation", "docs"],
        "reliability": ["reliable", "stability", "stable", "crash", "bug", "error"],
        "mobile": ["mobile", "app", "ios", "android", "phone", "tablet"],
        "security": ["security", "secure", "privacy", "safe", "encryption"]
    }
    
    # Keywords that indicate promotional/announcement content
    PROMOTIONAL_KEYWORDS = [
        "we're excited to launch",
        "we're super excited",
        "excited to announce",
        "thrilled to announce",
        "proud to announce",
        "launching today",
        "just launched",
        "we built",
        "we created",
        "we're launching",
        "check us out",
        "try the beta",
        "try it out",
        "we'd love",
        "share your thoughts",
        "feedback welcome",
        "early beta",
        "beta launch",
        "product launch",
        "introducing",
        "meet",
        "say hello to"
    ]
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[str] = None
    ):
        """
        Initialize SentimentAnalyzer with model name and device selection.
        
        Args:
            model_name: Name of the Hugging Face sentiment model
                       (default: cardiffnlp/twitter-roberta-base-sentiment-latest)
            device: Device to use ("cpu", "cuda", or None for auto-detection)
        
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> analyzer = SentimentAnalyzer(device="cuda")
        """
        self.model_name = model_name
        self.device = self._select_device(device)
        self.model = None
        self.tokenizer = None
        self.label_mapping = None
        
        logger.info(
            f"SentimentAnalyzer initialized with model={model_name}, device={self.device}"
        )
        
        # Load model during initialization
        self.load_model(model_name)
    
    def _select_device(self, device: Optional[str] = None) -> str:
        """
        Select device for model execution with GPU detection.
        
        Args:
            device: Requested device ("cpu", "cuda", or None for auto-detection)
            
        Returns:
            Selected device string ("cpu" or "cuda")
        """
        if device is not None:
            # User specified device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available, falling back to CPU"
                )
                return "cpu"
            logger.info(f"Using user-specified device: {device}")
            return device
        
        # Auto-detect device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(
                f"CUDA detected and available, using GPU: {torch.cuda.get_device_name(0)}"
            )
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
        
        return device
    
    def load_model(self, model_name: str) -> None:
        """
        Load sentiment model from Hugging Face.
        
        Args:
            model_name: Name of the model to load
            
        Raises:
            Exception: If model loading fails
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> analyzer.load_model("cardiffnlp/twitter-roberta-base-sentiment-latest")
        """
        try:
            start_time = time.time()
            logger.info(f"Loading sentiment model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Determine label mapping based on model config
            config = self.model.config
            if hasattr(config, 'id2label'):
                self.label_mapping = config.id2label
            else:
                # Default mapping for 3-class sentiment
                self.label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
            
            load_time = (time.time() - start_time) * 1000
            logger.info(
                f"Model loaded successfully in {load_time:.2f}ms "
                f"(labels: {list(self.label_mapping.values())})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
            raise Exception(f"Model loading failed: {e}")
    
    def analyze_sentiment(
        self,
        text: str,
        extract_aspects: bool = False
    ) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        Classify sentiment (positive, negative, neutral) with confidence scores.
        
        Args:
            text: Text to analyze
            extract_aspects: Whether to extract aspects (default: False)
            
        Returns:
            Dictionary with sentiment, confidence, and optionally aspects
            
        Raises:
            Exception: If sentiment analysis fails
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.analyze_sentiment("This product is amazing!")
            >>> result
            {'sentiment': 'positive', 'confidence': 0.95, 'aspects': []}
        """
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        if not text or not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "aspects": []
            }
        
        try:
            start_time = time.time()
            logger.debug(f"Analyzing sentiment for text: {text[:100]}...")
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, dim=-1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            
            # Map to sentiment label
            sentiment = self.label_mapping.get(predicted_class, "neutral")
            
            analysis_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Sentiment analysis complete: {sentiment} "
                f"(confidence: {confidence:.3f}) in {analysis_time:.2f}ms"
            )
            
            # Log low confidence predictions
            if confidence < 0.6:
                logger.warning(
                    f"Low confidence sentiment prediction: {confidence:.3f} "
                    f"for text: {text[:100]}..."
                )
            
            result = {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "aspects": []
            }
            
            # Extract aspects if requested
            if extract_aspects:
                result["aspects"] = self.extract_aspects(text)
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"CUDA out of memory error: {e}. Try using CPU.",
                exc_info=True
            )
            raise Exception(f"CUDA out of memory: {e}")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}", exc_info=True)
            raise Exception(f"Sentiment analysis failed: {e}")
    
    def is_promotional_content(self, text: str) -> bool:
        """
        Detect if text is promotional/announcement content rather than user feedback.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be promotional content, False otherwise
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> analyzer.is_promotional_content("We're excited to launch our new app!")
            True
            >>> analyzer.is_promotional_content("This app is terrible, crashes constantly")
            False
        """
        if not text or not text.strip():
            return False
        
        text_lower = text.lower()
        
        # Check for promotional keywords
        for keyword in self.PROMOTIONAL_KEYWORDS:
            if keyword in text_lower:
                logger.debug(f"Detected promotional content with keyword: '{keyword}'")
                return True
        
        # Check for common promotional patterns
        # Product announcements often start with "Hi everyone" or similar
        if text_lower.startswith(("hi everyone", "hey everyone", "hello everyone", "greetings")):
            logger.debug("Detected promotional content: greeting pattern")
            return True
        
        # Check for multiple exclamation marks (common in promotional content)
        if text.count("!") >= 3:
            logger.debug("Detected promotional content: multiple exclamation marks")
            return True
        
        return False
    
    def extract_aspects(self, text: str) -> List[Dict[str, Union[str, float]]]:
        """
        Identify feature mentions using keyword extraction and their sentiments.
        
        Args:
            text: Text to extract aspects from
            
        Returns:
            List of dictionaries with aspect, sentiment, and confidence
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> aspects = analyzer.extract_aspects("The UI is great but pricing is too high")
            >>> aspects
            [{'aspect': 'ui', 'sentiment': 'positive', 'confidence': 0.85},
             {'aspect': 'pricing', 'sentiment': 'negative', 'confidence': 0.78}]
        """
        if not text or not text.strip():
            return []
        
        text_lower = text.lower()
        aspects_found = []
        
        # Find all aspect mentions in text
        for aspect_category, keywords in self.ASPECT_KEYWORDS.items():
            for keyword in keywords:
                # Use word boundaries to match whole words
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    # Extract context around the keyword (±50 chars)
                    match = re.search(pattern, text_lower)
                    if match:
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end]
                        
                        # Analyze sentiment of the context
                        try:
                            sentiment_result = self.analyze_sentiment(context, extract_aspects=False)
                            
                            aspects_found.append({
                                "aspect": aspect_category,
                                "sentiment": sentiment_result["sentiment"],
                                "confidence": sentiment_result["confidence"]
                            })
                            
                            logger.debug(
                                f"Extracted aspect: {aspect_category} "
                                f"(sentiment: {sentiment_result['sentiment']}, "
                                f"confidence: {sentiment_result['confidence']:.3f})"
                            )
                            
                            # Only take first match per category
                            break
                        except Exception as e:
                            logger.warning(
                                f"Failed to analyze sentiment for aspect '{aspect_category}': {e}"
                            )
                            continue
        
        return aspects_found
    
    def batch_analyze(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True,
        extract_aspects: bool = False
    ) -> List[Dict[str, Union[str, float, List[Dict]]]]:
        """
        Efficient batch sentiment analysis with progress tracking.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process per batch (default: 16)
            show_progress: Whether to show progress (default: True)
            extract_aspects: Whether to extract aspects (default: False)
            
        Returns:
            List of sentiment analysis results
            
        Raises:
            Exception: If batch analysis fails
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> texts = ["Great product!", "Terrible experience", "It's okay"]
            >>> results = analyzer.batch_analyze(texts)
            >>> len(results)
            3
        """
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        if not texts:
            logger.warning("Empty text list provided, returning empty results")
            return []
        
        try:
            start_time = time.time()
            logger.info(
                f"Starting batch sentiment analysis for {len(texts)} texts "
                f"with batch_size={batch_size}"
            )
            
            results = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            promotional_count = 0
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                if show_progress:
                    logger.info(
                        f"Processing batch {batch_num}/{total_batches} "
                        f"({len(batch_texts)} texts)"
                    )
                
                # Process each text in batch
                for text in batch_texts:
                    try:
                        # Check if text is promotional content
                        is_promo = self.is_promotional_content(text)
                        if is_promo:
                            promotional_count += 1
                        
                        result = self.analyze_sentiment(text, extract_aspects=extract_aspects)
                        result["is_promotional"] = is_promo
                        results.append(result)
                    except Exception as e:
                        logger.warning(
                            f"Failed to analyze text: {text[:50]}... Error: {e}"
                        )
                        # Add default result for failed analysis
                        results.append({
                            "sentiment": "neutral",
                            "confidence": 0.0,
                            "aspects": [],
                            "is_promotional": False
                        })
            
            analysis_time = (time.time() - start_time) * 1000
            logger.info(
                f"Batch analysis complete: {len(results)} results in "
                f"{analysis_time:.2f}ms ({analysis_time/len(texts):.2f}ms per text). "
                f"Promotional content detected: {promotional_count}/{len(texts)}"
            )
            
            # Log memory usage if on CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                logger.info(
                    f"GPU memory: {memory_allocated:.2f}MB allocated, "
                    f"{memory_reserved:.2f}MB reserved"
                )
            
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"CUDA out of memory during batch processing: {e}. "
                f"Try reducing batch_size from {batch_size}.",
                exc_info=True
            )
            raise Exception(f"CUDA out of memory: {e}")
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}", exc_info=True)
            raise Exception(f"Batch analysis failed: {e}")
    
    def get_confidence_score(self, prediction: torch.Tensor) -> float:
        """
        Extract and return confidence values from model output.
        
        Args:
            prediction: Model output tensor (logits or probabilities)
            
        Returns:
            Confidence score as float
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> logits = torch.tensor([[0.1, 0.2, 2.5]])
            >>> confidence = analyzer.get_confidence_score(logits)
            >>> confidence > 0.5
            True
        """
        try:
            # Apply softmax if not already probabilities
            if prediction.max() > 1.0 or prediction.min() < 0.0:
                probabilities = torch.nn.functional.softmax(prediction, dim=-1)
            else:
                probabilities = prediction
            
            # Get maximum probability as confidence
            confidence = torch.max(probabilities).item()
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Failed to extract confidence score: {e}")
            return 0.0
    
    def __repr__(self) -> str:
        """String representation of SentimentAnalyzer."""
        return (
            f"SentimentAnalyzer(model='{self.model_name}', "
            f"device='{self.device}', labels={list(self.label_mapping.values()) if self.label_mapping else None})"
        )
