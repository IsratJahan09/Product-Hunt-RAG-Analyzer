"""
Text preprocessing module for Product Hunt RAG Analyzer.

This module provides text cleaning and normalization functionality including
HTML removal, special character handling, whitespace normalization, and
URL/email anonymization.
"""

import re
import time
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics_collector = get_metrics_collector()


class TextPreprocessor:
    """
    Text preprocessing orchestrator for cleaning and normalizing text data.
    
    Provides methods for HTML removal, special character handling, whitespace
    normalization, case conversion, and URL/email anonymization. Supports both
    single text and batch processing with parallel execution.
    """
    
    # Regex patterns compiled for performance
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9\s.,!?;:\'\"\-\[\]]')
    
    # Maximum text length before truncation
    MAX_TEXT_LENGTH = 512
    
    def __init__(self, max_length: int = 512, preserve_punctuation: bool = True):
        """
        Initialize TextPreprocessor.
        
        Args:
            max_length: Maximum text length before truncation (default: 512)
            preserve_punctuation: Whether to preserve punctuation marks (default: True)
        """
        self.max_length = max_length
        self.preserve_punctuation = preserve_punctuation
        logger.info(
            f"TextPreprocessor initialized with max_length={max_length}, "
            f"preserve_punctuation={preserve_punctuation}"
        )
    
    def clean_html(self, text: str) -> str:
        """
        Remove HTML tags from text using regex patterns.
        
        Args:
            text: Input text potentially containing HTML tags
            
        Returns:
            Text with HTML tags removed
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.clean_html("<p>Hello <b>world</b>!</p>")
            'Hello world!'
        """
        if not text:
            return text
        
        try:
            # Remove HTML tags
            cleaned = self.HTML_TAG_PATTERN.sub('', text)
            logger.debug(f"HTML tags removed from text (length: {len(text)} -> {len(cleaned)})")
            return cleaned
        except Exception as e:
            logger.warning(f"Error cleaning HTML: {e}, returning original text")
            return text
    
    def remove_special_chars(self, text: str) -> str:
        """
        Remove non-ASCII and special characters while preserving semantic meaning.
        
        Preserves alphanumeric characters, spaces, and optionally punctuation marks
        (.,!?;:'"-) to maintain readability and semantic content.
        
        Args:
            text: Input text with potential special characters
            
        Returns:
            Text with special characters removed
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.remove_special_chars("Hello™ world® 2024!")
            'Hello world 2024!'
        """
        if not text:
            return text
        
        try:
            if self.preserve_punctuation:
                # Keep alphanumeric, spaces, and common punctuation
                cleaned = self.SPECIAL_CHARS_PATTERN.sub('', text)
            else:
                # Keep only alphanumeric and spaces
                cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            
            logger.debug(f"Special characters removed (length: {len(text)} -> {len(cleaned)})")
            return cleaned
        except Exception as e:
            logger.warning(f"Error removing special characters: {e}, returning original text")
            return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Handle extra spaces, tabs, and line breaks.
        
        Replaces multiple consecutive whitespace characters (spaces, tabs, newlines)
        with a single space and strips leading/trailing whitespace.
        
        Args:
            text: Input text with potential extra whitespace
            
        Returns:
            Text with normalized whitespace
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.normalize_whitespace("Hello   world\\n\\t  test")
            'Hello world test'
        """
        if not text:
            return text
        
        try:
            # Replace multiple whitespace with single space
            normalized = self.WHITESPACE_PATTERN.sub(' ', text)
            # Strip leading and trailing whitespace
            normalized = normalized.strip()
            logger.debug(f"Whitespace normalized (length: {len(text)} -> {len(normalized)})")
            return normalized
        except Exception as e:
            logger.warning(f"Error normalizing whitespace: {e}, returning original text")
            return text
    
    def lowercase_text(self, text: str) -> str:
        """
        Convert text to lowercase for case normalization.
        
        Args:
            text: Input text with mixed case
            
        Returns:
            Lowercase text
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.lowercase_text("Hello WORLD")
            'hello world'
        """
        if not text:
            return text
        
        try:
            lowercased = text.lower()
            logger.debug("Text converted to lowercase")
            return lowercased
        except Exception as e:
            logger.warning(f"Error lowercasing text: {e}, returning original text")
            return text
    
    def remove_urls_emails(self, text: str) -> str:
        """
        Anonymize URLs and email addresses using regex.
        
        Replaces URLs with [URL] and email addresses with [EMAIL] to protect
        privacy and reduce noise in text analysis.
        
        Args:
            text: Input text potentially containing URLs and emails
            
        Returns:
            Text with URLs and emails anonymized
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.remove_urls_emails("Visit https://example.com or email test@example.com")
            'Visit [URL] or email [EMAIL]'
        """
        if not text:
            return text
        
        try:
            # Replace URLs
            text_no_urls = self.URL_PATTERN.sub('[URL]', text)
            # Replace emails
            text_no_emails = self.EMAIL_PATTERN.sub('[EMAIL]', text_no_urls)
            
            urls_removed = text != text_no_urls
            emails_removed = text_no_urls != text_no_emails
            
            if urls_removed or emails_removed:
                logger.debug(
                    f"Anonymized URLs: {urls_removed}, Emails: {emails_removed}"
                )
            
            return text_no_emails
        except Exception as e:
            logger.warning(f"Error removing URLs/emails: {e}, returning original text")
            return text
    
    def _handle_edge_cases(self, text: Optional[str]) -> Optional[str]:
        """
        Handle edge cases including empty strings, None values, and very long texts.
        
        Args:
            text: Input text that may be None, empty, or very long
            
        Returns:
            Processed text or None if input is None/empty
        """
        # Handle None
        if text is None:
            logger.debug("Received None value, returning None")
            return None
        
        # Handle empty string
        if not text or not text.strip():
            logger.debug("Received empty string, returning empty string")
            return ""
        
        # Handle very long texts - truncate to max_length
        if len(text) > self.max_length:
            logger.warning(
                f"Text length ({len(text)}) exceeds max_length ({self.max_length}), "
                f"truncating"
            )
            text = text[:self.max_length]
        
        return text
    
    def preprocess(self, text: Optional[str]) -> str:
        """
        Main entry point that chains all preprocessing steps.
        
        Applies the following transformations in order:
        1. Edge case handling (None, empty, long texts)
        2. HTML tag removal
        3. URL and email anonymization
        4. Special character removal
        5. Whitespace normalization
        6. Lowercase conversion
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Fully preprocessed text
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess("<p>Visit https://example.com for INFO!</p>")
            'visit [url] for info'
        """
        logger.debug(f"Starting preprocessing for text (length: {len(text) if text else 0})")
        
        # Handle edge cases
        text = self._handle_edge_cases(text)
        if text is None or text == "":
            return ""
        
        try:
            # Chain all preprocessing steps
            text = self.clean_html(text)
            text = self.remove_urls_emails(text)
            text = self.remove_special_chars(text)
            text = self.normalize_whitespace(text)
            text = self.lowercase_text(text)
            
            logger.debug(f"Preprocessing complete (final length: {len(text)})")
            return text
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}, returning empty string")
            return ""
    
    def preprocess_batch(
        self,
        texts: List[Optional[str]],
        max_workers: Optional[int] = None
    ) -> List[str]:
        """
        Efficient batch processing with parallel execution.
        
        Processes multiple texts in parallel using ThreadPoolExecutor for improved
        performance on large batches.
        
        Args:
            texts: List of texts to preprocess
            max_workers: Maximum number of worker threads (default: None = auto)
            
        Returns:
            List of preprocessed texts in the same order as input
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> texts = ["<p>Text 1</p>", "Text 2", None]
            >>> preprocessor.preprocess_batch(texts)
            ['text 1', 'text 2', '']
        """
        if not texts:
            logger.warning("Empty text list provided to preprocess_batch")
            return []
        
        logger.info(f"Starting batch preprocessing for {len(texts)} texts")
        start_time = time.time()
        
        try:
            # For small batches, process sequentially
            if len(texts) < 10:
                logger.debug("Small batch detected, processing sequentially")
                results = [self.preprocess(text) for text in texts]
            else:
                # For larger batches, use parallel processing
                logger.debug(f"Large batch detected, processing in parallel with max_workers={max_workers}")
                results = [None] * len(texts)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks with their indices
                    future_to_index = {
                        executor.submit(self.preprocess, text): i
                        for i, text in enumerate(texts)
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            results[index] = future.result()
                        except Exception as e:
                            logger.error(f"Error processing text at index {index}: {e}")
                            results[index] = ""
            
            # Track preprocessing time
            processing_time = (time.time() - start_time) * 1000
            per_text_time = processing_time / len(texts) if texts else 0
            metrics_collector.track_operation("preprocessing_batch", processing_time)
            metrics_collector.track_operation("preprocessing_per_text", per_text_time)
            
            logger.info(
                f"Batch preprocessing complete: {len(results)} texts processed "
                f"in {processing_time:.2f}ms ({per_text_time:.2f}ms per text)"
            )
            return results
        except Exception as e:
            logger.error(f"Error during batch preprocessing: {e}")
            # Return empty strings for all texts on failure
            return [""] * len(texts)
