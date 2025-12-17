"""
Embedding generation module for Product Hunt RAG Analyzer.

This module provides text-to-vector embedding generation using Sentence-Transformers
with support for batch processing, GPU acceleration, and progress tracking.
"""

import time
from typing import List, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics_collector = get_metrics_collector()


class EmbeddingGenerator:
    """
    Manages embedding generation using Sentence-Transformers.
    
    Provides methods to convert text into dense vector embeddings with support
    for batch processing, GPU acceleration, and memory-efficient operations.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        lazy_load: bool = False
    ):
        """
        Initialize EmbeddingGenerator with model name and device selection.
        
        Args:
            model_name: Name of the Sentence-Transformers model (default: all-MiniLM-L6-v2)
            device: Device to use ("cpu", "cuda", or None for auto-detection)
            lazy_load: If True, delay model loading until first use (default: False)
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> generator = EmbeddingGenerator(device="cuda")
            >>> generator = EmbeddingGenerator(lazy_load=True)  # For testing
        """
        self.model_name = model_name
        self.device = self._select_device(device)
        self.model = None
        self._embedding_dimension = None
        
        logger.info(
            f"EmbeddingGenerator initialized with model={model_name}, device={self.device}"
        )
        
        # Load model during initialization unless lazy_load is True
        if not lazy_load:
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
                logger.info("GPU acceleration: DISABLED (CUDA not available)")
                return "cpu"
            logger.info(f"Using user-specified device: {device}")
            if device == "cuda":
                logger.info(f"GPU acceleration: ENABLED - {torch.cuda.get_device_name(0)}")
            else:
                logger.info("GPU acceleration: DISABLED (CPU mode)")
            return device
        
        # Auto-detect device
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(
                f"GPU acceleration: ENABLED - {gpu_name} "
                f"({gpu_memory:.2f}GB memory)"
            )
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            device = "cpu"
            logger.info("GPU acceleration: DISABLED (CUDA not available)")
            logger.info("Consider installing CUDA for faster embedding generation")
        
        return device
    
    def load_model(self, model_name: str) -> None:
        """
        Load Sentence-Transformers model.
        
        Args:
            model_name: Name of the model to load from Sentence-Transformers
            
        Raises:
            Exception: If model loading fails
            
        Example:
            >>> generator = EmbeddingGenerator()
            >>> generator.load_model("all-MiniLM-L6-v2")
        """
        try:
            start_time = time.time()
            logger.info(f"Loading Sentence-Transformers model: {model_name}")
            
            # Load model
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Get embedding dimension
            self._embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            load_time = (time.time() - start_time) * 1000
            logger.info(
                f"Model loaded successfully in {load_time:.2f}ms "
                f"(dimension: {self._embedding_dimension})"
            )
            
            # Track model loading time
            metrics_collector.track_operation("model_loading", load_time)
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
            raise Exception(f"Model loading failed: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        Return the embedding dimension for the loaded model.
        
        Returns:
            Embedding dimension (384 for all-MiniLM-L6-v2)
            
        Example:
            >>> generator = EmbeddingGenerator()
            >>> generator.get_embedding_dimension()
            384
        """
        if self._embedding_dimension is None:
            if self.model is not None:
                self._embedding_dimension = self.model.get_sentence_embedding_dimension()
            else:
                logger.warning("Model not loaded, returning default dimension 384")
                return 384
        
        return self._embedding_dimension
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Create 384-dimensional vectors from single or multiple texts.
        
        Args:
            texts: Single text string or list of text strings
            normalize: Whether to normalize embeddings to unit length (default: True)
            show_progress: Whether to show progress bar (default: False)
            
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
            
        Raises:
            Exception: If embedding generation fails
            
        Example:
            >>> generator = EmbeddingGenerator()
            >>> embedding = generator.generate_embeddings("Hello world")
            >>> embedding.shape
            (1, 384)
            >>> embeddings = generator.generate_embeddings(["Text 1", "Text 2"])
            >>> embeddings.shape
            (2, 384)
        """
        if self.model is None:
            logger.info("Model not loaded, loading now...")
            self.load_model(self.model_name)
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        if not texts:
            logger.warning("Empty text list provided, returning empty array")
            return np.array([])
        
        try:
            start_time = time.time()
            logger.debug(f"Generating embeddings for {len(texts)} text(s)")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            generation_time = (time.time() - start_time) * 1000
            per_text_time = generation_time / len(texts)
            logger.info(
                f"Generated {len(texts)} embedding(s) in {generation_time:.2f}ms "
                f"({per_text_time:.2f}ms per text)"
            )
            
            # Track embedding generation metrics
            metrics_collector.track_operation("embedding_generation_batch", generation_time)
            metrics_collector.track_operation("embedding_generation_per_text", per_text_time)
            metrics_collector.track_memory_usage()
            
            # Ensure 2D array even for single input
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            return embeddings
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"CUDA out of memory error: {e}. Try reducing batch size or using CPU.",
                exc_info=True
            )
            raise Exception(f"CUDA out of memory: {e}")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise Exception(f"Embedding generation failed: {e}")
    
    def batch_generate(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Memory-efficient batch processing with progress tracking.
        
        Processes texts in batches to manage memory usage and provides
        progress tracking for large datasets.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per batch (default: 32)
            show_progress: Whether to show progress bar (default: True)
            normalize: Whether to normalize embeddings (default: True)
            
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
            
        Raises:
            Exception: If batch generation fails
            
        Example:
            >>> generator = EmbeddingGenerator()
            >>> texts = ["Text 1", "Text 2", ..., "Text 1000"]
            >>> embeddings = generator.batch_generate(texts, batch_size=32)
            >>> embeddings.shape
            (1000, 384)
        """
        if self.model is None:
            logger.info("Model not loaded, loading now...")
            self.load_model(self.model_name)
        
        if not texts:
            logger.warning("Empty text list provided, returning empty array")
            return np.array([])
        
        try:
            start_time = time.time()
            logger.info(
                f"Starting batch embedding generation for {len(texts)} texts "
                f"with batch_size={batch_size}"
            )
            
            # Use Sentence-Transformers' built-in batch processing
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            generation_time = (time.time() - start_time) * 1000
            per_text_time = generation_time / len(texts)
            logger.info(
                f"Batch generation complete: {len(texts)} embeddings in "
                f"{generation_time:.2f}ms ({per_text_time:.2f}ms per text)"
            )
            
            # Track batch generation metrics
            metrics_collector.track_operation("embedding_batch_generation", generation_time)
            metrics_collector.track_operation("embedding_batch_per_text", per_text_time)
            
            # Log and track memory usage
            memory_snapshot = metrics_collector.track_memory_usage()
            
            # Log GPU memory usage if on CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                logger.info(
                    f"GPU memory: {memory_allocated:.2f}MB allocated, "
                    f"{memory_reserved:.2f}MB reserved, {peak_memory:.2f}MB peak"
                )
            
            return embeddings
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"CUDA out of memory during batch processing: {e}. "
                f"Try reducing batch_size from {batch_size}.",
                exc_info=True
            )
            raise Exception(f"CUDA out of memory: {e}")
        except Exception as e:
            logger.error(f"Batch generation failed: {e}", exc_info=True)
            raise Exception(f"Batch generation failed: {e}")
    
    def __repr__(self) -> str:
        """String representation of EmbeddingGenerator."""
        return (
            f"EmbeddingGenerator(model='{self.model_name}', "
            f"device='{self.device}', dimension={self.get_embedding_dimension()})"
        )
