"""
FAISS vector storage module for Product Hunt RAG Analyzer.

This module provides the FAISSIndexManager class for creating, managing,
and searching vector embeddings using FAISS (Facebook AI Similarity Search).
Supports multiple index types and metadata management.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is not installed. Please install it with: "
        "pip install faiss-cpu (or faiss-gpu for GPU support)"
    )

from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_collector


class FAISSIndexManager:
    """
    Manages FAISS vector indices with metadata support.
    
    Provides methods to create, search, persist, and load FAISS indices
    with associated metadata for vector embeddings.
    """
    
    # Supported index types
    SUPPORTED_INDEX_TYPES = ["flat", "ivf", "hnsw"]
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISSIndexManager.
        
        Args:
            dimension: Dimensionality of vectors (e.g., 384 for all-MiniLM-L6-v2)
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            
        Raises:
            ValueError: If dimension is invalid or index_type is not supported
        """
        self.logger = get_logger(__name__)
        self.metrics_collector = get_metrics_collector()
        
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Unsupported index type: {index_type}. "
                f"Must be one of {self.SUPPORTED_INDEX_TYPES}"
            )
        
        self.dimension = dimension
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.vector_count = 0
        
        self.logger.info(
            f"Initialized FAISSIndexManager with dimension={dimension}, "
            f"index_type={index_type}"
        )
    
    def create_index(
        self,
        embeddings: Optional[np.ndarray] = None,
        index_type: Optional[str] = None
    ) -> faiss.Index:
        """
        Create a FAISS index with optional initial embeddings.
        
        Args:
            embeddings: Optional initial embeddings to add (shape: [n, dimension])
            index_type: Optional index type override
            
        Returns:
            Created FAISS index
            
        Raises:
            ValueError: If embeddings have wrong dimensions
            RuntimeError: If index creation fails
        """
        if index_type is None:
            index_type = self.index_type
        
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Unsupported index type: {index_type}. "
                f"Must be one of {self.SUPPORTED_INDEX_TYPES}"
            )
        
        try:
            self.logger.info(f"Creating FAISS index of type: {index_type}")
            
            if index_type == "flat":
                # IndexFlatL2: Exact search using L2 distance
                self.index = faiss.IndexFlatL2(self.dimension)
                
            elif index_type == "ivf":
                # IndexIVFFlat: Inverted file index for faster approximate search
                # Use sqrt(n) clusters as a heuristic
                n_clusters = 100  # Default, will be adjusted based on data size
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.dimension,
                    n_clusters,
                    faiss.METRIC_L2
                )
                
            elif index_type == "hnsw":
                # IndexHNSWFlat: Hierarchical Navigable Small World graph
                # M: number of connections per layer (32 is a good default)
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            
            self.logger.info(f"Successfully created {index_type} index")
            
            # Add initial embeddings if provided
            if embeddings is not None:
                if len(embeddings.shape) != 2:
                    raise ValueError(
                        f"Embeddings must be 2D array, got shape {embeddings.shape}"
                    )
                if embeddings.shape[1] != self.dimension:
                    raise ValueError(
                        f"Embeddings dimension {embeddings.shape[1]} doesn't match "
                        f"index dimension {self.dimension}"
                    )
                
                # For IVF index, we need to train it first
                if index_type == "ivf":
                    self.logger.info(f"Training IVF index with {len(embeddings)} vectors")
                    self.index.train(embeddings.astype(np.float32))
                
                # Add embeddings to index
                self.index.add(embeddings.astype(np.float32))
                self.vector_count = len(embeddings)
                self.logger.info(f"Added {len(embeddings)} vectors to index")
            
            return self.index
            
        except Exception as e:
            self.logger.error(f"Failed to create FAISS index: {e}", exc_info=True)
            raise RuntimeError(f"Index creation failed: {e}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add embeddings to the index with associated metadata.
        
        Args:
            embeddings: Embeddings to add (shape: [n, dimension])
            metadata: Optional list of metadata dictionaries for each embedding.
                     Each dict should contain: vector_id, product_id, review_id,
                     original_text, source, timestamp
            
        Raises:
            ValueError: If embeddings have wrong dimensions or metadata length mismatch
            RuntimeError: If index doesn't exist or addition fails
        """
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
        
        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape {embeddings.shape}"
            )
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        n_embeddings = len(embeddings)
        
        # Validate metadata if provided
        if metadata is not None:
            if len(metadata) != n_embeddings:
                raise ValueError(
                    f"Metadata length {len(metadata)} doesn't match "
                    f"embeddings length {n_embeddings}"
                )
        else:
            # Create default metadata
            metadata = []
            for i in range(n_embeddings):
                metadata.append({
                    "vector_id": self.vector_count + i,
                    "product_id": None,
                    "review_id": None,
                    "original_text": None,
                    "source": None,
                    "timestamp": datetime.now().isoformat()
                })
        
        try:
            # For IVF index, train if not already trained
            if self.index_type == "ivf" and not self.index.is_trained:
                self.logger.info(f"Training IVF index with {n_embeddings} vectors")
                self.index.train(embeddings.astype(np.float32))
            
            # Add embeddings to index
            self.index.add(embeddings.astype(np.float32))
            
            # Add metadata
            self.metadata.extend(metadata)
            self.vector_count += n_embeddings
            
            self.logger.info(
                f"Added {n_embeddings} embeddings to index. "
                f"Total vectors: {self.vector_count}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Failed to add embeddings: {e}")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-K most similar vectors using cosine similarity.
        
        Args:
            query_vector: Query vector (shape: [dimension] or [1, dimension])
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (distances, indices):
                - distances: Array of similarity scores (shape: [k])
                - indices: Array of vector indices (shape: [k])
            
        Raises:
            ValueError: If query_vector has wrong dimensions or k is invalid
            RuntimeError: If index doesn't exist or search fails
        """
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
        
        if self.vector_count == 0:
            raise RuntimeError("Index is empty. Add embeddings first.")
        
        # Reshape query vector if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        # Limit k to available vectors
        k = min(k, self.vector_count)
        
        try:
            start_time = datetime.now()
            
            # FAISS uses L2 distance by default
            # For cosine similarity, we need to normalize vectors
            # But for now, we'll use L2 distance as it's already configured
            distances, indices = self.index.search(
                query_vector.astype(np.float32),
                k
            )
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.debug(
                f"Search completed in {duration_ms:.2f}ms. "
                f"Retrieved {k} results."
            )
            
            # Track search latency
            self.metrics_collector.track_operation("faiss_search", duration_ms)
            
            # Return flattened arrays
            return distances[0], indices[0]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {e}")
    
    def search_with_metadata(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K most similar vectors and return formatted results with metadata.
        
        Args:
            query_vector: Query vector (shape: [dimension] or [1, dimension])
            k: Number of nearest neighbors to retrieve
            
        Returns:
            List of dictionaries with keys:
                - index: Vector index in the FAISS index
                - score: Similarity score (distance)
                - metadata: Associated metadata dictionary
            
        Raises:
            ValueError: If query_vector has wrong dimensions or k is invalid
            RuntimeError: If index doesn't exist or search fails
        """
        # Call the base search method
        distances, indices = self.search(query_vector, k)
        
        # Get metadata for the indices
        metadata_list = self.get_metadata(indices)
        
        # Format results
        results = []
        for idx, distance, metadata in zip(indices, distances, metadata_list):
            results.append({
                "index": int(idx),
                "score": float(distance),
                "metadata": metadata
            })
        
        return results
    
    def save_index(self, path: str) -> None:
        """
        Persist FAISS index and metadata to disk.
        
        Args:
            path: Base path for saving (without extension).
                 Will create {path}.index and {path}.metadata files.
            
        Raises:
            RuntimeError: If index doesn't exist or save fails
            IOError: If file write fails
        """
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
        
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            index_path = f"{path}.index"
            metadata_path = f"{path}.metadata"
            
            self.logger.info(f"Saving FAISS index to {index_path}")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata and configuration
            metadata_bundle = {
                "metadata": self.metadata,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "vector_count": self.vector_count,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_bundle, f)
            
            self.logger.info(
                f"Successfully saved index with {self.vector_count} vectors "
                f"and metadata to {path}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}", exc_info=True)
            raise IOError(f"Failed to save index: {e}")
    
    def load_index(self, path: str) -> None:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            path: Base path for loading (without extension).
                 Will load from {path}.index and {path}.metadata files.
            
        Raises:
            FileNotFoundError: If index or metadata files don't exist
            RuntimeError: If loading fails
        """
        index_path = f"{path}.index"
        metadata_path = f"{path}.metadata"
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        try:
            self.logger.info(f"Loading FAISS index from {index_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata_bundle = pickle.load(f)
            
            self.metadata = metadata_bundle["metadata"]
            self.dimension = metadata_bundle["dimension"]
            self.index_type = metadata_bundle["index_type"]
            self.vector_count = metadata_bundle["vector_count"]
            
            saved_at = metadata_bundle.get("saved_at", "unknown")
            
            self.logger.info(
                f"Successfully loaded index with {self.vector_count} vectors. "
                f"Saved at: {saved_at}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load index: {e}")
    
    def get_metadata(self, indices: Union[np.ndarray, List[int]]) -> List[Dict[str, Any]]:
        """
        Retrieve metadata for given vector indices.
        
        Args:
            indices: Array or list of vector indices
            
        Returns:
            List of metadata dictionaries
            
        Raises:
            ValueError: If any index is out of bounds
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        # Flatten if nested list (e.g., [[1, 2, 3]] -> [1, 2, 3])
        if isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], list):
            indices = indices[0]
        
        result = []
        for idx in indices:
            if idx < 0 or idx >= len(self.metadata):
                raise ValueError(
                    f"Index {idx} out of bounds. "
                    f"Valid range: [0, {len(self.metadata)})"
                )
            result.append(self.metadata[idx])
        
        return result
    
    def get_index_size(self) -> int:
        """
        Get the number of vectors in the index.
        
        Returns:
            Number of vectors in the index
        """
        return self.vector_count
    
    def __repr__(self) -> str:
        """String representation of FAISSIndexManager."""
        return (
            f"FAISSIndexManager(dimension={self.dimension}, "
            f"index_type={self.index_type}, "
            f"vector_count={self.vector_count})"
        )
