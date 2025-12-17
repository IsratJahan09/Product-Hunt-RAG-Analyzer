"""
Configuration management module for Product Hunt RAG Analyzer.

This module provides the ConfigManager class for loading, validating,
and accessing configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class ConfigManager:
    """
    Manages application configuration from YAML files.
    
    Provides methods to load, validate, and access configuration values
    with schema validation and default value handling.
    """
    
    # Required configuration keys for validation
    REQUIRED_KEYS = [
        "api",
        "models",
        "processing",
        "storage",
        "retrieval",
        "fastapi",
        "logging"
    ]
    
    # Required nested keys
    REQUIRED_NESTED = {
        "api": ["product_hunt"],
        "models": ["embedding", "sentiment", "llm"],
        "storage": ["data_dir", "indices_dir", "faiss"],
        "fastapi": ["host", "port"],
        "logging": ["level"]
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
        self._setup_directories()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Look for config in multiple locations
        possible_paths = [
            os.getenv("CONFIG_PATH"),
            "./config/default_config.yaml",
            "../config/default_config.yaml",
            "../../config/default_config.yaml"
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        # If no config found, use the standard location
        return "./config/default_config.yaml"
    
    def _load_config(self) -> None:
        """
        Load configuration from YAML file.
        
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            if self.config is None:
                raise ConfigValidationError("Configuration file is empty")
                
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Failed to parse YAML config: {e}")
    
    def _validate_config(self) -> None:
        """
        Validate configuration structure and required keys.
        
        Raises:
            ConfigValidationError: If validation fails.
        """
        # Check required top-level keys
        missing_keys = [key for key in self.REQUIRED_KEYS if key not in self.config]
        if missing_keys:
            raise ConfigValidationError(
                f"Missing required configuration keys: {', '.join(missing_keys)}"
            )
        
        # Check required nested keys
        for parent_key, required_children in self.REQUIRED_NESTED.items():
            if parent_key not in self.config:
                continue
            
            parent_config = self.config[parent_key]
            if not isinstance(parent_config, dict):
                raise ConfigValidationError(
                    f"Configuration key '{parent_key}' must be a dictionary"
                )
            
            missing_children = [
                child for child in required_children 
                if child not in parent_config
            ]
            if missing_children:
                raise ConfigValidationError(
                    f"Missing required keys in '{parent_key}': {', '.join(missing_children)}"
                )
        
        # Validate specific values
        self._validate_values()
    
    def _validate_values(self) -> None:
        """Validate specific configuration values."""
        # Validate logging level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_level = self.get("logging.level", "INFO")
        if log_level not in valid_log_levels:
            raise ConfigValidationError(
                f"Invalid logging level: {log_level}. Must be one of {valid_log_levels}"
            )
        
        # Validate FAISS index type
        valid_index_types = ["flat", "ivf", "hnsw"]
        index_type = self.get("storage.faiss.index_type", "flat")
        if index_type not in valid_index_types:
            raise ConfigValidationError(
                f"Invalid FAISS index type: {index_type}. Must be one of {valid_index_types}"
            )
        
        # Validate device
        valid_devices = ["cpu", "cuda"]
        embedding_device = self.get("models.embedding.device", "cpu")
        if embedding_device not in valid_devices:
            raise ConfigValidationError(
                f"Invalid embedding device: {embedding_device}. Must be one of {valid_devices}"
            )
        
        # Validate numeric values
        if self.get("fastapi.port", 8000) <= 0:
            raise ConfigValidationError("FastAPI port must be positive")
        
        if self.get("models.embedding.dimension", 384) <= 0:
            raise ConfigValidationError("Embedding dimension must be positive")
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.get("storage.data_dir", "./data"),
            self.get("storage.raw_dir", "./data/raw"),
            self.get("storage.processed_dir", "./data/processed"),
            self.get("storage.indices_dir", "./data/indices"),
        ]
        
        # Add log directory if file logging is enabled
        if self.get("logging.file.enabled", True):
            log_path = self.get("logging.file.path", "./logs/app.log")
            log_dir = os.path.dirname(log_path)
            if log_dir:
                directories.append(log_dir)
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., "models.embedding.name")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get("models.embedding.name")
            "all-MiniLM-L6-v2"
            >>> config.get("models.embedding.batch_size", 32)
            32
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get entire configuration dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        self._validate_config()
        self._setup_directories()
    
    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return f"ConfigManager(config_path='{self.config_path}')"
