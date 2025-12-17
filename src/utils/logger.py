"""
Centralized logging module for Product Hunt RAG Analyzer.

This module provides logging configuration and utilities with support for
console and file handlers, multiple log levels, and structured logging.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


class Logger:
    """
    Centralized logger for the application.
    
    Provides methods to configure and access loggers with console and file
    handlers, supporting multiple log levels and rotation.
    """
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def setup(
        cls,
        log_level: str = "INFO",
        log_format: Optional[str] = None,
        date_format: Optional[str] = None,
        console_enabled: bool = True,
        console_level: str = "INFO",
        file_enabled: bool = True,
        file_level: str = "DEBUG",
        file_path: str = "./logs/app.log",
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ) -> None:
        """
        Setup logging configuration for the application.
        
        Args:
            log_level: Default logging level
            log_format: Log message format string
            date_format: Date format string
            console_enabled: Enable console logging
            console_level: Console log level
            file_enabled: Enable file logging
            file_level: File log level
            file_path: Path to log file
            max_bytes: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        if cls._configured:
            return
        
        # Set default format if not provided
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"
        
        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Add console handler if enabled
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, console_level.upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if file_enabled:
            # Create log directory if it doesn't exist
            log_dir = Path(file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def setup_from_config(cls, config_manager) -> None:
        """
        Setup logging from ConfigManager instance.
        
        Args:
            config_manager: ConfigManager instance with logging configuration
        """
        cls.setup(
            log_level=config_manager.get("logging.level", "INFO"),
            log_format=config_manager.get(
                "logging.format",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            date_format=config_manager.get(
                "logging.date_format",
                "%Y-%m-%d %H:%M:%S"
            ),
            console_enabled=config_manager.get("logging.console.enabled", True),
            console_level=config_manager.get("logging.console.level", "INFO"),
            file_enabled=config_manager.get("logging.file.enabled", True),
            file_level=config_manager.get("logging.file.level", "DEBUG"),
            file_path=config_manager.get("logging.file.path", "./logs/app.log"),
            max_bytes=config_manager.get("logging.file.max_bytes", 10485760),
            backup_count=config_manager.get("logging.file.backup_count", 5)
        )
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name (typically __name__ of the module)
            
        Returns:
            Logger instance
            
        Example:
            >>> logger = Logger.get_logger(__name__)
            >>> logger.info("Processing started")
        """
        if not cls._configured:
            cls.setup()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def reset(cls) -> None:
        """Reset logging configuration (useful for testing)."""
        cls._configured = False
        cls._loggers.clear()
        logging.getLogger().handlers.clear()


# Convenience functions for quick logging
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return Logger.get_logger(name)


def log_operation(logger: logging.Logger, operation: str, level: str = "INFO") -> None:
    """
    Log an operation with timestamp.
    
    Args:
        logger: Logger instance
        operation: Operation description
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_method = getattr(logger, level.lower())
    log_method(f"Operation: {operation}")


def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """
    Log an error with context.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about the error
    """
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg = f"{context} - {error_msg}"
    logger.error(error_msg, exc_info=True)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    details: Optional[dict] = None
) -> None:
    """
    Log performance metrics for an operation.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        details: Additional performance details
    """
    msg = f"Performance: {operation} completed in {duration_ms:.2f}ms"
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        msg = f"{msg} ({detail_str})"
    logger.info(msg)


class LogContext:
    """
    Context manager for logging operation start and end.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, "data processing"):
        ...     # Do processing
        ...     pass
    """
    
    def __init__(self, logger: logging.Logger, operation: str, level: str = "INFO"):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            operation: Operation description
            level: Log level
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        """Enter context - log operation start."""
        self.start_time = datetime.now()
        log_method = getattr(self.logger, self.level.lower())
        log_method(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log operation end and duration."""
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        log_method = getattr(self.logger, self.level.lower())
        
        if exc_type is None:
            log_method(f"Completed: {self.operation} ({duration:.2f}ms)")
        else:
            self.logger.error(
                f"Failed: {self.operation} ({duration:.2f}ms) - {exc_val}",
                exc_info=True
            )
        
        return False  # Don't suppress exceptions
