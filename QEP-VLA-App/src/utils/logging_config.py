#!/usr/bin/env python3
"""
Logging configuration for QEP-VLA Platform
Provides structured and colored logging with file rotation
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

from config.settings import get_settings

settings = get_settings()

class StructuredFormatter(logging.Formatter):
    """JSON structured formatter for production environments"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class ColoredFormatter(logging.Formatter):
    """Colored formatter for development environments"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = False,
    structured: bool = False
) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_console: Enable console logging
        enable_file: Enable file logging
        structured: Use structured JSON logging
    """
    
    # Get log level from settings if not provided
    if log_level is None:
        log_level = settings.log_level
    
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if enable_file:
        log_dir = Path(log_file).parent if log_file else Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if structured:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        log_file_path = log_file or "logs/qep_vla.log"
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=5,
            encoding='utf-8'
        )
        
        file_handler.setLevel(numeric_level)
        
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Console logging: {enable_console}")
    logger.info(f"File logging: {enable_file}")
    logger.info(f"Structured logging: {structured}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def log_function_call(func):
    """Decorator to log function calls with parameters and timing"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} completed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    
    return wrapper

def log_performance(func):
    """Decorator to log function performance metrics"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        start_time = datetime.now()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            end_memory = get_memory_usage()
            memory_delta = end_memory - start_memory
            
            logger.info(f"{func.__name__} performance: {execution_time:.4f}s, memory: {memory_delta:+d}MB")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    
    return wrapper

def get_memory_usage() -> int:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return int(process.memory_info().rss / 1024 / 1024)
    except ImportError:
        return 0

class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, operation_name: str, logger_name: str = None):
        self.operation_name = operation_name
        self.logger = get_logger(logger_name or __name__)
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.start_memory = get_memory_usage()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = (datetime.now() - self.start_time).total_seconds()
        end_memory = get_memory_usage()
        memory_delta = end_memory - self.start_memory
        
        if exc_type is None:
            self.logger.info(f"{self.operation_name} completed: {execution_time:.4f}s, memory: {memory_delta:+d}MB")
        else:
            self.logger.error(f"{self.operation_name} failed after {execution_time:.4f}s: {exc_val}")
        
        return False  # Don't suppress exceptions

# Convenience function for performance logging
def log_performance_context(operation_name: str, logger_name: str = None):
    """Create a performance logging context manager"""
    return PerformanceLogger(operation_name, logger_name)
