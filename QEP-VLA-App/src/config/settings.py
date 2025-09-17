#!/usr/bin/env python3
"""
Configuration settings for QEP-VLA Platform
Uses Pydantic for environment variable management
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    app_name: str = Field(default="QEP-VLA Platform", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Database settings
    postgres_url: str = Field(
        default="postgresql://qepvla:qepvla123@localhost:5432/qepvla",
        description="PostgreSQL connection URL"
    )
    postgres_pool_size: int = Field(default=10, description="Database connection pool size")
    postgres_max_overflow: int = Field(default=20, description="Database max overflow connections")
    
    # Redis settings
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    redis_pool_size: int = Field(default=10, description="Redis connection pool size")
    
    # Blockchain settings
    ganache_url: str = Field(
        default="http://localhost:8545",
        description="Ganache blockchain endpoint"
    )
    web3_provider: str = Field(
        default="http://localhost:8545",
        description="Web3 provider URL"
    )
    
    # Privacy settings
    privacy_budget: float = Field(default=1.0, description="Default privacy budget")
    noise_scale: float = Field(default=0.1, description="Differential privacy noise scale")
    max_queries: int = Field(default=100, description="Maximum queries per privacy budget")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Model settings
    model_cache_dir: str = Field(default="./models", description="Model cache directory")
    model_download_timeout: int = Field(default=300, description="Model download timeout in seconds")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-here", description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration time")
    
    # CORS settings
    cors_origins: list = Field(default=["*"], description="Allowed CORS origins")
    cors_methods: list = Field(default=["*"], description="Allowed CORS methods")
    cors_headers: list = Field(default=["*"], description="Allowed CORS headers")
    
    # Trusted hosts
    trusted_hosts: list = Field(default=["*"], description="Trusted host patterns")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# Environment-specific settings
def get_environment_settings() -> Settings:
    """Get environment-specific settings"""
    env = os.getenv("QEP_VLA_ENV", "development")
    
    if env == "production":
        return Settings(
            environment="production",
            debug=False,
            log_level="WARNING",
            cors_origins=["https://yourdomain.com"],
            trusted_hosts=["yourdomain.com", "*.yourdomain.com"]
        )
    elif env == "staging":
        return Settings(
            environment="staging",
            debug=True,
            log_level="INFO"
        )
    else:
        return Settings(
            environment="development",
            debug=True,
            log_level="DEBUG"
        )
