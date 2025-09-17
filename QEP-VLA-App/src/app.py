#!/usr/bin/env python3
"""
QEP-VLA Platform Main Application
FastAPI application with privacy-preserving AI capabilities
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Import core components
from core.quantum_privacy_transform import QuantumPrivacyTransform
from core.federated_trainer import SecureFederatedTrainer
from core.edge_inference import AdaptiveEdgeInferenceEngine

# Import API routes
from api.routes import router as api_router

# Import utilities
from config.settings import get_settings
from utils.database import init_database, close_database
from utils.redis_client import init_redis, close_redis
from utils.logging_config import setup_logging

# Import monitoring
from monitoring.health_check import router as health_check_router

# Get settings
settings = get_settings()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting QEP-VLA Platform...")
    
    try:
        # Initialize database
        await init_database()
        
        # Initialize Redis
        await init_redis()
        
        # Initialize core components
        app.state.quantum_transformer = QuantumPrivacyTransform()
        app.state.federated_trainer = SecureFederatedTrainer()
        app.state.edge_engine = AdaptiveEdgeInferenceEngine()
        
        logger.info("✅ QEP-VLA Platform started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize QEP-VLA Platform: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down QEP-VLA Platform...")
    
    try:
        await close_database()
        await close_redis()
        logger.info("✅ QEP-VLA Platform shut down successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Quantum-Enhanced Privacy Vision-LiDAR-AI Platform",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts
)

# Add routes
app.include_router(health_check_router, prefix="/health", tags=["health"])
app.include_router(api_router, prefix="/api/v1", tags=["api"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to QEP-VLA Platform",
        "version": settings.app_version,
        "status": "running"
    }

@app.get("/info")
async def info():
    """Platform information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
