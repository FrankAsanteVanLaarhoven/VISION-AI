"""
Health check endpoints for QEP-VLA Platform
System health monitoring and status reporting
"""

from fastapi import APIRouter, Request, Depends
from typing import Dict, Any
import logging
from datetime import datetime

from utils.database import health_check as db_health_check
from utils.redis_client import health_check as redis_health_check

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_system_components(request: Request):
    """Get system components from request state"""
    return {
        'quantum_transformer': getattr(request.app.state, 'quantum_transformer', None),
        'federated_trainer': getattr(request.app.state, 'federated_trainer', None),
        'edge_engine': getattr(request.app.state, 'edge_engine', None)
    }

@router.get("/")
async def health_check():
    """
    Basic health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "QEP-VLA Platform Health Check"
    }

@router.get("/detailed")
async def detailed_health_check(
    components: Dict = Depends(get_system_components)
):
    """
    Detailed health check with component status
    """
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "external_services": {},
            "errors": []
        }
        
        # Check database health
        try:
            db_status = await db_health_check()
            health_status["external_services"]["database"] = db_status
            if db_status.get("status") != "healthy":
                health_status["overall_status"] = "degraded"
                health_status["errors"].append(f"Database: {db_status.get('error', 'Unknown error')}")
        except Exception as e:
            health_status["external_services"]["database"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
            health_status["errors"].append(f"Database: {str(e)}")
        
        # Check Redis health
        try:
            redis_status = await redis_health_check()
            health_status["external_services"]["redis"] = redis_status
            if redis_status.get("status") != "healthy":
                health_status["overall_status"] = "degraded"
                health_status["errors"].append(f"Redis: {redis_status.get('error', 'Unknown error')}")
        except Exception as e:
            health_status["external_services"]["redis"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
            health_status["errors"].append(f"Redis: {str(e)}")
        
        # Check core components
        if components['quantum_transformer']:
            try:
                quantum_health = components['quantum_transformer'].validate_privacy_guarantees()
                health_status["components"]["quantum_transformer"] = {
                    "status": "healthy" if quantum_health else "degraded",
                    "privacy_compliance": quantum_health
                }
                if not quantum_health:
                    health_status["overall_status"] = "degraded"
                    health_status["errors"].append("Quantum transformer: Privacy guarantees not met")
            except Exception as e:
                health_status["components"]["quantum_transformer"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
                health_status["errors"].append(f"Quantum transformer: {str(e)}")
        else:
            health_status["components"]["quantum_transformer"] = {"status": "not_initialized"}
        
        if components['edge_engine']:
            try:
                edge_health = components['edge_engine'].health_check()
                health_status["components"]["edge_engine"] = edge_health
                if edge_health.get("status") != "healthy":
                    health_status["overall_status"] = "degraded"
                    health_status["errors"].append(f"Edge engine: {edge_health.get('error', 'Unknown error')}")
            except Exception as e:
                health_status["components"]["edge_engine"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
                health_status["errors"].append(f"Edge engine: {str(e)}")
        else:
            health_status["components"]["edge_engine"] = {"status": "not_initialized"}
        
        if components['federated_trainer']:
            try:
                federated_metrics = components['federated_trainer'].get_training_metrics()
                health_status["components"]["federated_trainer"] = {
                    "status": "healthy",
                    "metrics": federated_metrics
                }
            except Exception as e:
                health_status["components"]["federated_trainer"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
                health_status["errors"].append(f"Federated trainer: {str(e)}")
        else:
            health_status["components"]["federated_trainer"] = {"status": "not_initialized"}
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/database")
async def database_health_check():
    """
    Database-specific health check
    """
    try:
        status = await db_health_check()
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "database",
            **status
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "database",
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/redis")
async def redis_health_check():
    """
    Redis-specific health check
    """
    try:
        status = await redis_health_check()
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "redis",
            **status
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "redis",
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/components")
async def components_health_check(
    components: Dict = Depends(get_system_components)
):
    """
    Core components health check
    """
    try:
        components_status = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check quantum transformer
        if components['quantum_transformer']:
            try:
                privacy_compliance = components['quantum_transformer'].validate_privacy_guarantees()
                metrics = components['quantum_transformer'].get_performance_metrics()
                
                components_status["components"]["quantum_transformer"] = {
                    "status": "healthy" if privacy_compliance else "degraded",
                    "privacy_compliance": privacy_compliance,
                    "performance_metrics": metrics
                }
            except Exception as e:
                components_status["components"]["quantum_transformer"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            components_status["components"]["quantum_transformer"] = {"status": "not_initialized"}
        
        # Check edge engine
        if components['edge_engine']:
            try:
                edge_health = components['edge_engine'].health_check()
                components_status["components"]["edge_engine"] = edge_health
            except Exception as e:
                components_status["components"]["edge_engine"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            components_status["components"]["edge_engine"] = {"status": "not_initialized"}
        
        # Check federated trainer
        if components['federated_trainer']:
            try:
                training_metrics = components['federated_trainer'].get_training_metrics()
                components_status["components"]["federated_trainer"] = {
                    "status": "healthy",
                    "metrics": training_metrics
                }
            except Exception as e:
                components_status["components"]["federated_trainer"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            components_status["components"]["federated_trainer"] = {"status": "not_initialized"}
        
        return components_status
        
    except Exception as e:
        logger.error(f"Components health check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/ready")
async def readiness_check(
    components: Dict = Depends(get_system_components)
):
    """
    Readiness check for Kubernetes/load balancer
    """
    try:
        # Check if all critical components are initialized
        critical_components = ['quantum_transformer', 'edge_engine', 'federated_trainer']
        missing_components = []
        
        for component_name in critical_components:
            if not components.get(component_name):
                missing_components.append(component_name)
        
        if missing_components:
            return {
                "status": "not_ready",
                "timestamp": datetime.now().isoformat(),
                "missing_components": missing_components,
                "message": "System not fully initialized"
            }
        
        # Check database connectivity
        try:
            db_status = await db_health_check()
            if db_status.get("status") != "healthy":
                return {
                    "status": "not_ready",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Database not accessible",
                    "database_status": db_status
                }
        except Exception as e:
            return {
                "status": "not_ready",
                "timestamp": datetime.now().isoformat(),
                "message": "Database health check failed",
                "error": str(e)
            }
        
        # Check Redis connectivity
        try:
            redis_status = await redis_health_check()
            if redis_status.get("status") != "healthy":
                return {
                    "status": "not_ready",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Redis not accessible",
                    "redis_status": redis_status
                }
        except Exception as e:
            return {
                "status": "not_ready",
                "timestamp": datetime.now().isoformat(),
                "message": "Redis health check failed",
                "error": str(e)
            }
        
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "message": "System ready to serve requests"
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "not_ready",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "message": "Service is running"
    }
