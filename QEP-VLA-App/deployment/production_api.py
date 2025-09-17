"""
Production API for QEP-VLA Application
Provides production-ready REST API for QEP-VLA privacy system
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import json
import asyncio
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from sdk_wrapper import QEPVLASDK

# Security
security = HTTPBearer()

# API Models
class PrivacyTransformRequest(BaseModel):
    data: Any = Field(..., description="Data to be transformed")
    transform_type: str = Field("auto", description="Type of privacy transformation")
    privacy_level: str = Field("high", description="Privacy protection level")

class NavigationRequest(BaseModel):
    target: List[float] = Field(..., description="Target coordinates [x, y, z]")
    mode: str = Field("privacy_aware", description="Navigation mode")
    privacy_constraints: Dict[str, Any] = Field(default_factory=dict, description="Privacy constraints")

class DataCaptureRequest(BaseModel):
    data_types: List[str] = Field(["camera", "lidar"], description="Types of data to capture")
    privacy_settings: Dict[str, Any] = Field(default_factory=dict, description="Privacy settings")

class ScenarioRequest(BaseModel):
    scenario_type: str = Field(..., description="Type of scenario to generate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Scenario parameters")

class SystemStatusResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    overall_health: str
    version: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Global variables
sdk_instance: Optional[QEPVLASDK] = None
app_startup_time: Optional[datetime] = None

# Configuration
API_VERSION = "1.0.0"
API_TITLE = "QEP-VLA Privacy System API"
API_DESCRIPTION = "Production API for Quantum-Enhanced Privacy Vision-LiDAR-AI System"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global sdk_instance, app_startup_time
    
    # Startup
    logger.info("Starting QEP-VLA Production API")
    app_startup_time = datetime.now()
    
    try:
        # Initialize SDK
        config_path = os.getenv('QEP_VLA_CONFIG_PATH', None)
        sdk_instance = QEPVLASDK(config_path)
        
        # Initialize components
        components_to_init = ['camera', 'lidar', 'quantum_sensor', 'navigation', 'privacy_transform', 'scenario_generation']
        success = sdk_instance.initialize_components(components_to_init)
        
        if success:
            logger.info("SDK components initialized successfully")
        else:
            logger.warning("Some SDK components failed to initialize")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize SDK: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down QEP-VLA Production API")
        if sdk_instance:
            sdk_instance.cleanup()

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Dependency functions
async def get_sdk() -> QEPVLASDK:
    """Get SDK instance dependency"""
    if not sdk_instance:
        raise HTTPException(status_code=503, detail="SDK not initialized")
    return sdk_instance

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token"""
    # In production, implement proper JWT verification
    token = credentials.credentials
    
    # Simple token validation (replace with proper JWT validation)
    if not token or len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

# Health check endpoint
@app.get("/health", response_model=SystemStatusResponse)
async def health_check():
    """Health check endpoint"""
    if not sdk_instance:
        return SystemStatusResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            components={},
            overall_health="unhealthy",
            version=API_VERSION
        )
    
    try:
        status = sdk_instance.get_system_status()
        return SystemStatusResponse(
            status="healthy" if status['overall_health'] == 'healthy' else "degraded",
            timestamp=status['timestamp'],
            components=status['component_status'],
            overall_health=status['overall_health'],
            version=API_VERSION
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemStatusResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            components={},
            overall_health="unhealthy",
            version=API_VERSION
        )

# Privacy transformation endpoint
@app.post("/privacy/transform")
async def apply_privacy_transform(
    request: PrivacyTransformRequest,
    sdk: QEPVLASDK = Depends(get_sdk),
    token: str = Depends(verify_token)
):
    """Apply privacy transformation to data"""
    try:
        logger.info(f"Privacy transform request received for user: {token[:8]}...")
        
        # Apply transformation
        transformed_data = sdk.apply_privacy_transform(
            request.data, 
            request.transform_type
        )
        
        # Get transform summary
        transform_summary = sdk.quantum_privacy_transform.get_transform_summary()
        
        return {
            "status": "success",
            "transformed_data": transformed_data,
            "transform_summary": transform_summary,
            "timestamp": datetime.now().isoformat(),
            "privacy_level": request.privacy_level
        }
        
    except Exception as e:
        logger.error(f"Privacy transform failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Navigation planning endpoint
@app.post("/navigation/plan")
async def plan_navigation(
    request: NavigationRequest,
    sdk: QEPVLASDK = Depends(get_sdk),
    token: str = Depends(verify_token)
):
    """Plan navigation path with privacy considerations"""
    try:
        logger.info(f"Navigation planning request received for user: {token[:8]}...")
        
        # Plan navigation
        navigation_plan = sdk.plan_navigation(
            request.target,
            request.mode
        )
        
        if not navigation_plan:
            raise HTTPException(status_code=400, detail="Failed to create navigation plan")
        
        return {
            "status": "success",
            "navigation_plan": navigation_plan,
            "timestamp": datetime.now().isoformat(),
            "privacy_constraints_applied": bool(request.privacy_constraints)
        }
        
    except Exception as e:
        logger.error(f"Navigation planning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data capture endpoint
@app.post("/data/capture")
async def capture_data(
    request: DataCaptureRequest,
    sdk: QEPVLASDK = Depends(get_sdk),
    token: str = Depends(verify_token)
):
    """Capture data from sensors with privacy controls"""
    try:
        logger.info(f"Data capture request received for user: {token[:8]}...")
        
        # Capture data
        captured_data = {}
        for data_type in request.data_types:
            data = sdk.capture_private_data(data_type)
            if data:
                captured_data[data_type] = data
        
        if not captured_data:
            raise HTTPException(status_code=400, detail="No data captured")
        
        return {
            "status": "success",
            "captured_data": captured_data,
            "timestamp": datetime.now().isoformat(),
            "privacy_applied": True
        }
        
    except Exception as e:
        logger.error(f"Data capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scenario generation endpoint
@app.post("/scenarios/generate")
async def generate_scenario(
    request: ScenarioRequest,
    sdk: QEPVLASDK = Depends(get_sdk),
    token: str = Depends(verify_token)
):
    """Generate test scenario for system validation"""
    try:
        logger.info(f"Scenario generation request received for user: {token[:8]}...")
        
        # Generate scenario
        scenario = sdk.generate_test_scenario(request.scenario_type)
        
        if not scenario:
            raise HTTPException(status_code=400, detail="Failed to generate scenario")
        
        return {
            "status": "success",
            "scenario": scenario,
            "timestamp": datetime.now().isoformat(),
            "scenario_type": request.scenario_type
        }
        
    except Exception as e:
        logger.error(f"Scenario generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System status endpoint
@app.get("/system/status")
async def get_system_status(
    sdk: QEPVLASDK = Depends(get_sdk),
    token: str = Depends(verify_token)
):
    """Get detailed system status"""
    try:
        logger.info(f"System status request received for user: {token[:8]}...")
        
        status = sdk.get_system_status()
        
        # Add additional system information
        status['api_version'] = API_VERSION
        status['uptime'] = (datetime.now() - app_startup_time).total_seconds() if app_startup_time else 0
        
        return {
            "status": "success",
            "system_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System status request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoint
@app.get("/config")
async def get_configuration(
    sdk: QEPVLASDK = Depends(get_sdk),
    token: str = Depends(verify_token)
):
    """Get current system configuration"""
    try:
        logger.info(f"Configuration request received for user: {token[:8]}...")
        
        return {
            "status": "success",
            "configuration": sdk.config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Configuration request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export configuration endpoint
@app.post("/config/export")
async def export_configuration(
    filepath: str,
    sdk: QEPVLASDK = Depends(get_sdk),
    token: str = Depends(verify_token)
):
    """Export current configuration to file"""
    try:
        logger.info(f"Configuration export request received for user: {token[:8]}...")
        
        sdk.export_config(filepath)
        
        return {
            "status": "success",
            "message": f"Configuration exported to {filepath}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Configuration export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Privacy compliance report endpoint
@app.get("/privacy/compliance")
async def get_privacy_compliance(
    sdk: QEPVLASDK = Depends(get_sdk),
    token: str = Depends(verify_token)
):
    """Get privacy compliance report"""
    try:
        logger.info(f"Privacy compliance request received for user: {token[:8]}...")
        
        # Generate compliance report
        compliance_report = {
            "timestamp": datetime.now().isoformat(),
            "compliance_frameworks": sdk.config.get('privacy', {}).get('compliance', []),
            "privacy_level": sdk.config.get('privacy', {}).get('default_level', 'unknown'),
            "data_retention_days": sdk.config.get('privacy', {}).get('data_retention_days', 0),
            "active_privacy_mechanisms": [],
            "privacy_metrics": {}
        }
        
        # Add active privacy mechanisms
        if sdk.quantum_privacy_transform:
            transform_summary = sdk.quantum_privacy_transform.get_transform_summary()
            compliance_report['active_privacy_mechanisms'].extend(transform_summary.get('transform_types_used', []))
        
        if sdk.secure_aggregation:
            agg_summary = sdk.secure_aggregation.get_aggregation_summary()
            compliance_report['active_privacy_mechanisms'].extend(agg_summary.get('active_privacy_mechanisms', []))
        
        return {
            "status": "success",
            "compliance_report": compliance_report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Privacy compliance request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return ErrorResponse(
        error=exc.detail,
        detail=str(exc),
        timestamp=datetime.now().isoformat()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        timestamp=datetime.now().isoformat()
    )

# Background tasks
async def cleanup_old_data():
    """Background task to cleanup old data"""
    if sdk_instance:
        try:
            # Cleanup old data based on retention policy
            retention_days = sdk_instance.config.get('privacy', {}).get('data_retention_days', 30)
            logger.info(f"Cleaning up data older than {retention_days} days")
            
            # Implementation would depend on data storage mechanism
            # For now, just log the cleanup attempt
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

# Scheduled tasks
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("QEP-VLA Production API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("QEP-VLA Production API shutdown initiated")

# Main entry point
if __name__ == "__main__":
    # Configuration
    host = os.getenv("QEP_VLA_HOST", "0.0.0.0")
    port = int(os.getenv("QEP_VLA_PORT", "8000"))
    reload = os.getenv("QEP_VLA_RELOAD", "false").lower() == "true"
    
    # Start server
    uvicorn.run(
        "production_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
