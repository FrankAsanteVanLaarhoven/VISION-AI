"""
Production-ready PVLA Navigation API
FastAPI-based REST API for PVLA Navigation System integration
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import numpy as np
import torch
import uvicorn
from contextlib import asynccontextmanager

# Import PVLA components
from core.pvla_navigation_system import PVLANavigationSystem, PVLAConfig
from core.unified_qep_vla_system import UnifiedQEPVLASystem, UnifiedSystemConfig, NavigationRequest, NavigationResponse
from config.settings import get_settings

settings = get_settings()

# Security
security = HTTPBearer()

# Global system instances
pvla_system: Optional[PVLANavigationSystem] = None
unified_system: Optional[UnifiedQEPVLASystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global pvla_system, unified_system
    
    # Startup
    logging.info("Initializing QEP-VLA Navigation Systems...")
    try:
        # Initialize unified QEP-VLA system
        unified_config = UnifiedSystemConfig()
        unified_system = UnifiedQEPVLASystem(unified_config)
        logging.info("Unified QEP-VLA System initialized successfully")
        
        # Initialize legacy PVLA system for backward compatibility
        pvla_system = PVLANavigationSystem()
        logging.info("Legacy PVLA Navigation System initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize systems: {e}")
        raise
    
    yield
    
    # Shutdown
    if unified_system:
        await unified_system.shutdown()
        logging.info("Unified QEP-VLA System shutdown complete")
    
    if pvla_system:
        pvla_system.shutdown()
        logging.info("Legacy PVLA Navigation System shutdown complete")

# FastAPI application
app = FastAPI(
    title="PVLA Navigation API",
    description="Privacy-Preserving Vision-Language-Action Navigation System API",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
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

# Pydantic models
class CameraFrame(BaseModel):
    """Camera frame data model"""
    frame_data: List[List[List[int]]] = Field(..., description="RGB frame data as nested lists")
    width: int = Field(..., description="Frame width")
    height: int = Field(..., description="Frame height")
    timestamp: float = Field(default_factory=time.time, description="Frame timestamp")
    
    @validator('frame_data')
    def validate_frame_data(cls, v):
        if not v or not v[0] or not v[0][0]:
            raise ValueError("Invalid frame data")
        return v

class LanguageCommand(BaseModel):
    """Language command data model"""
    command: str = Field(..., description="Natural language navigation command")
    confidence: Optional[float] = Field(None, description="Command confidence score")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

class NavigationContext(BaseModel):
    """Navigation context data model"""
    current_position: List[float] = Field(default=[0.0, 0.0, 0.0], description="Current position [x, y, z]")
    current_orientation: List[float] = Field(default=[0.0, 0.0, 0.0], description="Current orientation [roll, pitch, yaw]")
    target_position: List[float] = Field(default=[0.0, 0.0, 0.0], description="Target position [x, y, z]")
    environment_data: Dict[str, Any] = Field(default_factory=dict, description="Environment sensor data")
    safety_constraints: Dict[str, Any] = Field(default_factory=dict, description="Safety constraints")
    objectives: List[str] = Field(default_factory=list, description="Navigation objectives")

class NavigationRequest(BaseModel):
    """Complete navigation request model"""
    camera_frame: CameraFrame
    language_command: LanguageCommand
    navigation_context: NavigationContext
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    priority: int = Field(default=1, description="Request priority (1-10)")

class NavigationResponse(BaseModel):
    """Navigation response model"""
    request_id: Optional[str] = None
    navigation_action: int = Field(..., description="Selected navigation action index")
    explanation: str = Field(..., description="Human-readable explanation of the decision")
    confidence_score: float = Field(..., description="Confidence score (0.0-1.0)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    vision_metadata: Dict[str, Any] = Field(..., description="Vision processing metadata")
    language_metadata: Dict[str, Any] = Field(..., description="Language processing metadata")
    action_metadata: Dict[str, Any] = Field(..., description="Action selection metadata")
    adaptation_metadata: Optional[Dict[str, Any]] = Field(None, description="Meta-learning adaptation metadata")
    privacy_metadata: Optional[Dict[str, Any]] = Field(None, description="Privacy monitoring metadata")
    system_state: str = Field(..., description="Current system state")
    timestamp: float = Field(..., description="Response timestamp")

class SystemStatusResponse(BaseModel):
    """System status response model"""
    system_state: str = Field(..., description="Current system state")
    device: str = Field(..., description="Computing device")
    system_metrics: Dict[str, Any] = Field(..., description="System performance metrics")
    component_health: Dict[str, Any] = Field(..., description="Component health status")
    configuration: Dict[str, Any] = Field(..., description="System configuration")
    timestamp: float = Field(..., description="Status timestamp")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Health status")
    timestamp: float = Field(..., description="Check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")

# Utility functions
def convert_frame_to_numpy(frame_data: CameraFrame) -> np.ndarray:
    """Convert camera frame data to numpy array"""
    frame_array = np.array(frame_data.frame_data, dtype=np.uint8)
    return frame_array

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authentication dependency (placeholder)"""
    # In production, implement proper JWT validation
    return {"user_id": "default_user", "permissions": ["navigation"]}

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "PVLA Navigation API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if not pvla_system:
            return HealthCheckResponse(
                status="unhealthy",
                timestamp=time.time(),
                details={"error": "PVLA system not initialized"}
            )
        
        system_status = pvla_system.get_system_status()
        
        return HealthCheckResponse(
            status="healthy" if system_status['system_state'] == 'ready' else "degraded",
            timestamp=time.time(),
            details=system_status
        )
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=time.time(),
            details={"error": str(e)}
        )

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status(user: dict = Depends(get_current_user)):
    """Get comprehensive system status"""
    if not pvla_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PVLA system not available"
        )
    
    try:
        system_status = pvla_system.get_system_status()
        return SystemStatusResponse(**system_status)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )

@app.post("/navigate", response_model=NavigationResponse)
async def navigate(
    request: NavigationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Process navigation request through PVLA system
    
    This endpoint processes a complete navigation request including:
    - Privacy-preserving vision processing
    - Quantum-enhanced language understanding
    - Consciousness-driven action selection
    - Meta-learning adaptation
    - Privacy monitoring
    """
    if not pvla_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PVLA system not available"
        )
    
    try:
        # Convert camera frame to numpy array
        camera_frame = convert_frame_to_numpy(request.camera_frame)
        
        # Prepare navigation context
        navigation_context = {
            'context': request.navigation_context.current_position + request.navigation_context.current_orientation,
            'objectives': [1.0 if obj in request.navigation_context.objectives else 0.0 for obj in [
                'move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop',
                'accelerate', 'decelerate', 'avoid_obstacle', 'follow_path', 'reach_destination'
            ]],
            'goals': request.navigation_context.target_position + [0.0] * 253,  # Pad to 256
            'environment': list(request.navigation_context.environment_data.values())[:128] + [0.0] * (128 - len(request.navigation_context.environment_data)),
            'context': list(request.navigation_context.safety_constraints.values())[:128] + [0.0] * (128 - len(request.navigation_context.safety_constraints))
        }
        
        # Process navigation request
        result = await pvla_system.process_navigation_request(
            camera_frame=camera_frame,
            language_command=request.language_command.command,
            navigation_context=navigation_context
        )
        
        # Add request ID to result
        result['request_id'] = request.request_id
        
        # Schedule background tasks
        background_tasks.add_task(log_navigation_request, request, result)
        
        return NavigationResponse(**result)
        
    except Exception as e:
        logging.error(f"Navigation request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Navigation processing failed: {str(e)}"
        )

@app.post("/navigate/batch", response_model=List[NavigationResponse])
async def navigate_batch(
    requests: List[NavigationRequest],
    user: dict = Depends(get_current_user)
):
    """Process multiple navigation requests in batch"""
    if not pvla_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PVLA system not available"
        )
    
    try:
        # Process requests concurrently
        tasks = []
        for request in requests:
            task = process_single_navigation_request(request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Batch request {i} failed: {result}")
                responses.append(NavigationResponse(
                    request_id=requests[i].request_id,
                    navigation_action=0,  # Default to stop action
                    explanation="Error in processing",
                    confidence_score=0.0,
                    processing_time_ms=0.0,
                    vision_metadata={},
                    language_metadata={},
                    action_metadata={},
                    system_state="error",
                    timestamp=time.time()
                ))
            else:
                responses.append(NavigationResponse(**result))
        
        return responses
        
    except Exception as e:
        logging.error(f"Batch navigation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch navigation processing failed: {str(e)}"
        )

async def process_single_navigation_request(request: NavigationRequest) -> Dict[str, Any]:
    """Process a single navigation request"""
    camera_frame = convert_frame_to_numpy(request.camera_frame)
    
    navigation_context = {
        'context': request.navigation_context.current_position + request.navigation_context.current_orientation,
        'objectives': [1.0 if obj in request.navigation_context.objectives else 0.0 for obj in [
            'move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop',
            'accelerate', 'decelerate', 'avoid_obstacle', 'follow_path', 'reach_destination'
        ]],
        'goals': request.navigation_context.target_position + [0.0] * 253,
        'environment': list(request.navigation_context.environment_data.values())[:128] + [0.0] * (128 - len(request.navigation_context.environment_data)),
        'context': list(request.navigation_context.safety_constraints.values())[:128] + [0.0] * (128 - len(request.navigation_context.safety_constraints))
    }
    
    result = await pvla_system.process_navigation_request(
        camera_frame=camera_frame,
        language_command=request.language_command.command,
        navigation_context=navigation_context
    )
    
    result['request_id'] = request.request_id
    return result

@app.post("/system/reset")
async def reset_system_metrics(user: dict = Depends(get_current_user)):
    """Reset system performance metrics"""
    if not pvla_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PVLA system not available"
        )
    
    try:
        pvla_system.reset_system_metrics()
        return {"message": "System metrics reset successfully", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset metrics: {str(e)}"
        )

@app.post("/system/update-state")
async def update_navigation_state(
    position: List[float],
    orientation: List[float],
    user: dict = Depends(get_current_user)
):
    """Update current navigation state"""
    if not pvla_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PVLA system not available"
        )
    
    try:
        new_state = torch.tensor(position + orientation, device=pvla_system.device)
        pvla_system.update_navigation_state(new_state)
        return {"message": "Navigation state updated successfully", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update navigation state: {str(e)}"
        )

@app.post("/system/update-objectives")
async def update_navigation_objectives(
    objectives: List[float],
    user: dict = Depends(get_current_user)
):
    """Update navigation objectives"""
    if not pvla_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PVLA system not available"
        )
    
    try:
        new_objectives = torch.tensor(objectives, device=pvla_system.device)
        pvla_system.update_navigation_objectives(new_objectives)
        return {"message": "Navigation objectives updated successfully", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update navigation objectives: {str(e)}"
        )

# Unified QEP-VLA System Endpoints

@app.post("/api/v2/navigate", response_model=Dict[str, Any])
async def unified_navigate(
    request: NavigationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Unified QEP-VLA navigation endpoint with all Bo-Wei enhancements
    """
    if not unified_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unified QEP-VLA system not available"
        )
    
    try:
        # Process navigation request
        response = await unified_system.process_navigation_request(request)
        
        # Log request in background
        background_tasks.add_task(log_unified_navigation_request, request, response)
        
        return {
            "navigation_action": response.navigation_action,
            "confidence_score": response.confidence_score,
            "processing_time_ms": response.processing_time_ms,
            "privacy_guarantee": response.privacy_guarantee,
            "quantum_enhanced": response.quantum_enhanced,
            "explanation": response.explanation,
            "position_estimate": response.position_estimate,
            "performance_metrics": response.performance_metrics,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Navigation processing failed: {str(e)}"
        )

@app.get("/api/v2/system/status")
async def unified_system_status():
    """Get unified system status and health"""
    if not unified_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unified QEP-VLA system not available"
        )
    
    try:
        health_status = unified_system.health_check()
        return health_status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )

@app.get("/api/v2/system/metrics")
async def unified_system_metrics():
    """Get comprehensive system metrics"""
    if not unified_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unified QEP-VLA system not available"
        )
    
    try:
        metrics = unified_system.get_system_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )

@app.post("/api/v2/system/reset")
async def reset_unified_system():
    """Reset unified system metrics"""
    if not unified_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unified QEP-VLA system not available"
        )
    
    try:
        unified_system.reset_metrics()
        return {"message": "Unified system metrics reset successfully", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset system metrics: {str(e)}"
        )

# Background tasks
async def log_navigation_request(request: NavigationRequest, result: Dict[str, Any]):
    """Log navigation request for analysis"""
    logging.info(f"Navigation request {request.request_id} processed: "
                f"action={result['navigation_action']}, "
                f"confidence={result['confidence_score']:.3f}, "
                f"time={result['processing_time_ms']:.2f}ms")

async def log_unified_navigation_request(request: NavigationRequest, response: NavigationResponse):
    """Log unified navigation request for analysis"""
    logging.info(f"Unified navigation request processed: "
                f"action={response.navigation_action}, "
                f"confidence={response.confidence_score:.3f}, "
                f"time={response.processing_time_ms:.2f}ms, "
                f"quantum_enhanced={response.quantum_enhanced}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logging.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(
        "pvla_api:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
