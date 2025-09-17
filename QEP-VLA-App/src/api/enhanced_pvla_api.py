"""
Enhanced PVLA API with Bo-Wei Technologies Integration
Provides API endpoints for AI Reality Comprehension, Human-Robot Supply Chain Integration, and Safety & Privacy Asset Protection
"""

import time
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import enhanced unified system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.enhanced_unified_qep_vla_system import EnhancedUnifiedQEPVLASystem, EnhancedUnifiedSystemConfig
from core.ai_reality_comprehension import RealityComprehensionEngine, RealityComprehensionConfig
from core.human_robot_supply_chain import HumanRobotSupplyChainIntegration, SupplyChainConfig
from core.safety_privacy_protection import SafetyPrivacyAssetProtection, SafetyPrivacyConfig

# Import existing models
from core.unified_qep_vla_system import NavigationRequest, NavigationResponse

# Import enhanced features
from core.reality_aware_pvla_navigation import RealityAwarePVLANavigation
from core.safety_enhanced_federated_trainer import SafetyEnhancedFederatedTrainer

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced QEP-VLA Platform API",
    description="Quantum-Enhanced Privacy-preserving Vision-Language-Action Platform with Bo-Wei Technologies",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Global system instances
enhanced_system: Optional[EnhancedUnifiedQEPVLASystem] = None
reality_comprehension: Optional[RealityComprehensionEngine] = None
supply_chain_integration: Optional[HumanRobotSupplyChainIntegration] = None
safety_privacy_protection: Optional[SafetyPrivacyAssetProtection] = None
reality_aware_navigation: Optional[RealityAwarePVLANavigation] = None
safety_enhanced_trainer: Optional[SafetyEnhancedFederatedTrainer] = None

# Pydantic models for new endpoints
class RealityComprehensionRequest(BaseModel):
    """Request model for reality comprehension"""
    sensors: Dict[str, Any] = Field(..., description="Multi-modal sensor data")
    network: Dict[str, Any] = Field(..., description="Network state data")
    language: str = Field(..., description="Language input")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context data")

class SupplyChainIntegrationRequest(BaseModel):
    """Request model for supply chain integration"""
    human_agents: Dict[str, Any] = Field(..., description="Human agents data")
    robot_agents: Dict[str, Any] = Field(..., description="Robot agents data")
    environment_state: Dict[str, Any] = Field(..., description="Environment state")

class SafetyPrivacyProtectionRequest(BaseModel):
    """Request model for safety and privacy protection"""
    system_state: Dict[str, Any] = Field(..., description="System state data")
    privacy_requirements: Dict[str, Any] = Field(..., description="Privacy requirements")

class EnhancedNavigationRequest(BaseModel):
    """Enhanced navigation request with Bo-Wei technologies"""
    start_position: List[float] = Field(..., description="Start position [x, y, z]")
    target_position: List[float] = Field(..., description="Target position [x, y, z]")
    language_command: str = Field(..., description="Natural language navigation command")
    sensor_data: Dict[str, Any] = Field(default_factory=dict, description="Sensor data")
    network_state: Dict[str, Any] = Field(default_factory=dict, description="Network state")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context data including human/robot agents")
    privacy_requirements: Dict[str, Any] = Field(default_factory=dict, description="Privacy requirements")
    performance_requirements: Dict[str, Any] = Field(default_factory=dict, description="Performance requirements")

class SystemStatusResponse(BaseModel):
    """System status response"""
    status: str
    timestamp: float
    enhanced_features: Dict[str, Any]
    bo_wei_technologies: Dict[str, Any]

class SystemMetricsResponse(BaseModel):
    """System metrics response"""
    enhanced_metrics: Dict[str, Any]
    bo_wei_metrics: Dict[str, Any]
    performance_trends: Dict[str, Any]

# App lifespan management
@app.on_event("startup")
async def startup_event():
    """Initialize enhanced system on startup"""
    global enhanced_system, reality_comprehension, supply_chain_integration, safety_privacy_protection
    
    try:
        # Initialize enhanced unified system
        enhanced_config = EnhancedUnifiedSystemConfig(
            privacy_budget=0.1,
            quantum_enhancement=True,
            blockchain_validation=True,
            reality_comprehension_enabled=True,
            human_robot_integration_enabled=True,
            safety_privacy_protection_enabled=True
        )
        enhanced_system = EnhancedUnifiedQEPVLASystem(enhanced_config)
        
        # Initialize individual Bo-Wei technologies
        reality_config = RealityComprehensionConfig(quantum_enhancement=True)
        reality_comprehension = RealityComprehensionEngine(reality_config)
        
        supply_chain_config = SupplyChainConfig(
            privacy_protection_level='high',
            emergency_response_enabled=True,
            compliance_monitoring=True
        )
        supply_chain_integration = HumanRobotSupplyChainIntegration(supply_chain_config)
        
        safety_privacy_config = SafetyPrivacyConfig(
            privacy_budget=0.1,
            quantum_enhancement=True,
            blockchain_validation=True
        )
        safety_privacy_protection = SafetyPrivacyAssetProtection(safety_privacy_config)
        
        # Initialize enhanced features
        reality_aware_navigation = RealityAwarePVLANavigation()
        safety_enhanced_trainer = SafetyEnhancedFederatedTrainer()
        
        logging.info("Enhanced QEP-VLA system initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize enhanced system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global enhanced_system, reality_comprehension, supply_chain_integration, safety_privacy_protection
    
    try:
        if enhanced_system:
            del enhanced_system
        if reality_comprehension:
            del reality_comprehension
        if supply_chain_integration:
            del supply_chain_integration
        if safety_privacy_protection:
            del safety_privacy_protection
        
        logging.info("Enhanced QEP-VLA system shutdown complete")
        
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "enhanced_system": "available",
        "bo_wei_technologies": "integrated"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all components"""
    if not enhanced_system:
        raise HTTPException(status_code=503, detail="Enhanced system not initialized")
    
    try:
        health_status = enhanced_system.health_check()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Enhanced navigation endpoints
@app.post("/api/v3/navigate", response_model=Dict[str, Any])
async def enhanced_navigate(request: EnhancedNavigationRequest, background_tasks: BackgroundTasks):
    """Enhanced navigation with all Bo-Wei technologies"""
    if not enhanced_system:
        raise HTTPException(status_code=503, detail="Enhanced system not initialized")
    
    try:
        # Convert to NavigationRequest
        nav_request = NavigationRequest(
            start_position=request.start_position,
            target_position=request.target_position,
            language_command=request.language_command,
            sensor_data=request.sensor_data,
            network_state=request.network_state,
            context_data=request.context_data,
            privacy_requirements=request.privacy_requirements,
            performance_requirements=request.performance_requirements
        )
        
        # Process enhanced navigation
        response = enhanced_system.process_enhanced_navigation_request(nav_request)
        
        # Log request in background
        background_tasks.add_task(log_enhanced_navigation_request, request, response)
        
        return {
            "success": response.success,
            "path": response.path,
            "confidence": response.confidence,
            "processing_time_ms": response.processing_time_ms,
            "privacy_score": response.privacy_score,
            "quantum_enhancement_factor": response.quantum_enhancement_factor,
            "metadata": response.metadata,
            "error": response.error
        }
        
    except Exception as e:
        logging.error(f"Enhanced navigation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Navigation failed: {str(e)}")

# Bo-Wei Technology Endpoints

# AI Reality Comprehension endpoints
@app.post("/api/v3/reality/comprehend")
async def comprehend_reality(request: RealityComprehensionRequest):
    """Comprehend reality using AI Reality Comprehension Engine"""
    if not reality_comprehension:
        raise HTTPException(status_code=503, detail="Reality comprehension not available")
    
    try:
        multi_modal_input = {
            'sensors': request.sensors,
            'network': request.network,
            'language': request.language,
            'context': request.context
        }
        
        result = reality_comprehension.comprehend_reality(multi_modal_input)
        
        return {
            "success": 'error' not in result,
            "reality_model": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Reality comprehension failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reality comprehension failed: {str(e)}")

@app.get("/api/v3/reality/status")
async def reality_comprehension_status():
    """Get reality comprehension system status"""
    if not reality_comprehension:
        raise HTTPException(status_code=503, detail="Reality comprehension not available")
    
    try:
        health_status = reality_comprehension.health_check()
        metrics = reality_comprehension.get_performance_metrics()
        
        return {
            "status": health_status,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Human-Robot Supply Chain Integration endpoints
@app.post("/api/v3/supply-chain/integrate")
async def integrate_supply_chain(request: SupplyChainIntegrationRequest):
    """Integrate human-robot supply chain"""
    if not supply_chain_integration:
        raise HTTPException(status_code=503, detail="Supply chain integration not available")
    
    try:
        result = supply_chain_integration.integrate_supply_chain(
            request.human_agents,
            request.robot_agents,
            request.environment_state
        )
        
        return {
            "success": 'error' not in result,
            "integration_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Supply chain integration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Supply chain integration failed: {str(e)}")

@app.get("/api/v3/supply-chain/status")
async def supply_chain_status():
    """Get supply chain integration status"""
    if not supply_chain_integration:
        raise HTTPException(status_code=503, detail="Supply chain integration not available")
    
    try:
        health_status = supply_chain_integration.health_check()
        metrics = supply_chain_integration.get_performance_metrics()
        
        return {
            "status": health_status,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Safety & Privacy Asset Protection endpoints
@app.post("/api/v3/safety-privacy/protect")
async def protect_system_assets(request: SafetyPrivacyProtectionRequest):
    """Protect system assets with safety and privacy"""
    if not safety_privacy_protection:
        raise HTTPException(status_code=503, detail="Safety privacy protection not available")
    
    try:
        result = safety_privacy_protection.protect_system_assets(
            request.system_state,
            request.privacy_requirements
        )
        
        return {
            "success": 'error' not in result,
            "protection_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Asset protection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Asset protection failed: {str(e)}")

@app.get("/api/v3/safety-privacy/status")
async def safety_privacy_status():
    """Get safety and privacy protection status"""
    if not safety_privacy_protection:
        raise HTTPException(status_code=503, detail="Safety privacy protection not available")
    
    try:
        health_status = safety_privacy_protection.health_check()
        metrics = safety_privacy_protection.get_performance_metrics()
        
        return {
            "status": health_status,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# System status and metrics endpoints
@app.get("/api/v3/system/status", response_model=SystemStatusResponse)
async def get_enhanced_system_status():
    """Get enhanced system status with all Bo-Wei technologies"""
    if not enhanced_system:
        raise HTTPException(status_code=503, detail="Enhanced system not initialized")
    
    try:
        status = enhanced_system.get_enhanced_system_status()
        
        return SystemStatusResponse(
            status=status.get('status', 'unknown'),
            timestamp=time.time(),
            enhanced_features=status.get('enhanced_processing_metrics', {}),
            bo_wei_technologies=status.get('bo_wei_technologies', {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@app.get("/api/v3/system/metrics", response_model=SystemMetricsResponse)
async def get_enhanced_system_metrics():
    """Get enhanced system metrics with all Bo-Wei technologies"""
    if not enhanced_system:
        raise HTTPException(status_code=503, detail="Enhanced system not initialized")
    
    try:
        metrics = enhanced_system.get_enhanced_system_metrics()
        
        return SystemMetricsResponse(
            enhanced_metrics=metrics.get('enhanced_processing_metrics', {}),
            bo_wei_metrics={
                'reality_comprehension': metrics.get('reality_comprehension_metrics', {}),
                'supply_chain_integration': metrics.get('supply_chain_metrics', {}),
                'safety_privacy_protection': metrics.get('safety_privacy_metrics', {})
            },
            performance_trends={
                'trend': metrics.get('enhanced_processing_metrics', {}).get('performance_trend', 'unknown'),
                'total_requests': metrics.get('enhanced_processing_metrics', {}).get('total_enhanced_requests', 0)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.post("/api/v3/system/reset")
async def reset_enhanced_system_metrics():
    """Reset enhanced system metrics"""
    if not enhanced_system:
        raise HTTPException(status_code=503, detail="Enhanced system not initialized")
    
    try:
        enhanced_system.reset_enhanced_metrics()
        
        return {
            "success": True,
            "message": "Enhanced system metrics reset successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# Background task functions
async def log_enhanced_navigation_request(request: EnhancedNavigationRequest, response: NavigationResponse):
    """Log enhanced navigation request in background"""
    try:
        logging.info(f"Enhanced navigation request processed: {request.language_command}")
        logging.info(f"Response success: {response.success}, Processing time: {response.processing_time_ms}ms")
        
        # Log Bo-Wei technology usage
        if response.metadata:
            bo_wei_tech = response.metadata.get('bo_wei_technologies_active', [])
            if bo_wei_tech:
                logging.info(f"Bo-Wei technologies used: {', '.join(bo_wei_tech)}")
        
    except Exception as e:
        logging.error(f"Failed to log enhanced navigation request: {e}")

# Enhanced Feature Endpoints

# Reality-Aware Navigation
@app.post("/api/v3/navigation/comprehensive")
async def comprehensive_navigation(request: Dict[str, Any]):
    """Navigate with comprehensive reality awareness"""
    if not reality_aware_navigation:
        raise HTTPException(status_code=503, detail="Reality-aware navigation not available")
    
    try:
        result = reality_aware_navigation.navigate_with_comprehensive_awareness(request)
        
        return {
            "success": result.get('success', False),
            "navigation_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Comprehensive navigation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Navigation failed: {str(e)}")

@app.get("/api/v3/navigation/status")
async def navigation_status():
    """Get reality-aware navigation status"""
    if not reality_aware_navigation:
        raise HTTPException(status_code=503, detail="Reality-aware navigation not available")
    
    try:
        status = reality_aware_navigation.get_enhanced_navigation_status()
        
        return {
            "status": status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Safety-Enhanced Federated Training
@app.post("/api/v3/training/safety-enhanced")
async def safety_enhanced_training(request: Dict[str, Any]):
    """Train with safety and privacy protection"""
    if not safety_enhanced_trainer:
        raise HTTPException(status_code=503, detail="Safety-enhanced trainer not available")
    
    try:
        training_data = request.get('training_data', {})
        human_agents = request.get('human_agents', {})
        robot_agents = request.get('robot_agents', {})
        
        result = safety_enhanced_trainer.train_with_safety_privacy(
            training_data, human_agents, robot_agents
        )
        
        return {
            "success": result.get('success', False),
            "training_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Safety-enhanced training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/v3/training/status")
async def training_status():
    """Get safety-enhanced training status"""
    if not safety_enhanced_trainer:
        raise HTTPException(status_code=503, detail="Safety-enhanced trainer not available")
    
    try:
        status = safety_enhanced_trainer.get_enhanced_training_status()
        
        return {
            "status": status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Enhanced Human Safety Monitoring
@app.post("/api/v3/human-safety/monitor")
async def monitor_human_safety(request: Dict[str, Any]):
    """Monitor human safety with quantum privacy"""
    if not supply_chain_integration:
        raise HTTPException(status_code=503, detail="Supply chain integration not available")
    
    try:
        human_agents = request.get('human_agents', {})
        
        result = supply_chain_integration.human_integration.monitor_human_safety(human_agents)
        
        return {
            "success": 'error' not in result,
            "safety_data": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Human safety monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")

# Enhanced Robot Safety Coordination
@app.post("/api/v3/robot-safety/coordinate")
async def coordinate_robot_safety(request: Dict[str, Any]):
    """Coordinate robot safety with human awareness"""
    if not supply_chain_integration:
        raise HTTPException(status_code=503, detail="Supply chain integration not available")
    
    try:
        robot_agents = request.get('robot_agents', {})
        human_agents = request.get('human_agents', {})
        
        result = supply_chain_integration.robot_integration.coordinate_robot_safety(
            robot_agents, human_agents
        )
        
        return {
            "success": 'error' not in result,
            "coordination_data": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Robot safety coordination failed: {e}")
        raise HTTPException(status_code=500, detail=f"Coordination failed: {str(e)}")

# Comprehensive Asset Tracking
@app.post("/api/v3/assets/track-all")
async def track_all_assets(request: Dict[str, Any]):
    """Track all assets including humans, robots, and materials"""
    if not safety_privacy_protection:
        raise HTTPException(status_code=503, detail="Safety privacy protection not available")
    
    try:
        humans = request.get('humans', {})
        robots = request.get('robots', {})
        materials = request.get('materials', {})
        
        result = safety_privacy_protection.asset_tracking.track_all_assets(
            humans, robots, materials
        )
        
        return {
            "success": 'error' not in result,
            "tracking_data": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Asset tracking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tracking failed: {str(e)}")

# Demo endpoints for testing
@app.get("/api/v3/demo/reality")
async def demo_reality_comprehension():
    """Demo reality comprehension with sample data"""
    if not reality_comprehension:
        raise HTTPException(status_code=503, detail="Reality comprehension not available")
    
    try:
        # Sample multi-modal input
        sample_input = {
            'sensors': {
                'vision': [0.1, 0.2, 0.3, 0.4, 0.5],
                'lidar': [1.0, 2.0, 3.0, 4.0, 5.0],
                'imu': [0.01, 0.02, 0.03]
            },
            'network': {
                'blockchain': {'transaction_count': 100, 'block_count': 10},
                'iot_network': {'device_count': 50, 'connection_count': 100}
            },
            'language': 'Navigate to the target location safely',
            'context': {'location': 'warehouse', 'time': 'day'}
        }
        
        result = reality_comprehension.comprehend_reality(sample_input)
        
        return {
            "success": 'error' not in result,
            "demo_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")

@app.get("/api/v3/demo/supply-chain")
async def demo_supply_chain_integration():
    """Demo supply chain integration with sample data"""
    if not supply_chain_integration:
        raise HTTPException(status_code=503, detail="Supply chain integration not available")
    
    try:
        # Sample data
        sample_humans = {
            'human_1': {
                'biometric_data': {'vital_signs': {'heart_rate': 70, 'blood_pressure': 120}},
                'activity_patterns': {'movement': 'normal'},
                'work_duration': 4.0,
                'location': {'position': [0, 0, 0]},
                'vital_signs': {'heart_rate': 70}
            }
        }
        
        sample_robots = {
            'robot_1': {
                'current_position': [1, 1, 0],
                'target_position': [2, 2, 0],
                'sensor_data': {'camera': 'active'},
                'actuator_status': {'motors': 'operational'},
                'task_completion': {'completion_rate': 0.95},
                'energy_consumption': 50.0,
                'communication_log': [],
                'access_attempts': []
            }
        }
        
        sample_environment = {
            'hazard_level': 0.2,
            'workspace_size': 100.0
        }
        
        result = supply_chain_integration.integrate_supply_chain(
            sample_humans, sample_robots, sample_environment
        )
        
        return {
            "success": 'error' not in result,
            "demo_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")

@app.get("/api/v3/demo/safety-privacy")
async def demo_safety_privacy_protection():
    """Demo safety and privacy protection with sample data"""
    if not safety_privacy_protection:
        raise HTTPException(status_code=503, detail="Safety privacy protection not available")
    
    try:
        # Sample system state
        sample_system_state = {
            'agent_data': {
                'agent_1': {'data': 'test_data', 'sensitive': True}
            },
            'human_agents': {'human_1': {'status': 'active'}},
            'robot_agents': {'robot_1': {'status': 'operational'}},
            'environment': {'hazard_level': 0.2},
            'supply_chain': {'status': 'active'},
            'assets': {'asset_1': {'metadata': {}, 'current_state': {}}},
            'transactions': [],
            'regulatory_requirements': {'req_1': {'type': 'safety'}}
        }
        
        sample_privacy_requirements = {'epsilon': 0.1}
        
        result = safety_privacy_protection.protect_system_assets(
            sample_system_state, sample_privacy_requirements
        )
        
        return {
            "success": 'error' not in result,
            "demo_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_pvla_api:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info"
    )
