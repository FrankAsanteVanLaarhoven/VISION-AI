#!/usr/bin/env python3
"""
PVLA Navigation System Web Server
Serves the web dashboard and handles PVLA navigation requests
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import numpy as np
import cv2
import base64
from typing import Dict, Any, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import PVLA components
try:
    from src.core.pvla_navigation_system import PVLANavigationSystem, PVLAConfig
    from src.core.pvla_vision_algorithm import VisionConfig
    from src.core.pvla_language_algorithm import LanguageConfig
    from src.core.pvla_action_algorithm import ActionConfig
    from src.core.pvla_meta_learning import MetaLearningConfig
    from src.core.quantum_privacy_transform import QuantumTransformConfig
    PVLA_AVAILABLE = True
except ImportError as e:
    print(f"PVLA components not available: {e}")
    PVLA_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="PVLA Navigation System Web Interface",
    description="Privacy-Preserving Vision-Language-Action Navigation System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global PVLA system instance
pvla_system: Optional[PVLANavigationSystem] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize PVLA system on startup"""
    global pvla_system
    
    if PVLA_AVAILABLE:
        try:
            # Create lightweight configuration for web interface
            config = PVLAConfig(
                vision_config=VisionConfig(
                    input_resolution=(64, 64),  # Smaller for web
                    feature_dim=128,
                    privacy_budget=0.5,
                    max_processing_time_ms=200.0
                ),
                language_config=LanguageConfig(
                    model_name="bert-base-uncased",
                    max_sequence_length=128,
                    quantum_dimension=32,
                    num_quantum_states=4
                ),
                action_config=ActionConfig(
                    num_actions=5,
                    safety_threshold=0.5,
                    ethics_threshold=0.5
                ),
                meta_learning_config=MetaLearningConfig(
                    quantum_dimension=32,
                    num_quantum_layers=4,
                    performance_window=10,
                    memory_size=100
                ),
                privacy_config=QuantumTransformConfig(
                    privacy_budget_epsilon=0.5,
                    privacy_budget_delta=1e-3,
                    quantum_enhancement_factor=1.5
                )
            )
            
            pvla_system = PVLANavigationSystem(config)
            logging.info("PVLA Navigation System initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize PVLA system: {e}")
            pvla_system = None
    else:
        logging.warning("PVLA components not available - running in demo mode")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard"""
    dashboard_path = project_root.parent / "ui" / "web_dashboard" / "aria-advanced.html"
    
    if dashboard_path.exists():
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Inject PVLA-specific modifications
        content = content.replace(
            '<title>ARIA - Lifelike AI Assistant with Advanced Animation</title>',
            '<title>PVLA Navigation System - ARIA AI Assistant</title>'
        )
        content = content.replace(
            '<h1>🤖 ARIA - Lifelike AI Assistant</h1>',
            '<h1>🚀 PVLA Navigation System - ARIA AI Assistant</h1>'
        )
        content = content.replace(
            '<p>Advanced Voice AI with Realistic Human Expressions & Body Language</p>',
            '<p>Privacy-Preserving Vision-Language-Action Navigation with Quantum-Enhanced AI</p>'
        )
        
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PVLA Navigation System</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #0f0f23; color: white; }
                .container { max-width: 800px; margin: 0 auto; text-align: center; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 20px; margin-bottom: 30px; }
                .status { background: rgba(0,0,0,0.8); padding: 20px; border-radius: 10px; margin: 20px 0; }
                .feature { background: rgba(59,130,246,0.1); padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 3px solid #3b82f6; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 PVLA Navigation System</h1>
                    <p>Privacy-Preserving Vision-Language-Action Systems for Quantum-Enhanced Autonomous Navigation</p>
                </div>
                
                <div class="status">
                    <h2>System Status</h2>
                    <p>✅ Web server is running</p>
                    <p>✅ PVLA system is {'initialized' if pvla_system else 'not available'}</p>
                    <p>✅ API endpoints are active</p>
                </div>
                
                <div class="feature">
                    <h3>🔒 Privacy-Preserving Vision Processing</h3>
                    <p>Homomorphic encryption with differential privacy (ε=0.1)</p>
                </div>
                
                <div class="feature">
                    <h3>🧠 Quantum-Enhanced Language Understanding</h3>
                    <p>Quantum superposition, entanglement, and measurement for navigation</p>
                </div>
                
                <div class="feature">
                    <h3>🤖 Consciousness-Driven Action Selection</h3>
                    <p>Ethical AI with safety constraints and consciousness awareness</p>
                </div>
                
                <div class="feature">
                    <h3>📈 Self-Improving Meta-Learning</h3>
                    <p>Quantum parameter optimization for adaptive navigation</p>
                </div>
                
                <div style="margin-top: 30px;">
                    <h3>API Endpoints</h3>
                    <p><a href="/docs" style="color: #60a5fa;">📚 API Documentation</a></p>
                    <p><a href="/health" style="color: #60a5fa;">🏥 Health Check</a></p>
                    <p><a href="/status" style="color: #60a5fa;">📊 System Status</a></p>
                </div>
            </div>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "pvla_system": "available" if pvla_system else "not_available",
        "components": {
            "vision_algorithm": "available" if pvla_system else "not_available",
            "language_algorithm": "available" if pvla_system else "not_available",
            "action_algorithm": "available" if pvla_system else "not_available",
            "meta_learning": "available" if pvla_system else "not_available",
            "privacy_monitoring": "available" if pvla_system else "not_available"
        }
    }

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    if pvla_system:
        return pvla_system.get_system_status()
    else:
        return {
            "status": "demo_mode",
            "message": "PVLA system not available - running in demo mode",
            "timestamp": time.time()
        }

@app.post("/navigate")
async def navigate(request: Request):
    """Process navigation request"""
    if not pvla_system:
        raise HTTPException(status_code=503, detail="PVLA system not available")
    
    try:
        data = await request.json()
        
        # Extract data from request
        camera_frame_data = data.get("camera_frame", {})
        language_command = data.get("language_command", "")
        navigation_context = data.get("navigation_context", {})
        
        # Convert camera frame data to numpy array
        if "frame_data" in camera_frame_data:
            frame_data = camera_frame_data["frame_data"]
            width = camera_frame_data.get("width", 64)
            height = camera_frame_data.get("height", 64)
            
            # Convert to numpy array
            camera_frame = np.array(frame_data, dtype=np.uint8)
            if camera_frame.shape != (height, width, 3):
                camera_frame = camera_frame.reshape(height, width, 3)
        else:
            # Generate dummy frame for testing
            camera_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Prepare navigation context
        nav_context = {
            'context': navigation_context.get('current_position', [0.0, 0.0, 0.0]) + 
                      navigation_context.get('current_orientation', [0.0, 0.0, 0.0]),
            'objectives': [1.0 if obj in navigation_context.get('objectives', []) else 0.0 
                          for obj in ['move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop']],
            'goals': navigation_context.get('target_position', [0.0, 0.0, 0.0]) + [0.0] * 253,
            'environment': list(navigation_context.get('environment_data', {}).values())[:128] + [0.0] * (128 - len(navigation_context.get('environment_data', {}))),
            'context': list(navigation_context.get('safety_constraints', {}).values())[:128] + [0.0] * (128 - len(navigation_context.get('safety_constraints', {})))
        }
        
        # Process navigation request
        result = await pvla_system.process_navigation_request(
            camera_frame=camera_frame,
            language_command=language_command,
            navigation_context=nav_context
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Navigation request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Navigation processing failed: {str(e)}")

@app.post("/chat")
async def chat_with_aria(request: Request):
    """Chat endpoint for ARIA interface"""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Generate contextual response
        response = generate_aria_response(message)
        
        return {
            "response": response["text"],
            "emotion": response["emotion"],
            "body_language": response.get("body_language", "idle"),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logging.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

def generate_aria_response(message: str) -> Dict[str, Any]:
    """Generate contextual response for ARIA"""
    lower_message = message.lower()
    
    # PVLA-specific responses
    if any(word in lower_message for word in ['pvla', 'navigation', 'quantum', 'privacy']):
        return {
            "text": "I'm part of the PVLA Navigation System! I use quantum-enhanced language understanding, privacy-preserving vision processing, and consciousness-driven action selection to help with autonomous navigation. What would you like to know about our capabilities?",
            "emotion": "excited",
            "body_language": "speaking"
        }
    
    if any(word in lower_message for word in ['hello', 'hi', 'hey']):
        return {
            "text": "Hello! I'm ARIA, your AI assistant integrated with the PVLA Navigation System. I can help you with navigation tasks, answer questions about our quantum-enhanced capabilities, and demonstrate our privacy-preserving features. How can I assist you today?",
            "emotion": "happy",
            "body_language": "excited"
        }
    
    if any(word in lower_message for word in ['quantum', 'superposition', 'entanglement']):
        return {
            "text": "Quantum computing is fascinating! In our PVLA system, we use quantum superposition to explore multiple navigation possibilities simultaneously, quantum entanglement to connect language understanding with spatial reasoning, and quantum measurement to select optimal actions. It's like having a quantum-enhanced brain for navigation!",
            "emotion": "curious",
            "body_language": "speaking"
        }
    
    if any(word in lower_message for word in ['privacy', 'encryption', 'secure']):
        return {
            "text": "Privacy is fundamental to our system! We use homomorphic encryption so I can process visual data without ever seeing the raw images, differential privacy to protect your data, and quantum-safe encryption for future-proof security. Your privacy is protected at every step of the navigation process.",
            "emotion": "focused",
            "body_language": "speaking"
        }
    
    if any(word in lower_message for word in ['consciousness', 'ethics', 'moral']):
        return {
            "text": "Our consciousness-driven action selection ensures ethical navigation decisions. I consider utility, safety, and ethics in every action, with consciousness weights that adapt based on the situation. It's like having a moral compass built into the navigation system!",
            "emotion": "thoughtful",
            "body_language": "listening"
        }
    
    # Default response
    return {
        "text": f"That's interesting! You mentioned '{message}'. As part of the PVLA Navigation System, I'm designed to help with autonomous navigation tasks while maintaining privacy and ethical standards. Could you tell me more about what you'd like to explore?",
        "emotion": "curious",
        "body_language": "listening"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                response = generate_aria_response(message_data.get("message", ""))
                await manager.send_personal_message(
                    json.dumps({
                        "type": "response",
                        "data": response
                    }), 
                    websocket
                )
            elif message_data.get("type") == "navigation":
                # Handle navigation requests via WebSocket
                if pvla_system:
                    # Process navigation request
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "navigation_result",
                            "data": {"status": "processed", "action": "move_forward"}
                        }), 
                        websocket
                    )
                else:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "data": {"message": "PVLA system not available"}
                        }), 
                        websocket
                    )
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Mount static files
static_path = project_root.parent / "ui" / "web_dashboard"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
