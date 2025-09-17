"""
QEP-VLA Commercial SDK

A production-ready SDK for integrating quantum-enhanced privacy-preserving 
navigation into autonomous vehicle platforms.
"""

import requests
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NavigationResult:
    """
    Result object for navigation inference
    """
    action_probabilities: List[float]
    confidence_score: float
    processing_time_ms: float
    privacy_guarantee: str
    quantum_enhanced: bool
    meets_latency_requirement: bool
    model_complexity: str
    safety_fallback_triggered: bool
    
    @property
    def recommended_action(self) -> int:
        """Get the action with highest probability"""
        return int(np.argmax(self.action_probabilities))
    
    @property
    def is_safe(self) -> bool:
        """Check if the result meets safety requirements"""
        return (self.confidence_score > 0.7 and 
                self.meets_latency_requirement and 
                not self.safety_fallback_triggered)

class QEPVLANavigator:
    """
    Main SDK class for QEP-VLA Navigation System
    
    Usage:
        navigator = QEPVLANavigator(api_endpoint="http://localhost:8000")
        result = navigator.navigate(
            camera_data=camera_frame,
            lidar_data=point_cloud,
            language_command="Navigate to parking garage"
        )
    """
    
    def __init__(self, 
                 api_endpoint: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 privacy_level: str = "high",
                 quantum_enhanced: bool = True,
                 timeout: float = 10.0):
        """
        Initialize QEP-VLA Navigator
        
        Args:
            api_endpoint: QEP-VLA API server endpoint
            api_key: Optional API key for authentication
            privacy_level: Privacy level ('high', 'medium', 'low')
            quantum_enhanced: Enable quantum sensor enhancement
            timeout: Request timeout in seconds
        """
        
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.privacy_level = privacy_level
        self.quantum_enhanced = quantum_enhanced
        self.timeout = timeout
        
        # Performance tracking
        self.inference_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validate connection
        self._validate_connection()
    
    def _validate_connection(self) -> bool:
        """
        Validate connection to QEP-VLA API
        """
        try:
            response = requests.get(
                f"{self.api_endpoint}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            health_data = response.json()
            if health_data.get('status') == 'healthy':
                self.logger.info("Successfully connected to QEP-VLA API")
                return True
            else:
                raise ConnectionError("API health check failed")
                
        except Exception as e:
            self.logger.error(f"Failed to connect to QEP-VLA API: {e}")
            raise ConnectionError(f"Cannot connect to API at {self.api_endpoint}")
    
    def navigate(self,
                camera_data: np.ndarray,
                lidar_data: np.ndarray,
                language_command: str,
                imu_data: Optional[np.ndarray] = None,
                privacy_level: Optional[str] = None,
                quantum_enhanced: Optional[bool] = None) -> NavigationResult:
        """
        Perform navigation inference
        
        Args:
            camera_data: RGB camera image [H, W, 3]
            lidar_data: LiDAR point cloud [N, 3]
            language_command: Natural language navigation instruction
            imu_data: Optional IMU readings (accel + gyro)
            privacy_level: Override default privacy level
            quantum_enhanced: Override quantum enhancement setting
            
        Returns:
            NavigationResult object with action probabilities and metadata
        """
        
        start_time = time.time()
        
        # Use provided parameters or defaults
        privacy = privacy_level or self.privacy_level
        quantum = quantum_enhanced if quantum_enhanced is not None else self.quantum_enhanced
        
        # Default IMU data if not provided
        if imu_data is None:
            imu_data = np.zeros(6, dtype=np.float32)
        
        # Prepare request data
        request_data = {
            "camera_data": camera_data.tolist(),
            "lidar_data": lidar_data.tolist(),
            "imu_data": imu_data.tolist(),
            "language_command": language_command,
            "privacy_level": privacy,
            "quantum_enhanced": quantum
        }
        
        try:
            # Make API request
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                f"{self.api_endpoint}/api/v1/navigate",
                json=request_data,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result_data = response.json()
            
            # Create result object
            result = NavigationResult(
                action_probabilities=result_data["action_probabilities"],
                confidence_score=result_data["confidence_score"],
                processing_time_ms=result_data["processing_time_ms"],
                privacy_guarantee=result_data["privacy_guarantee"],
                quantum_enhanced=quantum,
                meets_latency_requirement=result_data["meets_latency_requirement"],
                model_complexity=result_data["model_complexity"],
                safety_fallback_triggered=result_data["safety_fallback_triggered"]
            )
            
            # Track performance
            total_time_ms = (time.time() - start_time) * 1000
            self.inference_history.append({
                "timestamp": time.time(),
                "processing_time_ms": result.processing_time_ms,
                "total_time_ms": total_time_ms,
                "confidence_score": result.confidence_score,
                "privacy_level": privacy,
                "quantum_enhanced": quantum,
                "model_complexity": result.model_complexity,
                "safety_fallback": result.safety_fallback_triggered
            })
            
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Navigation request failed: {e}")
            raise RuntimeError(f"Navigation inference failed: {e}")
    
    def batch_navigate(self,
                      batch_data: List[Dict[str, Any]],
                      max_workers: int = 4) -> List[NavigationResult]:
        """
        Perform batch navigation inference
        
        Args:
            batch_data: List of navigation requests
            max_workers: Maximum concurrent requests
            
        Returns:
            List of NavigationResult objects
        """
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_single(data):
            return self.navigate(**data)
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {
                executor.submit(process_single, data): data 
                for data in batch_data
            }
            
            for future in as_completed(future_to_data):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch navigation failed for item: {e}")
                    results.append(None)
        
        return results
    
    def apply_privacy_transform(self,
                              data: np.ndarray,
                              transform_type: str,
                              privacy_level: str = "high") -> np.ndarray:
        """
        Apply quantum privacy transformation to data
        
        Args:
            data: Input data array
            transform_type: Type of transformation
            privacy_level: Privacy level
            
        Returns:
            Transformed data with privacy guarantees
        """
        
        request_data = {
            "data": data.tolist(),
            "transform_type": transform_type,
            "privacy_level": privacy_level
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                f"{self.api_endpoint}/api/v1/privacy/transform",
                json=request_data,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result_data = response.json()
            return np.array(result_data["transformed_data"])
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Privacy transform failed: {e}")
            raise RuntimeError(f"Privacy transformation failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status and health information
        """
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                f"{self.api_endpoint}/api/v1/system/status",
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get system status: {e}")
            raise RuntimeError(f"System status check failed: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get detailed system performance metrics
        """
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                f"{self.api_endpoint}/api/v1/system/metrics",
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            raise RuntimeError(f"System metrics retrieval failed: {e}")
    
    def update_system_config(self,
                           privacy_budget_epsilon: Optional[float] = None,
                           quantum_enhancement_factor: Optional[float] = None,
                           max_latency_ms: Optional[float] = None,
                           blockchain_validation: Optional[bool] = None) -> bool:
        """
        Update system configuration parameters
        
        Returns:
            True if configuration updated successfully
        """
        
        request_data = {}
        if privacy_budget_epsilon is not None:
            request_data["privacy_budget_epsilon"] = privacy_budget_epsilon
        if quantum_enhancement_factor is not None:
            request_data["quantum_enhancement_factor"] = quantum_enhancement_factor
        if max_latency_ms is not None:
            request_data["max_latency_ms"] = max_latency_ms
        if blockchain_validation is not None:
            request_data["blockchain_validation"] = blockchain_validation
        
        if not request_data:
            return True  # No updates needed
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.put(
                f"{self.api_endpoint}/api/v1/system/config",
                json=request_data,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("success", False)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Configuration update failed: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get SDK performance metrics
        """
        if not self.inference_history:
            return {"message": "No inference history available"}
        
        history = self.inference_history
        
        avg_processing_time = np.mean([h["processing_time_ms"] for h in history])
        avg_total_time = np.mean([h["total_time_ms"] for h in history])
        avg_confidence = np.mean([h["confidence_score"] for h in history])
        
        latency_compliance = sum(
            1 for h in history if h["processing_time_ms"] < 50
        ) / len(history)
        
        safety_rate = sum(
            1 for h in history if not h["safety_fallback"]
        ) / len(history)
        
        model_complexity_distribution = {}
        for h in history:
            complexity = h.get("model_complexity", "unknown")
            model_complexity_distribution[complexity] = model_complexity_distribution.get(complexity, 0) + 1
        
        return {
            "total_inferences": len(history),
            "average_processing_time_ms": avg_processing_time,
            "average_total_time_ms": avg_total_time,
            "average_confidence_score": avg_confidence,
            "latency_compliance_rate": latency_compliance,
            "safety_rate": safety_rate,
            "model_complexity_distribution": model_complexity_distribution,
            "recent_inferences": history[-10:] if len(history) > 10 else history
        }
    
    def configure_privacy(self, level: str, custom_epsilon: Optional[float] = None):
        """
        Configure privacy settings
        
        Args:
            level: Privacy level ('high', 'medium', 'low', 'custom')
            custom_epsilon: Custom epsilon value for differential privacy
        """
        valid_levels = ['high', 'medium', 'low', 'custom']
        if level not in valid_levels:
            raise ValueError(f"Privacy level must be one of {valid_levels}")
        
        self.privacy_level = level
        
        if level == 'custom' and custom_epsilon is not None:
            # Update system configuration
            success = self.update_system_config(privacy_budget_epsilon=custom_epsilon)
            if success:
                self.logger.info(f"Custom privacy level set with ε={custom_epsilon}")
            else:
                self.logger.warning("Failed to update system configuration")
        
        self.logger.info(f"Privacy level set to: {level}")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.inference_history.clear()
        self.logger.info("Performance metrics reset")
    
    def export_metrics(self, filepath: str) -> bool:
        """
        Export performance metrics to file
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if export successful
        """
        try:
            metrics = self.get_performance_metrics()
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check
        
        Returns:
            Health status information
        """
        try:
            # Check API connection
            api_status = self._validate_connection()
            
            # Check system status
            system_status = self.get_system_status()
            
            # Check performance metrics
            performance_metrics = self.get_performance_metrics()
            
            return {
                "status": "healthy" if api_status and system_status.get("status") == "operational" else "degraded",
                "api_connection": api_status,
                "system_status": system_status,
                "performance_metrics": performance_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
