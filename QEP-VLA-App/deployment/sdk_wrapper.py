"""
SDK Wrapper for QEP-VLA Application
Provides easy-to-use interface for integrating QEP-VLA privacy features
"""

import sys
import os
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from datetime import datetime
import json
import yaml
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces.camera_interface import CameraInterface
from interfaces.lidar_interface import LiDARInterface
from interfaces.quantum_sensor_interface import QuantumSensorInterface
from core.federated_trainer import FederatedTrainer, TrainingConfig
from core.navigation_engine import NavigationEngine, NavigationConfig
from core.quantum_privacy_transform import QuantumPrivacyTransform, QuantumTransformConfig
from core.scenario_generator import ScenarioGenerator, ScenarioConfig
from core.secure_aggregation import SecureAggregation

class QEPVLASDK:
    """Main SDK class for QEP-VLA privacy system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.camera_interface = None
        self.lidar_interface = None
        self.quantum_sensor_interface = None
        self.federated_trainer = None
        self.navigation_engine = None
        self.quantum_privacy_transform = None
        self.scenario_generator = None
        self.secure_aggregation = None
        
        # Component status
        self.components_initialized = False
        
        # Initialize logging
        self._setup_logging()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                self.logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                config = self._get_default_config()
        else:
            config = self._get_default_config()
            self.logger.info("Using default configuration")
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'privacy': {
                'default_level': 'high',
                'compliance': ['GDPR', 'CCPA'],
                'data_retention_days': 30
            },
            'components': {
                'camera': True,
                'lidar': True,
                'quantum_sensor': True,
                'federated_learning': True,
                'navigation': True,
                'privacy_transform': True,
                'scenario_generation': True,
                'secure_aggregation': True
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('qep_vla_sdk.log')
            ]
        )
    
    def initialize_components(self, components: Optional[List[str]] = None) -> bool:
        """Initialize specified components or all components"""
        if components is None:
            components = list(self.config['components'].keys())
        
        self.logger.info(f"Initializing components: {components}")
        
        try:
            # Initialize camera interface
            if 'camera' in components and self.config['components']['camera']:
                self.camera_interface = CameraInterface(
                    camera_id=self.config.get('camera_id', 0),
                    privacy_config=self.config.get('camera_privacy', {})
                )
                self.logger.info("Camera interface initialized")
            
            # Initialize LiDAR interface
            if 'lidar' in components and self.config['components']['lidar']:
                self.lidar_interface = LiDARInterface(
                    device_id=self.config.get('lidar_device_id', 'default'),
                    privacy_config=self.config.get('lidar_privacy', {})
                )
                self.logger.info("LiDAR interface initialized")
            
            # Initialize quantum sensor interface
            if 'quantum_sensor' in components and self.config['components']['quantum_sensor']:
                self.quantum_sensor_interface = QuantumSensorInterface(
                    sensor_id=self.config.get('quantum_sensor_id', 'quantum_default'),
                    privacy_config=self.config.get('quantum_sensor_privacy', {})
                )
                self.logger.info("Quantum sensor interface initialized")
            
            # Initialize federated trainer
            if 'federated_learning' in components and self.config['components']['federated_learning']:
                # This would require a model to be passed in production
                self.logger.info("Federated trainer initialization deferred (requires model)")
            
            # Initialize navigation engine
            if 'navigation' in components and self.config['components']['navigation']:
                nav_config = NavigationConfig(**self.config.get('navigation_config', {}))
                self.navigation_engine = NavigationEngine(nav_config)
                self.logger.info("Navigation engine initialized")
            
            # Initialize quantum privacy transform
            if 'privacy_transform' in components and self.config['components']['privacy_transform']:
                transform_config = QuantumTransformConfig(**self.config.get('quantum_transform_config', {}))
                self.quantum_privacy_transform = QuantumPrivacyTransform(transform_config)
                self.logger.info("Quantum privacy transform initialized")
            
            # Initialize scenario generator
            if 'scenario_generation' in components and self.config['components']['scenario_generation']:
                scenario_config = ScenarioConfig(**self.config.get('scenario_config', {}))
                self.scenario_generator = ScenarioGenerator(scenario_config)
                self.logger.info("Scenario generator initialized")
            
            # Initialize secure aggregation
            if 'secure_aggregation' in components and self.config['components']['secure_aggregation']:
                self.secure_aggregation = SecureAggregation(
                    config=self.config.get('secure_aggregation_config', {})
                )
                self.logger.info("Secure aggregation initialized")
            
            self.components_initialized = True
            self.logger.info("All requested components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    def capture_private_data(self, data_type: str = 'all') -> Dict[str, Any]:
        """Capture data from sensors with privacy controls"""
        if not self.components_initialized:
            raise RuntimeError("Components not initialized. Call initialize_components() first.")
        
        captured_data = {}
        
        try:
            if data_type in ['all', 'camera'] and self.camera_interface:
                frame = self.camera_interface.capture_frame()
                if frame is not None:
                    captured_data['camera'] = {
                        'frame': frame,
                        'info': self.camera_interface.get_frame_info(),
                        'privacy_applied': True
                    }
            
            if data_type in ['all', 'lidar'] and self.lidar_interface:
                point_cloud = self.lidar_interface.scan_environment()
                if point_cloud is not None:
                    captured_data['lidar'] = {
                        'point_cloud': point_cloud,
                        'info': self.lidar_interface.get_scan_info(),
                        'privacy_applied': True
                    }
            
            if data_type in ['all', 'quantum'] and self.quantum_sensor_interface:
                measurement = self.quantum_sensor_interface.measure_quantum_state()
                if measurement is not None:
                    captured_data['quantum'] = {
                        'measurement': measurement,
                        'info': self.quantum_sensor_interface.get_sensor_info(),
                        'privacy_applied': True
                    }
            
            self.logger.info(f"Captured {data_type} data with privacy controls")
            return captured_data
            
        except Exception as e:
            self.logger.error(f"Failed to capture {data_type} data: {e}")
            return {}
    
    def apply_privacy_transform(self, data: Any, transform_type: str = 'auto') -> Any:
        """Apply privacy transformation to data"""
        if not self.quantum_privacy_transform:
            raise RuntimeError("Quantum privacy transform not initialized")
        
        try:
            if transform_type == 'auto':
                # Automatically select best transform based on data type
                transform_type = self._select_optimal_transform(data)
            
            transformed_data = self.quantum_privacy_transform.apply_transform(data, transform_type)
            self.logger.info(f"Applied {transform_type} privacy transformation")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Failed to apply privacy transform: {e}")
            return data
    
    def _select_optimal_transform(self, data: Any) -> str:
        """Select optimal privacy transform based on data characteristics"""
        # Simple heuristic for transform selection
        if hasattr(data, 'shape') and len(data.shape) > 1:
            return 'entanglement_masking'
        elif isinstance(data, (list, dict)):
            return 'quantum_key_encryption'
        else:
            return 'quantum_noise'
    
    def plan_navigation(self, target: List[float], mode: str = 'privacy_aware') -> Dict[str, Any]:
        """Plan navigation path with privacy considerations"""
        if not self.navigation_engine:
            raise RuntimeError("Navigation engine not initialized")
        
        try:
            target_array = np.array(target)
            path = self.navigation_engine.plan_path(target_array, mode)
            
            navigation_plan = {
                'target': target,
                'path': [point.tolist() for point in path],
                'mode': mode,
                'privacy_zones_avoided': len(self.navigation_engine.privacy_zones),
                'estimated_duration': len(path) * 2.0  # Rough estimate
            }
            
            self.logger.info(f"Navigation plan created for target {target}")
            return navigation_plan
            
        except Exception as e:
            self.logger.error(f"Failed to plan navigation: {e}")
            return {}
    
    def generate_test_scenario(self, scenario_type: str) -> Dict[str, Any]:
        """Generate test scenario for system validation"""
        if not self.scenario_generator:
            raise RuntimeError("Scenario generator not initialized")
        
        try:
            # Convert string to enum
            from core.scenario_generator import ScenarioType
            scenario_enum = ScenarioType(scenario_type)
            
            scenario = self.scenario_generator.generate_scenario(scenario_enum)
            self.logger.info(f"Generated {scenario_type} test scenario")
            return scenario
            
        except Exception as e:
            self.logger.error(f"Failed to generate scenario: {scenario_type}: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and component health"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'components_initialized': self.components_initialized,
            'component_status': {},
            'overall_health': 'unknown'
        }
        
        # Check component status
        if self.camera_interface:
            status['component_status']['camera'] = 'active'
        if self.lidar_interface:
            status['component_status']['lidar'] = 'active'
        if self.quantum_sensor_interface:
            status['component_status']['quantum_sensor'] = 'active'
        if self.navigation_engine:
            status['component_status']['navigation'] = 'active'
        if self.quantum_privacy_transform:
            status['component_status']['privacy_transform'] = 'active'
        if self.scenario_generator:
            status['component_status']['scenario_generator'] = 'active'
        if self.secure_aggregation:
            status['component_status']['secure_aggregation'] = 'active'
        
        # Determine overall health
        active_components = len([s for s in status['component_status'].values() if s == 'active'])
        total_components = len(self.config['components'])
        
        if active_components == total_components:
            status['overall_health'] = 'healthy'
        elif active_components > total_components // 2:
            status['overall_health'] = 'degraded'
        else:
            status['overall_health'] = 'unhealthy'
        
        return status
    
    def export_config(self, filepath: str):
        """Export current configuration to file"""
        try:
            with open(filepath, 'w') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
    
    def cleanup(self):
        """Cleanup resources and close connections"""
        try:
            if self.camera_interface:
                self.camera_interface.release()
            
            if self.lidar_interface:
                self.lidar_interface.disconnect()
            
            if self.quantum_sensor_interface:
                self.quantum_sensor_interface.shutdown()
            
            self.logger.info("SDK cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# Convenience functions for quick access
def create_qep_vla_sdk(config_path: Optional[str] = None) -> QEPVLASDK:
    """Create and return a QEP-VLA SDK instance"""
    return QEPVLASDK(config_path)

def quick_privacy_transform(data: Any, config_path: Optional[str] = None) -> Any:
    """Quick privacy transformation without full SDK initialization"""
    with QEPVLASDK(config_path) as sdk:
        sdk.initialize_components(['privacy_transform'])
        return sdk.apply_privacy_transform(data)

def quick_navigation_plan(target: List[float], config_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick navigation planning without full SDK initialization"""
    with QEPVLASDK(config_path) as sdk:
        sdk.initialize_components(['navigation'])
        return sdk.plan_navigation(target)
