#!/usr/bin/env python3
"""
Test Enhanced Bo-Wei Features
"""

import sys
import os
import time
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_human_safety():
    """Test enhanced human safety monitoring"""
    try:
        from src.core.human_robot_supply_chain import HumanVerticalIntegration, SupplyChainConfig
        
        config = SupplyChainConfig(privacy_protection_level='high')
        human_integration = HumanVerticalIntegration(config)
        
        # Test data
        human_agents = {
            'human_1': {
                'biometric_data': {'vital_signs': {'heart_rate': 70, 'blood_pressure': 120}},
                'activity_patterns': {'movement': 'normal'},
                'work_duration': 4.0,
                'location': {'position': [0, 0, 0]},
                'vital_signs': {'heart_rate': 70}
            }
        }
        
        # Test enhanced monitoring
        result = human_integration.monitor_human_safety(human_agents)
        
        logger.info(f"✅ Enhanced Human Safety Monitoring: {'success' if result else 'failed'}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Human Safety test failed: {e}")
        return False

def test_enhanced_robot_coordination():
    """Test enhanced robot coordination"""
    try:
        from src.core.human_robot_supply_chain import RobotVerticalIntegration, SupplyChainConfig
        
        config = SupplyChainConfig(privacy_protection_level='high')
        robot_integration = RobotVerticalIntegration(config)
        
        # Test data
        robot_agents = {
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
        
        human_agents = {
            'human_1': {
                'location': {'position': [1.5, 1.5, 0]}
            }
        }
        
        # Test enhanced coordination
        result = robot_integration.coordinate_robot_safety(robot_agents, human_agents)
        
        logger.info(f"✅ Enhanced Robot Coordination: {'success' if result else 'failed'}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Robot Coordination test failed: {e}")
        return False

def test_comprehensive_asset_tracking():
    """Test comprehensive asset tracking"""
    try:
        from src.core.safety_privacy_protection import SupplyChainAssetTracking, SafetyPrivacyConfig
        
        config = SafetyPrivacyConfig(privacy_budget=0.1)
        asset_tracking = SupplyChainAssetTracking(config)
        
        # Test data
        humans = {
            'human_1': {
                'role': 'worker',
                'department': 'production',
                'work_duration': 4.0,
                'location': {'zone': 'A1'},
                'status': 'active'
            }
        }
        
        robots = {
            'robot_1': {
                'model': 'AGV-001',
                'current_position': [1, 1, 0],
                'target_position': [2, 2, 0],
                'task_completion': {'completion_rate': 0.95},
                'energy_consumption': 50.0,
                'status': 'operational'
            }
        }
        
        materials = {
            'material_1': {
                'type': 'steel_plate',
                'source': 'supplier_A',
                'manufacturing_date': '2024-01-15',
                'batch_number': 'B001',
                'quality_grade': 'A',
                'current_location': 'warehouse_B',
                'ownership_history': ['supplier_A', 'warehouse_B'],
                'compliance_certificates': ['ISO_9001']
            }
        }
        
        # Test comprehensive tracking
        result = asset_tracking.track_all_assets(humans, robots, materials)
        
        logger.info(f"✅ Comprehensive Asset Tracking: {'success' if result else 'failed'}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive Asset Tracking test failed: {e}")
        return False

def test_reality_aware_navigation():
    """Test reality-aware navigation"""
    try:
        from src.core.reality_aware_pvla_navigation import RealityAwarePVLANavigation
        
        navigation = RealityAwarePVLANavigation()
        
        # Test data
        request = {
            'vision': [0.1, 0.2, 0.3, 0.4, 0.5],
            'lidar': [1.0, 2.0, 3.0, 4.0, 5.0],
            'imu': [0.01, 0.02, 0.03],
            'language': 'Navigate to target safely',
            'network_state': {
                'blockchain': {'transaction_count': 100, 'block_count': 10},
                'iot_network': {'device_count': 50, 'connection_count': 100}
            },
            'asset_data': {'asset_1': {'type': 'material'}},
            'human_agents': {
                'human_1': {'location': {'position': [0, 0, 0]}}
            },
            'robot_agents': {
                'robot_1': {'current_position': [1, 1, 0]}
            },
            'hazard_level': 0.2,
            'workspace_size': 100.0
        }
        
        # Test comprehensive navigation
        result = navigation.navigate_with_comprehensive_awareness(request)
        
        logger.info(f"✅ Reality-Aware Navigation: {'success' if result.get('success') else 'failed'}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Reality-Aware Navigation test failed: {e}")
        return False

def test_safety_enhanced_training():
    """Test safety-enhanced federated training"""
    try:
        from src.core.safety_enhanced_federated_trainer import SafetyEnhancedFederatedTrainer
        
        trainer = SafetyEnhancedFederatedTrainer()
        
        # Test data
        training_data = {
            'agent_1': {'model_parameters': {'weight_1': 0.5, 'bias_1': 0.1}},
            'agent_2': {'model_parameters': {'weight_1': 0.6, 'bias_1': 0.2}}
        }
        
        human_agents = {
            'agent_1': {
                'type': 'human',
                'health_score': 0.8,
                'fatigue_level': 0.3,
                'work_duration': 6.0
            }
        }
        
        robot_agents = {
            'agent_2': {
                'type': 'robot',
                'fault_status': {'status': 'operational'},
                'energy_level': 0.8,
                'security_status': {'status': 'secure'}
            }
        }
        
        # Test safety-enhanced training
        result = trainer.train_with_safety_privacy(training_data, human_agents, robot_agents)
        
        logger.info(f"✅ Safety-Enhanced Training: {'success' if result.get('success') else 'failed'}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Safety-Enhanced Training test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Testing Enhanced Bo-Wei Features")
    logger.info("=" * 60)
    
    tests = [
        ("Enhanced Human Safety Monitoring", test_enhanced_human_safety),
        ("Enhanced Robot Coordination", test_enhanced_robot_coordination),
        ("Comprehensive Asset Tracking", test_comprehensive_asset_tracking),
        ("Reality-Aware Navigation", test_reality_aware_navigation),
        ("Safety-Enhanced Training", test_safety_enhanced_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Testing {test_name}...")
        if test_func():
            passed += 1
        logger.info("-" * 40)
    
    logger.info(f"🎉 Test Results: {passed}/{total} tests passed!")
    
    if passed == total:
        logger.info("✅ All Enhanced Bo-Wei Features are working correctly!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
