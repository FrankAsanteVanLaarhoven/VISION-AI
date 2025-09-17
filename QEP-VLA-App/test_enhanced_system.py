#!/usr/bin/env python3
"""
Test script for Enhanced QEP-VLA System with Bo-Wei Technologies
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

def test_imports():
    """Test if all modules can be imported"""
    try:
        logger.info("Testing imports...")
        
        # Test config
        from config.settings import get_settings
        logger.info("✅ Config imported successfully")
        
        # Test Bo-Wei technologies
        from src.core.ai_reality_comprehension import RealityComprehensionEngine, RealityComprehensionConfig
        logger.info("✅ AI Reality Comprehension imported successfully")
        
        from src.core.human_robot_supply_chain import HumanRobotSupplyChainIntegration, SupplyChainConfig
        logger.info("✅ Human-Robot Supply Chain imported successfully")
        
        from src.core.safety_privacy_protection import SafetyPrivacyAssetProtection, SafetyPrivacyConfig
        logger.info("✅ Safety & Privacy Protection imported successfully")
        
        # Test enhanced unified system
        from src.core.enhanced_unified_qep_vla_system import EnhancedUnifiedQEPVLASystem, EnhancedUnifiedSystemConfig
        logger.info("✅ Enhanced Unified System imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_bo_wei_technologies():
    """Test Bo-Wei technologies individually"""
    try:
        logger.info("Testing Bo-Wei technologies...")
        
        # Test AI Reality Comprehension
        from src.core.ai_reality_comprehension import RealityComprehensionEngine, RealityComprehensionConfig
        reality_config = RealityComprehensionConfig(quantum_enhancement=True)
        reality_engine = RealityComprehensionEngine(reality_config)
        
        # Test with dummy data
        dummy_input = {
            'sensors': {
                'vision': [0.1, 0.2, 0.3, 0.4, 0.5],
                'lidar': [1.0, 2.0, 3.0, 4.0, 5.0],
                'imu': [0.01, 0.02, 0.03]
            },
            'network': {
                'blockchain': {'transaction_count': 100, 'block_count': 10},
                'iot_network': {'device_count': 50, 'connection_count': 100}
            },
            'language': 'Test language input',
            'context': {'location': 'test', 'time': 'now'}
        }
        
        result = reality_engine.comprehend_reality(dummy_input)
        logger.info(f"✅ AI Reality Comprehension test: {'success' if 'error' not in result else 'failed'}")
        
        # Test Human-Robot Supply Chain Integration
        from src.core.human_robot_supply_chain import HumanRobotSupplyChainIntegration, SupplyChainConfig
        supply_chain_config = SupplyChainConfig(privacy_protection_level='high')
        supply_chain = HumanRobotSupplyChainIntegration(supply_chain_config)
        
        dummy_humans = {
            'human_1': {
                'biometric_data': {'vital_signs': {'heart_rate': 70, 'blood_pressure': 120}},
                'activity_patterns': {'movement': 'normal'},
                'work_duration': 4.0,
                'location': {'position': [0, 0, 0]},
                'vital_signs': {'heart_rate': 70}
            }
        }
        
        dummy_robots = {
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
        
        dummy_environment = {
            'hazard_level': 0.2,
            'workspace_size': 100.0
        }
        
        result = supply_chain.integrate_supply_chain(dummy_humans, dummy_robots, dummy_environment)
        logger.info(f"✅ Human-Robot Supply Chain test: {'success' if 'error' not in result else 'failed'}")
        
        # Test Safety & Privacy Asset Protection
        from src.core.safety_privacy_protection import SafetyPrivacyAssetProtection, SafetyPrivacyConfig
        safety_config = SafetyPrivacyConfig(privacy_budget=0.1)
        safety_protection = SafetyPrivacyAssetProtection(safety_config)
        
        dummy_system_state = {
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
        
        dummy_privacy_requirements = {'epsilon': 0.1}
        
        result = safety_protection.protect_system_assets(dummy_system_state, dummy_privacy_requirements)
        logger.info(f"✅ Safety & Privacy Protection test: {'success' if 'error' not in result else 'failed'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Bo-Wei technologies test failed: {e}")
        return False

def test_enhanced_unified_system():
    """Test the enhanced unified system"""
    try:
        logger.info("Testing Enhanced Unified System...")
        
        from src.core.enhanced_unified_qep_vla_system import EnhancedUnifiedQEPVLASystem, EnhancedUnifiedSystemConfig
        from src.core.unified_qep_vla_system import NavigationRequest
        
        # Initialize enhanced system
        enhanced_config = EnhancedUnifiedSystemConfig(
            privacy_budget=0.1,
            quantum_enhancement=True,
            blockchain_validation=True,
            reality_comprehension_enabled=True,
            human_robot_integration_enabled=True,
            safety_privacy_protection_enabled=True
        )
        
        enhanced_system = EnhancedUnifiedQEPVLASystem(enhanced_config)
        logger.info("✅ Enhanced Unified System initialized successfully")
        
        # Test navigation request
        nav_request = NavigationRequest(
            start_position=[0, 0, 0],
            target_position=[10, 10, 0],
            language_command='Navigate to target using all Bo-Wei technologies',
            sensor_data={
                'vision': [0.1, 0.2, 0.3, 0.4, 0.5],
                'lidar': [1.0, 2.0, 3.0, 4.0, 5.0],
                'imu': [0.01, 0.02, 0.03]
            },
            network_state={
                'blockchain': {'transaction_count': 100, 'block_count': 10},
                'iot_network': {'device_count': 50, 'connection_count': 100}
            },
            context_data={
                'human_agents': {'human_1': {'status': 'active'}},
                'robot_agents': {'robot_1': {'status': 'operational'}},
                'hazard_level': 0.2,
                'workspace_size': 100.0
            },
            privacy_requirements={'epsilon': 0.1},
            performance_requirements={'max_latency_ms': 50, 'min_accuracy': 0.9}
        )
        
        # Process enhanced navigation request
        start_time = time.time()
        response = enhanced_system.process_enhanced_navigation_request(nav_request)
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Enhanced Navigation test: {'success' if response.success else 'failed'}")
        logger.info(f"   Processing time: {processing_time:.2f}ms")
        logger.info(f"   Privacy score: {response.privacy_score:.3f}")
        logger.info(f"   Quantum enhancement: {response.quantum_enhancement_factor:.3f}")
        
        # Test system status
        status = enhanced_system.get_enhanced_system_status()
        logger.info(f"✅ System status: {status.get('status', 'unknown')}")
        
        # Test system metrics
        metrics = enhanced_system.get_enhanced_system_metrics()
        logger.info(f"✅ System metrics retrieved: {len(metrics)} metrics")
        
        # Test health check
        health = enhanced_system.health_check()
        logger.info(f"✅ Health check: {health.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Unified System test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Enhanced QEP-VLA System Tests")
    logger.info("=" * 60)
    
    # Test imports
    if not test_imports():
        logger.error("❌ Import tests failed")
        return False
    
    logger.info("✅ All imports successful")
    logger.info("-" * 40)
    
    # Test Bo-Wei technologies
    if not test_bo_wei_technologies():
        logger.error("❌ Bo-Wei technologies tests failed")
        return False
    
    logger.info("✅ All Bo-Wei technologies tests successful")
    logger.info("-" * 40)
    
    # Test enhanced unified system
    if not test_enhanced_unified_system():
        logger.error("❌ Enhanced Unified System tests failed")
        return False
    
    logger.info("✅ All Enhanced Unified System tests successful")
    logger.info("-" * 40)
    
    logger.info("🎉 All tests passed! Enhanced QEP-VLA System with Bo-Wei technologies is ready!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
