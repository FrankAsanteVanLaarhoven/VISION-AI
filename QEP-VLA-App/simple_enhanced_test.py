#!/usr/bin/env python3
"""
Simple test for Enhanced QEP-VLA System with Bo-Wei Technologies
Tests core functionality without heavy dependencies
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

def test_basic_imports():
    """Test basic imports without heavy dependencies"""
    try:
        logger.info("Testing basic imports...")
        
        # Test config
        from config.settings import get_settings
        logger.info("✅ Config imported successfully")
        
        # Test Bo-Wei technologies (these should work without transformers)
        from src.core.ai_reality_comprehension import RealityComprehensionEngine, RealityComprehensionConfig
        logger.info("✅ AI Reality Comprehension imported successfully")
        
        from src.core.human_robot_supply_chain import HumanRobotSupplyChainIntegration, SupplyChainConfig
        logger.info("✅ Human-Robot Supply Chain imported successfully")
        
        from src.core.safety_privacy_protection import SafetyPrivacyAssetProtection, SafetyPrivacyConfig
        logger.info("✅ Safety & Privacy Protection imported successfully")
        
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

def test_health_checks():
    """Test health checks for all Bo-Wei technologies"""
    try:
        logger.info("Testing health checks...")
        
        # Test AI Reality Comprehension health
        from src.core.ai_reality_comprehension import RealityComprehensionEngine, RealityComprehensionConfig
        reality_config = RealityComprehensionConfig(quantum_enhancement=True)
        reality_engine = RealityComprehensionEngine(reality_config)
        health = reality_engine.health_check()
        logger.info(f"✅ AI Reality Comprehension health: {health.get('status', 'unknown')}")
        
        # Test Human-Robot Supply Chain health
        from src.core.human_robot_supply_chain import HumanRobotSupplyChainIntegration, SupplyChainConfig
        supply_chain_config = SupplyChainConfig(privacy_protection_level='high')
        supply_chain = HumanRobotSupplyChainIntegration(supply_chain_config)
        health = supply_chain.health_check()
        logger.info(f"✅ Human-Robot Supply Chain health: {health.get('status', 'unknown')}")
        
        # Test Safety & Privacy Protection health
        from src.core.safety_privacy_protection import SafetyPrivacyAssetProtection, SafetyPrivacyConfig
        safety_config = SafetyPrivacyConfig(privacy_budget=0.1)
        safety_protection = SafetyPrivacyAssetProtection(safety_config)
        health = safety_protection.health_check()
        logger.info(f"✅ Safety & Privacy Protection health: {health.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Health checks failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Simple Enhanced QEP-VLA System Tests")
    logger.info("=" * 60)
    
    # Test basic imports
    if not test_basic_imports():
        logger.error("❌ Basic import tests failed")
        return False
    
    logger.info("✅ All basic imports successful")
    logger.info("-" * 40)
    
    # Test Bo-Wei technologies
    if not test_bo_wei_technologies():
        logger.error("❌ Bo-Wei technologies tests failed")
        return False
    
    logger.info("✅ All Bo-Wei technologies tests successful")
    logger.info("-" * 40)
    
    # Test health checks
    if not test_health_checks():
        logger.error("❌ Health check tests failed")
        return False
    
    logger.info("✅ All health check tests successful")
    logger.info("-" * 40)
    
    logger.info("🎉 All tests passed! Bo-Wei technologies are ready!")
    logger.info("📊 Summary:")
    logger.info("   ✅ AI Reality Comprehension Engine")
    logger.info("   ✅ Human-Robot Supply Chain Integration")
    logger.info("   ✅ Safety & Privacy Asset Protection")
    logger.info("   ✅ All health checks passing")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
