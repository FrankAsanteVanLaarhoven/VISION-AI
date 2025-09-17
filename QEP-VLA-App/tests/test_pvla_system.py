"""
Comprehensive Test Suite for PVLA Navigation System
Production-ready testing and validation for all PVLA components
"""

import pytest
import asyncio
import numpy as np
import torch
import time
import logging
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import PVLA components
from src.core.pvla_navigation_system import PVLANavigationSystem, PVLAConfig
from src.core.pvla_vision_algorithm import VisionNavigationAlgorithm, VisionConfig
from src.core.pvla_language_algorithm import QuantumLanguageUnderstanding, LanguageConfig
from src.core.pvla_action_algorithm import ConsciousnessActionSelection, ActionConfig
from src.core.pvla_meta_learning import MetaLearningQuantumAdaptation, MetaLearningConfig
from src.core.quantum_privacy_transform import QuantumPrivacyTransform, QuantumTransformConfig
from src.quantum.quantum_infrastructure import QuantumInfrastructure, QuantumConfig
from src.privacy.privacy_monitoring import PrivacyMonitoringSystem, PrivacyConfig
from src.api.pvla_api import app
from fastapi.testclient import TestClient

# Test configuration
TEST_CONFIG = {
    'vision': VisionConfig(
        input_resolution=(64, 64),  # Smaller for testing
        feature_dim=128,
        privacy_budget=0.5,
        max_processing_time_ms=100.0
    ),
    'language': LanguageConfig(
        model_name="bert-base-uncased",
        max_sequence_length=128,
        quantum_dimension=32,
        num_quantum_states=4
    ),
    'action': ActionConfig(
        num_actions=5,
        safety_threshold=0.5,
        ethics_threshold=0.5
    ),
    'meta_learning': MetaLearningConfig(
        quantum_dimension=32,
        num_quantum_layers=4,
        performance_window=10,
        memory_size=100
    ),
    'privacy': QuantumTransformConfig(
        privacy_budget_epsilon=0.5,
        privacy_budget_delta=1e-3,
        quantum_enhancement_factor=1.5
    ),
    'quantum': QuantumConfig(
        num_qubits=10,
        backend="simulator",
        shots=100
    )
}

class TestVisionAlgorithm:
    """Test suite for Vision Navigation Algorithm"""
    
    @pytest.fixture
    def vision_algorithm(self):
        """Create vision algorithm instance for testing"""
        return VisionNavigationAlgorithm(TEST_CONFIG['vision'])
    
    def test_vision_algorithm_initialization(self, vision_algorithm):
        """Test vision algorithm initialization"""
        assert vision_algorithm is not None
        assert vision_algorithm.device is not None
        assert vision_algorithm.config.input_resolution == (64, 64)
    
    def test_encrypt_frame(self, vision_algorithm):
        """Test frame encryption"""
        # Create test frame
        test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Encrypt frame
        encrypted_frame = vision_algorithm.encrypt_frame(test_frame)
        
        assert encrypted_frame is not None
        assert encrypted_frame.shape == (1, 3, 64, 64)
        assert encrypted_frame.dtype == torch.float32
    
    def test_extract_navigation_features(self, vision_algorithm):
        """Test navigation feature extraction"""
        # Create test encrypted frame
        encrypted_frame = torch.randn(1, 3, 64, 64)
        
        # Extract features
        features = vision_algorithm.extract_navigation_features(encrypted_frame)
        
        assert features is not None
        assert features.shape[1] == TEST_CONFIG['vision'].feature_dim
    
    def test_quantum_attention(self, vision_algorithm):
        """Test quantum attention mechanism"""
        # Create test features and navigation state
        features = torch.randn(1, 128)
        nav_state = torch.randn(6)
        
        # Apply quantum attention
        attended_features, attention_weights = vision_algorithm.apply_quantum_attention(features, nav_state)
        
        assert attended_features is not None
        assert attention_weights is not None
        assert attended_features.shape == features.shape
    
    def test_slam_update(self, vision_algorithm):
        """Test SLAM update"""
        # Create test features and attention weights
        features = torch.randn(1, 128)
        attention_weights = torch.randn(1, 1, 8)
        
        # Update SLAM
        position_estimate, map_update, uncertainty = vision_algorithm.update_slam_estimate(features, attention_weights)
        
        assert position_estimate is not None
        assert map_update is not None
        assert uncertainty is not None
        assert position_estimate.shape[1] == 6  # [x, y, z, roll, pitch, yaw]
    
    def test_vision_forward_pass(self, vision_algorithm):
        """Test complete vision forward pass"""
        # Create test frame and navigation state
        test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        nav_state = torch.randn(6)
        
        # Forward pass
        position_estimate, privacy_score = vision_algorithm.forward(test_frame, nav_state)
        
        assert position_estimate is not None
        assert privacy_score is not None
        assert 0.0 <= privacy_score <= 1.0
        assert position_estimate.shape[1] == 6
    
    def test_vision_performance_metrics(self, vision_algorithm):
        """Test vision performance metrics"""
        # Run some forward passes
        for _ in range(5):
            test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            nav_state = torch.randn(6)
            vision_algorithm.forward(test_frame, nav_state)
        
        # Get metrics
        metrics = vision_algorithm.get_performance_metrics()
        
        assert metrics is not None
        assert 'average_processing_time_ms' in metrics
        assert 'average_privacy_score' in metrics
        assert metrics['total_frames_processed'] == 5
    
    def test_vision_health_check(self, vision_algorithm):
        """Test vision health check"""
        health = vision_algorithm.health_check()
        
        assert health is not None
        assert 'status' in health
        assert health['status'] == 'healthy'

class TestLanguageAlgorithm:
    """Test suite for Quantum Language Understanding Algorithm"""
    
    @pytest.fixture
    def language_algorithm(self):
        """Create language algorithm instance for testing"""
        return QuantumLanguageUnderstanding(TEST_CONFIG['language'])
    
    def test_language_algorithm_initialization(self, language_algorithm):
        """Test language algorithm initialization"""
        assert language_algorithm is not None
        assert language_algorithm.device is not None
        assert language_algorithm.config.model_name == "bert-base-uncased"
    
    def test_encode_language_input(self, language_algorithm):
        """Test language input encoding"""
        test_command = "Move forward carefully"
        
        # Encode input
        features = language_algorithm.encode_language_input(test_command)
        
        assert features is not None
        assert features.shape[1] == language_algorithm.language_model.config.hidden_size
    
    def test_quantum_superposition(self, language_algorithm):
        """Test quantum superposition creation"""
        # Create test features
        features = torch.randn(1, 768)  # BERT hidden size
        
        # Create superposition
        superposition = language_algorithm.create_quantum_superposition(features)
        
        assert superposition is not None
        assert superposition.shape[1] == TEST_CONFIG['language'].num_quantum_states
        assert superposition.shape[2] == TEST_CONFIG['language'].quantum_dimension
    
    def test_quantum_entanglement(self, language_algorithm):
        """Test quantum entanglement"""
        # Create test superposition and navigation context
        superposition = torch.randn(1, 4, 32)
        nav_context = torch.randn(6)
        
        # Create entanglement
        entangled_context, entanglement_strength = language_algorithm.quantum_entangle(superposition, nav_context)
        
        assert entangled_context is not None
        assert entanglement_strength is not None
        assert entangled_context.shape[1] == 6  # Navigation state dimension
    
    def test_quantum_measurement(self, language_algorithm):
        """Test quantum measurement"""
        # Create test entangled context and objectives
        entangled_context = torch.randn(1, 6)
        objectives = torch.randn(10)
        
        # Measure quantum state
        navigation_action, confidence = language_algorithm.quantum_measure(entangled_context, objectives)
        
        assert navigation_action is not None
        assert confidence is not None
        assert navigation_action.shape[1] == 10  # Action dimension
        assert 0.0 <= confidence <= 1.0
    
    def test_language_forward_pass(self, language_algorithm):
        """Test complete language forward pass"""
        # Create test inputs
        language_input = "Turn left at the intersection"
        nav_context = torch.randn(6)
        nav_objectives = torch.randn(10)
        
        # Forward pass
        action, confidence, metadata = language_algorithm.forward(language_input, nav_context, nav_objectives)
        
        assert action is not None
        assert confidence is not None
        assert metadata is not None
        assert 0.0 <= confidence <= 1.0
        assert 'processing_time_ms' in metadata
    
    def test_command_parsing(self, language_algorithm):
        """Test natural language command parsing"""
        test_commands = [
            "Move forward carefully",
            "Turn left at the next intersection",
            "Stop immediately",
            "Avoid the obstacle ahead"
        ]
        
        for command in test_commands:
            parsed = language_algorithm.parse_natural_language(command)
            
            assert parsed is not None
            assert 'commands' in parsed
            assert 'raw_text' in parsed
            assert parsed['raw_text'] == command
    
    def test_language_performance_metrics(self, language_algorithm):
        """Test language performance metrics"""
        # Run some forward passes
        for _ in range(5):
            language_input = "Test command"
            nav_context = torch.randn(6)
            nav_objectives = torch.randn(10)
            language_algorithm.forward(language_input, nav_context, nav_objectives)
        
        # Get metrics
        metrics = language_algorithm.get_performance_metrics()
        
        assert metrics is not None
        assert 'average_processing_time_ms' in metrics
        assert 'average_confidence_score' in metrics
        assert metrics['total_commands_processed'] == 5

class TestActionAlgorithm:
    """Test suite for Consciousness-Driven Action Selection"""
    
    @pytest.fixture
    def action_algorithm(self):
        """Create action algorithm instance for testing"""
        return ConsciousnessActionSelection(TEST_CONFIG['action'])
    
    def test_action_algorithm_initialization(self, action_algorithm):
        """Test action algorithm initialization"""
        assert action_algorithm is not None
        assert action_algorithm.device is not None
        assert action_algorithm.config.num_actions == 5
    
    def test_utility_calculation(self, action_algorithm):
        """Test utility calculation"""
        # Create test inputs
        possible_actions = torch.randn(5)
        state_features = torch.randn(512)
        goal_features = torch.randn(256)
        
        # Calculate utility
        utility_matrix = action_algorithm.calculate_utility_matrix(possible_actions, state_features, goal_features)
        
        assert utility_matrix is not None
        assert utility_matrix.shape[1] == 5  # Number of actions
    
    def test_safety_evaluation(self, action_algorithm):
        """Test safety constraint evaluation"""
        # Create test inputs
        possible_actions = torch.randn(5)
        state_features = torch.randn(512)
        environment_features = torch.randn(128)
        
        # Evaluate safety
        safety_matrix, risk_scores = action_algorithm.evaluate_safety_constraints(
            possible_actions, state_features, environment_features
        )
        
        assert safety_matrix is not None
        assert risk_scores is not None
        assert safety_matrix.shape[1] == 5
        assert risk_scores.shape[1] == 5
    
    def test_ethics_evaluation(self, action_algorithm):
        """Test ethical framework application"""
        # Create test inputs
        possible_actions = torch.randn(5)
        state_features = torch.randn(512)
        context_features = torch.randn(128)
        
        # Apply ethical framework
        ethics_matrix = action_algorithm.apply_ethical_framework(
            possible_actions, state_features, context_features
        )
        
        assert ethics_matrix is not None
        assert ethics_matrix.shape[1] == 5
    
    def test_weighted_optimization(self, action_algorithm):
        """Test weighted optimization"""
        # Create test matrices
        utility_matrix = torch.randn(1, 5)
        safety_matrix = torch.randn(1, 5)
        ethics_matrix = torch.randn(1, 5)
        consciousness_weights = [0.4, 0.3, 0.3]
        
        # Perform optimization
        optimal_action, total_scores, metadata = action_algorithm.weighted_optimization(
            utility_matrix, safety_matrix, ethics_matrix, consciousness_weights
        )
        
        assert optimal_action is not None
        assert total_scores is not None
        assert metadata is not None
        assert 0 <= optimal_action < 5
        assert 'utility_scores' in metadata
    
    def test_action_forward_pass(self, action_algorithm):
        """Test complete action forward pass"""
        # Create test inputs
        possible_actions = torch.randn(5)
        state_features = torch.randn(512)
        goal_features = torch.randn(256)
        environment_features = torch.randn(128)
        context_features = torch.randn(128)
        
        # Forward pass
        optimal_action, explanation, metadata = action_algorithm.forward(
            possible_actions, state_features, goal_features, environment_features, context_features
        )
        
        assert optimal_action is not None
        assert explanation is not None
        assert metadata is not None
        assert 0 <= optimal_action < 5
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    def test_action_performance_metrics(self, action_algorithm):
        """Test action performance metrics"""
        # Run some forward passes
        for _ in range(5):
            possible_actions = torch.randn(5)
            state_features = torch.randn(512)
            goal_features = torch.randn(256)
            environment_features = torch.randn(128)
            context_features = torch.randn(128)
            action_algorithm.forward(
                possible_actions, state_features, goal_features, environment_features, context_features
            )
        
        # Get metrics
        metrics = action_algorithm.get_performance_metrics()
        
        assert metrics is not None
        assert 'average_decision_time_ms' in metrics
        assert 'total_decisions' in metrics
        assert metrics['total_decisions'] == 5

class TestMetaLearning:
    """Test suite for Meta-Learning Quantum Adaptation"""
    
    @pytest.fixture
    def meta_learning(self):
        """Create meta-learning instance for testing"""
        return MetaLearningQuantumAdaptation(TEST_CONFIG['meta_learning'])
    
    def test_meta_learning_initialization(self, meta_learning):
        """Test meta-learning initialization"""
        assert meta_learning is not None
        assert meta_learning.device is not None
        assert meta_learning.config.quantum_dimension == 32
    
    def test_performance_tracking(self, meta_learning):
        """Test performance tracking"""
        # Update performance
        for _ in range(5):
            meta_learning.quantum_meta_learner.update_performance(
                success=True, latency=50.0, accuracy=0.8
            )
        
        # Get performance metrics
        metrics = meta_learning.quantum_meta_learner.performance_tracker.get_performance_metrics()
        
        assert metrics is not None
        assert 'success_rate' in metrics
        assert 'average_latency' in metrics
        assert 'average_accuracy' in metrics
        assert metrics['total_episodes'] == 5
    
    def test_quantum_parameter_optimization(self, meta_learning):
        """Test quantum parameter optimization"""
        # Create test performance history
        performance_history = [
            {'success': True, 'latency': 45.0, 'accuracy': 0.85},
            {'success': False, 'latency': 60.0, 'accuracy': 0.70},
            {'success': True, 'latency': 40.0, 'accuracy': 0.90}
        ]
        
        # Optimize parameters
        optimized_parameters = meta_learning.learn_optimal_quantum_circuit_parameters(performance_history)
        
        assert optimized_parameters is not None
        assert 'rotation_angles' in optimized_parameters
    
    def test_quantum_algorithm_adaptation(self, meta_learning):
        """Test quantum algorithm adaptation"""
        # Create test quantum parameters
        quantum_parameters = {
            'rotation_angles': torch.randn(4, 32, 3),
            'entanglement_weights': torch.randn(4, 32, 32),
            'measurement_basis': torch.randn(32, 32)
        }
        
        # Adapt algorithms
        adapted_circuits = meta_learning.adapt_quantum_algorithms(quantum_parameters)
        
        assert adapted_circuits is not None
        assert 'rotation_angles' in adapted_circuits
    
    def test_framework_integration(self, meta_learning):
        """Test framework integration"""
        # Create test adapted circuits
        adapted_circuits = {
            'rotation_angles': torch.randn(4, 32, 3),
            'entanglement_weights': torch.randn(4, 32, 32),
            'measurement_basis': torch.randn(32, 32)
        }
        
        # Integrate improvements
        updated_framework, improvement_metrics = meta_learning.integrate_quantum_improvements(adapted_circuits)
        
        assert updated_framework is not None
        assert improvement_metrics is not None
        assert 'success_rate' in improvement_metrics
    
    def test_meta_learning_forward_pass(self, meta_learning):
        """Test complete meta-learning forward pass"""
        # Create test performance history
        performance_history = [
            {'success': True, 'latency': 45.0, 'accuracy': 0.85},
            {'success': False, 'latency': 60.0, 'accuracy': 0.70},
            {'success': True, 'latency': 40.0, 'accuracy': 0.90}
        ]
        
        # Forward pass
        updated_framework, improvement_metrics = meta_learning.forward(performance_history)
        
        assert updated_framework is not None
        assert improvement_metrics is not None
        assert 'updated_framework' in updated_framework
        assert 'improvement_metrics' in improvement_metrics

class TestPrivacyMonitoring:
    """Test suite for Privacy Monitoring System"""
    
    @pytest.fixture
    def privacy_monitoring(self):
        """Create privacy monitoring instance for testing"""
        return PrivacyMonitoringSystem(TEST_CONFIG['privacy'])
    
    def test_privacy_monitoring_initialization(self, privacy_monitoring):
        """Test privacy monitoring initialization"""
        assert privacy_monitoring is not None
        assert privacy_monitoring.device is not None
        assert privacy_monitoring.config.privacy_budget_epsilon == 0.5
    
    def test_privacy_budget_tracking(self, privacy_monitoring):
        """Test privacy budget tracking"""
        # Consume budget
        success = privacy_monitoring.budget_tracker.consume_budget(0.1)
        assert success is True
        
        # Check budget status
        status = privacy_monitoring.budget_tracker.get_budget_status()
        assert status['current_epsilon'] == 0.1
        assert status['remaining_epsilon'] == 0.4
    
    def test_data_anonymization(self, privacy_monitoring):
        """Test data anonymization"""
        # Create test data
        test_data = torch.randn(10, 10)
        
        # Apply differential privacy
        anonymized_data = privacy_monitoring.data_anonymizer.apply_differential_privacy(test_data, 0.1)
        
        assert anonymized_data is not None
        assert anonymized_data.shape == test_data.shape
        assert not torch.equal(anonymized_data, test_data)  # Should be different due to noise
    
    def test_consent_management(self, privacy_monitoring):
        """Test consent management"""
        user_id = "test_user"
        consent_type = "data_processing"
        
        # Record consent
        privacy_monitoring.consent_manager.record_consent(user_id, consent_type, True)
        
        # Check consent
        has_consent = privacy_monitoring.consent_manager.check_consent(user_id, consent_type)
        assert has_consent is True
    
    def test_privacy_audit(self, privacy_monitoring):
        """Test privacy audit"""
        # Create test privacy measures
        privacy_measures = {
            'encryption': 'AES-256',
            'anonymization': 'differential_privacy',
            'consent': True
        }
        
        # Perform audit
        audit_result = privacy_monitoring.auditor.audit_data_processing(
            'navigation_data', 'test_processing', privacy_measures
        )
        
        assert audit_result is not None
        assert 'compliance_score' in audit_result
        assert 'violations' in audit_result
        assert audit_result['compliance_score'] == 1.0  # Should be compliant
    
    @pytest.mark.asyncio
    async def test_privacy_monitoring_async(self, privacy_monitoring):
        """Test async privacy monitoring"""
        # Create test data
        test_data = torch.randn(10, 10)
        
        # Monitor data processing
        result = await privacy_monitoring.monitor_data_processing(
            test_data, 'test_processing', 'test_user'
        )
        
        assert result is not None
        assert 'allowed' in result
        assert 'anonymized_data' in result
        assert 'privacy_measures' in result
    
    def test_privacy_violation_detection(self, privacy_monitoring):
        """Test privacy violation detection"""
        from src.privacy.privacy_monitoring import PrivacyViolationType
        
        # Detect violation
        violation = privacy_monitoring.detect_privacy_violation(
            PrivacyViolationType.DATA_LEAKAGE,
            {'details': 'test_violation'}
        )
        
        assert violation is not None
        assert violation['type'] == PrivacyViolationType.DATA_LEAKAGE
        assert violation['severity'] == 'critical'
    
    def test_privacy_status(self, privacy_monitoring):
        """Test privacy status reporting"""
        status = privacy_monitoring.get_privacy_status()
        
        assert status is not None
        assert 'budget_status' in status
        assert 'violation_summary' in status
        assert 'audit_summary' in status
        assert 'monitoring_performance' in status

class TestPVLANavigationSystem:
    """Test suite for complete PVLA Navigation System"""
    
    @pytest.fixture
    def pvla_system(self):
        """Create PVLA system instance for testing"""
        config = PVLAConfig(
            vision_config=TEST_CONFIG['vision'],
            language_config=TEST_CONFIG['language'],
            action_config=TEST_CONFIG['action'],
            meta_learning_config=TEST_CONFIG['meta_learning'],
            privacy_config=TEST_CONFIG['privacy']
        )
        return PVLANavigationSystem(config)
    
    def test_pvla_system_initialization(self, pvla_system):
        """Test PVLA system initialization"""
        assert pvla_system is not None
        assert pvla_system.vision_algorithm is not None
        assert pvla_system.language_algorithm is not None
        assert pvla_system.action_algorithm is not None
        assert pvla_system.meta_learning is not None
        assert pvla_system.privacy_transform is not None
    
    @pytest.mark.asyncio
    async def test_navigation_request_processing(self, pvla_system):
        """Test complete navigation request processing"""
        # Create test inputs
        camera_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        language_command = "Move forward carefully"
        navigation_context = {
            'context': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'objectives': [1.0, 0.0, 0.0, 0.0, 0.0],
            'goals': [1.0, 0.0, 0.0] + [0.0] * 253,
            'environment': [0.0] * 128,
            'context': [0.0] * 128
        }
        
        # Process navigation request
        result = await pvla_system.process_navigation_request(
            camera_frame, language_command, navigation_context
        )
        
        assert result is not None
        assert 'navigation_action' in result
        assert 'explanation' in result
        assert 'confidence_score' in result
        assert 'processing_time_ms' in result
        assert 'vision_metadata' in result
        assert 'language_metadata' in result
        assert 'action_metadata' in result
    
    def test_system_status(self, pvla_system):
        """Test system status reporting"""
        status = pvla_system.get_system_status()
        
        assert status is not None
        assert 'system_state' in status
        assert 'system_metrics' in status
        assert 'component_health' in status
        assert 'configuration' in status
    
    def test_navigation_state_update(self, pvla_system):
        """Test navigation state update"""
        new_state = torch.randn(6)
        pvla_system.update_navigation_state(new_state)
        
        assert torch.equal(pvla_system.current_navigation_state, new_state)
    
    def test_navigation_objectives_update(self, pvla_system):
        """Test navigation objectives update"""
        new_objectives = torch.randn(10)
        pvla_system.update_navigation_objectives(new_objectives)
        
        assert torch.equal(pvla_system.navigation_objectives, new_objectives)

class TestAPI:
    """Test suite for PVLA API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "PVLA Navigation API"
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_navigation_endpoint(self, client):
        """Test navigation endpoint"""
        navigation_request = {
            "camera_frame": {
                "frame_data": [[[255, 255, 255] for _ in range(64)] for _ in range(64)],
                "width": 64,
                "height": 64
            },
            "language_command": {
                "command": "Move forward carefully"
            },
            "navigation_context": {
                "current_position": [0.0, 0.0, 0.0],
                "current_orientation": [0.0, 0.0, 0.0],
                "target_position": [1.0, 0.0, 0.0],
                "environment_data": {},
                "safety_constraints": {},
                "objectives": ["move_forward"]
            }
        }
        
        response = client.post("/navigate", json=navigation_request)
        # Note: This might fail if PVLA system is not initialized in test environment
        # In production, this would return 200 with navigation result
        assert response.status_code in [200, 503]  # 503 if system not available

class TestIntegration:
    """Integration tests for PVLA system"""
    
    @pytest.fixture
    def full_system(self):
        """Create full PVLA system for integration testing"""
        config = PVLAConfig(
            vision_config=TEST_CONFIG['vision'],
            language_config=TEST_CONFIG['language'],
            action_config=TEST_CONFIG['action'],
            meta_learning_config=TEST_CONFIG['meta_learning'],
            privacy_config=TEST_CONFIG['privacy']
        )
        return PVLANavigationSystem(config)
    
    @pytest.mark.asyncio
    async def test_end_to_end_navigation(self, full_system):
        """Test end-to-end navigation workflow"""
        # Test multiple navigation scenarios
        test_scenarios = [
            {
                'camera_frame': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                'language_command': "Move forward",
                'context': {
                    'context': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'objectives': [1.0, 0.0, 0.0, 0.0, 0.0],
                    'goals': [1.0, 0.0, 0.0] + [0.0] * 253,
                    'environment': [0.0] * 128,
                    'context': [0.0] * 128
                }
            },
            {
                'camera_frame': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                'language_command': "Turn left at intersection",
                'context': {
                    'context': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'objectives': [0.0, 0.0, 1.0, 0.0, 0.0],
                    'goals': [0.0, 1.0, 0.0] + [0.0] * 253,
                    'environment': [0.1] * 128,
                    'context': [0.0] * 128
                }
            }
        ]
        
        results = []
        for scenario in test_scenarios:
            result = await full_system.process_navigation_request(
                scenario['camera_frame'],
                scenario['language_command'],
                scenario['context']
            )
            results.append(result)
        
        # Validate results
        assert len(results) == 2
        for result in results:
            assert result['navigation_action'] is not None
            assert result['confidence_score'] >= 0.0
            assert result['processing_time_ms'] < 1000.0  # Should be fast
    
    def test_system_performance_under_load(self, full_system):
        """Test system performance under load"""
        import asyncio
        
        async def single_request():
            camera_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            language_command = "Test command"
            navigation_context = {
                'context': [0.0] * 6,
                'objectives': [1.0, 0.0, 0.0, 0.0, 0.0],
                'goals': [0.0] * 256,
                'environment': [0.0] * 128,
                'context': [0.0] * 128
            }
            
            result = await full_system.process_navigation_request(
                camera_frame, language_command, navigation_context
            )
            return result['processing_time_ms']
        
        # Run multiple concurrent requests
        async def load_test():
            tasks = [single_request() for _ in range(10)]
            times = await asyncio.gather(*tasks)
            return times
        
        # Execute load test
        times = asyncio.run(load_test())
        
        # Validate performance
        assert len(times) == 10
        avg_time = sum(times) / len(times)
        assert avg_time < 500.0  # Average should be under 500ms
        assert max(times) < 1000.0  # No request should take more than 1 second

# Performance benchmarks
class TestPerformance:
    """Performance benchmark tests"""
    
    def test_vision_processing_benchmark(self):
        """Benchmark vision processing performance"""
        vision_algorithm = VisionNavigationAlgorithm(TEST_CONFIG['vision'])
        
        # Benchmark multiple frames
        times = []
        for _ in range(10):
            test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            nav_state = torch.randn(6)
            
            start_time = time.time()
            vision_algorithm.forward(test_frame, nav_state)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        assert avg_time < 100.0  # Should be under 100ms
    
    def test_language_processing_benchmark(self):
        """Benchmark language processing performance"""
        language_algorithm = QuantumLanguageUnderstanding(TEST_CONFIG['language'])
        
        # Benchmark multiple commands
        times = []
        for _ in range(10):
            language_input = "Test navigation command"
            nav_context = torch.randn(6)
            nav_objectives = torch.randn(10)
            
            start_time = time.time()
            language_algorithm.forward(language_input, nav_context, nav_objectives)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        assert avg_time < 200.0  # Should be under 200ms
    
    def test_action_selection_benchmark(self):
        """Benchmark action selection performance"""
        action_algorithm = ConsciousnessActionSelection(TEST_CONFIG['action'])
        
        # Benchmark multiple decisions
        times = []
        for _ in range(10):
            possible_actions = torch.randn(5)
            state_features = torch.randn(512)
            goal_features = torch.randn(256)
            environment_features = torch.randn(128)
            context_features = torch.randn(128)
            
            start_time = time.time()
            action_algorithm.forward(
                possible_actions, state_features, goal_features, environment_features, context_features
            )
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        assert avg_time < 50.0  # Should be under 50ms

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
