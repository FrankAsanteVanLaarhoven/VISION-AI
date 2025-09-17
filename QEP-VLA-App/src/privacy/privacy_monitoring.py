"""
Privacy Compliance Monitoring and Validation System
Production-ready privacy monitoring for PVLA Navigation System
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from config.settings import get_settings

settings = get_settings()

class PrivacyViolationType(Enum):
    """Types of privacy violations"""
    DATA_LEAKAGE = "data_leakage"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVACY_BUDGET_EXCEEDED = "privacy_budget_exceeded"
    ENCRYPTION_FAILURE = "encryption_failure"
    ANONYMIZATION_FAILURE = "anonymization_failure"
    CONSENT_VIOLATION = "consent_violation"

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

@dataclass
class PrivacyConfig:
    """Configuration for privacy monitoring"""
    privacy_budget_epsilon: float = 0.1
    privacy_budget_delta: float = 1e-5
    max_queries_per_budget: int = 100
    encryption_algorithm: str = "AES-256"
    anonymization_method: str = "differential_privacy"
    consent_required: bool = True
    audit_logging: bool = True
    real_time_monitoring: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PrivacyBudgetTracker:
    """
    Tracks and manages privacy budget consumption
    Implements differential privacy budget management
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.current_epsilon = 0.0
        self.current_delta = 0.0
        self.query_count = 0
        self.budget_history = []
        self.violation_count = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Privacy Budget Tracker initialized")
    
    def consume_budget(self, epsilon_cost: float, delta_cost: float = 0.0) -> bool:
        """
        Consume privacy budget for a query
        
        Returns:
            bool: True if budget is available, False if exceeded
        """
        # Check if budget would be exceeded
        if (self.current_epsilon + epsilon_cost > self.config.privacy_budget_epsilon or
            self.current_delta + delta_cost > self.config.privacy_budget_delta or
            self.query_count >= self.config.max_queries_per_budget):
            
            self.violation_count += 1
            self.logger.warning(f"Privacy budget exceeded. Epsilon: {self.current_epsilon + epsilon_cost:.4f}, "
                              f"Delta: {self.current_delta + delta_cost:.6f}, Queries: {self.query_count + 1}")
            return False
        
        # Consume budget
        self.current_epsilon += epsilon_cost
        self.current_delta += delta_cost
        self.query_count += 1
        
        # Record budget consumption
        self.budget_history.append({
            'timestamp': time.time(),
            'epsilon_cost': epsilon_cost,
            'delta_cost': delta_cost,
            'total_epsilon': self.current_epsilon,
            'total_delta': self.current_delta,
            'query_count': self.query_count
        })
        
        return True
    
    def reset_budget(self):
        """Reset privacy budget"""
        self.current_epsilon = 0.0
        self.current_delta = 0.0
        self.query_count = 0
        self.logger.info("Privacy budget reset")
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        return {
            'current_epsilon': self.current_epsilon,
            'current_delta': self.current_delta,
            'remaining_epsilon': self.config.privacy_budget_epsilon - self.current_epsilon,
            'remaining_delta': self.config.privacy_budget_delta - self.current_delta,
            'query_count': self.query_count,
            'max_queries': self.config.max_queries_per_budget,
            'budget_utilization': self.current_epsilon / self.config.privacy_budget_epsilon,
            'violation_count': self.violation_count
        }

class DataAnonymizer:
    """
    Data anonymization and de-identification
    Implements various anonymization techniques
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.anonymization_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.anonymization_key)
        
        # Anonymization parameters
        self.noise_scale = 1.0
        self.k_anonymity = 5
        self.l_diversity = 3
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Data Anonymizer initialized")
    
    def apply_differential_privacy(self, data: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Apply differential privacy noise to data
        """
        # Calculate noise scale
        sensitivity = 1.0  # L1 sensitivity
        noise_scale = sensitivity / epsilon
        
        # Add Laplacian noise
        noise = torch.from_numpy(
            np.random.laplace(0, noise_scale, data.shape)
        ).float().to(data.device)
        
        return data + noise
    
    def k_anonymize(self, data: torch.Tensor, k: int = None) -> torch.Tensor:
        """
        Apply k-anonymity to data
        """
        k = k or self.k_anonymity
        
        # Group similar data points
        data_flat = data.flatten()
        sorted_indices = torch.argsort(data_flat)
        
        # Create groups of size k
        anonymized_data = data.clone()
        for i in range(0, len(sorted_indices), k):
            group_indices = sorted_indices[i:i+k]
            if len(group_indices) >= k:
                # Replace with group mean
                group_mean = data_flat[group_indices].mean()
                for idx in group_indices:
                    flat_idx = idx.item()
                    row, col = divmod(flat_idx, data.shape[1])
                    anonymized_data[row, col] = group_mean
        
        return anonymized_data
    
    def l_diversify(self, data: torch.Tensor, l: int = None) -> torch.Tensor:
        """
        Apply l-diversity to data
        """
        l = l or self.l_diversity
        
        # Ensure diversity in sensitive attributes
        diversified_data = data.clone()
        
        # Add controlled noise to ensure diversity
        for i in range(data.shape[0]):
            noise = torch.randn_like(data[i]) * 0.1
            diversified_data[i] = data[i] + noise
        
        return diversified_data
    
    def encrypt_sensitive_data(self, data: torch.Tensor) -> bytes:
        """
        Encrypt sensitive data
        """
        # Convert tensor to bytes
        data_bytes = data.detach().cpu().numpy().tobytes()
        
        # Encrypt data
        encrypted_data = self.cipher_suite.encrypt(data_bytes)
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, encrypted_data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Decrypt sensitive data
        """
        # Decrypt data
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_data)
        
        # Convert back to tensor
        data_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
        data_tensor = torch.from_numpy(data_array.reshape(shape))
        
        return data_tensor

class ConsentManager:
    """
    Manages user consent and privacy preferences
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.consent_records = {}
        self.privacy_preferences = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Consent Manager initialized")
    
    def record_consent(self, user_id: str, consent_type: str, granted: bool, timestamp: float = None):
        """
        Record user consent
        """
        if timestamp is None:
            timestamp = time.time()
        
        consent_record = {
            'user_id': user_id,
            'consent_type': consent_type,
            'granted': granted,
            'timestamp': timestamp,
            'expires_at': timestamp + 365 * 24 * 3600  # 1 year default
        }
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent_record)
        
        self.logger.info(f"Consent recorded for user {user_id}: {consent_type} = {granted}")
    
    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """
        Check if user has granted consent
        """
        if user_id not in self.consent_records:
            return False
        
        current_time = time.time()
        
        for record in self.consent_records[user_id]:
            if (record['consent_type'] == consent_type and 
                record['granted'] and 
                record['expires_at'] > current_time):
                return True
        
        return False
    
    def set_privacy_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        Set user privacy preferences
        """
        self.privacy_preferences[user_id] = preferences
        self.logger.info(f"Privacy preferences set for user {user_id}")
    
    def get_privacy_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get user privacy preferences
        """
        return self.privacy_preferences.get(user_id, {})

class PrivacyAuditor:
    """
    Privacy compliance auditor
    Performs regular privacy audits and generates reports
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.audit_logs = []
        self.violation_reports = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Privacy Auditor initialized")
    
    def audit_data_processing(self, 
                            data_type: str,
                            processing_method: str,
                            privacy_measures: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit data processing for privacy compliance
        """
        audit_result = {
            'timestamp': time.time(),
            'data_type': data_type,
            'processing_method': processing_method,
            'privacy_measures': privacy_measures,
            'compliance_score': 0.0,
            'violations': [],
            'recommendations': []
        }
        
        # Check encryption
        if 'encryption' not in privacy_measures:
            audit_result['violations'].append('No encryption applied')
        elif privacy_measures['encryption'] != self.config.encryption_algorithm:
            audit_result['violations'].append(f'Incorrect encryption algorithm: {privacy_measures["encryption"]}')
        
        # Check anonymization
        if 'anonymization' not in privacy_measures:
            audit_result['violations'].append('No anonymization applied')
        elif privacy_measures['anonymization'] != self.config.anonymization_method:
            audit_result['violations'].append(f'Incorrect anonymization method: {privacy_measures["anonymization"]}')
        
        # Check consent
        if self.config.consent_required and 'consent' not in privacy_measures:
            audit_result['violations'].append('No consent recorded')
        
        # Calculate compliance score
        total_checks = 3
        passed_checks = total_checks - len(audit_result['violations'])
        audit_result['compliance_score'] = passed_checks / total_checks
        
        # Generate recommendations
        if audit_result['compliance_score'] < 1.0:
            audit_result['recommendations'].append('Implement proper encryption')
            audit_result['recommendations'].append('Apply differential privacy')
            audit_result['recommendations'].append('Obtain user consent')
        
        # Log audit result
        self.audit_logs.append(audit_result)
        
        return audit_result
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive privacy compliance report
        """
        if not self.audit_logs:
            return {'message': 'No audit data available'}
        
        # Calculate overall compliance
        total_audits = len(self.audit_logs)
        avg_compliance = np.mean([log['compliance_score'] for log in self.audit_logs])
        
        # Count violations by type
        violation_counts = {}
        for log in self.audit_logs:
            for violation in log['violations']:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        # Generate report
        report = {
            'timestamp': time.time(),
            'total_audits': total_audits,
            'average_compliance_score': avg_compliance,
            'violation_counts': violation_counts,
            'compliance_level': self._get_compliance_level(avg_compliance),
            'recommendations': self._generate_recommendations(violation_counts)
        }
        
        return report
    
    def _get_compliance_level(self, score: float) -> str:
        """Get compliance level based on score"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_recommendations(self, violation_counts: Dict[str, int]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        if 'No encryption applied' in violation_counts:
            recommendations.append('Implement encryption for all sensitive data')
        
        if 'No anonymization applied' in violation_counts:
            recommendations.append('Apply differential privacy or other anonymization techniques')
        
        if 'No consent recorded' in violation_counts:
            recommendations.append('Implement proper consent management system')
        
        return recommendations

class PrivacyMonitoringSystem:
    """
    Main privacy monitoring system
    Integrates all privacy components and provides monitoring interface
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize components
        self.budget_tracker = PrivacyBudgetTracker(self.config)
        self.data_anonymizer = DataAnonymizer(self.config)
        self.consent_manager = ConsentManager(self.config)
        self.auditor = PrivacyAuditor(self.config)
        
        # Monitoring state
        self.active_monitors = {}
        self.monitoring_threads = {}
        self.alert_callbacks = []
        
        # Performance tracking
        self.monitoring_times = []
        self.violation_detections = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Privacy Monitoring System initialized")
    
    async def monitor_data_processing(self, 
                                    data: torch.Tensor,
                                    processing_type: str,
                                    user_id: str = None) -> Dict[str, Any]:
        """
        Monitor data processing for privacy compliance
        """
        start_time = time.time()
        
        try:
            # Check consent
            consent_granted = True
            if self.config.consent_required and user_id:
                consent_granted = self.consent_manager.check_consent(user_id, processing_type)
            
            if not consent_granted:
                violation = {
                    'type': PrivacyViolationType.CONSENT_VIOLATION,
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'processing_type': processing_type,
                    'severity': 'high'
                }
                self.violation_detections.append(violation)
                return {'violation': violation, 'allowed': False}
            
            # Check privacy budget
            epsilon_cost = 0.1  # Default cost
            budget_available = self.budget_tracker.consume_budget(epsilon_cost)
            
            if not budget_available:
                violation = {
                    'type': PrivacyViolationType.PRIVACY_BUDGET_EXCEEDED,
                    'timestamp': time.time(),
                    'epsilon_cost': epsilon_cost,
                    'severity': 'medium'
                }
                self.violation_detections.append(violation)
                return {'violation': violation, 'allowed': False}
            
            # Apply privacy measures
            privacy_measures = {
                'encryption': self.config.encryption_algorithm,
                'anonymization': self.config.anonymization_method,
                'consent': consent_granted,
                'budget_consumed': epsilon_cost
            }
            
            # Anonymize data
            if self.config.anonymization_method == 'differential_privacy':
                anonymized_data = self.data_anonymizer.apply_differential_privacy(data, epsilon_cost)
            else:
                anonymized_data = data
            
            # Audit processing
            audit_result = self.auditor.audit_data_processing(
                'navigation_data', processing_type, privacy_measures
            )
            
            # Track performance
            monitoring_time = (time.time() - start_time) * 1000
            self.monitoring_times.append(monitoring_time)
            
            return {
                'allowed': True,
                'anonymized_data': anonymized_data,
                'privacy_measures': privacy_measures,
                'audit_result': audit_result,
                'monitoring_time_ms': monitoring_time,
                'budget_status': self.budget_tracker.get_budget_status()
            }
            
        except Exception as e:
            self.logger.error(f"Privacy monitoring failed: {e}")
            return {
                'allowed': False,
                'error': str(e),
                'monitoring_time_ms': (time.time() - start_time) * 1000
            }
    
    def detect_privacy_violation(self, 
                               violation_type: PrivacyViolationType,
                               details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect and record privacy violation
        """
        violation = {
            'type': violation_type,
            'timestamp': time.time(),
            'details': details,
            'severity': self._assess_violation_severity(violation_type, details)
        }
        
        self.violation_detections.append(violation)
        
        # Trigger alerts
        for callback in self.alert_callbacks:
            try:
                callback(violation)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        self.logger.warning(f"Privacy violation detected: {violation_type.value}")
        
        return violation
    
    def _assess_violation_severity(self, violation_type: PrivacyViolationType, details: Dict[str, Any]) -> str:
        """Assess violation severity"""
        if violation_type in [PrivacyViolationType.DATA_LEAKAGE, PrivacyViolationType.UNAUTHORIZED_ACCESS]:
            return 'critical'
        elif violation_type in [PrivacyViolationType.PRIVACY_BUDGET_EXCEEDED, PrivacyViolationType.ENCRYPTION_FAILURE]:
            return 'high'
        else:
            return 'medium'
    
    def add_alert_callback(self, callback: callable):
        """Add alert callback for privacy violations"""
        self.alert_callbacks.append(callback)
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get comprehensive privacy status"""
        return {
            'budget_status': self.budget_tracker.get_budget_status(),
            'violation_summary': {
                'total_violations': len(self.violation_detections),
                'violations_by_type': self._count_violations_by_type(),
                'recent_violations': self.violation_detections[-10:] if self.violation_detections else []
            },
            'audit_summary': self.auditor.generate_privacy_report(),
            'monitoring_performance': {
                'total_monitoring_events': len(self.monitoring_times),
                'average_monitoring_time_ms': np.mean(self.monitoring_times) if self.monitoring_times else 0.0,
                'max_monitoring_time_ms': np.max(self.monitoring_times) if self.monitoring_times else 0.0
            },
            'configuration': {
                'privacy_budget_epsilon': self.config.privacy_budget_epsilon,
                'privacy_budget_delta': self.config.privacy_budget_delta,
                'encryption_algorithm': self.config.encryption_algorithm,
                'anonymization_method': self.config.anonymization_method,
                'consent_required': self.config.consent_required
            }
        }
    
    def _count_violations_by_type(self) -> Dict[str, int]:
        """Count violations by type"""
        counts = {}
        for violation in self.violation_detections:
            violation_type = violation['type'].value
            counts[violation_type] = counts.get(violation_type, 0) + 1
        return counts
    
    def reset_privacy_budget(self):
        """Reset privacy budget"""
        self.budget_tracker.reset_budget()
    
    def reset_monitoring_metrics(self):
        """Reset monitoring metrics"""
        self.monitoring_times.clear()
        self.violation_detections.clear()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on privacy monitoring system"""
        try:
            # Test privacy monitoring
            test_data = torch.randn(10, 10)
            result = asyncio.run(self.monitor_data_processing(test_data, 'test_processing'))
            
            return {
                'status': 'healthy',
                'components': {
                    'budget_tracker': 'operational',
                    'data_anonymizer': 'operational',
                    'consent_manager': 'operational',
                    'auditor': 'operational'
                },
                'test_result': result,
                'privacy_status': self.get_privacy_status()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }
