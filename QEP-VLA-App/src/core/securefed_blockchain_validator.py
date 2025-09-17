"""
SecureFed Blockchain Validator - Bo-Wei Integration
Implements blockchain-based federated learning defense from SecureFed research

Features:
- Cosine similarity validation
- Blockchain consensus mechanism
- Malicious client detection (30% threshold)
- Secure model aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import hashlib
import json
import asyncio
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import math

from config.settings import get_settings

settings = get_settings()

class ValidationStatus(Enum):
    """Validation status for model updates"""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    PENDING = "pending"

class ConsensusType(Enum):
    """Types of consensus mechanisms"""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    THRESHOLD = "threshold"

@dataclass
class ModelUpdate:
    """Model update with validation metadata"""
    client_id: str
    model_params: Dict[str, torch.Tensor]
    sample_count: int
    timestamp: float
    validation_hash: str
    cosine_similarity: float
    validation_status: ValidationStatus
    blockchain_proof: Optional[str] = None

@dataclass
class SecureFedConfig:
    """Configuration for SecureFed blockchain validator"""
    cosine_similarity_threshold: float = 0.85
    malicious_client_threshold: float = 0.3  # 30% threshold
    consensus_threshold: float = 0.6
    blockchain_validation_enabled: bool = True
    max_validators: int = 10
    validation_timeout: float = 30.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class CosineSimilarityValidator:
    """
    Cosine similarity validator for model updates
    Implements the core validation mechanism from SecureFed
    """
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.similarity_history = defaultdict(list)
        
    def compute_cosine_similarity(self, 
                                model_update: Dict[str, torch.Tensor],
                                global_model: Dict[str, torch.Tensor]) -> float:
        """
        Compute cosine similarity between model update and global model
        """
        similarities = []
        
        for param_name in global_model.keys():
            if param_name in model_update:
                # Flatten parameters
                update_param = model_update[param_name].flatten()
                global_param = global_model[param_name].flatten()
                
                # Ensure same length
                min_len = min(len(update_param), len(global_param))
                update_param = update_param[:min_len]
                global_param = global_param[:min_len]
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    update_param.unsqueeze(0), 
                    global_param.unsqueeze(0)
                ).item()
                
                similarities.append(similarity)
        
        # Average similarity across all parameters
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return avg_similarity
    
    def validate_update(self, 
                       model_update: Dict[str, torch.Tensor],
                       global_model: Dict[str, torch.Tensor],
                       client_id: str) -> Tuple[bool, float]:
        """
        Validate model update using cosine similarity
        """
        similarity = self.compute_cosine_similarity(model_update, global_model)
        
        # Store similarity history
        self.similarity_history[client_id].append(similarity)
        
        # Check if similarity meets threshold
        is_valid = similarity >= self.threshold
        
        return is_valid, similarity
    
    def detect_malicious_client(self, client_id: str) -> bool:
        """
        Detect malicious client based on similarity history
        """
        if client_id not in self.similarity_history:
            return False
        
        similarities = self.similarity_history[client_id]
        if len(similarities) < 5:  # Need minimum history
            return False
        
        # Check if recent similarities are consistently low
        recent_similarities = similarities[-5:]
        avg_recent_similarity = np.mean(recent_similarities)
        
        # Client is malicious if average similarity is below threshold
        return avg_recent_similarity < self.threshold

class BlockchainConsensus:
    """
    Blockchain consensus mechanism for model validation
    Implements distributed validation from SecureFed framework
    """
    
    def __init__(self, max_validators: int = 10):
        self.max_validators = max_validators
        self.validators = {}
        self.consensus_history = []
        self.blockchain_state = self._initialize_blockchain()
        
    def _initialize_blockchain(self) -> str:
        """Initialize blockchain state"""
        timestamp = str(int(time.time()))
        return hashlib.sha256(f"securefed_blockchain_{timestamp}".encode()).hexdigest()
    
    def add_validator(self, validator_id: str, public_key: str) -> bool:
        """
        Add a new validator to the consensus network
        """
        if len(self.validators) >= self.max_validators:
            return False
        
        self.validators[validator_id] = {
            'public_key': public_key,
            'stake': 1.0,  # Equal stake for all validators
            'reputation': 1.0,
            'validation_count': 0
        }
        
        return True
    
    def generate_validation_proof(self, 
                                model_hash: str,
                                validation_result: bool,
                                validator_id: str) -> str:
        """
        Generate blockchain proof for validation
        """
        timestamp = time.time()
        proof_data = {
            'model_hash': model_hash,
            'validation_result': validation_result,
            'validator_id': validator_id,
            'timestamp': timestamp,
            'blockchain_state': self.blockchain_state
        }
        
        # Create proof hash
        proof_string = json.dumps(proof_data, sort_keys=True)
        proof_hash = hashlib.sha256(proof_string.encode()).hexdigest()
        
        # Update blockchain state
        self.blockchain_state = hashlib.sha256(
            f"{self.blockchain_state}_{proof_hash}".encode()
        ).hexdigest()
        
        return proof_hash
    
    def reach_consensus(self, 
                       validation_results: List[Tuple[str, bool, str]]) -> Tuple[bool, float]:
        """
        Reach consensus on model validation
        """
        if not validation_results:
            return False, 0.0
        
        # Count votes
        valid_votes = 0
        total_votes = len(validation_results)
        weighted_votes = 0.0
        total_weight = 0.0
        
        for validator_id, is_valid, proof in validation_results:
            if validator_id in self.validators:
                validator = self.validators[validator_id]
                weight = validator['stake'] * validator['reputation']
                
                if is_valid:
                    valid_votes += 1
                    weighted_votes += weight
                
                total_weight += weight
        
        # Simple majority consensus
        majority_consensus = valid_votes > total_votes / 2
        
        # Weighted consensus
        weighted_consensus = weighted_votes > total_weight / 2
        
        # Final consensus (both must agree)
        final_consensus = majority_consensus and weighted_consensus
        consensus_confidence = weighted_votes / total_weight if total_weight > 0 else 0.0
        
        return final_consensus, consensus_confidence

class SecureFedBlockchainValidator:
    """
    SecureFed Blockchain Validator implementing the complete framework
    
    Features:
    - Cosine similarity validation
    - Blockchain consensus mechanism
    - Malicious client detection
    - Secure model aggregation
    """
    
    def __init__(self, config: Optional[SecureFedConfig] = None):
        self.config = config or SecureFedConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize components
        self.cosine_validator = CosineSimilarityValidator(
            threshold=self.config.cosine_similarity_threshold
        )
        self.blockchain_consensus = BlockchainConsensus(
            max_validators=self.config.max_validators
        )
        
        # Global model state
        self.global_model = {}
        self.model_update_history = []
        self.client_reputation = defaultdict(lambda: 1.0)
        
        # Performance tracking
        self.validation_times = []
        self.consensus_times = []
        self.malicious_detections = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SecureFed Blockchain Validator initialized on {self.device}")
    
    def set_global_model(self, global_model: Dict[str, torch.Tensor]):
        """
        Set the current global model for validation
        """
        self.global_model = global_model
        self.logger.info("Global model updated for validation")
    
    def validate_model_update(self, 
                            client_id: str,
                            model_update: Dict[str, torch.Tensor],
                            sample_count: int) -> ModelUpdate:
        """
        Validate a single model update using SecureFed methodology
        """
        start_time = time.time()
        
        # Step 1: Cosine similarity validation
        is_valid, cosine_similarity = self.cosine_validator.validate_update(
            model_update, self.global_model, client_id
        )
        
        # Step 2: Check for malicious client
        is_malicious = self.cosine_validator.detect_malicious_client(client_id)
        
        # Step 3: Determine validation status
        if is_malicious:
            validation_status = ValidationStatus.INVALID
        elif is_valid:
            validation_status = ValidationStatus.VALID
        else:
            validation_status = ValidationStatus.SUSPICIOUS
        
        # Step 4: Generate model hash
        model_hash = self._generate_model_hash(model_update)
        
        # Step 5: Create model update object
        model_update_obj = ModelUpdate(
            client_id=client_id,
            model_params=model_update,
            sample_count=sample_count,
            timestamp=time.time(),
            validation_hash=model_hash,
            cosine_similarity=cosine_similarity,
            validation_status=validation_status
        )
        
        # Step 6: Blockchain validation (if enabled)
        if self.config.blockchain_validation_enabled:
            blockchain_proof = self._blockchain_validate(model_update_obj)
            model_update_obj.blockchain_proof = blockchain_proof
        
        # Track performance
        validation_time = (time.time() - start_time) * 1000
        self.validation_times.append(validation_time)
        
        # Update client reputation
        self._update_client_reputation(client_id, validation_status)
        
        # Store in history
        self.model_update_history.append(model_update_obj)
        
        return model_update_obj
    
    def _generate_model_hash(self, model_params: Dict[str, torch.Tensor]) -> str:
        """
        Generate hash for model parameters
        """
        # Convert model to string representation
        model_string = ""
        for param_name, param_tensor in sorted(model_params.items()):
            model_string += f"{param_name}:{param_tensor.flatten().tolist()}"
        
        # Generate hash
        model_hash = hashlib.sha256(model_string.encode()).hexdigest()
        return model_hash
    
    def _blockchain_validate(self, model_update: ModelUpdate) -> str:
        """
        Perform blockchain validation
        """
        # Simulate blockchain validation
        # In a real implementation, this would interact with a blockchain network
        
        validation_data = {
            'model_hash': model_update.validation_hash,
            'client_id': model_update.client_id,
            'cosine_similarity': model_update.cosine_similarity,
            'timestamp': model_update.timestamp
        }
        
        # Generate blockchain proof
        proof_string = json.dumps(validation_data, sort_keys=True)
        blockchain_proof = hashlib.sha256(proof_string.encode()).hexdigest()
        
        return blockchain_proof
    
    def _update_client_reputation(self, client_id: str, validation_status: ValidationStatus):
        """
        Update client reputation based on validation results
        """
        current_reputation = self.client_reputation[client_id]
        
        if validation_status == ValidationStatus.VALID:
            # Increase reputation for valid updates
            self.client_reputation[client_id] = min(1.0, current_reputation + 0.1)
        elif validation_status == ValidationStatus.INVALID:
            # Decrease reputation for invalid updates
            self.client_reputation[client_id] = max(0.0, current_reputation - 0.2)
        elif validation_status == ValidationStatus.SUSPICIOUS:
            # Slight decrease for suspicious updates
            self.client_reputation[client_id] = max(0.0, current_reputation - 0.05)
    
    async def validate_batch_updates(self, 
                                   batch_updates: List[Tuple[str, Dict[str, torch.Tensor], int]]) -> List[ModelUpdate]:
        """
        Validate a batch of model updates asynchronously
        """
        validation_tasks = []
        
        for client_id, model_update, sample_count in batch_updates:
            task = asyncio.create_task(
                self._async_validate_update(client_id, model_update, sample_count)
            )
            validation_tasks.append(task)
        
        # Wait for all validations to complete
        validated_updates = await asyncio.gather(*validation_tasks)
        
        return validated_updates
    
    async def _async_validate_update(self, 
                                   client_id: str,
                                   model_update: Dict[str, torch.Tensor],
                                   sample_count: int) -> ModelUpdate:
        """
        Asynchronous model update validation
        """
        # Run validation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.validate_model_update, 
            client_id, 
            model_update, 
            sample_count
        )
    
    def secure_aggregate(self, 
                        validated_updates: List[ModelUpdate],
                        aggregation_method: str = "weighted_average") -> Dict[str, torch.Tensor]:
        """
        Securely aggregate validated model updates
        """
        if not validated_updates:
            raise ValueError("No validated updates for aggregation")
        
        # Filter only valid updates
        valid_updates = [update for update in validated_updates 
                        if update.validation_status == ValidationStatus.VALID]
        
        if not valid_updates:
            raise ValueError("No valid updates for aggregation")
        
        # Choose aggregation method
        if aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(valid_updates)
        elif aggregation_method == "reputation_weighted":
            return self._reputation_weighted_aggregation(valid_updates)
        elif aggregation_method == "median":
            return self._median_aggregation(valid_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def _weighted_average_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """
        Weighted average aggregation based on sample count
        """
        total_samples = sum(update.sample_count for update in updates)
        
        global_model = {}
        for param_name in updates[0].model_params.keys():
            param_sum = torch.zeros_like(updates[0].model_params[param_name])
            
            for update in updates:
                weight = update.sample_count / total_samples
                param_sum += update.model_params[param_name] * weight
            
            global_model[param_name] = param_sum
        
        return global_model
    
    def _reputation_weighted_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """
        Reputation-weighted aggregation
        """
        total_reputation = sum(self.client_reputation[update.client_id] for update in updates)
        
        global_model = {}
        for param_name in updates[0].model_params.keys():
            param_sum = torch.zeros_like(updates[0].model_params[param_name])
            
            for update in updates:
                reputation_weight = self.client_reputation[update.client_id] / total_reputation
                param_sum += update.model_params[param_name] * reputation_weight
            
            global_model[param_name] = param_sum
        
        return global_model
    
    def _median_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """
        Median-based aggregation for robustness
        """
        global_model = {}
        for param_name in updates[0].model_params.keys():
            param_values = [update.model_params[param_name] for update in updates]
            
            # Stack parameters and compute median
            stacked_params = torch.stack(param_values, dim=0)
            median_params = torch.median(stacked_params, dim=0)[0]
            
            global_model[param_name] = median_params
        
        return global_model
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive validation metrics"""
        if not self.validation_times:
            return {}
        
        # Calculate malicious client percentage
        total_clients = len(self.client_reputation)
        malicious_clients = sum(1 for rep in self.client_reputation.values() if rep < 0.5)
        malicious_percentage = (malicious_clients / total_clients * 100) if total_clients > 0 else 0
        
        return {
            'total_validations': len(self.validation_times),
            'average_validation_time_ms': np.mean(self.validation_times),
            'min_validation_time_ms': np.min(self.validation_times),
            'max_validation_time_ms': np.max(self.validation_times),
            'total_clients': total_clients,
            'malicious_clients': malicious_clients,
            'malicious_percentage': malicious_percentage,
            'average_client_reputation': np.mean(list(self.client_reputation.values())),
            'consensus_enabled': self.config.blockchain_validation_enabled,
            'cosine_similarity_threshold': self.config.cosine_similarity_threshold
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.validation_times.clear()
        self.consensus_times.clear()
        self.malicious_detections.clear()
        self.model_update_history.clear()
    
    def update_config(self, new_config: SecureFedConfig):
        """Update configuration"""
        self.config = new_config
        self.cosine_validator.threshold = new_config.cosine_similarity_threshold
        self.logger.info(f"SecureFed configuration updated: {new_config}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test with dummy model update
            dummy_model = {
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10)
            }
            
            # Set dummy global model
            self.set_global_model(dummy_model)
            
            # Test validation
            model_update = self.validate_model_update(
                'test_client', dummy_model, 100
            )
            
            return {
                'status': 'healthy',
                'device': str(self.device),
                'validation_successful': model_update.validation_status == ValidationStatus.VALID,
                'cosine_similarity': model_update.cosine_similarity,
                'validation_metrics': self.get_validation_metrics()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }
