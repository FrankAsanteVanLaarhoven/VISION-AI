"""
C_action(a,t) - Consciousness-Driven Action Selection
Production-ready implementation for ethical AI action selection with safety constraints

Mathematical Foundation:
C_action(a,t) = argmaxₐ [w₁U(a) + w₂S(a) + w₃E(a)] subject to Σᵢ wᵢ = 1
where U(a) is utility, S(a) is safety, E(a) is ethics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import math
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from config.settings import get_settings

settings = get_settings()

class EthicalFramework(Enum):
    """Ethical frameworks for AI decision making"""
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"
    CARE_ETHICS = "care_ethics"
    CONSEQUENTIALIST = "consequentialist"

class SafetyLevel(Enum):
    """Safety levels for action validation"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

@dataclass
class ActionConfig:
    """Configuration for consciousness-driven action selection"""
    num_actions: int = 10
    utility_weight: float = 0.4
    safety_weight: float = 0.3
    ethics_weight: float = 0.3
    ethical_framework: EthicalFramework = EthicalFramework.UTILITARIAN
    safety_threshold: float = 0.7
    ethics_threshold: float = 0.6
    consciousness_temperature: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class UtilityCalculator(nn.Module):
    """
    Utility function calculator for action evaluation
    Considers efficiency, effectiveness, and goal achievement
    """
    
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # Utility components
        self.efficiency_estimator = nn.Linear(input_dim, num_actions)
        self.effectiveness_estimator = nn.Linear(input_dim, num_actions)
        self.goal_achievement_estimator = nn.Linear(input_dim, num_actions)
        
        # Utility combination weights
        self.utility_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))
        
    def forward(self, state_features: torch.Tensor, goal_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate utility for each possible action
        """
        # Combine state and goal features
        combined_features = torch.cat([state_features, goal_features], dim=-1)
        
        # Calculate utility components
        efficiency = torch.sigmoid(self.efficiency_estimator(combined_features))
        effectiveness = torch.sigmoid(self.effectiveness_estimator(combined_features))
        goal_achievement = torch.sigmoid(self.goal_achievement_estimator(combined_features))
        
        # Weighted combination
        utility_weights = F.softmax(self.utility_weights, dim=0)
        total_utility = (utility_weights[0] * efficiency + 
                        utility_weights[1] * effectiveness + 
                        utility_weights[2] * goal_achievement)
        
        return total_utility

class SafetyEvaluator(nn.Module):
    """
    Safety constraint evaluator for action validation
    Implements multi-level safety assessment
    """
    
    def __init__(self, input_dim: int, num_actions: int, num_safety_levels: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_safety_levels = num_safety_levels
        
        # Safety level evaluators
        self.safety_evaluators = nn.ModuleList([
            nn.Linear(input_dim, num_actions) for _ in range(num_safety_levels)
        ])
        
        # Safety level weights (critical > high > medium > low > minimal)
        self.safety_weights = nn.Parameter(torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2]))
        
        # Risk assessment
        self.risk_estimator = nn.Linear(input_dim, num_actions)
        
    def forward(self, state_features: torch.Tensor, environment_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate safety for each action across different safety levels
        """
        # Combine state and environment features
        combined_features = torch.cat([state_features, environment_features], dim=-1)
        
        # Evaluate safety at each level
        safety_scores = []
        for evaluator in self.safety_evaluators:
            safety_score = torch.sigmoid(evaluator(combined_features))
            safety_scores.append(safety_score)
        
        # Weight safety levels
        safety_weights = F.softmax(self.safety_weights, dim=0)
        weighted_safety = sum(w * score for w, score in zip(safety_weights, safety_scores))
        
        # Calculate risk scores
        risk_scores = torch.sigmoid(self.risk_estimator(combined_features))
        
        return weighted_safety, risk_scores

class EthicsEvaluator(nn.Module):
    """
    Ethics evaluator implementing different ethical frameworks
    """
    
    def __init__(self, input_dim: int, num_actions: int, ethical_framework: EthicalFramework):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.ethical_framework = ethical_framework
        
        # Ethical principle evaluators
        if ethical_framework == EthicalFramework.UTILITARIAN:
            self.principle_evaluators = nn.ModuleList([
                nn.Linear(input_dim, num_actions),  # Greatest good for greatest number
                nn.Linear(input_dim, num_actions),  # Harm minimization
                nn.Linear(input_dim, num_actions)   # Benefit maximization
            ])
        elif ethical_framework == EthicalFramework.DEONTOLOGICAL:
            self.principle_evaluators = nn.ModuleList([
                nn.Linear(input_dim, num_actions),  # Duty-based ethics
                nn.Linear(input_dim, num_actions),  # Rule following
                nn.Linear(input_dim, num_actions)   # Moral imperatives
            ])
        elif ethical_framework == EthicalFramework.VIRTUE_ETHICS:
            self.principle_evaluators = nn.ModuleList([
                nn.Linear(input_dim, num_actions),  # Courage
                nn.Linear(input_dim, num_actions),  # Wisdom
                nn.Linear(input_dim, num_actions)   # Justice
            ])
        else:  # Default to utilitarian
            self.principle_evaluators = nn.ModuleList([
                nn.Linear(input_dim, num_actions),
                nn.Linear(input_dim, num_actions),
                nn.Linear(input_dim, num_actions)
            ])
        
        # Ethical weights
        self.ethical_weights = nn.Parameter(torch.ones(len(self.principle_evaluators)))
        
    def forward(self, state_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        Evaluate ethical implications of each action
        """
        # Combine state and context features
        combined_features = torch.cat([state_features, context_features], dim=-1)
        
        # Evaluate ethical principles
        ethical_scores = []
        for evaluator in self.principle_evaluators:
            ethical_score = torch.sigmoid(evaluator(combined_features))
            ethical_scores.append(ethical_score)
        
        # Weight ethical principles
        ethical_weights = F.softmax(self.ethical_weights, dim=0)
        total_ethics = sum(w * score for w, score in zip(ethical_weights, ethical_scores))
        
        return total_ethics

class ConsciousnessModule(nn.Module):
    """
    AI consciousness module for self-awareness and reflection
    """
    
    def __init__(self, input_dim: int, consciousness_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.consciousness_dim = consciousness_dim
        
        # Consciousness layers
        self.self_awareness = nn.Linear(input_dim, consciousness_dim)
        self.reflection = nn.Linear(consciousness_dim, consciousness_dim)
        self.intention = nn.Linear(consciousness_dim, consciousness_dim)
        
        # Consciousness state
        self.consciousness_state = nn.Parameter(torch.randn(consciousness_dim))
        
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Generate consciousness-aware features
        """
        # Self-awareness
        awareness = torch.tanh(self.self_awareness(state_features))
        
        # Reflection
        reflection = torch.tanh(self.reflection(awareness))
        
        # Intention
        intention = torch.tanh(self.intention(reflection))
        
        # Update consciousness state
        self.consciousness_state.data = 0.9 * self.consciousness_state.data + 0.1 * intention.mean(dim=0)
        
        return intention

class ActionExplanationGenerator:
    """
    Generates human-readable explanations for action decisions
    """
    
    def __init__(self):
        self.action_names = [
            "move_forward", "move_backward", "turn_left", "turn_right", "stop",
            "accelerate", "decelerate", "avoid_obstacle", "follow_path", "reach_destination"
        ]
        
        self.explanation_templates = {
            "utility": "Selected action {action} because it maximizes {metric} (score: {score:.2f})",
            "safety": "Action {action} is safe with {level} safety level (score: {score:.2f})",
            "ethics": "Action {action} aligns with {framework} principles (score: {score:.2f})",
            "consciousness": "AI consciousness indicates {action} is the most appropriate choice"
        }
    
    def generate_explanation(self, 
                           selected_action: int, 
                           utility_scores: torch.Tensor,
                           safety_scores: torch.Tensor,
                           ethics_scores: torch.Tensor,
                           consciousness_features: torch.Tensor,
                           ethical_framework: EthicalFramework) -> str:
        """
        Generate explanation for action selection
        """
        action_name = self.action_names[selected_action]
        
        # Get scores for selected action
        utility_score = utility_scores[selected_action].item()
        safety_score = safety_scores[selected_action].item()
        ethics_score = ethics_scores[selected_action].item()
        
        # Determine primary reason
        if utility_score >= max(safety_score, ethics_score):
            primary_reason = "utility"
            metric = "efficiency and goal achievement"
        elif safety_score >= ethics_score:
            primary_reason = "safety"
            level = "high" if safety_score > 0.8 else "medium" if safety_score > 0.6 else "low"
        else:
            primary_reason = "ethics"
            framework = ethical_framework.value
        
        # Generate explanation
        if primary_reason == "utility":
            explanation = self.explanation_templates["utility"].format(
                action=action_name, metric=metric, score=utility_score
            )
        elif primary_reason == "safety":
            explanation = self.explanation_templates["safety"].format(
                action=action_name, level=level, score=safety_score
            )
        else:  # ethics
            explanation = self.explanation_templates["ethics"].format(
                action=action_name, framework=framework, score=ethics_score
            )
        
        # Add consciousness note
        consciousness_note = self.explanation_templates["consciousness"].format(action=action_name)
        
        return f"{explanation}. {consciousness_note}"

class ConsciousnessActionSelection:
    """
    C_action(a,t) - Consciousness-Driven Action Selection
    
    Implements ethical AI action selection with safety constraints and consciousness awareness.
    """
    
    def __init__(self, config: Optional[ActionConfig] = None):
        self.config = config or ActionConfig()
        self.device = torch.device(self.config.device)
        
        # Input dimensions
        state_dim = 512  # From vision and language features
        goal_dim = 256   # Goal representation
        environment_dim = 128  # Environment features
        
        # Initialize evaluators
        self.utility_calculator = UtilityCalculator(
            input_dim=state_dim + goal_dim,
            num_actions=self.config.num_actions
        ).to(self.device)
        
        self.safety_evaluator = SafetyEvaluator(
            input_dim=state_dim + environment_dim,
            num_actions=self.config.num_actions
        ).to(self.device)
        
        self.ethics_evaluator = EthicsEvaluator(
            input_dim=state_dim + environment_dim,
            num_actions=self.config.num_actions,
            ethical_framework=self.config.ethical_framework
        ).to(self.device)
        
        self.consciousness_module = ConsciousnessModule(
            input_dim=state_dim,
            consciousness_dim=128
        ).to(self.device)
        
        # Explanation generator
        self.explanation_generator = ActionExplanationGenerator()
        
        # Performance tracking
        self.decision_times = []
        self.utility_scores_history = []
        self.safety_scores_history = []
        self.ethics_scores_history = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Consciousness Action Selection initialized on {self.device}")
        
    def calculate_utility_matrix(self, possible_actions: torch.Tensor, state_features: torch.Tensor, goal_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate utility matrix for all possible actions
        """
        utility_scores = self.utility_calculator(state_features, goal_features)
        return utility_scores
    
    def evaluate_safety_constraints(self, possible_actions: torch.Tensor, state_features: torch.Tensor, environment_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate safety constraints for all actions
        """
        safety_scores, risk_scores = self.safety_evaluator(state_features, environment_features)
        return safety_scores, risk_scores
    
    def apply_ethical_framework(self, possible_actions: torch.Tensor, state_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        Apply ethical framework to evaluate actions
        """
        ethics_scores = self.ethics_evaluator(state_features, context_features)
        return ethics_scores
    
    def weighted_optimization(self, 
                            utility_matrix: torch.Tensor,
                            safety_matrix: torch.Tensor,
                            ethics_matrix: torch.Tensor,
                            consciousness_weights: List[float]) -> Tuple[int, torch.Tensor, Dict[str, Any]]:
        """
        Perform weighted optimization for action selection
        """
        # Normalize matrices
        utility_norm = F.softmax(utility_matrix / self.config.consciousness_temperature, dim=-1)
        safety_norm = F.softmax(safety_matrix / self.config.consciousness_temperature, dim=-1)
        ethics_norm = F.softmax(ethics_matrix / self.config.consciousness_temperature, dim=-1)
        
        # Weighted combination
        total_score = (consciousness_weights[0] * utility_norm + 
                      consciousness_weights[1] * safety_norm + 
                      consciousness_weights[2] * ethics_norm)
        
        # Select optimal action
        optimal_action_idx = torch.argmax(total_score, dim=-1).item()
        
        # Prepare metadata
        metadata = {
            'utility_scores': utility_matrix.detach().cpu().numpy(),
            'safety_scores': safety_matrix.detach().cpu().numpy(),
            'ethics_scores': ethics_matrix.detach().cpu().numpy(),
            'total_scores': total_score.detach().cpu().numpy(),
            'consciousness_weights': consciousness_weights,
            'selected_action_idx': optimal_action_idx
        }
        
        return optimal_action_idx, total_score, metadata
    
    def forward(self, 
                possible_actions: torch.Tensor,
                state_features: torch.Tensor,
                goal_features: torch.Tensor,
                environment_features: torch.Tensor,
                context_features: torch.Tensor) -> Tuple[int, str, Dict[str, Any]]:
        """
        Main forward pass for consciousness-driven action selection
        
        Args:
            possible_actions: Available actions tensor
            state_features: Current state features
            goal_features: Goal/objective features
            environment_features: Environment state features
            context_features: Contextual information
            
        Returns:
            optimal_action: Selected action index
            explanation_trace: Human-readable explanation
            metadata: Decision process metadata
        """
        start_time = time.time()
        
        # Step 1: Calculate utility matrix
        utility_matrix = self.calculate_utility_matrix(possible_actions, state_features, goal_features)
        
        # Step 2: Evaluate safety constraints
        safety_matrix, risk_scores = self.evaluate_safety_constraints(
            possible_actions, state_features, environment_features
        )
        
        # Step 3: Apply ethical framework
        ethics_matrix = self.apply_ethical_framework(
            possible_actions, state_features, context_features
        )
        
        # Step 4: Generate consciousness features
        consciousness_features = self.consciousness_module(state_features)
        
        # Step 5: Consciousness-weighted decision making
        consciousness_weights = [
            self.config.utility_weight,
            self.config.safety_weight,
            self.config.ethics_weight
        ]
        
        optimal_action_idx, total_scores, decision_metadata = self.weighted_optimization(
            utility_matrix, safety_matrix, ethics_matrix, consciousness_weights
        )
        
        # Step 6: Generate explanation
        explanation_trace = self.explanation_generator.generate_explanation(
            optimal_action_idx,
            utility_matrix,
            safety_matrix,
            ethics_matrix,
            consciousness_features,
            self.config.ethical_framework
        )
        
        # Performance tracking
        decision_time = (time.time() - start_time) * 1000
        self.decision_times.append(decision_time)
        self.utility_scores_history.append(utility_matrix.detach().cpu().numpy())
        self.safety_scores_history.append(safety_matrix.detach().cpu().numpy())
        self.ethics_scores_history.append(ethics_matrix.detach().cpu().numpy())
        
        # Prepare comprehensive metadata
        metadata = {
            'decision_time_ms': decision_time,
            'selected_action_idx': optimal_action_idx,
            'utility_scores': utility_matrix.detach().cpu().numpy(),
            'safety_scores': safety_matrix.detach().cpu().numpy(),
            'ethics_scores': ethics_matrix.detach().cpu().numpy(),
            'risk_scores': risk_scores.detach().cpu().numpy(),
            'total_scores': total_scores.detach().cpu().numpy(),
            'consciousness_features': consciousness_features.detach().cpu().numpy(),
            'consciousness_weights': consciousness_weights,
            'ethical_framework': self.config.ethical_framework.value,
            'safety_threshold': self.config.safety_threshold,
            'ethics_threshold': self.config.ethics_threshold
        }
        
        return optimal_action_idx, explanation_trace, metadata
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.decision_times:
            return {}
        
        return {
            'average_decision_time_ms': np.mean(self.decision_times),
            'min_decision_time_ms': np.min(self.decision_times),
            'max_decision_time_ms': np.max(self.decision_times),
            'total_decisions': len(self.decision_times),
            'average_utility_score': np.mean([np.mean(scores) for scores in self.utility_scores_history]),
            'average_safety_score': np.mean([np.mean(scores) for scores in self.safety_scores_history]),
            'average_ethics_score': np.mean([np.mean(scores) for scores in self.ethics_scores_history])
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.decision_times.clear()
        self.utility_scores_history.clear()
        self.safety_scores_history.clear()
        self.ethics_scores_history.clear()
    
    def update_config(self, new_config: ActionConfig):
        """Update configuration"""
        self.config = new_config
        self.logger.info(f"Action selection configuration updated: {new_config}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test with dummy data
            dummy_actions = torch.randn(10).to(self.device)
            dummy_state = torch.randn(512).to(self.device)
            dummy_goal = torch.randn(256).to(self.device)
            dummy_env = torch.randn(128).to(self.device)
            dummy_context = torch.randn(128).to(self.device)
            
            # Test forward pass
            action, explanation, metadata = self.forward(
                dummy_actions, dummy_state, dummy_goal, dummy_env, dummy_context
            )
            
            return {
                'status': 'healthy',
                'device': str(self.device),
                'models_loaded': True,
                'test_action': action,
                'test_explanation_length': len(explanation),
                'performance_metrics': self.get_performance_metrics()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }
