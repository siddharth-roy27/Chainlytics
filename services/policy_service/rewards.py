"""
Reward functions for logistics reinforcement learning
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import yaml
import os


@dataclass
class RewardComponents:
    """Individual components of the reward function"""
    service_level: float = 0.0
    cost_efficiency: float = 0.0
    resource_utilization: float = 0.0
    delay_penalty: float = 0.0
    customer_satisfaction: float = 0.0
    sustainability: float = 0.0
    safety_compliance: float = 0.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        'service_level': 0.25,
        'cost_efficiency': 0.20,
        'resource_utilization': 0.15,
        'delay_penalty': -0.20,
        'customer_satisfaction': 0.15,
        'sustainability': 0.10,
        'safety_compliance': 0.15
    })


class BaseRewardFunction(ABC):
    """Base class for reward functions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize reward function
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def compute_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                      next_state: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        """
        Compute reward based on state transition
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        pass


class LogisticsRewardFunction(BaseRewardFunction):
    """Reward function for logistics operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize logistics reward function
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Default weights
        self.weights = self.config.get('weights', {
            'service_level': 0.25,
            'cost_efficiency': 0.20,
            'resource_utilization': 0.15,
            'delay_penalty': -0.20,
            'customer_satisfaction': 0.15,
            'sustainability': 0.10,
            'safety_compliance': 0.15
        })
        
        # Thresholds and parameters
        self.service_level_target = self.config.get('service_level_target', 0.95)
        self.cost_efficiency_baseline = self.config.get('cost_efficiency_baseline', 1.0)
        self.resource_utilization_target = self.config.get('resource_utilization_target', 0.8)
    
    def compute_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                      next_state: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        """
        Compute reward based on logistics performance metrics
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        components = RewardComponents()
        
        # 1. Service Level Reward
        components.service_level = self._compute_service_level_reward(next_state)
        
        # 2. Cost Efficiency Reward
        components.cost_efficiency = self._compute_cost_efficiency_reward(state, action, next_state)
        
        # 3. Resource Utilization Reward
        components.resource_utilization = self._compute_resource_utilization_reward(next_state)
        
        # 4. Delay Penalty
        components.delay_penalty = self._compute_delay_penalty(next_state)
        
        # 5. Customer Satisfaction Reward
        components.customer_satisfaction = self._compute_customer_satisfaction_reward(next_state)
        
        # 6. Sustainability Reward
        components.sustainability = self._compute_sustainability_reward(state, action, next_state)
        
        # 7. Safety Compliance Reward
        components.safety_compliance = self._compute_safety_compliance_reward(next_state)
        
        # Compute weighted total reward
        total_reward = (
            components.service_level * self.weights['service_level'] +
            components.cost_efficiency * self.weights['cost_efficiency'] +
            components.resource_utilization * self.weights['resource_utilization'] +
            components.delay_penalty * self.weights['delay_penalty'] +
            components.customer_satisfaction * self.weights['customer_satisfaction'] +
            components.sustainability * self.weights['sustainability'] +
            components.safety_compliance * self.weights['safety_compliance']
        )
        
        return total_reward, components
    
    def _compute_service_level_reward(self, state: Dict[str, Any]) -> float:
        """Compute service level reward component"""
        # Service level = fulfilled_orders / total_orders
        fulfilled_orders = state.get('fulfilled_orders', 0)
        total_orders = state.get('total_orders', 1)  # Avoid division by zero
        
        if total_orders == 0:
            return 0.0
        
        service_level = fulfilled_orders / total_orders
        
        # Reward increases as service level approaches target
        # Using a sigmoid-like function for smooth transitions
        deviation = abs(service_level - self.service_level_target)
        reward = np.exp(-deviation * 5)  # Steeper penalty for deviations
        
        return reward
    
    def _compute_cost_efficiency_reward(self, state: Dict[str, Any], 
                                     action: Dict[str, Any], 
                                     next_state: Dict[str, Any]) -> float:
        """Compute cost efficiency reward component"""
        # Cost efficiency = baseline_cost / actual_cost
        actual_cost = next_state.get('total_cost', 1.0)
        
        if actual_cost <= 0:
            return 0.0
        
        efficiency = self.cost_efficiency_baseline / actual_cost
        
        # Cap efficiency reward to prevent extreme values
        efficiency = min(efficiency, 2.0)  # Max 2x efficiency bonus
        
        # Normalize to [0, 1] range
        normalized_efficiency = min(efficiency, 1.0)
        
        return normalized_efficiency
    
    def _compute_resource_utilization_reward(self, state: Dict[str, Any]) -> float:
        """Compute resource utilization reward component"""
        # Average utilization across all resources
        utilizations = state.get('resource_utilizations', [])
        
        if not utilizations:
            return 0.0
        
        avg_utilization = np.mean(utilizations)
        
        # Reward for being close to target utilization
        # Too low = underutilization, too high = overutilization
        deviation = abs(avg_utilization - self.resource_utilization_target)
        reward = np.exp(-deviation * 3)  # Exponential decay for deviation penalty
        
        return reward
    
    def _compute_delay_penalty(self, state: Dict[str, Any]) -> float:
        """Compute delay penalty component"""
        # Total delay hours across all shipments
        total_delay_hours = state.get('total_delay_hours', 0)
        
        # Exponential penalty for delays
        # Small delays have minimal penalty, large delays have significant penalty
        penalty = 1.0 - np.exp(-total_delay_hours / 10.0)
        
        return -abs(penalty)  # Negative reward (penalty)
    
    def _compute_customer_satisfaction_reward(self, state: Dict[str, Any]) -> float:
        """Compute customer satisfaction reward component"""
        # Average customer satisfaction score
        avg_satisfaction = state.get('avg_customer_satisfaction', 0.0)
        
        # Normalize to [0, 1] assuming satisfaction is on 0-100 scale
        normalized_satisfaction = avg_satisfaction / 100.0
        
        return normalized_satisfaction
    
    def _compute_sustainability_reward(self, state: Dict[str, Any], 
                                    action: Dict[str, Any], 
                                    next_state: Dict[str, Any]) -> float:
        """Compute sustainability reward component"""
        # Carbon emissions reduction compared to baseline
        current_emissions = next_state.get('carbon_emissions', 0.0)
        baseline_emissions = state.get('carbon_emissions', 1.0)
        
        if baseline_emissions <= 0:
            return 0.0
        
        # Reduction ratio (positive when emissions decrease)
        reduction_ratio = (baseline_emissions - current_emissions) / baseline_emissions
        
        # Normalize to [0, 1] range, with bonus for emissions reduction
        sustainability_score = max(0.0, min(1.0, 0.5 + reduction_ratio * 0.5))
        
        return sustainability_score
    
    def _compute_safety_compliance_reward(self, state: Dict[str, Any]) -> float:
        """Compute safety compliance reward component"""
        # Safety incidents (lower is better)
        safety_incidents = state.get('safety_incidents', 0)
        total_operations = state.get('total_operations', 1)
        
        if total_operations == 0:
            return 0.0
        
        # Safety rate (higher is better)
        safety_rate = 1.0 - (safety_incidents / total_operations)
        
        # Ensure non-negative reward
        return max(0.0, safety_rate)


class MultiObjectiveReward:
    """Combine multiple reward objectives"""
    
    def __init__(self, reward_functions: List[Tuple[BaseRewardFunction, float]]):
        """
        Initialize multi-objective reward
        
        Args:
            reward_functions: List of (reward_function, weight) tuples
        """
        self.reward_functions = reward_functions
    
    def compute_combined_reward(self, state: Dict[str, Any], 
                             action: Dict[str, Any], 
                             next_state: Dict[str, Any]) -> Tuple[float, List[RewardComponents]]:
        """
        Compute combined reward from multiple objectives
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Tuple of (combined_reward, list_of_components)
        """
        total_reward = 0.0
        all_components = []
        
        for reward_func, weight in self.reward_functions:
            reward, components = reward_func.compute_reward(state, action, next_state)
            total_reward += reward * weight
            all_components.append(components)
        
        return total_reward, all_components


class RewardShaping:
    """Apply reward shaping techniques"""
    
    def __init__(self):
        self.potential_functions = {}
    
    def add_potential_function(self, name: str, func: Callable[[Dict], float]):
        """
        Add a potential function for reward shaping
        
        Args:
            name: Name of the potential function
            func: Function that maps state to potential value
        """
        self.potential_functions[name] = func
    
    def compute_shaped_reward(self, state: Dict[str, Any], 
                           action: Dict[str, Any], 
                           next_state: Dict[str, Any],
                           base_reward: float,
                           gamma: float = 0.99) -> float:
        """
        Compute shaped reward using potential-based reward shaping
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            base_reward: Original reward
            gamma: Discount factor
            
        Returns:
            Shaped reward
        """
        shaped_reward = base_reward
        
        # Add potential differences for each potential function
        for potential_func in self.potential_functions.values():
            current_potential = potential_func(state)
            next_potential = potential_func(next_state)
            shaped_reward += gamma * next_potential - current_potential
        
        return shaped_reward


def load_reward_config(config_path: str) -> Dict[str, Any]:
    """
    Load reward configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        # Return default configuration
        return {
            'weights': {
                'service_level': 0.25,
                'cost_efficiency': 0.20,
                'resource_utilization': 0.15,
                'delay_penalty': -0.20,
                'customer_satisfaction': 0.15,
                'sustainability': 0.10,
                'safety_compliance': 0.15
            },
            'service_level_target': 0.95,
            'cost_efficiency_baseline': 1.0,
            'resource_utilization_target': 0.8
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Example usage
if __name__ == "__main__":
    # Create reward function
    reward_config = {
        'weights': {
            'service_level': 0.3,
            'cost_efficiency': 0.25,
            'resource_utilization': 0.15,
            'delay_penalty': -0.25,
            'customer_satisfaction': 0.2,
            'sustainability': 0.1,
            'safety_compliance': 0.15
        }
    }
    
    reward_func = LogisticsRewardFunction(reward_config)
    
    # Sample state transition
    state = {
        'fulfilled_orders': 90,
        'total_orders': 100,
        'total_cost': 5000,
        'resource_utilizations': [0.75, 0.85, 0.80],
        'total_delay_hours': 5,
        'avg_customer_satisfaction': 85,
        'carbon_emissions': 1000,
        'safety_incidents': 1,
        'total_operations': 100
    }
    
    action = {
        'route_changes': 3,
        'inventory_adjustments': 2
    }
    
    next_state = {
        'fulfilled_orders': 95,
        'total_orders': 100,
        'total_cost': 4800,
        'resource_utilizations': [0.80, 0.82, 0.78],
        'total_delay_hours': 3,
        'avg_customer_satisfaction': 88,
        'carbon_emissions': 950,
        'safety_incidents': 0,
        'total_operations': 100
    }
    
    # Compute reward
    total_reward, components = reward_func.compute_reward(state, action, next_state)
    
    print("Reward Components:")
    print(f"  Service Level: {components.service_level:.3f}")
    print(f"  Cost Efficiency: {components.cost_efficiency:.3f}")
    print(f"  Resource Utilization: {components.resource_utilization:.3f}")
    print(f"  Delay Penalty: {components.delay_penalty:.3f}")
    print(f"  Customer Satisfaction: {components.customer_satisfaction:.3f}")
    print(f"  Sustainability: {components.sustainability:.3f}")
    print(f"  Safety Compliance: {components.safety_compliance:.3f}")
    print(f"Total Reward: {total_reward:.3f}")
</file>