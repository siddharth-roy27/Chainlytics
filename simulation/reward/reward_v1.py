"""
Version 1 reward functions for logistics simulation
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging


class RewardFunctionV1:
    """Version 1 reward function for logistics operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward function
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('RewardFunctionV1')
        self.weights = self._initialize_weights()
        self.baseline_metrics = self._initialize_baselines()
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize reward component weights"""
        default_weights = {
            'service_level': 0.30,      # Weight for service level component
            'cost_efficiency': 0.25,    # Weight for cost efficiency component
            'resource_utilization': 0.15,  # Weight for resource utilization component
            'delivery_time': 0.15,      # Weight for delivery time component
            'customer_satisfaction': 0.10,  # Weight for customer satisfaction component
            'sustainability': 0.05      # Weight for sustainability component
        }
        
        # Override with config weights if provided
        config_weights = self.config.get('reward_weights', {})
        weights = default_weights.copy()
        weights.update(config_weights)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _initialize_baselines(self) -> Dict[str, float]:
        """Initialize baseline metrics for normalization"""
        default_baselines = {
            'baseline_cost_per_order': 50.0,      # Expected cost per order
            'baseline_delivery_time_hours': 24.0,  # Expected delivery time
            'baseline_service_level': 0.95,        # Expected service level
            'baseline_utilization': 0.70,          # Expected resource utilization
            'baseline_customer_satisfaction': 0.85  # Expected satisfaction
        }
        
        # Override with config baselines if provided
        config_baselines = self.config.get('baseline_metrics', {})
        baselines = default_baselines.copy()
        baselines.update(config_baselines)
        
        return baselines
    
    def calculate_reward(self, current_state: Dict[str, Any], 
                       previous_state: Dict[str, Any],
                       actions: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward based on current state and actions
        
        Args:
            current_state: Current simulation state
            previous_state: Previous simulation state
            actions: List of actions taken
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        # Calculate individual reward components
        components = {}
        
        # 1. Service Level Reward
        components['service_level'] = self._calculate_service_level_reward(current_state)
        
        # 2. Cost Efficiency Reward
        components['cost_efficiency'] = self._calculate_cost_efficiency_reward(
            current_state, previous_state
        )
        
        # 3. Resource Utilization Reward
        components['resource_utilization'] = self._calculate_resource_utilization_reward(
            current_state
        )
        
        # 4. Delivery Time Reward
        components['delivery_time'] = self._calculate_delivery_time_reward(current_state)
        
        # 5. Customer Satisfaction Reward
        components['customer_satisfaction'] = self._calculate_customer_satisfaction_reward(
            current_state
        )
        
        # 6. Sustainability Reward
        components['sustainability'] = self._calculate_sustainability_reward(
            current_state, previous_state
        )
        
        # Calculate weighted total reward
        total_reward = sum(
            components[component] * self.weights[component]
            for component in components
        )
        
        return total_reward, components
    
    def _calculate_service_level_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate service level reward component
        
        Args:
            state: Current state
            
        Returns:
            Service level reward (-1.0 to 1.0)
        """
        metrics = state.get('metrics', {})
        total_orders = metrics.get('total_orders', 1)
        fulfilled_orders = metrics.get('fulfilled_orders', 0)
        
        if total_orders == 0:
            return 0.0
        
        service_level = fulfilled_orders / total_orders
        
        # Normalize against baseline
        baseline = self.baseline_metrics['baseline_service_level']
        deviation = service_level - baseline
        
        # Use sigmoid-like function for smooth transitions
        # Positive deviation = positive reward, negative deviation = negative reward
        reward = 2 / (1 + np.exp(-deviation * 10)) - 1
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_cost_efficiency_reward(self, current_state: Dict[str, Any],
                                       previous_state: Dict[str, Any]) -> float:
        """
        Calculate cost efficiency reward component
        
        Args:
            current_state: Current state
            previous_state: Previous state
            
        Returns:
            Cost efficiency reward (-1.0 to 1.0)
        """
        current_metrics = current_state.get('metrics', {})
        previous_metrics = previous_state.get('metrics', {}) if previous_state else {}
        
        current_cost = current_metrics.get('total_cost', 0.0)
        current_orders = current_metrics.get('total_orders', 1)
        
        previous_cost = previous_metrics.get('total_cost', current_cost)
        previous_orders = previous_metrics.get('total_orders', current_orders)
        
        # Calculate cost per order
        current_cost_per_order = current_cost / max(1, current_orders)
        previous_cost_per_order = previous_cost / max(1, previous_orders)
        
        # Calculate improvement
        if previous_cost_per_order > 0:
            cost_improvement = (previous_cost_per_order - current_cost_per_order) / previous_cost_per_order
        else:
            cost_improvement = 0.0
        
        # Normalize against baseline
        baseline = self.baseline_metrics['baseline_cost_per_order']
        normalized_cost = current_cost_per_order / baseline if baseline > 0 else 1.0
        efficiency = 1.0 - normalized_cost
        
        # Combine improvement and efficiency
        reward = (cost_improvement * 0.3 + efficiency * 0.7)
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_resource_utilization_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate resource utilization reward component
        
        Args:
            state: Current state
            
        Returns:
            Resource utilization reward (-1.0 to 1.0)
        """
        metrics = state.get('metrics', {})
        utilization = metrics.get('resource_utilization', 0.0)
        
        # Target utilization (avoid too high or too low)
        target_utilization = 0.8
        deviation = abs(utilization - target_utilization)
        
        # Reward decreases as deviation increases
        reward = 1.0 - (deviation * 2)  # Scale factor to make it more sensitive
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_delivery_time_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate delivery time reward component
        
        Args:
            state: Current state
            
        Returns:
            Delivery time reward (-1.0 to 1.0)
        """
        metrics = state.get('metrics', {})
        avg_delivery_time = metrics.get('average_delivery_time', 0.0)
        
        # Normalize against baseline
        baseline = self.baseline_metrics['baseline_delivery_time_hours']
        if baseline > 0:
            normalized_time = avg_delivery_time / baseline
        else:
            normalized_time = 1.0
        
        # Faster delivery = positive reward, slower = negative reward
        reward = 1.0 - normalized_time
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_customer_satisfaction_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate customer satisfaction reward component
        
        Args:
            state: Current state
            
        Returns:
            Customer satisfaction reward (-1.0 to 1.0)
        """
        metrics = state.get('metrics', {})
        satisfaction = metrics.get('customer_satisfaction', 1.0)
        
        # Normalize against baseline
        baseline = self.baseline_metrics['baseline_customer_satisfaction']
        deviation = satisfaction - baseline
        
        # Use linear scaling
        reward = deviation * 2  # Scale to [-1, 1] range
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_sustainability_reward(self, current_state: Dict[str, Any],
                                      previous_state: Dict[str, Any]) -> float:
        """
        Calculate sustainability reward component
        
        Args:
            current_state: Current state
            previous_state: Previous state
            
        Returns:
            Sustainability reward (-1.0 to 1.0)
        """
        current_metrics = current_state.get('metrics', {})
        previous_metrics = previous_state.get('metrics', {}) if previous_state else {}
        
        # Simple sustainability metric: carbon emissions proxy
        current_emissions = current_metrics.get('total_emissions', 0.0)
        previous_emissions = previous_metrics.get('total_emissions', current_emissions)
        
        # Calculate emission reduction
        if previous_emissions > 0:
            emission_reduction = (previous_emissions - current_emissions) / previous_emissions
        else:
            emission_reduction = 0.0
        
        # Normalize to [-1, 1] range
        reward = max(-1.0, min(1.0, emission_reduction * 2))
        
        return reward
    
    def get_reward_breakdown(self, reward_components: Dict[str, float]) -> str:
        """
        Get human-readable breakdown of reward components
        
        Args:
            reward_components: Dictionary of reward components
            
        Returns:
            Formatted breakdown string
        """
        breakdown_lines = ["Reward Breakdown:"]
        
        total_weighted_reward = 0.0
        for component, value in reward_components.items():
            weight = self.weights.get(component, 0.0)
            weighted_value = value * weight
            total_weighted_reward += weighted_value
            
            breakdown_lines.append(
                f"  {component}: {value:+.3f} × {weight:.3f} = {weighted_value:+.3f}"
            )
        
        breakdown_lines.append(f"  Total Reward: {total_weighted_reward:+.3f}")
        
        return "\n".join(breakdown_lines)


class MultiObjectiveReward:
    """Handle multiple reward objectives"""
    
    def __init__(self, primary_reward: RewardFunctionV1, 
                 secondary_rewards: List[RewardFunctionV1] = None):
        """
        Initialize multi-objective reward handler
        
        Args:
            primary_reward: Primary reward function
            secondary_rewards: List of secondary reward functions
        """
        self.primary_reward = primary_reward
        self.secondary_rewards = secondary_rewards or []
        self.logger = logging.getLogger('MultiObjectiveReward')
    
    def calculate_combined_reward(self, current_state: Dict[str, Any],
                               previous_state: Dict[str, Any],
                               actions: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate combined reward from multiple objectives
        
        Args:
            current_state: Current simulation state
            previous_state: Previous simulation state
            actions: List of actions taken
            
        Returns:
            Tuple of (combined_reward, detailed_metrics)
        """
        # Calculate primary reward
        primary_reward, primary_components = self.primary_reward.calculate_reward(
            current_state, previous_state, actions
        )
        
        # Calculate secondary rewards
        secondary_rewards = []
        secondary_details = {}
        
        for i, reward_func in enumerate(self.secondary_rewards):
            sec_reward, sec_components = reward_func.calculate_reward(
                current_state, previous_state, actions
            )
            secondary_rewards.append(sec_reward)
            secondary_details[f'secondary_{i}'] = {
                'reward': sec_reward,
                'components': sec_components
            }
        
        # Combine rewards (simple weighted average)
        if secondary_rewards:
            # Give 80% weight to primary, 20% to secondary rewards equally divided
            primary_weight = 0.8
            secondary_weight_each = 0.2 / len(secondary_rewards)
            
            combined_reward = primary_reward * primary_weight
            for sec_reward in secondary_rewards:
                combined_reward += sec_reward * secondary_weight_each
        else:
            combined_reward = primary_reward
        
        # Compile detailed metrics
        detailed_metrics = {
            'primary_reward': primary_reward,
            'primary_components': primary_components,
            'secondary_rewards': secondary_details,
            'weights': {
                'primary': 0.8 if secondary_rewards else 1.0,
                'secondary_each': 0.2 / len(secondary_rewards) if secondary_rewards else 0.0
            }
        }
        
        return combined_reward, detailed_metrics


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = {
        'reward_weights': {
            'service_level': 0.35,
            'cost_efficiency': 0.25,
            'resource_utilization': 0.15,
            'delivery_time': 0.15,
            'customer_satisfaction': 0.10
        },
        'baseline_metrics': {
            'baseline_cost_per_order': 45.0,
            'baseline_delivery_time_hours': 20.0,
            'baseline_service_level': 0.92,
            'baseline_utilization': 0.75,
            'baseline_customer_satisfaction': 0.88
        }
    }
    
    # Create reward function
    reward_func = RewardFunctionV1(config)
    
    # Create sample states
    current_state = {
        'metrics': {
            'total_orders': 100,
            'fulfilled_orders': 95,
            'total_cost': 4200.0,
            'average_delivery_time': 18.5,
            'customer_satisfaction': 0.91,
            'resource_utilization': 0.78,
            'total_emissions': 1200.0
        }
    }
    
    previous_state = {
        'metrics': {
            'total_orders': 90,
            'fulfilled_orders': 85,
            'total_cost': 4000.0,
            'average_delivery_time': 22.0,
            'customer_satisfaction': 0.85,
            'resource_utilization': 0.72,
            'total_emissions': 1300.0
        }
    }
    
    # Calculate reward
    actions = [{'type': 'assign_vehicle', 'vehicle_id': 'VEH001'}]
    
    total_reward, components = reward_func.calculate_reward(
        current_state, previous_state, actions
    )
    
    print("Reward Calculation Results:")
    print(f"Total Reward: {total_reward:+.3f}")
    print()
    
    # Print component breakdown
    breakdown = reward_func.get_reward_breakdown(components)
    print(breakdown)
    
    # Test multi-objective reward
    print("\n" + "="*50)
    print("Testing Multi-Objective Reward:")
    
    # Create secondary reward function with different weights
    secondary_config = config.copy()
    secondary_config['reward_weights'] = {
        'service_level': 0.20,
        'cost_efficiency': 0.30,
        'resource_utilization': 0.20,
        'delivery_time': 0.15,
        'customer_satisfaction': 0.10,
        'sustainability': 0.05
    }
    
    secondary_reward = RewardFunctionV1(secondary_config)
    multi_reward = MultiObjectiveReward(reward_func, [secondary_reward])
    
    combined_reward, details = multi_reward.calculate_combined_reward(
        current_state, previous_state, actions
    )
    
    print(f"Combined Reward: {combined_reward:+.3f}")
    print(f"Primary Reward: {details['primary_reward']:+.3f}")
    print(f"Secondary Reward: {details['secondary_rewards']['secondary_0']['reward']:+.3f}")
</file>