"""
Version 2 reward functions for logistics simulation with advanced metrics
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging


class RewardFunctionV2:
    """Version 2 reward function with advanced logistics metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward function
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('RewardFunctionV2')
        self.weights = self._initialize_weights()
        self.penalty_factors = self._initialize_penalties()
        self.smoothing_factor = config.get('smoothing_factor', 0.1)
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize reward component weights"""
        default_weights = {
            'on_time_delivery': 0.25,        # Weight for on-time deliveries
            'cost_optimization': 0.20,       # Weight for cost optimization
            'resource_efficiency': 0.15,     # Weight for resource efficiency
            'customer_experience': 0.15,     # Weight for customer experience
            'operational_stability': 0.10,   # Weight for operational stability
            'sustainability': 0.10,          # Weight for sustainability
            'innovation_bonus': 0.05         # Weight for innovation/exploration
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
    
    def _initialize_penalties(self) -> Dict[str, float]:
        """Initialize penalty factors for violations"""
        return {
            'late_delivery': -0.5,           # Penalty for late deliveries
            'cancelled_order': -1.0,         # Penalty for cancelled orders
            'capacity_violation': -0.3,      # Penalty for capacity violations
            'safety_violation': -0.8,        # Penalty for safety violations
            'maintenance_neglect': -0.2      # Penalty for neglected maintenance
        }
    
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
        # Calculate positive reward components
        positive_components = {}
        
        # 1. On-Time Delivery Reward
        positive_components['on_time_delivery'] = self._calculate_on_time_delivery_reward(current_state)
        
        # 2. Cost Optimization Reward
        positive_components['cost_optimization'] = self._calculate_cost_optimization_reward(
            current_state, previous_state
        )
        
        # 3. Resource Efficiency Reward
        positive_components['resource_efficiency'] = self._calculate_resource_efficiency_reward(
            current_state
        )
        
        # 4. Customer Experience Reward
        positive_components['customer_experience'] = self._calculate_customer_experience_reward(
            current_state
        )
        
        # 5. Operational Stability Reward
        positive_components['operational_stability'] = self._calculate_operational_stability_reward(
            current_state, previous_state
        )
        
        # 6. Sustainability Reward
        positive_components['sustainability'] = self._calculate_sustainability_reward(
            current_state
        )
        
        # 7. Innovation Bonus (for exploration)
        positive_components['innovation_bonus'] = self._calculate_innovation_bonus(
            actions
        )
        
        # Calculate penalty components
        penalty_components = {}
        
        # Apply penalties based on violations in current state
        penalty_components['late_delivery'] = self._calculate_late_delivery_penalty(current_state)
        penalty_components['cancelled_order'] = self._calculate_cancelled_order_penalty(current_state)
        penalty_components['capacity_violation'] = self._calculate_capacity_violation_penalty(current_state)
        penalty_components['safety_violation'] = self._calculate_safety_violation_penalty(current_state)
        penalty_components['maintenance_neglect'] = self._calculate_maintenance_penalty(current_state)
        
        # Calculate weighted totals
        positive_reward = sum(
            positive_components[component] * self.weights[component]
            for component in positive_components
        )
        
        penalty_reward = sum(
            penalty_components[component] * abs(self.penalty_factors[component])
            for component in penalty_components
        )
        
        # Apply smoothing to prevent extreme fluctuations
        if previous_state and 'metrics' in previous_state:
            prev_total = previous_state['metrics'].get('total_reward_v2', 0.0)
            total_reward = (1 - self.smoothing_factor) * (positive_reward + penalty_reward) + \
                          self.smoothing_factor * prev_total
        else:
            total_reward = positive_reward + penalty_reward
        
        # Combine all components for return
        all_components = {**positive_components, **penalty_components}
        all_components['total_positive'] = positive_reward
        all_components['total_penalties'] = penalty_reward
        all_components['smoothed_total'] = total_reward
        
        return total_reward, all_components
    
    def _calculate_on_time_delivery_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate on-time delivery reward component
        
        Args:
            state: Current state
            
        Returns:
            On-time delivery reward (0.0 to 1.0)
        """
        orders = state.get('orders', {})
        if not orders:
            return 0.0
        
        on_time_count = 0
        total_delivered = 0
        
        for order in orders.values():
            if order.get('status') in ['fulfilled', 'delivered']:
                total_delivered += 1
                fulfilled_time = order.get('fulfilled_time')
                deadline = order.get('delivery_deadline')
                
                if fulfilled_time and deadline and fulfilled_time <= deadline:
                    on_time_count += 1
        
        if total_delivered == 0:
            return 0.0
        
        on_time_rate = on_time_count / total_delivered
        
        # Exponential scaling to emphasize high performance
        reward = 1.0 - np.exp(-on_time_rate * 5)
        
        return reward
    
    def _calculate_cost_optimization_reward(self, current_state: Dict[str, Any],
                                         previous_state: Dict[str, Any]) -> float:
        """
        Calculate cost optimization reward component
        
        Args:
            current_state: Current state
            previous_state: Previous state
            
        Returns:
            Cost optimization reward (-1.0 to 1.0)
        """
        current_metrics = current_state.get('metrics', {})
        previous_metrics = previous_state.get('metrics', {}) if previous_state else {}
        
        current_cost = current_metrics.get('total_cost', 0.0)
        current_orders = current_metrics.get('fulfilled_orders', 1)
        
        previous_cost = previous_metrics.get('total_cost', current_cost)
        previous_orders = previous_metrics.get('fulfilled_orders', current_orders)
        
        # Calculate cost per fulfilled order
        current_cost_per_order = current_cost / max(1, current_orders)
        previous_cost_per_order = previous_cost / max(1, previous_orders)
        
        # Calculate relative improvement
        if previous_cost_per_order > 0:
            relative_improvement = (previous_cost_per_order - current_cost_per_order) / previous_cost_per_order
        else:
            relative_improvement = 0.0
        
        # Normalize and bound
        reward = max(-1.0, min(1.0, relative_improvement * 2))
        
        return reward
    
    def _calculate_resource_efficiency_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate resource efficiency reward component
        
        Args:
            state: Current state
            
        Returns:
            Resource efficiency reward (0.0 to 1.0)
        """
        warehouses = state.get('warehouses', {})
        vehicles = state.get('vehicles', {})
        
        if not warehouses and not vehicles:
            return 0.0
        
        # Calculate warehouse efficiency (inventory turnover)
        warehouse_efficiencies = []
        for warehouse in warehouses.values():
            capacity = warehouse.get('capacity', 1)
            inventory = warehouse.get('current_inventory', 0)
            if capacity > 0:
                # Efficiency = 1 - (excess inventory / capacity)
                excess_ratio = max(0, inventory - capacity * 0.2) / capacity  # Keep 20% buffer
                efficiency = 1.0 - excess_ratio
                warehouse_efficiencies.append(max(0.0, efficiency))
        
        # Calculate vehicle efficiency (utilization)
        vehicle_efficiencies = []
        for vehicle in vehicles.values():
            capacity = vehicle.get('capacity', 1)
            assigned_orders = vehicle.get('assigned_orders', [])
            if capacity > 0 and assigned_orders:
                # Simplified: assume average order size
                avg_order_size = 10
                total_assigned = len(assigned_orders) * avg_order_size
                utilization = min(1.0, total_assigned / capacity)
                vehicle_efficiencies.append(utilization)
        
        # Average efficiencies
        avg_warehouse_eff = np.mean(warehouse_efficiencies) if warehouse_efficiencies else 0.0
        avg_vehicle_eff = np.mean(vehicle_efficiencies) if vehicle_efficiencies else 0.0
        
        # Combined efficiency
        combined_efficiency = (avg_warehouse_eff * 0.6 + avg_vehicle_eff * 0.4)
        
        return combined_efficiency
    
    def _calculate_customer_experience_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate customer experience reward component
        
        Args:
            state: Current state
            
        Returns:
            Customer experience reward (0.0 to 1.0)
        """
        metrics = state.get('metrics', {})
        satisfaction = metrics.get('customer_satisfaction', 1.0)
        
        # Sigmoid scaling to emphasize high satisfaction
        reward = 1.0 / (1.0 + np.exp(-(satisfaction - 0.8) * 10))
        
        return reward
    
    def _calculate_operational_stability_reward(self, current_state: Dict[str, Any],
                                            previous_state: Dict[str, Any]) -> float:
        """
        Calculate operational stability reward component
        
        Args:
            current_state: Current state
            previous_state: Previous state
            
        Returns:
            Operational stability reward (0.0 to 1.0)
        """
        current_metrics = current_state.get('metrics', {})
        previous_metrics = previous_state.get('metrics', {}) if previous_state else {}
        
        # Measure variance in key metrics
        current_variance = current_metrics.get('delivery_time_variance', 1.0)
        previous_variance = previous_metrics.get('delivery_time_variance', current_variance)
        
        # Lower variance = higher stability
        if previous_variance > 0:
            stability_improvement = (previous_variance - current_variance) / previous_variance
        else:
            stability_improvement = 0.0
        
        # Convert to reward (0 to 1, with 1 being perfectly stable)
        reward = max(0.0, min(1.0, 0.5 + stability_improvement * 0.5))
        
        return reward
    
    def _calculate_sustainability_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate sustainability reward component
        
        Args:
            state: Current state
            
        Returns:
            Sustainability reward (0.0 to 1.0)
        """
        metrics = state.get('metrics', {})
        emissions = metrics.get('total_emissions', 0.0)
        distance_traveled = metrics.get('total_distance_traveled', 1.0)
        
        if distance_traveled <= 0:
            return 0.0
        
        # Calculate emissions per distance (lower is better)
        emissions_per_km = emissions / distance_traveled
        
        # Normalize against a baseline (e.g., 0.1 kg CO2 per km)
        baseline_emissions = 0.1
        if baseline_emissions > 0:
            normalized_emissions = emissions_per_km / baseline_emissions
            # Invert and bound: 1.0 for perfect, 0.0 for very poor
            reward = max(0.0, min(1.0, 1.0 - normalized_emissions))
        else:
            reward = 1.0
        
        return reward
    
    def _calculate_innovation_bonus(self, actions: List[Dict[str, Any]]) -> float:
        """
        Calculate innovation/exploration bonus
        
        Args:
            actions: List of actions taken
            
        Returns:
            Innovation bonus (0.0 to 0.1)
        """
        # Simple bonus for trying new action combinations
        unique_actions = len(set(str(action) for action in actions))
        bonus = min(0.1, unique_actions * 0.01)  # Max 0.1 bonus
        
        return bonus
    
    def _calculate_late_delivery_penalty(self, state: Dict[str, Any]) -> float:
        """
        Calculate penalty for late deliveries
        
        Args:
            state: Current state
            
        Returns:
            Late delivery penalty (0.0 to -1.0)
        """
        orders = state.get('orders', {})
        if not orders:
            return 0.0
        
        late_count = 0
        total_delivered = 0
        
        for order in orders.values():
            if order.get('status') in ['fulfilled', 'delivered']:
                total_delivered += 1
                fulfilled_time = order.get('fulfilled_time')
                deadline = order.get('delivery_deadline')
                
                if fulfilled_time and deadline and fulfilled_time > deadline:
                    late_count += 1
        
        if total_delivered == 0:
            return 0.0
        
        late_rate = late_count / total_delivered
        penalty = -late_rate  # Scale to -1.0 max penalty
        
        return penalty
    
    def _calculate_cancelled_order_penalty(self, state: Dict[str, Any]) -> float:
        """
        Calculate penalty for cancelled orders
        
        Args:
            state: Current state
            
        Returns:
            Cancelled order penalty (0.0 to -1.0)
        """
        orders = state.get('orders', {})
        if not orders:
            return 0.0
        
        cancelled_count = sum(1 for order in orders.values() 
                            if order.get('status') == 'cancelled')
        total_orders = len(orders)
        
        if total_orders == 0:
            return 0.0
        
        cancellation_rate = cancelled_count / total_orders
        penalty = -cancellation_rate
        
        return penalty
    
    def _calculate_capacity_violation_penalty(self, state: Dict[str, Any]) -> float:
        """
        Calculate penalty for capacity violations
        
        Args:
            state: Current state
            
        Returns:
            Capacity violation penalty (0.0 to -1.0)
        """
        warehouses = state.get('warehouses', {})
        vehicles = state.get('vehicles', {})
        
        violations = 0
        total_entities = 0
        
        # Check warehouse capacity violations
        for warehouse in warehouses.values():
            total_entities += 1
            capacity = warehouse.get('capacity', 1)
            inventory = warehouse.get('current_inventory', 0)
            
            if inventory > capacity:
                violations += 1
        
        # Check vehicle capacity violations
        for vehicle in vehicles.values():
            total_entities += 1
            capacity = vehicle.get('capacity', 1)
            assigned_orders = vehicle.get('assigned_orders', [])
            
            # Simplified: assume average order size
            avg_order_size = 10
            total_assigned = len(assigned_orders) * avg_order_size
            
            if total_assigned > capacity:
                violations += 1
        
        if total_entities == 0:
            return 0.0
        
        violation_rate = violations / total_entities
        penalty = -violation_rate
        
        return penalty
    
    def _calculate_safety_violation_penalty(self, state: Dict[str, Any]) -> float:
        """
        Calculate penalty for safety violations
        
        Args:
            state: Current state
            
        Returns:
            Safety violation penalty (0.0 to -1.0)
        """
        # This would integrate with safety monitoring systems
        # For now, simulate based on metrics
        metrics = state.get('metrics', {})
        accident_rate = metrics.get('accident_rate', 0.0)
        
        # Cap penalty at -1.0
        penalty = -min(1.0, accident_rate * 10)
        
        return penalty
    
    def _calculate_maintenance_penalty(self, state: Dict[str, Any]) -> float:
        """
        Calculate penalty for neglected maintenance
        
        Args:
            state: Current state
            
        Returns:
            Maintenance penalty (0.0 to -1.0)
        """
        vehicles = state.get('vehicles', {})
        if not vehicles:
            return 0.0
        
        neglected_count = 0
        total_vehicles = 0
        
        for vehicle in vehicles.values():
            total_vehicles += 1
            # Simulate maintenance status
            maintenance_due = vehicle.get('maintenance_due', 0.0)
            
            # Threshold for neglect (arbitrary)
            if maintenance_due > 2.0:
                neglected_count += 1
        
        if total_vehicles == 0:
            return 0.0
        
        neglect_rate = neglected_count / total_vehicles
        penalty = -neglect_rate
        
        return penalty


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = {
        'reward_weights': {
            'on_time_delivery': 0.30,
            'cost_optimization': 0.20,
            'resource_efficiency': 0.15,
            'customer_experience': 0.15,
            'operational_stability': 0.10,
            'sustainability': 0.07,
            'innovation_bonus': 0.03
        },
        'smoothing_factor': 0.1
    }
    
    # Create reward function
    reward_func = RewardFunctionV2(config)
    
    # Create sample states
    current_state = {
        'orders': {
            'ORDER_001': {
                'status': 'fulfilled',
                'fulfilled_time': datetime.now(),
                'delivery_deadline': datetime.now() + timedelta(hours=1)
            },
            'ORDER_002': {
                'status': 'fulfilled',
                'fulfilled_time': datetime.now() + timedelta(hours=2),
                'delivery_deadline': datetime.now() + timedelta(hours=1)
            }
        },
        'warehouses': {
            'WH001': {
                'capacity': 1000,
                'current_inventory': 800
            }
        },
        'vehicles': {
            'VEH001': {
                'capacity': 500,
                'assigned_orders': ['ORDER_001']
            }
        },
        'metrics': {
            'total_cost': 1500.0,
            'fulfilled_orders': 2,
            'customer_satisfaction': 0.95,
            'delivery_time_variance': 0.5,
            'total_emissions': 50.0,
            'total_distance_traveled': 100.0,
            'accident_rate': 0.01
        }
    }
    
    previous_state = {
        'metrics': {
            'total_cost': 1600.0,
            'fulfilled_orders': 2,
            'customer_satisfaction': 0.90,
            'delivery_time_variance': 0.8,
            'total_emissions': 60.0,
            'total_distance_traveled': 120.0
        }
    }
    
    # Calculate reward
    actions = [
        {'type': 'assign_vehicle', 'vehicle_id': 'VEH001'},
        {'type': 'route_vehicle', 'vehicle_id': 'VEH001', 'destination': (10, 20)}
    ]
    
    total_reward, components = reward_func.calculate_reward(
        current_state, previous_state, actions
    )
    
    print("Version 2 Reward Calculation Results:")
    print(f"Total Reward: {total_reward:+.3f}")
    print()
    
    print("Reward Components:")
    for component, value in components.items():
        if component.startswith('total_'):
            print(f"  {component}: {value:+.3f}")
        else:
            weight = reward_func.weights.get(component, 0.0)
            print(f"  {component}: {value:+.3f} (weight: {weight:.3f})")
