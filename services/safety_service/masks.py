"""
Safety masks for constraining agent actions
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging


@dataclass
class SafetyMask:
    """Safety mask to constrain actions"""
    mask_id: str
    mask_type: str  # 'capacity', 'sla', 'legal', 'business', 'operational'
    constraints: Dict[str, Any]
    active: bool = True
    priority: int = 1  # Higher priority masks are applied first
    description: str = ""


@dataclass
class ActionConstraint:
    """Constraint on agent actions"""
    action_space: np.ndarray  # Valid action indices
    masked_actions: np.ndarray  # Invalid action indices
    constraint_vector: np.ndarray  # Continuous constraint vector
    violation_count: int = 0


class SafetyMaskEngine:
    """Engine for applying safety masks to agent actions"""
    
    def __init__(self):
        self.masks = {}
        self.logger = logging.getLogger('SafetyMaskEngine')
    
    def add_mask(self, mask: SafetyMask) -> None:
        """
        Add a safety mask
        
        Args:
            mask: Safety mask to add
        """
        self.masks[mask.mask_id] = mask
        self.logger.info(f"Added safety mask: {mask.mask_id}")
    
    def remove_mask(self, mask_id: str) -> bool:
        """
        Remove a safety mask
        
        Args:
            mask_id: ID of mask to remove
            
        Returns:
            True if removed, False if not found
        """
        if mask_id in self.masks:
            del self.masks[mask_id]
            self.logger.info(f"Removed safety mask: {mask_id}")
            return True
        return False
    
    def activate_mask(self, mask_id: str) -> bool:
        """
        Activate a safety mask
        
        Args:
            mask_id: ID of mask to activate
            
        Returns:
            True if activated, False if not found
        """
        if mask_id in self.masks:
            self.masks[mask_id].active = True
            self.logger.info(f"Activated safety mask: {mask_id}")
            return True
        return False
    
    def deactivate_mask(self, mask_id: str) -> bool:
        """
        Deactivate a safety mask
        
        Args:
            mask_id: ID of mask to deactivate
            
        Returns:
            True if deactivated, False if not found
        """
        if mask_id in self.masks:
            self.masks[mask_id].active = False
            self.logger.info(f"Deactivated safety mask: {mask_id}")
            return True
        return False
    
    def apply_masks(self, actions: np.ndarray, 
                   state: Dict[str, Any]) -> Tuple[np.ndarray, List[ActionConstraint]]:
        """
        Apply all active safety masks to actions
        
        Args:
            actions: Action array to constrain
            state: Current state
            
        Returns:
            Tuple of (constrained_actions, action_constraints)
        """
        # Sort masks by priority
        active_masks = [mask for mask in self.masks.values() if mask.active]
        active_masks.sort(key=lambda x: x.priority, reverse=True)
        
        constrained_actions = actions.copy()
        constraints = []
        
        # Apply each mask in priority order
        for mask in active_masks:
            try:
                constrained_actions, constraint = self._apply_mask(
                    mask, constrained_actions, state
                )
                constraints.append(constraint)
            except Exception as e:
                self.logger.warning(f"Failed to apply mask {mask.mask_id}: {str(e)}")
        
        return constrained_actions, constraints
    
    def _apply_mask(self, mask: SafetyMask, actions: np.ndarray, 
                   state: Dict[str, Any]) -> Tuple[np.ndarray, ActionConstraint]:
        """
        Apply a single safety mask
        
        Args:
            mask: Safety mask to apply
            actions: Actions to constrain
            state: Current state
            
        Returns:
            Tuple of (constrained_actions, action_constraint)
        """
        if mask.mask_type == 'capacity':
            return self._apply_capacity_mask(mask, actions, state)
        elif mask.mask_type == 'sla':
            return self._apply_sla_mask(mask, actions, state)
        elif mask.mask_type == 'legal':
            return self._apply_legal_mask(mask, actions, state)
        elif mask.mask_type == 'business':
            return self._apply_business_mask(mask, actions, state)
        elif mask.mask_type == 'operational':
            return self._apply_operational_mask(mask, actions, state)
        else:
            raise ValueError(f"Unsupported mask type: {mask.mask_type}")
    
    def _apply_capacity_mask(self, mask: SafetyMask, actions: np.ndarray, 
                           state: Dict[str, Any]) -> Tuple[np.ndarray, ActionConstraint]:
        """
        Apply capacity constraint mask
        
        Args:
            mask: Capacity mask
            actions: Actions to constrain
            state: Current state
            
        Returns:
            Tuple of (constrained_actions, action_constraint)
        """
        constraints = mask.constraints
        capacity_limits = constraints.get('limits', {})
        
        # Identify actions that would violate capacity constraints
        masked_indices = []
        valid_indices = []
        
        for i, action in enumerate(actions):
            # Check if action violates capacity constraints
            violates_capacity = False
            
            # This is a simplified check - in practice, you would have more
            # sophisticated capacity checking based on the specific action
            for resource, limit in capacity_limits.items():
                current_usage = state.get(f'{resource}_usage', 0)
                action_impact = self._get_action_impact(action, resource)
                
                if current_usage + action_impact > limit:
                    violates_capacity = True
                    break
            
            if violates_capacity:
                masked_indices.append(i)
            else:
                valid_indices.append(i)
        
        # Create constrained actions
        if valid_indices:
            constrained_actions = actions[valid_indices]
        else:
            # If all actions are masked, allow a safe default action
            constrained_actions = np.array([self._get_safe_default_action()])
            valid_indices = [0]
            masked_indices = list(range(len(actions)))
        
        # Create constraint object
        constraint = ActionConstraint(
            action_space=np.array(valid_indices),
            masked_actions=np.array(masked_indices),
            constraint_vector=self._create_capacity_constraint_vector(mask, state)
        )
        
        return constrained_actions, constraint
    
    def _apply_sla_mask(self, mask: SafetyMask, actions: np.ndarray, 
                       state: Dict[str, Any]) -> Tuple[np.ndarray, ActionConstraint]:
        """
        Apply SLA constraint mask
        
        Args:
            mask: SLA mask
            actions: Actions to constrain
            state: Current state
            
        Returns:
            Tuple of (constrained_actions, action_constraint)
        """
        # Similar implementation for SLA constraints
        # This would check if actions would violate service level agreements
        constraints = mask.constraints
        sla_requirements = constraints.get('requirements', {})
        
        masked_indices = []
        valid_indices = []
        
        for i, action in enumerate(actions):
            violates_sla = False
            
            # Check SLA violations
            for sla_metric, requirement in sla_requirements.items():
                current_performance = state.get(f'{sla_metric}_performance', 1.0)
                action_impact = self._get_action_sla_impact(action, sla_metric)
                
                if current_performance + action_impact < requirement:
                    violates_sla = True
                    break
            
            if violates_sla:
                masked_indices.append(i)
            else:
                valid_indices.append(i)
        
        # Create constrained actions
        if valid_indices:
            constrained_actions = actions[valid_indices]
        else:
            constrained_actions = np.array([self._get_safe_default_action()])
            valid_indices = [0]
            masked_indices = list(range(len(actions)))
        
        # Create constraint object
        constraint = ActionConstraint(
            action_space=np.array(valid_indices),
            masked_actions=np.array(masked_indices),
            constraint_vector=self._create_sla_constraint_vector(mask, state)
        )
        
        return constrained_actions, constraint
    
    def _get_action_impact(self, action: Any, resource: str) -> float:
        """
        Get the impact of an action on a resource
        
        Args:
            action: Action to evaluate
            resource: Resource to check impact on
            
        Returns:
            Resource impact value
        """
        # This is a simplified implementation
        # In practice, this would be more sophisticated based on action type
        impact_map = {
            'vehicle_capacity': 10.0,
            'warehouse_capacity': 100.0,
            'driver_hours': 8.0
        }
        return impact_map.get(resource, 0.0)
    
    def _get_action_sla_impact(self, action: Any, sla_metric: str) -> float:
        """
        Get the impact of an action on an SLA metric
        
        Args:
            action: Action to evaluate
            sla_metric: SLA metric to check impact on
            
        Returns:
            SLA impact value
        """
        # Simplified implementation
        impact_map = {
            'delivery_time': -0.5,  # Negative impact (worse delivery time)
            'order_accuracy': 0.1,   # Positive impact (better accuracy)
            'customer_satisfaction': 0.05
        }
        return impact_map.get(sla_metric, 0.0)
    
    def _get_safe_default_action(self) -> Any:
        """
        Get a safe default action when all actions are masked
        
        Returns:
            Safe default action
        """
        # Return a neutral/safe action
        return 0  # Simplified - would be more sophisticated in practice
    
    def _create_capacity_constraint_vector(self, mask: SafetyMask, 
                                        state: Dict[str, Any]) -> np.ndarray:
        """
        Create constraint vector for capacity constraints
        
        Args:
            mask: Capacity mask
            state: Current state
            
        Returns:
            Constraint vector
        """
        # Simplified constraint vector creation
        return np.array([1.0, 0.0, 0.0])  # Placeholder
    
    def _create_sla_constraint_vector(self, mask: SafetyMask, 
                                   state: Dict[str, Any]) -> np.ndarray:
        """
        Create constraint vector for SLA constraints
        
        Args:
            mask: SLA mask
            state: Current state
            
        Returns:
            Constraint vector
        """
        # Simplified constraint vector creation
        return np.array([0.0, 1.0, 0.0])  # Placeholder
    
    def _apply_legal_mask(self, mask: SafetyMask, actions: np.ndarray, 
                         state: Dict[str, Any]) -> Tuple[np.ndarray, ActionConstraint]:
        """Apply legal constraint mask"""
        # Implementation for legal constraints
        valid_indices = list(range(len(actions)))  # Simplified
        constraint = ActionConstraint(
            action_space=np.array(valid_indices),
            masked_actions=np.array([]),
            constraint_vector=np.array([0.0, 0.0, 1.0])
        )
        return actions, constraint
    
    def _apply_business_mask(self, mask: SafetyMask, actions: np.ndarray, 
                           state: Dict[str, Any]) -> Tuple[np.ndarray, ActionConstraint]:
        """Apply business constraint mask"""
        # Implementation for business constraints
        valid_indices = list(range(len(actions)))  # Simplified
        constraint = ActionConstraint(
            action_space=np.array(valid_indices),
            masked_actions=np.array([]),
            constraint_vector=np.array([0.0, 0.0, 1.0])
        )
        return actions, constraint
    
    def _apply_operational_mask(self, mask: SafetyMask, actions: np.ndarray, 
                              state: Dict[str, Any]) -> Tuple[np.ndarray, ActionConstraint]:
        """Apply operational constraint mask"""
        # Implementation for operational constraints
        valid_indices = list(range(len(actions)))  # Simplified
        constraint = ActionConstraint(
            action_space=np.array(valid_indices),
            masked_actions=np.array([]),
            constraint_vector=np.array([0.0, 0.0, 1.0])
        )
        return actions, constraint


class DynamicMaskManager:
    """Manage dynamic safety masks that adapt to changing conditions"""
    
    def __init__(self, mask_engine: SafetyMaskEngine):
        self.mask_engine = mask_engine
        self.dynamic_masks = {}
        self.condition_monitors = {}
    
    def add_dynamic_mask(self, mask_id: str, mask_template: SafetyMask,
                        condition_function: callable) -> None:
        """
        Add a dynamic mask that adapts based on conditions
        
        Args:
            mask_id: ID for the dynamic mask
            mask_template: Template for the mask
            condition_function: Function that determines when mask should be active
        """
        self.dynamic_masks[mask_id] = {
            'template': mask_template,
            'condition_function': condition_function,
            'current_state': False  # Whether mask is currently active
        }
    
    def update_dynamic_masks(self, current_state: Dict[str, Any]) -> None:
        """
        Update dynamic masks based on current state
        
        Args:
            current_state: Current system state
        """
        for mask_id, mask_info in self.dynamic_masks.items():
            condition_fn = mask_info['condition_function']
            should_be_active = condition_fn(current_state)
            current_state = mask_info['current_state']
            
            # Update mask activation state if needed
            if should_be_active and not current_state:
                self.mask_engine.activate_mask(mask_id)
                mask_info['current_state'] = True
            elif not should_be_active and current_state:
                self.mask_engine.deactivate_mask(mask_id)
                mask_info['current_state'] = False


# Example usage
if __name__ == "__main__":
    # Create safety mask engine
    mask_engine = SafetyMaskEngine()
    
    # Add capacity mask
    capacity_mask = SafetyMask(
        mask_id="capacity_limit_001",
        mask_type="capacity",
        constraints={
            "limits": {
                "vehicle_capacity": 1000,
                "warehouse_capacity": 5000
            }
        },
        priority=10,
        description="Prevent capacity violations"
    )
    
    mask_engine.add_mask(capacity_mask)
    
    # Add SLA mask
    sla_mask = SafetyMask(
        mask_id="sla_compliance_001",
        mask_type="sla",
        constraints={
            "requirements": {
                "delivery_time": 0.95,  # 95% on-time delivery
                "order_accuracy": 0.99   # 99% accuracy
            }
        },
        priority=5,
        description="Ensure SLA compliance"
    )
    
    mask_engine.add_mask(sla_mask)
    
    # Sample actions and state
    actions = np.array([1, 2, 3, 4, 5])
    state = {
        "vehicle_capacity_usage": 800,
        "warehouse_capacity_usage": 4500,
        "delivery_time_performance": 0.92,
        "order_accuracy_performance": 0.98
    }
    
    # Apply masks
    constrained_actions, constraints = mask_engine.apply_masks(actions, state)
    
    print("Original actions:", actions)
    print("Constrained actions:", constrained_actions)
    print("Number of constraints applied:", len(constraints))
</file>