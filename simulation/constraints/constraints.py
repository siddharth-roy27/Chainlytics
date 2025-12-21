"""
Constraint definitions and enforcement for logistics simulation
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class Constraint:
    """Definition of a constraint"""
    constraint_id: str
    constraint_type: str  # 'capacity', 'time', 'safety', 'business', 'legal'
    description: str
    check_function: callable
    violation_penalty: float
    active: bool = True
    priority: int = 1  # Higher priority constraints are checked first


class ConstraintChecker:
    """Check and enforce constraints in the simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize constraint checker
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('ConstraintChecker')
        self.constraints = self._initialize_constraints()
        self.violation_history = []
    
    def _initialize_constraints(self) -> List[Constraint]:
        """Initialize constraints from configuration"""
        constraints_config = self.config.get('constraints', [])
        constraints = []
        
        # Add default constraints if none configured
        if not constraints_config:
            constraints_config = self._get_default_constraints()
        
        for constraint_config in constraints_config:
            constraint = Constraint(
                constraint_id=constraint_config['id'],
                constraint_type=constraint_config['type'],
                description=constraint_config['description'],
                check_function=self._create_check_function(constraint_config),
                violation_penalty=constraint_config.get('penalty', -1.0),
                active=constraint_config.get('active', True),
                priority=constraint_config.get('priority', 1)
            )
            constraints.append(constraint)
        
        # Sort by priority (highest first)
        constraints.sort(key=lambda x: x.priority, reverse=True)
        
        return constraints
    
    def _get_default_constraints(self) -> List[Dict[str, Any]]:
        """Get default constraint definitions"""
        return [
            {
                'id': 'warehouse_capacity',
                'type': 'capacity',
                'description': 'Warehouse inventory must not exceed capacity',
                'check_function': 'check_warehouse_capacity',
                'penalty': -0.5,
                'active': True,
                'priority': 10
            },
            {
                'id': 'vehicle_capacity',
                'type': 'capacity',
                'description': 'Vehicle load must not exceed capacity',
                'check_function': 'check_vehicle_capacity',
                'penalty': -0.5,
                'active': True,
                'priority': 10
            },
            {
                'id': 'delivery_deadline',
                'type': 'time',
                'description': 'Orders must be delivered by deadline',
                'check_function': 'check_delivery_deadlines',
                'penalty': -1.0,
                'active': True,
                'priority': 8
            },
            {
                'id': 'driver_hours',
                'type': 'safety',
                'description': 'Drivers must not exceed maximum hours',
                'check_function': 'check_driver_hours',
                'penalty': -0.8,
                'active': True,
                'priority': 9
            },
            {
                'id': 'maintenance_schedule',
                'type': 'safety',
                'description': 'Vehicles must follow maintenance schedule',
                'check_function': 'check_maintenance',
                'penalty': -0.3,
                'active': True,
                'priority': 7
            }
        ]
    
    def _create_check_function(self, constraint_config: Dict[str, Any]) -> callable:
        """
        Create constraint check function from configuration
        
        Args:
            constraint_config: Constraint configuration
            
        Returns:
            Check function
        """
        function_name = constraint_config['check_function']
        
        if function_name == 'check_warehouse_capacity':
            return self._check_warehouse_capacity
        elif function_name == 'check_vehicle_capacity':
            return self._check_vehicle_capacity
        elif function_name == 'check_delivery_deadlines':
            return self._check_delivery_deadlines
        elif function_name == 'check_driver_hours':
            return self._check_driver_hours
        elif function_name == 'check_maintenance':
            return self._check_maintenance
        else:
            # Default check function that always passes
            return lambda state: (True, "No violation")
    
    def check_all_constraints(self, state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Check all active constraints against current state
        
        Args:
            state: Current simulation state
            
        Returns:
            Tuple of (violations, total_penalty)
        """
        violations = []
        total_penalty = 0.0
        
        for constraint in self.constraints:
            if not constraint.active:
                continue
            
            try:
                is_valid, message = constraint.check_function(state)
                
                if not is_valid:
                    violation = {
                        'constraint_id': constraint.constraint_id,
                        'constraint_type': constraint.constraint_type,
                        'description': constraint.description,
                        'message': message,
                        'penalty': constraint.violation_penalty,
                        'timestamp': datetime.now()
                    }
                    
                    violations.append(violation)
                    total_penalty += constraint.violation_penalty
                    
                    self.logger.warning(
                        f"Constraint violation: {constraint.constraint_id} - {message}"
                    )
                    
            except Exception as e:
                self.logger.error(
                    f"Error checking constraint {constraint.constraint_id}: {str(e)}"
                )
        
        # Record violations
        if violations:
            self.violation_history.extend(violations)
        
        return violations, total_penalty
    
    def _check_warehouse_capacity(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check warehouse capacity constraints
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (is_valid, message)
        """
        warehouses = state.get('warehouses', {})
        
        for wh_id, warehouse in warehouses.items():
            capacity = warehouse.get('capacity', 0)
            inventory = warehouse.get('current_inventory', 0)
            
            if inventory > capacity:
                excess = inventory - capacity
                return False, f"Warehouse {wh_id} over capacity by {excess} units"
        
        return True, "All warehouses within capacity limits"
    
    def _check_vehicle_capacity(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check vehicle capacity constraints
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (is_valid, message)
        """
        vehicles = state.get('vehicles', {})
        
        for veh_id, vehicle in vehicles.items():
            capacity = vehicle.get('capacity', 0)
            assigned_orders = vehicle.get('assigned_orders', [])
            
            # Simplified: assume average order size
            avg_order_size = self.config.get('average_order_size', 10)
            total_load = len(assigned_orders) * avg_order_size
            
            if total_load > capacity:
                excess = total_load - capacity
                return False, f"Vehicle {veh_id} over capacity by {excess} units"
        
        return True, "All vehicles within capacity limits"
    
    def _check_delivery_deadlines(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check delivery deadline constraints
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (is_valid, message)
        """
        orders = state.get('orders', {})
        current_time = state.get('timestamp', datetime.now())
        
        late_deliveries = []
        
        for order_id, order in orders.items():
            if order.get('status') in ['fulfilled', 'delivered']:
                deadline = order.get('delivery_deadline')
                fulfilled_time = order.get('fulfilled_time')
                
                if deadline and fulfilled_time and fulfilled_time > deadline:
                    late_deliveries.append(order_id)
        
        if late_deliveries:
            return False, f"Late deliveries for orders: {', '.join(late_deliveries[:5])}..."
        
        return True, "All deliveries on time"
    
    def _check_driver_hours(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check driver hour constraints
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (is_valid, message)
        """
        vehicles = state.get('vehicles', {})
        max_driver_hours = self.config.get('max_driver_hours', 10)
        
        violating_vehicles = []
        
        for veh_id, vehicle in vehicles.items():
            # Simulate driver hours based on operational time
            operational_time = vehicle.get('operational_time', 0)
            
            if operational_time > max_driver_hours:
                violating_vehicles.append(veh_id)
        
        if violating_vehicles:
            return False, f"Driver hours exceeded for vehicles: {', '.join(violating_vehicles)}"
        
        return True, "All driver hours within limits"
    
    def _check_maintenance(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check vehicle maintenance constraints
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (is_valid, message)
        """
        vehicles = state.get('vehicles', {})
        
        neglected_vehicles = []
        
        for veh_id, vehicle in vehicles.items():
            maintenance_due = vehicle.get('maintenance_due', 0)
            
            # Threshold for maintenance violation
            if maintenance_due > self.config.get('maintenance_threshold', 2.0):
                neglected_vehicles.append(veh_id)
        
        if neglected_vehicles:
            return False, f"Maintenance neglected for vehicles: {', '.join(neglected_vehicles)}"
        
        return True, "All vehicles maintenance up to date"
    
    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a new constraint
        
        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)
        # Re-sort by priority
        self.constraints.sort(key=lambda x: x.priority, reverse=True)
        self.logger.info(f"Added constraint: {constraint.constraint_id}")
    
    def remove_constraint(self, constraint_id: str) -> bool:
        """
        Remove a constraint
        
        Args:
            constraint_id: ID of constraint to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, constraint in enumerate(self.constraints):
            if constraint.constraint_id == constraint_id:
                del self.constraints[i]
                self.logger.info(f"Removed constraint: {constraint_id}")
                return True
        return False
    
    def activate_constraint(self, constraint_id: str) -> bool:
        """
        Activate a constraint
        
        Args:
            constraint_id: ID of constraint to activate
            
        Returns:
            True if activated, False if not found
        """
        for constraint in self.constraints:
            if constraint.constraint_id == constraint_id:
                constraint.active = True
                self.logger.info(f"Activated constraint: {constraint_id}")
                return True
        return False
    
    def deactivate_constraint(self, constraint_id: str) -> bool:
        """
        Deactivate a constraint
        
        Args:
            constraint_id: ID of constraint to deactivate
            
        Returns:
            True if deactivated, False if not found
        """
        for constraint in self.constraints:
            if constraint.constraint_id == constraint_id:
                constraint.active = False
                self.logger.info(f"Deactivated constraint: {constraint_id}")
                return True
        return False
    
    def get_violation_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get constraint violation report
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Violation report dictionary
        """
        if not self.violation_history:
            return {'total_violations': 0}
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_violations = [
            v for v in self.violation_history
            if v['timestamp'] >= cutoff_time
        ]
        
        if not recent_violations:
            return {'total_violations': 0}
        
        # Group by constraint type
        violations_by_type = {}
        total_penalty = 0.0
        
        for violation in recent_violations:
            constraint_type = violation['constraint_type']
            if constraint_type not in violations_by_type:
                violations_by_type[constraint_type] = {
                    'count': 0,
                    'penalty': 0.0,
                    'violations': []
                }
            
            violations_by_type[constraint_type]['count'] += 1
            violations_by_type[constraint_type]['penalty'] += violation['penalty']
            violations_by_type[constraint_type]['violations'].append({
                'constraint_id': violation['constraint_id'],
                'message': violation['message'],
                'timestamp': violation['timestamp'].isoformat()
            })
            total_penalty += violation['penalty']
        
        return {
            'total_violations': len(recent_violations),
            'total_penalty': total_penalty,
            'violations_by_type': violations_by_type,
            'period_start': cutoff_time.isoformat(),
            'period_end': datetime.now().isoformat()
        }
    
    def get_active_constraints(self) -> List[Dict[str, Any]]:
        """
        Get list of active constraints
        
        Returns:
            List of active constraint definitions
        """
        active_constraints = [
            {
                'constraint_id': c.constraint_id,
                'constraint_type': c.constraint_type,
                'description': c.description,
                'priority': c.priority
            }
            for c in self.constraints if c.active
        ]
        
        return active_constraints


class SoftConstraintEnforcer:
    """Enforce soft constraints by adjusting actions"""
    
    def __init__(self, constraint_checker: ConstraintChecker):
        """
        Initialize soft constraint enforcer
        
        Args:
            constraint_checker: Constraint checker instance
        """
        self.constraint_checker = constraint_checker
        self.logger = logging.getLogger('SoftConstraintEnforcer')
    
    def adjust_actions_for_constraints(self, actions: List[Dict[str, Any]], 
                                   current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adjust actions to avoid constraint violations
        
        Args:
            actions: Proposed actions
            current_state: Current state
            
        Returns:
            Adjusted actions
        """
        adjusted_actions = actions.copy()
        
        # Check constraints with proposed actions
        # This is a simplified implementation - in practice, you would simulate
        # the effect of actions on the state before checking constraints
        
        for constraint in self.constraint_checker.constraints:
            if not constraint.active:
                continue
            
            # Apply constraint-specific adjustments
            if constraint.constraint_id == 'warehouse_capacity':
                adjusted_actions = self._adjust_for_warehouse_capacity(
                    adjusted_actions, current_state
                )
            elif constraint.constraint_id == 'vehicle_capacity':
                adjusted_actions = self._adjust_for_vehicle_capacity(
                    adjusted_actions, current_state
                )
            elif constraint.constraint_id == 'driver_hours':
                adjusted_actions = self._adjust_for_driver_hours(
                    adjusted_actions, current_state
                )
        
        return adjusted_actions
    
    def _adjust_for_warehouse_capacity(self, actions: List[Dict[str, Any]], 
                                    state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adjust actions to respect warehouse capacity constraints
        
        Args:
            actions: Proposed actions
            state: Current state
            
        Returns:
            Adjusted actions
        """
        adjusted_actions = []
        
        for action in actions:
            if action.get('type') == 'restock_warehouse':
                warehouse_id = action.get('warehouse_id')
                quantity = action.get('quantity', 0)
                
                if warehouse_id and quantity > 0:
                    # Check if restocking would exceed capacity
                    warehouse = state.get('warehouses', {}).get(warehouse_id, {})
                    capacity = warehouse.get('capacity', 0)
                    current_inventory = warehouse.get('current_inventory', 0)
                    
                    if current_inventory + quantity > capacity:
                        # Adjust quantity to fit capacity
                        max_quantity = max(0, capacity - current_inventory)
                        action = action.copy()
                        action['quantity'] = max_quantity
                        self.logger.info(
                            f"Adjusted restock quantity for {warehouse_id}: "
                            f"{quantity} -> {max_quantity}"
                        )
            
            adjusted_actions.append(action)
        
        return adjusted_actions
    
    def _adjust_for_vehicle_capacity(self, actions: List[Dict[str, Any]], 
                                  state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adjust actions to respect vehicle capacity constraints
        
        Args:
            actions: Proposed actions
            state: Current state
            
        Returns:
            Adjusted actions
        """
        # This would implement vehicle load balancing logic
        # For now, return actions unchanged
        return actions
    
    def _adjust_for_driver_hours(self, actions: List[Dict[str, Any]], 
                              state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adjust actions to respect driver hour constraints
        
        Args:
            actions: Proposed actions
            state: Current state
            
        Returns:
            Adjusted actions
        """
        # This would implement driver scheduling logic
        # For now, return actions unchanged
        return actions


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = {
        'constraints': [
            {
                'id': 'warehouse_capacity',
                'type': 'capacity',
                'description': 'Warehouse inventory must not exceed capacity',
                'check_function': 'check_warehouse_capacity',
                'penalty': -0.5,
                'active': True,
                'priority': 10
            },
            {
                'id': 'vehicle_capacity',
                'type': 'capacity',
                'description': 'Vehicle load must not exceed capacity',
                'check_function': 'check_vehicle_capacity',
                'penalty': -0.5,
                'active': True,
                'priority': 10
            }
        ],
        'average_order_size': 15,
        'max_driver_hours': 8,
        'maintenance_threshold': 1.5
    }
    
    # Create constraint checker
    constraint_checker = ConstraintChecker(config)
    
    # Create sample state with violations
    current_state = {
        'timestamp': datetime.now(),
        'warehouses': {
            'WH001': {
                'capacity': 1000,
                'current_inventory': 1200  # Over capacity!
            }
        },
        'vehicles': {
            'VEH001': {
                'capacity': 500,
                'assigned_orders': ['ORDER_001', 'ORDER_002', 'ORDER_003'],  # Over capacity!
                'operational_time': 12  # Over driver hours!
            }
        },
        'orders': {
            'ORDER_001': {
                'status': 'fulfilled',
                'delivery_deadline': datetime.now() - timedelta(hours=2),
                'fulfilled_time': datetime.now()  # Late delivery!
            }
        }
    }
    
    # Check constraints
    violations, total_penalty = constraint_checker.check_all_constraints(current_state)
    
    print("Constraint Checking Results:")
    print(f"Total violations: {len(violations)}")
    print(f"Total penalty: {total_penalty}")
    print()
    
    for violation in violations:
        print(f"Violation: {violation['constraint_id']}")
        print(f"  Type: {violation['constraint_type']}")
        print(f"  Message: {violation['message']}")
        print(f"  Penalty: {violation['penalty']}")
        print()
    
    # Get violation report
    report = constraint_checker.get_violation_report(hours_back=24)
    print("Violation Report:")
    print(f"  Total violations: {report['total_violations']}")
    print(f"  Total penalty: {report['total_penalty']}")
    if 'violations_by_type' in report:
        for ctype, details in report['violations_by_type'].items():
            print(f"  {ctype}: {details['count']} violations")
    
    # Test constraint management
    print("\nConstraint Management:")
    active_constraints = constraint_checker.get_active_constraints()
    print(f"Active constraints: {len(active_constraints)}")
    
    # Test soft constraint enforcement
    soft_enforcer = SoftConstraintEnforcer(constraint_checker)
    
    actions = [
        {
            'type': 'restock_warehouse',
            'warehouse_id': 'WH001',
            'quantity': 300  # This would put it further over capacity
        }
    ]
    
    adjusted_actions = soft_enforcer.adjust_actions_for_constraints(actions, current_state)
    print(f"\nAction adjustment:")
    print(f"  Original action: {actions[0]}")
    print(f"  Adjusted action: {adjusted_actions[0]}")
</file>