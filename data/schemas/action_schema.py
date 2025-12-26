"""
Action Schema Validation
"""
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ActionSchema:
    """Validates action schemas for consistency and safety."""
    
    def __init__(self):
        pass
    
    def validate(self, action: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate an action against the schema.
        
        Args:
            action: Action to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = {}
        
        # Check required fields
        required_fields = ['type', 'timestamp']
        for field in required_fields:
            if field not in action:
                errors[field] = f"Missing required field: {field}"
        
        # Validate action type
        valid_types = [
            'route_assignment',
            'inventory_allocation',
            'vehicle_dispatch',
            'order_prioritization',
            'resource_reallocation',
            'maintenance_scheduling',
            'safe_hold'
        ]
        
        if 'type' in action and action['type'] not in valid_types:
            errors['type'] = f"Invalid action type: {action['type']}. Must be one of {valid_types}"
        
        # Validate timestamp format
        if 'timestamp' in action:
            try:
                datetime.fromisoformat(action['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                errors['timestamp'] = f"Invalid timestamp format: {action['timestamp']}"
        
        # Type-specific validations
        if 'type' in action:
            type_errors = self._validate_type_specific_fields(action)
            errors.update(type_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_type_specific_fields(self, action: Dict[str, Any]) -> Dict[str, str]:
        """Validate fields specific to each action type."""
        errors = {}
        action_type = action.get('type', '')
        
        if action_type == 'route_assignment':
            required_fields = ['vehicle_id', 'route', 'estimated_completion']
            for field in required_fields:
                if field not in action:
                    errors[field] = f"Missing required field for route_assignment: {field}"
        
        elif action_type == 'inventory_allocation':
            required_fields = ['warehouse_id', 'product_id', 'quantity']
            for field in required_fields:
                if field not in action:
                    errors[field] = f"Missing required field for inventory_allocation: {field}"
        
        elif action_type == 'vehicle_dispatch':
            required_fields = ['vehicle_id', 'destination', 'priority']
            for field in required_fields:
                if field not in action:
                    errors[field] = f"Missing required field for vehicle_dispatch: {field}"
        
        elif action_type == 'order_prioritization':
            required_fields = ['order_ids', 'priorities']
            for field in required_fields:
                if field not in action:
                    errors[field] = f"Missing required field for order_prioritization: {field}"
        
        return errors
    
    def sanitize(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize an action to ensure it conforms to the schema.
        
        Args:
            action: Action to sanitize
            
        Returns:
            Sanitized action
        """
        # Ensure required fields exist
        if 'type' not in action:
            action['type'] = 'safe_hold'
            
        if 'timestamp' not in action:
            action['timestamp'] = datetime.now().isoformat()
        
        # Add default fields if missing
        if 'reason' not in action:
            action['reason'] = 'automated_decision'
            
        if 'confidence' not in action:
            action['confidence'] = 0.0
            
        # Remove any unsafe fields
        unsafe_fields = ['override_safety', 'bypass_constraints']
        for field in unsafe_fields:
            if field in action:
                del action[field]
                logger.warning(f"Removed unsafe field from action: {field}")
        
        return action

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize validator
    validator = ActionSchema()
    
    # Valid action example
    valid_action = {
        "type": "route_assignment",
        "timestamp": datetime.now().isoformat(),
        "vehicle_id": "v123",
        "route": ["wh1", "cust456", "cust789"],
        "estimated_completion": "2023-12-01T15:30:00Z"
    }
    
    is_valid, errors = validator.validate(valid_action)
    print(f"Valid action validation: is_valid={is_valid}, errors={errors}")
    
    # Invalid action example
    invalid_action = {
        "type": "invalid_type",
        "timestamp": "not-a-timestamp"
    }
    
    is_valid, errors = validator.validate(invalid_action)
    print(f"Invalid action validation: is_valid={is_valid}, errors={errors}")
    
    # Sanitize example
    sanitized = validator.sanitize({"type": "test"})
    print(f"Sanitized action: {sanitized}")