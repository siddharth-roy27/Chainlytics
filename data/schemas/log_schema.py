"""
Log Schema for Decision Logging
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class DecisionLog:
    """Schema for logging decisions made by the system."""
    
    timestamp: str
    input_state: Dict[str, Any]
    action_proposed: Dict[str, Any]
    action_executed: Dict[str, Any]
    constraints_applied: Dict[str, Any]
    outcome: Optional[Dict[str, Any]]
    kpi_delta: Optional[Dict[str, float]]
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionLog':
        """Create DecisionLog from dictionary."""
        return cls(**data)

@dataclass
class StateSnapshot:
    """Schema for logging state snapshots."""
    
    timestamp: str
    warehouse_states: Dict[str, Any]
    vehicle_states: Dict[str, Any]
    order_states: Dict[str, Any]
    customer_states: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create StateSnapshot from dictionary."""
        return cls(**data)

@dataclass
class ActionLog:
    """Schema for logging actions taken."""
    
    timestamp: str
    action_type: str
    action_details: Dict[str, Any]
    actor: str  # Which service/component took the action
    duration_ms: float
    success: bool
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionLog':
        """Create ActionLog from dictionary."""
        return cls(**data)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example decision log
    decision_log = DecisionLog(
        timestamp=datetime.now().isoformat(),
        input_state={"warehouses": [{"id": "wh1", "inventory": 1000}]},
        action_proposed={"type": "route_assignment", "vehicle_id": "v1"},
        action_executed={"type": "route_assignment", "vehicle_id": "v1"},
        constraints_applied={"capacity_check": "passed"},
        outcome={"delivery_success": True},
        kpi_delta={"on_time_delivery": 0.02},
        processing_time_ms=45.2
    )
    
    print(f"Decision log created: {decision_log.to_dict()}")