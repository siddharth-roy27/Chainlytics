"""
Override mechanism for safety constraints
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import hashlib
import secrets


class OverrideLevel(Enum):
    """Levels of override authority"""
    OPERATOR = "operator"        # Basic operational override
    SUPERVISOR = "supervisor"    # Supervisory override
    MANAGER = "manager"         # Management override
    EMERGENCY = "emergency"     # Emergency override


class OverrideStatus(Enum):
    """Status of override requests"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OverrideRequest:
    """Request to override safety constraints"""
    request_id: str
    rule_id: str
    override_level: OverrideLevel
    requester_id: str
    reason: str
    timestamp: datetime
    expiration_time: Optional[datetime]
    status: OverrideStatus = OverrideStatus.PENDING
    approval_authority: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    justification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverrideLog:
    """Log of override activities"""
    override_id: str
    rule_id: str
    requester_id: str
    approver_id: Optional[str]
    request_time: datetime
    approval_time: Optional[datetime]
    override_level: OverrideLevel
    reason: str
    status: OverrideStatus
    duration_seconds: Optional[float] = None


class OverrideManager:
    """Manage override requests and approvals"""
    
    def __init__(self, approval_required: bool = True):
        """
        Initialize override manager
        
        Args:
            approval_required: Whether approvals are required for overrides
        """
        self.approval_required = approval_required
        self.override_requests = {}
        self.override_logs = []
        self.authorized_personnel = {}
        self.logger = logging.getLogger('OverrideManager')
    
    def add_authorized_personnel(self, personnel_id: str, 
                               authorized_levels: List[OverrideLevel]) -> None:
        """
        Add authorized personnel who can request/approve overrides
        
        Args:
            personnel_id: ID of authorized personnel
            authorized_levels: Levels this person is authorized for
        """
        self.authorized_personnel[personnel_id] = authorized_levels
        self.logger.info(f"Added authorized personnel: {personnel_id}")
    
    def request_override(self, rule_id: str, override_level: OverrideLevel,
                        requester_id: str, reason: str,
                        duration_minutes: Optional[int] = None) -> str:
        """
        Request an override of safety constraints
        
        Args:
            rule_id: ID of rule to override
            override_level: Level of override requested
            requester_id: ID of person requesting override
            reason: Reason for override
            duration_minutes: Duration of override in minutes (None = indefinite)
            
        Returns:
            Override request ID
        """
        # Validate requester authorization
        if not self._is_authorized(requester_id, override_level):
            raise PermissionError(f"Personnel {requester_id} not authorized for {override_level.value} overrides")
        
        # Generate request ID
        request_id = self._generate_request_id(rule_id, requester_id)
        
        # Calculate expiration time
        expiration_time = None
        if duration_minutes:
            expiration_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Create override request
        override_request = OverrideRequest(
            request_id=request_id,
            rule_id=rule_id,
            override_level=override_level,
            requester_id=requester_id,
            reason=reason,
            timestamp=datetime.now(),
            expiration_time=expiration_time
        )
        
        # Set status based on approval requirements
        if not self.approval_required or override_level == OverrideLevel.EMERGENCY:
            override_request.status = OverrideStatus.APPROVED
            override_request.approval_authority = requester_id
            override_request.approval_timestamp = datetime.now()
            self.logger.warning(f"Emergency override approved automatically: {request_id}")
        else:
            self.logger.info(f"Override request submitted: {request_id}")
        
        # Store request
        self.override_requests[request_id] = override_request
        
        return request_id
    
    def approve_override(self, request_id: str, approver_id: str,
                        justification: str = "") -> bool:
        """
        Approve an override request
        
        Args:
            request_id: ID of override request
            approver_id: ID of person approving override
            justification: Justification for approval
            
        Returns:
            True if approved, False if request not found or already processed
        """
        if request_id not in self.override_requests:
            self.logger.error(f"Override request not found: {request_id}")
            return False
        
        override_request = self.override_requests[request_id]
        
        # Check if already processed
        if override_request.status != OverrideStatus.PENDING:
            self.logger.warning(f"Override request {request_id} already processed")
            return False
        
        # Validate approver authorization
        required_level = override_request.override_level
        if not self._is_authorized(approver_id, required_level):
            raise PermissionError(f"Personnel {approver_id} not authorized to approve {required_level.value} overrides")
        
        # Approve request
        override_request.status = OverrideStatus.APPROVED
        override_request.approval_authority = approver_id
        override_request.approval_timestamp = datetime.now()
        override_request.justification = justification
        
        # Log approval
        self._log_override(override_request)
        
        self.logger.info(f"Override request approved: {request_id} by {approver_id}")
        return True
    
    def reject_override(self, request_id: str, rejector_id: str,
                       reason: str = "") -> bool:
        """
        Reject an override request
        
        Args:
            request_id: ID of override request
            rejector_id: ID of person rejecting override
            reason: Reason for rejection
            
        Returns:
            True if rejected, False if request not found or already processed
        """
        if request_id not in self.override_requests:
            return False
        
        override_request = self.override_requests[request_id]
        
        # Check if already processed
        if override_request.status != OverrideStatus.PENDING:
            return False
        
        # Validate rejector authorization
        if not self._is_authorized(rejector_id, override_request.override_level):
            raise PermissionError(f"Personnel {rejector_id} not authorized to process this override")
        
        # Reject request
        override_request.status = OverrideStatus.REJECTED
        override_request.approval_authority = rejector_id
        override_request.approval_timestamp = datetime.now()
        override_request.justification = reason
        
        self.logger.info(f"Override request rejected: {request_id} by {rejector_id}")
        return True
    
    def cancel_override(self, request_id: str, canceller_id: str) -> bool:
        """
        Cancel an active override
        
        Args:
            request_id: ID of override to cancel
            canceller_id: ID of person cancelling override
            
        Returns:
            True if cancelled, False if not found or not authorized
        """
        if request_id not in self.override_requests:
            return False
        
        override_request = self.override_requests[request_id]
        
        # Check if active
        if override_request.status != OverrideStatus.APPROVED:
            return False
        
        # Validate authorization
        if (canceller_id != override_request.requester_id and
            not self._is_authorized(canceller_id, override_request.override_level)):
            raise PermissionError("Not authorized to cancel this override")
        
        # Cancel override
        override_request.status = OverrideStatus.EXPIRED
        override_request.approval_timestamp = datetime.now()
        
        self.logger.info(f"Override cancelled: {request_id} by {canceller_id}")
        return True
    
    def check_override_status(self, rule_id: str, 
                            override_level: OverrideLevel = None) -> List[OverrideRequest]:
        """
        Check status of overrides for a specific rule
        
        Args:
            rule_id: ID of rule to check
            override_level: Specific override level to check (None = all levels)
            
        Returns:
            List of active override requests
        """
        active_overrides = []
        
        for override_request in self.override_requests.values():
            # Check if matches rule and is active
            if (override_request.rule_id == rule_id and
                override_request.status == OverrideStatus.APPROVED):
                
                # Check level if specified
                if override_level and override_request.override_level != override_level:
                    continue
                
                # Check expiration
                if (override_request.expiration_time and
                    datetime.now() > override_request.expiration_time):
                    # Expire override
                    override_request.status = OverrideStatus.EXPIRED
                    continue
                
                active_overrides.append(override_request)
        
        return active_overrides
    
    def _is_authorized(self, personnel_id: str, override_level: OverrideLevel) -> bool:
        """
        Check if personnel is authorized for a specific override level
        
        Args:
            personnel_id: ID of personnel
            override_level: Override level to check
            
        Returns:
            True if authorized, False otherwise
        """
        if personnel_id not in self.authorized_personnel:
            return False
        
        authorized_levels = self.authorized_personnel[personnel_id]
        
        # Check direct authorization
        if override_level in authorized_levels:
            return True
        
        # Check hierarchical authorization (higher levels can approve lower levels)
        level_hierarchy = [
            OverrideLevel.OPERATOR,
            OverrideLevel.SUPERVISOR,
            OverrideLevel.MANAGER,
            OverrideLevel.EMERGENCY
        ]
        
        personnel_level_indices = [level_hierarchy.index(level) for level in authorized_levels]
        required_level_index = level_hierarchy.index(override_level)
        
        return any(index >= required_level_index for index in personnel_level_indices)
    
    def _generate_request_id(self, rule_id: str, requester_id: str) -> str:
        """
        Generate unique request ID
        
        Args:
            rule_id: Rule ID
            requester_id: Requester ID
            
        Returns:
            Unique request ID
        """
        timestamp = datetime.now().isoformat()
        unique_string = f"{rule_id}_{requester_id}_{timestamp}_{secrets.token_hex(4)}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def _log_override(self, override_request: OverrideRequest) -> None:
        """
        Log override activity
        
        Args:
            override_request: Override request to log
        """
        duration = None
        if override_request.approval_timestamp and override_request.timestamp:
            duration = (override_request.approval_timestamp - override_request.timestamp).total_seconds()
        
        log_entry = OverrideLog(
            override_id=override_request.request_id,
            rule_id=override_request.rule_id,
            requester_id=override_request.requester_id,
            approver_id=override_request.approval_authority,
            request_time=override_request.timestamp,
            approval_time=override_request.approval_timestamp,
            override_level=override_request.override_level,
            reason=override_request.reason,
            status=override_request.status,
            duration_seconds=duration
        )
        
        self.override_logs.append(log_entry)
    
    def get_override_history(self, days: int = 30) -> List[OverrideLog]:
        """
        Get override history for specified number of days
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of override logs
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        return [
            log for log in self.override_logs
            if log.request_time >= cutoff_time
        ]


class SafetyOverrideEngine:
    """Integration point for safety overrides with rules and masks"""
    
    def __init__(self, override_manager: OverrideManager):
        """
        Initialize safety override engine
        
        Args:
            override_manager: Override manager instance
        """
        self.override_manager = override_manager
        self.logger = logging.getLogger('SafetyOverrideEngine')
    
    def check_override_permission(self, rule_id: str, 
                                override_level: OverrideLevel) -> bool:
        """
        Check if an override is permitted for a specific rule
        
        Args:
            rule_id: ID of rule
            override_level: Override level requested
            
        Returns:
            True if override permitted, False otherwise
        """
        # Check for active overrides
        active_overrides = self.override_manager.check_override_status(rule_id, override_level)
        
        # If there are active overrides at this level or higher, permit override
        return len(active_overrides) > 0
    
    def apply_override_filter(self, violations: List[Any], 
                            state: Dict[str, Any]) -> List[Any]:
        """
        Filter violations based on active overrides
        
        Args:
            violations: List of rule violations
            state: Current system state
            
        Returns:
            Filtered list of violations (permitted overrides removed)
        """
        filtered_violations = []
        
        for violation in violations:
            rule_id = getattr(violation, 'rule_id', None)
            if not rule_id:
                # Not a rule violation, pass through
                filtered_violations.append(violation)
                continue
            
            # Check for active overrides
            has_override = self.check_override_permission(rule_id, OverrideLevel.OPERATOR)
            
            if not has_override:
                filtered_violations.append(violation)
            else:
                self.logger.info(f"Violation suppressed by override: {rule_id}")
        
        return filtered_violations
    
    def get_override_summary(self) -> Dict[str, Any]:
        """
        Get summary of current override status
        
        Returns:
            Override summary
        """
        # Count active overrides by level
        override_counts = {}
        for override_request in self.override_manager.override_requests.values():
            if override_request.status == OverrideStatus.APPROVED:
                level = override_request.override_level.value
                override_counts[level] = override_counts.get(level, 0) + 1
        
        return {
            'active_overrides': override_counts,
            'total_overrides': len(self.override_manager.override_requests),
            'pending_requests': len([
                req for req in self.override_manager.override_requests.values()
                if req.status == OverrideStatus.PENDING
            ])
        }


# Example usage
if __name__ == "__main__":
    # Create override manager
    override_manager = OverrideManager(approval_required=True)
    
    # Add authorized personnel
    override_manager.add_authorized_personnel("operator_001", [OverrideLevel.OPERATOR])
    override_manager.add_authorized_personnel("supervisor_001", [OverrideLevel.OPERATOR, OverrideLevel.SUPERVISOR])
    override_manager.add_authorized_personnel("manager_001", [OverrideLevel.OPERATOR, OverrideLevel.SUPERVISOR, OverrideLevel.MANAGER])
    
    # Request override
    try:
        request_id = override_manager.request_override(
            rule_id="capacity_limit_001",
            override_level=OverrideLevel.SUPERVISOR,
            requester_id="operator_001",
            reason="Urgent shipment needs extra capacity",
            duration_minutes=60
        )
        print(f"Override request created: {request_id}")
    except PermissionError as e:
        print(f"Permission denied: {e}")
    
    # Approve override (supervisor)
    approved = override_manager.approve_override(
        request_id=request_id,
        approver_id="supervisor_001",
        justification="Valid business need, monitored closely"
    )
    print(f"Override approved: {approved}")
    
    # Check override status
    active_overrides = override_manager.check_override_status("capacity_limit_001")
    print(f"Active overrides: {len(active_overrides)}")
    
    # Create override engine
    override_engine = SafetyOverrideEngine(override_manager)
    summary = override_engine.get_override_summary()
    print(f"Override summary: {summary}")
