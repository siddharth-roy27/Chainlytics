"""
Safety rules engine for logistics operations
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta


class RuleType(Enum):
    """Types of safety rules"""
    CAPACITY_LIMIT = "capacity_limit"
    TIME_CONSTRAINT = "time_constraint"
    LEGAL_COMPLIANCE = "legal_compliance"
    BUSINESS_POLICY = "business_policy"
    OPERATIONAL_SAFETY = "operational_safety"
    QUALITY_CONTROL = "quality_control"


class RuleSeverity(Enum):
    """Severity levels for rule violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyRule:
    """Definition of a safety rule"""
    rule_id: str
    rule_type: RuleType
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    severity: RuleSeverity
    enabled: bool = True
    violation_count: int = 0
    last_violation: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleViolation:
    """Record of a rule violation"""
    rule_id: str
    timestamp: datetime
    severity: RuleSeverity
    violated_conditions: Dict[str, Any]
    corrective_action: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class SafetyRulesEngine:
    """Engine for evaluating and enforcing safety rules"""
    
    def __init__(self):
        self.rules = {}
        self.violations = []
        self.logger = logging.getLogger('SafetyRulesEngine')
    
    def add_rule(self, rule: SafetyRule) -> None:
        """
        Add a safety rule to the engine
        
        Args:
            rule: Safety rule to add
        """
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added safety rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a safety rule
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if removed, False if not found
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed safety rule: {rule_id}")
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable a safety rule
        
        Args:
            rule_id: ID of rule to enable
            
        Returns:
            True if enabled, False if not found
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.logger.info(f"Enabled safety rule: {rule_id}")
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable a safety rule
        
        Args:
            rule_id: ID of rule to disable
            
        Returns:
            True if disabled, False if not found
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.logger.info(f"Disabled safety rule: {rule_id}")
            return True
        return False
    
    def evaluate_rules(self, state: Dict[str, Any]) -> List[RuleViolation]:
        """
        Evaluate all enabled rules against current state
        
        Args:
            state: Current system state
            
        Returns:
            List of rule violations
        """
        violations = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check if rule condition is violated
                if not rule.condition(state):
                    # Rule violated - apply corrective action
                    corrective_action = rule.action(state)
                    
                    # Create violation record
                    violation = RuleViolation(
                        rule_id=rule_id,
                        timestamp=datetime.now(),
                        severity=rule.severity,
                        violated_conditions=state,
                        corrective_action=corrective_action
                    )
                    
                    violations.append(violation)
                    
                    # Update rule statistics
                    rule.violation_count += 1
                    rule.last_violation = datetime.now()
                    
                    self.logger.warning(
                        f"Rule violation: {rule_id} | Severity: {rule.severity.value} | "
                        f"Corrective action: {corrective_action}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule_id}: {str(e)}")
        
        # Store violations
        self.violations.extend(violations)
        
        return violations
    
    def get_violation_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate violation report for specified time window
        
        Args:
            time_window_hours: Time window in hours
            
        Returns:
            Violation report
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_violations = [
            v for v in self.violations 
            if v.timestamp >= cutoff_time
        ]
        
        # Group by severity
        severity_counts = {}
        for violation in recent_violations:
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Group by rule
        rule_counts = {}
        for violation in recent_violations:
            rule_counts[violation.rule_id] = rule_counts.get(violation.rule_id, 0) + 1
        
        return {
            'report_generated': datetime.now().isoformat(),
            'time_window_hours': time_window_hours,
            'total_violations': len(recent_violations),
            'violations_by_severity': severity_counts,
            'violations_by_rule': rule_counts,
            'recent_violations': [
                {
                    'rule_id': v.rule_id,
                    'timestamp': v.timestamp.isoformat(),
                    'severity': v.severity.value,
                    'resolved': v.resolved
                }
                for v in recent_violations[-10:]  # Last 10 violations
            ]
        }
    
    def resolve_violation(self, rule_id: str, resolution_details: Dict[str, Any] = None) -> bool:
        """
        Mark a violation as resolved
        
        Args:
            rule_id: ID of rule that was violated
            resolution_details: Details about resolution
            
        Returns:
            True if violation found and marked resolved, False otherwise
        """
        # Find most recent unresolved violation for this rule
        for violation in reversed(self.violations):
            if violation.rule_id == rule_id and not violation.resolved:
                violation.resolved = True
                violation.resolution_time = datetime.now()
                violation.resolution_details = resolution_details or {}
                self.logger.info(f"Resolved violation for rule: {rule_id}")
                return True
        
        return False


class PredefinedRules:
    """Collection of predefined safety rules"""
    
    @staticmethod
    def create_capacity_limit_rule(rule_id: str, capacity_type: str, 
                                limit: float) -> SafetyRule:
        """
        Create a capacity limit rule
        
        Args:
            rule_id: Unique rule ID
            capacity_type: Type of capacity to limit
            limit: Maximum capacity limit
            
        Returns:
            Safety rule
        """
        def condition(state: Dict[str, Any]) -> bool:
            current_capacity = state.get(f'{capacity_type}_usage', 0)
            return current_capacity <= limit
        
        def action(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                'type': 'capacity_violation',
                'capacity_type': capacity_type,
                'current_usage': state.get(f'{capacity_type}_usage', 0),
                'limit': limit,
                'recommended_action': 'reduce_load'
            }
        
        return SafetyRule(
            rule_id=rule_id,
            rule_type=RuleType.CAPACITY_LIMIT,
            description=f"Limit {capacity_type} usage to {limit}",
            condition=condition,
            action=action,
            severity=RuleSeverity.HIGH
        )
    
    @staticmethod
    def create_time_constraint_rule(rule_id: str, start_time: str, 
                                 end_time: str) -> SafetyRule:
        """
        Create a time constraint rule
        
        Args:
            rule_id: Unique rule ID
            start_time: Start time (HH:MM format)
            end_time: End time (HH:MM format)
            
        Returns:
            Safety rule
        """
        def condition(state: Dict[str, Any]) -> bool:
            current_time = datetime.now().time()
            start = datetime.strptime(start_time, "%H:%M").time()
            end = datetime.strptime(end_time, "%H:%M").time()
            
            # Handle overnight shifts
            if start <= end:
                return start <= current_time <= end
            else:
                return current_time >= start or current_time <= end
        
        def action(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                'type': 'time_violation',
                'current_time': datetime.now().strftime("%H:%M"),
                'allowed_window': f"{start_time}-{end_time}",
                'recommended_action': 'pause_operations'
            }
        
        return SafetyRule(
            rule_id=rule_id,
            rule_type=RuleType.TIME_CONSTRAINT,
            description=f"Operations only allowed between {start_time}-{end_time}",
            condition=condition,
            action=action,
            severity=RuleSeverity.MEDIUM
        )
    
    @staticmethod
    def create_legal_compliance_rule(rule_id: str, regulation: str, 
                                  requirement: str) -> SafetyRule:
        """
        Create a legal compliance rule
        
        Args:
            rule_id: Unique rule ID
            regulation: Regulation name
            requirement: Specific requirement
            
        Returns:
            Safety rule
        """
        def condition(state: Dict[str, Any]) -> bool:
            # In practice, this would check actual compliance status
            compliance_status = state.get(f'{regulation}_compliance', True)
            return compliance_status
        
        def action(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                'type': 'legal_violation',
                'regulation': regulation,
                'requirement': requirement,
                'recommended_action': 'halt_operations_and_notify_legal'
            }
        
        return SafetyRule(
            rule_id=rule_id,
            rule_type=RuleType.LEGAL_COMPLIANCE,
            description=f"Comply with {regulation}: {requirement}",
            condition=condition,
            action=action,
            severity=RuleSeverity.CRITICAL
        )


class RuleBasedSafetyController:
    """High-level controller that uses rules to ensure safety"""
    
    def __init__(self):
        self.rules_engine = SafetyRulesEngine()
        self.logger = logging.getLogger('RuleBasedSafetyController')
    
    def add_predefined_rules(self, rule_configs: List[Dict[str, Any]]) -> None:
        """
        Add predefined rules based on configuration
        
        Args:
            rule_configs: List of rule configuration dictionaries
        """
        for config in rule_configs:
            rule_type = config['type']
            rule_id = config['id']
            
            if rule_type == 'capacity_limit':
                rule = PredefinedRules.create_capacity_limit_rule(
                    rule_id, config['capacity_type'], config['limit']
                )
            elif rule_type == 'time_constraint':
                rule = PredefinedRules.create_time_constraint_rule(
                    rule_id, config['start_time'], config['end_time']
                )
            elif rule_type == 'legal_compliance':
                rule = PredefinedRules.create_legal_compliance_rule(
                    rule_id, config['regulation'], config['requirement']
                )
            else:
                self.logger.warning(f"Unknown rule type: {rule_type}")
                continue
            
            self.rules_engine.add_rule(rule)
    
    def check_safety(self, state: Dict[str, Any]) -> Tuple[bool, List[RuleViolation]]:
        """
        Check if current state is safe according to rules
        
        Args:
            state: Current system state
            
        Returns:
            Tuple of (is_safe, violations)
        """
        violations = self.rules_engine.evaluate_rules(state)
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def enforce_safety(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce safety by applying corrective actions for violations
        
        Args:
            state: Current system state
            
        Returns:
            Dictionary of corrective actions
        """
        violations = self.rules_engine.evaluate_rules(state)
        corrective_actions = {}
        
        for violation in violations:
            corrective_actions[violation.rule_id] = violation.corrective_action
        
        return corrective_actions


# Example usage
if __name__ == "__main__":
    # Create safety controller
    safety_controller = RuleBasedSafetyController()
    
    # Add predefined rules
    rule_configs = [
        {
            'type': 'capacity_limit',
            'id': 'vehicle_capacity_001',
            'capacity_type': 'vehicle',
            'limit': 1000.0
        },
        {
            'type': 'time_constraint',
            'id': 'operating_hours_001',
            'start_time': '08:00',
            'end_time': '18:00'
        }
    ]
    
    safety_controller.add_predefined_rules(rule_configs)
    
    # Test with sample state
    test_state = {
        'vehicle_usage': 800.0,
        'warehouse_usage': 4500.0
    }
    
    # Check safety
    is_safe, violations = safety_controller.check_safety(test_state)
    
    print(f"System is safe: {is_safe}")
    print(f"Violations found: {len(violations)}")
    
    for violation in violations:
        print(f"  - Rule: {violation.rule_id}")
        print(f"    Severity: {violation.severity.value}")
        print(f"    Action: {violation.corrective_action}")
    
    # Generate report
    report = safety_controller.rules_engine.get_violation_report()
    print(f"Violation report: {report['total_violations']} violations in last 24h")
</file>