"""
Audit trail for safety system activities
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import os


class AuditEventType(Enum):
    """Types of audit events"""
    RULE_VIOLATION = "rule_violation"
    MASK_APPLICATION = "mask_application"
    OVERRIDE_REQUEST = "override_request"
    OVERRIDE_APPROVAL = "override_approval"
    OVERRIDE_REJECTION = "override_rejection"
    SAFETY_DECISION = "safety_decision"
    SYSTEM_ALERT = "system_alert"
    CONFIGURATION_CHANGE = "configuration_change"


@dataclass
class AuditEvent:
    """Record of a safety system event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    actor: str  # Who performed the action
    resource: str  # What was affected
    action: str  # What was done
    details: Dict[str, Any]
    severity: str = "info"
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuditTrail:
    """Maintain audit trail of safety system activities"""
    
    def __init__(self, log_file: str = "logs/safety_audit.log", 
                 max_events: int = 10000):
        """
        Initialize audit trail
        
        Args:
            log_file: Path to audit log file
            max_events: Maximum number of events to keep in memory
        """
        self.log_file = log_file
        self.max_events = max_events
        self.events = []
        self.logger = logging.getLogger('AuditTrail')
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        
        # Set up file logging
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: AuditEventType, actor: str, resource: str,
                  action: str, details: Dict[str, Any], severity: str = "info",
                  session_id: Optional[str] = None, ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None) -> str:
        """
        Log a safety system event
        
        Args:
            event_type: Type of event
            actor: Who performed the action
            resource: What was affected
            action: What was done
            details: Additional event details
            severity: Event severity
            session_id: Session identifier
            ip_address: IP address of actor
            user_agent: User agent of actor
            
        Returns:
            Event ID
        """
        # Generate event ID
        event_id = self._generate_event_id(event_type, actor, resource, action)
        
        # Create event
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            actor=actor,
            resource=resource,
            action=action,
            details=details,
            severity=severity,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store in memory (with rotation)
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)  # Remove oldest event
        
        # Log to file
        self._log_to_file(event)
        
        # Also log to standard logger
        log_message = (
            f"Audit Event - ID: {event_id} | Type: {event_type.value} | "
            f"Actor: {actor} | Resource: {resource} | Action: {action} | "
            f"Severity: {severity}"
        )
        if severity == "critical":
            self.logger.critical(log_message)
        elif severity == "error":
            self.logger.error(log_message)
        elif severity == "warning":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        return event_id
    
    def _generate_event_id(self, event_type: AuditEventType, actor: str,
                          resource: str, action: str) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().isoformat()
        unique_string = f"{event_type.value}_{actor}_{resource}_{action}_{timestamp}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def _log_to_file(self, event: AuditEvent) -> None:
        """Log event to file in JSON format"""
        try:
            event_dict = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'actor': event.actor,
                'resource': event.resource,
                'action': event.action,
                'details': event.details,
                'severity': event.severity,
                'session_id': event.session_id,
                'ip_address': event.ip_address,
                'user_agent': event.user_agent
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write audit event to file: {str(e)}")
    
    def get_events(self, event_type: Optional[AuditEventType] = None,
                   actor: Optional[str] = None, resource: Optional[str] = None,
                   severity: Optional[str] = None,
                   hours_back: Optional[int] = None) -> List[AuditEvent]:
        """
        Retrieve audit events based on filters
        
        Args:
            event_type: Filter by event type
            actor: Filter by actor
            resource: Filter by resource
            severity: Filter by severity
            hours_back: Filter by time (hours ago)
            
        Returns:
            List of matching events
        """
        filtered_events = self.events.copy()
        
        # Apply filters
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if actor:
            filtered_events = [e for e in filtered_events if e.actor == actor]
        
        if resource:
            filtered_events = [e for e in filtered_events if e.resource == resource]
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        if hours_back:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            filtered_events = [e for e in filtered_events if e.timestamp >= cutoff_time]
        
        return filtered_events
    
    def get_event_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get summary of recent audit events
        
        Args:
            hours_back: Hours to look back
            
        Returns:
            Event summary
        """
        recent_events = self.get_events(hours_back=hours_back)
        
        # Group by event type
        event_type_counts = {}
        severity_counts = {}
        actor_activity = {}
        
        for event in recent_events:
            # Count by event type
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Count by severity
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            
            # Count by actor
            actor_activity[event.actor] = actor_activity.get(event.actor, 0) + 1
        
        return {
            'period_start': (datetime.now() - timedelta(hours=hours_back)).isoformat(),
            'period_end': datetime.now().isoformat(),
            'total_events': len(recent_events),
            'events_by_type': event_type_counts,
            'events_by_severity': severity_counts,
            'activity_by_actor': actor_activity,
            'most_active_actor': max(actor_activity.items(), key=lambda x: x[1])[0] if actor_activity else None
        }
    
    def export_events(self, filename: str, hours_back: Optional[int] = None) -> bool:
        """
        Export events to JSON file
        
        Args:
            filename: Output filename
            hours_back: Hours to look back (None = all events)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            events_to_export = self.get_events(hours_back=hours_back)
            
            export_data = []
            for event in events_to_export:
                export_data.append({
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'actor': event.actor,
                    'resource': event.resource,
                    'action': event.action,
                    'details': event.details,
                    'severity': event.severity,
                    'session_id': event.session_id,
                    'ip_address': event.ip_address,
                    'user_agent': event.user_agent
                })
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(events_to_export)} events to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export events: {str(e)}")
            return False


class ComplianceReporter:
    """Generate compliance reports from audit trail"""
    
    def __init__(self, audit_trail: AuditTrail):
        """
        Initialize compliance reporter
        
        Args:
            audit_trail: Audit trail instance
        """
        self.audit_trail = audit_trail
        self.logger = logging.getLogger('ComplianceReporter')
    
    def generate_compliance_report(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Generate compliance report for specified period
        
        Args:
            period_days: Period in days to report on
            
        Returns:
            Compliance report
        """
        hours_back = period_days * 24
        events = self.audit_trail.get_events(hours_back=hours_back)
        
        # Calculate compliance metrics
        total_events = len(events)
        critical_events = len([e for e in events if e.severity == 'critical'])
        warning_events = len([e for e in events if e.severity == 'warning'])
        override_events = len([e for e in events if 'override' in e.event_type.value])
        
        # Rule violation analysis
        violation_events = [e for e in events if e.event_type == AuditEventType.RULE_VIOLATION]
        unique_violations = len(set(e.resource for e in violation_events))
        
        # Override analysis
        override_requests = [e for e in events if e.event_type == AuditEventType.OVERRIDE_REQUEST]
        override_approvals = [e for e in events if e.event_type == AuditEventType.OVERRIDE_APPROVAL]
        override_rejections = [e for e in events if e.event_type == AuditEventType.OVERRIDE_REJECTION]
        
        approval_rate = (
            len(override_approvals) / len(override_requests) * 100 
            if override_requests else 0
        )
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'period_days': period_days,
            'period_start': (datetime.now() - timedelta(days=period_days)).isoformat(),
            'period_end': datetime.now().isoformat(),
            'summary': {
                'total_events': total_events,
                'critical_events': critical_events,
                'warning_events': warning_events,
                'override_events': override_events,
                'unique_rule_violations': unique_violations
            },
            'violation_analysis': {
                'total_violations': len(violation_events),
                'unique_rules_violated': unique_violations,
                'most_violated_rule': self._get_most_violated_rule(violation_events)
            },
            'override_analysis': {
                'total_requests': len(override_requests),
                'total_approvals': len(override_approvals),
                'total_rejections': len(override_rejections),
                'approval_rate_percent': round(approval_rate, 2)
            },
            'compliance_score': self._calculate_compliance_score(
                total_events, critical_events, warning_events, len(violation_events)
            )
        }
        
        return report
    
    def _get_most_violated_rule(self, violation_events: List[AuditEvent]) -> Optional[str]:
        """Get the most frequently violated rule"""
        if not violation_events:
            return None
        
        rule_counts = {}
        for event in violation_events:
            rule_counts[event.resource] = rule_counts.get(event.resource, 0) + 1
        
        return max(rule_counts.items(), key=lambda x: x[1])[0] if rule_counts else None
    
    def _calculate_compliance_score(self, total_events: int, critical_events: int,
                                 warning_events: int, violation_events: int) -> float:
        """Calculate overall compliance score (0-100)"""
        if total_events == 0:
            return 100.0
        
        # Weighted scoring: critical violations heavily penalized
        critical_penalty = critical_events * 10
        warning_penalty = warning_events * 2
        violation_penalty = violation_events * 1
        
        total_penalty = critical_penalty + warning_penalty + violation_penalty
        max_possible_penalty = total_events * 10  # Worst case scenario
        
        compliance_score = max(0, 100 - (total_penalty / max_possible_penalty * 100))
        return round(compliance_score, 2)
    
    def generate_violation_report(self, rule_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate detailed violation report
        
        Args:
            rule_id: Specific rule to report on (None = all rules)
            
        Returns:
            Violation report
        """
        # Get violation events
        filter_kwargs = {'event_type': AuditEventType.RULE_VIOLATION}
        if rule_id:
            filter_kwargs['resource'] = rule_id
        
        violation_events = self.audit_trail.get_events(**filter_kwargs)
        
        # Group by rule
        violations_by_rule = {}
        for event in violation_events:
            rule_id = event.resource
            if rule_id not in violations_by_rule:
                violations_by_rule[rule_id] = []
            violations_by_rule[rule_id].append(event)
        
        # Create detailed report
        report = {
            'report_generated': datetime.now().isoformat(),
            'total_violations': len(violation_events),
            'violations_by_rule': {}
        }
        
        for rule_id, events in violations_by_rule.items():
            # Group by actor
            violations_by_actor = {}
            for event in events:
                actor = event.actor
                if actor not in violations_by_actor:
                    violations_by_actor[actor] = []
                violations_by_actor[actor].append(event)
            
            report['violations_by_rule'][rule_id] = {
                'total_violations': len(events),
                'first_violation': min(e.timestamp for e in events).isoformat(),
                'last_violation': max(e.timestamp for e in events).isoformat(),
                'violations_by_actor': {
                    actor: len(actor_events)
                    for actor, actor_events in violations_by_actor.items()
                },
                'severity_distribution': self._get_severity_distribution(events)
            }
        
        return report
    
    def _get_severity_distribution(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Get distribution of events by severity"""
        severity_counts = {}
        for event in events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        return severity_counts


# Example usage
if __name__ == "__main__":
    # Create audit trail
    audit_trail = AuditTrail(log_file="logs/test_safety_audit.log")
    
    # Log some sample events
    audit_trail.log_event(
        event_type=AuditEventType.RULE_VIOLATION,
        actor="system",
        resource="capacity_limit_001",
        action="detected_violation",
        details={"current_usage": 1200, "limit": 1000, "location": "WH001"},
        severity="warning"
    )
    
    audit_trail.log_event(
        event_type=AuditEventType.OVERRIDE_REQUEST,
        actor="operator_001",
        resource="capacity_limit_001",
        action="requested_override",
        details={"reason": "urgent_shipment", "duration_minutes": 60},
        severity="info"
    )
    
    audit_trail.log_event(
        event_type=AuditEventType.OVERRIDE_APPROVAL,
        actor="supervisor_001",
        resource="capacity_limit_001",
        action="approved_override",
        details={"justification": "valid_business_need"},
        severity="info"
    )
    
    # Get event summary
    summary = audit_trail.get_event_summary(hours_back=24)
    print("Event Summary:")
    print(f"  Total events: {summary['total_events']}")
    print(f"  Events by type: {summary['events_by_type']}")
    
    # Generate compliance report
    compliance_reporter = ComplianceReporter(audit_trail)
    compliance_report = compliance_reporter.generate_compliance_report(period_days=7)
    
    print("\nCompliance Report:")
    print(f"  Compliance score: {compliance_report['compliance_score']}")
    print(f"  Critical events: {compliance_report['summary']['critical_events']}")
    print(f"  Override approval rate: {compliance_report['override_analysis']['approval_rate_percent']}%")
    
    # Export events
    exported = audit_trail.export_events("exported_events.json", hours_back=24)
    print(f"\nEvents exported: {exported}")
</file>