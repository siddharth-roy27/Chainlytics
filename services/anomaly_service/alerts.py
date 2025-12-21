"""
Alert system for anomaly detection notifications
"""

import smtplib
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertSeverity(Enum):
    """Severity levels for alerts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Channels for alert delivery"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DATABASE = "database"


@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    channel: AlertChannel
    recipients: List[str]
    anomaly_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent: bool = False
    sent_timestamp: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_timestamp: Optional[datetime] = None


class AlertManager:
    """Manage alert generation and delivery"""
    
    def __init__(self, config_file: str = "config/alerts.json"):
        """
        Initialize alert manager
        
        Args:
            config_file: Path to alert configuration file
        """
        self.config_file = config_file
        self.alerts = []
        self.alert_rules = {}
        self.notification_channels = {}
        self.logger = logging.getLogger('AlertManager')
        self._load_config()
    
    def _load_config(self) -> None:
        """Load alert configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                self.alert_rules = config.get('alert_rules', {})
                self.notification_channels = config.get('notification_channels', {})
                
                self.logger.info(f"Loaded alert configuration from {self.config_file}")
            except Exception as e:
                self.logger.error(f"Error loading alert configuration: {str(e)}")
        else:
            # Create default configuration
            self._create_default_config()
            self._save_config()
    
    def _save_config(self) -> None:
        """Save alert configuration"""
        try:
            config = {
                'alert_rules': self.alert_rules,
                'notification_channels': self.notification_channels
            }
            
            os.makedirs(os.path.dirname(self.config_file) if os.path.dirname(self.config_file) else '.', exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Saved alert configuration to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving alert configuration: {str(e)}")
    
    def _create_default_config(self) -> None:
        """Create default alert configuration"""
        self.alert_rules = {
            'demand_spike': {
                'severity': 'high',
                'channels': ['email', 'slack'],
                'recipients': ['ops-team@company.com'],
                'threshold': 2.0,  # 2x normal demand
                'cooldown_hours': 1
            },
            'cost_outlier': {
                'severity': 'medium',
                'channels': ['email'],
                'recipients': ['finance@company.com'],
                'threshold': 1.5,  # 1.5x normal cost
                'cooldown_hours': 2
            },
            'delay_anomaly': {
                'severity': 'critical',
                'channels': ['email', 'sms'],
                'recipients': ['ops-manager@company.com'],
                'threshold': 48,  # 48 hours delay
                'cooldown_hours': 0.5
            }
        }
        
        self.notification_channels = {
            'email': {
                'smtp_server': 'smtp.company.com',
                'smtp_port': 587,
                'username': 'alerts@company.com',
                'password': 'secure_password'
            },
            'slack': {
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            }
        }
    
    def add_alert_rule(self, rule_name: str, rule_config: Dict[str, Any]) -> None:
        """
        Add an alert rule
        
        Args:
            rule_name: Name of the rule
            rule_config: Configuration for the rule
        """
        self.alert_rules[rule_name] = rule_config
        self._save_config()
        self.logger.info(f"Added alert rule: {rule_name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if removed, False if not found
        """
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self._save_config()
            self.logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False
    
    def generate_alert(self, anomaly: Any, rule_name: str) -> Optional[Alert]:
        """
        Generate alert based on anomaly and rule
        
        Args:
            anomaly: Detected anomaly
            rule_name: Name of alert rule to apply
            
        Returns:
            Generated alert or None if no alert should be generated
        """
        if rule_name not in self.alert_rules:
            self.logger.warning(f"Alert rule not found: {rule_name}")
            return None
        
        rule = self.alert_rules[rule_name]
        
        # Check cooldown period
        if not self._check_cooldown(rule_name, anomaly):
            return None
        
        # Create alert
        alert_id = f"alert_{rule_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(anomaly)) % 10000}"
        
        # Determine severity
        severity_str = rule.get('severity', 'medium')
        severity = AlertSeverity(severity_str)
        
        # Create alert message
        title = f"{severity_str.title()} Anomaly Detected: {rule_name.replace('_', ' ').title()}"
        message = self._format_alert_message(anomaly, rule)
        
        # Get channels and recipients
        channels = [AlertChannel(channel) for channel in rule.get('channels', ['email'])]
        recipients = rule.get('recipients', [])
        
        # Create alert for each channel
        alerts = []
        for channel in channels:
            alert = Alert(
                alert_id=f"{alert_id}_{channel.value}",
                severity=severity,
                title=title,
                message=message,
                timestamp=datetime.now(),
                channel=channel,
                recipients=recipients,
                anomaly_details=anomaly.__dict__ if hasattr(anomaly, '__dict__') else {},
                metadata={'rule_name': rule_name, 'generated_by': 'anomaly_detection'}
            )
            alerts.append(alert)
        
        # Store alerts
        self.alerts.extend(alerts)
        
        self.logger.info(f"Generated {len(alerts)} alerts for rule: {rule_name}")
        return alerts[0] if alerts else None
    
    def _check_cooldown(self, rule_name: str, anomaly: Any) -> bool:
        """
        Check if alert should be generated based on cooldown period
        
        Args:
            rule_name: Name of alert rule
            anomaly: Detected anomaly
            
        Returns:
            True if alert should be generated, False if in cooldown
        """
        rule = self.alert_rules[rule_name]
        cooldown_hours = rule.get('cooldown_hours', 1)
        
        if cooldown_hours <= 0:
            return True
        
        cooldown_cutoff = datetime.now() - timedelta(hours=cooldown_hours)
        
        # Check recent alerts for same rule
        recent_alerts = [
            alert for alert in self.alerts
            if (alert.metadata.get('rule_name') == rule_name and
                alert.timestamp >= cooldown_cutoff)
        ]
        
        return len(recent_alerts) == 0
    
    def _format_alert_message(self, anomaly: Any, rule: Dict[str, Any]) -> str:
        """
        Format alert message
        
        Args:
            anomaly: Detected anomaly
            rule: Alert rule configuration
            
        Returns:
            Formatted alert message
        """
        if hasattr(anomaly, 'anomaly_type'):
            anomaly_type = anomaly.anomaly_type.value
            value = getattr(anomaly, 'value', 'N/A')
            expected = getattr(anomaly, 'expected_value', 'N/A')
            deviation = getattr(anomaly, 'deviation', 'N/A')
            timestamp = getattr(anomaly, 'timestamp', datetime.now())
            
            message = (
                f"Anomaly detected in {anomaly_type}:\n\n"
                f"Value: {value}\n"
                f"Expected: {expected}\n"
                f"Deviation: {deviation}\n"
                f"Timestamp: {timestamp}\n\n"
                f"Severity: {rule.get('severity', 'medium')}\n"
                f"Please investigate this anomaly."
            )
        else:
            message = f"Anomaly detected: {str(anomaly)}"
        
        return message
    
    def send_alerts(self) -> Dict[str, int]:
        """
        Send pending alerts through configured channels
        
        Returns:
            Dictionary with send statistics
        """
        pending_alerts = [alert for alert in self.alerts if not alert.sent]
        
        if not pending_alerts:
            return {'sent': 0, 'failed': 0, 'skipped': 0}
        
        stats = {'sent': 0, 'failed': 0, 'skipped': 0}
        
        for alert in pending_alerts:
            try:
                if self._send_alert(alert):
                    alert.sent = True
                    alert.sent_timestamp = datetime.now()
                    stats['sent'] += 1
                    self.logger.info(f"Sent alert {alert.alert_id} via {alert.channel.value}")
                else:
                    stats['failed'] += 1
                    self.logger.error(f"Failed to send alert {alert.alert_id}")
                    
            except Exception as e:
                stats['failed'] += 1
                self.logger.error(f"Error sending alert {alert.alert_id}: {str(e)}")
        
        return stats
    
    def _send_alert(self, alert: Alert) -> bool:
        """
        Send alert through specific channel
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        channel = alert.channel
        
        if channel == AlertChannel.EMAIL:
            return self._send_email_alert(alert)
        elif channel == AlertChannel.SLACK:
            return self._send_slack_alert(alert)
        elif channel == AlertChannel.DATABASE:
            return self._log_database_alert(alert)
        else:
            self.logger.warning(f"Unsupported alert channel: {channel.value}")
            return False
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """
        Send alert via email
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            email_config = self.notification_channels.get('email', {})
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config.get('username', 'alerts@company.com')
            msg['To'] = ', '.join(alert.recipients)
            msg['Subject'] = alert.title
            
            body = alert.message
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (simplified - in production, use proper SMTP setup)
            self.logger.info(f"Email alert would be sent to {alert.recipients}")
            self.logger.debug(f"Email content: {body}")
            
            return True  # Simulated success
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def _send_slack_alert(self, alert: Alert) -> bool:
        """
        Send alert via Slack webhook
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            slack_config = self.notification_channels.get('slack', {})
            webhook_url = slack_config.get('webhook_url')
            
            if not webhook_url:
                self.logger.warning("Slack webhook URL not configured")
                return False
            
            # In production, you would use requests to send to webhook
            self.logger.info(f"Slack alert would be sent to webhook: {webhook_url}")
            self.logger.debug(f"Slack message: {alert.message}")
            
            return True  # Simulated success
            
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {str(e)}")
            return False
    
    def _log_database_alert(self, alert: Alert) -> bool:
        """
        Log alert to database
        
        Args:
            alert: Alert to log
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            # In production, you would insert into a database
            self.logger.info(f"Database alert logged: {alert.alert_id}")
            self.logger.debug(f"Alert details: {alert.__dict__}")
            
            return True  # Simulated success
            
        except Exception as e:
            self.logger.error(f"Error logging database alert: {str(e)}")
            return False
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Person acknowledging the alert
            
        Returns:
            True if acknowledged, False if alert not found
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_timestamp = datetime.now()
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False
    
    def get_pending_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get pending (unsent) alerts
        
        Args:
            severity: Filter by severity (optional)
            
        Returns:
            List of pending alerts
        """
        pending_alerts = [alert for alert in self.alerts if not alert.sent]
        
        if severity:
            pending_alerts = [alert for alert in pending_alerts if alert.severity == severity]
        
        return pending_alerts
    
    def get_alert_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get alert statistics
        
        Args:
            hours_back: Hours to look back
            
        Returns:
            Alert statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
        
        # Group by severity
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Group by channel
        channel_counts = {}
        for alert in recent_alerts:
            channel = alert.channel.value
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        # Sent vs pending
        sent_count = len([alert for alert in recent_alerts if alert.sent])
        pending_count = len([alert for alert in recent_alerts if not alert.sent])
        acknowledged_count = len([alert for alert in recent_alerts if alert.acknowledged])
        
        return {
            'period_start': cutoff_time.isoformat(),
            'period_end': datetime.now().isoformat(),
            'total_alerts': len(recent_alerts),
            'alerts_by_severity': severity_counts,
            'alerts_by_channel': channel_counts,
            'sent_alerts': sent_count,
            'pending_alerts': pending_count,
            'acknowledged_alerts': acknowledged_count
        }


class AlertCorrelationEngine:
    """Correlate multiple alerts to identify patterns"""
    
    def __init__(self, alert_manager: AlertManager):
        """
        Initialize correlation engine
        
        Args:
            alert_manager: Alert manager instance
        """
        self.alert_manager = alert_manager
        self.logger = logging.getLogger('AlertCorrelationEngine')
    
    def correlate_alerts(self, time_window_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Correlate alerts within time window
        
        Args:
            time_window_minutes: Time window for correlation
            
        Returns:
            List of correlated alert patterns
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_alerts = [
            alert for alert in self.alert_manager.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        if len(recent_alerts) < 2:
            return []  # Need at least 2 alerts to correlate
        
        # Group alerts by time clusters
        clusters = self._cluster_alerts_by_time(recent_alerts, timedelta(minutes=10))
        
        # Identify patterns
        patterns = []
        for cluster in clusters:
            if len(cluster) > 1:
                pattern = {
                    'pattern_id': f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str([a.alert_id for a in cluster])) % 10000}",
                    'alerts': [alert.alert_id for alert in cluster],
                    'alert_types': list(set(alert.metadata.get('rule_name', 'unknown') for alert in cluster)),
                    'severity_distribution': self._get_severity_distribution(cluster),
                    'time_span_minutes': (max(a.timestamp for a in cluster) - min(a.timestamp for a in cluster)).total_seconds() / 60,
                    'correlation_score': self._calculate_correlation_score(cluster)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _cluster_alerts_by_time(self, alerts: List[Alert], time_threshold: timedelta) -> List[List[Alert]]:
        """
        Cluster alerts by time proximity
        
        Args:
            alerts: List of alerts to cluster
            time_threshold: Maximum time difference for clustering
            
        Returns:
            List of alert clusters
        """
        if not alerts:
            return []
        
        # Sort alerts by time
        sorted_alerts = sorted(alerts, key=lambda x: x.timestamp)
        
        clusters = []
        current_cluster = [sorted_alerts[0]]
        
        for alert in sorted_alerts[1:]:
            time_diff = alert.timestamp - current_cluster[-1].timestamp
            if time_diff <= time_threshold:
                current_cluster.append(alert)
            else:
                clusters.append(current_cluster)
                current_cluster = [alert]
        
        clusters.append(current_cluster)
        return clusters
    
    def _get_severity_distribution(self, alerts: List[Alert]) -> Dict[str, int]:
        """Get distribution of alerts by severity"""
        distribution = {}
        for alert in alerts:
            severity = alert.severity.value
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution
    
    def _calculate_correlation_score(self, alerts: List[Alert]) -> float:
        """
        Calculate correlation score for a group of alerts
        
        Args:
            alerts: List of alerts
            
        Returns:
            Correlation score (0-1)
        """
        if len(alerts) < 2:
            return 0.0
        
        # Simple scoring based on number of alerts and severity diversity
        num_alerts = len(alerts)
        unique_severities = len(set(alert.severity for alert in alerts))
        
        # Higher score for more alerts and more diverse severities
        score = min(1.0, (num_alerts / 10.0) * (unique_severities / 3.0))
        return round(score, 2)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create alert manager
    alert_manager = AlertManager("config/test_alerts.json")
    
    # Add a sample alert rule
    alert_manager.add_alert_rule('test_rule', {
        'severity': 'high',
        'channels': ['email'],
        'recipients': ['test@example.com'],
        'cooldown_hours': 1
    })
    
    # Create a mock anomaly
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class MockAnomaly:
        anomaly_type: Any
        value: float
        expected_value: float
        deviation: float
        timestamp: datetime
    
    # Mock anomaly type enum
    class MockAnomalyType:
        def __init__(self, value):
            self.value = value
    
    mock_anomaly = MockAnomaly(
        anomaly_type=MockAnomalyType("demand_spike"),
        value=300.0,
        expected_value=100.0,
        deviation=200.0,
        timestamp=datetime.now()
    )
    
    # Generate alert
    alert = alert_manager.generate_alert(mock_anomaly, 'test_rule')
    if alert:
        print(f"Generated alert: {alert.title}")
        print(f"Message: {alert.message}")
        print(f"Recipients: {alert.recipients}")
    
    # Send alerts
    stats = alert_manager.send_alerts()
    print(f"Alert sending stats: {stats}")
    
    # Get statistics
    stats = alert_manager.get_alert_statistics()
    print(f"Alert statistics: {stats}")
</file>