"""
Anomaly detection for logistics operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyType(Enum):
    """Types of anomalies"""
    DEMAND_SPIKE = "demand_spike"
    COST_OUTLIER = "cost_outlier"
    DELAY_ANOMALY = "delay_anomaly"
    INVENTORY_DEVIATION = "inventory_deviation"
    ROUTE_EFFICIENCY = "route_efficiency"
    CAPACITY_UTILIZATION = "capacity_utilization"


class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""
    anomaly_id: str
    anomaly_type: AnomalyType
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    severity: AnomalySeverity
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class AnomalyDetector:
    """Base class for anomaly detectors"""
    
    def __init__(self, name: str, anomaly_type: AnomalyType):
        self.name = name
        self.anomaly_type = anomaly_type
        self.logger = logging.getLogger(f'AnomalyDetector.{name}')
    
    def detect(self, data: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in data
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            List of detected anomalies
        """
        raise NotImplementedError("Subclasses must implement detect method")


class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detector using z-scores and percentiles"""
    
    def __init__(self, name: str, anomaly_type: AnomalyType, 
                 threshold_sigma: float = 3.0, window_size: int = 30):
        """
        Initialize statistical anomaly detector
        
        Args:
            name: Detector name
            anomaly_type: Type of anomaly to detect
            threshold_sigma: Number of standard deviations for anomaly threshold
            window_size: Size of rolling window for statistics
        """
        super().__init__(name, anomaly_type)
        self.threshold_sigma = threshold_sigma
        self.window_size = window_size
    
    def detect(self, data: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using statistical methods
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if data.empty:
            return anomalies
        
        # Assume the first numeric column is the value to monitor
        value_column = data.select_dtypes(include=[np.number]).columns[0]
        values = data[value_column]
        timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else data.iloc[:, 0]
        
        # Calculate rolling statistics
        rolling_mean = values.rolling(window=self.window_size, min_periods=1).mean()
        rolling_std = values.rolling(window=self.window_size, min_periods=1).std()
        
        # Calculate z-scores
        z_scores = np.abs((values - rolling_mean) / rolling_std)
        
        # Detect anomalies
        anomaly_indices = np.where(z_scores > self.threshold_sigma)[0]
        
        for idx in anomaly_indices:
            value = values.iloc[idx]
            expected = rolling_mean.iloc[idx]
            deviation = abs(value - expected)
            z_score = z_scores.iloc[idx]
            
            # Determine severity based on z-score
            if z_score > self.threshold_sigma * 2:
                severity = AnomalySeverity.CRITICAL
            elif z_score > self.threshold_sigma * 1.5:
                severity = AnomalySeverity.HIGH
            elif z_score > self.threshold_sigma:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW
            
            # Confidence based on how extreme the anomaly is
            confidence = min(1.0, z_score / (self.threshold_sigma * 3))
            
            anomaly = AnomalyDetectionResult(
                anomaly_id=f"{self.name}_{idx}_{timestamps[idx].strftime('%Y%m%d_%H%M%S')}",
                anomaly_type=self.anomaly_type,
                timestamp=timestamps[idx] if isinstance(timestamps[idx], datetime) else datetime.now(),
                value=float(value),
                expected_value=float(expected),
                deviation=float(deviation),
                severity=severity,
                confidence=float(confidence),
                details={
                    'z_score': float(z_score),
                    'rolling_mean': float(expected),
                    'rolling_std': float(rolling_std.iloc[idx]),
                    'window_size': self.window_size
                }
            )
            
            anomalies.append(anomaly)
        
        return anomalies


class IsolationForestDetector(AnomalyDetector):
    """Anomaly detector using Isolation Forest algorithm"""
    
    def __init__(self, name: str, anomaly_type: AnomalyType,
                 contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest detector
        
        Args:
            name: Detector name
            anomaly_type: Type of anomaly to detect
            contamination: Expected proportion of anomalies
            random_state: Random state for reproducibility
        """
        super().__init__(name, anomaly_type)
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def detect(self, data: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            data: DataFrame with features for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if data.empty:
            return anomalies
        
        # Select numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return anomalies
        
        feature_data = data[numeric_columns]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Train model if not already trained
        if not self.is_trained:
            self.model.fit(scaled_features)
            self.is_trained = True
        
        # Predict anomalies
        predictions = self.model.predict(scaled_features)
        anomaly_scores = self.model.decision_function(scaled_features)
        
        # Identify anomalies (predictions = -1 for anomalies)
        anomaly_indices = np.where(predictions == -1)[0]
        
        timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.RangeIndex(len(data))
        
        for idx in anomaly_indices:
            # Calculate severity based on anomaly score
            score = anomaly_scores[idx]
            normalized_score = (score + 0.5) / 1.0  # Normalize to 0-1 range
            
            if normalized_score < -0.5:
                severity = AnomalySeverity.CRITICAL
            elif normalized_score < -0.2:
                severity = AnomalySeverity.HIGH
            elif normalized_score < 0:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW
            
            # Confidence based on how anomalous the point is
            confidence = abs(normalized_score)
            
            # Get feature values for this point
            feature_values = feature_data.iloc[idx].to_dict()
            
            anomaly = AnomalyDetectionResult(
                anomaly_id=f"{self.name}_{idx}_{timestamps[idx] if isinstance(timestamps[idx], (str, int)) else datetime.now().strftime('%Y%m%d_%H%M%S')}",
                anomaly_type=self.anomaly_type,
                timestamp=timestamps[idx] if isinstance(timestamps[idx], datetime) else datetime.now(),
                value=float(feature_values[list(feature_values.keys())[0]]),  # First feature value
                expected_value=0.0,  # Isolation Forest doesn't provide expected values directly
                deviation=abs(float(anomaly_scores[idx])),
                severity=severity,
                confidence=float(confidence),
                details={
                    'anomaly_score': float(score),
                    'feature_values': feature_values,
                    'algorithm': 'isolation_forest'
                }
            )
            
            anomalies.append(anomaly)
        
        return anomalies


class ThresholdBasedDetector(AnomalyDetector):
    """Anomaly detector using predefined thresholds"""
    
    def __init__(self, name: str, anomaly_type: AnomalyType,
                 lower_threshold: Optional[float] = None,
                 upper_threshold: Optional[float] = None):
        """
        Initialize threshold-based detector
        
        Args:
            name: Detector name
            anomaly_type: Type of anomaly to detect
            lower_threshold: Lower threshold (None = no lower bound)
            upper_threshold: Upper threshold (None = no upper bound)
        """
        super().__init__(name, anomaly_type)
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
    
    def detect(self, data: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using thresholds
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if data.empty:
            return anomalies
        
        # Assume the first numeric column is the value to monitor
        value_column = data.select_dtypes(include=[np.number]).columns[0]
        values = data[value_column]
        timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else data.iloc[:, 0]
        
        # Detect threshold violations
        violation_indices = []
        
        if self.lower_threshold is not None:
            lower_violations = np.where(values < self.lower_threshold)[0]
            violation_indices.extend(lower_violations)
        
        if self.upper_threshold is not None:
            upper_violations = np.where(values > self.upper_threshold)[0]
            violation_indices.extend(upper_violations)
        
        # Remove duplicates and sort
        violation_indices = sorted(list(set(violation_indices)))
        
        for idx in violation_indices:
            value = values.iloc[idx]
            deviation = 0.0
            expected_value = 0.0
            
            # Calculate deviation and expected value
            if self.lower_threshold is not None and value < self.lower_threshold:
                deviation = self.lower_threshold - value
                expected_value = self.lower_threshold
            elif self.upper_threshold is not None and value > self.upper_threshold:
                deviation = value - self.upper_threshold
                expected_value = self.upper_threshold
            
            # Determine severity
            if self.lower_threshold is not None and value < self.lower_threshold * 0.5:
                severity = AnomalySeverity.CRITICAL
            elif self.upper_threshold is not None and value > self.upper_threshold * 1.5:
                severity = AnomalySeverity.CRITICAL
            else:
                severity = AnomalySeverity.HIGH
            
            anomaly = AnomalyDetectionResult(
                anomaly_id=f"{self.name}_{idx}_{timestamps[idx].strftime('%Y%m%d_%H%M%S') if hasattr(timestamps[idx], 'strftime') else 'unknown'}",
                anomaly_type=self.anomaly_type,
                timestamp=timestamps[idx] if isinstance(timestamps[idx], datetime) else datetime.now(),
                value=float(value),
                expected_value=float(expected_value),
                deviation=float(deviation),
                severity=severity,
                confidence=0.9,  # High confidence for threshold violations
                details={
                    'lower_threshold': self.lower_threshold,
                    'upper_threshold': self.upper_threshold,
                    'violation_type': 'below_lower' if value < (self.lower_threshold or float('-inf')) else 'above_upper'
                }
            )
            
            anomalies.append(anomaly)
        
        return anomalies


class CompositeAnomalyDetector:
    """Combine multiple anomaly detectors"""
    
    def __init__(self):
        self.detectors = []
        self.logger = logging.getLogger('CompositeAnomalyDetector')
    
    def add_detector(self, detector: AnomalyDetector) -> None:
        """
        Add an anomaly detector
        
        Args:
            detector: Anomaly detector to add
        """
        self.detectors.append(detector)
        self.logger.info(f"Added detector: {detector.name}")
    
    def detect_anomalies(self, data_dict: Dict[str, pd.DataFrame]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using all detectors
        
        Args:
            data_dict: Dictionary mapping detector names to data
            
        Returns:
            List of all detected anomalies
        """
        all_anomalies = []
        
        for detector in self.detectors:
            try:
                # Get data for this detector
                data = data_dict.get(detector.name, pd.DataFrame())
                if data.empty:
                    # Try to get data with similar name
                    matching_keys = [k for k in data_dict.keys() if detector.name.lower() in k.lower()]
                    if matching_keys:
                        data = data_dict[matching_keys[0]]
                
                if not data.empty:
                    anomalies = detector.detect(data)
                    all_anomalies.extend(anomalies)
                    self.logger.info(f"Detector {detector.name} found {len(anomalies)} anomalies")
                else:
                    self.logger.warning(f"No data found for detector: {detector.name}")
                    
            except Exception as e:
                self.logger.error(f"Error in detector {detector.name}: {str(e)}")
        
        return all_anomalies
    
    def get_detector_summary(self) -> Dict[str, int]:
        """
        Get summary of detectors and their types
        
        Returns:
            Dictionary mapping detector names to counts
        """
        summary = {}
        for detector in self.detectors:
            detector_type = type(detector).__name__
            summary[detector.name] = detector_type
        return summary


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Normal demand data with some outliers
    demand_data = np.random.poisson(100, 100)
    demand_data[10] = 300  # Spike
    demand_data[50] = 10   # Drop
    
    demand_df = pd.DataFrame({
        'timestamp': dates,
        'demand': demand_data
    }).set_index('timestamp')
    
    # Create detectors
    composite_detector = CompositeAnomalyDetector()
    
    # Add statistical detector
    stat_detector = StatisticalAnomalyDetector(
        name="demand_statistical",
        anomaly_type=AnomalyType.DEMAND_SPIKE,
        threshold_sigma=2.5
    )
    composite_detector.add_detector(stat_detector)
    
    # Add threshold detector
    threshold_detector = ThresholdBasedDetector(
        name="demand_threshold",
        anomaly_type=AnomalyType.DEMAND_SPIKE,
        lower_threshold=50,
        upper_threshold=200
    )
    composite_detector.add_detector(threshold_detector)
    
    # Detect anomalies
    data_dict = {"demand_statistical": demand_df, "demand_threshold": demand_df}
    anomalies = composite_detector.detect_anomalies(data_dict)
    
    print(f"Detected {len(anomalies)} anomalies:")
    for anomaly in anomalies[:5]:  # Show first 5
        print(f"  - {anomaly.anomaly_type.value}: {anomaly.value} at {anomaly.timestamp}")
        print(f"    Severity: {anomaly.severity.value}, Confidence: {anomaly.confidence:.2f}")
</file>