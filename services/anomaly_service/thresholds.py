"""
Dynamic threshold management for anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import json
import os


@dataclass
class DynamicThreshold:
    """Dynamic threshold with adaptive bounds"""
    metric_name: str
    lower_bound: float
    upper_bound: float
    baseline_value: float
    confidence_interval: Tuple[float, float]
    last_updated: datetime
    update_frequency: str  # 'hourly', 'daily', 'weekly'
    seasonality_adjusted: bool = False
    seasonal_factors: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThresholdManager:
    """Manage dynamic thresholds for anomaly detection"""
    
    def __init__(self, thresholds_file: str = "config/thresholds.json"):
        """
        Initialize threshold manager
        
        Args:
            thresholds_file: Path to thresholds configuration file
        """
        self.thresholds_file = thresholds_file
        self.thresholds = {}
        self.logger = logging.getLogger('ThresholdManager')
        self._load_thresholds()
    
    def _load_thresholds(self) -> None:
        """Load thresholds from configuration file"""
        if os.path.exists(self.thresholds_file):
            try:
                with open(self.thresholds_file, 'r') as f:
                    thresholds_data = json.load(f)
                
                for metric_name, threshold_data in thresholds_data.items():
                    # Convert datetime string to datetime object
                    if 'last_updated' in threshold_data:
                        threshold_data['last_updated'] = datetime.fromisoformat(
                            threshold_data['last_updated']
                        )
                    
                    self.thresholds[metric_name] = DynamicThreshold(**threshold_data)
                
                self.logger.info(f"Loaded {len(self.thresholds)} thresholds from {self.thresholds_file}")
            except Exception as e:
                self.logger.error(f"Error loading thresholds: {str(e)}")
        else:
            # Create default thresholds
            self._create_default_thresholds()
            self._save_thresholds()
    
    def _save_thresholds(self) -> None:
        """Save thresholds to configuration file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            thresholds_data = {}
            for metric_name, threshold in self.thresholds.items():
                threshold_dict = threshold.__dict__.copy()
                if isinstance(threshold_dict['last_updated'], datetime):
                    threshold_dict['last_updated'] = threshold_dict['last_updated'].isoformat()
                thresholds_data[metric_name] = threshold_dict
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.thresholds_file) if os.path.dirname(self.thresholds_file) else '.', exist_ok=True)
            
            with open(self.thresholds_file, 'w') as f:
                json.dump(thresholds_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.thresholds)} thresholds to {self.thresholds_file}")
        except Exception as e:
            self.logger.error(f"Error saving thresholds: {str(e)}")
    
    def _create_default_thresholds(self) -> None:
        """Create default thresholds for common metrics"""
        default_thresholds = {
            'daily_demand': DynamicThreshold(
                metric_name='daily_demand',
                lower_bound=50.0,
                upper_bound=500.0,
                baseline_value=200.0,
                confidence_interval=(150.0, 250.0),
                last_updated=datetime.now(),
                update_frequency='daily'
            ),
            'delivery_time_hours': DynamicThreshold(
                metric_name='delivery_time_hours',
                lower_bound=2.0,
                upper_bound=48.0,
                baseline_value=24.0,
                confidence_interval=(12.0, 36.0),
                last_updated=datetime.now(),
                update_frequency='daily'
            ),
            'cost_per_shipment': DynamicThreshold(
                metric_name='cost_per_shipment',
                lower_bound=10.0,
                upper_bound=500.0,
                baseline_value=100.0,
                confidence_interval=(50.0, 150.0),
                last_updated=datetime.now(),
                update_frequency='daily'
            ),
            'inventory_turnover_days': DynamicThreshold(
                metric_name='inventory_turnover_days',
                lower_bound=1.0,
                upper_bound=180.0,
                baseline_value=30.0,
                confidence_interval=(15.0, 60.0),
                last_updated=datetime.now(),
                update_frequency='weekly'
            )
        }
        
        self.thresholds = default_thresholds
    
    def get_threshold(self, metric_name: str) -> Optional[DynamicThreshold]:
        """
        Get threshold for a specific metric
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Dynamic threshold or None if not found
        """
        return self.thresholds.get(metric_name)
    
    def update_threshold(self, metric_name: str, threshold: DynamicThreshold) -> None:
        """
        Update threshold for a metric
        
        Args:
            metric_name: Name of metric
            threshold: New threshold
        """
        self.thresholds[metric_name] = threshold
        self._save_thresholds()
        self.logger.info(f"Updated threshold for {metric_name}")
    
    def adjust_threshold_based_on_history(self, metric_name: str, 
                                       historical_data: pd.Series,
                                       confidence_level: float = 0.95) -> DynamicThreshold:
        """
        Adjust threshold based on historical data
        
        Args:
            metric_name: Name of metric
            historical_data: Historical data series
            confidence_level: Confidence level for bounds (0.95 = 95%)
            
        Returns:
            Updated dynamic threshold
        """
        if historical_data.empty:
            raise ValueError("Historical data is empty")
        
        # Calculate statistics
        mean_value = float(historical_data.mean())
        std_value = float(historical_data.std())
        
        # Calculate confidence interval
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin_of_error = z_score * (std_value / np.sqrt(len(historical_data)))
        
        lower_bound = max(0, mean_value - margin_of_error)
        upper_bound = mean_value + margin_of_error
        confidence_interval = (lower_bound, upper_bound)
        
        # Create updated threshold
        updated_threshold = DynamicThreshold(
            metric_name=metric_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            baseline_value=mean_value,
            confidence_interval=confidence_interval,
            last_updated=datetime.now(),
            update_frequency=self.thresholds.get(metric_name, 
                                              DynamicThreshold('', 0, 0, 0, (0, 0), datetime.now(), 'daily')).update_frequency
        )
        
        # Update in manager
        self.update_threshold(metric_name, updated_threshold)
        
        return updated_threshold
    
    def apply_seasonal_adjustment(self, metric_name: str, 
                                seasonal_factors: Dict[str, float]) -> None:
        """
        Apply seasonal adjustment to threshold
        
        Args:
            metric_name: Name of metric
            seasonal_factors: Dictionary mapping time periods to adjustment factors
        """
        if metric_name not in self.thresholds:
            self.logger.warning(f"Threshold not found for {metric_name}")
            return
        
        threshold = self.thresholds[metric_name]
        threshold.seasonality_adjusted = True
        threshold.seasonal_factors = seasonal_factors
        threshold.last_updated = datetime.now()
        
        self.update_threshold(metric_name, threshold)
        self.logger.info(f"Applied seasonal adjustment to {metric_name}")
    
    def get_adjusted_threshold(self, metric_name: str, 
                             current_time: Optional[datetime] = None) -> Optional[DynamicThreshold]:
        """
        Get threshold adjusted for current conditions
        
        Args:
            metric_name: Name of metric
            current_time: Current time (defaults to now)
            
        Returns:
            Adjusted dynamic threshold or None if not found
        """
        threshold = self.get_threshold(metric_name)
        if not threshold:
            return None
        
        # Apply seasonal adjustments if applicable
        if threshold.seasonality_adjusted and current_time:
            adjusted_threshold = DynamicThreshold(**threshold.__dict__)
            
            # Apply seasonal factor based on current time
            seasonal_factor = 1.0
            if threshold.seasonal_factors:
                # Simple seasonal adjustment based on month
                month_key = current_time.strftime('%m')
                seasonal_factor = threshold.seasonal_factors.get(month_key, 1.0)
            
            # Adjust bounds
            adjusted_threshold.lower_bound *= seasonal_factor
            adjusted_threshold.upper_bound *= seasonal_factor
            adjusted_threshold.baseline_value *= seasonal_factor
            
            return adjusted_threshold
        
        return threshold
    
    def validate_thresholds(self) -> List[str]:
        """
        Validate all thresholds for consistency
        
        Returns:
            List of validation issues
        """
        issues = []
        
        for metric_name, threshold in self.thresholds.items():
            # Check bounds consistency
            if threshold.lower_bound >= threshold.upper_bound:
                issues.append(f"Invalid bounds for {metric_name}: lower ({threshold.lower_bound}) >= upper ({threshold.upper_bound})")
            
            # Check confidence interval
            ci_lower, ci_upper = threshold.confidence_interval
            if ci_lower >= ci_upper:
                issues.append(f"Invalid confidence interval for {metric_name}: lower ({ci_lower}) >= upper ({ci_upper})")
            
            # Check baseline within bounds
            if not (threshold.lower_bound <= threshold.baseline_value <= threshold.upper_bound):
                issues.append(f"Baseline value for {metric_name} outside bounds")
        
        return issues
    
    def get_thresholds_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of all thresholds
        
        Returns:
            Dictionary of threshold summaries
        """
        summary = {}
        for metric_name, threshold in self.thresholds.items():
            summary[metric_name] = {
                'lower_bound': threshold.lower_bound,
                'upper_bound': threshold.upper_bound,
                'baseline_value': threshold.baseline_value,
                'last_updated': threshold.last_updated.isoformat()
            }
        return summary


class AdaptiveThresholdEngine:
    """Engine for adaptive threshold management"""
    
    def __init__(self, threshold_manager: ThresholdManager):
        """
        Initialize adaptive threshold engine
        
        Args:
            threshold_manager: Threshold manager instance
        """
        self.threshold_manager = threshold_manager
        self.logger = logging.getLogger('AdaptiveThresholdEngine')
    
    def update_thresholds_from_data(self, data_dict: Dict[str, pd.Series]) -> Dict[str, DynamicThreshold]:
        """
        Update thresholds based on new data
        
        Args:
            data_dict: Dictionary mapping metric names to data series
            
        Returns:
            Dictionary of updated thresholds
        """
        updated_thresholds = {}
        
        for metric_name, data_series in data_dict.items():
            try:
                if not data_series.empty:
                    # Update threshold based on historical data
                    updated_threshold = self.threshold_manager.adjust_threshold_based_on_history(
                        metric_name, data_series
                    )
                    updated_thresholds[metric_name] = updated_threshold
                    self.logger.info(f"Updated threshold for {metric_name} based on {len(data_series)} data points")
                else:
                    self.logger.warning(f"No data available for {metric_name}")
                    
            except Exception as e:
                self.logger.error(f"Error updating threshold for {metric_name}: {str(e)}")
        
        return updated_thresholds
    
    def apply_business_rules(self, metric_name: str, 
                           business_constraints: Dict[str, Any]) -> None:
        """
        Apply business rules to threshold constraints
        
        Args:
            metric_name: Name of metric
            business_constraints: Business constraint definitions
        """
        threshold = self.threshold_manager.get_threshold(metric_name)
        if not threshold:
            self.logger.warning(f"Threshold not found for {metric_name}")
            return
        
        # Apply business constraints
        if 'min_value' in business_constraints:
            threshold.lower_bound = max(threshold.lower_bound, business_constraints['min_value'])
        
        if 'max_value' in business_constraints:
            threshold.upper_bound = min(threshold.upper_bound, business_constraints['max_value'])
        
        if 'hard_min' in business_constraints:
            threshold.lower_bound = business_constraints['hard_min']
        
        if 'hard_max' in business_constraints:
            threshold.upper_bound = business_constraints['hard_max']
        
        threshold.last_updated = datetime.now()
        self.threshold_manager.update_threshold(metric_name, threshold)
        
        self.logger.info(f"Applied business rules to {metric_name}")


# Example usage
if __name__ == "__main__":
    # Create threshold manager
    threshold_manager = ThresholdManager("config/test_thresholds.json")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    demand_data = pd.Series(np.random.normal(200, 50, 365), index=dates)
    
    # Add some outliers
    demand_data.iloc[10] = 400
    demand_data.iloc[100] = 50
    
    # Create adaptive engine
    adaptive_engine = AdaptiveThresholdEngine(threshold_manager)
    
    # Update threshold based on data
    data_dict = {'daily_demand': demand_data}
    updated_thresholds = adaptive_engine.update_thresholds_from_data(data_dict)
    
    print("Updated thresholds:")
    for metric_name, threshold in updated_thresholds.items():
        print(f"  {metric_name}: [{threshold.lower_bound:.2f}, {threshold.upper_bound:.2f}] (baseline: {threshold.baseline_value:.2f})")
    
    # Validate thresholds
    validation_issues = threshold_manager.validate_thresholds()
    if validation_issues:
        print("Validation issues:")
        for issue in validation_issues:
            print(f"  - {issue}")
    else:
        print("All thresholds are valid")
    
    # Get summary
    summary = threshold_manager.get_thresholds_summary()
    print(f"Threshold summary: {len(summary)} metrics")
</file>