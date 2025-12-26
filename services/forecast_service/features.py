"""
Feature engineering for demand forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class DemandFeatureExtractor:
    """Extract features for demand forecasting"""
    
    def __init__(self):
        self.feature_columns = []
        
    def extract_temporal_features(self, date: datetime) -> Dict[str, float]:
        """Extract temporal features from date"""
        features = {
            'hour': date.hour,
            'day_of_week': date.weekday(),
            'day_of_month': date.day,
            'month': date.month,
            'is_weekend': 1 if date.weekday() >= 5 else 0,
        }
        return features
    
    def extract_lag_features(self, historical_data: pd.DataFrame, 
                           lag_periods: List[int] = [1, 3, 7, 14]) -> Dict[str, float]:
        """Extract lag features from historical demand data"""
        features = {}
        for period in lag_periods:
            col_name = f'demand_lag_{period}'
            if len(historical_data) >= period:
                features[col_name] = historical_data['demand'].iloc[-period]
            else:
                features[col_name] = 0.0
        return features
    
    def extract_rolling_features(self, historical_data: pd.DataFrame,
                               windows: List[int] = [3, 7, 14]) -> Dict[str, float]:
        """Extract rolling statistics features"""
        features = {}
        for window in windows:
            if len(historical_data) >= window:
                recent_data = historical_data['demand'].tail(window)
                features[f'demand_mean_{window}'] = recent_data.mean()
                features[f'demand_std_{window}'] = recent_data.std()
                features[f'demand_max_{window}'] = recent_data.max()
            else:
                features[f'demand_mean_{window}'] = 0.0
                features[f'demand_std_{window}'] = 0.0
                features[f'demand_max_{window}'] = 0.0
        return features
    
    def create_feature_vector(self, date: datetime, 
                            historical_data: pd.DataFrame,
                            warehouse_id: str = None) -> Dict[str, float]:
        """Create complete feature vector for demand prediction"""
        features = {}
        
        # Temporal features
        temporal_features = self.extract_temporal_features(date)
        features.update(temporal_features)
        
        # Lag features
        lag_features = self.extract_lag_features(historical_data)
        features.update(lag_features)
        
        # Rolling features
        rolling_features = self.extract_rolling_features(historical_data)
        features.update(rolling_features)
        
        # Warehouse identifier (if provided)
        if warehouse_id:
            features['warehouse_id'] = hash(warehouse_id) % 1000  # Simple encoding
            
        return features
    
    def prepare_training_data(self, historical_orders: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical orders"""
        # Assuming historical_orders has columns: timestamp, warehouse_id, demand
        features_list = []
        targets = []
        
        # Sort by timestamp
        historical_orders = historical_orders.sort_values('timestamp')
        
        # For each order, create features and target
        for i in range(len(historical_orders)):
            current_row = historical_orders.iloc[i]
            
            # Get historical data up to current point
            historical_subset = historical_orders.iloc[:i+1]
            
            # Create feature vector
            features = self.create_feature_vector(
                date=current_row['timestamp'],
                historical_data=historical_subset,
                warehouse_id=current_row.get('warehouse_id', None)
            )
            
            features_list.append(list(features.values()))
            targets.append(current_row['demand'])
            
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(targets)
        
        return X, y