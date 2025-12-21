"""
Evaluation metrics for demand forecasting
"""

import numpy as np
from typing import Dict, Any
from .model import DemandForecastModel


class ForecastEvaluator:
    """Evaluate forecast model performance"""
    
    def __init__(self):
        pass
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_coverage(self, y_true: np.ndarray, lower_bound: np.ndarray, 
                          upper_bound: np.ndarray) -> float:
        """Calculate coverage probability for prediction intervals"""
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        return np.mean(within_interval)
    
    def evaluate_model(self, model: DemandForecastModel, X: np.ndarray, 
                      y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        predictions = []
        p50_preds = []
        p90_preds = []
        p99_preds = []
        
        # Make predictions for each sample
        for i in range(len(X)):
            p50, p90, p99 = model.predict(X[i:i+1])
            p50_preds.append(p50)
            p90_preds.append(p90)
            p99_preds.append(p99)
            predictions.append(p50)  # Use median as point forecast
        
        # Convert to numpy arrays
        y_pred = np.array(predictions)
        p50_preds = np.array(p50_preds)
        p90_preds = np.array(p90_preds)
        p99_preds = np.array(p99_preds)
        
        # Calculate metrics
        metrics = {
            'mae': self.calculate_mae(y_true, y_pred),
            'rmse': self.calculate_rmse(y_true, y_pred),
            'mape': self.calculate_mape(y_true, y_pred),
            'p50_coverage': self.calculate_coverage(y_true, p50_preds * 0.8, p50_preds * 1.2),
            'p90_coverage': self.calculate_coverage(y_true, p50_preds * 0.9, p90_preds * 1.1),
            'p99_coverage': self.calculate_coverage(y_true, p50_preds * 0.95, p99_preds * 1.05)
        }
        
        return metrics