"""
Demand Forecasting Model Implementation
Supports LightGBM (MVP) and TFT (Advanced)
"""

import numpy as np
from typing import Dict, Tuple, Any
import lightgbm as lgb
from dataclasses import dataclass


@dataclass
class ForecastOutput:
    """Standardized output format for demand forecasts"""
    warehouse_id: str
    time_window: str
    p50: float
    p90: float
    p99: float


class DemandForecastModel:
    """Base class for demand forecasting models"""
    
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train the forecasting model"""
        raise NotImplementedError("Subclasses must implement train method")
        
    def predict(self, features: np.ndarray) -> Tuple[float, float, float]:
        """Predict demand with confidence intervals (P50, P90, P99)"""
        raise NotImplementedError("Subclasses must implement predict method")
        
    def save_model(self, path: str) -> None:
        """Save trained model to disk"""
        raise NotImplementedError("Subclasses must implement save_model method")
        
    def load_model(self, path: str) -> None:
        """Load trained model from disk"""
        raise NotImplementedError("Subclasses must implement load_model method")


class LightGBMDemandModel(DemandForecastModel):
    """LightGBM implementation for demand forecasting (MVP)"""
    
    def __init__(self):
        super().__init__("lightgbm")
        self.model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.5,  # For median prediction
            n_estimators=100,
            random_state=42
        )
        
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train LightGBM model on historical orders data"""
        self.model.fit(features, targets)
        self.is_trained = True
        
    def predict(self, features: np.ndarray) -> Tuple[float, float, float]:
        """Predict demand with confidence intervals"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # For simplicity, using separate models for different quantiles
        # In practice, would use quantile regression
        p50_pred = self.model.predict(features)
        
        # Mock confidence intervals (would use actual quantile models)
        p90_pred = p50_pred * 1.2
        p99_pred = p50_pred * 1.5
        
        return float(p50_pred[0]), float(p90_pred[0]), float(p99_pred[0])
        
    def save_model(self, path: str) -> None:
        """Save trained model to disk"""
        import joblib
        joblib.dump(self.model, path)
        
    def load_model(self, path: str) -> None:
        """Load trained model from disk"""
        import joblib
        self.model = joblib.load(path)
        self.is_trained = True