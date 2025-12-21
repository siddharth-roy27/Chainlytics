"""
Inference module for demand forecasting
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any
from datetime import datetime
from .model import LightGBMDemandModel, ForecastOutput
from .features import DemandFeatureExtractor


class ForecastInferenceEngine:
    """Inference engine for demand forecasting"""
    
    def __init__(self, model_path: str):
        """
        Initialize inference engine with trained model
        
        Args:
            model_path: Path to trained model
        """
        self.model = LightGBMDemandModel()
        self.model.load_model(model_path)
        self.feature_extractor = DemandFeatureExtractor()
    
    def predict_demand(self, warehouse_id: str, time_window: str, 
                      historical_data: pd.DataFrame) -> ForecastOutput:
        """
        Predict demand for a specific warehouse and time window
        
        Args:
            warehouse_id: ID of the warehouse
            time_window: Time window for prediction
            historical_data: Historical demand data
            
        Returns:
            ForecastOutput with P50, P90, P99 predictions
        """
        # Create feature vector for prediction
        # For simplicity, we'll use current time
        current_time = datetime.now()
        features_dict = self.feature_extractor.create_feature_vector(
            date=current_time,
            historical_data=historical_data,
            warehouse_id=warehouse_id
        )
        
        # Convert to numpy array
        features = np.array(list(features_dict.values())).reshape(1, -1)
        
        # Make prediction
        p50, p90, p99 = self.model.predict(features)
        
        # Create standardized output
        forecast_output = ForecastOutput(
            warehouse_id=warehouse_id,
            time_window=time_window,
            p50=p50,
            p90=p90,
            p99=p99
        )
        
        return forecast_output


def load_latest_model():
    """Load the latest trained model from registry"""
    # In practice, this would check model registry for latest version
    # For now, we'll assume v1.0 exists
    model_path = "registry/models/demand/v1.0/model.pkl"
    return ForecastInferenceEngine(model_path)


if __name__ == "__main__":
    # Example usage
    # This would typically be called by the decision orchestrator
    
    # Load model
    engine = load_latest_model()
    
    # Create mock historical data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    mock_data = pd.DataFrame({
        'timestamp': dates,
        'warehouse_id': 'WH001',
        'demand': np.random.poisson(100, len(dates))
    })
    
    # Make prediction
    forecast = engine.predict_demand(
        warehouse_id="WH001",
        time_window="2024-01-01_2024-01-07",
        historical_data=mock_data
    )
    
    print(f"Demand Forecast:")
    print(f"Warehouse: {forecast.warehouse_id}")
    print(f"Time Window: {forecast.time_window}")
    print(f"P50: {forecast.p50:.2f}")
    print(f"P90: {forecast.p90:.2f}")
    print(f"P99: {forecast.p99:.2f}")