"""
Training pipeline for demand forecasting model
"""

import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any
from .model import LightGBMDemandModel
from .features import DemandFeatureExtractor
from .metrics import ForecastEvaluator


def load_historical_data(data_path: str) -> pd.DataFrame:
    """Load historical orders data for training"""
    # This would connect to your data store in practice
    # For now, we'll create mock data or load from file
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # Create mock data for demonstration
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        warehouses = ['WH001', 'WH002', 'WH003']
        
        data = []
        for date in dates:
            for warehouse in warehouses:
                # Simulate some seasonality and trend
                base_demand = 100 + np.sin(2 * np.pi * date.dayofyear / 365) * 20
                noise = np.random.normal(0, 10)
                demand = max(0, base_demand + noise)
                
                data.append({
                    'timestamp': date,
                    'warehouse_id': warehouse,
                    'demand': demand
                })
        
        return pd.DataFrame(data)


def train_forecast_model(config: Dict[str, Any]) -> str:
    """
    Train the demand forecasting model
    
    Args:
        config: Training configuration
        
    Returns:
        Path to saved model
    """
    # Load data
    data_path = config.get('data_path', 'data/historical_orders.csv')
    historical_data = load_historical_data(data_path)
    
    # Extract features
    feature_extractor = DemandFeatureExtractor()
    X, y = feature_extractor.prepare_training_data(historical_data)
    
    # Train model
    model = LightGBMDemandModel()
    model.train(X, y)
    
    # Evaluate model
    evaluator = ForecastEvaluator()
    metrics = evaluator.evaluate_model(model, X, y)
    print(f"Model Performance: {metrics}")
    
    # Save model
    model_version = config.get('version', 'v1.0')
    model_path = f"registry/models/demand/{model_version}/model.pkl"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    
    # Save training config
    config_path = f"registry/models/demand/{model_version}/config.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    # Example training configuration
    config = {
        'data_path': 'data/historical_orders.csv',
        'version': 'v1.0',
        'model_type': 'lightgbm',
        'training_params': {
            'n_estimators': 100,
            'random_state': 42
        }
    }
    
    # Train model
    model_path = train_forecast_model(config)
    print(f"Training completed. Model saved at: {model_path}")