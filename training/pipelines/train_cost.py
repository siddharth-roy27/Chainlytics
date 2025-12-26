"""
Cost Model Training Pipeline
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
import yaml
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.cost_service.cost_model import CostModel
from services.cost_service.marginal_cost import MarginalCostCalculator
from data.loaders.cost_loader import CostDataLoader
from registry.models.cost_registry import CostModelRegistry

logger = logging.getLogger(__name__)

class CostTrainingPipeline:
    """Training pipeline for logistics cost models."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.data_loader = CostDataLoader()
        self.registry = CostModelRegistry()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'training': {
                    'test_size': 0.2,
                    'validation_size': 0.1,
                    'random_state': 42
                },
                'model': {
                    'algorithm': 'linear_regression',
                    'params': {
                        'fit_intercept': True,
                        'normalize': False
                    }
                }
            }
    
    def run_training(self,
                     data_start_date: str = None,
                     data_end_date: str = None,
                     version: str = None) -> Dict[str, Any]:
        """
        Run the complete cost model training pipeline.
        
        Args:
            data_start_date: Start date for training data
            data_end_date: End date for training data
            version: Model version identifier
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting cost model training pipeline")
        
        # Load and prepare data
        logger.info("Loading training data")
        raw_data = self.data_loader.load_historical_costs(
            start_date=data_start_date,
            end_date=data_end_date
        )
        
        # Prepare features and targets
        logger.info("Preparing features and targets")
        features, targets = self._prepare_features_targets(raw_data)
        
        # Split data
        test_size = self.config['training']['test_size']
        val_size = self.config['training']['validation_size']
        
        train_features, val_features, test_features = self._split_data(
            features, test_size, val_size
        )
        train_targets, val_targets, test_targets = self._split_data(
            targets, test_size, val_size
        )
        
        # Train model
        logger.info("Training cost model")
        model = CostModel(algorithm=self.config['model']['algorithm'])
        model.train(train_features, train_targets, self.config['model']['params'])
        
        # Evaluate model
        logger.info("Evaluating model performance")
        val_predictions = model.predict(val_features)
        test_predictions = model.predict(test_features)
        
        val_metrics = self._evaluate_cost_model(val_targets, val_predictions)
        test_metrics = self._evaluate_cost_model(test_targets, test_predictions)
        
        # Create version if not provided
        if not version:
            version = f"cost_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Register model
        logger.info(f"Registering model version: {version}")
        model_metadata = {
            'version': version,
            'algorithm': self.config['model']['algorithm'],
            'params': self.config['model']['params'],
            'training_date': datetime.now().isoformat(),
            'metrics': {
                'validation': val_metrics,
                'test': test_metrics
            },
            'data_range': {
                'start_date': data_start_date,
                'end_date': data_end_date
            }
        }
        
        self.registry.register_model(model, model_metadata, version)
        
        results = {
            'version': version,
            'metrics': {
                'validation': val_metrics,
                'test': test_metrics
            },
            'training_completed': datetime.now().isoformat()
        }
        
        logger.info("Cost model training pipeline completed successfully")
        return results
    
    def _prepare_features_targets(self, data: pd.DataFrame):
        """Prepare features and targets from raw data."""
        # Extract features (distance, weight, time, etc.)
        feature_columns = ['distance_km', 'weight_kg', 'time_hours', 'fuel_price', 'traffic_index']
        features = data[feature_columns].values
        
        # Target is total cost
        targets = data['total_cost'].values
        
        return features, targets
    
    def _split_data(self, data, test_size: float, val_size: float):
        """Split data into train/validation/test sets."""
        n = len(data)
        test_split = int(n * (1 - test_size))
        val_split = int(test_split * (1 - val_size))
        
        train_data = data[:val_split]
        val_data = data[val_split:test_split]
        test_data = data[test_split:]
        
        return train_data, val_data, test_data
    
    def _evaluate_cost_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate cost model performance."""
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run training pipeline
    pipeline = CostTrainingPipeline()
    results = pipeline.run_training(
        data_start_date="2023-01-01",
        data_end_date="2023-12-31"
    )
    
    print(f"Training completed with results: {results}")