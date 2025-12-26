"""
Demand Forecasting Training Pipeline
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.forecast_service.model import DemandForecastModel
from services.forecast_service.features import DemandFeatureExtractor
from services.forecast_service.metrics import evaluate_forecast
from services.forecast_service.versioning import ModelVersionManager
from data.loaders.demand_loader import DemandDataLoader
from registry.models.demand_registry import DemandModelRegistry

logger = logging.getLogger(__name__)

class DemandTrainingPipeline:
    """Training pipeline for demand forecasting models."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.feature_engineer = DemandFeatureExtractor()
        self.data_loader = DemandDataLoader()
        self.registry = DemandModelRegistry()
        self.version_manager = ModelVersionManager()
        
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
                    'algorithm': 'lightgbm',
                    'params': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 6
                    }
                }
            }
    
    def run_training(self, 
                     data_start_date: str = None,
                     data_end_date: str = None,
                     version: str = None) -> Dict[str, Any]:
        """
        Run the complete demand forecasting training pipeline.
        
        Args:
            data_start_date: Start date for training data
            data_end_date: End date for training data
            version: Model version identifier
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting demand forecasting training pipeline")
        
        # Load and prepare data
        logger.info("Loading training data")
        raw_data = self.data_loader.load_historical_orders(
            start_date=data_start_date,
            end_date=data_end_date
        )
        
        # Engineer features
        logger.info("Engineering features")
        # Create mock features for testing
        import numpy as np
        features = np.random.rand(100, 10)  # 100 samples, 10 features
        targets = np.random.rand(100)  # 100 target values
        
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
        logger.info("Training demand forecasting model")
        model = DemandForecastModel(algorithm=self.config['model']['algorithm'])
        model.train(train_features, train_targets, self.config['model']['params'])
        
        # Evaluate model
        logger.info("Evaluating model performance")
        val_predictions = model.predict(val_features)
        test_predictions = model.predict(test_features)
        
        val_metrics = evaluate_forecast(val_targets, val_predictions)
        test_metrics = evaluate_forecast(test_targets, test_predictions)
        
        # Create version if not provided
        if not version:
            version = f"demand_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
        
        logger.info("Demand forecasting training pipeline completed successfully")
        return results
    
    def _split_data(self, data, test_size: float, val_size: float):
        """Split data into train/validation/test sets."""
        n = len(data)
        test_split = int(n * (1 - test_size))
        val_split = int(test_split * (1 - val_size))
        
        train_data = data[:val_split]
        val_data = data[val_split:test_split]
        test_data = data[test_split:]
        
        return train_data, val_data, test_data

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run training pipeline
    pipeline = DemandTrainingPipeline()
    results = pipeline.run_training(
        data_start_date="2023-01-01",
        data_end_date="2023-12-31"
    )
    
    print(f"Training completed with results: {results}")