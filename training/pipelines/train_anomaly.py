"""
Anomaly Detection Training Pipeline
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.anomaly_service.detector import AnomalyDetector
from services.anomaly_service.thresholds import AnomalyThresholds
from data.loaders.anomaly_loader import AnomalyDataLoader
from registry.models.anomaly_registry import AnomalyModelRegistry

logger = logging.getLogger(__name__)

class AnomalyTrainingPipeline:
    """Training pipeline for anomaly detection models."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.data_loader = AnomalyDataLoader()
        self.registry = AnomalyModelRegistry()
        self.scaler = StandardScaler()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'training': {
                    'contamination': 0.1,
                    'random_state': 42,
                    'n_estimators': 100
                },
                'features': {
                    'demand_features': ['order_volume', 'order_value', 'customer_count'],
                    'cost_features': ['transport_cost', 'holding_cost', 'labor_cost'],
                    'delay_features': ['delivery_time', 'pickup_delay', 'route_deviation']
                }
            }
    
    def run_training(self,
                     data_start_date: str = None,
                     data_end_date: str = None,
                     version: str = None) -> Dict[str, Any]:
        """
        Run the complete anomaly detection training pipeline.
        
        Args:
            data_start_date: Start date for training data
            data_end_date: End date for training data
            version: Model version identifier
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting anomaly detection training pipeline")
        
        # Load and prepare data
        logger.info("Loading training data")
        raw_data = self.data_loader.load_historical_data(
            start_date=data_start_date,
            end_date=data_end_date
        )
        
        # Prepare features for different anomaly types
        logger.info("Preparing features for anomaly detection")
        demand_features = self._prepare_demand_features(raw_data)
        cost_features = self._prepare_cost_features(raw_data)
        delay_features = self._prepare_delay_features(raw_data)
        
        # Train models for each anomaly type
        logger.info("Training anomaly detection models")
        demand_model = self._train_isolation_forest(demand_features, "demand")
        cost_model = self._train_isolation_forest(cost_features, "cost")
        delay_model = self._train_isolation_forest(delay_features, "delay")
        
        # Evaluate models
        logger.info("Evaluating model performance")
        demand_metrics = self._evaluate_anomaly_model(demand_model, demand_features)
        cost_metrics = self._evaluate_anomaly_model(cost_model, cost_features)
        delay_metrics = self._evaluate_anomaly_model(delay_model, delay_features)
        
        # Create version if not provided
        if not version:
            version = f"anomaly_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Register models
        logger.info(f"Registering model version: {version}")
        models = {
            'demand': demand_model,
            'cost': cost_model,
            'delay': delay_model
        }
        
        model_metadata = {
            'version': version,
            'training_date': datetime.now().isoformat(),
            'features': self.config['features'],
            'metrics': {
                'demand': demand_metrics,
                'cost': cost_metrics,
                'delay': delay_metrics
            },
            'data_range': {
                'start_date': data_start_date,
                'end_date': data_end_date
            }
        }
        
        self.registry.register_models(models, model_metadata, version)
        
        results = {
            'version': version,
            'metrics': model_metadata['metrics'],
            'training_completed': datetime.now().isoformat()
        }
        
        logger.info("Anomaly detection training pipeline completed successfully")
        return results
    
    def _prepare_demand_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for demand anomaly detection."""
        feature_cols = self.config['features']['demand_features']
        features = data[feature_cols].values
        return self.scaler.fit_transform(features)
    
    def _prepare_cost_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for cost anomaly detection."""
        feature_cols = self.config['features']['cost_features']
        features = data[feature_cols].values
        return self.scaler.fit_transform(features)
    
    def _prepare_delay_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for delay anomaly detection."""
        feature_cols = self.config['features']['delay_features']
        features = data[feature_cols].values
        return self.scaler.fit_transform(features)
    
    def _train_isolation_forest(self, features: np.ndarray, model_type: str) -> IsolationForest:
        """Train Isolation Forest model."""
        model = IsolationForest(
            contamination=self.config['training']['contamination'],
            random_state=self.config['training']['random_state'],
            n_estimators=self.config['training']['n_estimators']
        )
        model.fit(features)
        return model
    
    def _evaluate_anomaly_model(self, model: IsolationForest, features: np.ndarray) -> Dict[str, float]:
        """Evaluate anomaly detection model."""
        # Predict anomalies
        predictions = model.predict(features)
        anomaly_scores = model.decision_function(features)
        
        # Calculate metrics
        anomaly_count = np.sum(predictions == -1)
        anomaly_rate = anomaly_count / len(predictions)
        
        # Average anomaly score
        avg_anomaly_score = np.mean(anomaly_scores)
        
        return {
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'avg_anomaly_score': float(avg_anomaly_score)
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run training pipeline
    pipeline = AnomalyTrainingPipeline()
    results = pipeline.run_training(
        data_start_date="2023-01-01",
        data_end_date="2023-12-31"
    )
    
    print(f"Training completed with results: {results}")