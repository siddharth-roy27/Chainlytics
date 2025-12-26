"""
Training pipeline for anomaly detection models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from .detector import StatisticalAnomalyDetector, IsolationForestDetector, CompositeAnomalyDetector
from .thresholds import ThresholdManager, DynamicThreshold


class AnomalyDetectionTrainer:
    """Train anomaly detection models"""
    
    def __init__(self, model_save_path: str = "registry/models/anomaly"):
        """
        Initialize trainer
        
        Args:
            model_save_path: Path to save trained models
        """
        self.model_save_path = model_save_path
        self.threshold_manager = ThresholdManager()
        self.logger = logging.getLogger('AnomalyDetectionTrainer')
        os.makedirs(model_save_path, exist_ok=True)
    
    def prepare_training_data(self, data_sources: Dict[str, pd.DataFrame],
                            label_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare data for anomaly detection training
        
        Args:
            data_sources: Dictionary of data sources
            label_column: Column containing anomaly labels (if supervised)
            
        Returns:
            Prepared training data
        """
        prepared_data = {}
        
        for source_name, df in data_sources.items():
            if df.empty:
                continue
            
            # Handle missing values
            df_clean = df.dropna()
            
            # Convert to appropriate format
            if label_column and label_column in df_clean.columns:
                # Supervised case
                X = df_clean.drop(columns=[label_column])
                y = df_clean[label_column]
                prepared_data[source_name] = {'X': X, 'y': y}
            else:
                # Unsupervised case
                X = df_clean.select_dtypes(include=[np.number])
                prepared_data[source_name] = {'X': X}
        
        return prepared_data
    
    def train_statistical_detectors(self, training_data: Dict[str, Dict[str, pd.DataFrame]],
                                 threshold_config: Dict[str, Dict[str, float]]) -> List[StatisticalAnomalyDetector]:
        """
        Train statistical anomaly detectors
        
        Args:
            training_data: Prepared training data
            threshold_config: Configuration for statistical thresholds
            
        Returns:
            List of trained statistical detectors
        """
        detectors = []
        
        for source_name, data_dict in training_data.items():
            X = data_dict['X']
            
            # Create detector for each numeric column
            for column in X.select_dtypes(include=[np.number]).columns:
                metric_name = f"{source_name}_{column}"
                
                # Get threshold configuration
                config = threshold_config.get(metric_name, {})
                threshold_sigma = config.get('threshold_sigma', 3.0)
                window_size = config.get('window_size', 30)
                
                # Create detector
                detector = StatisticalAnomalyDetector(
                    name=f"stat_{metric_name}",
                    anomaly_type=self._infer_anomaly_type(column),
                    threshold_sigma=threshold_sigma,
                    window_size=window_size
                )
                
                detectors.append(detector)
                self.logger.info(f"Created statistical detector for {metric_name}")
        
        return detectors
    
    def train_isolation_forest_detectors(self, training_data: Dict[str, Dict[str, pd.DataFrame]],
                                      contamination_rates: Dict[str, float]) -> List[IsolationForestDetector]:
        """
        Train Isolation Forest detectors
        
        Args:
            training_data: Prepared training data
            contamination_rates: Expected contamination rates for each source
            
        Returns:
            List of trained Isolation Forest detectors
        """
        detectors = []
        
        for source_name, data_dict in training_data.items():
            X = data_dict['X']
            
            if X.empty:
                continue
            
            # Get contamination rate
            contamination = contamination_rates.get(source_name, 0.1)
            
            # Create detector
            detector = IsolationForestDetector(
                name=f"isoforest_{source_name}",
                anomaly_type=self._infer_anomaly_type(source_name),
                contamination=contamination,
                random_state=42
            )
            
            # Train detector
            try:
                detector.detect(X)
                detectors.append(detector)
                self.logger.info(f"Trained Isolation Forest detector for {source_name}")
            except Exception as e:
                self.logger.error(f"Error training Isolation Forest for {source_name}: {str(e)}")
        
        return detectors
    
    def _infer_anomaly_type(self, metric_name: str) -> Any:
        """
        Infer anomaly type from metric name
        
        Args:
            metric_name: Name of metric
            
        Returns:
            AnomalyType enum value
        """
        from .detector import AnomalyType
        
        metric_name_lower = metric_name.lower()
        
        if 'demand' in metric_name_lower:
            return AnomalyType.DEMAND_SPIKE
        elif 'cost' in metric_name_lower or 'price' in metric_name_lower:
            return AnomalyType.COST_OUTLIER
        elif 'delay' in metric_name_lower or 'time' in metric_name_lower:
            return AnomalyType.DELAY_ANOMALY
        elif 'inventory' in metric_name_lower:
            return AnomalyType.INVENTORY_DEVIATION
        elif 'route' in metric_name_lower:
            return AnomalyType.ROUTE_EFFICIENCY
        elif 'capacity' in metric_name_lower:
            return AnomalyType.CAPACITY_UTILIZATION
        else:
            return AnomalyType.DEMAND_SPIKE  # Default
    
    def evaluate_detectors(self, detectors: List[Any], 
                         test_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained detectors
        
        Args:
            detectors: List of trained detectors
            test_data: Test data for evaluation
            
        Returns:
            Evaluation metrics for each detector
        """
        evaluation_results = {}
        
        for detector in detectors:
            try:
                # Get test data for this detector
                data_key = detector.name.replace('stat_', '').replace('isoforest_', '')
                test_df = test_data.get(data_key, pd.DataFrame())
                
                if test_df.empty:
                    # Try partial match
                    matching_keys = [k for k in test_data.keys() if data_key.split('_')[0] in k]
                    if matching_keys:
                        test_df = test_data[matching_keys[0]]
                
                if test_df.empty:
                    evaluation_results[detector.name] = {'error': 'No test data'}
                    continue
                
                # Detect anomalies
                anomalies = detector.detect(test_df)
                
                # Calculate metrics (simplified for unsupervised case)
                metrics = {
                    'anomalies_detected': len(anomalies),
                    'anomaly_rate': len(anomalies) / len(test_df) if len(test_df) > 0 else 0,
                    'avg_confidence': np.mean([a.confidence for a in anomalies]) if anomalies else 0
                }
                
                evaluation_results[detector.name] = metrics
                self.logger.info(f"Evaluation for {detector.name}: {metrics}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {detector.name}: {str(e)}")
                evaluation_results[detector.name] = {'error': str(e)}
        
        return evaluation_results
    
    def train_composite_model(self, training_data: Dict[str, Dict[str, pd.DataFrame]],
                           threshold_config: Dict[str, Dict[str, float]],
                           contamination_rates: Dict[str, float]) -> CompositeAnomalyDetector:
        """
        Train composite anomaly detection model
        
        Args:
            training_data: Prepared training data
            threshold_config: Statistical threshold configuration
            contamination_rates: Contamination rates for Isolation Forest
            
        Returns:
            Trained composite detector
        """
        # Create composite detector
        composite_detector = CompositeAnomalyDetector()
        
        # Train statistical detectors
        stat_detectors = self.train_statistical_detectors(training_data, threshold_config)
        for detector in stat_detectors:
            composite_detector.add_detector(detector)
        
        # Train Isolation Forest detectors
        iso_detectors = self.train_isolation_forest_detectors(training_data, contamination_rates)
        for detector in iso_detectors:
            composite_detector.add_detector(detector)
        
        return composite_detector
    
    def save_model(self, model: Any, model_name: str) -> str:
        """
        Save trained model
        
        Args:
            model: Trained model
            model_name: Name for the model
            
        Returns:
            Path to saved model
        """
        try:
            # Create model directory
            model_dir = os.path.join(self.model_save_path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'saved_at': datetime.now().isoformat(),
                'model_type': type(model).__name__
            }
            
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved model {model_name} to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
            raise
    
    def train_pipeline(self, data_sources: Dict[str, pd.DataFrame],
                     test_size: float = 0.2,
                     threshold_config: Optional[Dict[str, Dict[str, float]]] = None,
                     contamination_rates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Complete training pipeline
        
        Args:
            data_sources: Dictionary of training data sources
            test_size: Proportion of data to use for testing
            threshold_config: Statistical threshold configuration
            contamination_rates: Contamination rates for Isolation Forest
            
        Returns:
            Training results
        """
        if threshold_config is None:
            threshold_config = {}
        
        if contamination_rates is None:
            contamination_rates = {}
        
        start_time = datetime.now()
        self.logger.info("Starting anomaly detection training pipeline")
        
        try:
            # Prepare data
            self.logger.info("Preparing training data...")
            prepared_data = self.prepare_training_data(data_sources)
            
            # Split data
            train_data = {}
            test_data = {}
            
            for source_name, data_dict in prepared_data.items():
                X = data_dict['X']
                if len(X) > 10:  # Need minimum data
                    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
                    train_data[source_name] = {'X': X_train}
                    test_data[source_name] = X_test
                else:
                    train_data[source_name] = data_dict
                    test_data[source_name] = X
            
            # Train composite model
            self.logger.info("Training composite model...")
            composite_model = self.train_composite_model(
                train_data, threshold_config, contamination_rates
            )
            
            # Evaluate model
            self.logger.info("Evaluating model...")
            evaluation_results = self.evaluate_detectors(
                composite_model.detectors, test_data
            )
            
            # Save model
            model_name = f"composite_anomaly_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.save_model(composite_model, model_name)
            
            # Update thresholds based on training data
            self.logger.info("Updating dynamic thresholds...")
            self._update_thresholds_from_training(train_data)
            
            # Compile results
            training_duration = (datetime.now() - start_time).total_seconds()
            
            results = {
                'model_name': model_name,
                'model_path': model_path,
                'training_duration_seconds': training_duration,
                'num_detectors': len(composite_model.detectors),
                'detector_summary': composite_model.get_detector_summary(),
                'evaluation_results': evaluation_results,
                'training_completed': datetime.now().isoformat()
            }
            
            self.logger.info(f"Training completed in {training_duration:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            raise
    
    def _update_thresholds_from_training(self, training_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Update dynamic thresholds based on training data
        
        Args:
            training_data: Training data used for threshold updates
        """
        try:
            data_dict = {}
            for source_name, data_dict_item in training_data.items():
                X = data_dict_item['X']
                # Combine all numeric columns into series for threshold adjustment
                for column in X.select_dtypes(include=[np.number]).columns:
                    metric_name = f"{source_name}_{column}"
                    data_dict[metric_name] = X[column]
            
            # Update thresholds
            from .thresholds import AdaptiveThresholdEngine
            adaptive_engine = AdaptiveThresholdEngine(self.threshold_manager)
            adaptive_engine.update_thresholds_from_data(data_dict)
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {str(e)}")


def load_training_data(data_directory: str) -> Dict[str, pd.DataFrame]:
    """
    Load training data from directory
    
    Args:
        data_directory: Directory containing training data files
        
    Returns:
        Dictionary of dataframes
    """
    data_sources = {}
    
    if not os.path.exists(data_directory):
        return data_sources
    
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            try:
                file_path = os.path.join(data_directory, filename)
                df = pd.read_csv(file_path)
                
                # Use filename (without extension) as source name
                source_name = os.path.splitext(filename)[0]
                data_sources[source_name] = df
                
                logging.info(f"Loaded {len(df)} rows from {filename}")
                
            except Exception as e:
                logging.error(f"Error loading {filename}: {str(e)}")
    
    return data_sources


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample training data
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    # Normal demand data with some anomalies
    demand_data = np.random.poisson(100, 1000)
    # Insert some anomalies
    anomaly_indices = np.random.choice(1000, 50, replace=False)
    demand_data[anomaly_indices] = np.random.poisson(300, 50)  # High demand anomalies
    
    demand_df = pd.DataFrame({
        'date': dates,
        'demand': demand_data
    })
    
    # Cost data
    cost_data = np.random.normal(100, 20, 1000)
    # Insert some anomalies
    cost_data[anomaly_indices] = np.random.normal(300, 50, 50)  # High cost anomalies
    
    cost_df = pd.DataFrame({
        'date': dates,
        'cost': cost_data
    })
    
    # Create data sources
    data_sources = {
        'daily_demand': demand_df,
        'shipment_cost': cost_df
    }
    
    # Configuration
    threshold_config = {
        'daily_demand_demand': {'threshold_sigma': 3.0, 'window_size': 30},
        'shipment_cost_cost': {'threshold_sigma': 2.5, 'window_size': 30}
    }
    
    contamination_rates = {
        'daily_demand': 0.05,
        'shipment_cost': 0.05
    }
    
    # Create trainer
    trainer = AnomalyDetectionTrainer("registry/models/test_anomaly")
    
    # Run training pipeline
    try:
        results = trainer.train_pipeline(
            data_sources=data_sources,
            threshold_config=threshold_config,
            contamination_rates=contamination_rates
        )
        
        print("Training completed successfully!")
        print(f"Model saved to: {results['model_path']}")
        print(f"Number of detectors: {results['num_detectors']}")
        print(f"Training duration: {results['training_duration_seconds']:.2f} seconds")
        
        print("\nDetector summary:")
        for name, detector_type in results['detector_summary'].items():
            print(f"  - {name}: {detector_type}")
        
        print("\nEvaluation results:")
        for detector_name, metrics in results['evaluation_results'].items():
            if 'error' not in metrics:
                print(f"  - {detector_name}: {metrics}")
            else:
                print(f"  - {detector_name}: Error - {metrics['error']}")
                
    except Exception as e:
        print(f"Training failed: {str(e)}")
</file>