"""
Model versioning for demand forecasting
"""

import os
import yaml
from typing import Dict, Any, List
from datetime import datetime


class ModelVersionManager:
    """Manage model versions and metadata"""
    
    def __init__(self, model_registry_path: str = "registry/models/demand"):
        self.registry_path = model_registry_path
        self.metadata_file = os.path.join(self.registry_path, "metadata.yaml")
        
    def register_model(self, version: str, metrics: Dict[str, float], 
                      config: Dict[str, Any]) -> None:
        """
        Register a new model version with metrics and config
        
        Args:
            version: Model version identifier
            metrics: Evaluation metrics
            config: Training configuration
        """
        # Create version directory
        version_path = os.path.join(self.registry_path, version)
        os.makedirs(version_path, exist_ok=True)
        
        # Save metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'config': config
        }
        
        metadata_path = os.path.join(version_path, "metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        # Update registry metadata
        self._update_registry_metadata(version, metadata)
    
    def _update_registry_metadata(self, version: str, metadata: Dict[str, Any]) -> None:
        """Update the main registry metadata file"""
        # Load existing metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                registry_metadata = yaml.safe_load(f) or {}
        else:
            registry_metadata = {'versions': {}}
        
        # Add new version
        registry_metadata['versions'][version] = metadata
        
        # Save updated metadata
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            yaml.dump(registry_metadata, f)
    
    def get_latest_version(self) -> str:
        """Get the latest model version"""
        if not os.path.exists(self.metadata_file):
            return None
            
        with open(self.metadata_file, 'r') as f:
            registry_metadata = yaml.safe_load(f) or {}
        
        versions = registry_metadata.get('versions', {})
        if not versions:
            return None
            
        # Return the most recently created version
        sorted_versions = sorted(
            versions.items(), 
            key=lambda x: x[1].get('created_at', ''), 
            reverse=True
        )
        
        return sorted_versions[0][0] if sorted_versions else None
    
    def get_version_metrics(self, version: str) -> Dict[str, float]:
        """Get metrics for a specific model version"""
        version_path = os.path.join(self.registry_path, version)
        metadata_path = os.path.join(version_path, "metadata.yaml")
        
        if not os.path.exists(metadata_path):
            return {}
            
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
            
        return metadata.get('metrics', {})
    
    def list_versions(self) -> List[str]:
        """List all registered model versions"""
        if not os.path.exists(self.metadata_file):
            return []
            
        with open(self.metadata_file, 'r') as f:
            registry_metadata = yaml.safe_load(f) or {}
        
        versions = registry_metadata.get('versions', {})
        return list(versions.keys())
    
    def is_version_promotable(self, version: str, 
                            threshold_metrics: Dict[str, float] = None) -> bool:
        """
        Check if a model version meets promotion criteria
        
        Args:
            version: Model version to check
            threshold_metrics: Minimum required metrics values
            
        Returns:
            True if model meets promotion criteria
        """
        metrics = self.get_version_metrics(version)
        
        # Check if model has metrics
        if not metrics:
            return False
            
        # If no thresholds specified, model is promotable
        if not threshold_metrics:
            return True
            
        # Check against threshold metrics
        for metric_name, threshold_value in threshold_metrics.items():
            if metric_name not in metrics:
                return False
            if metrics[metric_name] > threshold_value:  # Lower is better for error metrics
                return False
                
        return True


# Example usage
if __name__ == "__main__":
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Example metrics from training
    example_metrics = {
        'mae': 15.2,
        'rmse': 22.1,
        'mape': 8.5,
        'p50_coverage': 0.85,
        'p90_coverage': 0.92
    }
    
    # Example config
    example_config = {
        'model_type': 'lightgbm',
        'training_params': {
            'n_estimators': 100,
            'random_state': 42
        },
        'data_source': 'historical_orders_2023'
    }
    
    # Register model
    version_manager.register_model("v1.0", example_metrics, example_config)
    
    # Check latest version
    latest = version_manager.get_latest_version()
    print(f"Latest version: {latest}")
    
    # List all versions
    versions = version_manager.list_versions()
    print(f"All versions: {versions}")