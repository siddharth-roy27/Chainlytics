"""
Versioning for state encoding models and embeddings
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class StateEncodingVersion:
    """Version information for state encoding"""
    version: str
    created_at: str
    model_type: str
    model_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_data_info: Dict[str, Any]
    compatible_with: List[str]  # List of compatible versions


class StateVersionManager:
    """Manage versions of state encoding models and embeddings"""
    
    def __init__(self, registry_path: str = "registry/models/state"):
        """
        Initialize version manager
        
        Args:
            registry_path: Path to model registry
        """
        self.registry_path = registry_path
        self.metadata_file = os.path.join(registry_path, "versions.yaml")
        os.makedirs(registry_path, exist_ok=True)
    
    def register_version(self, version_info: StateEncodingVersion) -> None:
        """
        Register a new version of the state encoding model
        
        Args:
            version_info: Version information
        """
        # Create version directory
        version_path = os.path.join(self.registry_path, version_info.version)
        os.makedirs(version_path, exist_ok=True)
        
        # Save version metadata
        metadata_path = os.path.join(version_path, "metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(asdict(version_info), f, default_flow_style=False)
        
        # Update main registry
        self._update_registry(version_info)
    
    def _update_registry(self, version_info: StateEncodingVersion) -> None:
        """Update the main version registry"""
        # Load existing registry
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                registry = yaml.safe_load(f) or {}
        else:
            registry = {'versions': {}}
        
        # Add new version
        registry['versions'][version_info.version] = asdict(version_info)
        
        # Save updated registry
        with open(self.metadata_file, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False)
    
    def get_version_info(self, version: str) -> Optional[StateEncodingVersion]:
        """
        Get information about a specific version
        
        Args:
            version: Version identifier
            
        Returns:
            Version information or None if not found
        """
        version_path = os.path.join(self.registry_path, version, "metadata.yaml")
        
        if not os.path.exists(version_path):
            return None
        
        with open(version_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        return StateEncodingVersion(**metadata)
    
    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest version
        
        Returns:
            Latest version identifier or None if no versions exist
        """
        if not os.path.exists(self.metadata_file):
            return None
        
        with open(self.metadata_file, 'r') as f:
            registry = yaml.safe_load(f) or {}
        
        versions = registry.get('versions', {})
        if not versions:
            return None
        
        # Sort by creation time
        sorted_versions = sorted(
            versions.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )
        
        return sorted_versions[0][0] if sorted_versions else None
    
    def list_versions(self) -> List[str]:
        """
        List all available versions
        
        Returns:
            List of version identifiers
        """
        if not os.path.exists(self.metadata_file):
            return []
        
        with open(self.metadata_file, 'r') as f:
            registry = yaml.safe_load(f) or {}
        
        versions = registry.get('versions', {})
        return list(versions.keys())
    
    def is_compatible(self, version: str, target_version: str) -> bool:
        """
        Check if a version is compatible with a target version
        
        Args:
            version: Version to check
            target_version: Target version
            
        Returns:
            True if compatible, False otherwise
        """
        version_info = self.get_version_info(version)
        if not version_info:
            return False
        
        # Direct compatibility
        if target_version in version_info.compatible_with:
            return True
        
        # Self-compatibility
        if version == target_version:
            return True
        
        return False
    
    def get_compatible_versions(self, version: str) -> List[str]:
        """
        Get all versions compatible with a given version
        
        Args:
            version: Version to check compatibility for
            
        Returns:
            List of compatible versions
        """
        version_info = self.get_version_info(version)
        if not version_info:
            return []
        
        compatible = version_info.compatible_with[:]
        
        # Add self
        if version not in compatible:
            compatible.append(version)
        
        return compatible
    
    def promote_version(self, version: str, criteria: Dict[str, Any] = None) -> bool:
        """
        Promote a version to production based on criteria
        
        Args:
            version: Version to promote
            criteria: Promotion criteria (performance thresholds, etc.)
            
        Returns:
            True if promoted, False otherwise
        """
        version_info = self.get_version_info(version)
        if not version_info:
            return False
        
        # Check promotion criteria
        if criteria:
            metrics = version_info.performance_metrics
            for metric, threshold in criteria.items():
                if metric not in metrics:
                    return False
                if metrics[metric] < threshold:
                    return False
        
        # Mark as promoted (in a real system, this would update a "production" pointer)
        promoted_file = os.path.join(self.registry_path, version, "promoted")
        with open(promoted_file, 'w') as f:
            f.write(datetime.now().isoformat())
        
        return True
    
    def is_promoted(self, version: str) -> bool:
        """
        Check if a version is promoted to production
        
        Args:
            version: Version to check
            
        Returns:
            True if promoted, False otherwise
        """
        promoted_file = os.path.join(self.registry_path, version, "promoted")
        return os.path.exists(promoted_file)


class EmbeddingVersionManager:
    """Manage versions of state embeddings"""
    
    def __init__(self, embeddings_path: str = "data/embeddings"):
        """
        Initialize embedding version manager
        
        Args:
            embeddings_path: Path to embeddings storage
        """
        self.embeddings_path = embeddings_path
        os.makedirs(embeddings_path, exist_ok=True)
    
    def save_embedding_version(self, embeddings: Any, 
                             metadata: Dict[str, Any],
                             version: str) -> str:
        """
        Save a version of embeddings
        
        Args:
            embeddings: Embeddings data
            metadata: Metadata about the embeddings
            version: Version identifier
            
        Returns:
            Path to saved embeddings
        """
        import pickle
        
        # Create version directory
        version_path = os.path.join(self.embeddings_path, version)
        os.makedirs(version_path, exist_ok=True)
        
        # Save embeddings
        embeddings_file = os.path.join(version_path, "embeddings.pkl")
        with open(embeddings_file, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'metadata': metadata,
                'saved_at': datetime.now().isoformat()
            }, f)
        
        # Save metadata
        metadata_file = os.path.join(version_path, "info.yaml")
        version_info = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata
        }
        with open(metadata_file, 'w') as f:
            yaml.dump(version_info, f, default_flow_style=False)
        
        return embeddings_file
    
    def load_embedding_version(self, version: str) -> tuple:
        """
        Load a specific version of embeddings
        
        Args:
            version: Version identifier
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        import pickle
        
        embeddings_file = os.path.join(self.embeddings_path, version, "embeddings.pkl")
        
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings version {version} not found")
        
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['embeddings'], data['metadata']
    
    def get_embedding_versions(self) -> List[str]:
        """
        Get all available embedding versions
        
        Returns:
            List of version identifiers
        """
        if not os.path.exists(self.embeddings_path):
            return []
        
        versions = []
        for item in os.listdir(self.embeddings_path):
            item_path = os.path.join(self.embeddings_path, item)
            if os.path.isdir(item_path):
                versions.append(item)
        
        return versions


# Example usage
if __name__ == "__main__":
    # State version manager example
    state_manager = StateVersionManager()
    
    # Register a version
    version_info = StateEncodingVersion(
        version="v1.0.0",
        created_at=datetime.now().isoformat(),
        model_type="gnn",
        model_parameters={
            "embedding_dim": 128,
            "hidden_dims": [256, 128],
            "node_feature_dim": 64
        },
        performance_metrics={
            "encoding_accuracy": 0.95,
            "processing_time_ms": 15.2
        },
        training_data_info={
            "dataset": "logistics_graphs_2023",
            "samples": 10000
        },
        compatible_with=["v0.9.0", "v0.9.1"]
    )
    
    state_manager.register_version(version_info)
    print(f"Registered version: {version_info.version}")
    
    # Check latest version
    latest = state_manager.get_latest_version()
    print(f"Latest version: {latest}")
    
    # List all versions
    versions = state_manager.list_versions()
    print(f"Available versions: {versions}")
    
    # Embedding version manager example
    embedding_manager = EmbeddingVersionManager()
    
    # Create sample embeddings
    import numpy as np
    sample_embeddings = np.random.randn(100, 128)
    metadata = {
        "graph_size": 100,
        "timestamp": datetime.now().isoformat(),
        "source": "simulation_run_001"
    }
    
    # Save version
    version_path = embedding_manager.save_embedding_version(
        sample_embeddings, metadata, "emb_v1.0.0"
    )
    print(f"Saved embeddings to: {version_path}")
    
    # List versions
    emb_versions = embedding_manager.get_embedding_versions()
    print(f"Embedding versions: {emb_versions}")
</file>