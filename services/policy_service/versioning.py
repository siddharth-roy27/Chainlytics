"""
Versioning for policy models and training artifacts
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class PolicyVersionInfo:
    """Information about a policy version"""
    version: str
    created_at: str
    policy_type: str
    agent_configurations: List[Dict[str, Any]]
    training_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    compatible_versions: List[str]
    promoted_to_production: bool = False
    promotion_date: Optional[str] = None


class PolicyVersionManager:
    """Manage versions of policy models"""
    
    def __init__(self, registry_path: str = "registry/policies"):
        """
        Initialize policy version manager
        
        Args:
            registry_path: Path to policy registry
        """
        self.registry_path = registry_path
        self.metadata_file = os.path.join(registry_path, "policy_versions.yaml")
        os.makedirs(registry_path, exist_ok=True)
    
    def register_version(self, version_info: PolicyVersionInfo) -> None:
        """
        Register a new policy version
        
        Args:
            version_info: Policy version information
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
    
    def _update_registry(self, version_info: PolicyVersionInfo) -> None:
        """Update the main policy registry"""
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
    
    def get_version_info(self, version: str) -> Optional[PolicyVersionInfo]:
        """
        Get information about a specific policy version
        
        Args:
            version: Version identifier
            
        Returns:
            Policy version information or None if not found
        """
        version_path = os.path.join(self.registry_path, version, "metadata.yaml")
        
        if not os.path.exists(version_path):
            return None
        
        with open(version_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        return PolicyVersionInfo(**metadata)
    
    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest policy version
        
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
        List all available policy versions
        
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
        Check if a policy version is compatible with a target version
        
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
        if target_version in version_info.compatible_versions:
            return True
        
        # Self-compatibility
        if version == target_version:
            return True
        
        return False
    
    def promote_version(self, version: str) -> bool:
        """
        Promote a policy version to production
        
        Args:
            version: Version to promote
            
        Returns:
            True if promoted successfully, False otherwise
        """
        version_info = self.get_version_info(version)
        if not version_info:
            return False
        
        # Update version info
        version_info.promoted_to_production = True
        version_info.promotion_date = datetime.now().isoformat()
        
        # Save updated metadata
        version_path = os.path.join(self.registry_path, version)
        metadata_path = os.path.join(version_path, "metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(asdict(version_info), f, default_flow_style=False)
        
        # Update registry
        self._update_registry(version_info)
        
        return True
    
    def get_production_version(self) -> Optional[str]:
        """
        Get the currently promoted production version
        
        Returns:
            Production version identifier or None if none promoted
        """
        versions = self.list_versions()
        
        for version in versions:
            version_info = self.get_version_info(version)
            if version_info and version_info.promoted_to_production:
                return version
        
        return None


@dataclass
class TrainingArtifactVersion:
    """Version information for training artifacts"""
    artifact_type: str  # 'model_weights', 'training_logs', 'evaluation_results'
    version: str
    created_at: str
    parent_policy_version: str
    metrics: Dict[str, float]
    artifact_path: str
    size_bytes: int


class ArtifactVersionManager:
    """Manage versions of training artifacts"""
    
    def __init__(self, artifacts_path: str = "registry/artifacts"):
        """
        Initialize artifact version manager
        
        Args:
            artifacts_path: Path to artifacts storage
        """
        self.artifacts_path = artifacts_path
        self.metadata_file = os.path.join(artifacts_path, "artifact_versions.yaml")
        os.makedirs(artifacts_path, exist_ok=True)
    
    def register_artifact(self, artifact_info: TrainingArtifactVersion) -> None:
        """
        Register a training artifact version
        
        Args:
            artifact_info: Artifact version information
        """
        # Create artifact directory
        artifact_dir = os.path.join(self.artifacts_path, artifact_info.version)
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Save artifact metadata
        metadata_path = os.path.join(artifact_dir, "artifact_metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(asdict(artifact_info), f, default_flow_style=False)
        
        # Update main registry
        self._update_registry(artifact_info)
    
    def _update_registry(self, artifact_info: TrainingArtifactVersion) -> None:
        """Update the main artifact registry"""
        # Load existing registry
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                registry = yaml.safe_load(f) or {}
        else:
            registry = {'artifacts': {}}
        
        # Add new artifact
        key = f"{artifact_info.artifact_type}_{artifact_info.version}"
        registry['artifacts'][key] = asdict(artifact_info)
        
        # Save updated registry
        with open(self.metadata_file, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False)
    
    def get_artifact_info(self, artifact_type: str, version: str) -> Optional[TrainingArtifactVersion]:
        """
        Get information about a specific artifact version
        
        Args:
            artifact_type: Type of artifact
            version: Version identifier
            
        Returns:
            Artifact version information or None if not found
        """
        key = f"{artifact_type}_{version}"
        artifact_dir = os.path.join(self.artifacts_path, version)
        metadata_path = os.path.join(artifact_dir, "artifact_metadata.yaml")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        return TrainingArtifactVersion(**metadata)
    
    def list_artifacts(self, artifact_type: Optional[str] = None) -> List[str]:
        """
        List all available artifact versions
        
        Args:
            artifact_type: Filter by artifact type (optional)
            
        Returns:
            List of artifact version identifiers
        """
        if not os.path.exists(self.metadata_file):
            return []
        
        with open(self.metadata_file, 'r') as f:
            registry = yaml.safe_load(f) or {}
        
        artifacts = registry.get('artifacts', {})
        
        if artifact_type:
            # Filter by artifact type
            filtered_artifacts = {
                k: v for k, v in artifacts.items() 
                if v['artifact_type'] == artifact_type
            }
            return list(filtered_artifacts.keys())
        else:
            return list(artifacts.keys())
    
    def get_latest_artifact(self, artifact_type: str) -> Optional[str]:
        """
        Get the latest version of a specific artifact type
        
        Args:
            artifact_type: Type of artifact
            
        Returns:
            Latest artifact version identifier or None if none exist
        """
        artifacts = self.list_artifacts(artifact_type)
        
        if not artifacts:
            return None
        
        # Parse timestamps and sort
        artifact_info_list = []
        for artifact_key in artifacts:
            artifact_info = self.get_artifact_info(
                artifact_type, 
                artifact_key.replace(f"{artifact_type}_", "")
            )
            if artifact_info:
                artifact_info_list.append((artifact_key, artifact_info.created_at))
        
        if not artifact_info_list:
            return None
        
        # Sort by creation time
        sorted_artifacts = sorted(
            artifact_info_list,
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_artifacts[0][0] if sorted_artifacts else None


class ModelRollbackManager:
    """Manage rollback operations for policy models"""
    
    def __init__(self, policy_version_manager: PolicyVersionManager,
                 artifact_version_manager: ArtifactVersionManager):
        """
        Initialize rollback manager
        
        Args:
            policy_version_manager: Policy version manager
            artifact_version_manager: Artifact version manager
        """
        self.policy_manager = policy_version_manager
        self.artifact_manager = artifact_version_manager
        self.rollback_history = []
    
    def rollback_to_version(self, version: str) -> bool:
        """
        Rollback to a specific policy version
        
        Args:
            version: Version to rollback to
            
        Returns:
            True if rollback successful, False otherwise
        """
        # Check if version exists
        version_info = self.policy_manager.get_version_info(version)
        if not version_info:
            return False
        
        # Record rollback operation
        rollback_record = {
            'rollback_version': version,
            'previous_production_version': self.policy_manager.get_production_version(),
            'rollback_timestamp': datetime.now().isoformat(),
            'status': 'initiated'
        }
        
        try:
            # Promote the target version to production
            success = self.policy_manager.promote_version(version)
            
            if success:
                rollback_record['status'] = 'completed'
                rollback_record['success'] = True
            else:
                rollback_record['status'] = 'failed'
                rollback_record['success'] = False
                
        except Exception as e:
            rollback_record['status'] = 'failed'
            rollback_record['success'] = False
            rollback_record['error'] = str(e)
        
        # Record rollback operation
        self.rollback_history.append(rollback_record)
        
        return rollback_record['success']
    
    def get_rollback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get rollback operation history
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of rollback operations
        """
        return self.rollback_history[-limit:]
    
    def can_rollback(self, version: str) -> bool:
        """
        Check if rollback to a specific version is possible
        
        Args:
            version: Version to check
            
        Returns:
            True if rollback is possible, False otherwise
        """
        # Check if version exists
        version_info = self.policy_manager.get_version_info(version)
        if not version_info:
            return False
        
        # Check if version has required artifacts
        # This would involve checking if model weights and other artifacts exist
        # For now, we'll assume if the version exists, rollback is possible
        return True


# Example usage
if __name__ == "__main__":
    # Create policy version manager
    policy_manager = PolicyVersionManager()
    
    # Register a policy version
    version_info = PolicyVersionInfo(
        version="policy_v1.0.0",
        created_at=datetime.now().isoformat(),
        policy_type="cooperative",
        agent_configurations=[
            {
                'id': 'warehouse_1',
                'type': 'ppo',
                'observation_dim': 64,
                'action_dim': 10
            }
        ],
        training_metrics={
            'final_reward': 15.2,
            'training_stability': 0.95,
            'convergence_rate': 0.87
        },
        training_config={
            'episodes': 1000,
            'learning_rate': 3e-4,
            'batch_size': 32
        },
        compatible_versions=["policy_v0.9.0", "policy_v0.9.1"],
        promoted_to_production=False
    )
    
    policy_manager.register_version(version_info)
    print(f"Registered policy version: {version_info.version}")
    
    # List versions
    versions = policy_manager.list_versions()
    print(f"Available versions: {versions}")
    
    # Get latest version
    latest = policy_manager.get_latest_version()
    print(f"Latest version: {latest}")
    
    # Create artifact version manager
    artifact_manager = ArtifactVersionManager()
    
    # Register an artifact
    artifact_info = TrainingArtifactVersion(
        artifact_type="model_weights",
        version="weights_v1.0.0",
        created_at=datetime.now().isoformat(),
        parent_policy_version="policy_v1.0.0",
        metrics={'accuracy': 0.92, 'loss': 0.05},
        artifact_path="/path/to/model/weights",
        size_bytes=1024000
    )
    
    artifact_manager.register_artifact(artifact_info)
    print(f"Registered artifact: {artifact_info.version}")
    
    # Create rollback manager
    rollback_manager = ModelRollbackManager(policy_manager, artifact_manager)
    
    # Check if rollback is possible
    can_rollback = rollback_manager.can_rollback("policy_v1.0.0")
    print(f"Can rollback to policy_v1.0.0: {can_rollback}")
</file>