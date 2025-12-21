"""
Scenario persistence and storage management
"""

import os
import json
import gzip
import shutil
from typing import List, Dict, Any
from dataclasses import asdict
from datetime import datetime
import pandas as pd

from .generator import Scenario


class ScenarioStorageManager:
    """Manage storage and retrieval of scenarios"""
    
    def __init__(self, storage_path: str = "data/scenarios"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
    def save_scenario_batch(self, scenarios: List[Scenario], 
                          batch_id: str = None,
                          compress: bool = True) -> str:
        """
        Save a batch of scenarios to persistent storage
        
        Args:
            scenarios: List of scenarios to save
            batch_id: Identifier for this batch
            compress: Whether to compress the output
            
        Returns:
            Path to saved batch
        """
        if batch_id is None:
            batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Create batch directory
        batch_dir = os.path.join(self.storage_path, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Convert scenarios to serializable format
        scenarios_data = []
        for scenario in scenarios:
            scenario_dict = {
                'scenario_id': scenario.scenario_id,
                'timestamp': scenario.timestamp.isoformat(),
                'parameters': scenario.parameters,
                'shocks': [asdict(shock) for shock in scenario.shocks],
                'metadata': scenario.metadata
            }
            scenarios_data.append(scenario_dict)
        
        # Save scenarios
        if compress:
            file_path = os.path.join(batch_dir, "scenarios.json.gz")
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                json.dump(scenarios_data, f, indent=2)
        else:
            file_path = os.path.join(batch_dir, "scenarios.json")
            with open(file_path, 'w') as f:
                json.dump(scenarios_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'batch_id': batch_id,
            'n_scenarios': len(scenarios),
            'saved_at': datetime.now().isoformat(),
            'compressed': compress,
            'version': '1.0'
        }
        
        metadata_path = os.path.join(batch_dir, "batch_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved batch {batch_id} with {len(scenarios)} scenarios to {file_path}")
        return batch_dir
    
    def load_scenario_batch(self, batch_id: str) -> List[Scenario]:
        """
        Load a batch of scenarios from storage
        
        Args:
            batch_id: Identifier of batch to load
            
        Returns:
            List of scenarios
        """
        batch_dir = os.path.join(self.storage_path, batch_id)
        metadata_path = os.path.join(batch_dir, "batch_metadata.json")
        
        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found for batch {batch_id}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Determine file path
        compressed = metadata.get('compressed', True)
        if compressed:
            file_path = os.path.join(batch_dir, "scenarios.json.gz")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                scenarios_data = json.load(f)
        else:
            file_path = os.path.join(batch_dir, "scenarios.json")
            with open(file_path, 'r') as f:
                scenarios_data = json.load(f)
        
        # Convert back to Scenario objects
        scenarios = []
        for data in scenarios_data:
            # Note: For simplicity, we're not reconstructing DisruptionShock objects
            # In a production system, we would need proper deserialization
            scenario = Scenario(
                scenario_id=data['scenario_id'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                parameters=data['parameters'],
                shocks=[],  # Simplified - would need proper reconstruction
                metadata=data['metadata']
            )
            scenarios.append(scenario)
        
        print(f"Loaded batch {batch_id} with {len(scenarios)} scenarios")
        return scenarios
    
    def list_available_batches(self) -> List[Dict[str, Any]]:
        """
        List all available scenario batches
        
        Returns:
            List of batch metadata
        """
        batches = []
        
        # Iterate through batch directories
        for batch_dir_name in os.listdir(self.storage_path):
            batch_dir = os.path.join(self.storage_path, batch_dir_name)
            if os.path.isdir(batch_dir):
                metadata_path = os.path.join(batch_dir, "batch_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    batches.append(metadata)
        
        # Sort by save time
        batches.sort(key=lambda x: x['saved_at'], reverse=True)
        return batches
    
    def save_scenario_summary(self, scenarios: List[Scenario], 
                            batch_id: str) -> str:
        """
        Save a summary of scenarios for quick analysis
        
        Args:
            scenarios: List of scenarios
            batch_id: Batch identifier
            
        Returns:
            Path to summary file
        """
        batch_dir = os.path.join(self.storage_path, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Extract key metrics
        summary_data = []
        for scenario in scenarios:
            row = {
                'scenario_id': scenario.scenario_id,
                'timestamp': scenario.timestamp.isoformat(),
                **scenario.parameters
            }
            summary_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save as CSV for easy analysis
        csv_path = os.path.join(batch_dir, "scenarios_summary.csv")
        df.to_csv(csv_path, index=False)
        
        # Save statistical summary
        stats_path = os.path.join(batch_dir, "scenarios_stats.json")
        stats = {
            'batch_id': batch_id,
            'n_scenarios': len(scenarios),
            'parameter_statistics': {}
        }
        
        # Calculate statistics for numerical parameters
        for column in df.columns:
            if column not in ['scenario_id', 'timestamp']:
                stats['parameter_statistics'][column] = {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'median': float(df[column].median())
                }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved summary for batch {batch_id} to {csv_path}")
        return csv_path
    
    def archive_old_batches(self, older_than_days: int = 30) -> int:
        """
        Archive batches older than specified days
        
        Args:
            older_than_days: Age threshold for archiving
            
        Returns:
            Number of batches archived
        """
        archived_count = 0
        cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 3600)
        
        # Create archive directory
        archive_dir = os.path.join(self.storage_path, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        
        # Check each batch
        for batch_dir_name in os.listdir(self.storage_path):
            if batch_dir_name == "archive":
                continue
                
            batch_dir = os.path.join(self.storage_path, batch_dir_name)
            if os.path.isdir(batch_dir):
                metadata_path = os.path.join(batch_dir, "batch_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    saved_time = datetime.fromisoformat(metadata['saved_at']).timestamp()
                    if saved_time < cutoff_time:
                        # Move to archive
                        archive_path = os.path.join(archive_dir, batch_dir_name)
                        shutil.move(batch_dir, archive_path)
                        archived_count += 1
                        print(f"Archived batch {batch_dir_name}")
        
        return archived_count


# Example usage
if __name__ == "__main__":
    # Initialize storage manager
    storage_manager = ScenarioStorageManager()
    
    # Show available batches
    batches = storage_manager.list_available_batches()
    print(f"Available batches: {len(batches)}")
    for batch in batches[:3]:  # Show first 3
        print(f"  - {batch['batch_id']}: {batch['n_scenarios']} scenarios")