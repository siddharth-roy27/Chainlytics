"""
Scenario generator that combines sampling and shock injection
"""

import numpy as np
import pandas as pd
import uuid
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os

from .sampler import MonteCarloSampler, ScenarioSample
from .shocks import ShockInjector, DisruptionShock, ShockType


@dataclass
class Scenario:
    """Complete scenario definition"""
    scenario_id: str
    timestamp: datetime
    parameters: Dict[str, float]
    shocks: List[DisruptionShock]
    metadata: Dict[str, str]


class ScenarioGenerator:
    """Generate complete scenarios combining sampling and shocks"""
    
    def __init__(self, seed: int = 42):
        self.sampler = MonteCarloSampler(seed)
        self.shock_injector = ShockInjector(seed)
        self.rng = np.random.default_rng(seed)
        
    def generate_base_parameters(self, n_scenarios: int = 1000) -> List[Dict[str, float]]:
        """
        Generate base scenario parameters using Monte Carlo sampling
        
        Args:
            n_scenarios: Number of scenarios to generate
            
        Returns:
            List of parameter dictionaries
        """
        samples = self.sampler.generate_samples(n_scenarios)
        
        parameters = []
        for sample in samples:
            param_dict = {
                'demand_multiplier': sample.demand_multiplier,
                'cost_multiplier': sample.cost_multiplier,
                'disruption_probability': sample.disruption_probability,
                'weather_factor': sample.weather_factor,
                'timestamp': sample.timestamp.isoformat(),
                'lead_time_multiplier': 1.0,
                'transport_cost_multiplier': 1.0,
                'speed_reduction': 1.0,
                'operating_cost_increase': 1.0
            }
            parameters.append(param_dict)
            
        return parameters
    
    def generate_scenarios(self, n_scenarios: int = 1000,
                          timeline_start: datetime = None,
                          timeline_end: datetime = None,
                          entity_list: List[str] = None) -> List[Scenario]:
        """
        Generate complete scenarios with parameters and shocks
        
        Args:
            n_scenarios: Number of scenarios to generate
            timeline_start: Start of simulation timeline
            timeline_end: End of simulation timeline
            entity_list: List of entities in system
            
        Returns:
            List of complete scenarios
        """
        if timeline_start is None:
            timeline_start = datetime.now()
        if timeline_end is None:
            timeline_end = timeline_start + timedelta(days=90)  # 3 months default
        if entity_list is None:
            entity_list = [f"WH{i:03d}" for i in range(1, 21)]  # Default 20 warehouses
            
        # Generate base parameters
        base_parameters = self.generate_base_parameters(n_scenarios)
        
        # Generate common shocks for all scenarios
        common_shocks = self.shock_injector.inject_shocks_into_timeline(
            timeline_start, timeline_end, entity_list
        )
        
        # Create scenarios
        scenarios = []
        for i, params in enumerate(base_parameters):
            # Add some scenario-specific shocks
            scenario_specific_shocks = []
            if self.rng.random() < params['disruption_probability']:
                # Add a random shock specific to this scenario
                shock_types = list(ShockType)
                shock_type = self.rng.choice(shock_types)
                shock_time = timeline_start + timedelta(
                    days=self.rng.integers(0, (timeline_end - timeline_start).days)
                )
                specific_shock = self.shock_injector.generate_shock(
                    shock_type, shock_time, entity_list
                )
                scenario_specific_shocks.append(specific_shock)
            
            # Combine common and specific shocks
            all_shocks = common_shocks + scenario_specific_shocks
            
            # Apply shock effects to parameters
            modified_params = self.shock_injector.apply_shock_effects(params, all_shocks)
            
            scenario = Scenario(
                scenario_id=f"SCENARIO_{i:06d}_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.fromisoformat(modified_params['timestamp']),
                parameters=modified_params,
                shocks=all_shocks,
                metadata={
                    'generated_at': datetime.now().isoformat(),
                    'source': 'monte_carlo_with_shocks',
                    'version': '1.0'
                }
            )
            
            scenarios.append(scenario)
            
        return scenarios
    
    def perturb_scenario(self, base_scenario: Scenario, 
                        perturbation_strength: float = 0.1) -> Scenario:
        """
        Create a perturbed version of a base scenario
        
        Args:
            base_scenario: Base scenario to perturb
            perturbation_strength: Strength of perturbation (0.0 to 1.0)
            
        Returns:
            Perturbed scenario
        """
        # Copy base scenario
        perturbed_params = base_scenario.parameters.copy()
        
        # Apply small perturbations to numerical parameters
        for key, value in perturbed_params.items():
            if isinstance(value, (int, float)) and key not in ['timestamp']:
                # Add gaussian noise
                noise = self.rng.normal(0, perturbation_strength * abs(value))
                perturbed_params[key] = value + noise
        
        # Create new scenario with perturbed parameters
        perturbed_scenario = Scenario(
            scenario_id=f"Perturbed_{base_scenario.scenario_id}",
            timestamp=base_scenario.timestamp,
            parameters=perturbed_params,
            shocks=base_scenario.shocks,  # Keep same shocks
            metadata={
                **base_scenario.metadata,
                'perturbed_from': base_scenario.scenario_id,
                'perturbation_strength': perturbation_strength
            }
        )
        
        return perturbed_scenario


def generate_scenario_batch(n_scenarios: int = 1000,
                          batch_id: str = None,
                          output_dir: str = "data/scenarios") -> str:
    """
    Generate a batch of scenarios and save to disk
    
    Args:
        n_scenarios: Number of scenarios to generate
        batch_id: Identifier for this batch
        output_dir: Directory to save scenarios
        
    Returns:
        Path to saved batch
    """
    if batch_id is None:
        batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    batch_dir = os.path.join(output_dir, batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    # Generate scenarios
    generator = ScenarioGenerator()
    scenarios = generator.generate_scenarios(n_scenarios)
    
    # Save scenarios
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
    
    # Save to JSON
    output_path = os.path.join(batch_dir, "scenarios.json")
    with open(output_path, 'w') as f:
        json.dump(scenarios_data, f, indent=2)
    
    # Save summary
    summary = {
        'batch_id': batch_id,
        'n_scenarios': len(scenarios),
        'generated_at': datetime.now().isoformat(),
        'parameter_ranges': {}
    }
    
    # Calculate parameter ranges
    if scenarios:
        param_keys = scenarios[0].parameters.keys()
        for key in param_keys:
            if key != 'timestamp':
                values = [s.parameters[key] for s in scenarios]
                summary['parameter_ranges'][key] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
    
    summary_path = os.path.join(batch_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated batch {batch_id} with {len(scenarios)} scenarios")
    print(f"Saved to {batch_dir}")
    
    return batch_dir


# Example usage
if __name__ == "__main__":
    # Generate a small batch of scenarios
    batch_path = generate_scenario_batch(n_scenarios=100, batch_id="TEST_BATCH_001")
    print(f"Batch saved to: {batch_path}")