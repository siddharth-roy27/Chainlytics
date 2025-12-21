"""
Monte Carlo sampling for scenario generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ScenarioSample:
    """Represents a single scenario sample"""
    sample_id: str
    timestamp: datetime
    demand_multiplier: float
    cost_multiplier: float
    disruption_probability: float
    weather_factor: float


class MonteCarloSampler:
    """Monte Carlo sampler for generating scenario samples"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
    def sample_demand_multiplier(self, distribution: str = "normal", 
                               params: Dict[str, float] = None) -> float:
        """
        Sample demand multiplier
        
        Args:
            distribution: Distribution type ('normal', 'lognormal', 'uniform')
            params: Distribution parameters
            
        Returns:
            Demand multiplier
        """
        if params is None:
            params = {'mean': 1.0, 'std': 0.2}
            
        if distribution == "normal":
            return max(0.1, self.rng.normal(params['mean'], params['std']))
        elif distribution == "lognormal":
            return self.rng.lognormal(params.get('mean', 0), params.get('sigma', 0.2))
        elif distribution == "uniform":
            return self.rng.uniform(params.get('low', 0.5), params.get('high', 1.5))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    
    def sample_cost_multiplier(self, distribution: str = "normal",
                              params: Dict[str, float] = None) -> float:
        """
        Sample cost multiplier
        
        Args:
            distribution: Distribution type
            params: Distribution parameters
            
        Returns:
            Cost multiplier
        """
        if params is None:
            params = {'mean': 1.0, 'std': 0.15}
            
        return max(0.1, self.rng.normal(params['mean'], params['std']))
    
    def sample_disruption_probability(self, base_prob: float = 0.05,
                                   volatility: float = 0.02) -> float:
        """
        Sample disruption probability
        
        Args:
            base_prob: Base disruption probability
            volatility: Volatility of disruption probability
            
        Returns:
            Disruption probability
        """
        return min(0.99, max(0.01, self.rng.normal(base_prob, volatility)))
    
    def sample_weather_factor(self, severity_levels: List[float] = None) -> float:
        """
        Sample weather factor
        
        Args:
            severity_levels: Possible weather severity levels
            
        Returns:
            Weather factor
        """
        if severity_levels is None:
            severity_levels = [0.8, 0.9, 1.0, 1.1, 1.2]  # Impact on operations
            
        return self.rng.choice(severity_levels)
    
    def generate_samples(self, n_samples: int = 1000,
                        distributions: Dict[str, Dict] = None) -> List[ScenarioSample]:
        """
        Generate multiple scenario samples
        
        Args:
            n_samples: Number of samples to generate
            distributions: Custom distribution parameters
            
        Returns:
            List of scenario samples
        """
        if distributions is None:
            distributions = {}
            
        samples = []
        base_time = datetime.now()
        
        for i in range(n_samples):
            sample = ScenarioSample(
                sample_id=f"SAMPLE_{i:06d}",
                timestamp=base_time + timedelta(hours=i),
                demand_multiplier=self.sample_demand_multiplier(
                    **distributions.get('demand', {})
                ),
                cost_multiplier=self.sample_cost_multiplier(
                    **distributions.get('cost', {})
                ),
                disruption_probability=self.sample_disruption_probability(),
                weather_factor=self.sample_weather_factor()
            )
            samples.append(sample)
            
        return samples


# Example usage
if __name__ == "__main__":
    # Initialize sampler
    sampler = MonteCarloSampler(seed=42)
    
    # Generate samples
    samples = sampler.generate_samples(n_samples=100)
    
    # Print first few samples
    print("Generated Scenario Samples:")
    for i, sample in enumerate(samples[:5]):
        print(f"Sample {i+1}: {sample}")