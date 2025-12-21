"""
Disruption modeling and shock injection for scenario generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class ShockType(Enum):
    """Types of disruptions that can occur"""
    SUPPLIER_DELAY = "supplier_delay"
    TRANSPORT_DISRUPTION = "transport_disruption"
    WEATHER_EVENT = "weather_event"
    DEMAND_SPIKE = "demand_spike"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    REGULATORY_CHANGE = "regulatory_change"


@dataclass
class DisruptionShock:
    """Represents a specific disruption event"""
    shock_id: str
    shock_type: ShockType
    start_time: datetime
    end_time: Optional[datetime]
    intensity: float  # 0.0 to 1.0
    affected_entities: List[str]  # e.g., warehouse IDs, route segments
    description: str


class ShockInjector:
    """Inject realistic disruptions into scenarios"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.shock_templates = self._initialize_shock_templates()
    
    def _initialize_shock_templates(self) -> Dict[ShockType, Dict]:
        """Initialize templates for different shock types"""
        return {
            ShockType.SUPPLIER_DELAY: {
                'duration_range': (1, 7),  # days
                'intensity_range': (0.3, 0.9),
                'frequency': 0.2  # probability per month
            },
            ShockType.TRANSPORT_DISRUPTION: {
                'duration_range': (0.5, 3),  # days
                'intensity_range': (0.4, 0.8),
                'frequency': 0.15
            },
            ShockType.WEATHER_EVENT: {
                'duration_range': (1, 5),  # days
                'intensity_range': (0.5, 1.0),
                'frequency': 0.1
            },
            ShockType.DEMAND_SPIKE: {
                'duration_range': (2, 14),  # days
                'intensity_range': (1.5, 3.0),  # demand multiplier
                'frequency': 0.25
            },
            ShockType.INFRASTRUCTURE_FAILURE: {
                'duration_range': (3, 30),  # days
                'intensity_range': (0.7, 1.0),
                'frequency': 0.05
            },
            ShockType.REGULATORY_CHANGE: {
                'duration_range': (30, 365),  # days (long term)
                'intensity_range': (0.2, 0.6),  # usually negative impact
                'frequency': 0.02
            }
        }
    
    def generate_shock(self, shock_type: ShockType, 
                      start_time: datetime,
                      affected_entities: List[str] = None) -> DisruptionShock:
        """
        Generate a specific disruption shock
        
        Args:
            shock_type: Type of disruption
            start_time: When the disruption starts
            affected_entities: Entities affected by the disruption
            
        Returns:
            DisruptionShock object
        """
        template = self.shock_templates[shock_type]
        
        # Sample duration and intensity
        duration_days = self.rng.integers(
            template['duration_range'][0], 
            template['duration_range'][1] + 1
        )
        end_time = start_time + timedelta(days=duration_days)
        
        intensity = self.rng.uniform(
            template['intensity_range'][0],
            template['intensity_range'][1]
        )
        
        if affected_entities is None:
            # Randomly select affected entities
            all_entities = [f"WH{i:03d}" for i in range(1, 21)]  # 20 warehouses
            n_affected = self.rng.integers(1, min(5, len(all_entities)) + 1)
            affected_entities = list(self.rng.choice(all_entities, n_affected, replace=False))
        
        shock = DisruptionShock(
            shock_id=f"SHOCK_{shock_type.value.upper()}_{int(start_time.timestamp())}",
            shock_type=shock_type,
            start_time=start_time,
            end_time=end_time,
            intensity=intensity,
            affected_entities=affected_entities,
            description=f"{shock_type.value.replace('_', ' ').title()} disruption"
        )
        
        return shock
    
    def inject_shocks_into_timeline(self, timeline_start: datetime,
                                  timeline_end: datetime,
                                  entity_list: List[str] = None) -> List[DisruptionShock]:
        """
        Inject multiple shocks into a timeline
        
        Args:
            timeline_start: Start of simulation timeline
            timeline_end: End of simulation timeline
            entity_list: List of entities in the system
            
        Returns:
            List of disruption shocks
        """
        shocks = []
        current_time = timeline_start
        
        # Calculate number of periods based on frequency
        total_days = (timeline_end - timeline_start).days
        
        for shock_type, template in self.shock_templates.items():
            # Calculate expected number of events
            expected_events = total_days * template['frequency'] / 30  # monthly frequency
            n_events = max(0, int(self.rng.poisson(expected_events)))
            
            # Generate events
            for _ in range(n_events):
                # Random time within timeline
                event_day = self.rng.integers(0, total_days)
                event_time = timeline_start + timedelta(days=event_day)
                
                # Generate shock
                shock = self.generate_shock(shock_type, event_time, entity_list)
                shocks.append(shock)
        
        # Sort by start time
        shocks.sort(key=lambda x: x.start_time)
        return shocks
    
    def apply_shock_effects(self, base_scenario: Dict, 
                           shocks: List[DisruptionShock]) -> Dict:
        """
        Apply shock effects to a base scenario
        
        Args:
            base_scenario: Base scenario parameters
            shocks: List of shocks to apply
            
        Returns:
            Modified scenario with shock effects
        """
        modified_scenario = base_scenario.copy()
        
        # Apply each shock
        for shock in shocks:
            if shock.shock_type == ShockType.DEMAND_SPIKE:
                # Increase demand
                modified_scenario['demand_multiplier'] *= shock.intensity
            elif shock.shock_type == ShockType.SUPPLIER_DELAY:
                # Increase lead times
                modified_scenario['lead_time_multiplier'] = modified_scenario.get('lead_time_multiplier', 1.0) * shock.intensity
            elif shock.shock_type == ShockType.TRANSPORT_DISRUPTION:
                # Increase transport costs
                modified_scenario['transport_cost_multiplier'] = modified_scenario.get('transport_cost_multiplier', 1.0) * shock.intensity
            elif shock.shock_type == ShockType.WEATHER_EVENT:
                # Affect multiple factors
                modified_scenario['speed_reduction'] = modified_scenario.get('speed_reduction', 1.0) * shock.intensity
                modified_scenario['operating_cost_increase'] = modified_scenario.get('operating_cost_increase', 1.0) * (2 - shock.intensity)
        
        return modified_scenario


# Example usage
if __name__ == "__main__":
    # Initialize shock injector
    shock_injector = ShockInjector(seed=42)
    
    # Define timeline
    start_time = datetime.now()
    end_time = start_time + timedelta(days=90)  # 3 months
    
    # Generate shocks
    shocks = shock_injector.inject_shocks_into_timeline(start_time, end_time)
    
    print(f"Generated {len(shocks)} shocks:")
    for i, shock in enumerate(shocks[:5]):  # Show first 5
        print(f"  {i+1}. {shock.shock_type.value}: {shock.description}")
        print(f"     Intensity: {shock.intensity:.2f}")
        print(f"     Duration: {shock.start_time} to {shock.end_time}")
        print(f"     Affected: {shock.affected_entities}")
        print()