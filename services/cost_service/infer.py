"""
Inference module for cost calculations
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
import joblib

from .cost_model import LinearCostModel, CostBreakdown
from .marginal_cost import MarginalCostAnalyzer


class CostInferenceEngine:
    """Inference engine for cost calculations"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model (optional)
            config_path: Path to model configuration (optional)
        """
        # Load calibrated parameters if available
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            parameters = config.get('calibrated_parameters', {})
            self.cost_model = LinearCostModel(parameters)
        else:
            # Use default parameters
            self.cost_model = LinearCostModel()
        
        self.analyzer = MarginalCostAnalyzer(self.cost_model)
    
    def calculate_costs(self, **kwargs) -> CostBreakdown:
        """
        Calculate costs for given parameters
        
        Args:
            **kwargs: Cost calculation parameters
            
        Returns:
            CostBreakdown with detailed cost components
        """
        return self.cost_model.calculate_total_cost(**kwargs)
    
    def calculate_marginal_costs(self, base_params: Dict, 
                               variables: List[str]) -> Dict[str, float]:
        """
        Calculate marginal costs for decision analysis
        
        Args:
            base_params: Base parameters for cost calculation
            variables: Variables to analyze
            
        Returns:
            Dictionary mapping variables to marginal costs
        """
        return self.analyzer.calculate_marginal_costs_multi_dimensional(
            base_params, variables
        )
    
    def optimize_decision(self, options: List[Dict]) -> Tuple[int, CostBreakdown]:
        """
        Find the option with minimum cost
        
        Args:
            options: List of parameter dictionaries for each option
            
        Returns:
            Tuple of (best_option_index, cost_breakdown)
        """
        min_cost = float('inf')
        best_index = -1
        best_breakdown = None
        
        for i, params in enumerate(options):
            cost_breakdown = self.cost_model.calculate_total_cost(**params)
            total_cost = cost_breakdown.total_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_index = i
                best_breakdown = cost_breakdown
                
        return best_index, best_breakdown
    
    def what_if_analysis(self, base_params: Dict, 
                        scenarios: List[Dict]) -> List[Tuple[CostBreakdown, Dict]]:
        """
        Perform what-if analysis for different scenarios
        
        Args:
            base_params: Base parameters
            scenarios: List of scenario modifications
            
        Returns:
            List of (cost_breakdown, scenario_description) tuples
        """
        results = []
        
        # Base case
        base_costs = self.cost_model.calculate_total_cost(**base_params)
        results.append((base_costs, {"scenario": "base_case"}))
        
        # Scenario cases
        for i, scenario_changes in enumerate(scenarios):
            # Apply scenario changes
            scenario_params = base_params.copy()
            scenario_params.update(scenario_changes)
            
            # Calculate costs
            scenario_costs = self.cost_model.calculate_total_cost(**scenario_params)
            results.append((scenario_costs, scenario_changes))
            
        return results


def load_latest_cost_model():
    """Load the latest trained cost model"""
    model_path = "registry/models/cost/v1.0/model.pkl"
    config_path = "registry/models/cost/v1.0/config.json"
    
    if os.path.exists(config_path):
        return CostInferenceEngine(config_path=config_path)
    else:
        return CostInferenceEngine()


def calculate_shipment_costs(distance: float, weight: float, 
                           inventory_value: float, holding_days: float,
                           delay_hours: float = 0.0, shipment_value: float = 0.0,
                           shortage_units: float = 0.0, units_handled: float = 0.0,
                           labor_hours: float = 0.0, vehicle_type: str = "standard") -> CostBreakdown:
    """
    Convenience function to calculate costs for a shipment
    
    Returns:
        CostBreakdown with detailed cost components
    """
    engine = load_latest_cost_model()
    
    return engine.calculate_costs(
        distance=distance,
        weight=weight,
        inventory_value=inventory_value,
        holding_days=holding_days,
        delay_hours=delay_hours,
        shipment_value=shipment_value,
        shortage_units=shortage_units,
        units_handled=units_handled,
        labor_hours=labor_hours,
        vehicle_type=vehicle_type
    )


def compare_route_options(route_options: List[Dict]) -> Tuple[int, CostBreakdown, List[CostBreakdown]]:
    """
    Compare multiple route options and find the cheapest
    
    Args:
        route_options: List of route parameter dictionaries
        
    Returns:
        Tuple of (best_index, best_cost_breakdown, all_cost_breakdowns)
    """
    engine = load_latest_cost_model()
    
    all_breakdowns = []
    for params in route_options:
        breakdown = engine.calculate_costs(**params)
        all_breakdowns.append(breakdown)
    
    best_index, best_breakdown = engine.optimize_decision(route_options)
    
    return best_index, best_breakdown, all_breakdowns


# Example usage
if __name__ == "__main__":
    # Initialize inference engine
    engine = load_latest_cost_model()
    
    # Calculate costs for a sample shipment
    costs = engine.calculate_costs(
        distance=500.0,      # km
        weight=1200.0,       # kg
        inventory_value=50000.0,  # $ value
        holding_days=5.0,
        delay_hours=3.0,
        shipment_value=25000.0,
        shortage_units=10.0,
        units_handled=100.0,
        labor_hours=8.0,
        vehicle_type="express"
    )
    
    print("Shipment Cost Analysis:")
    print(f"Transport Cost: ${costs.transport_cost:.2f}")
    print(f"Holding Cost: ${costs.holding_cost:.2f}")
    print(f"Delay Penalty: ${costs.delay_penalty:.2f}")
    print(f"Shortage Cost: ${costs.shortage_cost:.2f}")
    print(f"Handling Cost: ${costs.handling_cost:.2f}")
    print(f"Fuel Cost: ${costs.fuel_cost:.2f}")
    print(f"Labor Cost: ${costs.labor_cost:.2f}")
    print(f"Maintenance Cost: ${costs.maintenance_cost:.2f}")
    print(f"Total Cost: ${costs.total_cost:.2f}")
    
    # Calculate marginal costs
    base_params = {
        'distance': 500.0,
        'weight': 1200.0,
        'inventory_value': 50000.0,
        'holding_days': 5.0
    }
    
    marginal_costs = engine.calculate_marginal_costs(
        base_params, 
        ['distance', 'weight', 'holding_days']
    )
    
    print("\nMarginal Costs:")
    for variable, cost in marginal_costs.items():
        print(f"  Marginal cost of {variable}: ${cost:.2f} per unit")
    
    # Compare route options
    route_options = [
        {'distance': 500, 'weight': 1000, 'labor_hours': 8},
        {'distance': 480, 'weight': 1100, 'labor_hours': 9},
        {'distance': 520, 'weight': 950, 'labor_hours': 7}
    ]
    
    best_index, best_costs, all_costs = compare_route_options(route_options)
    
    print("\nRoute Comparison:")
    for i, costs in enumerate(all_costs):
        marker = " (*)" if i == best_index else ""
        print(f"  Option {i+1}: ${costs.total_cost:.2f}{marker}")