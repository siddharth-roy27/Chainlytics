"""
Marginal cost analysis for decision making
"""

import numpy as np
from typing import Dict, List, Callable, Tuple
from .cost_model import LogisticsCostModel, LinearCostModel, CostBreakdown


class MarginalCostAnalyzer:
    """Analyze marginal costs for logistics decisions"""
    
    def __init__(self, cost_model: LogisticsCostModel = None):
        self.cost_model = cost_model or LinearCostModel()
    
    def calculate_marginal_cost(self, base_params: Dict, 
                              variable: str, 
                              delta: float = 1.0) -> Tuple[float, CostBreakdown]:
        """
        Calculate marginal cost for a specific variable
        
        Args:
            base_params: Base parameters for cost calculation
            variable: Variable to analyze (e.g., 'distance', 'weight')
            delta: Change in variable for marginal calculation
            
        Returns:
            Tuple of (marginal_cost, detailed_cost_breakdown)
        """
        # Calculate base cost
        base_costs = self.cost_model.calculate_total_cost(**base_params)
        base_total = base_costs.total_cost
        
        # Calculate cost with delta change
        modified_params = base_params.copy()
        if variable in modified_params:
            modified_params[variable] += delta
        else:
            modified_params[variable] = delta
            
        modified_costs = self.cost_model.calculate_total_cost(**modified_params)
        modified_total = modified_costs.total_cost
        
        # Calculate marginal cost
        marginal_cost = modified_total - base_total
        
        return marginal_cost, modified_costs
    
    def calculate_marginal_costs_multi_dimensional(self, base_params: Dict,
                                                variables: List[str],
                                                deltas: List[float] = None) -> Dict[str, float]:
        """
        Calculate marginal costs for multiple variables
        
        Args:
            base_params: Base parameters for cost calculation
            variables: List of variables to analyze
            deltas: List of deltas for each variable (default: 1.0 for each)
            
        Returns:
            Dictionary mapping variables to their marginal costs
        """
        if deltas is None:
            deltas = [1.0] * len(variables)
        
        marginal_costs = {}
        for var, delta in zip(variables, deltas):
            mc, _ = self.calculate_marginal_cost(base_params, var, delta)
            marginal_costs[var] = mc
            
        return marginal_costs
    
    def find_optimal_inventory_level(self, demand_forecast: float,
                                   holding_cost_rate: float,
                                   ordering_cost: float,
                                   unit_cost: float) -> float:
        """
        Calculate optimal inventory level using Economic Order Quantity (EOQ) model
        
        Args:
            demand_forecast: Annual demand forecast
            holding_cost_rate: Annual holding cost rate (as fraction of unit cost)
            ordering_cost: Cost per order
            unit_cost: Cost per unit
            
        Returns:
            Optimal order quantity
        """
        # EOQ formula: sqrt(2 * D * S / H)
        # Where D = annual demand, S = ordering cost, H = holding cost per unit
        holding_cost_per_unit = unit_cost * holding_cost_rate
        if holding_cost_per_unit <= 0:
            return float('inf')  # Avoid division by zero
            
        eoq = np.sqrt(2 * demand_forecast * ordering_cost / holding_cost_per_unit)
        return eoq
    
    def calculate_cost_sensitivity(self, base_params: Dict,
                                 variable: str,
                                 range_percent: float = 0.2,
                                 steps: int = 11) -> Tuple[List[float], List[float]]:
        """
        Calculate cost sensitivity over a range of variable values
        
        Args:
            base_params: Base parameters for cost calculation
            variable: Variable to analyze
            range_percent: Range as percentage of base value (±range_percent)
            steps: Number of steps to calculate
            
        Returns:
            Tuple of (variable_values, total_costs)
        """
        # Get base value
        base_value = base_params.get(variable, 1.0)
        
        # Calculate range
        min_value = base_value * (1 - range_percent)
        max_value = base_value * (1 + range_percent)
        
        # Generate values
        values = np.linspace(min_value, max_value, steps)
        costs = []
        
        # Calculate costs for each value
        for value in values:
            params = base_params.copy()
            params[variable] = value
            cost_breakdown = self.cost_model.calculate_total_cost(**params)
            costs.append(cost_breakdown.total_cost)
            
        return values.tolist(), costs
    
    def optimize_route_costs(self, route_options: List[Dict]) -> Tuple[int, float]:
        """
        Find the route option with minimum cost
        
        Args:
            route_options: List of route parameter dictionaries
            
        Returns:
            Tuple of (best_option_index, minimum_cost)
        """
        min_cost = float('inf')
        best_index = -1
        
        for i, params in enumerate(route_options):
            cost_breakdown = self.cost_model.calculate_total_cost(**params)
            total_cost = cost_breakdown.total_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_index = i
                
        return best_index, min_cost
    
    def calculate_breakeven_point(self, fixed_costs: float,
                                selling_price: float,
                                variable_cost: float) -> float:
        """
        Calculate breakeven point in units
        
        Args:
            fixed_costs: Total fixed costs
            selling_price: Price per unit
            variable_cost: Variable cost per unit
            
        Returns:
            Breakeven point in units
        """
        contribution_margin = selling_price - variable_cost
        if contribution_margin <= 0:
            return float('inf')  # No breakeven point
            
        return fixed_costs / contribution_margin


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MarginalCostAnalyzer()
    
    # Base parameters for a shipment
    base_params = {
        'distance': 500.0,
        'weight': 1000.0,
        'inventory_value': 50000.0,
        'holding_days': 5.0,
        'delay_hours': 2.0,
        'shipment_value': 30000.0,
        'shortage_units': 5.0,
        'units_handled': 100.0,
        'labor_hours': 8.0
    }
    
    # Calculate marginal cost of distance
    marginal_distance_cost, cost_details = analyzer.calculate_marginal_cost(
        base_params, 'distance', 1.0  # 1 km increase
    )
    
    print(f"Marginal cost of 1 km: ${marginal_distance_cost:.2f}")
    print(f"New total cost: ${cost_details.total_cost:.2f}")
    
    # Calculate multi-dimensional marginal costs
    variables = ['distance', 'weight', 'holding_days']
    marginal_costs = analyzer.calculate_marginal_costs_multi_dimensional(
        base_params, variables
    )
    
    print("\nMulti-dimensional marginal costs:")
    for var, cost in marginal_costs.items():
        print(f"  {var}: ${cost:.2f} per unit")
    
    # Find optimal inventory level
    optimal_q = analyzer.find_optimal_inventory_level(
        demand_forecast=10000,
        holding_cost_rate=0.2,  # 20% annual holding cost
        ordering_cost=100,
        unit_cost=50
    )
    
    print(f"\nOptimal order quantity (EOQ): {optimal_q:.0f} units")
    
    # Compare route options
    route_options = [
        {'distance': 500, 'weight': 1000, 'labor_hours': 8},
        {'distance': 480, 'weight': 1100, 'labor_hours': 9},
        {'distance': 520, 'weight': 950, 'labor_hours': 7}
    ]
    
    best_route, min_cost = analyzer.optimize_route_costs(route_options)
    print(f"\nBest route option: #{best_route + 1} with cost ${min_cost:.2f}")