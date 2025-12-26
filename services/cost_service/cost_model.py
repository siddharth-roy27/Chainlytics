"""
Cost modeling for logistics operations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CostBreakdown:
    """Detailed breakdown of cost components"""
    transport_cost: float
    holding_cost: float
    delay_penalty: float
    shortage_cost: float
    handling_cost: float
    fuel_cost: float
    labor_cost: float
    maintenance_cost: float
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost"""
        return (
            self.transport_cost + self.holding_cost + self.delay_penalty +
            self.shortage_cost + self.handling_cost + self.fuel_cost +
            self.labor_cost + self.maintenance_cost
        )


class LogisticsCostModel:
    """Base class for logistics cost modeling"""
    
    def __init__(self):
        self.parameters = {}
        self.is_calibrated = False
    
    def calculate_transport_cost(self, distance: float, weight: float, 
                              vehicle_type: str = "standard") -> float:
        """Calculate transportation cost"""
        raise NotImplementedError("Subclasses must implement transport cost calculation")
    
    def calculate_holding_cost(self, inventory_value: float, 
                             holding_period_days: float) -> float:
        """Calculate inventory holding cost"""
        raise NotImplementedError("Subclasses must implement holding cost calculation")
    
    def calculate_delay_penalty(self, delay_hours: float, 
                              shipment_value: float) -> float:
        """Calculate delay penalty cost"""
        raise NotImplementedError("Subclasses must implement delay penalty calculation")
    
    def calculate_total_cost(self, **kwargs) -> CostBreakdown:
        """Calculate complete cost breakdown"""
        raise NotImplementedError("Subclasses must implement total cost calculation")


class LinearCostModel(LogisticsCostModel):
    """Linear cost model for logistics operations"""
    
    def __init__(self, config: Dict = None):
        super().__init__()
        if config is None:
            config = self._default_config()
        self.parameters = config
        self.is_calibrated = True
    
    def _default_config(self) -> Dict:
        """Default cost parameters"""
        return {
            'transport_rate_per_km': 2.5,
            'transport_rate_per_kg': 0.8,
            'holding_rate_daily': 0.005,  # 0.5% of inventory value per day
            'delay_penalty_hourly': 50.0,
            'shortage_cost_per_unit': 25.0,
            'handling_cost_per_unit': 2.0,
            'fuel_cost_per_km': 0.3,
            'labor_cost_per_hour': 35.0,
            'maintenance_cost_per_km': 0.1
        }
    
    def calculate_transport_cost(self, distance: float, weight: float,
                               vehicle_type: str = "standard") -> float:
        """Calculate transportation cost based on distance and weight"""
        base_cost = (
            distance * self.parameters['transport_rate_per_km'] +
            weight * self.parameters['transport_rate_per_kg']
        )
        
        # Adjust for vehicle type
        vehicle_multipliers = {
            'standard': 1.0,
            'express': 1.5,
            'economy': 0.8,
            'heavy': 2.0
        }
        
        multiplier = vehicle_multipliers.get(vehicle_type, 1.0)
        return base_cost * multiplier
    
    def calculate_holding_cost(self, inventory_value: float,
                             holding_period_days: float) -> float:
        """Calculate inventory holding cost"""
        return inventory_value * self.parameters['holding_rate_daily'] * holding_period_days
    
    def calculate_delay_penalty(self, delay_hours: float,
                              shipment_value: float) -> float:
        """Calculate delay penalty cost"""
        return delay_hours * self.parameters['delay_penalty_hourly']
    
    def calculate_shortage_cost(self, shortage_units: float) -> float:
        """Calculate cost of stockout/shortage"""
        return shortage_units * self.parameters['shortage_cost_per_unit']
    
    def calculate_handling_cost(self, units_handled: float) -> float:
        """Calculate goods handling cost"""
        return units_handled * self.parameters['handling_cost_per_unit']
    
    def calculate_fuel_cost(self, distance: float) -> float:
        """Calculate fuel cost component"""
        return distance * self.parameters['fuel_cost_per_km']
    
    def calculate_labor_cost(self, labor_hours: float) -> float:
        """Calculate labor cost component"""
        return labor_hours * self.parameters['labor_cost_per_hour']
    
    def calculate_maintenance_cost(self, distance: float) -> float:
        """Calculate vehicle maintenance cost"""
        return distance * self.parameters['maintenance_cost_per_km']
    
    def calculate_total_cost(self, distance: float = 0.0, weight: float = 0.0,
                           inventory_value: float = 0.0, holding_days: float = 0.0,
                           delay_hours: float = 0.0, shipment_value: float = 0.0,
                           shortage_units: float = 0.0, units_handled: float = 0.0,
                           labor_hours: float = 0.0, vehicle_type: str = "standard") -> CostBreakdown:
        """Calculate complete cost breakdown"""
        
        transport_cost = self.calculate_transport_cost(distance, weight, vehicle_type)
        holding_cost = self.calculate_holding_cost(inventory_value, holding_days)
        delay_penalty = self.calculate_delay_penalty(delay_hours, shipment_value)
        shortage_cost = self.calculate_shortage_cost(shortage_units)
        handling_cost = self.calculate_handling_cost(units_handled)
        fuel_cost = self.calculate_fuel_cost(distance)
        labor_cost = self.calculate_labor_cost(labor_hours)
        maintenance_cost = self.calculate_maintenance_cost(distance)
        
        return CostBreakdown(
            transport_cost=transport_cost,
            holding_cost=holding_cost,
            delay_penalty=delay_penalty,
            shortage_cost=shortage_cost,
            handling_cost=handling_cost,
            fuel_cost=fuel_cost,
            labor_cost=labor_cost,
            maintenance_cost=maintenance_cost
        )


# Example usage
if __name__ == "__main__":
    # Initialize cost model
    cost_model = LinearCostModel()
    
    # Calculate costs for a sample shipment
    costs = cost_model.calculate_total_cost(
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
    
    print("Cost Breakdown:")
    print(f"Transport Cost: ${costs.transport_cost:.2f}")
    print(f"Holding Cost: ${costs.holding_cost:.2f}")
    print(f"Delay Penalty: ${costs.delay_penalty:.2f}")
    print(f"Shortage Cost: ${costs.shortage_cost:.2f}")
    print(f"Handling Cost: ${costs.handling_cost:.2f}")
    print(f"Fuel Cost: ${costs.fuel_cost:.2f}")
    print(f"Labor Cost: ${costs.labor_cost:.2f}")
    print(f"Maintenance Cost: ${costs.maintenance_cost:.2f}")
    print(f"Total Cost: ${costs.total_cost:.2f}")