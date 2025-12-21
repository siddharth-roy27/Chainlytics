"""
Vehicle dynamics for logistics simulation
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class VehicleProfile:
    """Profile defining vehicle characteristics"""
    vehicle_type: str          # 'standard', 'express', 'heavy', 'eco'
    capacity: float           # Cargo capacity (units)
    speed: float             # Average speed (km/h)
    fuel_efficiency: float   # Fuel consumption (L/100km)
    maintenance_rate: float  # Maintenance needs per hour
    cost_per_km: float      # Operating cost per kilometer
    availability: float      # Base availability (0.0-1.0)


@dataclass
class TrafficCondition:
    """Traffic conditions affecting vehicle movement"""
    congestion_level: float   # 0.0 (free) to 1.0 (gridlock)
    road_quality: float      # 0.0 (poor) to 1.0 (excellent)
    weather_impact: float    # 0.0 (clear) to 1.0 (severe)
    construction_zones: List[Tuple[float, float]]  # Areas with construction


class VehicleDynamics:
    """Manage vehicle movement and behavior dynamics"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vehicle dynamics
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('VehicleDynamics')
        self.profiles = self._initialize_profiles()
        self.traffic_conditions = self._initialize_traffic_conditions()
        self.vehicle_states = {}  # Track individual vehicle states
    
    def _initialize_profiles(self) -> Dict[str, VehicleProfile]:
        """Initialize vehicle profiles from configuration"""
        profiles_config = self.config.get('vehicle_profiles', {})
        profiles = {}
        
        # Default profiles if none configured
        if not profiles_config:
            profiles_config = {
                'standard': {
                    'vehicle_type': 'standard',
                    'capacity': 1000.0,
                    'speed': 60.0,  # km/h
                    'fuel_efficiency': 8.0,  # L/100km
                    'maintenance_rate': 0.02,  # Maintenance per hour
                    'cost_per_km': 2.5,
                    'availability': 0.95
                },
                'express': {
                    'vehicle_type': 'express',
                    'capacity': 500.0,
                    'speed': 80.0,
                    'fuel_efficiency': 10.0,
                    'maintenance_rate': 0.03,
                    'cost_per_km': 3.5,
                    'availability': 0.90
                },
                'heavy': {
                    'vehicle_type': 'heavy',
                    'capacity': 5000.0,
                    'speed': 45.0,
                    'fuel_efficiency': 25.0,
                    'maintenance_rate': 0.05,
                    'cost_per_km': 5.0,
                    'availability': 0.85
                },
                'eco': {
                    'vehicle_type': 'eco',
                    'capacity': 800.0,
                    'speed': 50.0,
                    'fuel_efficiency': 4.0,
                    'maintenance_rate': 0.01,
                    'cost_per_km': 2.0,
                    'availability': 0.92
                }
            }
        
        for profile_name, profile_config in profiles_config.items():
            profiles[profile_name] = VehicleProfile(**profile_config)
        
        return profiles
    
    def _initialize_traffic_conditions(self) -> TrafficCondition:
        """Initialize traffic conditions"""
        traffic_config = self.config.get('traffic_conditions', {})
        
        return TrafficCondition(
            congestion_level=traffic_config.get('congestion_level', 0.3),
            road_quality=traffic_config.get('road_quality', 0.8),
            weather_impact=traffic_config.get('weather_impact', 0.1),
            construction_zones=traffic_config.get('construction_zones', [])
        )
    
    def move_vehicle(self, vehicle: Dict[str, Any], 
                    destination: Tuple[float, float],
                    timestep_duration: timedelta) -> Dict[str, Any]:
        """
        Move vehicle towards destination
        
        Args:
            vehicle: Vehicle dictionary
            destination: Destination coordinates (x, y)
            timestep_duration: Duration of timestep
            
        Returns:
            Updated vehicle dictionary
        """
        # Get vehicle profile
        profile = self.profiles.get(vehicle['type'], list(self.profiles.values())[0])
        
        # Calculate current position
        current_pos = (vehicle['current_location']['x'], vehicle['current_location']['y'])
        
        # Calculate distance to destination
        distance_to_dest = self._calculate_distance(current_pos, destination)
        
        if distance_to_dest < 0.1:  # Arrived at destination
            vehicle['current_location'] = {'x': destination[0], 'y': destination[1]}
            vehicle['status'] = 'at_destination'
            return vehicle
        
        # Calculate effective speed considering traffic conditions
        effective_speed = self._calculate_effective_speed(profile, current_pos, destination)
        
        # Calculate distance traveled in this timestep
        timestep_hours = timestep_duration.total_seconds() / 3600.0
        distance_traveled = effective_speed * timestep_hours
        
        # Update position
        if distance_traveled >= distance_to_dest:
            # Arrive at destination
            vehicle['current_location'] = {'x': destination[0], 'y': destination[1]}
            vehicle['status'] = 'at_destination'
        else:
            # Move towards destination
            direction_x = destination[0] - current_pos[0]
            direction_y = destination[1] - current_pos[1]
            distance = max(0.001, np.sqrt(direction_x**2 + direction_y**2))  # Avoid division by zero
            
            # Normalize and scale by distance traveled
            unit_x = direction_x / distance
            unit_y = direction_y / distance
            
            new_x = current_pos[0] + unit_x * distance_traveled
            new_y = current_pos[1] + unit_y * distance_traveled
            
            vehicle['current_location'] = {'x': new_x, 'y': new_y}
            vehicle['status'] = 'moving'
        
        # Update vehicle state
        self._update_vehicle_state(vehicle, timestep_duration)
        
        return vehicle
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Distance between positions
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_effective_speed(self, profile: VehicleProfile,
                                 current_pos: Tuple[float, float],
                                 destination: Tuple[float, float]) -> float:
        """
        Calculate effective speed considering traffic conditions
        
        Args:
            profile: Vehicle profile
            current_pos: Current position
            destination: Destination position
            
        Returns:
            Effective speed in km/h
        """
        base_speed = profile.speed
        
        # Apply traffic congestion factor
        congestion_factor = 1.0 - (self.traffic_conditions.congestion_level * 0.7)
        
        # Apply road quality factor
        road_factor = 0.5 + (self.traffic_conditions.road_quality * 0.5)
        
        # Apply weather impact
        weather_factor = 1.0 - (self.traffic_conditions.weather_impact * 0.4)
        
        # Check for construction zones
        construction_factor = 1.0
        for zone_start, zone_end in self.traffic_conditions.construction_zones:
            # Simple check if route passes through construction zone
            if (zone_start <= current_pos[0] <= zone_end or 
                zone_start <= destination[0] <= zone_end):
                construction_factor = 0.7  # 30% slower in construction zones
                break
        
        # Calculate effective speed
        effective_speed = (base_speed * congestion_factor * road_factor * 
                          weather_factor * construction_factor)
        
        # Ensure reasonable bounds
        effective_speed = max(5.0, min(base_speed, effective_speed))
        
        return effective_speed
    
    def _update_vehicle_state(self, vehicle: Dict[str, Any], 
                           timestep_duration: timedelta) -> None:
        """
        Update vehicle state (fuel, maintenance, costs)
        
        Args:
            vehicle: Vehicle dictionary
            timestep_duration: Duration of timestep
        """
        vehicle_id = vehicle['id']
        
        # Initialize vehicle state if not exists
        if vehicle_id not in self.vehicle_states:
            self.vehicle_states[vehicle_id] = {
                'fuel_level': 100.0,  # Percentage
                'maintenance_due': 0.0,  # Accumulated maintenance needs
                'total_distance': 0.0,
                'total_cost': 0.0,
                'operational_time': 0.0  # Hours
            }
        
        state = self.vehicle_states[vehicle_id]
        
        # Get profile
        profile = self.profiles.get(vehicle['type'], list(self.profiles.values())[0])
        
        # Update operational time
        timestep_hours = timestep_duration.total_seconds() / 3600.0
        state['operational_time'] += timestep_hours
        
        # Update maintenance needs
        state['maintenance_due'] += profile.maintenance_rate * timestep_hours
        
        # Update fuel (simplified - assumes vehicle consumes fuel while moving)
        if vehicle['status'] == 'moving':
            # Estimate fuel consumption based on distance and profile
            fuel_consumption_rate = profile.fuel_efficiency / 100.0  # L/km
            estimated_distance = profile.speed * timestep_hours * 0.5  # Assume half max speed on average
            fuel_consumed = fuel_consumption_rate * estimated_distance
            state['fuel_level'] = max(0.0, state['fuel_level'] - fuel_consumed)
        
        # Update costs
        if vehicle['status'] == 'moving':
            estimated_distance = profile.speed * timestep_hours * 0.5
            state['total_cost'] += estimated_distance * profile.cost_per_km
            state['total_distance'] += estimated_distance
    
    def check_vehicle_availability(self, vehicle_type: str) -> bool:
        """
        Check if vehicle of given type is available
        
        Args:
            vehicle_type: Type of vehicle to check
            
        Returns:
            True if vehicle is available, False otherwise
        """
        profile = self.profiles.get(vehicle_type)
        if not profile:
            return False
        
        # Base availability plus some randomness
        availability_probability = profile.availability * np.random.uniform(0.8, 1.2)
        return np.random.random() < min(1.0, availability_probability)
    
    def get_vehicle_performance(self, vehicle_id: str) -> Dict[str, float]:
        """
        Get performance metrics for a vehicle
        
        Args:
            vehicle_id: ID of vehicle
            
        Returns:
            Performance metrics dictionary
        """
        if vehicle_id not in self.vehicle_states:
            return {}
        
        state = self.vehicle_states[vehicle_id]
        profile = None
        
        # Find profile for this vehicle (would need vehicle type information)
        # For now, use default profile
        if self.profiles:
            profile = list(self.profiles.values())[0]
        
        metrics = state.copy()
        
        # Add calculated metrics
        if profile and state['total_distance'] > 0:
            metrics['average_cost_per_km'] = state['total_cost'] / state['total_distance']
            metrics['fuel_efficiency_actual'] = (
                (profile.fuel_efficiency * state['total_distance']) / 
                max(1.0, 100.0 - state['fuel_level'])  # Simplified
            )
        
        metrics['utilization_rate'] = min(1.0, state['operational_time'] / 24.0)  # Daily rate
        
        return metrics
    
    def schedule_maintenance(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """
        Schedule maintenance if needed
        
        Args:
            vehicle_id: ID of vehicle
            
        Returns:
            Maintenance schedule or None if not needed
        """
        if vehicle_id not in self.vehicle_states:
            return None
        
        state = self.vehicle_states[vehicle_id]
        
        # Check if maintenance is due (arbitrary threshold)
        if state['maintenance_due'] > 1.0:
            maintenance_schedule = {
                'vehicle_id': vehicle_id,
                'type': 'routine',
                'estimated_duration_hours': 2.0 + np.random.exponential(1.0),
                'cost': 200.0 + np.random.exponential(100.0),
                'scheduled_time': datetime.now() + timedelta(hours=1)
            }
            
            # Reset maintenance counter
            state['maintenance_due'] = 0.0
            
            self.logger.info(f"Scheduled maintenance for vehicle {vehicle_id}")
            return maintenance_schedule
        
        return None
    
    def refuel_vehicle(self, vehicle_id: str) -> bool:
        """
        Refuel vehicle if needed
        
        Args:
            vehicle_id: ID of vehicle
            
        Returns:
            True if refueled, False otherwise
        """
        if vehicle_id not in self.vehicle_states:
            return False
        
        state = self.vehicle_states[vehicle_id]
        
        # Refuel if fuel level is low
        if state['fuel_level'] < 20.0:
            state['fuel_level'] = 100.0
            state['total_cost'] += 150.0  # Refueling cost
            self.logger.info(f"Refueled vehicle {vehicle_id}")
            return True
        
        return False
    
    def get_fleet_statistics(self) -> Dict[str, Any]:
        """
        Get fleet-wide statistics
        
        Returns:
            Fleet statistics dictionary
        """
        if not self.vehicle_states:
            return {'total_vehicles': 0}
        
        stats = {
            'total_vehicles': len(self.vehicle_states),
            'vehicles_by_type': {},
            'average_fuel_level': 0.0,
            'total_maintenance_due': 0.0,
            'total_fleet_distance': 0.0,
            'total_fleet_cost': 0.0
        }
        
        fuel_levels = []
        maintenance_needs = []
        distances = []
        costs = []
        
        for vehicle_id, state in self.vehicle_states.items():
            fuel_levels.append(state['fuel_level'])
            maintenance_needs.append(state['maintenance_due'])
            distances.append(state['total_distance'])
            costs.append(state['total_cost'])
        
        if fuel_levels:
            stats['average_fuel_level'] = np.mean(fuel_levels)
        if maintenance_needs:
            stats['total_maintenance_due'] = np.sum(maintenance_needs)
        if distances:
            stats['total_fleet_distance'] = np.sum(distances)
        if costs:
            stats['total_fleet_cost'] = np.sum(costs)
        
        return stats


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = {
        'vehicle_profiles': {
            'standard': {
                'vehicle_type': 'standard',
                'capacity': 1000.0,
                'speed': 60.0,
                'fuel_efficiency': 8.0,
                'maintenance_rate': 0.02,
                'cost_per_km': 2.5,
                'availability': 0.95
            },
            'express': {
                'vehicle_type': 'express',
                'capacity': 500.0,
                'speed': 80.0,
                'fuel_efficiency': 10.0,
                'maintenance_rate': 0.03,
                'cost_per_km': 3.5,
                'availability': 0.90
            }
        },
        'traffic_conditions': {
            'congestion_level': 0.3,
            'road_quality': 0.8,
            'weather_impact': 0.1,
            'construction_zones': [(25.0, 30.0), (70.0, 75.0)]
        }
    }
    
    # Create vehicle dynamics
    vehicle_dynamics = VehicleDynamics(config)
    
    # Create sample vehicle
    sample_vehicle = {
        'id': 'VEH001',
        'type': 'standard',
        'current_location': {'x': 0.0, 'y': 0.0},
        'status': 'available',
        'capacity_utilization': 0.0
    }
    
    # Test vehicle movement
    destination = (50.0, 50.0)
    timestep = timedelta(minutes=30)
    
    print("Testing vehicle movement:")
    print(f"Initial position: {sample_vehicle['current_location']}")
    
    # Move vehicle for several timesteps
    for i in range(10):
        sample_vehicle = vehicle_dynamics.move_vehicle(
            sample_vehicle, destination, timestep
        )
        position = sample_vehicle['current_location']
        distance_to_dest = vehicle_dynamics._calculate_distance(
            (position['x'], position['y']), destination
        )
        print(f"Timestep {i+1}: Position ({position['x']:.2f}, {position['y']:.2f}), "
              f"Distance to dest: {distance_to_dest:.2f}, Status: {sample_vehicle['status']}")
        
        if sample_vehicle['status'] == 'at_destination':
            print("Vehicle arrived at destination!")
            break
    
    # Check vehicle performance
    performance = vehicle_dynamics.get_vehicle_performance('VEH001')
    print(f"\nVehicle performance: {performance}")
    
    # Check fleet statistics
    fleet_stats = vehicle_dynamics.get_fleet_statistics()
    print(f"Fleet statistics: {fleet_stats}")
    
    # Test maintenance scheduling
    maintenance = vehicle_dynamics.schedule_maintenance('VEH001')
    if maintenance:
        print(f"Maintenance scheduled: {maintenance}")
    else:
        print("No maintenance needed")
    
    # Test availability check
    available = vehicle_dynamics.check_vehicle_availability('standard')
    print(f"Standard vehicle available: {available}")
</file>