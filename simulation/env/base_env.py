"""
Base simulation environment for logistics operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta


@dataclass
class SimulationState:
    """Current state of the simulation"""
    timestep: int
    timestamp: datetime
    warehouses: Dict[str, Dict[str, Any]]
    vehicles: Dict[str, Dict[str, Any]]
    orders: Dict[str, Dict[str, Any]]
    customers: Dict[str, Dict[str, Any]]
    metrics: Dict[str, float]
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SimulationAction:
    """Action to be executed in the simulation"""
    action_type: str
    agent_id: str
    parameters: Dict[str, Any]
    timestamp: datetime


class BaseSimulationEnv(ABC):
    """Base class for simulation environments"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulation environment
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.current_state = None
        self.timestep = 0
        self.start_time = datetime.now()
        self.logger = logging.getLogger('BaseSimulationEnv')
        
        # Initialize environment
        self._initialize_environment()
    
    @abstractmethod
    def _initialize_environment(self) -> None:
        """Initialize the simulation environment"""
        pass
    
    @abstractmethod
    def reset(self) -> SimulationState:
        """
        Reset the simulation environment
        
        Returns:
            Initial simulation state
        """
        pass
    
    @abstractmethod
    def step(self, actions: List[SimulationAction]) -> Tuple[SimulationState, float, bool, Dict[str, Any]]:
        """
        Execute one step in the simulation
        
        Args:
            actions: List of actions to execute
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get the observation space definition
        
        Returns:
            Observation space specification
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """
        Get the action space definition
        
        Returns:
            Action space specification
        """
        pass
    
    def render(self, mode: str = 'human') -> Optional[Any]:
        """
        Render the current state of the simulation
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            Rendered output or None
        """
        if mode == 'human':
            self._print_state_summary()
        elif mode == 'json':
            return self._get_state_json()
    
    def _print_state_summary(self) -> None:
        """Print a summary of the current state"""
        if self.current_state:
            print(f"=== Simulation State (Timestep {self.current_state.timestep}) ===")
            print(f"Timestamp: {self.current_state.timestamp}")
            print(f"Active Warehouses: {len(self.current_state.warehouses)}")
            print(f"Active Vehicles: {len(self.current_state.vehicles)}")
            print(f"Pending Orders: {len(self.current_state.orders)}")
            print(f"Metrics: {self.current_state.metrics}")
            if self.current_state.events:
                print(f"Recent Events: {len(self.current_state.events)}")
    
    def _get_state_json(self) -> Dict[str, Any]:
        """
        Get current state as JSON-serializable dictionary
        
        Returns:
            State dictionary
        """
        if not self.current_state:
            return {}
        
        # Convert datetime objects to strings
        state_dict = {
            'timestep': self.current_state.timestep,
            'timestamp': self.current_state.timestamp.isoformat(),
            'warehouses': self.current_state.warehouses,
            'vehicles': self.current_state.vehicles,
            'orders': self.current_state.orders,
            'customers': self.current_state.customers,
            'metrics': self.current_state.metrics,
            'events': self.current_state.events
        }
        
        return state_dict
    
    def close(self) -> None:
        """Clean up simulation environment"""
        self.logger.info("Simulation environment closed")
    
    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set random seed for reproducibility
        
        Args:
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            self.logger.info(f"Random seed set to {seed}")


class LogisticsSimulationEnv(BaseSimulationEnv):
    """Concrete implementation of logistics simulation environment"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize logistics simulation environment
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
    
    def _initialize_environment(self) -> None:
        """Initialize the logistics simulation environment"""
        self.logger.info("Initializing logistics simulation environment")
        
        # Set up simulation parameters
        self.max_timesteps = self.config.get('max_timesteps', 1000)
        self.timestep_duration = timedelta(
            minutes=self.config.get('timestep_duration_minutes', 60)
        )
        
        # Initialize components
        self.warehouses = {}
        self.vehicles = {}
        self.orders = {}
        self.customers = {}
        self.metrics = {}
        
        # Set random seed if provided
        seed = self.config.get('random_seed')
        if seed is not None:
            self.seed(seed)
    
    def reset(self) -> SimulationState:
        """Reset the simulation environment"""
        self.logger.info("Resetting simulation environment")
        
        # Reset counters
        self.timestep = 0
        self.start_time = datetime.now()
        
        # Initialize components
        self._initialize_warehouses()
        self._initialize_vehicles()
        self._initialize_customers()
        self._initialize_orders()
        self._initialize_metrics()
        
        # Create initial state
        self.current_state = SimulationState(
            timestep=self.timestep,
            timestamp=self.start_time,
            warehouses=self.warehouses.copy(),
            vehicles=self.vehicles.copy(),
            orders=self.orders.copy(),
            customers=self.customers.copy(),
            metrics=self.metrics.copy(),
            events=[]
        )
        
        return self.current_state
    
    def _initialize_warehouses(self) -> None:
        """Initialize warehouse entities"""
        warehouse_config = self.config.get('warehouses', [])
        
        for i, wh_config in enumerate(warehouse_config):
            wh_id = wh_config.get('id', f'WH{i:03d}')
            self.warehouses[wh_id] = {
                'id': wh_id,
                'location': wh_config.get('location', {'x': 0, 'y': 0}),
                'capacity': wh_config.get('capacity', 10000),
                'current_inventory': wh_config.get('initial_inventory', 5000),
                'operational': True,
                'last_updated': self.start_time
            }
        
        # If no warehouses configured, create defaults
        if not self.warehouses:
            for i in range(3):
                wh_id = f'WH{i:03d}'
                self.warehouses[wh_id] = {
                    'id': wh_id,
                    'location': {'x': np.random.randint(0, 100), 'y': np.random.randint(0, 100)},
                    'capacity': 10000,
                    'current_inventory': np.random.randint(3000, 8000),
                    'operational': True,
                    'last_updated': self.start_time
                }
    
    def _initialize_vehicles(self) -> None:
        """Initialize vehicle entities"""
        vehicle_config = self.config.get('vehicles', [])
        
        for i, veh_config in enumerate(vehicle_config):
            veh_id = veh_config.get('id', f'VEH{i:03d}')
            self.vehicles[veh_id] = {
                'id': veh_id,
                'type': veh_config.get('type', 'standard'),
                'capacity': veh_config.get('capacity', 1000),
                'current_location': veh_config.get('initial_location', {'x': 0, 'y': 0}),
                'status': 'available',  # available, busy, maintenance
                'assigned_orders': [],
                'last_updated': self.start_time
            }
        
        # If no vehicles configured, create defaults
        if not self.vehicles:
            vehicle_types = ['standard', 'express', 'heavy']
            for i in range(5):
                veh_id = f'VEH{i:03d}'
                veh_type = np.random.choice(vehicle_types)
                capacity = 500 if veh_type == 'standard' else (1000 if veh_type == 'express' else 2000)
                self.vehicles[veh_id] = {
                    'id': veh_id,
                    'type': veh_type,
                    'capacity': capacity,
                    'current_location': {'x': np.random.randint(0, 100), 'y': np.random.randint(0, 100)},
                    'status': 'available',
                    'assigned_orders': [],
                    'last_updated': self.start_time
                }
    
    def _initialize_customers(self) -> None:
        """Initialize customer entities"""
        customer_config = self.config.get('customers', [])
        
        for i, cust_config in enumerate(customer_config):
            cust_id = cust_config.get('id', f'CUST{i:04d}')
            self.customers[cust_id] = {
                'id': cust_id,
                'location': cust_config.get('location', {'x': np.random.randint(0, 100), 'y': np.random.randint(0, 100)}),
                'priority': cust_config.get('priority', 1),
                'service_level_agreement': cust_config.get('sla', 24),  # hours
                'last_updated': self.start_time
            }
        
        # If no customers configured, create defaults
        if not self.customers:
            for i in range(20):
                cust_id = f'CUST{i:04d}'
                self.customers[cust_id] = {
                    'id': cust_id,
                    'location': {'x': np.random.randint(0, 100), 'y': np.random.randint(0, 100)},
                    'priority': np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1]),
                    'service_level_agreement': np.random.choice([24, 48, 72]),
                    'last_updated': self.start_time
                }
    
    def _initialize_orders(self) -> None:
        """Initialize order entities"""
        # Orders are typically generated dynamically during simulation
        self.orders = {}
    
    def _initialize_metrics(self) -> None:
        """Initialize performance metrics"""
        self.metrics = {
            'total_orders': 0,
            'fulfilled_orders': 0,
            'delayed_orders': 0,
            'total_cost': 0.0,
            'average_delivery_time': 0.0,
            'customer_satisfaction': 1.0,
            'resource_utilization': 0.0
        }
    
    def step(self, actions: List[SimulationAction]) -> Tuple[SimulationState, float, bool, Dict[str, Any]]:
        """
        Execute one step in the simulation
        
        Args:
            actions: List of actions to execute
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Execute actions
        self._execute_actions(actions)
        
        # Advance time
        self.timestep += 1
        current_timestamp = self.start_time + (self.timestep * self.timestep_duration)
        
        # Update entities
        self._update_warehouses()
        self._update_vehicles()
        self._update_orders()
        self._update_customers()
        self._update_metrics()
        
        # Generate new orders
        self._generate_new_orders()
        
        # Check termination conditions
        done = self.timestep >= self.max_timesteps
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Create events log
        events = self._generate_events(actions)
        
        # Create new state
        self.current_state = SimulationState(
            timestep=self.timestep,
            timestamp=current_timestamp,
            warehouses=self.warehouses.copy(),
            vehicles=self.vehicles.copy(),
            orders=self.orders.copy(),
            customers=self.customers.copy(),
            metrics=self.metrics.copy(),
            events=events
        )
        
        # Info dictionary
        info = {
            'timestep': self.timestep,
            'actions_executed': len(actions),
            'new_orders_generated': len([e for e in events if e.get('type') == 'order_created'])
        }
        
        return self.current_state, reward, done, info
    
    def _execute_actions(self, actions: List[SimulationAction]) -> None:
        """Execute simulation actions"""
        for action in actions:
            try:
                if action.action_type == 'assign_vehicle':
                    self._assign_vehicle(action)
                elif action.action_type == 'route_vehicle':
                    self._route_vehicle(action)
                elif action.action_type == 'process_order':
                    self._process_order(action)
                elif action.action_type == 'restock_warehouse':
                    self._restock_warehouse(action)
                else:
                    self.logger.warning(f"Unknown action type: {action.action_type}")
            except Exception as e:
                self.logger.error(f"Error executing action {action.action_type}: {str(e)}")
    
    def _assign_vehicle(self, action: SimulationAction) -> None:
        """Assign vehicle to orders"""
        vehicle_id = action.parameters.get('vehicle_id')
        order_ids = action.parameters.get('order_ids', [])
        
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            if vehicle['status'] == 'available':
                vehicle['status'] = 'busy'
                vehicle['assigned_orders'].extend(order_ids)
                self.logger.info(f"Assigned vehicle {vehicle_id} to {len(order_ids)} orders")
    
    def _route_vehicle(self, action: SimulationAction) -> None:
        """Route vehicle to destinations"""
        vehicle_id = action.parameters.get('vehicle_id')
        destination = action.parameters.get('destination')
        
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            vehicle['current_location'] = destination
            self.logger.info(f"Routed vehicle {vehicle_id} to {destination}")
    
    def _process_order(self, action: SimulationAction) -> None:
        """Process order fulfillment"""
        order_id = action.parameters.get('order_id')
        warehouse_id = action.parameters.get('warehouse_id')
        
        if order_id in self.orders and warehouse_id in self.warehouses:
            order = self.orders[order_id]
            warehouse = self.warehouses[warehouse_id]
            
            # Check inventory
            if warehouse['current_inventory'] >= order['quantity']:
                # Fulfill order
                warehouse['current_inventory'] -= order['quantity']
                order['status'] = 'fulfilled'
                order['fulfilled_time'] = self.current_state.timestamp if self.current_state else datetime.now()
                self.metrics['fulfilled_orders'] += 1
                self.logger.info(f"Fulfilled order {order_id} from warehouse {warehouse_id}")
            else:
                # Insufficient inventory
                order['status'] = 'delayed'
                self.metrics['delayed_orders'] += 1
                self.logger.warning(f"Delayed order {order_id} due to insufficient inventory at {warehouse_id}")
    
    def _restock_warehouse(self, action: SimulationAction) -> None:
        """Restock warehouse inventory"""
        warehouse_id = action.parameters.get('warehouse_id')
        quantity = action.parameters.get('quantity', 0)
        
        if warehouse_id in self.warehouses:
            warehouse = self.warehouses[warehouse_id]
            warehouse['current_inventory'] = min(
                warehouse['current_inventory'] + quantity,
                warehouse['capacity']
            )
            self.logger.info(f"Restocked warehouse {warehouse_id} with {quantity} units")
    
    def _update_warehouses(self) -> None:
        """Update warehouse states"""
        for warehouse in self.warehouses.values():
            warehouse['last_updated'] = self.current_state.timestamp if self.current_state else datetime.now()
    
    def _update_vehicles(self) -> None:
        """Update vehicle states"""
        for vehicle in self.vehicles.values():
            vehicle['last_updated'] = self.current_state.timestamp if self.current_state else datetime.now()
            
            # Simple vehicle movement simulation
            if vehicle['status'] == 'busy' and vehicle['assigned_orders']:
                # Move vehicle towards destination (simplified)
                current_loc = vehicle['current_location']
                # Move randomly for simulation purposes
                vehicle['current_location'] = {
                    'x': max(0, min(100, current_loc['x'] + np.random.randint(-5, 6))),
                    'y': max(0, min(100, current_loc['y'] + np.random.randint(-5, 6)))
                }
    
    def _update_orders(self) -> None:
        """Update order states"""
        current_time = self.current_state.timestamp if self.current_state else datetime.now()
        
        for order in self.orders.values():
            order['last_updated'] = current_time
            
            # Check for order delays
            if order['status'] == 'pending':
                order_creation_time = order.get('creation_time', current_time)
                time_elapsed = (current_time - order_creation_time).total_seconds() / 3600  # hours
                
                if time_elapsed > order.get('sla_hours', 24):
                    order['status'] = 'delayed'
                    self.metrics['delayed_orders'] += 1
    
    def _update_customers(self) -> None:
        """Update customer states"""
        current_time = self.current_state.timestamp if self.current_state else datetime.now()
        
        for customer in self.customers.values():
            customer['last_updated'] = current_time
    
    def _update_metrics(self) -> None:
        """Update performance metrics"""
        # Update resource utilization
        total_vehicles = len(self.vehicles)
        busy_vehicles = len([v for v in self.vehicles.values() if v['status'] == 'busy'])
        self.metrics['resource_utilization'] = busy_vehicles / total_vehicles if total_vehicles > 0 else 0
        
        # Update customer satisfaction (simplified)
        total_orders = self.metrics['total_orders']
        delayed_orders = self.metrics['delayed_orders']
        if total_orders > 0:
            self.metrics['customer_satisfaction'] = 1.0 - (delayed_orders / total_orders)
    
    def _generate_new_orders(self) -> None:
        """Generate new orders for current timestep"""
        # Simple order generation model
        num_new_orders = np.random.poisson(3)  # Average 3 orders per timestep
        
        for i in range(num_new_orders):
            order_id = f"ORDER_{self.timestep:04d}_{i:02d}"
            customer_id = np.random.choice(list(self.customers.keys()))
            customer = self.customers[customer_id]
            
            self.orders[order_id] = {
                'id': order_id,
                'customer_id': customer_id,
                'quantity': np.random.randint(1, 50),
                'priority': customer['priority'],
                'sla_hours': customer['service_level_agreement'],
                'creation_time': self.current_state.timestamp if self.current_state else datetime.now(),
                'status': 'pending',
                'last_updated': self.current_state.timestamp if self.current_state else datetime.now()
            }
            
            self.metrics['total_orders'] += 1
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward for current timestep
        
        Returns:
            Reward value
        """
        # Simple reward function based on key metrics
        fulfilled_ratio = (self.metrics['fulfilled_orders'] / 
                          max(1, self.metrics['total_orders']))
        delay_penalty = -(self.metrics['delayed_orders'] / 
                         max(1, self.metrics['total_orders']))
        cost_efficiency = -self.metrics['total_cost'] / 10000  # Normalize cost
        satisfaction_bonus = self.metrics['customer_satisfaction']
        
        reward = (fulfilled_ratio * 0.4 + 
                 delay_penalty * 0.3 + 
                 cost_efficiency * 0.2 + 
                 satisfaction_bonus * 0.1)
        
        return reward
    
    def _generate_events(self, actions: List[SimulationAction]) -> List[Dict[str, Any]]:
        """
        Generate events log for current timestep
        
        Args:
            actions: Actions executed in current timestep
            
        Returns:
            List of events
        """
        events = []
        current_time = self.current_state.timestamp if self.current_state else datetime.now()
        
        # Add action events
        for action in actions:
            events.append({
                'type': 'action_executed',
                'action_type': action.action_type,
                'agent_id': action.agent_id,
                'timestamp': current_time.isoformat(),
                'parameters': action.parameters
            })
        
        # Add order creation events
        new_orders = [oid for oid in self.orders.keys() 
                     if self.orders[oid].get('creation_time', current_time) == current_time]
        for order_id in new_orders:
            events.append({
                'type': 'order_created',
                'order_id': order_id,
                'timestamp': current_time.isoformat()
            })
        
        # Add fulfillment events
        fulfilled_orders = [oid for oid in self.orders.keys() 
                           if self.orders[oid].get('status') == 'fulfilled' and
                           self.orders[oid].get('fulfilled_time', current_time) == current_time]
        for order_id in fulfilled_orders:
            events.append({
                'type': 'order_fulfilled',
                'order_id': order_id,
                'timestamp': current_time.isoformat()
            })
        
        return events
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space definition"""
        return {
            'warehouses': {
                'count': len(self.warehouses),
                'features': ['id', 'location', 'capacity', 'current_inventory', 'operational']
            },
            'vehicles': {
                'count': len(self.vehicles),
                'features': ['id', 'type', 'capacity', 'current_location', 'status']
            },
            'orders': {
                'count': len(self.orders),
                'features': ['id', 'customer_id', 'quantity', 'priority', 'status']
            },
            'metrics': {
                'features': list(self.metrics.keys())
            }
        }
    
    def get_action_space(self) -> Dict[str, Any]:
        """Get action space definition"""
        return {
            'action_types': [
                'assign_vehicle',
                'route_vehicle', 
                'process_order',
                'restock_warehouse'
            ],
            'parameters': {
                'assign_vehicle': ['vehicle_id', 'order_ids'],
                'route_vehicle': ['vehicle_id', 'destination'],
                'process_order': ['order_id', 'warehouse_id'],
                'restock_warehouse': ['warehouse_id', 'quantity']
            }
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = {
        'max_timesteps': 100,
        'timestep_duration_minutes': 60,
        'random_seed': 42,
        'warehouses': [
            {'id': 'WH001', 'location': {'x': 10, 'y': 20}, 'capacity': 15000, 'initial_inventory': 8000},
            {'id': 'WH002', 'location': {'x': 80, 'y': 60}, 'capacity': 12000, 'initial_inventory': 6000}
        ],
        'vehicles': [
            {'id': 'VEH001', 'type': 'standard', 'capacity': 1000, 'initial_location': {'x': 10, 'y': 20}},
            {'id': 'VEH002', 'type': 'express', 'capacity': 1500, 'initial_location': {'x': 80, 'y': 60}}
        ]
    }
    
    # Create simulation environment
    env = LogisticsSimulationEnv(config)
    
    # Reset environment
    initial_state = env.reset()
    print("Initial state created")
    print(f"Warehouses: {len(initial_state.warehouses)}")
    print(f"Vehicles: {len(initial_state.vehicles)}")
    print(f"Customers: {len(initial_state.customers)}")
    
    # Run a few simulation steps
    for step in range(5):
        # Create dummy actions
        actions = [
            SimulationAction(
                action_type='assign_vehicle',
                agent_id='dispatcher',
                parameters={'vehicle_id': 'VEH001', 'order_ids': ['ORDER_TEST']},
                timestamp=datetime.now()
            )
        ]
        
        # Execute step
        next_state, reward, done, info = env.step(actions)
        
        print(f"\nStep {step + 1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
        print(f"  Metrics: {next_state.metrics}")
    
    # Render final state
    env.render(mode='human')
    
    # Close environment
    env.close()
    print("Simulation completed")
</file>