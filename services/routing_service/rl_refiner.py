"""
Reinforcement learning based route refinement
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

from .ortools_solver import RouteSolution, RoutingProblem


@dataclass
class RouteRefinementState:
    """State representation for route refinement"""
    current_routes: List[List[int]]
    distance_matrix: np.ndarray
    demands: List[int]
    vehicle_capacities: List[int]
    current_objective: float
    time_step: int
    max_time_steps: int = 100


@dataclass
class RouteRefinementAction:
    """Action for route refinement"""
    action_type: str  # 'swap', 'move', 'exchange', 'two_opt'
    route_indices: List[int]  # Which routes to modify
    node_indices: List[int]  # Which nodes to modify
    improvement: float = 0.0  # Expected improvement


class RouteRefinementNetwork(nn.Module):
    """Neural network for route refinement decisions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize refinement network
        
        Args:
            state_dim: Dimension of state representation
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
        """
        super(RouteRefinementNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Build layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Action head
        self.action_head = nn.Linear(input_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(input_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.hidden_layers(state)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        return action_logits, value


class RLRouteRefiner:
    """Reinforcement learning based route refiner"""
    
    def __init__(self, state_dim: int = 128, action_dim: int = 16, 
                 hidden_dims: List[int] = None, lr: float = 3e-4):
        """
        Initialize RL route refiner
        
        Args:
            state_dim: Dimension of state representation
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = RouteRefinementNetwork(state_dim, action_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.logger = logging.getLogger('RLRouteRefiner')
        
        # Action space definition
        self.action_types = ['swap', 'move', 'exchange', 'two_opt']
    
    def encode_state(self, state: RouteRefinementState) -> torch.Tensor:
        """
        Encode state to tensor representation
        
        Args:
            state: Route refinement state
            
        Returns:
            State tensor
        """
        # Simple encoding - in practice, you'd use more sophisticated encoding
        # that captures route structure, distances, capacities, etc.
        
        # Flatten routes and pad/truncate to fixed size
        flat_routes = []
        for route in state.current_routes:
            flat_routes.extend(route)
            # Add padding
            flat_routes.extend([0] * max(0, 20 - len(route)))  # Assume max 20 nodes per route
        
        # Truncate if too long
        flat_routes = flat_routes[:64]  # Assume 64 nodes max
        
        # Pad if too short
        flat_routes.extend([0] * max(0, 64 - len(flat_routes)))
        
        # Add scalar features
        features = flat_routes + [
            state.current_objective,
            state.time_step,
            state.max_time_steps,
            np.mean(state.demands) if state.demands else 0,
            np.mean(state.vehicle_capacities) if state.vehicle_capacities else 0
        ]
        
        # Pad to fixed size
        features.extend([0] * max(0, self.state_dim - len(features)))
        features = features[:self.state_dim]
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def select_action(self, state: RouteRefinementState, 
                     deterministic: bool = False) -> RouteRefinementAction:
        """
        Select refinement action using policy network
        
        Args:
            state: Current state
            deterministic: Whether to select deterministic action
            
        Returns:
            Selected action
        """
        # Encode state
        state_tensor = self.encode_state(state)
        
        # Get action probabilities
        with torch.no_grad():
            action_logits, _ = self.network(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
        
        # Select action
        if deterministic:
            action_idx = torch.argmax(action_probs, dim=-1).item()
        else:
            action_idx = torch.multinomial(action_probs, 1).item()
        
        # Decode action
        action = self._decode_action(action_idx, state)
        return action
    
    def _decode_action(self, action_idx: int, state: RouteRefinementState) -> RouteRefinementAction:
        """
        Decode action index to action object
        
        Args:
            action_idx: Action index
            state: Current state
            
        Returns:
            Route refinement action
        """
        # Simple decoding - in practice, you'd use more sophisticated decoding
        # that considers the actual route structure
        
        num_routes = len(state.current_routes)
        if num_routes == 0:
            return RouteRefinementAction('swap', [], [])
        
        # Select action type
        action_type_idx = action_idx % len(self.action_types)
        action_type = self.action_types[action_type_idx]
        
        # Select routes (simplified)
        route_indices = [action_idx % num_routes, (action_idx + 1) % num_routes]
        
        # Select nodes (simplified)
        node_indices = [action_idx % 10, (action_idx + 3) % 10]  # Max 10 nodes per route assumed
        
        return RouteRefinementAction(
            action_type=action_type,
            route_indices=route_indices,
            node_indices=node_indices
        )
    
    def refine_routes(self, initial_solution: RouteSolution, 
                     problem: RoutingProblem, 
                     max_iterations: int = 50) -> RouteSolution:
        """
        Refine routes using reinforcement learning
        
        Args:
            initial_solution: Initial route solution
            problem: Routing problem definition
            max_iterations: Maximum refinement iterations
            
        Returns:
            Refined route solution
        """
        current_solution = initial_solution
        current_objective = initial_solution.objective_value
        
        for iteration in range(max_iterations):
            # Create state
            state = RouteRefinementState(
                current_routes=current_solution.routes,
                distance_matrix=problem.distance_matrix,
                demands=problem.demands,
                vehicle_capacities=problem.vehicle_capacities,
                current_objective=current_objective,
                time_step=iteration,
                max_time_steps=max_iterations
            )
            
            # Select action
            action = self.select_action(state, deterministic=True)
            
            # Apply action (simplified)
            refined_solution = self._apply_action(current_solution, action, problem)
            
            # Check improvement
            if refined_solution and refined_solution.objective_value < current_objective:
                improvement = current_objective - refined_solution.objective_value
                action.improvement = improvement
                current_solution = refined_solution
                current_objective = refined_solution.objective_value
                
                self.logger.debug(f"Iteration {iteration}: Improved by {improvement:.2f}")
        
        return current_solution
    
    def _apply_action(self, solution: RouteSolution, action: RouteRefinementAction,
                     problem: RoutingProblem) -> Optional[RouteSolution]:
        """
        Apply refinement action to route solution
        
        Args:
            solution: Current route solution
            action: Refinement action
            problem: Routing problem
            
        Returns:
            Modified solution or None if invalid
        """
        # This is a simplified implementation
        # In practice, you would implement proper route modification operators
        
        try:
            # Create copy of routes
            new_routes = [route[:] for route in solution.routes]
            
            # Apply action based on type
            if action.action_type == 'swap' and len(action.route_indices) >= 2:
                route1_idx = action.route_indices[0] % len(new_routes)
                route2_idx = action.route_indices[1] % len(new_routes)
                
                if (len(new_routes[route1_idx]) > 2 and len(new_routes[route2_idx]) > 2 and
                    len(action.node_indices) >= 2):
                    node1_idx = action.node_indices[0] % len(new_routes[route1_idx])
                    node2_idx = action.node_indices[1] % len(new_routes[route2_idx])
                    
                    # Swap nodes (avoid depot)
                    if node1_idx > 0 and node2_idx > 0:
                        node1 = new_routes[route1_idx][node1_idx]
                        node2 = new_routes[route2_idx][node2_idx]
                        new_routes[route1_idx][node1_idx] = node2
                        new_routes[route2_idx][node2_idx] = node1
            
            # Recalculate objective (simplified)
            new_objective = self._calculate_objective(new_routes, problem)
            
            # Create new solution
            new_solution = RouteSolution(
                routes=new_routes,
                vehicle_loads=solution.vehicle_loads[:],  # Simplified
                total_distance=int(new_objective),
                total_time=solution.total_time,
                objective_value=new_objective
            )
            
            return new_solution
            
        except Exception as e:
            self.logger.warning(f"Failed to apply action: {str(e)}")
            return None
    
    def _calculate_objective(self, routes: List[List[int]], 
                           problem: RoutingProblem) -> float:
        """
        Calculate objective value for routes
        
        Args:
            routes: Route sequences
            problem: Routing problem
            
        Returns:
            Objective value
        """
        total_distance = 0
        
        for route in routes:
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                if (from_node < len(problem.distance_matrix) and 
                    to_node < len(problem.distance_matrix)):
                    total_distance += problem.distance_matrix[from_node][to_node]
        
        return float(total_distance)
    
    def update_network(self, states: List[torch.Tensor], 
                      actions: List[int], 
                      returns: List[float]) -> Dict[str, float]:
        """
        Update network parameters using collected experiences
        
        Args:
            states: List of state tensors
            actions: List of action indices
            returns: List of returns
            
        Returns:
            Training metrics
        """
        if not states:
            return {}
        
        # Convert to tensors
        states_tensor = torch.cat(states)
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = torch.FloatTensor(returns)
        
        # Forward pass
        action_logits, values = self.network(states_tensor)
        
        # Calculate losses
        # Policy loss
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions_tensor.unsqueeze(1))
        advantages = returns_tensor - values.squeeze()
        policy_loss = -(selected_log_probs.squeeze() * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns_tensor)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item()
        }


class RouteRefinementBuffer:
    """Buffer for storing route refinement experiences"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize experience buffer
        
        Args:
            capacity: Buffer capacity
        """
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.returns = []
        self.position = 0
    
    def add(self, state: torch.Tensor, action: int, return_val: float) -> None:
        """
        Add experience to buffer
        
        Args:
            state: State tensor
            action: Action index
            return_val: Return value
        """
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.returns.append(return_val)
        else:
            self.states[self.position] = state
            self.actions[self.position] = action
            self.returns[self.position] = return_val
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        """
        Sample batch of experiences
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (states, actions, returns)
        """
        if len(self.states) < batch_size:
            return self.states[:], self.actions[:], self.returns[:]
        
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        return (
            [self.states[i] for i in indices],
            [self.actions[i] for i in indices],
            [self.returns[i] for i in indices]
        )
    
    def clear(self) -> None:
        """Clear buffer"""
        self.states.clear()
        self.actions.clear()
        self.returns.clear()
        self.position = 0


# Example usage
if __name__ == "__main__":
    # Create sample data
    routes = [[0, 1, 2, 0], [0, 3, 0]]
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    # Create initial solution
    initial_solution = RouteSolution(
        routes=routes,
        vehicle_loads=[30, 20],
        total_distance=100,
        objective_value=100.0
    )
    
    # Create problem
    problem = RoutingProblem(
        num_vehicles=2,
        depot_index=0,
        distance_matrix=distance_matrix,
        demands=[0, 10, 20, 15],
        vehicle_capacities=[30, 30]
    )
    
    # Create refiner
    refiner = RLRouteRefiner(state_dim=128, action_dim=16)
    
    # Refine routes
    refined_solution = refiner.refine_routes(initial_solution, problem, max_iterations=10)
    
    print("Initial solution:")
    print(f"  Routes: {initial_solution.routes}")
    print(f"  Objective: {initial_solution.objective_value}")
    
    print("Refined solution:")
    print(f"  Routes: {refined_solution.routes}")
    print(f"  Objective: {refined_solution.objective_value}")
</file>