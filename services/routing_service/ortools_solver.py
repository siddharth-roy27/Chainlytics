"""
OR-Tools based routing solver for logistics optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging

# Try to import OR-Tools
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    OR_TOOLS_AVAILABLE = True
except ImportError:
    OR_TOOLS_AVAILABLE = False
    print("OR-Tools not available. Install with: pip install ortools")


@dataclass
class RoutingProblem:
    """Routing problem definition"""
    num_vehicles: int
    depot_index: int
    distance_matrix: np.ndarray
    demands: List[int]
    vehicle_capacities: List[int]
    time_windows: List[Tuple[int, int]] = field(default_factory=list)
    service_times: List[int] = field(default_factory=list)
    pickup_deliveries: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class RouteSolution:
    """Solution to a routing problem"""
    routes: List[List[int]]
    vehicle_loads: List[int]
    total_distance: int
    total_time: int = 0
    objective_value: float = 0.0


class ORToolsRoutingSolver:
    """Routing solver using Google OR-Tools"""
    
    def __init__(self, time_limit: int = 30):
        """
        Initialize OR-Tools routing solver
        
        Args:
            time_limit: Time limit for solver in seconds
        """
        if not OR_TOOLS_AVAILABLE:
            raise ImportError("OR-Tools is not available. Please install it with: pip install ortools")
        
        self.time_limit = time_limit
        self.logger = logging.getLogger('ORToolsRoutingSolver')
    
    def solve_cvrp(self, problem: RoutingProblem) -> Optional[RouteSolution]:
        """
        Solve Capacitated Vehicle Routing Problem
        
        Args:
            problem: Routing problem definition
            
        Returns:
            Route solution or None if no solution found
        """
        try:
            # Create routing model
            manager = pywrapcp.RoutingIndexManager(
                len(problem.distance_matrix),
                problem.num_vehicles,
                problem.depot_index
            )
            
            routing = pywrapcp.RoutingModel(manager)
            
            # Create distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return problem.distance_matrix[from_node][to_node]
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Add capacity constraints
            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return problem.demands[from_node]
            
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                problem.vehicle_capacities,  # vehicle maximum capacities
                True,  # start cumul to zero
                'Capacity'
            )
            
            # Add time windows if provided
            if problem.time_windows:
                self._add_time_windows(routing, manager, problem)
            
            # Add pickup and delivery constraints if provided
            if problem.pickup_deliveries:
                self._add_pickup_delivery_constraints(routing, manager, problem)
            
            # Set search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.FromSeconds(self.time_limit)
            search_parameters.log_search = False
            
            # Solve the problem
            solution = routing.SolveWithParameters(search_parameters)
            
            if solution:
                return self._extract_solution(solution, routing, manager, problem)
            else:
                self.logger.warning("No solution found for CVRP")
                return None
                
        except Exception as e:
            self.logger.error(f"Error solving CVRP: {str(e)}")
            return None
    
    def _add_time_windows(self, routing, manager, problem: RoutingProblem) -> None:
        """Add time window constraints to the routing model"""
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return problem.distance_matrix[from_node][to_node]
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        routing.AddDimension(
            time_callback_index,
            30,  # allow waiting time
            30000,  # maximum time per vehicle
            False,  # Don't force start cumul to zero
            'Time'
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add time window constraints for each location
        for location_idx, (start, end) in enumerate(problem.time_windows):
            index = manager.NodeToIndex(location_idx)
            if index != -1:  # Skip if location is not in the model
                time_dimension.CumulVar(index).SetRange(start, end)
    
    def _add_pickup_delivery_constraints(self, routing, manager, problem: RoutingProblem) -> None:
        """Add pickup and delivery constraints"""
        for pickup, delivery in problem.pickup_deliveries:
            pickup_index = manager.NodeToIndex(pickup)
            delivery_index = manager.NodeToIndex(delivery)
            if pickup_index != -1 and delivery_index != -1:
                routing.AddPickupAndDelivery(pickup_index, delivery_index)
                routing.solver().Add(
                    routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
                )
                routing.solver().Add(
                    routing.NextVar(pickup_index) != delivery_index  # Not directly connected
                )
    
    def _extract_solution(self, solution, routing, manager, problem: RoutingProblem) -> RouteSolution:
        """Extract solution from OR-Tools result"""
        routes = []
        vehicle_loads = []
        total_distance = 0
        total_time = 0
        
        for vehicle_id in range(problem.num_vehicles):
            route = []
            vehicle_load = 0
            
            index = routing.Start(vehicle_id)
            route.append(manager.IndexToNode(index))
            
            while not routing.IsEnd(index):
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                node = manager.IndexToNode(index)
                route.append(node)
                
                # Accumulate distance
                if not routing.IsEnd(index):
                    total_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            routes.append(route)
            
            # Calculate vehicle load
            capacity_dimension = routing.GetDimensionOrDie('Capacity')
            vehicle_load = solution.Value(capacity_dimension.CumulVar(routing.End(vehicle_id)))
            vehicle_loads.append(vehicle_load)
        
        return RouteSolution(
            routes=routes,
            vehicle_loads=vehicle_loads,
            total_distance=total_distance,
            total_time=total_time,
            objective_value=solution.ObjectiveValue()
        )
    
    def solve_vrptw(self, problem: RoutingProblem) -> Optional[RouteSolution]:
        """
        Solve Vehicle Routing Problem with Time Windows
        
        Args:
            problem: Routing problem with time windows
            
        Returns:
            Route solution or None if no solution found
        """
        # VRPTW is handled by the CVRP solver with time windows added
        return self.solve_cvrp(problem)
    
    def solve_pdptw(self, problem: RoutingProblem) -> Optional[RouteSolution]:
        """
        Solve Pickup and Delivery Problem with Time Windows
        
        Args:
            problem: Routing problem with pickup-delivery pairs and time windows
            
        Returns:
            Route solution or None if no solution found
        """
        # PDPTW is handled by the CVRP solver with pickup-delivery constraints
        return self.solve_cvrp(problem)


class RoutingOptimizer:
    """High-level routing optimizer that handles different problem types"""
    
    def __init__(self, solver_type: str = "ortools", time_limit: int = 30):
        """
        Initialize routing optimizer
        
        Args:
            solver_type: Type of solver to use ("ortools")
            time_limit: Time limit for solver in seconds
        """
        self.solver_type = solver_type
        self.time_limit = time_limit
        
        if solver_type == "ortools":
            self.solver = ORToolsRoutingSolver(time_limit)
        else:
            raise ValueError(f"Unsupported solver type: {solver_type}")
    
    def optimize_routes(self, problem_definition: Dict[str, Any]) -> Optional[RouteSolution]:
        """
        Optimize routes based on problem definition
        
        Args:
            problem_definition: Dictionary with problem parameters
            
        Returns:
            Route solution or None if no solution found
        """
        try:
            # Convert problem definition to RoutingProblem
            problem = self._parse_problem_definition(problem_definition)
            
            # Solve based on problem type
            problem_type = problem_definition.get('problem_type', 'cvrp')
            
            if problem_type == 'cvrp':
                return self.solver.solve_cvrp(problem)
            elif problem_type == 'vrptw':
                return self.solver.solve_vrptw(problem)
            elif problem_type == 'pdptw':
                return self.solver.solve_pdptw(problem)
            else:
                # Default to CVRP
                return self.solver.solve_cvrp(problem)
                
        except Exception as e:
            logging.error(f"Error optimizing routes: {str(e)}")
            return None
    
    def _parse_problem_definition(self, problem_def: Dict[str, Any]) -> RoutingProblem:
        """Parse problem definition dictionary into RoutingProblem object"""
        return RoutingProblem(
            num_vehicles=problem_def['num_vehicles'],
            depot_index=problem_def['depot_index'],
            distance_matrix=np.array(problem_def['distance_matrix']),
            demands=problem_def['demands'],
            vehicle_capacities=problem_def['vehicle_capacities'],
            time_windows=problem_def.get('time_windows', []),
            service_times=problem_def.get('service_times', []),
            pickup_deliveries=problem_def.get('pickup_deliveries', [])
        )


# Example usage
if __name__ == "__main__":
    if OR_TOOLS_AVAILABLE:
        # Create sample problem
        distance_matrix = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]
        
        problem_def = {
            'problem_type': 'cvrp',
            'num_vehicles': 2,
            'depot_index': 0,
            'distance_matrix': distance_matrix,
            'demands': [0, 10, 20, 15],
            'vehicle_capacities': [30, 30]
        }
        
        # Create optimizer
        optimizer = RoutingOptimizer("ortools", time_limit=10)
        
        # Solve problem
        solution = optimizer.optimize_routes(problem_def)
        
        if solution:
            print("Solution found:")
            print(f"Total distance: {solution.total_distance}")
            print(f"Routes: {solution.routes}")
            print(f"Vehicle loads: {solution.vehicle_loads}")
        else:
            print("No solution found")
    else:
        print("OR-Tools not available. Skipping example.")
</file>