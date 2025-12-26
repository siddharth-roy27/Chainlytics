"""
Hybrid routing solver combining OR-Tools and RL refinement
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import time

from .ortools_solver import ORToolsRoutingSolver, RoutingProblem, RouteSolution
from .rl_refiner import RLRouteRefiner, RouteRefinementState


class HybridRoutingSolver:
    """Hybrid routing solver combining exact methods and reinforcement learning"""
    
    def __init__(self, time_limit: int = 30, rl_refinement_iterations: int = 50):
        """
        Initialize hybrid routing solver
        
        Args:
            time_limit: Time limit for exact solver in seconds
            rl_refinement_iterations: Number of RL refinement iterations
        """
        self.exact_solver = ORToolsRoutingSolver(time_limit)
        self.rl_refiner = RLRouteRefiner()
        self.rl_refinement_iterations = rl_refinement_iterations
        self.logger = logging.getLogger('HybridRoutingSolver')
    
    def solve(self, problem: RoutingProblem, 
              use_rl_refinement: bool = True) -> Optional[RouteSolution]:
        """
        Solve routing problem using hybrid approach
        
        Args:
            problem: Routing problem definition
            use_rl_refinement: Whether to use RL refinement
            
        Returns:
            Route solution or None if no solution found
        """
        start_time = time.time()
        
        try:
            # Step 1: Solve with exact method (OR-Tools)
            self.logger.info("Step 1: Solving with OR-Tools...")
            exact_solution = self.exact_solver.solve_cvrp(problem)
            
            if not exact_solution:
                self.logger.warning("OR-Tools failed to find solution")
                return None
            
            self.logger.info(f"OR-Tools solution found with objective: {exact_solution.objective_value}")
            
            # Step 2: RL refinement (optional)
            if use_rl_refinement:
                self.logger.info("Step 2: Applying RL refinement...")
                refined_solution = self.rl_refiner.refine_routes(
                    exact_solution, problem, self.rl_refinement_iterations
                )
                
                if refined_solution:
                    improvement = exact_solution.objective_value - refined_solution.objective_value
                    self.logger.info(f"RL refinement improved solution by: {improvement:.2f}")
                    return refined_solution
                else:
                    self.logger.warning("RL refinement failed, returning OR-Tools solution")
                    return exact_solution
            else:
                return exact_solution
                
        except Exception as e:
            self.logger.error(f"Error in hybrid solving: {str(e)}")
            return None
        
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Hybrid solving completed in {elapsed_time:.2f} seconds")
    
    def solve_with_adaptive_refinement(self, problem: RoutingProblem, 
                                     quality_threshold: float = 0.05) -> Optional[RouteSolution]:
        """
        Solve with adaptive RL refinement based on solution quality
        
        Args:
            problem: Routing problem definition
            quality_threshold: Threshold for applying refinement (fractional improvement)
            
        Returns:
            Route solution or None if no solution found
        """
        # Solve with exact method
        exact_solution = self.exact_solver.solve_cvrp(problem)
        if not exact_solution:
            return None
        
        # Estimate potential for improvement
        potential_improvement = self._estimate_improvement_potential(problem, exact_solution)
        
        # Apply RL refinement if potential is high
        if potential_improvement > quality_threshold:
            self.logger.info(f"High improvement potential ({potential_improvement:.3f}), applying RL refinement")
            refined_solution = self.rl_refiner.refine_routes(
                exact_solution, problem, self.rl_refinement_iterations
            )
            return refined_solution or exact_solution
        else:
            self.logger.info(f"Low improvement potential ({potential_improvement:.3f}), skipping RL refinement")
            return exact_solution
    
    def _estimate_improvement_potential(self, problem: RoutingProblem, 
                                      solution: RouteSolution) -> float:
        """
        Estimate potential for improvement in the solution
        
        Args:
            problem: Routing problem
            solution: Current solution
            
        Returns:
            Estimated improvement potential (0.0 to 1.0)
        """
        # Simple heuristic: potential based on route structure
        # More complex routes (more nodes, longer distances) have higher potential
        
        total_nodes = sum(len(route) for route in solution.routes) - len(solution.routes)  # Exclude depots
        avg_route_length = total_nodes / len(solution.routes) if solution.routes else 0
        
        # Potential increases with average route length
        # Normalize to [0, 1] range
        potential = min(1.0, avg_route_length / 20.0)  # Assume 20 nodes is high complexity
        
        return potential
    
    def solve_batch(self, problems: List[RoutingProblem], 
                   parallel: bool = False) -> List[Optional[RouteSolution]]:
        """
        Solve multiple routing problems
        
        Args:
            problems: List of routing problems
            parallel: Whether to solve in parallel (not implemented in this example)
            
        Returns:
            List of solutions (None for unsolved problems)
        """
        solutions = []
        
        for i, problem in enumerate(problems):
            self.logger.info(f"Solving problem {i+1}/{len(problems)}")
            solution = self.solve(problem)
            solutions.append(solution)
        
        return solutions


class SolutionQualityAssessor:
    """Assess quality of routing solutions"""
    
    def __init__(self):
        self.logger = logging.getLogger('SolutionQualityAssessor')
    
    def assess_solution(self, solution: RouteSolution, problem: RoutingProblem) -> Dict[str, float]:
        """
        Assess various quality metrics for a solution
        
        Args:
            solution: Route solution
            problem: Routing problem
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {}
        
        # 1. Objective value
        metrics['objective_value'] = solution.objective_value
        
        # 2. Route balance (how evenly distributed)
        if solution.vehicle_loads:
            load_variance = np.var(solution.vehicle_loads)
            metrics['load_balance'] = 1.0 / (1.0 + load_variance)  # Higher is better
        
        # 3. Route efficiency (directness of routes)
        efficiency = self._calculate_route_efficiency(solution, problem)
        metrics['route_efficiency'] = efficiency
        
        # 4. Capacity utilization
        if problem.vehicle_capacities and solution.vehicle_loads:
            utilizations = [
                load / capacity if capacity > 0 else 0 
                for load, capacity in zip(solution.vehicle_loads, problem.vehicle_capacities)
            ]
            metrics['avg_capacity_utilization'] = np.mean(utilizations)
            metrics['min_capacity_utilization'] = np.min(utilizations)
        
        # 5. Number of vehicles used
        metrics['vehicles_used'] = len([route for route in solution.routes if len(route) > 2])
        
        return metrics
    
    def _calculate_route_efficiency(self, solution: RouteSolution, 
                                 problem: RoutingProblem) -> float:
        """
        Calculate route efficiency (how direct the routes are)
        
        Args:
            solution: Route solution
            problem: Routing problem
            
        Returns:
            Efficiency score (0.0 to 1.0)
        """
        if not solution.routes or not problem.distance_matrix.size:
            return 0.0
        
        total_direct_distance = 0
        total_actual_distance = 0
        
        for route in solution.routes:
            if len(route) < 3:  # Need at least depot + one customer + depot
                continue
            
            # Direct distance from depot to all customers and back
            depot = route[0]
            customers = route[1:-1]  # Exclude depots
            
            for customer in customers:
                if (depot < len(problem.distance_matrix) and 
                    customer < len(problem.distance_matrix)):
                    total_direct_distance += 2 * problem.distance_matrix[depot][customer]
            
            # Actual route distance
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                if (from_node < len(problem.distance_matrix) and 
                    to_node < len(problem.distance_matrix)):
                    total_actual_distance += problem.distance_matrix[from_node][to_node]
        
        if total_actual_distance == 0:
            return 1.0
        
        # Efficiency = direct distance / actual distance (higher is better)
        efficiency = total_direct_distance / total_actual_distance if total_actual_distance > 0 else 0
        return min(1.0, efficiency)
    
    def compare_solutions(self, solution1: RouteSolution, solution2: RouteSolution,
                         problem: RoutingProblem) -> Dict[str, Any]:
        """
        Compare two solutions
        
        Args:
            solution1: First solution
            solution2: Second solution
            problem: Routing problem
            
        Returns:
            Comparison results
        """
        metrics1 = self.assess_solution(solution1, problem)
        metrics2 = self.assess_solution(solution2, problem)
        
        comparison = {
            'solution1_metrics': metrics1,
            'solution2_metrics': metrics2,
            'improvements': {}
        }
        
        # Calculate improvements
        for metric_name in metrics1:
            if metric_name in metrics2:
                improvement = metrics2[metric_name] - metrics1[metric_name]
                comparison['improvements'][metric_name] = improvement
        
        return comparison


class AdaptiveRoutingSolver:
    """Adaptive routing solver that learns from problem characteristics"""
    
    def __init__(self):
        self.hybrid_solver = HybridRoutingSolver()
        self.quality_assessor = SolutionQualityAssessor()
        self.problem_history = []
        self.logger = logging.getLogger('AdaptiveRoutingSolver')
    
    def solve_with_learning(self, problem: RoutingProblem) -> Optional[RouteSolution]:
        """
        Solve routing problem with learning from past problems
        
        Args:
            problem: Routing problem
            
        Returns:
            Route solution
        """
        # Analyze problem characteristics
        problem_features = self._extract_problem_features(problem)
        
        # Select appropriate solving strategy based on features
        strategy = self._select_strategy(problem_features)
        
        # Solve with selected strategy
        if strategy == 'exact_only':
            solution = self.hybrid_solver.exact_solver.solve_cvrp(problem)
        elif strategy == 'hybrid_standard':
            solution = self.hybrid_solver.solve(problem, use_rl_refinement=True)
        else:  # adaptive_refinement
            solution = self.hybrid_solver.solve_with_adaptive_refinement(problem)
        
        # Assess solution quality
        if solution:
            quality_metrics = self.quality_assessor.assess_solution(solution, problem)
            self.logger.info(f"Solution quality: {quality_metrics}")
            
            # Store problem and solution for learning
            self.problem_history.append({
                'problem_features': problem_features,
                'solution_metrics': quality_metrics,
                'strategy_used': strategy
            })
        
        return solution
    
    def _extract_problem_features(self, problem: RoutingProblem) -> Dict[str, float]:
        """
        Extract features from routing problem
        
        Args:
            problem: Routing problem
            
        Returns:
            Feature dictionary
        """
        num_customers = len(problem.demands) - 1  # Exclude depot
        total_demand = sum(problem.demands)
        avg_capacity = np.mean(problem.vehicle_capacities) if problem.vehicle_capacities else 1
        capacity_utilization = total_demand / (len(problem.vehicle_capacities) * avg_capacity) if problem.vehicle_capacities else 0
        
        # Distance matrix statistics
        if problem.distance_matrix.size > 0:
            avg_distance = np.mean(problem.distance_matrix[problem.distance_matrix > 0])
            max_distance = np.max(problem.distance_matrix)
        else:
            avg_distance = 0
            max_distance = 0
        
        return {
            'num_customers': num_customers,
            'total_demand': total_demand,
            'capacity_utilization': capacity_utilization,
            'avg_distance': avg_distance,
            'max_distance': max_distance,
            'num_vehicles': problem.num_vehicles
        }
    
    def _select_strategy(self, features: Dict[str, float]) -> str:
        """
        Select solving strategy based on problem features
        
        Args:
            features: Problem features
            
        Returns:
            Strategy name
        """
        num_customers = features['num_customers']
        capacity_utilization = features['capacity_utilization']
        
        # Simple rule-based strategy selection
        if num_customers < 20:
            # Small problems: exact method only
            return 'exact_only'
        elif num_customers < 100:
            # Medium problems: standard hybrid
            return 'hybrid_standard'
        else:
            # Large problems: adaptive refinement
            return 'adaptive_refinement'


# Example usage
if __name__ == "__main__":
    # Create sample problem
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    problem = RoutingProblem(
        num_vehicles=2,
        depot_index=0,
        distance_matrix=distance_matrix,
        demands=[0, 10, 20, 15],
        vehicle_capacities=[30, 30]
    )
    
    # Create hybrid solver
    hybrid_solver = HybridRoutingSolver(time_limit=10, rl_refinement_iterations=20)
    
    # Solve problem
    solution = hybrid_solver.solve(problem, use_rl_refinement=True)
    
    if solution:
        print("Hybrid solution found:")
        print(f"  Objective value: {solution.objective_value}")
        print(f"  Routes: {solution.routes}")
        print(f"  Vehicle loads: {solution.vehicle_loads}")
    else:
        print("No solution found")
    
    # Create adaptive solver
    adaptive_solver = AdaptiveRoutingSolver()
    
    # Solve with learning
    adaptive_solution = adaptive_solver.solve_with_learning(problem)
    
    if adaptive_solution:
        print("Adaptive solution found:")
        print(f"  Objective value: {adaptive_solution.objective_value}")
</file>