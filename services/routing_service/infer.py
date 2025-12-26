"""
Inference module for routing service
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import logging
from datetime import datetime

from .ortools_solver import ORToolsRoutingSolver, RoutingProblem, RouteSolution
from .rl_refiner import RLRouteRefiner
from .hybrid import HybridRoutingSolver, AdaptiveRoutingSolver


class RoutingInferenceEngine:
    """Inference engine for routing decisions"""
    
    def __init__(self, solver_type: str = "hybrid"):
        """
        Initialize routing inference engine
        
        Args:
            solver_type: Type of solver to use ("ortools", "rl", "hybrid", "adaptive")
        """
        self.solver_type = solver_type
        self.logger = logging.getLogger('RoutingInferenceEngine')
        
        # Initialize appropriate solver
        if solver_type == "ortools":
            self.solver = ORToolsRoutingSolver(time_limit=30)
        elif solver_type == "rl":
            self.solver = RLRouteRefiner()
        elif solver_type == "hybrid":
            self.solver = HybridRoutingSolver()
        elif solver_type == "adaptive":
            self.solver = AdaptiveRoutingSolver()
        else:
            raise ValueError(f"Unsupported solver type: {solver_type}")
    
    def solve_routing_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve routing problem and return structured results
        
        Args:
            problem_data: Dictionary containing problem definition
            
        Returns:
            Dictionary with solution and metadata
        """
        try:
            start_time = datetime.now()
            
            # Convert problem data to RoutingProblem
            problem = self._parse_problem_data(problem_data)
            
            # Solve problem based on solver type
            solution = self._solve_problem(problem)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Format results
            if solution:
                result = {
                    'success': True,
                    'solution': self._format_solution(solution),
                    'processing_time': processing_time,
                    'solver_type': self.solver_type,
                    'timestamp': start_time.isoformat()
                }
            else:
                result = {
                    'success': False,
                    'error': 'No solution found',
                    'processing_time': processing_time,
                    'solver_type': self.solver_type,
                    'timestamp': start_time.isoformat()
                }
            
            self.logger.info(f"Routing problem solved in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error solving routing problem: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'solver_type': self.solver_type,
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_problem_data(self, problem_data: Dict[str, Any]) -> RoutingProblem:
        """
        Parse problem data dictionary into RoutingProblem object
        
        Args:
            problem_data: Problem data dictionary
            
        Returns:
            RoutingProblem object
        """
        return RoutingProblem(
            num_vehicles=problem_data['num_vehicles'],
            depot_index=problem_data['depot_index'],
            distance_matrix=np.array(problem_data['distance_matrix']),
            demands=problem_data['demands'],
            vehicle_capacities=problem_data['vehicle_capacities'],
            time_windows=problem_data.get('time_windows', []),
            service_times=problem_data.get('service_times', []),
            pickup_deliveries=problem_data.get('pickup_deliveries', [])
        )
    
    def _solve_problem(self, problem: RoutingProblem) -> Optional[RouteSolution]:
        """
        Solve problem using appropriate solver
        
        Args:
            problem: Routing problem
            
        Returns:
            Route solution or None if no solution found
        """
        if self.solver_type == "ortools":
            return self.solver.solve_cvrp(problem)
        elif self.solver_type == "rl":
            # RL solver needs an initial solution
            exact_solver = ORToolsRoutingSolver()
            initial_solution = exact_solver.solve_cvrp(problem)
            if initial_solution:
                return self.solver.refine_routes(initial_solution, problem)
            return None
        elif self.solver_type == "hybrid":
            return self.solver.solve(problem)
        elif self.solver_type == "adaptive":
            return self.solver.solve_with_learning(problem)
        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}")
    
    def _format_solution(self, solution: RouteSolution) -> Dict[str, Any]:
        """
        Format solution for output
        
        Args:
            solution: Route solution
            
        Returns:
            Formatted solution dictionary
        """
        return {
            'routes': solution.routes,
            'vehicle_loads': solution.vehicle_loads,
            'total_distance': solution.total_distance,
            'total_time': solution.total_time,
            'objective_value': solution.objective_value
        }


class BatchRoutingProcessor:
    """Process multiple routing problems in batch"""
    
    def __init__(self, solver_type: str = "hybrid"):
        """
        Initialize batch processor
        
        Args:
            solver_type: Type of solver to use
        """
        self.inference_engine = RoutingInferenceEngine(solver_type)
        self.processing_history = []
    
    def process_batch(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process batch of routing problems
        
        Args:
            problems: List of problem data dictionaries
            
        Returns:
            List of solution results
        """
        results = []
        
        for i, problem_data in enumerate(problems):
            try:
                self.logger.info(f"Processing problem {i+1}/{len(problems)}")
                result = self.inference_engine.solve_routing_problem(problem_data)
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'problem_index': i,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Store batch processing record
        batch_record = {
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'num_problems': len(problems),
            'num_successful': sum(1 for r in results if r.get('success', False)),
            'processing_time': sum(r.get('processing_time', 0) for r in results),
            'timestamp': datetime.now().isoformat()
        }
        self.processing_history.append(batch_record)
        
        return results
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get processing history
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of processing records
        """
        return self.processing_history[-limit:]


class RoutingAPIInterface:
    """Interface for routing API calls"""
    
    def __init__(self, default_solver: str = "hybrid"):
        """
        Initialize API interface
        
        Args:
            default_solver: Default solver type
        """
        self.default_solver = default_solver
        self.logger = logging.getLogger('RoutingAPIInterface')
    
    def handle_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle routing API request
        
        Args:
            request_data: API request data
            
        Returns:
            API response
        """
        try:
            # Extract solver type from request or use default
            solver_type = request_data.get('solver_type', self.default_solver)
            
            # Create inference engine
            inference_engine = RoutingInferenceEngine(solver_type)
            
            # Process request
            if request_data.get('batch_mode', False):
                # Batch processing
                problems = request_data['problems']
                batch_processor = BatchRoutingProcessor(solver_type)
                results = batch_processor.process_batch(problems)
                
                return {
                    'status': 'success',
                    'results': results,
                    'batch_processed': len(problems)
                }
            else:
                # Single problem
                problem_data = request_data['problem']
                result = inference_engine.solve_routing_problem(problem_data)
                
                return {
                    'status': 'success',
                    'result': result
                }
                
        except Exception as e:
            self.logger.error(f"Error handling API request: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }


def load_routing_engine(solver_type: str = "hybrid") -> RoutingInferenceEngine:
    """
    Load routing inference engine
    
    Args:
        solver_type: Type of solver to use
        
    Returns:
        Routing inference engine
    """
    return RoutingInferenceEngine(solver_type)


def solve_single_problem(problem_data: Dict[str, Any], 
                        solver_type: str = "hybrid") -> Dict[str, Any]:
    """
    Solve a single routing problem
    
    Args:
        problem_data: Problem definition
        solver_type: Solver type to use
        
    Returns:
        Solution result
    """
    engine = RoutingInferenceEngine(solver_type)
    return engine.solve_routing_problem(problem_data)


# Example usage
if __name__ == "__main__":
    # Create sample problem data
    problem_data = {
        'num_vehicles': 2,
        'depot_index': 0,
        'distance_matrix': [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ],
        'demands': [0, 10, 20, 15],
        'vehicle_capacities': [30, 30]
    }
    
    # Solve with different solvers
    for solver_type in ["ortools", "hybrid"]:
        print(f"\nSolving with {solver_type} solver:")
        
        try:
            result = solve_single_problem(problem_data, solver_type)
            
            if result['success']:
                print(f"Success! Objective value: {result['solution']['objective_value']}")
                print(f"Routes: {result['solution']['routes']}")
                print(f"Processing time: {result['processing_time']:.2f}s")
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error with {solver_type}: {str(e)}")
