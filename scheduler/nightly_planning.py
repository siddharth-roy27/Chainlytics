"""
Nightly Planning Scheduler Job
"""
import logging
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any

from training.pipelines.evaluate_all import ComprehensiveEvaluationPipeline
from inference.decision_orchestrator import DecisionOrchestrator
from registry.models.model_registry import ModelRegistry
from registry.policies.policy_registry import PolicyRegistry
from services.forecast_service.infer import DemandForecaster
from data.loaders.historical_loader import HistoricalDataLoader

logger = logging.getLogger(__name__)

class NightlyPlanningJob:
    """Nightly planning job for demand forecasting and initial planning."""
    
    def __init__(self):
        self.evaluator = ComprehensiveEvaluationPipeline()
        self.orchestrator = DecisionOrchestrator()
        self.model_registry = ModelRegistry()
        self.policy_registry = PolicyRegistry()
        self.demand_forecaster = DemandForecaster()
        self.data_loader = HistoricalDataLoader()
        
    async def run_planning(self, date: str = None) -> Dict[str, Any]:
        """
        Run nightly planning job.
        
        Args:
            date: Date for planning (defaults to tomorrow)
            
        Returns:
            Planning results
        """
        if date is None:
            # Default to tomorrow
            date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
        logger.info(f"Starting nightly planning for {date}")
        start_time = datetime.now()
        
        # 1. Load and analyze historical data
        logger.info("Loading historical data")
        historical_data = self.data_loader.load_recent_data(days=30)
        
        # 2. Generate demand forecasts
        logger.info("Generating demand forecasts")
        forecasts = self._generate_forecasts(historical_data, date)
        
        # 3. Evaluate current models
        logger.info("Evaluating current models")
        evaluation_results = self.evaluator.run_evaluation()
        
        # 4. Check for model updates
        logger.info("Checking for model updates")
        model_updates = self._check_model_updates(evaluation_results)
        
        # 5. Generate initial plan
        logger.info("Generating initial plan")
        initial_plan = self._generate_initial_plan(forecasts, date)
        
        # 6. Log planning results
        end_time = datetime.now()
        planning_duration = (end_time - start_time).total_seconds()
        
        results = {
            'date': date,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': planning_duration,
            'forecasts': forecasts,
            'evaluation_results': evaluation_results,
            'model_updates': model_updates,
            'initial_plan': initial_plan
        }
        
        logger.info(f"Nightly planning completed in {planning_duration:.2f} seconds")
        return results
    
    def _generate_forecasts(self, historical_data: Dict[str, Any], date: str) -> Dict[str, Any]:
        """Generate demand forecasts for planning."""
        # In a real implementation, this would use the demand forecaster
        # For now, returning simulated forecasts
        return {
            'warehouse_forecasts': [
                {
                    'warehouse_id': 'wh1',
                    'date': date,
                    'p50': 120,
                    'p90': 180,
                    'p99': 240
                }
            ],
            'generated_at': datetime.now().isoformat()
        }
    
    def _check_model_updates(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if models need updates based on evaluation results."""
        # Simplified logic - in reality, this would be more complex
        updates_needed = []
        
        # Check if forecast accuracy dropped below threshold
        if evaluation_results.get('forecast', {}).get('accuracy_p50', 1.0) < 0.8:
            updates_needed.append('forecast_model')
            
        # Check if policy performance degraded
        if evaluation_results.get('policy', {}).get('success_rate', 1.0) < 0.85:
            updates_needed.append('policy_model')
            
        return {
            'updates_needed': updates_needed,
            'evaluation_results': evaluation_results
        }
    
    def _generate_initial_plan(self, forecasts: Dict[str, Any], date: str) -> Dict[str, Any]:
        """Generate initial logistics plan based on forecasts."""
        # In a real implementation, this would be much more complex
        # For now, returning a simple plan structure
        return {
            'date': date,
            'warehouse_allocations': [
                {
                    'warehouse_id': 'wh1',
                    'allocated_inventory': 1000,
                    'forecasted_demand': forecasts['warehouse_forecasts'][0]['p50']
                }
            ],
            'vehicle_assignments': [],
            'generated_at': datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run nightly planning
    async def main():
        job = NightlyPlanningJob()
        results = await job.run_planning()
        print(f"Nightly planning completed: {results}")
    
    asyncio.run(main())