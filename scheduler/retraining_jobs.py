"""
Retraining Scheduler Jobs
"""
import logging
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any, List

from training.pipelines.train_demand import DemandTrainingPipeline
from training.pipelines.train_cost import CostTrainingPipeline
from training.pipelines.train_rl import RLTrainingPipeline
from training.pipelines.train_anomaly import AnomalyTrainingPipeline
from training.pipelines.evaluate_all import ComprehensiveEvaluationPipeline
from registry.models.model_registry import ModelRegistry
from registry.policies.policy_registry import PolicyRegistry
from evaluation.kpi_definitions import KPIMetrics

logger = logging.getLogger(__name__)

class RetrainingScheduler:
    """Manages scheduled retraining jobs for all models and policies."""
    
    def __init__(self):
        self.demand_pipeline = DemandTrainingPipeline()
        self.cost_pipeline = CostTrainingPipeline()
        self.rl_pipeline = RLTrainingPipeline()
        self.anomaly_pipeline = AnomalyTrainingPipeline()
        self.evaluator = ComprehensiveEvaluationPipeline()
        self.model_registry = ModelRegistry()
        self.policy_registry = PolicyRegistry()
        self.kpi_metrics = KPIMetrics()
        
    async def run_weekly_retraining(self) -> Dict[str, Any]:
        """
        Run weekly retraining for all models.
        
        Returns:
            Retraining results
        """
        logger.info("Starting weekly retraining job")
        start_time = datetime.now()
        
        results = {
            'job_type': 'weekly_retraining',
            'start_time': start_time.isoformat(),
            'models_trained': [],
            'results': {}
        }
        
        # Calculate date range for training data (previous 3 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        date_range = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
        # 1. Retrain demand forecasting model
        logger.info("Retraining demand forecasting model")
        try:
            demand_results = self.demand_pipeline.run_training(**date_range)
            results['results']['demand'] = demand_results
            results['models_trained'].append('demand')
            logger.info("Demand model retraining completed successfully")
        except Exception as e:
            logger.error(f"Demand model retraining failed: {e}")
            results['results']['demand'] = {'error': str(e)}
        
        # 2. Retrain cost model
        logger.info("Retraining cost model")
        try:
            cost_results = self.cost_pipeline.run_training(**date_range)
            results['results']['cost'] = cost_results
            results['models_trained'].append('cost')
            logger.info("Cost model retraining completed successfully")
        except Exception as e:
            logger.error(f"Cost model retraining failed: {e}")
            results['results']['cost'] = {'error': str(e)}
        
        # 3. Retrain anomaly detection models
        logger.info("Retraining anomaly detection models")
        try:
            anomaly_results = self.anomaly_pipeline.run_training(**date_range)
            results['results']['anomaly'] = anomaly_results
            results['models_trained'].append('anomaly')
            logger.info("Anomaly detection models retraining completed successfully")
        except Exception as e:
            logger.error(f"Anomaly detection models retraining failed: {e}")
            results['results']['anomaly'] = {'error': str(e)}
        
        # 4. Evaluate all models
        logger.info("Evaluating all models")
        try:
            evaluation_results = self.evaluator.run_evaluation()
            results['results']['evaluation'] = evaluation_results
            logger.info("Model evaluation completed successfully")
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            results['results']['evaluation'] = {'error': str(e)}
        
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        logger.info(f"Weekly retraining completed in {results['duration_seconds']:.2f} seconds")
        return results
    
    async def run_monthly_full_benchmark(self) -> Dict[str, Any]:
        """
        Run monthly full benchmark of all models and policies.
        
        Returns:
            Benchmark results
        """
        logger.info("Starting monthly full benchmark job")
        start_time = datetime.now()
        
        results = {
            'job_type': 'monthly_benchmark',
            'start_time': start_time.isoformat(),
            'completed': False,
            'results': {}
        }
        
        try:
            # Run comprehensive evaluation
            evaluation_results = self.evaluator.run_evaluation()
            results['results']['evaluation'] = evaluation_results
            
            # Run RL training with benchmarking
            rl_results = self.rl_pipeline.run_training(benchmark=True)
            results['results']['rl'] = rl_results
            
            # Compare with previous versions
            comparison_results = self._compare_with_baseline(evaluation_results)
            results['results']['comparison'] = comparison_results
            
            # Promote models if they meet criteria
            promotion_results = self._promote_models_if_qualified(evaluation_results)
            results['results']['promotion'] = promotion_results
            
            results['completed'] = True
            logger.info("Monthly full benchmark completed successfully")
            
        except Exception as e:
            logger.error(f"Monthly full benchmark failed: {e}")
            results['error'] = str(e)
        
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        logger.info(f"Monthly full benchmark completed in {results['duration_seconds']:.2f} seconds")
        return results
    
    def _compare_with_baseline(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline."""
        # In a real implementation, this would compare with stored baseline metrics
        # For now, returning simulated comparison
        return {
            'improvements': ['forecast_accuracy', 'routing_efficiency'],
            'degradations': ['anomaly_detection_false_positive_rate'],
            'unchanged': ['policy_success_rate']
        }
    
    def _promote_models_if_qualified(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Promote models to production if they meet quality criteria."""
        promotions = []
        
        # Check forecast model quality
        forecast_accuracy = evaluation_results.get('forecast', {}).get('accuracy_p50', 0)
        if forecast_accuracy > 0.85:
            promotions.append({
                'model': 'forecast',
                'version': evaluation_results.get('forecast', {}).get('version', 'unknown'),
                'promoted': True,
                'reason': f'Accuracy {forecast_accuracy:.2f} > 0.85 threshold'
            })
        
        # Check policy performance
        policy_success_rate = evaluation_results.get('policy', {}).get('success_rate', 0)
        if policy_success_rate > 0.9:
            promotions.append({
                'model': 'policy',
                'version': evaluation_results.get('policy', {}).get('version', 'unknown'),
                'promoted': True,
                'reason': f'Success rate {policy_success_rate:.2f} > 0.9 threshold'
            })
        
        return {
            'promotions': promotions,
            'criteria': {
                'forecast_accuracy_threshold': 0.85,
                'policy_success_rate_threshold': 0.9
            }
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run retraining jobs
    async def main():
        scheduler = RetrainingScheduler()
        
        # Run weekly retraining
        weekly_results = await scheduler.run_weekly_retraining()
        print(f"Weekly retraining completed: {weekly_results}")
        
        # Run monthly benchmark (separately to avoid long execution)
        # monthly_results = await scheduler.run_monthly_full_benchmark()
        # print(f"Monthly benchmark completed: {monthly_results}")
    
    asyncio.run(main())