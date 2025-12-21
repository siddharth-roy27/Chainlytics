"""
Comprehensive Evaluation Pipeline
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.forecast_service.metrics import evaluate_forecast
from services.cost_service.marginal_cost import MarginalCostCalculator
from services.policy_service.policy import PolicyNetwork
from services.anomaly_service.detector import AnomalyDetector
from services.routing_service.hybrid import HybridRoutingOptimizer
from evaluation.kpi_definitions import KPIMetrics
from evaluation.policy_comparison import PolicyComparator
from evaluation.regression_tests import RegressionTester
from registry.metrics.metrics_registry import MetricsRegistry

logger = logging.getLogger(__name__)

class ComprehensiveEvaluationPipeline:
    """Comprehensive evaluation pipeline for all services."""
    
    def __init__(self):
        self.kpi_metrics = KPIMetrics()
        self.policy_comparator = PolicyComparator()
        self.regression_tester = RegressionTester()
        self.metrics_registry = MetricsRegistry()
        
    def run_evaluation(self, 
                      baseline_version: str = None,
                      candidate_version: str = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all services.
        
        Args:
            baseline_version: Baseline version for comparison
            candidate_version: Candidate version to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting comprehensive evaluation pipeline")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_version': baseline_version,
            'candidate_version': candidate_version
        }
        
        # Evaluate forecasting performance
        logger.info("Evaluating forecast service")
        forecast_results = self._evaluate_forecast_service()
        results['forecast'] = forecast_results
        
        # Evaluate cost modeling
        logger.info("Evaluating cost service")
        cost_results = self._evaluate_cost_service()
        results['cost'] = cost_results
        
        # Evaluate routing optimization
        logger.info("Evaluating routing service")
        routing_results = self._evaluate_routing_service()
        results['routing'] = routing_results
        
        # Evaluate policy performance
        logger.info("Evaluating policy service")
        policy_results = self._evaluate_policy_service(baseline_version, candidate_version)
        results['policy'] = policy_results
        
        # Evaluate anomaly detection
        logger.info("Evaluating anomaly service")
        anomaly_results = self._evaluate_anomaly_service()
        results['anomaly'] = anomaly_results
        
        # Calculate overall KPIs
        logger.info("Calculating overall KPIs")
        kpi_results = self._calculate_overall_kpis(results)
        results['kpis'] = kpi_results
        
        # Run regression tests
        logger.info("Running regression tests")
        regression_results = self.regression_tester.run_all_tests()
        results['regression'] = regression_results
        
        # Register metrics
        self.metrics_registry.register_evaluation_metrics(results)
        
        logger.info("Comprehensive evaluation pipeline completed successfully")
        return results
    
    def _evaluate_forecast_service(self) -> Dict[str, Any]:
        """Evaluate forecast service performance."""
        # This would typically load test data and compare predictions to actuals
        # For now, returning placeholder results
        return {
            'accuracy_p50': 0.85,
            'accuracy_p90': 0.78,
            'accuracy_p99': 0.72,
            'mape': 12.5,
            'bias': 0.02
        }
    
    def _evaluate_cost_service(self) -> Dict[str, Any]:
        """Evaluate cost service performance."""
        # This would typically validate cost predictions against actual costs
        # For now, returning placeholder results
        return {
            'mae': 15.2,
            'rmse': 22.1,
            'mape': 8.3,
            'r_squared': 0.89
        }
    
    def _evaluate_routing_service(self) -> Dict[str, Any]:
        """Evaluate routing service performance."""
        # This would typically compare route efficiency and constraints satisfaction
        # For now, returning placeholder results
        return {
            'avg_route_efficiency': 0.87,
            'constraint_satisfaction_rate': 0.95,
            'avg_delivery_time_improvement': 0.12,
            'cost_savings': 0.08
        }
    
    def _evaluate_policy_service(self, baseline_version: str, candidate_version: str) -> Dict[str, Any]:
        """Evaluate policy service performance."""
        # Compare policies if both versions are provided
        if baseline_version and candidate_version:
            comparison = self.policy_comparator.compare_policies(
                baseline_version, candidate_version
            )
            return comparison
        else:
            # Return standalone evaluation
            return {
                'reward_mean': 125.4,
                'reward_std': 22.1,
                'success_rate': 0.89,
                'constraint_violations': 0.03
            }
    
    def _evaluate_anomaly_service(self) -> Dict[str, Any]:
        """Evaluate anomaly service performance."""
        # This would typically measure detection accuracy and false positive rates
        # For now, returning placeholder results
        return {
            'detection_rate': 0.92,
            'false_positive_rate': 0.08,
            'precision': 0.88,
            'recall': 0.91
        }
    
    def _calculate_overall_kpis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall KPIs from individual service evaluations."""
        kpis = {}
        
        # Operational efficiency KPI
        kpis['operational_efficiency'] = (
            evaluation_results['forecast']['accuracy_p50'] * 0.3 +
            evaluation_results['routing']['avg_route_efficiency'] * 0.4 +
            evaluation_results['policy']['success_rate'] * 0.3
        )
        
        # Cost effectiveness KPI
        kpis['cost_effectiveness'] = (
            (1 - evaluation_results['cost']['mape'] / 100) * 0.5 +
            (1 - evaluation_results['routing']['cost_savings']) * 0.3 +
            evaluation_results['anomaly']['detection_rate'] * 0.2
        )
        
        # Service quality KPI
        kpis['service_quality'] = (
            evaluation_results['forecast']['accuracy_p90'] * 0.4 +
            evaluation_results['policy']['success_rate'] * 0.3 +
            (1 - evaluation_results['anomaly']['false_positive_rate']) * 0.3
        )
        
        # Risk management KPI
        kpis['risk_management'] = (
            evaluation_results['anomaly']['detection_rate'] * 0.5 +
            (1 - evaluation_results['policy']['constraint_violations']) * 0.3 +
            evaluation_results['routing']['constraint_satisfaction_rate'] * 0.2
        )
        
        return kpis
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation pipeline
    pipeline = ComprehensiveEvaluationPipeline()
    results = pipeline.run_evaluation(
        baseline_version="v1.0.0",
        candidate_version="v1.1.0"
    )
    
    # Save results
    pipeline.save_results(results, "./results/comprehensive_evaluation.json")
    
    print(f"Evaluation completed with results: {results}")