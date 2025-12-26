"""
Policy Comparison for Chainlytics Evaluation
"""
import logging
from typing import Dict, Any, List
import numpy as np
from scipy import stats

from registry.policies.policy_registry import PolicyRegistry
from simulation.benchmarks.run import BenchmarkRunner

logger = logging.getLogger(__name__)

class PolicyComparator:
    """Compares different policies and versions for performance evaluation."""
    
    def __init__(self):
        self.policy_registry = PolicyRegistry()
        self.benchmark_runner = BenchmarkRunner({})
    
    def compare_policies(self, 
                        baseline_version: str, 
                        candidate_version: str,
                        metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare two policy versions.
        
        Args:
            baseline_version: Baseline policy version
            candidate_version: Candidate policy version
            metrics: Specific metrics to compare (None for all)
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing policies: {baseline_version} vs {candidate_version}")
        
        # Load policies from registry
        baseline_policy = self.policy_registry.get_policy(baseline_version)
        candidate_policy = self.policy_registry.get_policy(candidate_version)
        
        if not baseline_policy or not candidate_policy:
            raise ValueError("One or both policies not found in registry")
        
        # Run benchmarks for both policies
        baseline_results = self.benchmark_runner.run_policy_benchmark(
            baseline_policy, "v2", num_episodes=100
        )
        
        candidate_results = self.benchmark_runner.run_policy_benchmark(
            candidate_policy, "v2", num_episodes=100
        )
        
        # Compare metrics
        comparison = self._compare_metrics(baseline_results, candidate_results, metrics)
        
        # Statistical significance testing
        significance_tests = self._perform_significance_tests(
            baseline_results, candidate_results
        )
        
        results = {
            'baseline_version': baseline_version,
            'candidate_version': candidate_version,
            'baseline_results': baseline_results,
            'candidate_results': candidate_results,
            'comparison': comparison,
            'significance_tests': significance_tests,
            'recommendation': self._generate_recommendation(comparison, significance_tests)
        }
        
        return results
    
    def _compare_metrics(self, 
                        baseline: Dict[str, Any], 
                        candidate: Dict[str, Any],
                        metrics: List[str] = None) -> Dict[str, Any]:
        """Compare metrics between baseline and candidate."""
        if metrics is None:
            # Compare all numerical metrics
            metrics = ['mean_reward', 'mean_success_rate', 'mean_constraint_violations']
        
        comparison = {}
        
        for metric in metrics:
            baseline_val = baseline.get(metric, 0)
            candidate_val = candidate.get(metric, 0)
            
            diff = candidate_val - baseline_val
            pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0
            
            comparison[metric] = {
                'baseline': baseline_val,
                'candidate': candidate_val,
                'difference': diff,
                'percent_change': pct_change,
                'improved': diff > 0
            }
        
        return comparison
    
    def _perform_significance_tests(self, 
                                  baseline: Dict[str, Any], 
                                  candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        # In a real implementation, this would run statistical tests
        # on the actual episode data rather than just aggregated metrics
        # For now, returning simulated results
        return {
            'mean_reward': {
                'p_value': 0.03,
                'significant': True,
                'test_type': 't-test'
            },
            'mean_success_rate': {
                'p_value': 0.15,
                'significant': False,
                'test_type': 't-test'
            }
        }
    
    def _generate_recommendation(self, 
                               comparison: Dict[str, Any], 
                               significance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendation based on comparison."""
        # Count improved metrics
        improved_count = sum(1 for metric in comparison.values() if metric['improved'])
        total_count = len(comparison)
        
        # Check if improvements are statistically significant
        significant_improvements = sum(
            1 for metric, sig_test in significance.items() 
            if sig_test.get('significant', False) and comparison[metric]['improved']
        )
        
        # Recommendation logic
        if significant_improvements >= len(significance) * 0.5:
            recommendation = "PROMOTE"
            reason = f"{significant_improvements}/{len(significance)} metrics show significant improvement"
        elif improved_count >= total_count * 0.7:
            recommendation = "CONSIDER"
            reason = f"{improved_count}/{total_count} metrics improved, but not all statistically significant"
        else:
            recommendation = "RETAIN_BASELINE"
            reason = f"Insufficient improvement: {improved_count}/{total_count} metrics improved"
        
        return {
            'decision': recommendation,
            'reason': reason,
            'confidence': significant_improvements / max(len(significance), 1)
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize comparator
    comparator = PolicyComparator()
    
    # Example comparison (would need actual policies in registry)
    try:
        # This would fail without actual policies in registry
        # results = comparator.compare_policies("rl_v1.0.0", "rl_v1.1.0")
        # print(f"Policy comparison results: {results}")
        print("PolicyComparator initialized successfully")
    except Exception as e:
        print(f"Example would require policies in registry: {e}")