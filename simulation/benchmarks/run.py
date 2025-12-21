"""
Benchmark Runner for Simulation Engine
"""
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime
import json
import os

from ..env.base_env import LogisticsEnv
from ..reward.reward_v1 import RewardFunctionV1
from ..reward.reward_v2 import RewardFunctionV2

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Run benchmarks for comparing different policies and reward functions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        
    def run_policy_benchmark(self, 
                           policy: Any, 
                           reward_version: str = "v2",
                           num_episodes: int = 100) -> Dict[str, float]:
        """
        Run benchmark for a specific policy.
        
        Args:
            policy: The policy to benchmark
            reward_version: Version of reward function to use ("v1" or "v2")
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary of benchmark metrics
        """
        logger.info(f"Running benchmark for policy with reward version {reward_version}")
        
        # Initialize environment
        env = LogisticsEnv(seed=42)
        
        # Select reward function
        if reward_version == "v1":
            reward_fn = RewardFunctionV1()
        else:
            reward_fn = RewardFunctionV2()
            
        # Metrics tracking
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        constraint_violations = []
        
        # Run episodes
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            violations = 0
            
            done = False
            while not done:
                # Get action from policy
                action = policy.predict(state)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Track metrics
                total_reward += reward
                steps += 1
                
                # Count constraint violations
                if info.get('constraint_violation', False):
                    violations += 1
                    
                state = next_state
                
            # Store episode results
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            success_rates.append(info.get('success', 0))
            constraint_violations.append(violations)
            
            if episode % 10 == 0:
                logger.info(f"Completed episode {episode}/{num_episodes}")
                
        # Calculate aggregate metrics
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'mean_success_rate': float(np.mean(success_rates)),
            'mean_constraint_violations': float(np.mean(constraint_violations)),
            'reward_version': reward_version,
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
        
    def compare_policies(self, 
                        policies: List[Any],
                        policy_names: List[str],
                        reward_versions: List[str] = ["v1", "v2"]) -> Dict[str, Any]:
        """
        Compare multiple policies across different reward versions.
        
        Args:
            policies: List of policies to compare
            policy_names: Names of the policies
            reward_versions: Reward versions to test
            
        Returns:
            Comparison results dictionary
        """
        results = {}
        
        for i, policy in enumerate(policies):
            policy_results = {}
            policy_name = policy_names[i]
            
            for reward_version in reward_versions:
                logger.info(f"Testing policy {policy_name} with reward {reward_version}")
                metrics = self.run_policy_benchmark(policy, reward_version)
                policy_results[reward_version] = metrics
                
            results[policy_name] = policy_results
            
        self.results = results
        return results
        
    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Benchmark results saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # This would typically be run from a training pipeline
    config = {
        "benchmark_episodes": 100,
        "save_path": "./results/benchmark_results.json"
    }
    
    runner = BenchmarkRunner(config)
    print("Benchmark runner initialized")