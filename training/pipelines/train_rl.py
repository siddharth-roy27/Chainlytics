"""
Reinforcement Learning Training Pipeline
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List
import yaml
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.policy_service.agents import MultiAgentPPO
from services.policy_service.policy import PolicyNetwork
from services.policy_service.critic import CriticNetwork
from services.state_service.encoder import StateEncoder
from simulation.env.base_env import LogisticsEnv
from simulation.reward.reward_v2 import RewardFunctionV2
from simulation.benchmarks.run import BenchmarkRunner
from registry.policies.policy_registry import PolicyRegistry

logger = logging.getLogger(__name__)

class RLTrainingPipeline:
    """Training pipeline for reinforcement learning policies."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.env = LogisticsEnv()
        self.reward_fn = RewardFunctionV2()
        self.state_encoder = StateEncoder()
        self.registry = PolicyRegistry()
        self.benchmark_runner = BenchmarkRunner({})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'training': {
                    'episodes': 1000,
                    'max_steps': 500,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'gamma': 0.99,
                    'epsilon': 0.2,
                    'entropy_coeff': 0.01
                },
                'network': {
                    'hidden_layers': [256, 128, 64],
                    'activation': 'relu'
                }
            }
    
    def run_training(self,
                     version: str = None,
                     benchmark: bool = True) -> Dict[str, Any]:
        """
        Run the complete RL training pipeline.
        
        Args:
            version: Policy version identifier
            benchmark: Whether to run benchmarks after training
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting RL training pipeline")
        
        # Initialize agents
        logger.info("Initializing RL agents")
        agents = self._initialize_agents()
        
        # Training parameters
        episodes = self.config['training']['episodes']
        max_steps = self.config['training']['max_steps']
        
        # Metrics tracking
        episode_rewards = []
        episode_lengths = []
        policy_losses = []
        
        # Training loop
        for episode in range(episodes):
            state = self.env.reset()
            encoded_state = self.state_encoder.encode(state)
            
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Get actions from all agents
                actions = {}
                log_probs = {}
                
                for agent_id, agent in agents.items():
                    action, log_prob = agent.select_action(encoded_state)
                    actions[agent_id] = action
                    log_probs[agent_id] = log_prob
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(actions)
                encoded_next_state = self.state_encoder.encode(next_state)
                
                # Store experiences for all agents
                for agent_id, agent in agents.items():
                    agent.store_experience(
                        encoded_state, 
                        actions[agent_id], 
                        reward, 
                        encoded_next_state, 
                        log_probs[agent_id], 
                        done
                    )
                
                # Update agents periodically
                if (episode * max_steps + step) % self.config['training']['batch_size'] == 0:
                    for agent in agents.values():
                        loss = agent.update()
                        if loss is not None:
                            policy_losses.append(loss)
                
                episode_reward += reward
                steps += 1
                
                if done:
                    break
                    
                state = next_state
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                logger.info(f"Episode {episode}/{episodes} - "
                           f"Avg Reward: {avg_reward:.2f}, "
                           f"Avg Length: {avg_length:.2f}")
        
        # Create version if not provided
        if not version:
            version = f"rl_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Register policy
        logger.info(f"Registering policy version: {version}")
        policy_metadata = {
            'version': version,
            'training_date': datetime.now().isoformat(),
            'episodes': episodes,
            'config': self.config,
            'metrics': {
                'final_avg_reward': float(np.mean(episode_rewards[-100:])),
                'final_avg_length': float(np.mean(episode_lengths[-100:])),
                'total_rewards': float(np.sum(episode_rewards)),
                'policy_loss_final': float(np.mean(policy_losses[-100:])) if policy_losses else 0.0
            }
        }
        
        # Save trained agents
        self.registry.register_policy(agents, policy_metadata, version)
        
        results = {
            'version': version,
            'metrics': policy_metadata['metrics'],
            'training_completed': datetime.now().isoformat()
        }
        
        # Run benchmarks if requested
        if benchmark:
            logger.info("Running policy benchmarks")
            benchmark_results = self.benchmark_runner.run_policy_benchmark(
                agents, "v2", num_episodes=50
            )
            results['benchmarks'] = benchmark_results
        
        logger.info("RL training pipeline completed successfully")
        return results
    
    def _initialize_agents(self) -> Dict[str, MultiAgentPPO]:
        """Initialize RL agents for each entity type."""
        agents = {}
        
        # Initialize agents for different entity types
        entity_types = ['warehouse', 'vehicle', 'driver']
        
        for entity_type in entity_types:
            agent = MultiAgentPPO(
                state_dim=self.state_encoder.get_encoding_dim(),
                action_dim=10,  # Placeholder, should match actual action space
                hidden_layers=self.config['network']['hidden_layers'],
                lr=self.config['training']['learning_rate'],
                gamma=self.config['training']['gamma'],
                epsilon=self.config['training']['epsilon'],
                entropy_coeff=self.config['training']['entropy_coeff']
            )
            agents[entity_type] = agent
            
        return agents

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run training pipeline
    pipeline = RLTrainingPipeline()
    results = pipeline.run_training(benchmark=True)
    
    print(f"Training completed with results: {results}")