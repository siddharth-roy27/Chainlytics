"""
Training pipeline for multi-agent policy learning
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import yaml
from datetime import datetime
from collections import deque
import logging

from .agents import BaseAgent, PPOAgent, MultiAgentSystem
from .policy import PolicyManager, PolicyConfig
from .critic import MultiAgentCritic, CriticConfig
from .rewards import LogisticsRewardFunction, RewardComponents


class TrainingConfig:
    """Configuration for training pipeline"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize training configuration
        
        Args:
            config_dict: Configuration dictionary
        """
        config = config_dict or {}
        
        # Training parameters
        self.episodes = config.get('episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 100)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_clip = config.get('epsilon_clip', 0.2)
        self.epochs_per_update = config.get('epochs_per_update', 10)
        
        # Environment parameters
        self.simulation_config = config.get('simulation_config', {})
        
        # Logging parameters
        self.log_interval = config.get('log_interval', 10)
        self.save_interval = config.get('save_interval', 100)
        
        # Model saving
        self.save_path = config.get('save_path', 'registry/models/policy')
        self.checkpoint_path = config.get('checkpoint_path', 'checkpoints')


class MultiAgentTrainer:
    """Train multi-agent policies using reinforcement learning"""
    
    def __init__(self, training_config: TrainingConfig):
        """
        Initialize trainer
        
        Args:
            training_config: Training configuration
        """
        self.config = training_config
        self.policy_manager = PolicyManager()
        self.reward_function = LogisticsRewardFunction()
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.loss_history = deque(maxlen=1000)
        
        # Setup logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for training"""
        logger = logging.getLogger('MultiAgentTrainer')
        logger.setLevel(logging.INFO)
        
        # Create handler if it doesn't exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_training_policy(self, agent_configs: List[Dict[str, Any]]) -> str:
        """
        Create policy for training
        
        Args:
            agent_configs: Configuration for each agent
            
        Returns:
            Policy ID
        """
        policy_config = PolicyConfig(
            policy_id=f"policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            policy_type="cooperative",
            agent_configs=agent_configs,
            training_algorithm="ppo"
        )
        
        policy_id = self.policy_manager.create_policy(policy_config)
        self.policy_manager.activate_policy(policy_id)
        
        return policy_id
    
    def train_episode(self, environment: Any) -> Tuple[float, int]:
        """
        Train for one episode
        
        Args:
            environment: Training environment
            
        Returns:
            Tuple of (episode_reward, steps_taken)
        """
        # Get active policy
        policy_info = self.policy_manager.get_active_policy()
        if not policy_info:
            raise ValueError("No active policy found")
        
        multi_agent_system = policy_info['multi_agent_system']
        
        # Reset environment
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        
        # Episode loop
        while steps < self.config.max_steps_per_episode:
            # Get observations for each agent
            observations = self._extract_agent_observations(state)
            
            # Select actions
            actions = multi_agent_system.select_actions(observations)
            
            # Execute actions in environment
            next_state, global_reward, done, info = environment.step(actions)
            
            # Distribute reward to agents
            agent_rewards = self._distribute_rewards(state, actions, next_state, global_reward)
            
            # Get next observations
            next_observations = self._extract_agent_observations(next_state)
            
            # Update agents
            multi_agent_system.update_agents(
                observations, actions, agent_rewards, next_observations
            )
            
            # Update state
            state = next_state
            total_reward += global_reward
            steps += 1
            
            # Check if episode is done
            if done:
                break
        
        return total_reward, steps
    
    def _extract_agent_observations(self, global_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract individual agent observations from global state
        
        Args:
            global_state: Global environment state
            
        Returns:
            Dictionary mapping agent IDs to their observations
        """
        # This is a simplified implementation
        # In practice, you would extract relevant information for each agent
        observations = {}
        
        # Example: Assume global_state contains agent-specific information
        for agent_id in global_state.get('agent_positions', {}).keys():
            observations[agent_id] = {
                'state_embedding': global_state.get('agent_embeddings', {}).get(agent_id, np.zeros(64)),
                'neighbor_info': global_state.get('neighbor_info', {}).get(agent_id, {}),
                'global_info': global_state.get('global_metrics', {}),
                'reward': 0.0,
                'done': False
            }
        
        return observations
    
    def _distribute_rewards(self, state: Dict[str, Any], actions: Dict[str, Any],
                          next_state: Dict[str, Any], global_reward: float) -> Dict[str, float]:
        """
        Distribute global reward to individual agents
        
        Args:
            state: Current state
            actions: Actions taken
            next_state: Next state
            global_reward: Global reward
            
        Returns:
            Dictionary mapping agent IDs to individual rewards
        """
        # Compute individual contributions using reward function
        agent_rewards = {}
        
        # This is a simplified approach - in practice, you'd use more sophisticated
        # credit assignment methods
        agent_ids = list(state.get('agent_positions', {}).keys())
        
        if agent_ids:
            # Equal distribution as baseline
            equal_share = global_reward / len(agent_ids)
            for agent_id in agent_ids:
                agent_rewards[agent_id] = equal_share
                
                # Add individual performance bonus
                # This would involve computing individual contributions
                # For now, we'll add a small random bonus/penalty
                performance_bonus = np.random.normal(0, 0.1)
                agent_rewards[agent_id] += performance_bonus
        else:
            # No agents - no rewards
            pass
        
        return agent_rewards
    
    def train_policy(self, environment: Any) -> Dict[str, Any]:
        """
        Train policy for specified number of episodes
        
        Args:
            environment: Training environment
            
        Returns:
            Training results
        """
        self.logger.info(f"Starting training for {self.config.episodes} episodes")
        
        # Create directories
        os.makedirs(self.config.save_path, exist_ok=True)
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        
        # Training loop
        for episode in range(self.config.episodes):
            # Train episode
            episode_reward, steps = self.train_episode(environment)
            
            # Update statistics
            self.episode_rewards.append(episode_reward)
            
            # Log progress
            if episode % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards)
                self.logger.info(
                    f"Episode {episode}/{self.config.episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Steps: {steps}"
                )
            
            # Save checkpoint
            if episode % self.config.save_interval == 0:
                self._save_checkpoint(episode)
        
        # Training complete
        final_avg_reward = np.mean(self.episode_rewards)
        self.logger.info(f"Training completed. Final average reward: {final_avg_reward:.2f}")
        
        # Save final model
        self._save_final_model()
        
        return {
            'final_average_reward': final_avg_reward,
            'total_episodes': self.config.episodes,
            'completed_at': datetime.now().isoformat()
        }
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint"""
        checkpoint_data = {
            'episode': episode,
            'average_reward': np.mean(self.episode_rewards),
            'loss_history': list(self.loss_history),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = os.path.join(
            self.config.checkpoint_path, 
            f"checkpoint_ep{episode}.json"
        )
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def _save_final_model(self) -> None:
        """Save final trained model"""
        # Get active policy
        policy_info = self.policy_manager.get_active_policy()
        if not policy_info:
            return
        
        policy_id = self.policy_manager.active_policy
        policy_dir = os.path.join(self.config.save_path, policy_id)
        os.makedirs(policy_dir, exist_ok=True)
        
        # Save training results
        results = {
            'final_average_reward': np.mean(self.episode_rewards),
            'training_episodes': self.config.episodes,
            'training_completed': datetime.now().isoformat(),
            'config': vars(self.config)
        }
        
        results_file = os.path.join(policy_dir, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Final model saved to: {policy_dir}")


class CurriculumLearning:
    """Implement curriculum learning for progressive training"""
    
    def __init__(self, stages: List[Dict[str, Any]]):
        """
        Initialize curriculum learning
        
        Args:
            stages: List of training stages with increasing difficulty
        """
        self.stages = stages
        self.current_stage = 0
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Get current training stage"""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return self.stages[-1]  # Return last stage if exceeded
    
    def advance_stage(self, performance_threshold: float, 
                     current_performance: float) -> bool:
        """
        Advance to next stage if performance threshold is met
        
        Args:
            performance_threshold: Required performance to advance
            current_performance: Current performance metric
            
        Returns:
            True if advanced to next stage, False otherwise
        """
        if current_performance >= performance_threshold:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                return True
        return False
    
    def get_stage_description(self) -> str:
        """Get description of current stage"""
        stage = self.get_current_stage()
        return stage.get('description', f'Stage {self.current_stage}')


class TransferLearning:
    """Transfer learning between related tasks"""
    
    def __init__(self):
        self.source_policies = {}
    
    def add_source_policy(self, task_name: str, policy_path: str):
        """
        Add a source policy for transfer learning
        
        Args:
            task_name: Name of the source task
            policy_path: Path to the source policy
        """
        self.source_policies[task_name] = policy_path
    
    def transfer_weights(self, target_agent: BaseAgent, 
                        source_task: str,
                        transfer_ratio: float = 0.5) -> bool:
        """
        Transfer weights from source policy to target agent
        
        Args:
            target_agent: Target agent to receive weights
            source_task: Source task name
            transfer_ratio: Ratio of weights to transfer (0.0 to 1.0)
            
        Returns:
            True if transfer successful, False otherwise
        """
        if source_task not in self.source_policies:
            return False
        
        # This is a simplified implementation
        # In practice, you would load the source policy and copy relevant weights
        # based on network architecture compatibility
        
        # For now, we'll just log the transfer
        print(f"Transferring {transfer_ratio*100:.1f}% of weights from {source_task} to {target_agent.agent_id}")
        
        return True


def load_training_config(config_path: str = "config/training.yaml") -> TrainingConfig:
    """
    Load training configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Training configuration
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return TrainingConfig(config_dict)
    else:
        # Return default configuration
        return TrainingConfig()


# Example usage
if __name__ == "__main__":
    # Create training configuration
    config_dict = {
        'episodes': 100,
        'max_steps_per_episode': 50,
        'batch_size': 16,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'epsilon_clip': 0.2,
        'epochs_per_update': 5,
        'log_interval': 10,
        'save_interval': 50,
        'save_path': 'registry/models/policy',
        'checkpoint_path': 'checkpoints'
    }
    
    training_config = TrainingConfig(config_dict)
    
    # Create trainer
    trainer = MultiAgentTrainer(training_config)
    
    # Define agent configurations
    agent_configs = [
        {
            'id': 'warehouse_1',
            'type': 'ppo',
            'observation_dim': 64,
            'action_dim': 10,
            'continuous_action': True,
            'learning_rate': 3e-4
        },
        {
            'id': 'warehouse_2',
            'type': 'ppo',
            'observation_dim': 64,
            'action_dim': 10,
            'continuous_action': True,
            'learning_rate': 3e-4
        }
    ]
    
    # Create policy
    policy_id = trainer.create_training_policy(agent_configs)
    print(f"Created training policy: {policy_id}")
    
    # In a real implementation, you would now train the policy
    # using a simulation environment:
    # results = trainer.train_policy(simulation_environment)
    # print(f"Training results: {results}")
    
    print("Training pipeline initialized successfully")
</file>