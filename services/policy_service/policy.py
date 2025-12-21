"""
Policy definitions and management for multi-agent RL system
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from datetime import datetime

from .agents import BaseAgent, PPOAgent, MultiAgentSystem


class PolicyType(Enum):
    """Types of policies"""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"


@dataclass
class PolicyConfig:
    """Configuration for a policy"""
    policy_id: str
    policy_type: PolicyType
    agent_configs: List[Dict[str, Any]]
    shared_network: bool = False
    coordination_mechanism: str = "independent"
    communication_protocol: str = "none"
    training_algorithm: str = "ppo"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


class PolicyManager:
    """Manage policies for multi-agent system"""
    
    def __init__(self, registry_path: str = "registry/policies"):
        """
        Initialize policy manager
        
        Args:
            registry_path: Path to policy registry
        """
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        self.policies = {}
        self.active_policy = None
    
    def create_policy(self, config: PolicyConfig) -> str:
        """
        Create a new policy based on configuration
        
        Args:
            config: Policy configuration
            
        Returns:
            Policy ID
        """
        # Create agents based on configuration
        agents = []
        for agent_config in config.agent_configs:
            agent = self._create_agent(agent_config)
            agents.append(agent)
        
        # Create multi-agent system
        mas = MultiAgentSystem(agents)
        
        # Store policy
        self.policies[config.policy_id] = {
            'config': config,
            'multi_agent_system': mas,
            'created_at': datetime.now().isoformat()
        }
        
        # Save policy configuration
        self._save_policy_config(config)
        
        return config.policy_id
    
    def _create_agent(self, agent_config: Dict[str, Any]) -> BaseAgent:
        """Create an agent from configuration"""
        agent_type = agent_config.get('type', 'ppo')
        agent_id = agent_config['id']
        observation_dim = agent_config['observation_dim']
        action_dim = agent_config['action_dim']
        
        if agent_type == 'ppo':
            return PPOAgent(
                agent_id=agent_id,
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=agent_config.get('hidden_dims', [256, 128]),
                continuous_action=agent_config.get('continuous_action', False),
                lr=agent_config.get('learning_rate', 3e-4),
                gamma=agent_config.get('gamma', 0.99),
                eps_clip=agent_config.get('eps_clip', 0.2)
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    def _save_policy_config(self, config: PolicyConfig) -> None:
        """Save policy configuration to file"""
        policy_dir = os.path.join(self.registry_path, config.policy_id)
        os.makedirs(policy_dir, exist_ok=True)
        
        config_path = os.path.join(policy_dir, "config.json")
        with open(config_path, 'w') as f:
            # Convert to serializable format
            config_dict = {
                'policy_id': config.policy_id,
                'policy_type': config.policy_type.value,
                'agent_configs': config.agent_configs,
                'shared_network': config.shared_network,
                'coordination_mechanism': config.coordination_mechanism,
                'communication_protocol': config.communication_protocol,
                'training_algorithm': config.training_algorithm,
                'hyperparameters': config.hyperparameters,
                'created_at': datetime.now().isoformat()
            }
            json.dump(config_dict, f, indent=2)
    
    def load_policy(self, policy_id: str) -> bool:
        """
        Load a policy from registry
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            True if loaded successfully, False otherwise
        """
        policy_dir = os.path.join(self.registry_path, policy_id)
        config_path = os.path.join(policy_dir, "config.json")
        
        if not os.path.exists(config_path):
            return False
        
        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert back to PolicyConfig
        config = PolicyConfig(
            policy_id=config_dict['policy_id'],
            policy_type=PolicyType(config_dict['policy_type']),
            agent_configs=config_dict['agent_configs'],
            shared_network=config_dict['shared_network'],
            coordination_mechanism=config_dict['coordination_mechanism'],
            communication_protocol=config_dict['communication_protocol'],
            training_algorithm=config_dict['training_algorithm'],
            hyperparameters=config_dict['hyperparameters']
        )
        
        # Recreate policy
        self.create_policy(config)
        return True
    
    def activate_policy(self, policy_id: str) -> bool:
        """
        Activate a policy for use
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            True if activated, False otherwise
        """
        if policy_id not in self.policies:
            return False
        
        self.active_policy = policy_id
        return True
    
    def get_active_policy(self) -> Optional[Dict]:
        """
        Get the currently active policy
        
        Returns:
            Active policy or None if none active
        """
        if self.active_policy is None:
            return None
        return self.policies.get(self.active_policy)
    
    def list_policies(self) -> List[str]:
        """
        List all available policies
        
        Returns:
            List of policy IDs
        """
        return list(self.policies.keys())
    
    def get_policy_config(self, policy_id: str) -> Optional[PolicyConfig]:
        """
        Get configuration for a policy
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            Policy configuration or None if not found
        """
        if policy_id not in self.policies:
            return None
        return self.policies[policy_id]['config']


class CooperativePolicy:
    """Cooperative policy for multi-agent coordination"""
    
    def __init__(self, agents: List[BaseAgent], shared_reward: bool = True):
        """
        Initialize cooperative policy
        
        Args:
            agents: List of agents
            shared_reward: Whether to share rewards among agents
        """
        self.agents = {agent.agent_id: agent for agent in agents}
        self.shared_reward = shared_reward
        self.team_reward = 0.0
    
    def coordinate_actions(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate actions among agents
        
        Args:
            observations: Observations for each agent
            
        Returns:
            Coordinated actions
        """
        actions = {}
        
        # In a more sophisticated implementation, this would implement
        # coordination mechanisms like:
        # 1. Centralized action selection
        # 2. Communication protocols
        # 3. Consensus algorithms
        # 4. Auction mechanisms
        
        # For now, we'll use independent action selection
        for agent_id, agent in self.agents.items():
            if agent_id in observations:
                # This is a placeholder - in practice, you'd pass coordinated observations
                actions[agent_id] = agent.select_action(observations[agent_id])
        
        return actions
    
    def distribute_reward(self, global_reward: float) -> Dict[str, float]:
        """
        Distribute global reward among agents
        
        Args:
            global_reward: Global team reward
            
        Returns:
            Individual rewards for each agent
        """
        self.team_reward += global_reward
        rewards = {}
        
        if self.shared_reward:
            # Equal distribution
            equal_share = global_reward / len(self.agents) if self.agents else 0
            for agent_id in self.agents:
                rewards[agent_id] = equal_share
        else:
            # Reward based on individual contribution
            # This is a simplified approach - in practice, you'd use more sophisticated
            # credit assignment methods
            for agent_id in self.agents:
                rewards[agent_id] = global_reward * np.random.dirichlet([1] * len(self.agents))[0]
        
        return rewards


class HierarchicalPolicy:
    """Hierarchical policy with manager and worker agents"""
    
    def __init__(self, manager_agent: BaseAgent, worker_agents: List[BaseAgent]):
        """
        Initialize hierarchical policy
        
        Args:
            manager_agent: High-level manager agent
            worker_agents: Low-level worker agents
        """
        self.manager = manager_agent
        self.workers = {agent.agent_id: agent for agent in worker_agents}
        self.task_assignments = {}
    
    def assign_tasks(self, global_observation: Any) -> Dict[str, Any]:
        """
        Assign tasks to worker agents
        
        Args:
            global_observation: Global system observation
            
        Returns:
            Task assignments for each worker
        """
        # Manager selects high-level strategy
        # This is a simplified implementation
        strategy = self.manager.select_action(global_observation)
        
        # Assign tasks to workers based on strategy
        assignments = {}
        worker_ids = list(self.workers.keys())
        
        # Simple round-robin assignment
        for i, worker_id in enumerate(worker_ids):
            task_type = f"task_type_{i % 3}"  # Simplified task types
            assignments[worker_id] = {
                'task_type': task_type,
                'priority': i,
                'resources': 1.0 / len(worker_ids)
            }
        
        self.task_assignments = assignments
        return assignments
    
    def execute_tasks(self, worker_observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute assigned tasks
        
        Args:
            worker_observations: Observations for each worker
            
        Returns:
            Actions taken by each worker
        """
        actions = {}
        for worker_id, worker in self.workers.items():
            if worker_id in worker_observations:
                # Worker executes assigned task
                actions[worker_id] = worker.select_action(worker_observations[worker_id])
        
        return actions


# Example usage
if __name__ == "__main__":
    # Create policy manager
    policy_manager = PolicyManager()
    
    # Define policy configuration
    agent_configs = [
        {
            'id': 'warehouse_1',
            'type': 'ppo',
            'observation_dim': 64,
            'action_dim': 10,
            'continuous_action': True
        },
        {
            'id': 'warehouse_2',
            'type': 'ppo',
            'observation_dim': 64,
            'action_dim': 10,
            'continuous_action': True
        }
    ]
    
    policy_config = PolicyConfig(
        policy_id='logistics_policy_v1',
        policy_type=PolicyType.COOPERATIVE,
        agent_configs=agent_configs,
        coordination_mechanism='shared_reward',
        training_algorithm='ppo'
    )
    
    # Create policy
    policy_id = policy_manager.create_policy(policy_config)
    print(f"Created policy: {policy_id}")
    
    # List policies
    policies = policy_manager.list_policies()
    print(f"Available policies: {policies}")
    
    # Activate policy
    activated = policy_manager.activate_policy(policy_id)
    print(f"Policy activated: {activated}")
