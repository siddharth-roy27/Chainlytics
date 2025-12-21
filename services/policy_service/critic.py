"""
Critic networks for multi-agent reinforcement learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CriticConfig:
    """Configuration for critic networks"""
    critic_type: str  # 'centralized', 'decentralized', 'global_state'
    input_dim: int
    hidden_dims: List[int]
    output_dim: int = 1  # Value estimation
    learning_rate: float = 3e-4
    gamma: float = 0.99


class BaseCritic(nn.Module, ABC):
    """Base class for critic networks"""
    
    def __init__(self, config: CriticConfig):
        """
        Initialize critic
        
        Args:
            config: Critic configuration
        """
        super(BaseCritic, self).__init__()
        self.config = config
        self.optimizer = None
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass"""
        pass
    
    @abstractmethod
    def compute_value(self, *args, **kwargs) -> float:
        """Compute value estimate"""
        pass
    
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """Update critic parameters"""
        pass


class StateValueCritic(BaseCritic):
    """State-value critic V(s)"""
    
    def __init__(self, config: CriticConfig):
        """
        Initialize state-value critic
        
        Args:
            config: Critic configuration
        """
        super(StateValueCritic, self).__init__(config)
        
        # Build network layers
        layers = []
        input_dim = config.input_dim
        hidden_dims = config.hidden_dims
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            Value estimate
        """
        return self.network(state)
    
    def compute_value(self, state: np.ndarray) -> float:
        """
        Compute value estimate for state
        
        Args:
            state: State array
            
        Returns:
            Value estimate
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.forward(state_tensor)
            return value.item()


class ActionValueCritic(BaseCritic):
    """Action-value critic Q(s,a)"""
    
    def __init__(self, config: CriticConfig):
        """
        Initialize action-value critic
        
        Args:
            config: Critic configuration
        """
        super(ActionValueCritic, self).__init__(config)
        
        # State processing layers
        state_layers = []
        state_input_dim = config.input_dim
        hidden_dims = config.hidden_dims[:-1] if len(config.hidden_dims) > 1 else [256]
        
        for hidden_dim in hidden_dims:
            state_layers.append(nn.Linear(state_input_dim, hidden_dim))
            state_layers.append(nn.ReLU())
            state_layers.append(nn.LayerNorm(hidden_dim))
            state_input_dim = hidden_dim
        
        self.state_processor = nn.Sequential(*state_layers)
        
        # Action processing layers
        action_layers = []
        action_input_dim = config.output_dim  # Assuming action dim is output_dim
        action_hidden_dims = [128]
        
        for hidden_dim in action_hidden_dims:
            action_layers.append(nn.Linear(action_input_dim, hidden_dim))
            action_layers.append(nn.ReLU())
            action_layers.append(nn.LayerNorm(hidden_dim))
            action_input_dim = hidden_dim
        
        self.action_processor = nn.Sequential(*action_layers)
        
        # Combined layers
        combined_dim = state_input_dim + action_input_dim
        final_layers = []
        final_hidden_dims = [config.hidden_dims[-1]] if config.hidden_dims else [128]
        
        for hidden_dim in final_hidden_dims:
            final_layers.append(nn.Linear(combined_dim, hidden_dim))
            final_layers.append(nn.ReLU())
            final_layers.append(nn.LayerNorm(hidden_dim))
            combined_dim = hidden_dim
        
        final_layers.append(nn.Linear(combined_dim, 1))
        self.final_processor = nn.Sequential(*final_layers)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value estimate
        """
        state_features = self.state_processor(state)
        action_features = self.action_processor(action)
        combined = torch.cat([state_features, action_features], dim=-1)
        q_value = self.final_processor(combined)
        return q_value
    
    def compute_value(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Compute Q-value estimate
        
        Args:
            state: State array
            action: Action array
            
        Returns:
            Q-value estimate
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            q_value = self.forward(state_tensor, action_tensor)
            return q_value.item()


class CentralizedCritic(BaseCritic):
    """Centralized critic for multi-agent systems"""
    
    def __init__(self, config: CriticConfig, num_agents: int):
        """
        Initialize centralized critic
        
        Args:
            config: Critic configuration
            num_agents: Number of agents in the system
        """
        super(CentralizedCritic, self).__init__(config)
        self.num_agents = num_agents
        
        # Global state processor (concatenated states from all agents)
        global_input_dim = config.input_dim * num_agents
        
        layers = []
        input_dim = global_input_dim
        hidden_dims = config.hidden_dims
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, config.output_dim))
        self.network = nn.Sequential(*layers)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            global_state: Concatenated states from all agents
            
        Returns:
            Global value estimate
        """
        return self.network(global_state)
    
    def compute_value(self, global_state: np.ndarray) -> float:
        """
        Compute global value estimate
        
        Args:
            global_state: Concatenated states from all agents
            
        Returns:
            Global value estimate
        """
        with torch.no_grad():
            global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
            value = self.forward(global_state_tensor)
            return value.item()


class MultiAgentCritic:
    """Manage critics for multi-agent systems"""
    
    def __init__(self, critic_configs: Dict[str, CriticConfig]):
        """
        Initialize multi-agent critic system
        
        Args:
            critic_configs: Dictionary mapping agent IDs to critic configurations
        """
        self.critics = {}
        self.global_critic = None
        
        # Create individual critics
        for agent_id, config in critic_configs.items():
            if config.critic_type == 'state_value':
                self.critics[agent_id] = StateValueCritic(config)
            elif config.critic_type == 'action_value':
                self.critics[agent_id] = ActionValueCritic(config)
            else:
                raise ValueError(f"Unsupported critic type: {config.critic_type}")
        
        # Check if we need a global critic
        if any(config.critic_type == 'centralized' for config in critic_configs.values()):
            # Find the centralized config
            centralized_config = None
            num_agents = 0
            for config in critic_configs.values():
                if config.critic_type == 'centralized':
                    centralized_config = config
                    num_agents += 1
            
            if centralized_config:
                self.global_critic = CentralizedCritic(centralized_config, num_agents)
    
    def compute_individual_values(self, agent_states: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute value estimates for individual agents
        
        Args:
            agent_states: Dictionary mapping agent IDs to their states
            
        Returns:
            Dictionary mapping agent IDs to value estimates
        """
        values = {}
        for agent_id, state in agent_states.items():
            if agent_id in self.critics:
                values[agent_id] = self.critics[agent_id].compute_value(state)
        return values
    
    def compute_global_value(self, global_state: np.ndarray) -> float:
        """
        Compute global value estimate
        
        Args:
            global_state: Concatenated states from all agents
            
        Returns:
            Global value estimate
        """
        if self.global_critic:
            return self.global_critic.compute_value(global_state)
        return 0.0
    
    def update_critic(self, agent_id: str, states: torch.Tensor, 
                     targets: torch.Tensor) -> Dict[str, float]:
        """
        Update a specific critic
        
        Args:
            agent_id: Agent identifier
            states: State tensors
            targets: Target values
            
        Returns:
            Training metrics
        """
        if agent_id not in self.critics:
            return {}
        
        critic = self.critics[agent_id]
        critic.optimizer.zero_grad()
        
        # Compute predictions
        predictions = critic(states)
        
        # Compute loss
        loss = F.mse_loss(predictions, targets)
        
        # Update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic.optimizer.step()
        
        return {'critic_loss': loss.item()}
    
    def update_global_critic(self, global_states: torch.Tensor,
                           targets: torch.Tensor) -> Dict[str, float]:
        """
        Update global critic
        
        Args:
            global_states: Global state tensors
            targets: Target values
            
        Returns:
            Training metrics
        """
        if not self.global_critic:
            return {}
        
        self.global_critic.optimizer.zero_grad()
        
        # Compute predictions
        predictions = self.global_critic(global_states)
        
        # Compute loss
        loss = F.mse_loss(predictions, targets)
        
        # Update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), 0.5)
        self.global_critic.optimizer.step()
        
        return {'global_critic_loss': loss.item()}


class AdvantageEstimator:
    """Estimate advantages for policy gradient methods"""
    
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize advantage estimator
        
        Args:
            gamma: Discount factor
            lam: GAE lambda parameter
        """
        self.gamma = gamma
        self.lam = lam
    
    def compute_td_errors(self, rewards: List[float], values: List[float],
                         next_values: List[float], dones: List[bool]) -> List[float]:
        """
        Compute TD errors
        
        Args:
            rewards: Rewards received
            values: Current state values
            next_values: Next state values
            dones: Done flags
            
        Returns:
            TD errors
        """
        td_errors = []
        for i in range(len(rewards)):
            if dones[i]:
                td_error = rewards[i] - values[i]
            else:
                td_error = rewards[i] + self.gamma * next_values[i] - values[i]
            td_errors.append(td_error)
        return td_errors
    
    def compute_advantages(self, rewards: List[float], values: List[float],
                          next_values: List[float], dones: List[bool]) -> List[float]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Rewards received
            values: Current state values
            next_values: Next state values
            dones: Done flags
            
        Returns:
            Advantages
        """
        # Compute TD errors
        td_errors = self.compute_td_errors(rewards, values, next_values, dones)
        
        # Compute GAE
        advantages = []
        gae = 0
        
        for i in reversed(range(len(td_errors))):
            gae = td_errors[i] + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def compute_returns(self, rewards: List[float], values: List[float],
                       next_values: List[float], dones: List[bool]) -> List[float]:
        """
        Compute discounted returns
        
        Args:
            rewards: Rewards received
            values: Current state values
            next_values: Next state values
            dones: Done flags
            
        Returns:
            Discounted returns
        """
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return returns


# Example usage
if __name__ == "__main__":
    # Create critic configurations
    critic_configs = {
        'agent_1': CriticConfig(
            critic_type='state_value',
            input_dim=64,
            hidden_dims=[256, 128],
            learning_rate=3e-4
        ),
        'agent_2': CriticConfig(
            critic_type='state_value',
            input_dim=64,
            hidden_dims=[256, 128],
            learning_rate=3e-4
        )
    }
    
    # Create multi-agent critic system
    ma_critic = MultiAgentCritic(critic_configs)
    
    print(f"Created multi-agent critic with {len(ma_critic.critics)} individual critics")
    
    # Test advantage estimation
    advantage_estimator = AdvantageEstimator()
    
    # Sample data
    rewards = [1.0, 2.0, 1.5, 0.5]
    values = [0.8, 1.2, 1.0, 0.6]
    next_values = [1.2, 1.0, 0.6, 0.0]  # Last next value is 0 (terminal)
    dones = [False, False, False, True]
    
    # Compute advantages
    advantages = advantage_estimator.compute_advantages(rewards, values, next_values, dones)
    returns = advantage_estimator.compute_returns(rewards, values, next_values, dones)
    
    print(f"Advantages: {[round(a, 3) for a in advantages]}")
    print(f"Returns: {[round(r, 3) for r in returns]}")
</file>