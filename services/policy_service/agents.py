"""
Multi-agent reinforcement learning agents for logistics decision making
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AgentAction:
    """Action taken by an agent"""
    agent_id: str
    action_type: str
    action_params: Dict[str, Any]
    action_value: float = 0.0


@dataclass
class AgentObservation:
    """Observation received by an agent"""
    agent_id: str
    state_embedding: np.ndarray
    neighbor_info: Dict[str, Any]
    global_info: Dict[str, Any]
    reward: float = 0.0
    done: bool = False


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, observation_dim: int, action_dim: int):
        """
        Initialize agent
        
        Args:
            agent_id: Unique identifier for the agent
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
        """
        self.agent_id = agent_id
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.total_reward = 0.0
        self.step_count = 0
    
    @abstractmethod
    def select_action(self, observation: AgentObservation) -> AgentAction:
        """Select an action based on observation"""
        pass
    
    @abstractmethod
    def update(self, observation: AgentObservation, action: AgentAction, 
               reward: float, next_observation: AgentObservation) -> None:
        """Update agent based on experience"""
        pass
    
    def reset(self) -> None:
        """Reset agent state"""
        self.total_reward = 0.0
        self.step_count = 0


class PolicyNetwork(nn.Module):
    """Neural network for policy function"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 continuous_action: bool = False):
        """
        Initialize policy network
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            continuous_action: Whether action space is continuous
        """
        super(PolicyNetwork, self).__init__()
        
        self.continuous_action = continuous_action
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layers
        self.policy_head = nn.Linear(prev_dim, output_dim)
        
        if continuous_action:
            self.log_std = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (action_logits, log_std) for continuous actions
            or (action_logits, None) for discrete actions
        """
        x = self.hidden_layers(x)
        action_logits = self.policy_head(x)
        
        if self.continuous_action:
            return action_logits, self.log_std
        else:
            return action_logits, None


class ValueNetwork(nn.Module):
    """Neural network for value function"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Initialize value network
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
        """
        super(ValueNetwork, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Value estimate
        """
        return self.network(x)


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent"""
    
    def __init__(self, agent_id: str, observation_dim: int, action_dim: int,
                 hidden_dims: List[int] = None, continuous_action: bool = False,
                 lr: float = 3e-4, gamma: float = 0.99, eps_clip: float = 0.2):
        """
        Initialize PPO agent
        
        Args:
            agent_id: Unique identifier for the agent
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            continuous_action: Whether action space is continuous
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
        """
        super(PPOAgent, self).__init__(agent_id, observation_dim, action_dim)
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.continuous_action = continuous_action
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        # Networks
        self.policy_net = PolicyNetwork(observation_dim, hidden_dims, action_dim, continuous_action)
        self.value_net = ValueNetwork(observation_dim, hidden_dims)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Storage for experience
        self.memory = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'returns': [],
            'values': [],
            'advantages': []
        }
    
    def select_action(self, observation: AgentObservation) -> AgentAction:
        """
        Select action using current policy
        
        Args:
            observation: Current observation
            
        Returns:
            Selected action
        """
        # Convert observation to tensor
        state_tensor = torch.FloatTensor(observation.state_embedding).unsqueeze(0)
        
        # Get action probabilities
        with torch.no_grad():
            if self.continuous_action:
                action_mean, log_std = self.policy_net(state_tensor)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(action_mean, std)
            else:
                action_logits, _ = self.policy_net(state_tensor)
                dist = torch.distributions.Categorical(logits=action_logits)
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Get value estimate
            value = self.value_net(state_tensor)
        
        # Store experience
        self.memory['observations'].append(state_tensor)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        
        # Convert to action
        if self.continuous_action:
            action_params = {'action_vector': action.squeeze(0).cpu().numpy()}
        else:
            action_params = {'action_index': action.item()}
        
        return AgentAction(
            agent_id=self.agent_id,
            action_type='continuous' if self.continuous_action else 'discrete',
            action_params=action_params,
            action_value=value.item()
        )
    
    def update(self, observation: AgentObservation, action: AgentAction,
               reward: float, next_observation: AgentObservation) -> None:
        """
        Update agent (store reward, but actual training happens in batch)
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
        """
        # Store reward
        self.memory['returns'].append(reward)
        self.total_reward += reward
        self.step_count += 1
    
    def compute_returns_and_advantages(self, next_value: float) -> None:
        """
        Compute returns and advantages for PPO update
        
        Args:
            next_value: Value of next state
        """
        returns = []
        advantages = []
        
        # Compute returns
        gae = 0
        for i in reversed(range(len(self.memory['returns']))):
            if i == len(self.memory['returns']) - 1:
                next_value = next_value
            else:
                next_value = self.memory['values'][i + 1]
            
            delta = self.memory['returns'][i] + self.gamma * next_value - self.memory['values'][i]
            gae = delta + self.gamma * 0.95 * gae  # GAE lambda = 0.95
            returns.insert(0, gae + self.memory['values'][i])
            advantages.insert(0, gae)
        
        self.memory['returns'] = returns
        self.memory['advantages'] = advantages
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step
        
        Returns:
            Training metrics
        """
        if len(self.memory['observations']) == 0:
            return {}
        
        # Convert to tensors
        observations = torch.cat(self.memory['observations'])
        actions = torch.cat(self.memory['actions'])
        old_log_probs = torch.cat(self.memory['log_probs'])
        returns = torch.FloatTensor(self.memory['returns'])
        advantages = torch.FloatTensor(self.memory['advantages'])
        values = torch.cat(self.memory['values'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        if self.continuous_action:
            action_mean, log_std = self.policy_net(observations)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_mean, std)
        else:
            action_logits, _ = self.policy_net(observations)
            dist = torch.distributions.Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(torch.cat(self.memory['values']), returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        # Clear memory
        self.memory = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'returns': [],
            'values': [],
            'advantages': []
        }
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_return': returns.mean().item()
        }


class MultiAgentSystem:
    """Coordinate multiple agents"""
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize multi-agent system
        
        Args:
            agents: List of agents
        """
        self.agents = {agent.agent_id: agent for agent in agents}
        self.agent_ids = list(self.agents.keys())
    
    def select_actions(self, observations: Dict[str, AgentObservation]) -> Dict[str, AgentAction]:
        """
        Select actions for all agents
        
        Args:
            observations: Observations for each agent
            
        Returns:
            Actions for each agent
        """
        actions = {}
        for agent_id, agent in self.agents.items():
            if agent_id in observations:
                actions[agent_id] = agent.select_action(observations[agent_id])
        return actions
    
    def update_agents(self, observations: Dict[str, AgentObservation],
                     actions: Dict[str, AgentAction],
                     rewards: Dict[str, float],
                     next_observations: Dict[str, AgentObservation]) -> None:
        """
        Update all agents
        
        Args:
            observations: Previous observations
            actions: Actions taken
            rewards: Rewards received
            next_observations: New observations
        """
        for agent_id, agent in self.agents.items():
            if (agent_id in observations and agent_id in actions and 
                agent_id in rewards and agent_id in next_observations):
                agent.update(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id]
                )
    
    def train_step(self) -> Dict[str, Dict[str, float]]:
        """
        Perform training step for all agents
        
        Returns:
            Training metrics for each agent
        """
        metrics = {}
        for agent_id, agent in self.agents.items():
            if isinstance(agent, PPOAgent):
                metrics[agent_id] = agent.train_step()
        return metrics
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get a specific agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent or None if not found
        """
        return self.agents.get(agent_id)
    
    def reset_all(self) -> None:
        """Reset all agents"""
        for agent in self.agents.values():
            agent.reset()


# Example usage
if __name__ == "__main__":
    # Create sample agents
    agents = []
    for i in range(3):  # 3 warehouse agents
        agent = PPOAgent(
            agent_id=f"warehouse_{i}",
            observation_dim=64,  # State embedding dimension
            action_dim=10,       # Action dimension
            continuous_action=True
        )
        agents.append(agent)
    
    # Create multi-agent system
    mas = MultiAgentSystem(agents)
    
    print(f"Created multi-agent system with {len(agents)} agents")
    print(f"Agent IDs: {mas.agent_ids}")
