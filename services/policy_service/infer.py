"""
Inference module for multi-agent policy execution
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from datetime import datetime
import logging

from .agents import BaseAgent, AgentAction, AgentObservation
from .policy import PolicyManager, PolicyConfig
from .critic import MultiAgentCritic


class PolicyInferenceEngine:
    """Execute trained policies for decision making"""
    
    def __init__(self, policy_path: str = "registry/policies"):
        """
        Initialize inference engine
        
        Args:
            policy_path: Path to policy registry
        """
        self.policy_manager = PolicyManager(policy_path)
        self.active_policy_id = None
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for inference"""
        logger = logging.getLogger('PolicyInferenceEngine')
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
    
    def load_policy(self, policy_id: str) -> bool:
        """
        Load a trained policy for inference
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            success = self.policy_manager.load_policy(policy_id)
            if success:
                self.active_policy_id = policy_id
                self.logger.info(f"Loaded policy: {policy_id}")
            else:
                self.logger.error(f"Failed to load policy: {policy_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error loading policy {policy_id}: {str(e)}")
            return False
    
    def select_actions(self, observations: Dict[str, AgentObservation]) -> Dict[str, AgentAction]:
        """
        Select actions using the active policy
        
        Args:
            observations: Dictionary mapping agent IDs to observations
            
        Returns:
            Dictionary mapping agent IDs to actions
        """
        if not self.active_policy_id:
            raise ValueError("No active policy loaded")
        
        # Get active policy
        policy_info = self.policy_manager.get_active_policy()
        if not policy_info:
            raise ValueError("Active policy not found")
        
        multi_agent_system = policy_info['multi_agent_system']
        
        # Select actions
        actions = multi_agent_system.select_actions(observations)
        
        return actions
    
    def evaluate_policy(self, observations: Dict[str, AgentObservation]) -> Dict[str, Any]:
        """
        Evaluate policy performance on current observations
        
        Args:
            observations: Dictionary mapping agent IDs to observations
            
        Returns:
            Policy evaluation metrics
        """
        if not self.active_policy_id:
            raise ValueError("No active policy loaded")
        
        # Get active policy
        policy_info = self.policy_manager.get_active_policy()
        if not policy_info:
            raise ValueError("Active policy not found")
        
        # This is a simplified evaluation
        # In practice, you would compute more detailed metrics
        metrics = {
            'policy_id': self.active_policy_id,
            'num_agents': len(observations),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def get_policy_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the active policy
        
        Returns:
            Policy information or None if no active policy
        """
        if not self.active_policy_id:
            return None
        
        policy_info = self.policy_manager.get_active_policy()
        if not policy_info:
            return None
        
        # Extract relevant information
        config = policy_info['config']
        return {
            'policy_id': config.policy_id,
            'policy_type': config.policy_type.value,
            'num_agents': len(config.agent_configs),
            'created_at': policy_info['created_at'],
            'agent_ids': [agent_config['id'] for agent_config in config.agent_configs]
        }


class RealTimeDecisionMaker:
    """Make real-time decisions using trained policies"""
    
    def __init__(self, inference_engine: PolicyInferenceEngine):
        """
        Initialize real-time decision maker
        
        Args:
            inference_engine: Policy inference engine
        """
        self.inference_engine = inference_engine
        self.decision_history = []
        self.logger = logging.getLogger('RealTimeDecisionMaker')
    
    def make_decisions(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decisions based on current state
        
        Args:
            current_state: Current system state
            
        Returns:
            Decision results
        """
        try:
            # Convert state to agent observations
            observations = self._state_to_observations(current_state)
            
            # Select actions using policy
            actions = self.inference_engine.select_actions(observations)
            
            # Record decision
            decision_record = {
                'timestamp': datetime.now().isoformat(),
                'input_state': current_state,
                'observations': {k: {'agent_id': v.agent_id, 'state_shape': v.state_embedding.shape} 
                               for k, v in observations.items()},
                'actions': {k: {'agent_id': v.agent_id, 'action_type': v.action_type} 
                          for k, v in actions.items()}
            }
            self.decision_history.append(decision_record)
            
            # Convert actions to executable format
            executable_actions = self._actions_to_executable(actions)
            
            return {
                'actions': executable_actions,
                'decision_record': decision_record,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error making decisions: {str(e)}")
            return {
                'actions': {},
                'error': str(e),
                'success': False
            }
    
    def _state_to_observations(self, state: Dict[str, Any]) -> Dict[str, AgentObservation]:
        """
        Convert system state to agent observations
        
        Args:
            state: System state
            
        Returns:
            Dictionary mapping agent IDs to observations
        """
        observations = {}
        
        # Extract agent-specific information
        agent_positions = state.get('agent_positions', {})
        agent_embeddings = state.get('agent_embeddings', {})
        global_metrics = state.get('global_metrics', {})
        
        for agent_id, position in agent_positions.items():
            observation = AgentObservation(
                agent_id=agent_id,
                state_embedding=agent_embeddings.get(agent_id, np.zeros(64)),
                neighbor_info={},  # Would be populated with neighbor information
                global_info=global_metrics,
                reward=0.0,
                done=False
            )
            observations[agent_id] = observation
        
        return observations
    
    def _actions_to_executable(self, actions: Dict[str, AgentAction]) -> Dict[str, Any]:
        """
        Convert agent actions to executable format
        
        Args:
            actions: Agent actions
            
        Returns:
            Executable actions dictionary
        """
        executable = {}
        
        for agent_id, action in actions.items():
            if action.action_type == 'continuous':
                executable[agent_id] = {
                    'type': 'vector_action',
                    'vector': action.action_params.get('action_vector', []).tolist()
                }
            elif action.action_type == 'discrete':
                executable[agent_id] = {
                    'type': 'index_action',
                    'index': action.action_params.get('action_index', 0)
                }
            else:
                executable[agent_id] = {
                    'type': 'custom_action',
                    'params': action.action_params
                }
        
        return executable
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent decision history
        
        Args:
            limit: Maximum number of decisions to return
            
        Returns:
            List of recent decisions
        """
        return self.decision_history[-limit:]


class BatchDecisionProcessor:
    """Process decisions in batch mode"""
    
    def __init__(self, inference_engine: PolicyInferenceEngine):
        """
        Initialize batch decision processor
        
        Args:
            inference_engine: Policy inference engine
        """
        self.inference_engine = inference_engine
        self.batch_results = []
    
    def process_batch(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of states
        
        Args:
            states: List of system states
            
        Returns:
            List of decision results
        """
        results = []
        
        for i, state in enumerate(states):
            try:
                # Convert state to observations
                observations = self._state_to_observations(state)
                
                # Select actions
                actions = self.inference_engine.select_actions(observations)
                
                # Convert to executable format
                executable_actions = self._actions_to_executable(actions)
                
                results.append({
                    'batch_index': i,
                    'state': state,
                    'actions': executable_actions,
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'batch_index': i,
                    'state': state,
                    'error': str(e),
                    'success': False
                })
        
        # Store results
        batch_result = {
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'processed_at': datetime.now().isoformat(),
            'results': results,
            'successful_decisions': sum(1 for r in results if r['success']),
            'total_decisions': len(results)
        }
        self.batch_results.append(batch_result)
        
        return results
    
    def _state_to_observations(self, state: Dict[str, Any]) -> Dict[str, AgentObservation]:
        """Convert state to observations (same as in RealTimeDecisionMaker)"""
        observations = {}
        agent_positions = state.get('agent_positions', {})
        agent_embeddings = state.get('agent_embeddings', {})
        global_metrics = state.get('global_metrics', {})
        
        for agent_id, position in agent_positions.items():
            observation = AgentObservation(
                agent_id=agent_id,
                state_embedding=agent_embeddings.get(agent_id, np.zeros(64)),
                neighbor_info={},
                global_info=global_metrics,
                reward=0.0,
                done=False
            )
            observations[agent_id] = observation
        
        return observations
    
    def _actions_to_executable(self, actions: Dict[str, AgentAction]) -> Dict[str, Any]:
        """Convert actions to executable format (same as in RealTimeDecisionMaker)"""
        executable = {}
        
        for agent_id, action in actions.items():
            if action.action_type == 'continuous':
                executable[agent_id] = {
                    'type': 'vector_action',
                    'vector': action.action_params.get('action_vector', []).tolist()
                }
            elif action.action_type == 'discrete':
                executable[agent_id] = {
                    'type': 'index_action',
                    'index': action.action_params.get('action_index', 0)
                }
            else:
                executable[agent_id] = {
                    'type': 'custom_action',
                    'params': action.action_params
                }
        
        return executable


def load_latest_policy() -> PolicyInferenceEngine:
    """
    Load the latest trained policy for inference
    
    Returns:
        Policy inference engine with latest policy loaded
    """
    inference_engine = PolicyInferenceEngine()
    
    # Try to load the latest policy
    policy_manager = PolicyManager()
    latest_policy = policy_manager.get_latest_version()
    
    if latest_policy:
        inference_engine.load_policy(latest_policy)
    
    return inference_engine


# Example usage
if __name__ == "__main__":
    # Create inference engine
    inference_engine = PolicyInferenceEngine()
    
    # Create sample observations
    sample_observations = {
        'warehouse_1': AgentObservation(
            agent_id='warehouse_1',
            state_embedding=np.random.randn(64),
            neighbor_info={},
            global_info={'total_cost': 5000, 'service_level': 0.95},
            reward=0.0,
            done=False
        ),
        'warehouse_2': AgentObservation(
            agent_id='warehouse_2',
            state_embedding=np.random.randn(64),
            neighbor_info={},
            global_info={'total_cost': 5000, 'service_level': 0.95},
            reward=0.0,
            done=False
        )
    }
    
    print("Policy inference engine initialized")
    print(f"Sample observations created for {len(sample_observations)} agents")
    
    # In a real implementation, you would load a trained policy and make decisions:
    # inference_engine.load_policy("policy_v1.0")
    # actions = inference_engine.select_actions(sample_observations)
    # print(f"Selected actions for {len(actions)} agents")