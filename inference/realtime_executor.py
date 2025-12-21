"""
Real-time Executor for Chainlytics Inference
"""
import logging
from typing import Dict, Any
from datetime import datetime
import asyncio
import json

from .decision_orchestrator import DecisionOrchestrator
from data.schemas.action_schema import ActionSchema
from services.safety_service.audit import SafetyAuditor

logger = logging.getLogger(__name__)

class RealtimeExecutor:
    """Executes decisions in real-time and manages the execution lifecycle."""
    
    def __init__(self, execution_mode: str = "simulation"):
        self.orchestrator = DecisionOrchestrator()
        self.safety_auditor = SafetyAuditor()
        self.execution_mode = execution_mode  # "simulation", "dry_run", or "production"
        self.execution_queue = []
        self.is_running = False
        
    async def start_execution_loop(self, interval_seconds: int = 60):
        """
        Start the real-time execution loop.
        
        Args:
            interval_seconds: How often to check for new decisions (in seconds)
        """
        logger.info(f"Starting real-time execution loop with {interval_seconds}s interval")
        self.is_running = True
        
        while self.is_running:
            try:
                await self._execute_pending_decisions()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def stop_execution_loop(self):
        """Stop the real-time execution loop."""
        logger.info("Stopping real-time execution loop")
        self.is_running = False
    
    async def submit_state_for_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a state for decision making and queue for execution.
        
        Args:
            state: Current logistics state
            
        Returns:
            Decision result
        """
        logger.info("Submitting state for decision making")
        
        # Make decision
        decision = self.orchestrator.make_decision(state)
        
        # Validate action
        action_validator = ActionSchema()
        is_valid, validation_errors = action_validator.validate(decision['action'])
        
        if not is_valid:
            logger.error(f"Invalid action generated: {validation_errors}")
            # Fallback to safe action
            safe_action = self._generate_safe_action(state)
            decision['action'] = safe_action
            decision['fallback_applied'] = True
        
        # Audit decision for safety
        audit_result = self.safety_auditor.audit_decision(decision)
        decision['safety_audit'] = audit_result
        
        # Queue for execution based on mode
        if self.execution_mode == "production":
            self.execution_queue.append(decision)
            logger.info("Decision queued for production execution")
        elif self.execution_mode == "dry_run":
            logger.info("Dry-run mode: Decision validated but not executed")
        else:  # simulation
            logger.info("Simulation mode: Decision validated but not executed")
        
        return decision
    
    async def _execute_pending_decisions(self):
        """Execute all pending decisions in the queue."""
        if not self.execution_queue:
            return
            
        logger.info(f"Executing {len(self.execution_queue)} pending decisions")
        
        executed_decisions = []
        
        for decision in self.execution_queue:
            try:
                # Execute action (this would interface with actual logistics systems)
                execution_result = await self._execute_action(decision['action'])
                
                # Update decision with execution results
                decision['execution_result'] = execution_result
                decision['executed_at'] = datetime.now().isoformat()
                
                # Log successful execution
                logger.info(f"Action executed successfully: {execution_result}")
                
                executed_decisions.append(decision)
                
            except Exception as e:
                logger.error(f"Failed to execute action: {e}")
                decision['execution_error'] = str(e)
                decision['executed_at'] = datetime.now().isoformat()
        
        # Clear executed decisions from queue
        self.execution_queue = [d for d in self.execution_queue if d not in executed_decisions]
        
        # Update outcomes in decision log
        self._update_decision_outcomes(executed_decisions)
    
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action (interface with logistics systems).
        
        Args:
            action: Action to execute
            
        Returns:
            Execution result
        """
        # In a real implementation, this would interface with:
        # - Warehouse management systems
        # - Fleet management systems
        # - Route optimization APIs
        # - Driver mobile apps
        # - Customer notification systems
        
        logger.info(f"Executing action: {action}")
        
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        # Return simulated execution result
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'details': 'Action executed successfully in simulation mode'
        }
    
    def _generate_safe_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a safe fallback action when validation fails."""
        # Simple safe action - do nothing or minimal action
        return {
            'type': 'safe_hold',
            'reason': 'validation_failed',
            'timestamp': datetime.now().isoformat(),
            'entities': []
        }
    
    def _update_decision_outcomes(self, executed_decisions: list):
        """Update decision log with execution outcomes."""
        # In a real implementation, this would update the decision log
        # with actual outcomes and KPI deltas
        for decision in executed_decisions:
            logger.info(f"Updated outcome for decision: {decision.get('execution_result')}")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize executor
    executor = RealtimeExecutor(execution_mode="simulation")
    
    # Example state
    example_state = {
        "warehouses": [
            {"id": "wh1", "location": {"lat": 40.7128, "lng": -74.0060}, "inventory": 1000}
        ],
        "vehicles": [
            {"id": "v1", "location": {"lat": 40.7128, "lng": -74.0060}, "capacity": 50}
        ],
        "orders": [
            {"id": "o1", "destination": {"lat": 40.7589, "lng": -73.9851}, "priority": 1}
        ]
    }
    
    # Run async execution
    async def main():
        # Submit state for decision
        decision = await executor.submit_state_for_decision(example_state)
        print(f"Decision: {decision}")
        
        # Start execution loop for 5 seconds then stop
        execution_task = asyncio.create_task(executor.start_execution_loop(1))
        await asyncio.sleep(5)
        await executor.stop_execution_loop()
    
    # Run the async main function
    asyncio.run(main())