"""
Decision Orchestrator for Chainlytics Inference
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
import json

from services.state_service.encoder import StateEncoder
from services.forecast_service.infer import DemandForecaster
from services.policy_service.infer import PolicyInference
from services.routing_service.hybrid import HybridRoutingOptimizer
from services.safety_service.override import SafetyOverride
from services.cost_service.infer import CostPredictor
from services.anomaly_service.detector import AnomalyDetector
from services.explain_service.output import ExplanationGenerator
from simulation.constraints.constraints import ConstraintChecker
from data.schemas.log_schema import DecisionLog

logger = logging.getLogger(__name__)

class DecisionOrchestrator:
    """Orchestrates the decision-making flow for logistics optimization."""
    
    def __init__(self):
        # Initialize all services
        self.state_encoder = StateEncoder()
        self.demand_forecaster = DemandForecaster()
        self.policy_inference = PolicyInference()
        self.routing_optimizer = HybridRoutingOptimizer()
        self.safety_override = SafetyOverride()
        self.cost_predictor = CostPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.explanation_generator = ExplanationGenerator()
        self.constraint_checker = ConstraintChecker()
        
        # Decision log
        self.decision_log = []
        
    def make_decision(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a complete decision based on the current state.
        
        Args:
            raw_state: Raw logistics state from sensors/systems
            
        Returns:
            Decision result with action, explanation, and metadata
        """
        logger.info("Starting decision orchestration")
        decision_start_time = datetime.now()
        
        # 1. Encode state
        logger.info("Encoding state")
        encoded_state = self.state_encoder.encode(raw_state)
        
        # 2. Check for anomalies
        logger.info("Checking for anomalies")
        anomalies = self.anomaly_detector.detect(encoded_state)
        if anomalies:
            logger.warning(f"Anomalies detected: {anomalies}")
        
        # 3. Get demand forecast
        logger.info("Getting demand forecast")
        forecast = self.demand_forecaster.predict(encoded_state)
        
        # 4. Generate RL proposal
        logger.info("Generating RL policy proposal")
        rl_action = self.policy_inference.predict(encoded_state, forecast)
        
        # 5. Optimize routes
        logger.info("Optimizing routes")
        optimized_routes = self.routing_optimizer.optimize(rl_action)
        
        # 6. Check constraints
        logger.info("Checking constraints")
        constraint_violations = self.constraint_checker.check_all(optimized_routes)
        
        # 7. Apply safety overrides
        logger.info("Applying safety overrides")
        final_action = self.safety_override.apply(optimized_routes, constraint_violations)
        
        # 8. Predict costs
        logger.info("Predicting costs")
        cost_prediction = self.cost_predictor.predict(final_action)
        
        # 9. Generate explanation
        logger.info("Generating explanation")
        explanation = self.explanation_generator.generate(
            encoded_state, rl_action, final_action, constraint_violations
        )
        
        # 10. Log decision
        decision_result = {
            'action': final_action,
            'explanation': explanation,
            'forecast': forecast,
            'cost_prediction': cost_prediction,
            'anomalies': anomalies,
            'constraint_violations': constraint_violations,
            'processing_time_ms': (datetime.now() - decision_start_time).total_seconds() * 1000
        }
        
        # Log the decision
        self._log_decision(raw_state, decision_result)
        
        logger.info("Decision orchestration completed successfully")
        return decision_result
    
    def _log_decision(self, input_state: Dict[str, Any], decision_result: Dict[str, Any]):
        """Log the decision for learning and auditing."""
        log_entry = DecisionLog(
            timestamp=datetime.now().isoformat(),
            input_state=input_state,
            action_proposed=decision_result['action'],
            action_executed=decision_result['action'],  # In this case, same as proposed
            constraints_applied=decision_result['constraint_violations'],
            outcome=None,  # Would be filled in after execution
            kpi_delta=None,  # Would be filled in after execution
            processing_time_ms=decision_result['processing_time_ms']
        )
        
        self.decision_log.append(log_entry)
        
        # Also log to file/system
        logger.info(f"Decision logged: {log_entry}")
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decision history."""
        return [log.to_dict() for log in self.decision_log[-limit:]]
    
    def export_logs(self, filepath: str):
        """Export decision logs to file."""
        log_data = [log.to_dict() for log in self.decision_log]
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        logger.info(f"Decision logs exported to {filepath}")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize orchestrator
    orchestrator = DecisionOrchestrator()
    
    # Example state (would come from actual logistics systems)
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
    
    # Make decision
    decision = orchestrator.make_decision(example_state)
    print(f"Decision made: {decision}")