"""
Explainability service for ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json


class ExplanationType(Enum):
    """Types of explanations"""
    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_BOUNDARY = "decision_boundary"
    COUNTERFACTUAL = "counterfactual"
    LOCAL_EXPLANATION = "local_explanation"
    GLOBAL_EXPLANATION = "global_explanation"
    CONFIDENCE_SCORE = "confidence_score"


class ExplanationMethod(Enum):
    """Explanation methods"""
    SHAP = "shap"
    LIME = "lime"
    GRADIENT = "gradient"
    RULE_BASED = "rule_based"
    TEMPLATE_BASED = "template_based"


@dataclass
class Explanation:
    """Explanation result"""
    explanation_id: str
    explanation_type: ExplanationType
    method: ExplanationMethod
    target_instance: Dict[str, Any]
    explanation_result: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    alternative_actions: Optional[List[Dict[str, Any]]] = None


class BaseExplainer:
    """Base class for explainers"""
    
    def __init__(self, model: Any, method: ExplanationMethod):
        self.model = model
        self.method = method
        self.logger = logging.getLogger(f'BaseExplainer.{method.value}')
    
    def explain(self, instance: Any, **kwargs) -> Explanation:
        """
        Generate explanation for instance
        
        Args:
            instance: Instance to explain
            **kwargs: Additional parameters
            
        Returns:
            Explanation result
        """
        raise NotImplementedError("Subclasses must implement explain method")


class FeatureImportanceExplainer(BaseExplainer):
    """Explain predictions based on feature importance"""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize feature importance explainer
        
        Args:
            model: Trained model with feature importance capability
            feature_names: Names of features
        """
        super().__init__(model, ExplanationMethod.SHAP)
        self.feature_names = feature_names
    
    def explain(self, instance: Union[np.ndarray, pd.DataFrame], 
               explanation_type: ExplanationType = ExplanationType.FEATURE_IMPORTANCE) -> Explanation:
        """
        Explain instance using feature importance
        
        Args:
            instance: Instance to explain
            explanation_type: Type of explanation
            
        Returns:
            Explanation result
        """
        # Convert instance to numpy array if needed
        if isinstance(instance, pd.DataFrame):
            instance_array = instance.values
        else:
            instance_array = np.asarray(instance)
        
        # Get feature importance (simplified - in practice, use SHAP or similar)
        feature_importance = self._calculate_feature_importance(instance_array)
        
        # Create explanation result
        explanation_result = {
            'feature_values': self._get_feature_values(instance_array),
            'importance_scores': feature_importance,
            'top_features': self._get_top_features(feature_importance, top_k=5)
        }
        
        # Calculate confidence based on feature importance distribution
        confidence = self._calculate_confidence(feature_importance)
        
        explanation = Explanation(
            explanation_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(instance_array)) % 10000}",
            explanation_type=explanation_type,
            method=self.method,
            target_instance=self._serialize_instance(instance),
            explanation_result=explanation_result,
            confidence=confidence,
            timestamp=datetime.now(),
            feature_importance=dict(zip(self.feature_names, feature_importance)),
            metadata={'model_type': type(self.model).__name__}
        )
        
        return explanation
    
    def _calculate_feature_importance(self, instance: np.ndarray) -> np.ndarray:
        """
        Calculate feature importance for instance
        
        Args:
            instance: Input instance
            
        Returns:
            Feature importance scores
        """
        # Simplified implementation - in practice, use SHAP or permutation importance
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            return np.abs(self.model.coef_)
        else:
            # Default: uniform importance
            return np.ones(instance.shape[1]) / instance.shape[1]
    
    def _get_feature_values(self, instance: np.ndarray) -> Dict[str, float]:
        """
        Get feature values from instance
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary of feature values
        """
        if len(instance.shape) == 1:
            values = instance
        else:
            values = instance[0]  # Take first row if 2D
        
        return dict(zip(self.feature_names, values))
    
    def _get_top_features(self, importance_scores: np.ndarray, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Get top features by importance
        
        Args:
            importance_scores: Importance scores
            top_k: Number of top features to return
            
        Returns:
            List of top features with scores
        """
        # Get indices of top features
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        
        # Create list of feature names and scores
        top_features = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                top_features.append({
                    'feature': self.feature_names[idx],
                    'importance': float(importance_scores[idx])
                })
        
        return top_features
    
    def _calculate_confidence(self, importance_scores: np.ndarray) -> float:
        """
        Calculate confidence based on feature importance distribution
        
        Args:
            importance_scores: Importance scores
            
        Returns:
            Confidence score (0-1)
        """
        # Simple confidence measure based on entropy of importance distribution
        # Lower entropy = higher confidence
        normalized_scores = importance_scores / np.sum(importance_scores)
        entropy = -np.sum(normalized_scores * np.log(normalized_scores + 1e-10))
        max_entropy = np.log(len(importance_scores))
        
        # Convert to confidence (1 - normalized entropy)
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        return max(0.0, min(1.0, confidence))
    
    def _serialize_instance(self, instance: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Serialize instance for storage
        
        Args:
            instance: Input instance
            
        Returns:
            Serialized instance
        """
        if isinstance(instance, pd.DataFrame):
            return instance.to_dict('records')[0] if len(instance) > 0 else {}
        elif isinstance(instance, np.ndarray):
            if len(instance.shape) == 1:
                values = instance
            else:
                values = instance[0]  # Take first row
            return dict(zip(self.feature_names, values))
        else:
            return {'instance': str(instance)}


class CounterfactualExplainer(BaseExplainer):
    """Generate counterfactual explanations"""
    
    def __init__(self, model: Any, feature_names: List[str], 
                 feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize counterfactual explainer
        
        Args:
            model: Trained model
            feature_names: Names of features
            feature_bounds: Bounds for feature values (min, max)
        """
        super().__init__(model, ExplanationMethod.RULE_BASED)
        self.feature_names = feature_names
        self.feature_bounds = feature_bounds or {}
    
    def explain(self, instance: Union[np.ndarray, pd.DataFrame], 
               desired_outcome: Any = None,
               max_changes: int = 3) -> Explanation:
        """
        Generate counterfactual explanation
        
        Args:
            instance: Instance to explain
            desired_outcome: Desired outcome (None = opposite of current prediction)
            max_changes: Maximum number of feature changes
            
        Returns:
            Counterfactual explanation
        """
        # Convert instance to numpy array
        if isinstance(instance, pd.DataFrame):
            instance_array = instance.values[0] if len(instance) > 0 else np.array([])
        else:
            instance_array = np.asarray(instance)
            if len(instance_array.shape) > 1:
                instance_array = instance_array[0]
        
        if instance_array.size == 0:
            raise ValueError("Empty instance provided")
        
        # Get current prediction
        current_prediction = self.model.predict(instance_array.reshape(1, -1))[0]
        
        # Determine desired outcome
        if desired_outcome is None:
            # For binary classification, flip the prediction
            if isinstance(current_prediction, (int, float)) and current_prediction in [0, 1]:
                desired_outcome = 1 - current_prediction
            else:
                desired_outcome = current_prediction  # For non-binary, keep same
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(
            instance_array, current_prediction, desired_outcome, max_changes
        )
        
        # Create explanation result
        explanation_result = {
            'current_prediction': float(current_prediction),
            'desired_outcome': float(desired_outcome),
            'counterfactuals': counterfactuals,
            'num_changes': len(counterfactuals)
        }
        
        # Calculate confidence (based on how close counterfactuals are to desired outcome)
        confidence = self._calculate_counterfactual_confidence(counterfactuals, desired_outcome)
        
        explanation = Explanation(
            explanation_id=f"cf_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(instance_array)) % 10000}",
            explanation_type=ExplanationType.COUNTERFACTUAL,
            method=self.method,
            target_instance=self._serialize_instance(instance),
            explanation_result=explanation_result,
            confidence=confidence,
            timestamp=datetime.now(),
            alternative_actions=counterfactuals,
            metadata={
                'model_type': type(self.model).__name__,
                'current_prediction': float(current_prediction),
                'desired_outcome': float(desired_outcome)
            }
        )
        
        return explanation
    
    def _generate_counterfactuals(self, instance: np.ndarray, 
                                current_prediction: Any,
                                desired_outcome: Any,
                                max_changes: int) -> List[Dict[str, Any]]:
        """
        Generate counterfactual examples
        
        Args:
            instance: Original instance
            current_prediction: Current model prediction
            desired_outcome: Desired outcome
            max_changes: Maximum number of changes
            
        Returns:
            List of counterfactual examples
        """
        counterfactuals = []
        
        # Try changing each feature individually
        for i, feature_name in enumerate(self.feature_names):
            if len(counterfactuals) >= max_changes:
                break
            
            # Get feature bounds
            feature_min, feature_max = self.feature_bounds.get(
                feature_name, 
                (np.min(instance) - 1, np.max(instance) + 1)
            )
            
            # Try increasing and decreasing the feature value
            original_value = instance[i]
            
            # Generate candidate values
            candidates = []
            if feature_max > original_value:
                candidates.append(min(feature_max, original_value + (feature_max - original_value) * 0.1))
            if feature_min < original_value:
                candidates.append(max(feature_min, original_value - (original_value - feature_min) * 0.1))
            
            # Test each candidate
            for candidate_value in candidates:
                if len(counterfactuals) >= max_changes:
                    break
                
                # Create modified instance
                modified_instance = instance.copy()
                modified_instance[i] = candidate_value
                
                # Get prediction
                try:
                    prediction = self.model.predict(modified_instance.reshape(1, -1))[0]
                    
                    # Check if prediction moves toward desired outcome
                    if self._is_better_prediction(prediction, current_prediction, desired_outcome):
                        counterfactual = {
                            'feature_changed': feature_name,
                            'original_value': float(original_value),
                            'new_value': float(candidate_value),
                            'predicted_outcome': float(prediction),
                            'improvement': abs(float(prediction) - float(current_prediction))
                        }
                        counterfactuals.append(counterfactual)
                        
                except Exception as e:
                    self.logger.warning(f"Error predicting for counterfactual: {str(e)}")
        
        return counterfactuals
    
    def _is_better_prediction(self, new_prediction: Any, 
                            current_prediction: Any, 
                            desired_outcome: Any) -> bool:
        """
        Check if new prediction is better than current (closer to desired)
        
        Args:
            new_prediction: New prediction
            current_prediction: Current prediction
            desired_outcome: Desired outcome
            
        Returns:
            True if new prediction is better
        """
        try:
            new_distance = abs(float(new_prediction) - float(desired_outcome))
            current_distance = abs(float(current_prediction) - float(desired_outcome))
            return new_distance < current_distance
        except (ValueError, TypeError):
            return str(new_prediction) != str(current_prediction)
    
    def _calculate_counterfactual_confidence(self, counterfactuals: List[Dict[str, Any]], 
                                          desired_outcome: Any) -> float:
        """
        Calculate confidence based on counterfactual quality
        
        Args:
            counterfactuals: Generated counterfactuals
            desired_outcome: Desired outcome
            
        Returns:
            Confidence score (0-1)
        """
        if not counterfactuals:
            return 0.0
        
        # Calculate average improvement
        total_improvement = sum(cf.get('improvement', 0) for cf in counterfactuals)
        avg_improvement = total_improvement / len(counterfactuals)
        
        # Normalize to 0-1 range (assuming improvement of 1.0 is good)
        confidence = min(1.0, avg_improvement)
        return confidence
    
    def _serialize_instance(self, instance: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Serialize instance for storage"""
        if isinstance(instance, pd.DataFrame):
            return instance.to_dict('records')[0] if len(instance) > 0 else {}
        elif isinstance(instance, np.ndarray):
            if len(instance.shape) == 1:
                values = instance
            else:
                values = instance[0]
            return dict(zip(self.feature_names, values))
        else:
            return {'instance': str(instance)}


class ExplanationAggregator:
    """Aggregate multiple explanations"""
    
    def __init__(self):
        self.explanations = []
        self.logger = logging.getLogger('ExplanationAggregator')
    
    def add_explanation(self, explanation: Explanation) -> None:
        """
        Add explanation to aggregator
        
        Args:
            explanation: Explanation to add
        """
        self.explanations.append(explanation)
        self.logger.info(f"Added explanation: {explanation.explanation_id}")
    
    def get_consensus_explanation(self, target_instance: Any) -> Optional[Explanation]:
        """
        Get consensus explanation for target instance
        
        Args:
            target_instance: Target instance
            
        Returns:
            Consensus explanation or None if no explanations available
        """
        if not self.explanations:
            return None
        
        # Filter explanations for target instance
        instance_explanations = [
            exp for exp in self.explanations 
            if self._instances_match(exp.target_instance, target_instance)
        ]
        
        if not instance_explanations:
            return None
        
        # Aggregate feature importance if available
        aggregated_importance = self._aggregate_feature_importance(instance_explanations)
        
        # Get most common explanation type
        type_counts = {}
        for exp in instance_explanations:
            type_counts[exp.explanation_type] = type_counts.get(exp.explanation_type, 0) + 1
        
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average confidence
        avg_confidence = np.mean([exp.confidence for exp in instance_explanations])
        
        # Create consensus explanation
        consensus_explanation = Explanation(
            explanation_id=f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(target_instance)) % 10000}",
            explanation_type=most_common_type,
            method=ExplanationMethod.TEMPLATE_BASED,  # Consensus method
            target_instance=target_instance,
            explanation_result={
                'individual_explanations': len(instance_explanations),
                'type_distribution': {k.value: v for k, v in type_counts.items()},
                'aggregated_feature_importance': aggregated_importance
            },
            confidence=float(avg_confidence),
            timestamp=datetime.now(),
            feature_importance=aggregated_importance,
            metadata={'aggregation_method': 'consensus'}
        )
        
        return consensus_explanation
    
    def _instances_match(self, instance1: Any, instance2: Any) -> bool:
        """
        Check if two instances match
        
        Args:
            instance1: First instance
            instance2: Second instance
            
        Returns:
            True if instances match
        """
        try:
            return json.dumps(instance1, sort_keys=True) == json.dumps(instance2, sort_keys=True)
        except:
            return str(instance1) == str(instance2)
    
    def _aggregate_feature_importance(self, explanations: List[Explanation]) -> Optional[Dict[str, float]]:
        """
        Aggregate feature importance across explanations
        
        Args:
            explanations: List of explanations
            
        Returns:
            Aggregated feature importance
        """
        if not explanations or not explanations[0].feature_importance:
            return None
        
        # Collect all feature importance scores
        all_importances = {}
        for exp in explanations:
            if exp.feature_importance:
                for feature, importance in exp.feature_importance.items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
        
        # Calculate mean importance for each feature
        aggregated_importance = {}
        for feature, importances in all_importances.items():
            aggregated_importance[feature] = float(np.mean(importances))
        
        return aggregated_importance
    
    def get_explanation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about explanations
        
        Returns:
            Explanation statistics
        """
        if not self.explanations:
            return {'total_explanations': 0}
        
        # Group by explanation type
        type_counts = {}
        method_counts = {}
        confidences = []
        
        for exp in self.explanations:
            # Count types
            type_counts[exp.explanation_type.value] = type_counts.get(exp.explanation_type.value, 0) + 1
            
            # Count methods
            method_counts[exp.method.value] = method_counts.get(exp.method.value, 0) + 1
            
            # Collect confidences
            confidences.append(exp.confidence)
        
        return {
            'total_explanations': len(self.explanations),
            'explanations_by_type': type_counts,
            'explanations_by_method': method_counts,
            'average_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'confidence_std': float(np.std(confidences)) if confidences else 0.0,
            'time_range': {
                'earliest': min(exp.timestamp for exp in self.explanations).isoformat(),
                'latest': max(exp.timestamp for exp in self.explanations).isoformat()
            }
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create mock model (simple linear model for demonstration)
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    feature_names = [f'feature_{i}' for i in range(5)]
    
    # Train simple model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Create feature importance explainer
    fi_explainer = FeatureImportanceExplainer(model, feature_names)
    
    # Explain a sample instance
    sample_instance = X[0].reshape(1, -1)
    explanation = fi_explainer.explain(sample_instance)
    
    print("Feature Importance Explanation:")
    print(f"  ID: {explanation.explanation_id}")
    print(f"  Type: {explanation.explanation_type.value}")
    print(f"  Confidence: {explanation.confidence:.3f}")
    print(f"  Top features: {explanation.explanation_result['top_features']}")
    
    # Create counterfactual explainer
    feature_bounds = {name: (np.min(X[:, i]), np.max(X[:, i])) for i, name in enumerate(feature_names)}
    cf_explainer = CounterfactualExplainer(model, feature_names, feature_bounds)
    
    # Generate counterfactual explanation
    cf_explanation = cf_explainer.explain(sample_instance, max_changes=2)
    
    print("\nCounterfactual Explanation:")
    print(f"  ID: {cf_explanation.explanation_id}")
    print(f"  Current prediction: {cf_explanation.explanation_result['current_prediction']}")
    print(f"  Desired outcome: {cf_explanation.explanation_result['desired_outcome']}")
    print(f"  Number of counterfactuals: {cf_explanation.explanation_result['num_changes']}")
    
    # Create explanation aggregator
    aggregator = ExplanationAggregator()
    aggregator.add_explanation(explanation)
    aggregator.add_explanation(cf_explanation)
    
    # Get statistics
    stats = aggregator.get_explanation_statistics()
    print(f"\nExplanation Statistics: {stats}")
</file>