"""
SHAP adapter for explainability service
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")


@dataclass
class SHAPExplanation:
    """SHAP explanation result"""
    shap_values: np.ndarray
    expected_value: float
    feature_names: List[str]
    instance_values: np.ndarray
    base_values: np.ndarray
    explanation_type: str  # 'local' or 'global'
    model_type: str
    timestamp: datetime


class SHAPExplainerAdapter:
    """Adapter for SHAP explainers"""
    
    def __init__(self, model: Any, feature_names: List[str], 
                 explainer_type: str = "auto"):
        """
        Initialize SHAP explainer adapter
        
        Args:
            model: Trained model
            feature_names: Names of features
            explainer_type: Type of SHAP explainer ('tree', 'linear', 'deep', 'kernel', 'auto')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available. Please install it with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.explainer = None
        self.logger = logging.getLogger('SHAPExplainerAdapter')
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize appropriate SHAP explainer"""
        try:
            if self.explainer_type == "auto":
                # Auto-detect explainer type based on model
                self.explainer = self._detect_and_create_explainer()
            elif self.explainer_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "linear":
                self.explainer = shap.LinearExplainer(self.model)
            elif self.explainer_type == "deep":
                self.explainer = shap.DeepExplainer(self.model)
            elif self.explainer_type == "kernel":
                self.explainer = shap.KernelExplainer(self.model.predict, self._get_background_data())
            else:
                raise ValueError(f"Unsupported explainer type: {self.explainer_type}")
            
            self.logger.info(f"Initialized {self.explainer_type} SHAP explainer")
            
        except Exception as e:
            self.logger.error(f"Error initializing SHAP explainer: {str(e)}")
            # Fall back to kernel explainer
            try:
                self.explainer = shap.KernelExplainer(self.model.predict, self._get_background_data())
                self.logger.info("Falling back to KernelExplainer")
            except Exception as fallback_e:
                self.logger.error(f"Fallback also failed: {str(fallback_e)}")
                raise
    
    def _detect_and_create_explainer(self):
        """Auto-detect and create appropriate explainer"""
        model_class = type(self.model).__module__ + "." + type(self.model).__name__
        
        # Tree-based models
        tree_models = [
            'sklearn.ensemble.RandomForestClassifier',
            'sklearn.ensemble.RandomForestRegressor',
            'sklearn.ensemble.GradientBoostingClassifier',
            'sklearn.ensemble.GradientBoostingRegressor',
            'sklearn.tree.DecisionTreeClassifier',
            'sklearn.tree.DecisionTreeRegressor',
            'xgboost.sklearn.XGBClassifier',
            'xgboost.sklearn.XGBRegressor',
            'lightgbm.sklearn.LGBMClassifier',
            'lightgbm.sklearn.LGBMRegressor'
        ]
        
        if model_class in tree_models:
            return shap.TreeExplainer(self.model)
        
        # Linear models
        linear_models = [
            'sklearn.linear_model.LogisticRegression',
            'sklearn.linear_model.LinearRegression',
            'sklearn.linear_model.Ridge',
            'sklearn.linear_model.Lasso'
        ]
        
        if model_class in linear_models:
            return shap.LinearExplainer(self.model)
        
        # Default to kernel explainer
        return shap.KernelExplainer(self.model.predict, self._get_background_data())
    
    def _get_background_data(self, sample_size: int = 100) -> np.ndarray:
        """Get background data for KernelExplainer"""
        # This is a placeholder - in practice, you'd use actual training data
        # For now, we'll create dummy data
        if hasattr(self.model, 'X_train'):
            # Use training data if available
            background_data = self.model.X_train
            if len(background_data) > sample_size:
                # Sample if too large
                indices = np.random.choice(len(background_data), sample_size, replace=False)
                background_data = background_data[indices]
            return background_data
        else:
            # Create dummy background data
            return np.random.randn(sample_size, len(self.feature_names))
    
    def explain_local(self, instance: Union[np.ndarray, pd.DataFrame]) -> SHAPExplanation:
        """
        Generate local SHAP explanation for single instance
        
        Args:
            instance: Single instance to explain
            
        Returns:
            SHAP explanation
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialized")
        
        try:
            # Convert instance to appropriate format
            if isinstance(instance, pd.DataFrame):
                instance_array = instance.values
            else:
                instance_array = np.asarray(instance)
            
            # Ensure 2D array
            if len(instance_array.shape) == 1:
                instance_array = instance_array.reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(instance_array)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case - take first class
                shap_values = shap_values[0]
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[0]  # Take first element
            else:
                expected_value = 0.0
            
            # Get base values (expected values for each instance)
            if hasattr(self.explainer, 'shap_values'):
                # For some explainers, we need to get base values differently
                base_values = np.full(len(instance_array), expected_value)
            else:
                base_values = np.array([expected_value])
            
            explanation = SHAPExplanation(
                shap_values=shap_values,
                expected_value=float(expected_value),
                feature_names=self.feature_names,
                instance_values=instance_array[0] if len(instance_array) > 0 else np.array([]),
                base_values=base_values,
                explanation_type='local',
                model_type=type(self.model).__name__,
                timestamp=datetime.now()
            )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating local SHAP explanation: {str(e)}")
            raise
    
    def explain_global(self, background_data: Union[np.ndarray, pd.DataFrame], 
                      max_evals: int = 1000) -> SHAPExplanation:
        """
        Generate global SHAP explanation
        
        Args:
            background_data: Background data for global explanation
            max_evals: Maximum number of evaluations
            
        Returns:
            Global SHAP explanation
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialized")
        
        try:
            # Convert background data to appropriate format
            if isinstance(background_data, pd.DataFrame):
                background_array = background_data.values
            else:
                background_array = np.asarray(background_data)
            
            # Sample if too large
            if len(background_array) > max_evals:
                indices = np.random.choice(len(background_array), max_evals, replace=False)
                background_array = background_array[indices]
            
            # Calculate SHAP values for background data
            shap_values = self.explainer.shap_values(background_array)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case - take first class or average
                if len(shap_values) > 0:
                    shap_values = np.mean(shap_values, axis=0)
                else:
                    shap_values = shap_values[0]
            
            # Calculate mean SHAP values for global importance
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = np.mean(expected_value)
            else:
                expected_value = 0.0
            
            explanation = SHAPExplanation(
                shap_values=mean_shap_values,
                expected_value=float(expected_value),
                feature_names=self.feature_names,
                instance_values=np.mean(background_array, axis=0),
                base_values=np.full(len(mean_shap_values), expected_value),
                explanation_type='global',
                model_type=type(self.model).__name__,
                timestamp=datetime.now()
            )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating global SHAP explanation: {str(e)}")
            raise
    
    def get_feature_importance(self, shap_explanation: SHAPExplanation) -> Dict[str, float]:
        """
        Extract feature importance from SHAP explanation
        
        Args:
            shap_explanation: SHAP explanation
            
        Returns:
            Dictionary of feature importances
        """
        # Use absolute SHAP values for importance
        importance_scores = np.abs(shap_explanation.shap_values)
        
        # Normalize to sum to 1
        if np.sum(importance_scores) > 0:
            importance_scores = importance_scores / np.sum(importance_scores)
        
        # Create dictionary mapping feature names to importance scores
        feature_importance = dict(zip(shap_explanation.feature_names, importance_scores))
        
        return feature_importance
    
    def visualize_explanation(self, shap_explanation: SHAPExplanation, 
                           plot_type: str = "waterfall") -> Any:
        """
        Create visualization of SHAP explanation
        
        Args:
            shap_explanation: SHAP explanation
            plot_type: Type of plot ('waterfall', 'bar', 'beeswarm')
            
        Returns:
            Visualization object (matplotlib figure or None)
        """
        try:
            import matplotlib.pyplot as plt
            
            if plot_type == "waterfall":
                # Create waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get feature contributions
                feature_names = shap_explanation.feature_names
                shap_values = shap_explanation.shap_values
                instance_values = shap_explanation.instance_values
                
                # Sort by absolute SHAP values
                sorted_indices = np.argsort(np.abs(shap_values))[::-1]
                
                # Plot
                y_pos = np.arange(len(feature_names))
                ax.barh(y_pos, shap_values[sorted_indices])
                ax.set_yticks(y_pos)
                ax.set_yticklabels([feature_names[i] for i in sorted_indices])
                ax.set_xlabel('SHAP Value')
                ax.set_title('Feature Contributions (Waterfall)')
                ax.grid(axis='x', alpha=0.3)
                
                return fig
                
            elif plot_type == "bar":
                # Create bar plot of feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                
                feature_importance = self.get_feature_importance(shap_explanation)
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                features, importances = zip(*sorted_features)
                y_pos = np.arange(len(features))
                
                ax.barh(y_pos, importances)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Importance')
                ax.set_title('Global Feature Importance (SHAP)')
                ax.grid(axis='x', alpha=0.3)
                
                return fig
                
            else:
                self.logger.warning(f"Unsupported plot type: {plot_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return None


class SHAPExplanationConverter:
    """Convert SHAP explanations to standard format"""
    
    @staticmethod
    def to_dict(shap_explanation: SHAPExplanation) -> Dict[str, Any]:
        """
        Convert SHAP explanation to dictionary
        
        Args:
            shap_explanation: SHAP explanation
            
        Returns:
            Dictionary representation
        """
        return {
            'shap_values': shap_explanation.shap_values.tolist(),
            'expected_value': shap_explanation.expected_value,
            'feature_names': shap_explanation.feature_names,
            'instance_values': shap_explanation.instance_values.tolist(),
            'base_values': shap_explanation.base_values.tolist(),
            'explanation_type': shap_explanation.explanation_type,
            'model_type': shap_explanation.model_type,
            'timestamp': shap_explanation.timestamp.isoformat()
        }
    
    @staticmethod
    def from_dict(explanation_dict: Dict[str, Any]) -> SHAPExplanation:
        """
        Create SHAP explanation from dictionary
        
        Args:
            explanation_dict: Dictionary representation
            
        Returns:
            SHAP explanation
        """
        from datetime import datetime
        
        return SHAPExplanation(
            shap_values=np.array(explanation_dict['shap_values']),
            expected_value=explanation_dict['expected_value'],
            feature_names=explanation_dict['feature_names'],
            instance_values=np.array(explanation_dict['instance_values']),
            base_values=np.array(explanation_dict['base_values']),
            explanation_type=explanation_dict['explanation_type'],
            model_type=explanation_dict['model_type'],
            timestamp=datetime.fromisoformat(explanation_dict['timestamp'])
        )


# Example usage
if __name__ == "__main__":
    if SHAP_AVAILABLE:
        # Import required libraries
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import warnings
        warnings.filterwarnings("ignore")
        
        # Create sample data
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Create DataFrame for easier handling
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Add training data to model for background data
        model.X_train = X[:100]  # Use first 100 samples as background
        
        # Create SHAP explainer adapter
        shap_adapter = SHAPExplainerAdapter(model, feature_names, explainer_type="auto")
        
        # Explain single instance
        sample_instance = X[0].reshape(1, -1)
        local_explanation = shap_adapter.explain_local(sample_instance)
        
        print("Local SHAP Explanation:")
        print(f"  Expected value: {local_explanation.expected_value:.3f}")
        print(f"  SHAP values shape: {local_explanation.shap_values.shape}")
        print(f"  Model type: {local_explanation.model_type}")
        
        # Get feature importance
        feature_importance = shap_adapter.get_feature_importance(local_explanation)
        print("\nTop 5 Feature Importances:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"  {feature}: {importance:.3f}")
        
        # Global explanation
        background_data = X[:100]  # Use subset for efficiency
        global_explanation = shap_adapter.explain_global(background_data)
        
        print("\nGlobal SHAP Explanation:")
        print(f"  Expected value: {global_explanation.expected_value:.3f}")
        print(f"  Feature names: {len(global_explanation.feature_names)}")
        
        # Convert to dictionary and back
        explanation_dict = SHAPExplanationConverter.to_dict(local_explanation)
        reconstructed = SHAPExplanationConverter.from_dict(explanation_dict)
        
        print(f"\nConversion test passed: {np.allclose(local_explanation.shap_values, reconstructed.shap_values)}")
        
        # Create visualization (if matplotlib is available)
        try:
            fig = shap_adapter.visualize_explanation(local_explanation, plot_type="waterfall")
            if fig:
                print("\nVisualization created successfully")
                # plt.show()  # Uncomment to display plot
                plt.close(fig)  # Close to free memory
        except Exception as e:
            print(f"\nVisualization error: {str(e)}")
    else:
        print("SHAP not available. Install with: pip install shap")
</file>