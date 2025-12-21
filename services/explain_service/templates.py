"""
Template-based explanations for logistics ML models
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import os


@dataclass
class ExplanationTemplate:
    """Template for generating explanations"""
    template_id: str
    template_name: str
    template_text: str
    variables: List[str]
    model_types: List[str]
    explanation_type: str
    severity_levels: List[str]
    metadata: Dict[str, Any]


class TemplateManager:
    """Manage explanation templates"""
    
    def __init__(self, templates_file: str = "config/explanation_templates.json"):
        """
        Initialize template manager
        
        Args:
            templates_file: Path to templates configuration file
        """
        self.templates_file = templates_file
        self.templates = {}
        self.logger = logging.getLogger('TemplateManager')
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load templates from configuration file"""
        if os.path.exists(self.templates_file):
            try:
                with open(self.templates_file, 'r') as f:
                    templates_data = json.load(f)
                
                for template_id, template_data in templates_data.items():
                    self.templates[template_id] = ExplanationTemplate(**template_data)
                
                self.logger.info(f"Loaded {len(self.templates)} templates from {self.templates_file}")
            except Exception as e:
                self.logger.error(f"Error loading templates: {str(e)}")
        else:
            # Create default templates
            self._create_default_templates()
            self._save_templates()
    
    def _save_templates(self) -> None:
        """Save templates to configuration file"""
        try:
            # Convert templates to serializable format
            templates_data = {}
            for template_id, template in self.templates.items():
                templates_data[template_id] = {
                    'template_id': template.template_id,
                    'template_name': template.template_name,
                    'template_text': template.template_text,
                    'variables': template.variables,
                    'model_types': template.model_types,
                    'explanation_type': template.explanation_type,
                    'severity_levels': template.severity_levels,
                    'metadata': template.metadata
                }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.templates_file) if os.path.dirname(self.templates_file) else '.', exist_ok=True)
            
            with open(self.templates_file, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.templates)} templates to {self.templates_file}")
        except Exception as e:
            self.logger.error(f"Error saving templates: {str(e)}")
    
    def _create_default_templates(self) -> None:
        """Create default explanation templates"""
        default_templates = {
            'demand_forecast_high': ExplanationTemplate(
                template_id='demand_forecast_high',
                template_name='High Demand Forecast',
                template_text=(
                    "The {model_name} model predicts unusually high demand of {predicted_value} units "
                    "for the next {time_period}. This represents a {percentage_change}% increase from "
                    "the typical demand of {baseline_value} units.\n\n"
                    "Key contributing factors:\n{top_factors}\n\n"
                    "Recommended actions:\n{recommendations}"
                ),
                variables=['model_name', 'predicted_value', 'time_period', 'percentage_change', 
                          'baseline_value', 'top_factors', 'recommendations'],
                model_types=['forecast_model'],
                explanation_type='forecast',
                severity_levels=['medium', 'high'],
                metadata={'domain': 'demand_planning', 'created_by': 'system'}
            ),
            'route_optimization': ExplanationTemplate(
                template_id='route_optimization',
                template_name='Route Optimization Decision',
                template_text=(
                    "The {model_name} recommends optimizing delivery routes to reduce costs by "
                    "{cost_savings_percentage}%. The proposed solution involves {num_vehicles} vehicles "
                    "and covers {num_stops} stops.\n\n"
                    "Primary benefits:\n{benefits}\n\n"
                    "Implementation considerations:\n{considerations}"
                ),
                variables=['model_name', 'cost_savings_percentage', 'num_vehicles', 'num_stops',
                          'benefits', 'considerations'],
                model_types=['routing_model'],
                explanation_type='optimization',
                severity_levels=['low', 'medium'],
                metadata={'domain': 'logistics', 'created_by': 'system'}
            ),
            'anomaly_detection': ExplanationTemplate(
                template_id='anomaly_detection',
                template_name='Anomaly Detection Alert',
                template_text=(
                    "An anomaly was detected in {metric_name} with a value of {anomalous_value}, "
                    "which deviates {deviation_amount} from the expected value of {expected_value}.\n\n"
                    "Confidence level: {confidence}%\n"
                    "Severity: {severity}\n\n"
                    "Potential causes:\n{possible_causes}\n\n"
                    "Recommended investigation steps:\n{investigation_steps}"
                ),
                variables=['metric_name', 'anomalous_value', 'deviation_amount', 'expected_value',
                          'confidence', 'severity', 'possible_causes', 'investigation_steps'],
                model_types=['anomaly_detector'],
                explanation_type='anomaly',
                severity_levels=['low', 'medium', 'high', 'critical'],
                metadata={'domain': 'monitoring', 'created_by': 'system'}
            ),
            'inventory_recommendation': ExplanationTemplate(
                template_id='inventory_recommendation',
                template_name='Inventory Replenishment Recommendation',
                template_text=(
                    "Based on {model_name} analysis, recommend replenishing inventory for {product_name} "
                    "({product_id}). Current stock level of {current_stock} units is below the optimal "
                    "threshold of {optimal_threshold} units.\n\n"
                    "Projected stockout risk: {stockout_risk}%\n"
                    "Recommended order quantity: {recommended_quantity} units\n"
                    "Estimated delivery time: {delivery_time_days} days\n\n"
                    "Financial impact:\n{financial_impact}"
                ),
                variables=['model_name', 'product_name', 'product_id', 'current_stock',
                          'optimal_threshold', 'stockout_risk', 'recommended_quantity',
                          'delivery_time_days', 'financial_impact'],
                model_types=['inventory_model'],
                explanation_type='recommendation',
                severity_levels=['low', 'medium', 'high'],
                metadata={'domain': 'inventory_management', 'created_by': 'system'}
            )
        }
        
        self.templates = default_templates
    
    def get_template(self, template_id: str) -> Optional[ExplanationTemplate]:
        """
        Get template by ID
        
        Args:
            template_id: Template identifier
            
        Returns:
            Explanation template or None if not found
        """
        return self.templates.get(template_id)
    
    def find_templates(self, model_type: str = None, explanation_type: str = None,
                      severity_level: str = None) -> List[ExplanationTemplate]:
        """
        Find templates matching criteria
        
        Args:
            model_type: Model type to match
            explanation_type: Explanation type to match
            severity_level: Severity level to match
            
        Returns:
            List of matching templates
        """
        matching_templates = []
        
        for template in self.templates.values():
            # Check model type
            if model_type and model_type not in template.model_types:
                continue
            
            # Check explanation type
            if explanation_type and template.explanation_type != explanation_type:
                continue
            
            # Check severity level
            if severity_level and severity_level not in template.severity_levels:
                continue
            
            matching_templates.append(template)
        
        return matching_templates
    
    def add_template(self, template: ExplanationTemplate) -> None:
        """
        Add a new template
        
        Args:
            template: Template to add
        """
        self.templates[template.template_id] = template
        self._save_templates()
        self.logger.info(f"Added template: {template.template_name}")
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove a template
        
        Args:
            template_id: ID of template to remove
            
        Returns:
            True if removed, False if not found
        """
        if template_id in self.templates:
            del self.templates[template_id]
            self._save_templates()
            self.logger.info(f"Removed template: {template_id}")
            return True
        return False
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about templates
        
        Returns:
            Template statistics
        """
        if not self.templates:
            return {'total_templates': 0}
        
        # Count by explanation type
        type_counts = {}
        model_type_counts = {}
        severity_counts = {}
        
        for template in self.templates.values():
            # Explanation type counts
            exp_type = template.explanation_type
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
            
            # Model type counts
            for model_type in template.model_types:
                model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1
            
            # Severity level counts
            for severity in template.severity_levels:
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_templates': len(self.templates),
            'templates_by_explanation_type': type_counts,
            'templates_by_model_type': model_type_counts,
            'templates_by_severity': severity_counts,
            'domains': list(set(template.metadata.get('domain', 'unknown') 
                              for template in self.templates.values()))
        }


class TemplateBasedExplainer:
    """Generate explanations using templates"""
    
    def __init__(self, template_manager: TemplateManager):
        """
        Initialize template-based explainer
        
        Args:
            template_manager: Template manager instance
        """
        self.template_manager = template_manager
        self.logger = logging.getLogger('TemplateBasedExplainer')
    
    def generate_explanation(self, template_id: str, variables: Dict[str, Any]) -> str:
        """
        Generate explanation using template
        
        Args:
            template_id: ID of template to use
            variables: Variables to substitute in template
            
        Returns:
            Generated explanation text
        """
        template = self.template_manager.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Validate required variables
        missing_variables = [var for var in template.variables if var not in variables]
        if missing_variables:
            raise ValueError(f"Missing required variables: {missing_variables}")
        
        # Generate explanation by substituting variables
        explanation_text = template.template_text.format(**variables)
        
        self.logger.info(f"Generated explanation using template: {template_id}")
        return explanation_text
    
    def generate_explanation_with_context(self, model_type: str, explanation_type: str,
                                       severity_level: str, variables: Dict[str, Any]) -> Optional[str]:
        """
        Generate explanation by finding appropriate template based on context
        
        Args:
            model_type: Type of model
            explanation_type: Type of explanation
            severity_level: Severity level
            variables: Variables to substitute
            
        Returns:
            Generated explanation text or None if no suitable template found
        """
        # Find matching templates
        matching_templates = self.template_manager.find_templates(
            model_type=model_type,
            explanation_type=explanation_type,
            severity_level=severity_level
        )
        
        if not matching_templates:
            self.logger.warning(
                f"No templates found for model_type={model_type}, "
                f"explanation_type={explanation_type}, severity_level={severity_level}"
            )
            return None
        
        # Use the first matching template
        template = matching_templates[0]
        
        # Validate required variables
        missing_variables = [var for var in template.variables if var not in variables]
        if missing_variables:
            self.logger.warning(f"Missing variables for template {template.template_id}: {missing_variables}")
            # Try to fill missing variables with defaults or skip validation
            for var in missing_variables:
                variables[var] = "N/A"
        
        # Generate explanation
        try:
            explanation_text = template.template_text.format(**variables)
            self.logger.info(f"Generated contextual explanation using template: {template.template_id}")
            return explanation_text
        except KeyError as e:
            self.logger.error(f"Variable substitution error in template {template.template_id}: {str(e)}")
            return None
    
    def format_feature_list(self, features: Dict[str, float], top_k: int = 5) -> str:
        """
        Format feature importance list for templates
        
        Args:
            features: Dictionary of features and their importance/values
            top_k: Number of top features to include
            
        Returns:
            Formatted string
        """
        # Sort features by importance/value
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
        
        # Take top k features
        top_features = sorted_features[:top_k]
        
        # Format as bullet points
        formatted_lines = []
        for feature, value in top_features:
            if isinstance(value, float):
                formatted_lines.append(f"• {feature}: {value:.3f}")
            else:
                formatted_lines.append(f"• {feature}: {value}")
        
        return "\n".join(formatted_lines) if formatted_lines else "No significant features identified."
    
    def format_recommendations(self, recommendations: List[str]) -> str:
        """
        Format recommendations list for templates
        
        Args:
            recommendations: List of recommendation strings
            
        Returns:
            Formatted string
        """
        if not recommendations:
            return "No specific recommendations available."
        
        formatted_lines = [f"• {rec}" for rec in recommendations]
        return "\n".join(formatted_lines)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create template manager
    template_manager = TemplateManager("config/test_explanation_templates.json")
    
    # Get template statistics
    stats = template_manager.get_template_statistics()
    print("Template Statistics:")
    print(f"  Total templates: {stats['total_templates']}")
    print(f"  By explanation type: {stats['templates_by_explanation_type']}")
    
    # Create template-based explainer
    explainer = TemplateBasedExplainer(template_manager)
    
    # Generate explanation using specific template
    try:
        variables = {
            'model_name': 'DemandForecastModel-v2.1',
            'predicted_value': 1250,
            'time_period': 'next week',
            'percentage_change': 25,
            'baseline_value': 1000,
            'top_factors': explainer.format_feature_list({
                'seasonal_trend': 0.45,
                'marketing_campaign': 0.32,
                'economic_indicator': 0.18
            }),
            'recommendations': explainer.format_recommendations([
                "Increase inventory levels by 20%",
                "Adjust staffing schedules",
                "Coordinate with suppliers for additional capacity"
            ])
        }
        
        explanation = explainer.generate_explanation('demand_forecast_high', variables)
        print("\nGenerated Explanation:")
        print(explanation)
        
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
    
    # Generate contextual explanation
    contextual_explanation = explainer.generate_explanation_with_context(
        model_type='forecast_model',
        explanation_type='forecast',
        severity_level='high',
        variables=variables
    )
    
    if contextual_explanation:
        print("\nContextual Explanation:")
        print(contextual_explanation)
    else:
        print("\nNo contextual explanation generated")
</file>