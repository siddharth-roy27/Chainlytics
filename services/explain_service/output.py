"""
Output formatting and serialization for explanations
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
import logging
from datetime import datetime


class ExplanationOutputFormatter:
    """Format explanations for different output formats"""
    
    def __init__(self):
        self.logger = logging.getLogger('ExplanationOutputFormatter')
    
    def to_json(self, explanation: Any, pretty: bool = True) -> str:
        """
        Convert explanation to JSON format
        
        Args:
            explanation: Explanation object or dictionary
            pretty: Whether to format JSON with indentation
            
        Returns:
            JSON string
        """
        try:
            # Handle dataclass objects
            if hasattr(explanation, '__dataclass_fields__'):
                explanation_dict = asdict(explanation)
            elif isinstance(explanation, dict):
                explanation_dict = explanation
            else:
                # Try to convert to dictionary
                explanation_dict = explanation.__dict__
            
            # Handle datetime objects
            explanation_dict = self._serialize_datetimes(explanation_dict)
            
            if pretty:
                return json.dumps(explanation_dict, indent=2, default=str)
            else:
                return json.dumps(explanation_dict, default=str)
                
        except Exception as e:
            self.logger.error(f"Error converting to JSON: {str(e)}")
            raise
    
    def to_dict(self, explanation: Any) -> Dict[str, Any]:
        """
        Convert explanation to dictionary format
        
        Args:
            explanation: Explanation object
            
        Returns:
            Dictionary representation
        """
        try:
            if hasattr(explanation, '__dataclass_fields__'):
                return asdict(explanation)
            elif isinstance(explanation, dict):
                return explanation
            else:
                return explanation.__dict__
        except Exception as e:
            self.logger.error(f"Error converting to dictionary: {str(e)}")
            raise
    
    def to_dataframe(self, explanations: List[Any]) -> pd.DataFrame:
        """
        Convert list of explanations to DataFrame
        
        Args:
            explanations: List of explanation objects
            
        Returns:
            DataFrame with explanation data
        """
        try:
            # Convert each explanation to dictionary
            explanation_dicts = []
            for exp in explanations:
                exp_dict = self.to_dict(exp)
                # Flatten nested dictionaries for better DataFrame representation
                flattened_dict = self._flatten_dict(exp_dict)
                explanation_dicts.append(flattened_dict)
            
            # Create DataFrame
            df = pd.DataFrame(explanation_dicts)
            
            # Convert timestamp columns if present
            timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
            for col in timestamp_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {str(e)}")
            raise
    
    def to_markdown(self, explanation: Any, include_charts: bool = False) -> str:
        """
        Convert explanation to Markdown format
        
        Args:
            explanation: Explanation object
            include_charts: Whether to include chart placeholders
            
        Returns:
            Markdown string
        """
        try:
            exp_dict = self.to_dict(explanation)
            
            markdown_lines = []
            markdown_lines.append(f"# Explanation: {exp_dict.get('explanation_id', 'Unknown')}")
            markdown_lines.append("")
            
            # Basic information
            markdown_lines.append("## Basic Information")
            markdown_lines.append(f"- **Type**: {exp_dict.get('explanation_type', 'Unknown')}")
            markdown_lines.append(f"- **Method**: {exp_dict.get('method', 'Unknown')}")
            markdown_lines.append(f"- **Confidence**: {exp_dict.get('confidence', 0.0):.3f}")
            markdown_lines.append(f"- **Timestamp**: {exp_dict.get('timestamp', 'Unknown')}")
            markdown_lines.append("")
            
            # Target instance
            markdown_lines.append("## Target Instance")
            target_instance = exp_dict.get('target_instance', {})
            if target_instance:
                for key, value in target_instance.items():
                    markdown_lines.append(f"- **{key}**: {value}")
            else:
                markdown_lines.append("- No instance data available")
            markdown_lines.append("")
            
            # Explanation result
            markdown_lines.append("## Explanation Result")
            explanation_result = exp_dict.get('explanation_result', {})
            if explanation_result:
                for key, value in explanation_result.items():
                    if isinstance(value, (list, dict)):
                        markdown_lines.append(f"- **{key}**:")
                        markdown_lines.append(f"  ```json")
                        markdown_lines.append(f"  {json.dumps(value, indent=2, default=str)}")
                        markdown_lines.append(f"  ```")
                    else:
                        markdown_lines.append(f"- **{key}**: {value}")
            else:
                markdown_lines.append("- No explanation result available")
            markdown_lines.append("")
            
            # Feature importance
            feature_importance = exp_dict.get('feature_importance')
            if feature_importance:
                markdown_lines.append("## Feature Importance")
                markdown_lines.append("| Feature | Importance |")
                markdown_lines.append("|---------|------------|")
                for feature, importance in sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True):
                    markdown_lines.append(f"| {feature} | {importance:.4f} |")
                markdown_lines.append("")
            
            # Alternative actions
            alternative_actions = exp_dict.get('alternative_actions')
            if alternative_actions:
                markdown_lines.append("## Alternative Actions")
                for i, action in enumerate(alternative_actions):
                    markdown_lines.append(f"### Action {i+1}")
                    for key, value in action.items():
                        markdown_lines.append(f"- **{key}**: {value}")
                    markdown_lines.append("")
            
            # Metadata
            metadata = exp_dict.get('metadata', {})
            if metadata:
                markdown_lines.append("## Metadata")
                for key, value in metadata.items():
                    markdown_lines.append(f"- **{key}**: {value}")
                markdown_lines.append("")
            
            # Chart placeholder
            if include_charts:
                markdown_lines.append("## Visualization")
                markdown_lines.append("![Explanation Chart](chart_placeholder.png)")
                markdown_lines.append("*Chart would be generated here*")
                markdown_lines.append("")
            
            return "\n".join(markdown_lines)
            
        except Exception as e:
            self.logger.error(f"Error converting to Markdown: {str(e)}")
            raise
    
    def to_html(self, explanation: Any, include_styles: bool = True) -> str:
        """
        Convert explanation to HTML format
        
        Args:
            explanation: Explanation object
            include_styles: Whether to include CSS styles
            
        Returns:
            HTML string
        """
        try:
            exp_dict = self.to_dict(explanation)
            
            html_lines = []
            
            # HTML header
            html_lines.append("<!DOCTYPE html>")
            html_lines.append("<html>")
            html_lines.append("<head>")
            html_lines.append("<meta charset='UTF-8'>")
            html_lines.append("<title>Explanation Report</title>")
            
            # CSS styles
            if include_styles:
                html_lines.append("<style>")
                html_lines.append("""
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2, h3 { color: #333; }
                    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .section { margin: 20px 0; }
                    .key-value { margin: 5px 0; }
                    .key { font-weight: bold; }
                """)
                html_lines.append("</style>")
            
            html_lines.append("</head>")
            html_lines.append("<body>")
            
            # Title
            html_lines.append(f"<h1>Explanation: {exp_dict.get('explanation_id', 'Unknown')}</h1>")
            
            # Basic information section
            html_lines.append("<div class='section'>")
            html_lines.append("<h2>Basic Information</h2>")
            html_lines.append("<div class='key-value'><span class='key'>Type:</span> "
                            f"{exp_dict.get('explanation_type', 'Unknown')}</div>")
            html_lines.append("<div class='key-value'><span class='key'>Method:</span> "
                            f"{exp_dict.get('method', 'Unknown')}</div>")
            html_lines.append("<div class='key-value'><span class='key'>Confidence:</span> "
                            f"{exp_dict.get('confidence', 0.0):.3f}</div>")
            html_lines.append("<div class='key-value'><span class='key'>Timestamp:</span> "
                            f"{exp_dict.get('timestamp', 'Unknown')}</div>")
            html_lines.append("</div>")
            
            # Target instance section
            html_lines.append("<div class='section'>")
            html_lines.append("<h2>Target Instance</h2>")
            target_instance = exp_dict.get('target_instance', {})
            if target_instance:
                html_lines.append("<table>")
                html_lines.append("<tr><th>Feature</th><th>Value</th></tr>")
                for key, value in target_instance.items():
                    html_lines.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                html_lines.append("</table>")
            else:
                html_lines.append("<p>No instance data available</p>")
            html_lines.append("</div>")
            
            # Explanation result section
            html_lines.append("<div class='section'>")
            html_lines.append("<h2>Explanation Result</h2>")
            explanation_result = exp_dict.get('explanation_result', {})
            if explanation_result:
                html_lines.append("<table>")
                html_lines.append("<tr><th>Key</th><th>Value</th></tr>")
                for key, value in explanation_result.items():
                    if isinstance(value, (list, dict)):
                        value_str = json.dumps(value, indent=2, default=str)
                        html_lines.append(f"<tr><td>{key}</td><td><pre>{value_str}</pre></td></tr>")
                    else:
                        html_lines.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                html_lines.append("</table>")
            else:
                html_lines.append("<p>No explanation result available</p>")
            html_lines.append("</div>")
            
            # Feature importance section
            feature_importance = exp_dict.get('feature_importance')
            if feature_importance:
                html_lines.append("<div class='section'>")
                html_lines.append("<h2>Feature Importance</h2>")
                html_lines.append("<table>")
                html_lines.append("<tr><th>Feature</th><th>Importance</th></tr>")
                for feature, importance in sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True):
                    html_lines.append(f"<tr><td>{feature}</td><td>{importance:.4f}</td></tr>")
                html_lines.append("</table>")
                html_lines.append("</div>")
            
            # Footer
            html_lines.append("</body>")
            html_lines.append("</html>")
            
            return "\n".join(html_lines)
            
        except Exception as e:
            self.logger.error(f"Error converting to HTML: {str(e)}")
            raise
    
    def _serialize_datetimes(self, obj: Any) -> Any:
        """
        Recursively serialize datetime objects to ISO format strings
        
        Args:
            obj: Object to serialize
            
        Returns:
            Object with datetime objects converted to strings
        """
        if isinstance(obj, dict):
            return {key: self._serialize_datetimes(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetimes(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionary
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # Handle list of dictionaries
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)


class ExplanationExporter:
    """Export explanations to various formats and destinations"""
    
    def __init__(self, formatter: ExplanationOutputFormatter):
        """
        Initialize exporter
        
        Args:
            formatter: Output formatter instance
        """
        self.formatter = formatter
        self.logger = logging.getLogger('ExplanationExporter')
    
    def export_to_file(self, explanation: Any, filepath: str, 
                      format: str = 'json') -> bool:
        """
        Export explanation to file
        
        Args:
            explanation: Explanation to export
            filepath: Path to output file
            format: Output format ('json', 'markdown', 'html', 'csv')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format.lower() == 'json':
                content = self.formatter.to_json(explanation, pretty=True)
            elif format.lower() == 'markdown':
                content = self.formatter.to_markdown(explanation)
            elif format.lower() == 'html':
                content = self.formatter.to_html(explanation)
            elif format.lower() == 'csv':
                # For CSV, we need to convert to DataFrame first
                df = self.formatter.to_dataframe([explanation])
                df.to_csv(filepath, index=False)
                self.logger.info(f"Exported explanation to CSV: {filepath}")
                return True
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Exported explanation to {format}: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to {format} file: {str(e)}")
            return False
    
    def export_multiple_to_file(self, explanations: List[Any], filepath: str,
                              format: str = 'json') -> bool:
        """
        Export multiple explanations to file
        
        Args:
            explanations: List of explanations to export
            filepath: Path to output file
            format: Output format ('json', 'csv')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format.lower() == 'json':
                # Convert all explanations to dictionaries
                exp_dicts = [self.formatter.to_dict(exp) for exp in explanations]
                content = json.dumps(exp_dicts, indent=2, default=str)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            elif format.lower() == 'csv':
                # Convert to DataFrame and save as CSV
                df = self.formatter.to_dataframe(explanations)
                df.to_csv(filepath, index=False)
                
            else:
                raise ValueError(f"Unsupported format for multiple export: {format}")
            
            self.logger.info(f"Exported {len(explanations)} explanations to {format}: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting multiple explanations: {str(e)}")
            return False
    
    def export_to_database(self, explanation: Any, connection: Any,
                         table_name: str = 'explanations') -> bool:
        """
        Export explanation to database
        
        Args:
            explanation: Explanation to export
            connection: Database connection object
            table_name: Name of table to insert into
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to dictionary
            exp_dict = self.formatter.to_dict(explanation)
            
            # Flatten for database insertion
            flattened_dict = self.formatter._flatten_dict(exp_dict)
            
            # Convert datetime to string for database compatibility
            for key, value in flattened_dict.items():
                if isinstance(value, datetime):
                    flattened_dict[key] = value.isoformat()
            
            # Insert into database (simplified - actual implementation depends on DB type)
            columns = ', '.join(flattened_dict.keys())
            placeholders = ', '.join(['%s'] * len(flattened_dict))
            values = list(flattened_dict.values())
            
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            
            # Execute query (this is a placeholder - actual implementation varies)
            cursor = connection.cursor()
            cursor.execute(query, values)
            connection.commit()
            
            self.logger.info(f"Exported explanation to database table: {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to database: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample explanation data
    sample_explanation = {
        'explanation_id': 'exp_20231201_123456_1234',
        'explanation_type': 'feature_importance',
        'method': 'shap',
        'target_instance': {
            'feature_1': 0.5,
            'feature_2': 1.2,
            'feature_3': -0.3
        },
        'explanation_result': {
            'feature_values': {'feature_1': 0.5, 'feature_2': 1.2, 'feature_3': -0.3},
            'importance_scores': [0.4, 0.35, 0.25],
            'top_features': [
                {'feature': 'feature_1', 'importance': 0.4},
                {'feature': 'feature_2', 'importance': 0.35}
            ]
        },
        'confidence': 0.85,
        'timestamp': datetime.now(),
        'feature_importance': {
            'feature_1': 0.4,
            'feature_2': 0.35,
            'feature_3': 0.25
        },
        'metadata': {
            'model_type': 'RandomForest',
            'dataset': 'logistics_data_v1'
        }
    }
    
    # Create formatter and exporter
    formatter = ExplanationOutputFormatter()
    exporter = ExplanationExporter(formatter)
    
    # Test JSON output
    json_output = formatter.to_json(sample_explanation)
    print("JSON Output:")
    print(json_output[:200] + "..." if len(json_output) > 200 else json_output)
    print()
    
    # Test Markdown output
    markdown_output = formatter.to_markdown(sample_explanation)
    print("Markdown Output:")
    print(markdown_output[:200] + "..." if len(markdown_output) > 200 else markdown_output)
    print()
    
    # Test HTML output
    html_output = formatter.to_html(sample_explanation)
    print("HTML Output (first 200 chars):")
    print(html_output[:200] + "..." if len(html_output) > 200 else html_output)
    print()
    
    # Test DataFrame conversion
    try:
        df = formatter.to_dataframe([sample_explanation])
        print("DataFrame Output:")
        print(df.head())
        print()
    except Exception as e:
        print(f"DataFrame conversion error: {str(e)}")
    
    # Test file export (uncomment to actually create files)
    # success = exporter.export_to_file(sample_explanation, "sample_explanation.json", "json")
    # print(f"JSON export successful: {success}")
    
    # success = exporter.export_to_file(sample_explanation, "sample_explanation.md", "markdown")
    # print(f"Markdown export successful: {success}")
    
    # success = exporter.export_to_file(sample_explanation, "sample_explanation.html", "html")
    # print(f"HTML export successful: {success}")
</file>