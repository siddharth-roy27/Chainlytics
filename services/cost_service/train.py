"""
Training pipeline for cost models
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from .cost_model import LinearCostModel
from .marginal_cost import MarginalCostAnalyzer


def load_historical_cost_data(data_path: str) -> pd.DataFrame:
    """
    Load historical cost data for training
    
    Args:
        data_path: Path to cost data
        
    Returns:
        DataFrame with historical cost data
    """
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        distance = np.random.uniform(50, 1000, n_samples)  # km
        weight = np.random.uniform(100, 5000, n_samples)   # kg
        holding_days = np.random.uniform(1, 30, n_samples)
        delay_hours = np.random.exponential(2, n_samples)
        units_handled = np.random.uniform(10, 500, n_samples)
        
        # Generate realistic cost components with some noise
        transport_cost = (
            distance * 2.5 + 
            weight * 0.8 + 
            np.random.normal(0, 50, n_samples)
        )
        
        holding_cost = (
            np.random.uniform(1000, 100000, n_samples) *  # inventory value
            0.005 * holding_days + 
            np.random.normal(0, 20, n_samples)
        )
        
        delay_penalty = (
            delay_hours * 50 + 
            np.random.normal(0, 10, n_samples)
        )
        
        # Total cost (target variable)
        total_cost = (
            transport_cost + holding_cost + delay_penalty +
            np.random.normal(0, 30, n_samples)
        )
        
        # Ensure non-negative costs
        total_cost = np.maximum(total_cost, 0)
        transport_cost = np.maximum(transport_cost, 0)
        holding_cost = np.maximum(holding_cost, 0)
        delay_penalty = np.maximum(delay_penalty, 0)
        
        data = pd.DataFrame({
            'distance': distance,
            'weight': weight,
            'holding_days': holding_days,
            'delay_hours': delay_hours,
            'units_handled': units_handled,
            'transport_cost': transport_cost,
            'holding_cost': holding_cost,
            'delay_penalty': delay_penalty,
            'total_cost': total_cost
        })
        
        return data


def train_cost_model(historical_data: pd.DataFrame,
                    target_variable: str = 'total_cost') -> Tuple[LinearRegression, Dict]:
    """
    Train a linear regression model on historical cost data
    
    Args:
        historical_data: DataFrame with historical cost data
        target_variable: Target variable to predict
        
    Returns:
        Tuple of (trained_model, model_metrics)
    """
    # Select feature columns
    feature_columns = [
        'distance', 'weight', 'holding_days', 
        'delay_hours', 'units_handled'
    ]
    
    # Prepare data
    X = historical_data[feature_columns]
    y = historical_data[target_variable]
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Evaluate model
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Feature importance (coefficients)
    feature_importance = dict(zip(feature_columns, model.coef_))
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'feature_importance': feature_importance,
        'intercept': model.intercept_
    }
    
    return model, metrics


def calibrate_cost_model(training_data_path: str = "data/historical_costs.csv",
                        model_save_path: str = "registry/models/cost/v1.0/model.pkl",
                        config_save_path: str = "registry/models/cost/v1.0/config.json") -> Dict:
    """
    Calibrate the cost model based on historical data
    
    Args:
        training_data_path: Path to historical cost data
        model_save_path: Path to save trained model
        config_save_path: Path to save model configuration
        
    Returns:
        Calibration results
    """
    # Load data
    historical_data = load_historical_cost_data(training_data_path)
    print(f"Loaded {len(historical_data)} historical cost records")
    
    # Train model for total cost
    model, metrics = train_cost_model(historical_data, 'total_cost')
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    
    # Generate calibrated parameters for LinearCostModel
    feature_importance = metrics['feature_importance']
    
    calibrated_params = {
        'transport_rate_per_km': max(0, feature_importance.get('distance', 2.5)),
        'transport_rate_per_kg': max(0, feature_importance.get('weight', 0.8)),
        'holding_rate_daily': max(0, feature_importance.get('holding_days', 0.005) / 10000),  # Normalize
        'delay_penalty_hourly': max(0, feature_importance.get('delay_hours', 50.0)),
        'handling_cost_per_unit': max(0, feature_importance.get('units_handled', 2.0) / 100),  # Normalize
        'fuel_cost_per_km': max(0, feature_importance.get('distance', 2.5) * 0.12),  # Assume 12% of transport
        'labor_cost_per_hour': 35.0,  # Fixed assumption
        'maintenance_cost_per_km': max(0, feature_importance.get('distance', 2.5) * 0.04)  # Assume 4% of transport
    }
    
    # Save configuration
    config = {
        'calibrated_parameters': calibrated_params,
        'training_metrics': metrics,
        'training_data_size': len(historical_data),
        'features_used': list(feature_importance.keys())
    }
    
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to {model_save_path}")
    print(f"Configuration saved to {config_save_path}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    
    return config


def validate_cost_model(test_data_path: str = None) -> Dict:
    """
    Validate the cost model performance
    
    Args:
        test_data_path: Path to test data (optional)
        
    Returns:
        Validation results
    """
    # Load test data or use a portion of training data
    if test_data_path and os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
    else:
        # Generate synthetic test data
        np.random.seed(123)  # Different seed for test data
        test_data = load_historical_cost_data("dummy_path")
    
    # Initialize cost model
    cost_model = LinearCostModel()
    analyzer = MarginalCostAnalyzer(cost_model)
    
    # Calculate predicted costs
    predicted_costs = []
    actual_costs = test_data['total_cost'].values
    
    for _, row in test_data.iterrows():
        cost_breakdown = cost_model.calculate_total_cost(
            distance=row['distance'],
            weight=row['weight'],
            holding_days=row['holding_days'],
            delay_hours=row['delay_hours'],
            units_handled=row['units_handled']
        )
        predicted_costs.append(cost_breakdown.total_cost)
    
    # Calculate validation metrics
    predicted_costs = np.array(predicted_costs)
    mse = mean_squared_error(actual_costs, predicted_costs)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_costs - predicted_costs))
    r2 = r2_score(actual_costs, predicted_costs)
    
    # Calculate accuracy within thresholds
    within_10_percent = np.mean(np.abs(actual_costs - predicted_costs) / actual_costs <= 0.1) * 100
    within_20_percent = np.mean(np.abs(actual_costs - predicted_costs) / actual_costs <= 0.2) * 100
    
    validation_results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy_within_10_percent': within_10_percent,
        'accuracy_within_20_percent': within_20_percent,
        'test_sample_size': len(test_data)
    }
    
    print("Validation Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Accuracy within 10%: {within_10_percent:.1f}%")
    print(f"  Accuracy within 20%: {within_20_percent:.1f}%")
    
    return validation_results


# Main training function
def main():
    """Main training function"""
    print("Starting cost model calibration...")
    
    # Calibrate model
    calibration_results = calibrate_cost_model()
    
    # Validate model
    validation_results = validate_cost_model()
    
    # Combine results
    results = {
        'calibration': calibration_results,
        'validation': validation_results
    }
    
    # Save results
    results_path = "registry/models/cost/v1.0/results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed. Results saved to {results_path}")
    return results


if __name__ == "__main__":
    main()