"""
Order dynamics for logistics simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class OrderPattern:
    """Pattern for order generation"""
    pattern_type: str  # 'constant', 'sinusoidal', 'trending', 'spike'
    base_rate: float   # Base orders per hour
    amplitude: float   # Amplitude for periodic patterns
    frequency: float   # Frequency for periodic patterns
    trend_slope: float # Slope for trending patterns
    spike_probability: float  # Probability of spikes
    spike_multiplier: float   # Multiplier for spike orders


class OrderDynamics:
    """Manage order generation and dynamics"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize order dynamics
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('OrderDynamics')
        self.patterns = self._initialize_patterns()
        self.order_history = []
    
    def _initialize_patterns(self) -> Dict[str, OrderPattern]:
        """Initialize order patterns from configuration"""
        patterns_config = self.config.get('order_patterns', {})
        patterns = {}
        
        # Default patterns if none configured
        if not patterns_config:
            patterns_config = {
                'weekday': {
                    'pattern_type': 'sinusoidal',
                    'base_rate': 2.0,
                    'amplitude': 1.5,
                    'frequency': 24.0,  # Daily cycle
                    'trend_slope': 0.0,
                    'spike_probability': 0.05,
                    'spike_multiplier': 3.0
                },
                'weekend': {
                    'pattern_type': 'constant',
                    'base_rate': 1.0,
                    'amplitude': 0.0,
                    'frequency': 1.0,
                    'trend_slope': 0.0,
                    'spike_probability': 0.02,
                    'spike_multiplier': 2.0
                }
            }
        
        for pattern_name, pattern_config in patterns_config.items():
            patterns[pattern_name] = OrderPattern(**pattern_config)
        
        return patterns
    
    def generate_orders(self, timestamp: datetime, customer_pool: List[str], 
                       previous_orders: int = 0) -> List[Dict[str, Any]]:
        """
        Generate orders for a given timestamp
        
        Args:
            timestamp: Current timestamp
            customer_pool: List of available customer IDs
            previous_orders: Number of orders in previous timestep
            
        Returns:
            List of generated orders
        """
        if not customer_pool:
            return []
        
        # Determine pattern based on day/time
        pattern = self._get_appropriate_pattern(timestamp)
        
        # Calculate expected order rate
        order_rate = self._calculate_order_rate(pattern, timestamp, previous_orders)
        
        # Generate number of orders using Poisson distribution
        num_orders = np.random.poisson(order_rate)
        
        # Apply spikes
        if np.random.random() < pattern.spike_probability:
            num_orders = int(num_orders * pattern.spike_multiplier)
            self.logger.info(f"Order spike detected: {num_orders} orders")
        
        # Generate individual orders
        orders = []
        for i in range(num_orders):
            order = self._generate_single_order(timestamp, customer_pool)
            orders.append(order)
        
        # Track order history
        self.order_history.append({
            'timestamp': timestamp,
            'num_orders': num_orders,
            'pattern_used': pattern.pattern_type,
            'rate': order_rate
        })
        
        return orders
    
    def _get_appropriate_pattern(self, timestamp: datetime) -> OrderPattern:
        """
        Get appropriate pattern based on timestamp
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Appropriate order pattern
        """
        # Weekend vs weekday
        if timestamp.weekday() >= 5:  # Saturday or Sunday
            return self.patterns.get('weekend', list(self.patterns.values())[0])
        else:
            return self.patterns.get('weekday', list(self.patterns.values())[0])
    
    def _calculate_order_rate(self, pattern: OrderPattern, timestamp: datetime, 
                            previous_orders: int) -> float:
        """
        Calculate order rate based on pattern
        
        Args:
            pattern: Order pattern to use
            timestamp: Current timestamp
            previous_orders: Number of orders in previous timestep
            
        Returns:
            Calculated order rate
        """
        base_rate = pattern.base_rate
        rate = base_rate
        
        # Apply pattern-specific calculations
        if pattern.pattern_type == 'sinusoidal':
            # Sinusoidal pattern (e.g., daily cycles)
            hours_since_midnight = timestamp.hour + timestamp.minute / 60.0
            rate += pattern.amplitude * np.sin(2 * np.pi * hours_since_midnight / pattern.frequency)
        
        elif pattern.pattern_type == 'trending':
            # Trending pattern
            days_since_start = (timestamp - datetime(timestamp.year, 1, 1)).days
            rate += pattern.trend_slope * days_since_start
        
        elif pattern.pattern_type == 'constant':
            # Constant rate
            pass
        
        # Ensure non-negative rate
        rate = max(0.0, rate)
        
        return rate
    
    def _generate_single_order(self, timestamp: datetime, 
                             customer_pool: List[str]) -> Dict[str, Any]:
        """
        Generate a single order
        
        Args:
            timestamp: Current timestamp
            customer_pool: List of available customers
            
        Returns:
            Generated order dictionary
        """
        # Select random customer
        customer_id = np.random.choice(customer_pool)
        
        # Generate order properties
        order_properties = {
            'id': f"ORDER_{timestamp.strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
            'customer_id': customer_id,
            'timestamp': timestamp,
            'quantity': self._generate_order_quantity(),
            'priority': self._generate_order_priority(),
            'delivery_deadline': timestamp + timedelta(
                hours=self._generate_delivery_deadline_hours()
            ),
            'product_type': self._generate_product_type(),
            'source_warehouse': None,  # Will be assigned later
            'status': 'pending',
            'estimated_cost': 0.0
        }
        
        return order_properties
    
    def _generate_order_quantity(self) -> int:
        """
        Generate order quantity
        
        Returns:
            Order quantity
        """
        # Most orders are small, some are large
        if np.random.random() < 0.8:
            # Small orders (1-10 units)
            return np.random.randint(1, 11)
        else:
            # Large orders (11-100 units)
            return np.random.randint(11, 101)
    
    def _generate_order_priority(self) -> int:
        """
        Generate order priority
        
        Returns:
            Order priority (1-3, where 3 is highest)
        """
        # Priority distribution: 70% normal, 20% high, 10% urgent
        return np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
    
    def _generate_delivery_deadline_hours(self) -> float:
        """
        Generate delivery deadline in hours from now
        
        Returns:
            Delivery deadline in hours
        """
        # Most deadlines are 24-48 hours, some shorter or longer
        return np.random.gamma(shape=2.0, scale=24.0)
    
    def _generate_product_type(self) -> str:
        """
        Generate product type
        
        Returns:
            Product type
        """
        product_types = ['standard', 'perishable', 'fragile', 'hazardous']
        # Distribution: 60% standard, 20% perishable, 15% fragile, 5% hazardous
        return np.random.choice(product_types, p=[0.6, 0.2, 0.15, 0.05])
    
    def get_order_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get order statistics for recent period
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Order statistics
        """
        if not self.order_history:
            return {'total_orders': 0}
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_orders = [
            entry for entry in self.order_history
            if entry['timestamp'] >= cutoff_time
        ]
        
        if not recent_orders:
            return {'total_orders': 0}
        
        total_orders = sum(entry['num_orders'] for entry in recent_orders)
        avg_rate = np.mean([entry['rate'] for entry in recent_orders])
        pattern_distribution = {}
        
        for entry in recent_orders:
            pattern = entry['pattern_used']
            pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + entry['num_orders']
        
        return {
            'total_orders': total_orders,
            'average_rate': avg_rate,
            'pattern_distribution': pattern_distribution,
            'time_period_hours': hours_back,
            'period_start': cutoff_time.isoformat(),
            'period_end': datetime.now().isoformat()
        }
    
    def adjust_patterns_based_on_feedback(self, feedback_metrics: Dict[str, float]) -> None:
        """
        Adjust order patterns based on feedback metrics
        
        Args:
            feedback_metrics: Dictionary of performance metrics
        """
        # This is a simplified adaptation mechanism
        # In practice, you would use more sophisticated learning algorithms
        
        satisfaction = feedback_metrics.get('customer_satisfaction', 1.0)
        delay_rate = feedback_metrics.get('delay_rate', 0.0)
        
        # Adjust patterns based on performance
        for pattern_name, pattern in self.patterns.items():
            if satisfaction < 0.8:
                # Reduce order rate if satisfaction is low
                pattern.base_rate *= 0.95
                self.logger.info(f"Reduced {pattern_name} pattern rate due to low satisfaction")
            elif delay_rate > 0.1:
                # Reduce order rate if delay rate is high
                pattern.base_rate *= 0.98
                self.logger.info(f"Reduced {pattern_name} pattern rate due to high delays")
            elif satisfaction > 0.95 and delay_rate < 0.05:
                # Increase order rate if performance is good
                pattern.base_rate *= 1.02
                self.logger.info(f"Increased {pattern_name} pattern rate due to good performance")
            
            # Ensure reasonable bounds
            pattern.base_rate = max(0.1, min(10.0, pattern.base_rate))


class SeasonalOrderAdjuster:
    """Adjust order patterns for seasonal variations"""
    
    def __init__(self):
        self.seasonal_factors = self._initialize_seasonal_factors()
        self.logger = logging.getLogger('SeasonalOrderAdjuster')
    
    def _initialize_seasonal_factors(self) -> Dict[str, float]:
        """Initialize seasonal adjustment factors"""
        # Factors relative to base rate (1.0 = no adjustment)
        return {
            'new_year': 0.7,      # January
            'valentines': 1.3,    # February
            'spring': 1.1,        # March-May
            'summer': 0.9,        # June-August
            'back_to_school': 1.4, # August-September
            'holiday_season': 1.8, # November-December
            'default': 1.0
        }
    
    def get_seasonal_factor(self, timestamp: datetime) -> float:
        """
        Get seasonal factor for timestamp
        
        Args:
            timestamp: Timestamp to evaluate
            
        Returns:
            Seasonal adjustment factor
        """
        month = timestamp.month
        
        if month == 1:
            return self.seasonal_factors['new_year']
        elif month == 2:
            return self.seasonal_factors['valentines']
        elif month in [3, 4, 5]:
            return self.seasonal_factors['spring']
        elif month in [6, 7, 8]:
            return self.seasonal_factors['summer']
        elif month in [8, 9]:
            return self.seasonal_factors['back_to_school']
        elif month in [11, 12]:
            return self.seasonal_factors['holiday_season']
        else:
            return self.seasonal_factors['default']
    
    def apply_seasonal_adjustment(self, base_rate: float, timestamp: datetime) -> float:
        """
        Apply seasonal adjustment to base rate
        
        Args:
            base_rate: Base order rate
            timestamp: Current timestamp
            
        Returns:
            Seasonally adjusted rate
        """
        factor = self.get_seasonal_factor(timestamp)
        adjusted_rate = base_rate * factor
        return adjusted_rate


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = {
        'order_patterns': {
            'weekday': {
                'pattern_type': 'sinusoidal',
                'base_rate': 2.5,
                'amplitude': 1.8,
                'frequency': 24.0,
                'trend_slope': 0.0,
                'spike_probability': 0.03,
                'spike_multiplier': 4.0
            },
            'weekend': {
                'pattern_type': 'constant',
                'base_rate': 1.2,
                'amplitude': 0.0,
                'frequency': 1.0,
                'trend_slope': 0.0,
                'spike_probability': 0.01,
                'spike_multiplier': 2.5
            }
        }
    }
    
    # Create order dynamics
    order_dynamics = OrderDynamics(config)
    
    # Create customer pool
    customer_pool = [f'CUST{i:04d}' for i in range(100)]
    
    # Generate orders for different times
    test_times = [
        datetime(2023, 6, 15, 10, 0),  # Thursday morning
        datetime(2023, 6, 17, 14, 0),  # Saturday afternoon
        datetime(2023, 6, 19, 18, 0)   # Monday evening
    ]
    
    for timestamp in test_times:
        orders = order_dynamics.generate_orders(timestamp, customer_pool)
        print(f"\nGenerated {len(orders)} orders for {timestamp}")
        if orders:
            print(f"Sample order: {orders[0]}")
    
    # Get statistics
    stats = order_dynamics.get_order_statistics(hours_back=48)
    print(f"\nOrder statistics (48 hours): {stats}")
    
    # Test seasonal adjuster
    seasonal_adjuster = SeasonalOrderAdjuster()
    
    # Test different seasons
    test_dates = [
        datetime(2023, 1, 15),   # Winter
        datetime(2023, 7, 15),   # Summer
        datetime(2023, 12, 15)   # Holiday season
    ]
    
    for date in test_dates:
        factor = seasonal_adjuster.get_seasonal_factor(date)
        adjusted_rate = seasonal_adjuster.apply_seasonal_adjustment(2.0, date)
        print(f"{date.strftime('%B')}: factor={factor:.2f}, adjusted_rate={adjusted_rate:.2f}")
</file>