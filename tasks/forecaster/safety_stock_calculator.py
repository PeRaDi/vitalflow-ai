"""
Advanced Safety Stock Calculator with AI Forecast Integration

This module calculates safety stock for medical inventory items using:
1. ABC/XYZ analysis classification
2. AI forecast integration
3. Weighted historical data analysis
4. Dynamic service level calculation
"""

import numpy as np
from typing import Dict, List, Tuple

class SafetyStockCalculator:
    """
    Safety stock calculator that integrates AI forecasting with ABC/XYZ analysis
    """
    
    def __init__(self, db, item_id: int, lead_time: int, forecast_horizon_days: int, ai_forecast: float, forecast_accuracy: float = 0.85):
        self.db = db
        self.item_id = item_id
        self.lead_time = lead_time
        self.ai_forecast = float(ai_forecast)
        self.forecast_horizon_days = forecast_horizon_days
        self.forecast_accuracy = forecast_accuracy
        self.analysis_window = forecast_horizon_days * 6  # 90 days for 15-day forecast
        
        # Realistic safety stock factors (% of forecast) for AI-integrated approach
        self.safety_factors = {
            'AX': 0.12, 'AY': 0.18, 'AZ': 0.25,  # 12-25% for A items
            'BX': 0.08, 'BY': 0.15, 'BZ': 0.20,  # 8-20% for B items  
            'CX': 0.05, 'CY': 0.10, 'CZ': 0.15,  # 5-15% for C items
        }
        
        # Corresponding service levels
        self.service_levels = {
            'AX': 0.88, 'AY': 0.84, 'AZ': 0.80,  # 80-88% for A items
            'BX': 0.85, 'BY': 0.80, 'BZ': 0.75,  # 75-85% for B items
            'CX': 0.82, 'CY': 0.78, 'CZ': 0.70,  # 70-82% for C items
        }
        
        # Lead time multipliers by ABC category
        self.lead_time_multipliers = {
            'A': 1.1,  # 10% buffer for critical items
            'B': 1.0,  # Standard lead time
            'C': 0.9,  # 10% reduction for low-value items
        }
    
    def prepare_data(self, data: List[Tuple]) -> Dict:
        return {
            'id': self.item_id,
            'usage_history': [(str(date), float(quantity)) for date, quantity in data]
        }
            
    def calculate_abc_xyz_classification(self, item_data: Dict) -> Tuple[str, str]:
        """
        Calculate ABC and XYZ classification for the item
        For single item analysis, we'll use simplified thresholds
        """
        
        usage_values = [usage[1] for usage in item_data['usage_history']]
        total_usage = sum(usage_values)
        
        # Simplified ABC classification (without other items for comparison)
        # Using usage intensity as proxy
        daily_avg = total_usage / len(usage_values) if usage_values else 0
        
        if daily_avg > 800:  # High usage
            abc_class = 'A'
        elif daily_avg > 400:  # Medium usage
            abc_class = 'B'
        else:  # Low usage
            abc_class = 'C'
        
        # XYZ classification based on coefficient of variation
        if len(usage_values) > 1:
            mean_usage = np.mean(usage_values)
            std_usage = np.std(usage_values, ddof=1)
            cv = std_usage / mean_usage if mean_usage > 0 else 0
            
            if cv <= 0.25:
                xyz_class = 'X'  # Low variability
            elif cv <= 0.35:
                xyz_class = 'Y'  # Medium variability
            else:
                xyz_class = 'Z'  # High variability
        else:
            xyz_class = 'X'  # Default for insufficient data
        
        return abc_class, xyz_class
    
    def get_weighted_statistics(self, usage_history: List[Tuple], analysis_window: int) -> Dict:
        """
        Calculate weighted statistics using recent data with exponential weighting
        """
        
        # Get recent data within analysis window
        recent_usage = usage_history[-analysis_window:] if len(usage_history) > analysis_window else usage_history
        usage_values = [usage[1] for usage in recent_usage]
        
        if len(usage_values) < 2:
            return {
                'weighted_mean': usage_values[0] if usage_values else 0,
                'weighted_std': 0,
                'cv': 0,
                'data_points': len(usage_values),
                'trend_factor': 1.0
            }
        
        # Apply exponential weighting (recent data more important)
        weights = np.exp(np.linspace(-0.5, 0, len(usage_values)))
        weights = weights / weights.sum()  # Normalize weights
        
        # Calculate weighted statistics
        weighted_mean = np.average(usage_values, weights=weights)
        weighted_variance = np.average((np.array(usage_values) - weighted_mean)**2, weights=weights)
        weighted_std = np.sqrt(weighted_variance)
        
        cv = weighted_std / weighted_mean if weighted_mean > 0 else 0
        
        # Calculate trend factor (recent vs overall average)
        recent_month = usage_values[-30:] if len(usage_values) >= 30 else usage_values
        recent_avg = np.mean(recent_month)
        trend_factor = recent_avg / weighted_mean if weighted_mean > 0 else 1.0
        
        return {
            'weighted_mean': weighted_mean,
            'weighted_std': weighted_std,
            'cv': cv,
            'data_points': len(usage_values),
            'trend_factor': trend_factor
        }
    
    def exec(self):
        data = self.db.get_item_data(self.item_id)
    
        data = self.prepare_data(data)

        # Get ABC/XYZ classification
        abc_class, xyz_class = self.calculate_abc_xyz_classification(data)
        category = f"{abc_class}{xyz_class}"

        # Get weighted historical statistics
        stats = self.get_weighted_statistics(data['usage_history'], self.analysis_window)

        # Calculate daily forecast from n-days total
        daily_forecast = self.ai_forecast / self.forecast_horizon_days

        # Get safety factor for this category
        safety_factor = self.safety_factors.get(category, 0.10)
        service_level = self.service_levels.get(category, 0.80)
        
        # Adjust safety factor based on forecast confidence and variability
        confidence_adjustment = 1 + (1 - self.forecast_accuracy) * 0.3
        variability_adjustment = 1 + min(stats['cv'], 0.5)  # Cap at 50% adjustment

        adjusted_safety_factor = safety_factor * confidence_adjustment * variability_adjustment

        # Calculate safety stock as percentage of forecast
        base_safety_stock = self.ai_forecast * adjusted_safety_factor
        
        # Apply lead time adjustment
        abc_category = category[0]
        adjusted_lead_time = self.lead_time * self.lead_time_multipliers.get(abc_category, 1.0)
        
        # Final safety stock calculation
        safety_stock = base_safety_stock * np.sqrt(adjusted_lead_time / self.lead_time)  # Scale by lead time

        # Calculate reorder point
        reorder_point = (daily_forecast * adjusted_lead_time) + safety_stock
        
        # Calculate days of supply
        days_of_supply = safety_stock / daily_forecast if daily_forecast > 0 else 0

        return {
            'item_id': self.item_id,
            'abc_xyz_category': category,
            'ai_forecast': self.ai_forecast,
            'daily_forecast': daily_forecast,
            'historical_daily_avg': stats['weighted_mean'],
            'historical_std': stats['weighted_std'],
            'cv': stats['cv'],
            'trend_factor': stats['trend_factor'],
            'base_safety_factor': safety_factor,
            'adjusted_safety_factor': adjusted_safety_factor,
            'confidence_adjustment': confidence_adjustment,
            'variability_adjustment': variability_adjustment,
            'service_level': service_level,
            'lead_time_days': adjusted_lead_time,
            'safety_stock': round(safety_stock, 0),
            'reorder_point': round(reorder_point, 0),
            'days_of_supply': round(days_of_supply, 1),
            'safety_stock_percentage': round((safety_stock / self.ai_forecast * 100), 1) if self.ai_forecast > 0 else 0,
            'data_points_analyzed': stats['data_points']
        }