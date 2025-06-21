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

    # def calculate_safety_stock(self, item_id: int, ai_forecast_15_days: float) -> Dict:
    #     """
    #     Calculate safety stock using integrated AI forecast and ABC/XYZ analysis
    #     """
        
    #     # Load item data
    #     print(f"📋 Loading data for item ID {item_id}...")
    #     item_data = self.load_item_data(item_id)
        
    #     # Get ABC/XYZ classification
    #     abc_class, xyz_class = self.calculate_abc_xyz_classification(item_data)
    #     category = f"{abc_class}{xyz_class}"
        
    #     # Get weighted historical statistics
    #     stats = self.get_weighted_statistics(item_data['usage_history'], self.analysis_window)
        
    #     # Calculate daily forecast from 15-day total
    #     daily_forecast = ai_forecast_15_days / self.forecast_horizon
        
    #     # Get safety factor for this category
    #     safety_factor = self.safety_factors.get(category, 0.10)
    #     service_level = self.service_levels.get(category, 0.80)
        
    #     # Adjust safety factor based on forecast confidence and variability
    #     confidence_adjustment = 1 + (1 - self.forecast_accuracy) * 0.3
    #     variability_adjustment = 1 + min(stats['cv'], 0.5)  # Cap at 50% adjustment
        
    #     adjusted_safety_factor = safety_factor * confidence_adjustment * variability_adjustment
        
    #     # Calculate safety stock as percentage of forecast
    #     base_safety_stock = ai_forecast_15_days * adjusted_safety_factor
        
    #     # Apply lead time adjustment
    #     lead_time_days = 7  # Default 7-day lead time
    #     abc_category = category[0]
    #     adjusted_lead_time = lead_time_days * self.lead_time_multipliers.get(abc_category, 1.0)
        
    #     # Final safety stock calculation
    #     safety_stock = base_safety_stock * np.sqrt(adjusted_lead_time / 7)  # Scale by lead time
        
    #     # Calculate reorder point
    #     reorder_point = (daily_forecast * adjusted_lead_time) + safety_stock
        
    #     # Calculate days of supply
    #     days_of_supply = safety_stock / daily_forecast if daily_forecast > 0 else 0
        
    #     return {
    #         'item_id': item_id,
    #         'item_name': item_data['name'],
    #         'abc_xyz_category': category,
    #         'ai_forecast_15_days': ai_forecast_15_days,
    #         'daily_forecast': daily_forecast,
    #         'historical_daily_avg': stats['weighted_mean'],
    #         'historical_std': stats['weighted_std'],
    #         'cv': stats['cv'],
    #         'trend_factor': stats['trend_factor'],
    #         'base_safety_factor': safety_factor,
    #         'adjusted_safety_factor': adjusted_safety_factor,
    #         'confidence_adjustment': confidence_adjustment,
    #         'variability_adjustment': variability_adjustment,
    #         'service_level': service_level,
    #         'lead_time_days': adjusted_lead_time,
    #         'safety_stock': round(safety_stock, 0),
    #         'reorder_point': round(reorder_point, 0),
    #         'days_of_supply': round(days_of_supply, 1),
    #         'safety_stock_percentage': round((safety_stock / ai_forecast_15_days * 100), 1) if ai_forecast_15_days > 0 else 0,
    #         'data_points_analyzed': stats['data_points']
    #     }
    
    # def print_detailed_analysis(self, results: Dict):
    #     """Print comprehensive safety stock analysis"""
        
    #     print("\n" + "="*80)
    #     print("🏥 ADVANCED SAFETY STOCK ANALYSIS")
    #     print("="*80)
        
    #     print(f"\n📦 ITEM INFORMATION:")
    #     print(f"   ID: {results['item_id']}")
    #     print(f"   Name: {results['item_name']}")
    #     print(f"   ABC/XYZ Category: {results['abc_xyz_category']}")
        
    #     print(f"\n🤖 AI FORECAST DATA:")
    #     print(f"   15-Day Forecast: {results['ai_forecast_15_days']:,.0f} units")
    #     print(f"   Daily Forecast: {results['daily_forecast']:.1f} units/day")
        
    #     print(f"\n📊 HISTORICAL ANALYSIS:")
    #     print(f"   Data Points Analyzed: {results['data_points_analyzed']} days")
    #     print(f"   Historical Daily Avg: {results['historical_daily_avg']:.1f} units/day")
    #     print(f"   Standard Deviation: {results['historical_std']:.1f}")
    #     print(f"   Coefficient of Variation: {results['cv']:.3f}")
    #     print(f"   Trend Factor: {results['trend_factor']:.2f} (recent vs historical)")
        
    #     print(f"\n⚙️ SAFETY STOCK CALCULATION:")
    #     print(f"   Base Safety Factor: {results['base_safety_factor']:.1%}")
    #     print(f"   Confidence Adjustment: {results['confidence_adjustment']:.2f}x")
    #     print(f"   Variability Adjustment: {results['variability_adjustment']:.2f}x")
    #     print(f"   Final Safety Factor: {results['adjusted_safety_factor']:.1%}")
    #     print(f"   Service Level Target: {results['service_level']:.1%}")
    #     print(f"   Lead Time: {results['lead_time_days']:.1f} days")
        
    #     print(f"\n🎯 RESULTS:")
    #     print(f"   Safety Stock: {results['safety_stock']:.0f} units ({results['safety_stock_percentage']:.1f}% of forecast)")
    #     print(f"   Reorder Point: {results['reorder_point']:.0f} units")
    #     print(f"   Days of Supply: {results['days_of_supply']:.1f} days")
        
    #     # Strategic insights
    #     category = results['abc_xyz_category']
    #     abc_cat, xyz_cat = category[0], category[1]
        
    #     print(f"\n💡 STRATEGIC INSIGHTS:")
        
    #     abc_insight = {
    #         'A': "High-value item - Monitor closely, optimize procurement",
    #         'B': "Medium-value item - Standard monitoring, regular review", 
    #         'C': "Low-value item - Bulk ordering, minimal monitoring"
    #     }
        
    #     xyz_insight = {
    #         'X': "Predictable demand - Reliable planning, efficient ordering",
    #         'Y': "Moderate variability - Flexible procurement, buffer stock",
    #         'Z': "High variability - Safety margins, supplier diversity"
    #     }
        
    #     print(f"   ABC Strategy: {abc_insight.get(abc_cat, 'Standard approach')}")
    #     print(f"   XYZ Strategy: {xyz_insight.get(xyz_cat, 'Standard approach')}")
        
    #     # Forecast vs Historical comparison
    #     forecast_vs_historical = (results['daily_forecast'] / results['historical_daily_avg'] - 1) * 100 if results['historical_daily_avg'] > 0 else 0
        
    #     if abs(forecast_vs_historical) > 20:
    #         print(f"   ⚠️  Significant demand change: AI forecast is {forecast_vs_historical:+.1f}% vs historical")
    #     else:
    #         print(f"   ✅ Forecast aligned: AI forecast is {forecast_vs_historical:+.1f}% vs historical")
        
    #     print("="*80)

# def main():
#     """Main function to run safety stock calculation from command line"""
    
#     if len(sys.argv) != 3:
#         print("❌ Usage: python safety_stock_calculator.py <item_id> <ai_forecast_15_days>")
#         print("📝 Example: python safety_stock_calculator.py 15 12500")
#         sys.exit(1)
    
#     try:
#         item_id = int(sys.argv[1])
#         ai_forecast_15_days = float(sys.argv[2])
        
#         if item_id <= 0:
#             raise ValueError("Item ID must be positive")
#         if ai_forecast_15_days < 0:
#             raise ValueError("AI forecast must be non-negative")
            
#     except ValueError as e:
#         print(f"❌ Invalid arguments: {e}")
#         print("📝 Example: python safety_stock_calculator.py 15 12500")
#         sys.exit(1)
    
#     try:
#         # Initialize calculator
#         calculator = AdvancedSafetyStockCalculator(
#             forecast_horizon_days=15,
#             forecast_accuracy=0.85
#         )
        
#         # Calculate safety stock
#         results = calculator.calculate_safety_stock(item_id, ai_forecast_15_days)
        
#         # Print detailed analysis
#         calculator.print_detailed_analysis(results)
        
#     except FileNotFoundError as e:
#         print(f"❌ File not found: {e}")
#         print("📁 Make sure CSV files are in ../data/ directory")
#         sys.exit(1)
#     except ValueError as e:
#         print(f"❌ Data error: {e}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
