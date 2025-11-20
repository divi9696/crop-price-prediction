import pandas as pd
import numpy as np
from typing import List, Dict
import yaml

class SyntheticDataGenerator:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rng = np.random.RandomState(self.config['model']['random_seed'])
        
    def generate_base_data(self) -> pd.DataFrame:
        """Generate base synthetic dataset"""
        data_config = self.config['data']
        
        # Define Indian markets with states
        markets_info = [
            {"market": "Kolkata", "state": "West Bengal", "district": "Kolkata"},
            {"market": "Delhi", "state": "Delhi", "district": "Delhi"},
            {"market": "Mumbai", "state": "Maharashtra", "district": "Mumbai"},
            {"market": "Chennai", "state": "Tamil Nadu", "district": "Chennai"},
            {"market": "Hyderabad", "state": "Telangana", "district": "Hyderabad"},
            {"market": "Bangalore", "state": "Karnataka", "district": "Bangalore"},
            {"market": "Ahmedabad", "state": "Gujarat", "district": "Ahmedabad"},
            {"market": "Pune", "state": "Maharashtra", "district": "Pune"},
            {"market": "Jaipur", "state": "Rajasthan", "district": "Jaipur"},
            {"market": "Lucknow", "state": "Uttar Pradesh", "district": "Lucknow"}
        ]
        
        crops_varieties = {
            "rice": ["PR 126", "Basmati", "Sona Masoori", "Swarna"],
            "wheat": ["Sharbati", "Lokwan", "MP Wheat", "Punjab Wheat"],
            "maize": ["Hybrid", "Desi", "Sweet Corn"]
        }
        
        dates = pd.date_range(
            start=data_config['start_date'],
            end=data_config['end_date'],
            freq='D'
        )
        
        records = []
        
        for date in dates:
            for market_info in markets_info:
                for crop, varieties in crops_varieties.items():
                    variety = self.rng.choice(varieties)
                    
                    # Base price with seasonality
                    base_price = self._generate_base_price(crop, date, market_info['state'])
                    
                    # Add random variations
                    price_variation = self.rng.normal(0, 50)
                    modal_price = max(1000, base_price + price_variation)
                    
                    # Generate min/max prices around modal
                    min_price = modal_price * self.rng.uniform(0.85, 0.98)
                    max_price = modal_price * self.rng.uniform(1.02, 1.15)
                    
                    # Generate arrival quantity with seasonality
                    arrival_qty = self._generate_arrival_qty(crop, date)
                    
                    # Generate weather data
                    rainfall, temp = self._generate_weather_data(date, market_info['state'])
                    
                    # Generate external factors
                    govt_policy = 1 if self.rng.random() < 0.02 else 0
                    holiday = 1 if date.weekday() in [5, 6] or self.rng.random() < 0.01 else 0
                    
                    records.append({
                        'date': date,
                        'state': market_info['state'],
                        'district': market_info['district'],
                        'market': market_info['market'],
                        'crop': crop,
                        'variety': variety,
                        'min_price': round(min_price, 2),
                        'max_price': round(max_price, 2),
                        'modal_price': round(modal_price, 2),
                        'arrival_qty': round(arrival_qty, 2),
                        'acreage_estimate': self._get_acreage_estimate(crop, market_info['state']),
                        'rainfall_mm': round(rainfall, 1),
                        'temp_c': round(temp, 1),
                        'govt_policy_flag': govt_policy,
                        'holiday_flag': holiday
                    })
        
        return pd.DataFrame(records)
    
    def _generate_base_price(self, crop: str, date: pd.Timestamp, state: str) -> float:
        """Generate base price with seasonality and trends"""
        # Base prices by crop (INR per quintal)
        base_prices = {
            "rice": 2000,
            "wheat": 1800,
            "maize": 1500
        }
        
        base = base_prices.get(crop, 2000)
        
        # State premium/discount
        state_factors = {
            "Delhi": 1.1, "Maharashtra": 1.05, "West Bengal": 1.0,
            "Tamil Nadu": 0.95, "Karnataka": 0.98, "Gujarat": 1.02,
            "Rajasthan": 0.92, "Uttar Pradesh": 0.90, "Telangana": 0.95
        }
        base *= state_factors.get(state, 1.0)
        
        # Seasonality - rice harvest peaks in Oct-Nov
        month = date.month
        if crop == "rice":
            # Lower prices during harvest (Oct-Nov), higher in lean season
            if month in [10, 11]:
                base *= 0.85  # Harvest season discount
            elif month in [6, 7]:
                base *= 1.15  # Lean season premium
        
        # Long-term trend (slight inflation)
        days_from_start = (date - pd.Timestamp(self.config['data']['start_date'])).days
        trend = 1 + (days_from_start / 365) * 0.03  # 3% annual increase
        
        return base * trend
    
    def _generate_arrival_qty(self, crop: str, date: pd.Timestamp) -> float:
        """Generate arrival quantity with seasonality"""
        base_qty = 1000  # quintals
        
        # Seasonality based on crop harvesting patterns
        month = date.month
        if crop == "rice":
            # Kharif crop - main harvest Oct-Nov
            if month in [10, 11]:
                base_qty *= 3.0
            elif month in [9, 12]:
                base_qty *= 1.5
        elif crop == "wheat":
            # Rabi crop - main harvest Mar-Apr
            if month in [3, 4]:
                base_qty *= 2.5
            elif month in [2, 5]:
                base_qty *= 1.3
        
        # Random variation
        variation = self.rng.lognormal(0, 0.3)
        return base_qty * variation
    
    def _generate_weather_data(self, date: pd.Timestamp, state: str) -> tuple:
        """Generate realistic weather data"""
        month = date.month
        
        # Monsoon season: Jun-Sept
        if month in [6, 7, 8, 9]:
            rainfall = self.rng.gamma(2, 15)  # Higher rainfall
            temp = self.rng.uniform(25, 32)
        else:
            rainfall = self.rng.gamma(1, 2)   # Lower rainfall
            if month in [4, 5, 10]:
                temp = self.rng.uniform(28, 38)  # Hot months
            else:
                temp = self.rng.uniform(18, 28)  # Cooler months
        
        return rainfall, temp
    
    def _get_acreage_estimate(self, crop: str, state: str) -> float:
        """Generate acreage estimates"""
        base_acreage = {
            "rice": 50000,
            "wheat": 45000,
            "maize": 30000
        }
        
        base = base_acreage.get(crop, 40000)
        # Add some state-wise variation
        variation = self.rng.uniform(0.8, 1.2)
        return base * variation

def main():
    """Generate and save synthetic data"""
    generator = SyntheticDataGenerator()
    print("Generating synthetic crop price data...")
    df = generator.generate_base_data()
    
    # Save to CSV
    output_path = "data/synthetic_crop_prices.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Generated {len(df):,} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Markets: {df['market'].nunique()}")
    print(f"Crops: {df['crop'].unique().tolist()}")

if __name__ == "__main__":
    main()