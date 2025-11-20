import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading and validation class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.expected_columns = [
            'date', 'state', 'district', 'market', 'crop', 'variety',
            'min_price', 'max_price', 'modal_price', 'arrival_qty',
            'acreage_estimate', 'rainfall_mm', 'temp_c', 
            'govt_policy_flag', 'holiday_flag'
        ]
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            
            # Validate schema
            self._validate_schema(df)
            
            # Sort by date and market
            df = df.sort_values(['market', 'crop', 'date']).reset_index(drop=True)
            
            logger.info(f"✅ Loaded {len(df):,} records")
            logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"🏪 Markets: {df['market'].nunique()}")
            logger.info(f"🌾 Crops: {df['crop'].unique().tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate data schema"""
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Check for critical nulls
        critical_cols = ['date', 'market', 'crop', 'modal_price']
        null_counts = df[critical_cols].isnull().sum()
        
        for col, null_count in null_counts.items():
            if null_count > 0:
                logger.warning(f"Column {col} has {null_count} null values")
        
        # Basic data quality checks
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        
        if df['modal_price'].min() <= 0:
            logger.warning("Some price values are zero or negative")

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Data preprocessing pipeline"""
    
    def __init__(self, config: Dict, target_col: str = 'modal_price'):
        self.config = config
        self.target_col = target_col
        self.aggregation_freq = config['model']['aggregation']
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps"""
        logger.info("🔄 Starting data preprocessing")
        
        if len(df) == 0:
            raise ValueError("Cannot preprocess empty dataframe")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Remove outliers
        df_processed = self._handle_outliers(df_processed)
        
        # Aggregate data
        df_processed = self._aggregate_data(df_processed)
        
        logger.info(f"✅ Data shape after preprocessing: {df_processed.shape}")
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        df_filled = df.copy()
        
        # Group by market and crop for time-series imputation
        group_cols = ['market', 'crop', 'variety']
        
        # Forward fill then backward fill for time-series data
        price_cols = ['min_price', 'max_price', 'modal_price']
        for col in price_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled.groupby(group_cols)[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(x.median())
                )
        
        # Fill remaining numeric columns with median
        numeric_cols = ['arrival_qty', 'rainfall_mm', 'temp_c', 'acreage_estimate']
        for col in numeric_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled.groupby(group_cols)[col].transform(
                    lambda x: x.fillna(x.median())
                )
        
        # Fill binary flags with 0
        binary_cols = ['govt_policy_flag', 'holiday_flag']
        for col in binary_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(0)
        
        # Check if any missing values remain
        remaining_nulls = df_filled.isnull().sum().sum()
        if remaining_nulls > 0:
            logger.warning(f"Still have {remaining_nulls} missing values after imputation")
            df_filled = df_filled.fillna(0)
        
        return df_filled
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using quantile-based method"""
        df_clean = df.copy()
        
        price_cols = ['min_price', 'max_price', 'modal_price']
        for col in price_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.05)  # Use 5% to be less aggressive
                Q3 = df_clean[col].quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers before capping
                outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if outliers_before > 0:
                    logger.info(f"Capping {outliers_before} outliers in {col}")
                
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
        return df_clean
    
    def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to specified frequency"""
        if self.aggregation_freq == 'daily':
            logger.info("Using daily frequency (no aggregation)")
            return df
        
        group_cols = ['market', 'crop', 'variety', 'state', 'district']
        
        logger.info(f"Aggregating data to {self.aggregation_freq} frequency")
        
        try:
            if self.aggregation_freq == 'weekly':
                # Group by week starting Monday
                df_agg = df.set_index('date').groupby([
                    pd.Grouper(freq='W-MON')
                ] + group_cols).agg({
                    'min_price': 'mean',
                    'max_price': 'mean', 
                    'modal_price': 'mean',
                    'arrival_qty': 'sum',
                    'acreage_estimate': 'last',
                    'rainfall_mm': 'sum',
                    'temp_c': 'mean',
                    'govt_policy_flag': 'max',
                    'holiday_flag': 'max'
                }).reset_index()
                
            elif self.aggregation_freq == 'monthly':
                df_agg = df.set_index('date').groupby([
                    pd.Grouper(freq='M')
                ] + group_cols).agg({
                    'min_price': 'mean',
                    'max_price': 'mean',
                    'modal_price': 'mean',
                    'arrival_qty': 'sum',
                    'acreage_estimate': 'last',
                    'rainfall_mm': 'sum',
                    'temp_c': 'mean',
                    'govt_policy_flag': 'max',
                    'holiday_flag': 'max'
                }).reset_index()
            else:
                logger.warning(f"Unknown aggregation frequency: {self.aggregation_freq}, using daily")
                return df
            
            logger.info(f"✅ Shape after aggregation: {df_agg.shape}")
            return df_agg
            
        except Exception as e:
            logger.error(f"Error during aggregation: {e}")
            return df

def prepare_features_for_training(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features for model training by handling missing values and data types"""
    logger.info("🛠️ Preparing features for training")
    
    if X.empty or y.empty:
        raise ValueError("Cannot prepare features from empty data")
    
    X_processed = X.copy()
    
    # Convert object columns to numeric where possible
    object_columns = X_processed.select_dtypes(include=['object']).columns
    for col in object_columns:
        try:
            # Try direct conversion to numeric
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
            converted_count = X_processed[col].notna().sum()
            logger.info(f"Converted {converted_count}/{len(X_processed)} values in {col} to numeric")
        except Exception as e:
            logger.warning(f"Could not convert column {col} to numeric: {e}")
            # If conversion fails, use frequency encoding
            freq_encoding = X_processed[col].value_counts(normalize=True)
            X_processed[col] = X_processed[col].map(freq_encoding)
            X_processed[col] = X_processed[col].fillna(0)
            logger.info(f"Frequency encoded column {col}")
    
    # Handle missing values in numeric columns
    numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        # FIX: Use explicit boolean check instead of implicit
        null_mask = X_processed[col].isnull()
        if null_mask.any():  # Explicit .any() call
            null_count = null_mask.sum()
            median_val = X_processed[col].median()
            X_processed[col] = X_processed[col].fillna(median_val)
            logger.info(f"Filled {null_count} missing values in {col} with median: {median_val:.4f}")
    
    # Fill any remaining NaNs with 0
    remaining_nulls = X_processed.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"Filling {remaining_nulls} remaining NaN values with 0")
        X_processed = X_processed.fillna(0)
    
    # Remove any infinite values
    inf_count = np.isinf(X_processed.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        logger.warning(f"Replacing {inf_count} infinite values with 0")
        X_processed = X_processed.replace([np.inf, -np.inf], 0)
    
    # Handle target variable
    y_processed = y.copy()
    valid_indices = ~y_processed.isnull()
    
    if valid_indices.sum() == 0:  # Explicit .sum() call
        raise ValueError("No valid target values after processing")
    
    X_final = X_processed[valid_indices]
    y_final = y_processed[valid_indices]
    
    logger.info(f"✅ Final processed features: {X_final.shape}")
    logger.info(f"✅ Valid target samples: {len(y_final)}")
    
    return X_final, y_final

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return summary statistics"""
    quality_report = {
        'total_records': len(df),
        'date_range': {
            'start': df['date'].min(),
            'end': df['date'].max()
        },
        'missing_values': df.isnull().sum().to_dict(),
        'unique_counts': {
            'markets': df['market'].nunique(),
            'crops': df['crop'].nunique(),
            'states': df['state'].nunique()
        },
        'price_statistics': {
            'min_price': df['modal_price'].min(),
            'max_price': df['modal_price'].max(),
            'mean_price': df['modal_price'].mean(),
            'median_price': df['modal_price'].median()
        }
    }
    
    return quality_report