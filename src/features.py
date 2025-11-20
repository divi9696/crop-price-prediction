import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for time series data"""
    
    def __init__(self, config: Dict, target_col: str = 'modal_price'):
        self.config = config
        self.target_col = target_col
        self.feature_config = config['features']
        
        # Encoding objects
        self.target_encoder_market = None
        self.target_encoder_state = None
        self.onehot_encoder = None
        self.scaler = None
        
        # Track feature names
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders on training data"""
        logger.info("🔧 Fitting feature engineering transformers")
        
        # Fit target encoders
        if self.feature_config['encoding']['market_encoding'] == 'target':
            self.target_encoder_market = ce.TargetEncoder()
            self.target_encoder_market.fit(X[['market']], X[self.target_col])
            logger.info("Fitted target encoder for market")
        
        if self.feature_config['encoding']['state_encoding'] == 'target':
            self.target_encoder_state = ce.TargetEncoder()
            self.target_encoder_state.fit(X[['state']], X[self.target_col])
            logger.info("Fitted target encoder for state")
        
        # Fit one-hot encoder for variety
        if self.feature_config['encoding']['variety_encoding'] == 'onehot':
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.onehot_encoder.fit(X[['variety']])
            logger.info("Fitted one-hot encoder for variety")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply feature engineering and return features and target"""
        logger.info("🔄 Starting feature engineering")
        
        if len(df) == 0:
            raise ValueError("Cannot engineer features from empty dataframe")
        
        df_featured = df.copy()
        
        # Create time features
        df_featured = self._create_time_features(df_featured)
        
        # Create lag features
        df_featured = self._create_lag_features(df_featured)
        
        # Create rolling features
        df_featured = self._create_rolling_features(df_featured)
        
        # Encode categorical variables
        df_featured = self._encode_categoricals(df_featured)
        
        # Identify feature columns
        feature_columns = self._get_feature_columns(df_featured)
        self.feature_names_ = feature_columns
        
        # Separate features and target
        X = df_featured[feature_columns]
        y = df_featured[self.target_col]
        
        logger.info(f"✅ Final feature matrix: {X.shape}")
        logger.info(f"📊 Feature dtypes: {dict(X.dtypes.value_counts())}")
        logger.info(f"🎯 Target variable: {y.name}")
        
        return X, y
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of appropriate columns for feature matrix"""
        
        # Always exclude these columns
        exclude_cols = ['date', 'district', 'min_price', 'max_price', 'modal_price',
                       'state', 'market', 'crop', 'variety']  # Raw categoricals
        
        # Define columns that should be features
        feature_candidates = []
        
        # Time-based features
        time_features = ['year', 'month', 'quarter', 'week_of_year', 'day_of_week', 
                        'is_harvest_month', 'month_sin', 'month_cos']
        
        # Lag and rolling features (pattern matching)
        lag_rolling_features = [col for col in df.columns if any([
            col.startswith('lag_'),
            col.startswith('rolling_'),
            col.startswith('arrival_'),
            'encoded' in col,
            'variety_' in col
        ])]
        
        # Basic numeric features
        basic_numeric = ['arrival_qty', 'acreage_estimate', 'rainfall_mm', 'temp_c',
                        'govt_policy_flag', 'holiday_flag']
        
        # Combine all potential features
        all_candidates = time_features + lag_rolling_features + basic_numeric
        
        # Filter to existing columns and exclude unwanted ones
        feature_columns = [col for col in all_candidates 
                          if col in df.columns and col not in exclude_cols]
        
        # Ensure they are numeric
        final_features = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                final_features.append(col)
            else:
                logger.warning(f"Skipping non-numeric column: {col} (dtype: {df[col].dtype})")
        
        logger.info(f"Selected {len(final_features)} feature columns")
        return final_features
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df_featured = df.copy()
        
        # Basic time features
        df_featured['year'] = df_featured['date'].dt.year
        df_featured['month'] = df_featured['date'].dt.month
        df_featured['quarter'] = df_featured['date'].dt.quarter
        df_featured['week_of_year'] = df_featured['date'].dt.isocalendar().week.astype(int)
        df_featured['day_of_week'] = df_featured['date'].dt.dayofweek
        
        # Seasonal features
        df_featured['is_harvest_month'] = self._get_harvest_month_flag(
            df_featured['month'], df_featured['crop']
        )
        
        # Cyclical encoding for month
        df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
        df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)
        
        logger.info("Created time-based features")
        return df_featured
    
    def _get_harvest_month_flag(self, month_series: pd.Series, crop_series: pd.Series) -> pd.Series:
        """Create harvest month flag based on crop type"""
        harvest_months = {
            'rice': [9, 10, 11],    # Kharif harvest
            'wheat': [3, 4, 5],     # Rabi harvest
            'maize': [9, 10, 11],   # Kharif harvest
            'pulses': [2, 3, 10, 11], # Varies by type
            'cotton': [10, 11, 12], # Kharif harvest
            'sugarcane': [12, 1, 2] # Year-round with peak
        }
        
        def _check_harvest(month, crop):
            crop_harvest = harvest_months.get(crop, [])
            return 1 if month in crop_harvest else 0
        
        return pd.Series([
            _check_harvest(m, c) for m, c in zip(month_series, crop_series)
        ])
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series"""
        df_featured = df.sort_values(['market', 'crop', 'date']).copy()
        
        group_cols = ['market', 'crop']
        lag_periods = self.feature_config['lags']
        
        logger.info(f"Creating lag features for periods: {lag_periods}")
        
        for lag in lag_periods:
            df_featured[f'lag_{lag}'] = df_featured.groupby(group_cols)[self.target_col].shift(lag)
            df_featured[f'arrival_lag_{lag}'] = df_featured.groupby(group_cols)['arrival_qty'].shift(lag)
        
        return df_featured
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        df_featured = df.sort_values(['market', 'crop', 'date']).copy()
        
        group_cols = ['market', 'crop']
        windows = self.feature_config['rolling_windows']
        
        logger.info(f"Creating rolling features with windows: {windows}")
        
        for window in windows:
            # Rolling statistics for target
            df_featured[f'rolling_mean_{window}'] = df_featured.groupby(group_cols)[self.target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df_featured[f'rolling_std_{window}'] = df_featured.groupby(group_cols)[self.target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Rolling statistics for arrivals
            df_featured[f'arrival_rolling_mean_{window}'] = df_featured.groupby(group_cols)['arrival_qty'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        return df_featured
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Target encoding for market
        if self.target_encoder_market is not None:
            try:
                market_encoded = self.target_encoder_market.transform(df_encoded[['market']])
                df_encoded['market_encoded'] = market_encoded
                logger.info("Applied target encoding for market")
            except Exception as e:
                logger.warning(f"Failed to target encode market: {e}")
        
        # Target encoding for state
        if self.target_encoder_state is not None:
            try:
                state_encoded = self.target_encoder_state.transform(df_encoded[['state']])
                df_encoded['state_encoded'] = state_encoded
                logger.info("Applied target encoding for state")
            except Exception as e:
                logger.warning(f"Failed to target encode state: {e}")
        
        # One-hot encoding for variety
        if self.onehot_encoder is not None:
            try:
                variety_encoded = self.onehot_encoder.transform(df_encoded[['variety']])
                variety_cols = [f'variety_{cat}' for cat in self.onehot_encoder.categories_[0]]
                variety_df = pd.DataFrame(variety_encoded, columns=variety_cols, index=df_encoded.index)
                df_encoded = pd.concat([df_encoded, variety_df], axis=1)
                logger.info(f"Applied one-hot encoding for variety ({len(variety_cols)} categories)")
            except Exception as e:
                logger.warning(f"Failed to one-hot encode variety: {e}")
        
        return df_encoded
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names_

def create_feature_summary(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Create summary of features and target"""
    summary = {
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'feature_names': X.columns.tolist(),
        'feature_dtypes': X.dtypes.astype(str).to_dict(),
        'target_stats': {
            'mean': y.mean(),
            'std': y.std(),
            'min': y.min(),
            'max': y.max()
        },
        'feature_stats': {}
    }
    
    # Add stats for each feature
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            summary['feature_stats'][col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'null_count': X[col].isnull().sum()
            }
    
    return summary