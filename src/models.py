import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import joblib
import json
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictor:
    """Base class for price prediction models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.model_type = None
        self.feature_columns = []
        self.training_date = None
        
    def save_model(self, filepath: str) -> None:
        """Save model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type == 'lightgbm':
            joblib.dump(self.model, f"{filepath}.joblib")
        elif self.model_type == 'lstm':
            self.model.save(f"{filepath}.h5")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'training_date': self.training_date or datetime.now().isoformat(),
            'config': self.config,
            'feature_columns': self.feature_columns,
            'feature_count': len(self.feature_columns)
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model and metadata"""
        # Load metadata first
        try:
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.model_type = metadata['model_type']
            self.training_date = metadata.get('training_date')
            self.feature_columns = metadata.get('feature_columns', [])
            
            if self.model_type == 'lightgbm':
                self.model = joblib.load(f"{filepath}.joblib")
            elif self.model_type == 'lstm':
                self.model = tf.keras.models.load_model(f"{filepath}.h5")
            
            logger.info(f"✅ Model loaded from {filepath}")
            logger.info(f"📊 Model type: {self.model_type}")
            logger.info(f"📅 Training date: {self.training_date}")
            logger.info(f"🔢 Features: {len(self.feature_columns)}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

class LightGBMPredictor(PricePredictor):
    """LightGBM based price predictor"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_type = 'lightgbm'
        self.lgb_config = config['lightgbm']
        
    def build_model(self) -> lgb.LGBMRegressor:
        """Build LightGBM model"""
        model = lgb.LGBMRegressor(
            n_estimators=self.lgb_config['n_estimators'],
            learning_rate=self.lgb_config['learning_rate'],
            max_depth=self.lgb_config['max_depth'],
            subsample=self.lgb_config['subsample'],
            colsample_bytree=self.lgb_config['colsample_bytree'],
            reg_alpha=self.lgb_config.get('reg_alpha', 0.1),
            reg_lambda=self.lgb_config.get('reg_lambda', 0.1),
            random_state=self.config['model']['random_seed'],
            n_jobs=-1,
            verbose=-1
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the model"""
        logger.info("🚀 Training LightGBM model")
        
        self.model = self.build_model()
        
        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append('valid')
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=10
        )
        
        self.training_date = datetime.now().isoformat()
        logger.info("✅ LightGBM training completed")
        
        # Log feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            total_importance = sum(self.model.feature_importances_)
            if total_importance > 0:
                logger.info(f"📈 Model has {len(self.model.feature_importances_)} features with importance")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            predictions = self.model.predict(X)
            logger.info(f"📊 Made predictions for {len(predictions)} samples")
            return predictions
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise

class LSTMPredictor(PricePredictor):
    """LSTM based price predictor"""
    
    def __init__(self, config: Dict, sequence_length: int = 12):
        super().__init__(config)
        self.model_type = 'lstm'
        self.lstm_config = config['lstm']
        self.sequence_length = sequence_length
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model"""
        model = Sequential()
        
        # Convolutional layer for feature extraction
        model.add(Conv1D(
            filters=32, kernel_size=3, activation='relu',
            input_shape=input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        
        # LSTM layers
        for units in self.lstm_config['units']:
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(self.lstm_config['dropout']))
        
        # Final LSTM layer without return sequences
        model.add(LSTM(self.lstm_config['units'][-1] // 2))
        model.add(Dropout(self.lstm_config['dropout']))
        
        # Output layer
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=self.lstm_config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info("✅ LSTM model built successfully")
        return model
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logger.info(f"📦 Created sequences: {X_seq.shape} -> {y_seq.shape}")
        return X_seq, y_seq
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the LSTM model"""
        logger.info("🚀 Training LSTM model")
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_train, y_train)
        
        # Build model
        self.model = self.build_model((X_seq.shape[1], X_seq.shape[2]))
        
        # Training callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
            logger.info(f"📊 Using validation data: {X_val_seq.shape}")
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.lstm_config['epochs'],
            batch_size=self.lstm_config['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_date = datetime.now().isoformat()
        
        # Log training results
        final_loss = history.history['loss'][-1]
        final_mae = history.history['mae'][-1]
        logger.info(f"✅ LSTM training completed - Final loss: {final_loss:.4f}, MAE: {final_mae:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure we have enough data for sequences
        if len(X) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} observations for prediction")
        
        # Use the last sequence_length points
        X_seq = X[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        predictions = self.model.predict(X_seq, verbose=0)
        
        logger.info(f"📊 LSTM prediction made for sequence")
        return predictions.flatten()

class MultiHorizonPredictor:
    """Predictor for multiple horizons"""
    
    def __init__(self, base_predictor: PricePredictor, horizons: List[int]):
        self.base_predictor = base_predictor
        self.horizons = horizons
        self.models = {}
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train models for each horizon"""
        logger.info(f"🎯 Training multi-horizon models for horizons: {self.horizons}")
        
        for horizon in self.horizons:
            logger.info(f"  Training horizon {horizon}...")
            
            # Create shifted target for each horizon
            y_horizon = np.roll(y, -horizon)
            y_horizon = y_horizon[:-horizon]  # Remove last horizon points
            X_horizon = X[:-horizon]
            
            # Skip if not enough data
            if len(X_horizon) < 100:
                logger.warning(f"Not enough data for horizon {horizon}, skipping")
                continue
            
            # Train model for this horizon
            horizon_predictor = type(self.base_predictor)(self.base_predictor.config)
            horizon_predictor.train(X_horizon, y_horizon)
            
            self.models[horizon] = horizon_predictor
            logger.info(f"  ✅ Horizon {horizon} model trained")
    
    def predict(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Predict for specific horizon"""
        if horizon not in self.models:
            raise ValueError(f"No model trained for horizon {horizon}")
        
        return self.models[horizon].predict(X)

def load_predictor(model_path: str) -> PricePredictor:
    """Convenience function to load a predictor"""
    # Determine model type from metadata
    try:
        with open(f"{model_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata['model_type']
        
        if model_type == 'lightgbm':
            predictor = LightGBMPredictor(metadata['config'])
        elif model_type == 'lstm':
            predictor = LSTMPredictor(metadata['config'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        predictor.load_model(model_path)
        return predictor
        
    except Exception as e:
        logger.error(f"Error loading predictor: {e}")
        raise