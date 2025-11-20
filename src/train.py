from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict
import yaml
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataLoader, DataPreprocessor, prepare_features_for_training, validate_data_quality
from src.features import FeatureEngineer, create_feature_summary
from src.models import LightGBMPredictor, LSTMPredictor, MultiHorizonPredictor
from src.eval import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_directory(directory: str):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"📁 Created directory: {directory}")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train crop price prediction models')
    parser.add_argument('--config', type=str, default='config.yml', 
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/synthetic_crop_prices.csv',
                       help='Path to data file')
    parser.add_argument('--model', type=str, choices=['lightgbm', 'lstm', 'both'],
                       default='lightgbm', help='Model to train')
    parser.add_argument('--output', type=str, default='models/',
                       help='Output directory for models')
    parser.add_argument('--crop', type=str, default=None,
                       help='Specific crop to train on (default: from config)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🚀 Starting crop price prediction training")
    logger.info(f"📁 Output directory: {args.output}")
    logger.info(f"📊 Model type: {args.model}")
    
    # Ensure output directory exists
    ensure_directory(args.output)
    
    # Load configuration
    config = load_config(args.config)
    logger.info("✅ Configuration loaded")
    
    # Load and preprocess data
    data_loader = DataLoader(config)
    
    try:
        df = data_loader.load_from_csv(args.data)
    except FileNotFoundError:
        logger.error(f"❌ Data file not found: {args.data}")
        logger.info("💡 Please generate synthetic data first: python data/generate_synthetic.py")
        return
    
    # Validate data quality
    quality_report = validate_data_quality(df)
    logger.info(f"📋 Data quality report: {quality_report['total_records']} records")
    
    # Filter for specific crop
    target_crop = args.crop or config['model'].get('target_crop', 'rice')
    df_filtered = df[df['crop'] == target_crop].copy()
    logger.info(f"🌾 Filtered data for {target_crop}: {len(df_filtered):,} records")
    
    if len(df_filtered) == 0:
        logger.error(f"❌ No data found for crop: {target_crop}")
        logger.info(f"💡 Available crops: {df['crop'].unique().tolist()}")
        return
    
    # Preprocess data
    preprocessor = DataPreprocessor(config)
    df_processed = preprocessor.transform(df_filtered)
    
    # Feature engineering
    feature_engineer = FeatureEngineer(config)
    feature_engineer.fit(df_processed)
    X, y = feature_engineer.transform(df_processed)
    
    # Create feature summary
    feature_summary = create_feature_summary(X, y)
    logger.info(f"🔢 Features: {feature_summary['n_features']}")
    logger.info(f"📈 Target stats - Mean: {feature_summary['target_stats']['mean']:.2f}, "
                f"Std: {feature_summary['target_stats']['std']:.2f}")
    
    # Handle missing values and prepare features
    X_processed, y_processed = prepare_features_for_training(X, y)
    
    logger.info(f"✅ Final dataset: X={X_processed.shape}, y={y_processed.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    # Save scaler for later use
    scaler_path = os.path.join(args.output, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"💾 Scaler saved to {scaler_path}")
    
    # Save feature names
    feature_engineer.feature_columns = X_processed.columns.tolist()
    
    # Split data (time-series aware)
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_processed[:split_idx], y_processed[split_idx:]
    
    logger.info(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Train models
    models = {}
    
    if args.model in ['lightgbm', 'both']:
        logger.info("🎯 Training LightGBM model...")
        try:
            lgb_predictor = LightGBMPredictor(config)
            lgb_predictor.feature_columns = feature_engineer.feature_columns
            lgb_predictor.train(X_train, y_train.values, X_test, y_test.values)
            
            # Evaluate
            y_pred_lgb = lgb_predictor.predict(X_test)
            metrics_lgb = evaluator.calculate_metrics(y_test.values, y_pred_lgb)
            
            logger.info("📈 LightGBM Test Metrics:")
            for metric, value in metrics_lgb.items():
                if metric != 'horizon':
                    logger.info(f"  {metric.upper()}: {value:.4f}")
            
            models['lightgbm'] = lgb_predictor
            
            # Save model
            lgb_predictor.save_model(f"{args.output}/lightgbm_model")
            
        except Exception as e:
            logger.error(f"❌ LightGBM training failed: {e}")
    
    if args.model in ['lstm', 'both']:
        logger.info("🎯 Training LSTM model...")
        try:
            lstm_predictor = LSTMPredictor(config, sequence_length=12)
            lstm_predictor.feature_columns = feature_engineer.feature_columns
            lstm_predictor.train(X_train, y_train.values, X_test, y_test.values)
            
            # Evaluate
            y_pred_lstm = lstm_predictor.predict(X_test)
            metrics_lstm = evaluator.calculate_metrics(y_test.values, y_pred_lstm)
            
            logger.info("📈 LSTM Test Metrics:")
            for metric, value in metrics_lstm.items():
                if metric != 'horizon':
                    logger.info(f"  {metric.upper()}: {value:.4f}")
            
            models['lstm'] = lstm_predictor
            
            # Save model
            lstm_predictor.save_model(f"{args.output}/lstm_model")
            
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
    
    # Multi-horizon prediction
    if 'lightgbm' in models:
        logger.info("🎯 Training multi-horizon models...")
        try:
            horizons = config['model']['horizons']
            base_predictor = LightGBMPredictor(config)
            multi_predictor = MultiHorizonPredictor(base_predictor, horizons)
            multi_predictor.train(X_train, y_train.values)
            
            # Evaluate multi-horizon
            logger.info("📊 Multi-horizon performance:")
            for horizon in horizons:
                if horizon in multi_predictor.models:
                    y_pred_multi = multi_predictor.predict(X_test, horizon)
                    # Adjust test set for horizon
                    y_test_horizon = y_test.values[horizon:]
                    y_pred_horizon = y_pred_multi[:-horizon] if len(y_pred_multi) > horizon else y_pred_multi
                    
                    if len(y_test_horizon) == len(y_pred_horizon) and len(y_test_horizon) > 0:
                        metrics_multi = evaluator.calculate_metrics(y_test_horizon, y_pred_horizon, horizon)
                        logger.info(f"  Horizon {horizon} weeks - "
                                  f"RMSE: {metrics_multi['rmse']:.2f}, "
                                  f"MAPE: {metrics_multi['mape']:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Multi-horizon training failed: {e}")
    
    # Save training summary
    training_summary = {
        'training_date': datetime.now().isoformat(),
        'config': config,
        'data_info': {
            'original_records': len(df),
            'filtered_records': len(df_filtered),
            'processed_records': len(X_processed),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        },
        'feature_info': feature_summary,
        'models_trained': list(models.keys())
    }
    
    summary_path = os.path.join(args.output, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    logger.info(f"💾 Training summary saved to {summary_path}")
    logger.info("✅ Training completed successfully!")

if __name__ == "__main__":
    main()