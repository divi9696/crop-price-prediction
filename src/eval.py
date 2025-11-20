import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import shap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Model evaluation and visualization class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_history = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         horizon: int = 1) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'horizon': horizon
        }
        
        # Store in history
        if horizon not in self.metrics_history:
            self.metrics_history[horizon] = []
        self.metrics_history[horizon].append(metrics)
        
        return metrics
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        dates: Optional[pd.DatetimeIndex] = None,
                        title: str = "Actual vs Predicted Prices") -> plt.Figure:
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Time series plot
        if dates is not None:
            axes[0].plot(dates, y_true, label='Actual', alpha=0.7)
            axes[0].plot(dates, y_pred, label='Predicted', alpha=0.7)
            axes[0].set_xlabel('Date')
        else:
            axes[0].plot(y_true, label='Actual', alpha=0.7)
            axes[0].plot(y_pred, label='Predicted', alpha=0.7)
            axes[0].set_xlabel('Time Index')
        
        axes[0].set_ylabel('Price (INR)')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(range(len(residuals)), residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Time Index')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Prediction Residuals')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_by_horizon(self) -> plt.Figure:
        """Plot metrics across different horizons"""
        if not self.metrics_history:
            raise ValueError("No metrics calculated yet")
        
        horizons = sorted(self.metrics_history.keys())
        metrics_df = pd.DataFrame([
            {**metric, 'horizon': horizon} 
            for horizon in horizons 
            for metric in self.metrics_history[horizon]
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics_to_plot = ['rmse', 'mae', 'mape']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]
            sns.boxplot(data=metrics_df, x='horizon', y=metric, ax=ax)
            ax.set_title(f'{metric.upper()} by Horizon')
            ax.set_xlabel('Horizon (weeks)')
            ax.set_ylabel(metric.upper())
        
        # Hide empty subplot
        axes[1, 1].axis('off')
        plt.tight_layout()
        return fig
    
    def generate_shap_analysis(self, model, X: np.ndarray, 
                             feature_names: List[str]) -> plt.Figure:
        """Generate SHAP analysis for tree-based models"""
        try:
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Summary plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Feature importance
            shap.summary_plot(shap_values, X, feature_names=feature_names, 
                            show=False, plot_type="bar", max_display=15)
            axes[0].set_title("SHAP Feature Importance")
            
            # Beeswarm plot
            shap.summary_plot(shap_values, X, feature_names=feature_names, 
                            show=False, max_display=15)
            axes[1].set_title("SHAP Value Distribution")
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            # Return empty figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "SHAP analysis not available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def cross_validate_time_series(self, model, X: np.ndarray, y: np.ndarray,
                                 n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform time series cross-validation"""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {'rmse': [], 'mae': [], 'mape': []}
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and predict
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            for key in cv_scores.keys():
                cv_scores[key].append(metrics[key])
        
        return cv_scores

def plot_feature_importance(feature_importance: Dict[str, float], 
                          top_n: int = 15) -> plt.Figure:
    """Plot feature importance"""
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importance = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(features))
    
    ax.barh(y_pos, importance)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    
    plt.tight_layout()
    return fig