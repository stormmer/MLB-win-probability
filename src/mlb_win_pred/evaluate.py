"""Evaluate trained MLB win prediction models."""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
    confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve
import logging
from mlb_win_pred.config import Config
from mlb_win_pred.utils import (
    get_data_path, get_model_path, get_report_path, get_figure_path,
    setup_logger
)
from mlb_win_pred.dataset_builder import train_val_test_split

logger = setup_logger()

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_model(config: Config, model_path: Path = None) -> Dict:
    """Load trained model.
    
    Args:
        config: Configuration object
        model_path: Path to model file (defaults to win_model_xgb.pkl)
        
    Returns:
        Dictionary with model, scaler, feature_cols, and model_type
    """
    if model_path is None:
        # Try XGBoost first, then logistic regression
        xgb_path = get_model_path(config, "win_model_xgb.pkl")
        lr_path = get_model_path(config, "win_model_lr.pkl")
        
        if xgb_path.exists():
            model_path = xgb_path
        elif lr_path.exists():
            model_path = lr_path
        else:
            raise FileNotFoundError(f"No model found. Expected {xgb_path} or {lr_path}")
    
    logger.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    return model_data


def evaluate_model(config: Config, model_path: Path = None) -> Dict:
    """Evaluate model on test set.
    
    Args:
        config: Configuration object
        model_path: Path to model file
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting model evaluation...")
    
    # Load model
    model_data = load_model(config, model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_cols = model_data['feature_cols']
    model_type = model_data.get('model_type', 'unknown')
    
    # Load test data
    processed_file = get_data_path(config, "games_processed.csv", subdir="processed")
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
    
    df = pd.read_csv(processed_file)
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Get test set
    _, _, test_df = train_val_test_split(df, config)
    
    if test_df.empty:
        logger.warning("Test set is empty. Using validation set instead.")
        train_df, val_df, _ = train_val_test_split(df, config)
        test_df = val_df
    
    # Prepare features
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['win'].values
    
    # Make predictions
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'model_type': model_type,
        'n_test_samples': len(y_test)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Generate plots
    generate_plots(y_test, y_pred_proba, model, feature_cols, config)
    
    # Save metrics
    metrics_path = get_report_path(config, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return metrics


def generate_plots(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    model,
    feature_cols: list,
    config: Config
) -> None:
    """Generate evaluation plots.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model: Trained model
        feature_cols: List of feature names
        config: Configuration object
    """
    logger.info("Generating evaluation plots...")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_figure_path(config, "roc_curve.png"), dpi=150)
    plt.close()
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_figure_path(config, "calibration_curve.png"), dpi=150)
    plt.close()
    
    # Feature Importance (if XGBoost)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        
        plt.figure()
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(get_figure_path(config, "feature_importance.png"), dpi=150)
        plt.close()
    
    # Probability Distribution
    plt.figure()
    plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='Losses', density=True)
    plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Wins', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Predicted Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_figure_path(config, "probability_distribution.png"), dpi=150)
    plt.close()
    
    logger.info("Plots saved to reports/figures/")


if __name__ == "__main__":
    from mlb_win_pred.config import get_config
    
    config = get_config()
    metrics = evaluate_model(config)
    print("\nEvaluation complete!")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test Log Loss: {metrics['log_loss']:.4f}")

