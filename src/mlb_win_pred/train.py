"""Train MLB win prediction models."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb
from mlb_win_pred.config import Config
from mlb_win_pred.utils import (
    get_data_path, get_model_path, set_seed, setup_logger
)
from mlb_win_pred.dataset_builder import build_full_dataset, train_val_test_split

logger = setup_logger()


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature column names from dataset.
    
    Args:
        df: Dataset with features
        
    Returns:
        List of feature column names
    """
    # Get all columns that start with team_ or opp_ or are context features
    feature_cols = []
    
    for col in df.columns:
        if col.startswith('team_') or col.startswith('opp_'):
            feature_cols.append(col)
        elif col in ['is_home', 'month', 'day_of_week']:
            feature_cols.append(col)
    
    return sorted(feature_cols)


def prepare_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for training.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        feature_cols: List of feature column names
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Select features and target
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['win'].values
    
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df['win'].values
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['win'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train baseline logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Tuple of (model, scaler)
    """
    logger.info("Training baseline logistic regression model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, y_pred_proba)
    val_logloss = log_loss(y_val, y_pred_proba)
    
    logger.info(f"Baseline model - Val AUC: {val_auc:.4f}, Val Log Loss: {val_logloss:.4f}")
    
    return model, scaler


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config
) -> xgb.XGBClassifier:
    """Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration object
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model...")
    
    model = xgb.XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        learning_rate=config.xgb_learning_rate,
        max_depth=config.xgb_max_depth,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        random_state=config.random_seed,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Use callbacks for early stopping in newer XGBoost versions (2.0+)
    try:
        from xgboost import callback
        callbacks = [callback.EarlyStopping(rounds=20, save_best=True)]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
            verbose=False
        )
    except (ImportError, AttributeError, TypeError):
        # Fallback: try without early stopping or use eval_set only
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        except Exception:
            # Final fallback: just fit without eval_set
            model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_pred_proba)
    val_logloss = log_loss(y_val, y_pred_proba)
    
    logger.info(f"XGBoost model - Val AUC: {val_auc:.4f}, Val Log Loss: {val_logloss:.4f}")
    
    return model


def train_models(config: Config) -> None:
    """Train models and save the best one.
    
    Args:
        config: Configuration object
    """
    set_seed(config.random_seed)
    
    logger.info("Starting model training...")
    
    # Load or build dataset
    processed_file = get_data_path(config, "games_processed.csv", subdir="processed")
    
    if processed_file.exists():
        logger.info(f"Loading processed dataset from {processed_file}")
        df = pd.read_csv(processed_file)
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        logger.info("Processed dataset not found. Building from raw data...")
        df = build_full_dataset(config, save=True)
    
    if df.empty:
        logger.error("Dataset is empty. Cannot train models.")
        return
    
    # Split into train/val/test
    train_df, val_df, test_df = train_val_test_split(df, config)
    
    if train_df.empty:
        logger.error("Training set is empty. Check date splits in config.")
        return
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        train_df, val_df, test_df, feature_cols
    )
    
    # Train baseline model
    baseline_model, scaler = train_baseline_model(X_train, y_train, X_val, y_val)
    
    # Train XGBoost model
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val, config)
    
    # Compare models and choose best
    baseline_proba = baseline_model.predict_proba(scaler.transform(X_val))[:, 1]
    xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
    
    baseline_auc = roc_auc_score(y_val, baseline_proba)
    xgb_auc = roc_auc_score(y_val, xgb_proba)
    
    logger.info(f"Baseline AUC: {baseline_auc:.4f}")
    logger.info(f"XGBoost AUC: {xgb_auc:.4f}")
    
    # Retrain best model on train+val
    if xgb_auc >= baseline_auc:
        logger.info("XGBoost is best model. Retraining on train+val...")
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        
        final_model = xgb.XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            learning_rate=config.xgb_learning_rate,
            max_depth=config.xgb_max_depth,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            random_state=config.random_seed,
            eval_metric='logloss',
            use_label_encoder=False
        )
        final_model.fit(X_train_val, y_train_val, verbose=False)
        
        model_name = "win_model_xgb.pkl"
        model_type = "xgboost"
    else:
        logger.info("Baseline is best model. Retraining on train+val...")
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        
        X_train_val_scaled = scaler.fit_transform(X_train_val)
        final_model = LogisticRegression(random_state=42, max_iter=1000)
        final_model.fit(X_train_val_scaled, y_train_val)
        
        model_name = "win_model_lr.pkl"
        model_type = "logistic_regression"
    
    # Save model
    model_path = get_model_path(config, model_name)
    model_data = {
        'model': final_model,
        'scaler': scaler if model_type == "logistic_regression" else None,
        'feature_cols': feature_cols,
        'model_type': model_type
    }
    joblib.dump(model_data, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save feature list
    feature_list_path = get_model_path(config, "feature_list.txt")
    with open(feature_list_path, 'w') as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")
    logger.info(f"Saved feature list to {feature_list_path}")


if __name__ == "__main__":
    from mlb_win_pred.config import get_config
    
    config = get_config()
    train_models(config)

