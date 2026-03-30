"""
XGBoost model: training with randomized hyperparameter search, saving, loading,
and inference with confidence scores.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label_encoder: Any,
    param_dist: Optional[Dict] = None,
    n_iter: int = cfg.XGBOOST_N_ITER,
    cv_folds: int = cfg.XGBOOST_CV_FOLDS,
    random_seed: int = cfg.RANDOM_SEED,
) -> Tuple[XGBClassifier, Dict]:
    """Train an XGBoost classifier with randomized hyperparameter search.

    Args:
        X_train: Feature matrix of shape ``(n_samples, n_features)``.
        y_train: Integer-encoded labels of shape ``(n_samples,)``.
        label_encoder: Fitted label encoder (for class names in logging).
        param_dist: Hyperparameter search space.  Defaults to config values.
        n_iter: Number of random search iterations.
        cv_folds: Number of stratified CV folds.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (best_model, best_params).
    """
    if param_dist is None:
        param_dist = cfg.XGBOOST_PARAM_DIST

    sample_weights = compute_sample_weight("balanced", y_train)

    base_model = XGBClassifier(
        tree_method=cfg.XGBOOST_TREE_METHOD,
        random_state=random_seed,
        eval_metric="mlogloss",
        n_jobs=-1,
    )

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_seed
    )

    logger.info(
        "Starting RandomizedSearchCV: %d iterations, %d-fold CV.",
        n_iter,
        cv_folds,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_seed,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train, sample_weight=sample_weights)

    best_model = search.best_estimator_
    best_params = search.best_params_

    logger.info("Best CV macro-F1: %.4f", search.best_score_)
    logger.info("Best params: %s", best_params)

    return best_model, best_params


def predict_with_confidence(
    model: XGBClassifier,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and per-class probability scores.

    Args:
        model: Trained XGBoost model.
        X: Feature matrix of shape ``(n_samples, n_features)``.

    Returns:
        Tuple of (predicted_labels, probabilities) where labels has shape
        ``(n_samples,)`` and probabilities has shape ``(n_samples, n_classes)``.
    """
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities


def save_model(model: XGBClassifier, path: Path) -> None:
    """Save an XGBoost model to disk via pickle.

    Args:
        model: Trained model to save.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("XGBoost model saved to %s.", path)


def load_model(path: Path) -> XGBClassifier:
    """Load an XGBoost model from disk.

    Args:
        path: Path to the pickled model file.

    Returns:
        Loaded :class:`XGBClassifier`.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("XGBoost model loaded from %s.", path)
    return model
