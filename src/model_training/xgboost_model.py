"""
XGBoost model: training with randomized hyperparameter search, saving, loading,
inference with confidence scores, and end-to-end training pipeline.

Run directly to train::

    python -m src.xgboost_model
"""

import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config as cfg
from src.utils import get_logger, load_json, load_pickle, save_json, save_pickle

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


# ── End-to-end training pipeline ─────────────────────────────────────────


def main() -> None:
    """Train XGBoost on Mal-API-2019 with TF-IDF + statistical + category features."""
    from src.model_training.feature_engineering import build_feature_matrix, build_tfidf_vectorizer

    # ── Load cached preprocessed data ────────────────────────────────────
    logger.info("Loading cached preprocessed data...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)
    label_encoder = load_pickle(cfg.CACHE_DIR / "label_encoder.pkl")

    y_train = label_encoder.transform([s["label"] for s in train_samples])
    y_test = label_encoder.transform([s["label"] for s in test_samples])

    # ── Build or load features ──────────────────────────────────────────
    cache_train = cfg.FEATURES_DIR / "X_train_xgb_v2.pkl"
    cache_test = cfg.FEATURES_DIR / "X_test_xgb_v2.pkl"

    if cache_train.exists() and cache_test.exists():
        logger.info("Loading cached feature matrices...")
        X_train = load_pickle(cache_train)
        X_test = load_pickle(cache_test)
    else:
        logger.info("Building TF-IDF vectorizer...")
        tfidf_vec = build_tfidf_vectorizer(train_samples)

        logger.info("Building feature matrices...")
        X_train = build_feature_matrix(train_samples, tfidf_vec)
        X_test = build_feature_matrix(test_samples, tfidf_vec)

        # Cache features and vectorizer
        save_pickle(tfidf_vec, cfg.CACHE_DIR / "tfidf_vectorizer_v2.pkl")
        save_pickle(X_train, cache_train)
        save_pickle(X_test, cache_test)

    logger.info("Feature matrix shape: train=%s, test=%s", X_train.shape, X_test.shape)

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("Starting XGBoost training...")
    start = time.time()
    model, best_params = train_xgboost(X_train, y_train, label_encoder)
    elapsed = (time.time() - start) / 60.0
    logger.info("Training time: %.1f minutes.", elapsed)

    # Save model
    save_model(model, cfg.XGBOOST_MODEL_DIR / "best_model_v2.pkl")
    save_pickle(best_params, cfg.XGBOOST_MODEL_DIR / "best_params_v2.pkl")

    # ── Evaluate ─────────────────────────────────────────────────────────
    preds, probs = predict_with_confidence(model, X_test)
    test_acc = np.mean(preds == y_test)
    test_macro_f1 = f1_score(y_test, preds, average="macro")

    train_preds, _ = predict_with_confidence(model, X_train)
    train_acc = np.mean(train_preds == y_train)
    train_macro_f1 = f1_score(y_train, train_preds, average="macro")

    report = classification_report(
        y_test, preds,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    report_str = classification_report(
        y_test, preds,
        target_names=label_encoder.classes_,
    )

    logger.info("XGBoost — Train accuracy: %.4f, Train macro-F1: %.4f",
                train_acc, train_macro_f1)
    logger.info("XGBoost — Test accuracy: %.4f, Test macro-F1: %.4f",
                test_acc, test_macro_f1)
    logger.info("\n%s", report_str)

    # Save results
    results = {
        "best_params": best_params,
        "train_accuracy": round(train_acc, 4),
        "train_macro_f1": round(train_macro_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "training_minutes": round(elapsed, 1),
        "feature_matrix_shape": list(X_train.shape),
        "per_class_f1": {
            name: round(report[name]["f1-score"], 4)
            for name in label_encoder.classes_
        },
    }
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(results, cfg.METRICS_DIR / "xgboost_v2_results.json")
    logger.info("Results saved.")


if __name__ == "__main__":
    main()
