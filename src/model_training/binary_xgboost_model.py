"""
Stage-1 Binary XGBoost: malware vs benign detection.

Trained on Olivera-preprocessed Mal-API (malware) + Olivera benign samples.
Uses TF-IDF + statistical + category features, with seq_length and
unique_ratio removed (near-constant at fixed 100-call sequences).

Run directly::

    python -m src.model_training.binary_xgboost_model
"""

import math
import time
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, f1_score

import config as cfg
from src.model_training.feature_engineering import (
    build_tfidf_vectorizer,
    compute_category_features,
    tfidf_transform,
)
from src.model_training.xgboost_model import (
    predict_with_confidence,
    save_model,
    train_xgboost,
)
from src.utils import get_logger, load_pickle, save_json, save_pickle

logger = get_logger(__name__)

Sample = Dict[str, object]


# ── Binary-specific statistical features ─────────────────────────────────


def compute_binary_statistical_features(
    samples: List[Sample],
    top_k: int = cfg.TOP_K_API_FREQUENCIES,
) -> np.ndarray:
    """Compute statistical features for binary classification.

    Same as the family classifier's statistical features, but with
    ``seq_length`` and ``unique_ratio`` removed since all sequences
    are Olivera-preprocessed to ~100 calls (making length near-constant
    and unique_ratio redundant with unique_count).

    Features per sample:
    1. Unique API call count
    2-6. Top-k API call frequencies (as ratios)
    7. Shannon entropy of the API call frequency distribution

    Args:
        samples: Samples with ``sequence`` field.
        top_k: Number of top API call frequency ratios to include.

    Returns:
        Array of shape ``(n_samples, 1 + top_k + 1)``.
    """
    n_features = 1 + top_k + 1  # unique, top_k ratios, entropy
    features = np.zeros((len(samples), n_features), dtype=np.float64)

    for i, sample in enumerate(samples):
        seq = sample["sequence"]
        total = len(seq)

        if total == 0:
            continue

        counts = Counter(seq)
        unique = len(counts)

        # Top-k frequency ratios (descending)
        sorted_counts = sorted(counts.values(), reverse=True)
        top_k_ratios = [c / total for c in sorted_counts[:top_k]]
        top_k_ratios += [0.0] * (top_k - len(top_k_ratios))

        # Shannon entropy
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        features[i, 0] = unique
        features[i, 1 : 1 + top_k] = top_k_ratios
        features[i, 1 + top_k] = entropy

    logger.info("Binary statistical features computed: shape %s.", features.shape)
    return features


BINARY_STATISTICAL_FEATURE_NAMES = (
    ["unique_count"]
    + [f"top{k+1}_freq_ratio" for k in range(cfg.TOP_K_API_FREQUENCIES)]
    + ["shannon_entropy"]
)


def build_binary_feature_matrix(
    samples: List[Sample],
    tfidf_vectorizer,
    top_k: int = cfg.TOP_K_API_FREQUENCIES,
) -> np.ndarray:
    """Build the feature matrix for binary XGBoost.

    Concatenates:
    1. TF-IDF features
    2. Statistical features (without seq_length and unique_ratio)
    3. API category ratio features

    Args:
        samples: Samples with ``sequence`` field.
        tfidf_vectorizer: Fitted TF-IDF vectorizer.
        top_k: Number of top-k frequency features.

    Returns:
        Dense array of shape ``(n_samples, total_features)``.
    """
    parts = []
    part_names = []

    tfidf = tfidf_transform(samples, tfidf_vectorizer)
    parts.append(tfidf)
    part_names.append(f"tfidf={tfidf.shape[1]}")

    stats = compute_binary_statistical_features(samples, top_k)
    parts.append(stats)
    part_names.append(f"stats={stats.shape[1]}")

    cats = compute_category_features(samples)
    parts.append(cats)
    part_names.append(f"cats={cats.shape[1]}")

    combined = np.hstack(parts)
    logger.info(
        "Binary feature matrix: %s (%s).",
        combined.shape,
        ", ".join(part_names),
    )
    return combined


# ── Training pipeline ────────────────────────────────────────────────────


def main() -> None:
    """Train Stage-1 binary XGBoost (malware vs benign)."""

    # ── Load cached binary dataset ───────────────────────────────────────
    logger.info("Loading binary preprocessed data...")
    train_samples = load_pickle(cfg.BINARY_PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.BINARY_PREPROCESSED_TEST_PATH)
    label_encoder = load_pickle(cfg.BINARY_LABEL_ENCODER_PATH)

    y_train = label_encoder.transform([s["label"] for s in train_samples])
    y_test = label_encoder.transform([s["label"] for s in test_samples])

    logger.info(
        "Binary dataset: %d train, %d test, classes=%s.",
        len(train_samples), len(test_samples), list(label_encoder.classes_),
    )

    # ── Build features ───────────────────────────────────────────────────
    cfg.BINARY_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_train = cfg.BINARY_FEATURES_DIR / "X_train_binary_xgb.pkl"
    cache_test = cfg.BINARY_FEATURES_DIR / "X_test_binary_xgb.pkl"

    if cache_train.exists() and cache_test.exists():
        logger.info("Loading cached binary feature matrices...")
        X_train = load_pickle(cache_train)
        X_test = load_pickle(cache_test)
    else:
        logger.info("Building TF-IDF vectorizer for binary task...")
        tfidf_vec = build_tfidf_vectorizer(train_samples)

        logger.info("Building binary feature matrices...")
        X_train = build_binary_feature_matrix(train_samples, tfidf_vec)
        X_test = build_binary_feature_matrix(test_samples, tfidf_vec)

        # Cache
        save_pickle(tfidf_vec, cfg.BINARY_CACHE_DIR / "tfidf_vectorizer.pkl")
        save_pickle(X_train, cache_train)
        save_pickle(X_test, cache_test)

    logger.info("Feature matrix shape: train=%s, test=%s", X_train.shape, X_test.shape)

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("Starting binary XGBoost training...")
    start = time.time()
    model, best_params = train_xgboost(X_train, y_train, label_encoder)
    elapsed = (time.time() - start) / 60.0
    logger.info("Training time: %.1f minutes.", elapsed)

    # Save model
    cfg.BINARY_XGBOOST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, cfg.BINARY_XGBOOST_MODEL_DIR / "best_model.pkl")
    save_pickle(best_params, cfg.BINARY_XGBOOST_MODEL_DIR / "best_params.pkl")

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

    logger.info("Binary XGBoost — Train acc: %.4f, macro-F1: %.4f",
                train_acc, train_macro_f1)
    logger.info("Binary XGBoost — Test acc: %.4f, macro-F1: %.4f",
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
        "features_removed": ["seq_length", "unique_ratio"],
        "per_class_f1": {
            name: round(report[name]["f1-score"], 4)
            for name in label_encoder.classes_
        },
    }
    cfg.BINARY_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(results, cfg.BINARY_METRICS_DIR / "binary_xgboost_results.json")
    logger.info("Results saved.")

    # Print summary
    print(f"\n{'='*55}")
    print("Stage-1 Binary XGBoost Results")
    print(f"{'='*55}")
    print(f"Train: acc={train_acc:.4f}, macro-F1={train_macro_f1:.4f}")
    print(f"Test:  acc={test_acc:.4f}, macro-F1={test_macro_f1:.4f}")
    print(f"Best params: {best_params}")
    print(f"\n{report_str}")


if __name__ == "__main__":
    main()
