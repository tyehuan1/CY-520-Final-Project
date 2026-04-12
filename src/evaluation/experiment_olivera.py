"""
Olivera generalizability experiment.

Train XGBoost and LSTM on Olivera-formatted Mal-API data (100-call,
deduped, lowercased), then evaluate on:
  1. Mal-API held-out test split (in-distribution baseline)
  2. VT-labeled Olivera malware (cross-dataset generalizability)

Both datasets share the same Cuckoo sandbox, the same ~240-token
vocabulary, and identical 100-call sequence length — isolating
dataset-level distribution shift from vocabulary/sandbox mismatch.

XGBoost feature set:
  - Length-normalised TF (max 1000, unigrams + bigrams)
  - Statistical features (no log_seq_length — near-constant at 100)
  - API category ratios (8 features)
  - Bigram transition proportions (64 features)

LSTM:
  - Sequence length = 100 (matches the data exactly, no padding/truncation
    mismatch), trained with the standard BiLSTM architecture.

Usage::

    python -m src.evaluation.experiment_olivera
"""

import time

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import config as cfg
from src.data_loading.preprocessing import pad_sequences
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_f1,
    plot_roc_curves,
)
from src.model_training.feature_engineering import (
    build_tfidf_vectorizer,
    compute_bigram_transition_features,
    compute_category_features,
    compute_statistical_features,
    tfidf_transform,
)
from src.model_training.lstm_model import (
    build_lstm_model,
    predict_with_confidence as lstm_predict,
    train_lstm,
)
from src.model_training.xgboost_model import (
    predict_with_confidence as xgb_predict,
    save_model as save_xgb_model,
    train_xgboost,
)
from src.utils import get_logger, load_json, load_pickle, save_json, save_pickle

logger = get_logger(__name__)

OLIVERA_LSTM_SEQ_LEN = cfg.OLIVERA_SEQ_COLUMNS  # 100


def build_olivera_features(
    samples: list,
    tfidf_vectorizer,
) -> np.ndarray:
    """Build the Olivera experiment feature matrix (for XGBoost).

    Concatenates TF features (reduced dimensionality) with engineered
    features.  Drops ``log_seq_length`` (index 0 of statistical features)
    because all sequences are ~100 calls.

    Args:
        samples: Samples with ``sequence`` field.
        tfidf_vectorizer: Fitted TF vectorizer.

    Returns:
        Dense feature array.
    """
    tf = tfidf_transform(samples, tfidf_vectorizer)

    stats = compute_statistical_features(samples, log_dampen=False)
    # Drop log_seq_length (column 0) — near-constant for 100-call sequences
    stats = stats[:, 1:]

    cats = compute_category_features(samples, log_dampen=False)
    bigrams = compute_bigram_transition_features(samples, log_dampen=False)

    combined = np.hstack([tf, stats, cats, bigrams])
    logger.info(
        "Olivera features: shape %s (tf=%d, stats=%d, cats=%d, bigrams=%d).",
        combined.shape, tf.shape[1], stats.shape[1],
        cats.shape[1], bigrams.shape[1],
    )
    return combined


def _evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: list,
    tag: str,
    metrics_dir,
    plots_dir,
    olivera_class_indices: list = None,
    olivera_class_names: list = None,
) -> dict:
    """Evaluate a model and save metrics + confusion matrix.

    Args:
        model_name: "xgboost" or "lstm".
        y_true: Ground-truth integer labels.
        preds: Predicted integer labels.
        probs: Prediction probabilities.
        class_names: All class names.
        tag: Dataset tag for filenames (e.g. "malapi_test", "olivera_test").
        metrics_dir: Path to save JSON metrics.
        plots_dir: Path to save plots.
        olivera_class_indices: If set, also log restricted report.
        olivera_class_names: Names for restricted classes.

    Returns:
        Metrics dict.
    """
    metrics = compute_all_metrics(y_true, preds, probs, class_names)
    save_json(metrics, metrics_dir / f"{tag}_{model_name}.json")

    logger.info(
        "%s %s: acc=%.4f  macro-F1=%.4f",
        tag, model_name, metrics["accuracy"], metrics["macro_f1"],
    )

    # Use restricted labels for Olivera (some classes missing)
    if olivera_class_indices is not None:
        logger.info(
            "\n%s",
            classification_report(
                y_true, preds,
                labels=olivera_class_indices,
                target_names=olivera_class_names, zero_division=0,
            ),
        )
    else:
        logger.info(
            "\n%s",
            classification_report(
                y_true, preds,
                target_names=class_names, zero_division=0,
            ),
        )

    plot_confusion_matrix(
        y_true, preds, class_names,
        f"Olivera {model_name.upper()} — {tag}",
        plots_dir / f"{tag}_{model_name}_confusion.png",
    )
    plot_per_class_f1(
        metrics,
        f"Olivera {model_name.upper()} — {tag} Per-Class F1",
        plots_dir / f"{tag}_{model_name}_per_class_f1.png",
    )
    plot_roc_curves(
        y_true, probs, class_names,
        f"Olivera {model_name.upper()} — {tag} ROC",
        plots_dir / f"{tag}_{model_name}_roc.png",
    )
    return metrics


def main() -> None:
    cfg.OLIVERA_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OLIVERA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load preprocessed data ──────────────────────────────────────────
    logger.info("Loading Olivera experiment datasets...")
    train_samples = load_pickle(cfg.OLIVERA_TRAIN_PATH)
    test_samples = load_pickle(cfg.OLIVERA_TEST_PATH)
    olivera_samples = load_pickle(cfg.OLIVERA_EXT_TEST_PATH)
    label_encoder: LabelEncoder = load_pickle(cfg.OLIVERA_LABEL_ENCODER_PATH)
    vocab = load_json(cfg.OLIVERA_VOCABULARY_PATH)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    y_train = label_encoder.transform([s["label"] for s in train_samples])
    y_test = label_encoder.transform([s["label"] for s in test_samples])

    # Olivera may have a subset of classes (small classes excluded)
    olivera_classes = sorted(set(s["label"] for s in olivera_samples))
    olivera_class_indices = [
        i for i, c in enumerate(class_names) if c in olivera_classes
    ]
    olivera_class_names = [class_names[i] for i in olivera_class_indices]
    y_olivera = label_encoder.transform([s["label"] for s in olivera_samples])

    logger.info(
        "Train: %d | MalAPI test: %d | Olivera test: %d | Classes: %s",
        len(train_samples), len(test_samples), len(olivera_samples),
        class_names,
    )
    logger.info("Olivera classes (after min-size filter): %s", olivera_classes)

    # ==================================================================
    # XGBoost
    # ==================================================================
    logger.info("=" * 60)
    logger.info("XGBOOST TRAINING")
    logger.info("=" * 60)

    # Build features
    logger.info("Fitting TF vectorizer on training data...")
    tfidf_vec = build_tfidf_vectorizer(
        train_samples,
        max_features=cfg.OLIVERA_TFIDF_MAX_FEATURES,
        ngram_range=cfg.OLIVERA_TFIDF_NGRAM_RANGE,
    )
    save_pickle(tfidf_vec, cfg.OLIVERA_TFIDF_PATH)

    X_train_xgb = build_olivera_features(train_samples, tfidf_vec)
    X_test_xgb = build_olivera_features(test_samples, tfidf_vec)
    X_oliv_xgb = build_olivera_features(olivera_samples, tfidf_vec)

    # Train
    xgb_start = time.time()
    xgb_model, xgb_params = train_xgboost(
        X_train_xgb, y_train, label_encoder,
        param_dist=cfg.OLIVERA_XGBOOST_PARAM_DIST,
    )
    xgb_elapsed = (time.time() - xgb_start) / 60.0
    logger.info("XGBoost training: %.1f minutes.", xgb_elapsed)

    cfg.OLIVERA_XGBOOST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_xgb_model(xgb_model, cfg.OLIVERA_XGBOOST_MODEL_DIR / "best_model.pkl")

    # Evaluate XGBoost
    xgb_preds_test, xgb_probs_test = xgb_predict(xgb_model, X_test_xgb)
    xgb_malapi = _evaluate_model(
        "xgboost", y_test, xgb_preds_test, xgb_probs_test, class_names,
        "malapi_test", cfg.OLIVERA_METRICS_DIR, cfg.OLIVERA_PLOTS_DIR,
    )

    xgb_preds_oliv, xgb_probs_oliv = xgb_predict(xgb_model, X_oliv_xgb)
    xgb_olivera = _evaluate_model(
        "xgboost", y_olivera, xgb_preds_oliv, xgb_probs_oliv, class_names,
        "olivera_test", cfg.OLIVERA_METRICS_DIR, cfg.OLIVERA_PLOTS_DIR,
        olivera_class_indices, olivera_class_names,
    )

    # ==================================================================
    # LSTM
    # ==================================================================
    logger.info("=" * 60)
    logger.info("LSTM TRAINING (seq_len=%d)", OLIVERA_LSTM_SEQ_LEN)
    logger.info("=" * 60)

    # Pad/truncate encoded sequences to 100
    X_train_lstm = pad_sequences(
        [s["encoded"] for s in train_samples], max_len=OLIVERA_LSTM_SEQ_LEN,
    )
    X_test_lstm = pad_sequences(
        [s["encoded"] for s in test_samples], max_len=OLIVERA_LSTM_SEQ_LEN,
    )
    X_oliv_lstm = pad_sequences(
        [s["encoded"] for s in olivera_samples], max_len=OLIVERA_LSTM_SEQ_LEN,
    )

    y_train_onehot = to_categorical(y_train, num_classes=num_classes)

    # Build and train LSTM
    lstm_model = build_lstm_model(
        vocab_size=len(vocab),
        max_seq_len=OLIVERA_LSTM_SEQ_LEN,
        num_classes=num_classes,
    )

    lstm_model_dir = cfg.MODELS_DIR / "lstm_olivera"
    lstm_model_path = lstm_model_dir / f"best_len{OLIVERA_LSTM_SEQ_LEN}.keras"

    lstm_start = time.time()
    train_lstm(
        lstm_model, X_train_lstm, y_train_onehot, y_train,
        model_path=lstm_model_path,
    )
    lstm_elapsed = (time.time() - lstm_start) / 60.0
    logger.info("LSTM training: %.1f minutes.", lstm_elapsed)

    # Evaluate LSTM
    lstm_preds_test, lstm_probs_test = lstm_predict(lstm_model, X_test_lstm)
    lstm_malapi = _evaluate_model(
        "lstm", y_test, lstm_preds_test, lstm_probs_test, class_names,
        "malapi_test", cfg.OLIVERA_METRICS_DIR, cfg.OLIVERA_PLOTS_DIR,
    )

    lstm_preds_oliv, lstm_probs_oliv = lstm_predict(lstm_model, X_oliv_lstm)
    lstm_olivera = _evaluate_model(
        "lstm", y_olivera, lstm_preds_oliv, lstm_probs_oliv, class_names,
        "olivera_test", cfg.OLIVERA_METRICS_DIR, cfg.OLIVERA_PLOTS_DIR,
        olivera_class_indices, olivera_class_names,
    )

    # ==================================================================
    # Generalization gap
    # ==================================================================
    gap = {}
    for model_name, malapi_m, olivera_m, elapsed in [
        ("xgboost", xgb_malapi, xgb_olivera, xgb_elapsed),
        ("lstm", lstm_malapi, lstm_olivera, lstm_elapsed),
    ]:
        ma_f1 = malapi_m["macro_f1"]
        ol_f1 = olivera_m["macro_f1"]
        drop = round(100 * (ma_f1 - ol_f1) / ma_f1, 1) if ma_f1 > 0 else None

        malapi_per = malapi_m.get("per_class", {})
        olivera_per = olivera_m.get("per_class", {})

        gap[model_name] = {
            "training_minutes": round(elapsed, 1),
            "malapi_macro_f1": ma_f1,
            "olivera_macro_f1": ol_f1,
            "drop_pct": drop,
            "per_class": {
                cls: {
                    "malapi_f1": malapi_per.get(cls, {}).get("f1", 0.0),
                    "olivera_f1": olivera_per.get(cls, {}).get("f1", 0.0),
                    "delta": round(
                        olivera_per.get(cls, {}).get("f1", 0.0)
                        - malapi_per.get(cls, {}).get("f1", 0.0),
                        4,
                    ),
                }
                for cls in class_names
            },
        }
    save_json(gap, cfg.OLIVERA_METRICS_DIR / "generalization_gap.json")

    # Save experiment summary
    comparison = {
        "experiment": "Olivera generalizability (Olivera-format Mal-API training)",
        "xgboost": {
            "training_minutes": round(xgb_elapsed, 1),
            "best_params": xgb_params,
            "feature_shape": list(X_train_xgb.shape),
            "tfidf_max_features": cfg.OLIVERA_TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": list(cfg.OLIVERA_TFIDF_NGRAM_RANGE),
        },
        "lstm": {
            "training_minutes": round(lstm_elapsed, 1),
            "seq_len": OLIVERA_LSTM_SEQ_LEN,
            "vocab_size": len(vocab),
        },
        "malapi_test_n": len(test_samples),
        "olivera_test_n": len(olivera_samples),
        "olivera_classes": olivera_classes,
    }
    save_json(comparison, cfg.OLIVERA_METRICS_DIR / "experiment_summary.json")

    # Model comparison charts (XGBoost vs LSTM side-by-side)
    plot_model_comparison(
        xgb_malapi, lstm_malapi,
        cfg.OLIVERA_PLOTS_DIR / "malapi_test_model_comparison.png",
    )
    plot_model_comparison(
        xgb_olivera, lstm_olivera,
        cfg.OLIVERA_PLOTS_DIR / "olivera_test_model_comparison.png",
    )

    # ==================================================================
    # Print summary
    # ==================================================================
    print(f"\n{'='*65}")
    print("Olivera Generalizability Experiment — Results")
    print(f"{'='*65}")
    print(f"XGBoost: {xgb_elapsed:.1f} min | Features: {X_train_xgb.shape[1]}")
    print(f"LSTM:    {lstm_elapsed:.1f} min | Seq len: {OLIVERA_LSTM_SEQ_LEN}")
    print()
    print(f"  {'Model':<10} {'Dataset':<16} {'Samples':>8} {'Acc':>7} {'F1':>7}")
    print(f"  {'-'*50}")
    for name, ma_m, ol_m in [
        ("xgboost", xgb_malapi, xgb_olivera),
        ("lstm", lstm_malapi, lstm_olivera),
    ]:
        print(
            f"  {name:<10} {'Mal-API test':<16} {len(test_samples):>8} "
            f"{ma_m['accuracy']:>7.4f} {ma_m['macro_f1']:>7.4f}"
        )
        print(
            f"  {'':<10} {'Olivera test':<16} {len(olivera_samples):>8} "
            f"{ol_m['accuracy']:>7.4f} {ol_m['macro_f1']:>7.4f}"
        )
        g = gap[name]
        print(f"  {'':<10} {'Drop':<16} {'':>8} {'':>7} {g['drop_pct']:>6.1f}%")

    print(f"\n  {'Class':>12}  {'XGB MalAPI':>10} {'XGB Oliv':>9}  "
          f"{'LSTM MalAPI':>11} {'LSTM Oliv':>10}")
    print(f"  {'-'*58}")
    for cls in class_names:
        xg = gap["xgboost"]["per_class"][cls]
        lg = gap["lstm"]["per_class"][cls]
        marker = " *" if cls not in olivera_classes else ""
        print(
            f"  {cls:>12}  {xg['malapi_f1']:>10.4f} {xg['olivera_f1']:>9.4f}  "
            f"{lg['malapi_f1']:>11.4f} {lg['olivera_f1']:>10.4f}{marker}"
        )
    if set(class_names) - set(olivera_classes):
        print(f"\n  * = class not present in Olivera test set")

    print(f"\nResults saved to: {cfg.OLIVERA_RESULTS_DIR}")


if __name__ == "__main__":
    main()
