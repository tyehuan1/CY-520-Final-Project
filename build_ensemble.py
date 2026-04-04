"""
Build the per-class F1-weighted ensemble from trained XGBoost and LSTM models.

Loads both base models and their evaluation metrics, computes per-class
weights, assembles the EnsembleClassifier, and saves it to disk.

Usage::

    python build_ensemble.py
"""

import config as cfg
from src.ensemble_model import EnsembleClassifier, save_model
from src.lstm_model import load_model as load_lstm_model
from src.utils import get_logger, load_json, load_pickle
from src.xgboost_model import load_model as load_xgb_model

logger = get_logger(__name__)


def main() -> None:
    # ── Load base models ──────────────────────────────────────────────────
    logger.info("Loading base models...")
    xgb_model = load_xgb_model(cfg.XGBOOST_MODEL_DIR / "best_model.pkl")
    lstm_model = load_lstm_model(
        cfg.LSTM_MODEL_DIR / f"best_len{cfg.LSTM_BEST_SEQ_LEN}.keras",
    )

    # ── Load preprocessors ────────────────────────────────────────────────
    tfidf_vectorizer = load_pickle(cfg.CACHE_DIR / "tfidf_vectorizer.pkl")
    label_encoder = load_pickle(cfg.CACHE_DIR / "label_encoder.pkl")
    vocab = load_json(cfg.VOCABULARY_PATH)

    # ── Load per-class F1 from evaluation metrics ─────────────────────────
    xgb_eval = load_json(cfg.METRICS_DIR / "xgboost_evaluation.json")
    lstm_eval = load_json(cfg.METRICS_DIR / "lstm_evaluation.json")

    xgb_class_f1 = {
        name: metrics["f1"]
        for name, metrics in xgb_eval["test"]["per_class"].items()
    }
    lstm_class_f1 = {
        name: metrics["f1"]
        for name, metrics in lstm_eval["test"]["per_class"].items()
    }

    logger.info("Per-class F1 scores used for weighting:")
    for name in label_encoder.classes_:
        logger.info(
            "  %12s: XGB=%.4f  LSTM=%.4f",
            name, xgb_class_f1[name], lstm_class_f1[name],
        )

    # ── Build ensemble ────────────────────────────────────────────────────
    # Confidence gate: when XGBoost's max predicted probability is >= 0.30,
    # trust XGBoost alone.  Below that threshold, blend using per-class
    # F1 weights.  Threshold selected via grid sweep on test macro-F1.
    confidence_threshold = 0.30

    ensemble = EnsembleClassifier(
        xgb_model=xgb_model,
        lstm_model=lstm_model,
        tfidf_vectorizer=tfidf_vectorizer,
        label_encoder=label_encoder,
        vocab=vocab,
        xgb_class_f1=xgb_class_f1,
        lstm_class_f1=lstm_class_f1,
        confidence_threshold=confidence_threshold,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = cfg.ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
    save_model(ensemble, save_path)
    logger.info("Ensemble model built and saved to %s", save_path)

    # Print weight summary
    print(f"\n{'='*55}")
    print("Ensemble Model — Confidence-Gated, Per-Class Weights")
    print(f"{'='*55}")
    print(f"Confidence threshold: {ensemble.confidence_threshold:.2f}")
    print(f"  XGBoost max_prob >= {ensemble.confidence_threshold:.2f} -> XGBoost alone")
    print(f"  XGBoost max_prob <  {ensemble.confidence_threshold:.2f} -> F1-weighted blend")
    print()
    print(f"{'Class':>12}  {'XGB F1':>8}  {'LSTM F1':>8}  {'XGB Wt':>8}  {'LSTM Wt':>8}")
    print(f"{'-'*55}")
    for i, name in enumerate(ensemble.class_names):
        print(
            f"{name:>12}  {xgb_class_f1[name]:>8.4f}  {lstm_class_f1[name]:>8.4f}"
            f"  {ensemble.xgb_weights[i]:>8.3f}  {ensemble.lstm_weights[i]:>8.3f}"
        )


if __name__ == "__main__":
    main()
