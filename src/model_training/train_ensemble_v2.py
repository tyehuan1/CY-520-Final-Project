"""
Build the V2 confidence-gated ensemble from V2 base models.

Uses per-class F1 weights derived from the V2 XGBoost and LSTM test
results, with a confidence gate of 0.50 (higher than the 0.30 default
to force more uncertain samples through the LSTM-aware blend under
distribution shift).

Usage::

    python -m src.model_training.train_ensemble_v2
"""

import config as cfg
from src.model_training.ensemble_model import EnsembleClassifier, save_model
from src.model_training.lstm_model import load_model as load_lstm_model
from src.model_training.xgboost_model import load_model as load_xgb_model
from src.utils import get_logger, load_json, load_pickle

logger = get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.50


def main() -> None:
    # ── Load V2 base models ──────────────────────────────────────────────
    logger.info("Loading V2 base models (8-class, with Trojan)...")
    xgb_model = load_xgb_model(cfg.V2_XGBOOST_MODEL_DIR / "best_model.pkl")
    lstm_model = load_lstm_model(
        cfg.V2_LSTM_MODEL_DIR / f"best_len{cfg.LSTM_V2_BEST_SEQ_LEN}.keras",
    )

    # ── Load V2 preprocessors ────────────────────────────────────────────
    tfidf_vectorizer = load_pickle(cfg.V2_TFIDF_PATH)
    label_encoder = load_pickle(cfg.V2_LABEL_ENCODER_PATH)
    vocab = load_json(cfg.VOCABULARY_PATH)  # with-Trojan vocab

    # ── Per-class F1 from V2 evaluation results ──────────────────────────
    xgb_results = load_json(cfg.V2_METRICS_DIR / "xgboost_v2_results.json")
    lstm_results = load_json(
        cfg.V2_METRICS_DIR / f"lstm_v2_len{cfg.LSTM_V2_BEST_SEQ_LEN}_results.json",
    )

    xgb_class_f1 = xgb_results["per_class_f1"]
    lstm_class_f1 = lstm_results["per_class_f1"]

    logger.info("Per-class F1 scores used for weighting:")
    for name in label_encoder.classes_:
        logger.info(
            "  %12s: XGB=%.4f  LSTM=%.4f",
            name, xgb_class_f1[name], lstm_class_f1[name],
        )

    # ── Build ensemble ───────────────────────────────────────────────────
    ensemble = EnsembleClassifier(
        xgb_model=xgb_model,
        lstm_model=lstm_model,
        tfidf_vectorizer=tfidf_vectorizer,
        label_encoder=label_encoder,
        vocab=vocab,
        xgb_class_f1=xgb_class_f1,
        lstm_class_f1=lstm_class_f1,
        seq_len=cfg.LSTM_V2_BEST_SEQ_LEN,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = cfg.V2_ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
    save_model(ensemble, save_path)

    print(f"\n{'='*55}")
    print("V2 Ensemble — Confidence-Gated, Per-Class Weights")
    print(f"{'='*55}")
    print(f"Confidence threshold: {ensemble.confidence_threshold:.2f}")
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
