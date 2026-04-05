"""
Phase 6 — Run full evaluation for both XGBoost and LSTM models.

Loads cached data, generates predictions, computes metrics, and produces
all plots and SHAP explanations.  Outputs saved under ``results/``.

Usage::

    python -m src.evaluation.evaluate_models
"""

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

import config as cfg
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_f1,
    plot_roc_curves,
    plot_three_model_comparison,
    run_shap_analysis,
)
from src.model_training.feature_engineering import (
    CATEGORY_FEATURE_NAMES,
    STATISTICAL_FEATURE_NAMES,
    build_feature_matrix,
    get_feature_names,
)
from src.model_training.lstm_model import load_model as load_lstm_model
from src.model_training.lstm_model import predict_with_confidence as lstm_predict
from src.data_loading.preprocessing import pad_sequences
from src.utils import get_logger, load_json, load_pickle, save_json
from src.model_training.xgboost_model import load_model as load_xgb_model
from src.model_training.xgboost_model import predict_with_confidence as xgb_predict

logger = get_logger(__name__)


def main() -> None:
    # ── Load cached preprocessed data ─────────────────────────────────────
    logger.info("Loading cached data...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)
    vocab = load_json(cfg.VOCABULARY_PATH)
    label_encoder = load_pickle(cfg.CACHE_DIR / "label_encoder.pkl")

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    y_train_int = label_encoder.transform([s["label"] for s in train_samples])
    y_test_int = label_encoder.transform([s["label"] for s in test_samples])

    logger.info(
        "Data loaded: %d train, %d test, %d classes.",
        len(train_samples), len(test_samples), num_classes,
    )

    # ── Class distribution plot ───────────────────────────────────────────
    logger.info("Plotting class distribution...")
    plot_class_distribution(
        y_train_int, y_test_int, class_names,
        cfg.PLOTS_DIR / "class_distribution.png",
    )

    # ==================================================================
    # XGBoost Evaluation
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EVALUATING XGBOOST (v1)")
    logger.info("=" * 70)

    # Load model and build features
    xgb_model = load_xgb_model(cfg.XGBOOST_MODEL_DIR / "best_model.pkl")
    tfidf_vectorizer = load_pickle(cfg.CACHE_DIR / "tfidf_vectorizer.pkl")

    # v1 uses the original vectorizer with no skip-grams
    X_train_xgb = load_pickle(cfg.FEATURES_DIR / "X_train_xgb.pkl")
    X_test_xgb = load_pickle(cfg.FEATURES_DIR / "X_test_xgb.pkl")

    # Feature names: TF-IDF(5000) + Stats(9) + Categories(8) = 5017
    feature_names = (
        tfidf_vectorizer.get_feature_names_out().tolist()
        + list(STATISTICAL_FEATURE_NAMES)
        + CATEGORY_FEATURE_NAMES
    )
    assert len(feature_names) == X_test_xgb.shape[1], (
        f"Feature name count ({len(feature_names)}) != matrix columns ({X_test_xgb.shape[1]})"
    )

    logger.info(
        "XGBoost feature matrix: train=%s, test=%s",
        X_train_xgb.shape, X_test_xgb.shape,
    )

    # Predictions
    xgb_train_preds, xgb_train_probs = xgb_predict(xgb_model, X_train_xgb)
    xgb_test_preds, xgb_test_probs = xgb_predict(xgb_model, X_test_xgb)

    # Metrics
    xgb_train_metrics = compute_all_metrics(
        y_train_int, xgb_train_preds, xgb_train_probs, class_names,
    )
    xgb_test_metrics = compute_all_metrics(
        y_test_int, xgb_test_preds, xgb_test_probs, class_names,
    )

    logger.info("XGBoost train: acc=%.4f, macro-F1=%.4f",
                xgb_train_metrics["accuracy"], xgb_train_metrics["macro_f1"])
    logger.info("XGBoost test:  acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
                xgb_test_metrics["accuracy"], xgb_test_metrics["macro_f1"],
                xgb_test_metrics["roc_auc_macro"] or 0)

    # Save metrics
    save_json(
        {"train": xgb_train_metrics, "test": xgb_test_metrics},
        cfg.METRICS_DIR / "xgboost_evaluation.json",
    )

    # Classification report (text)
    xgb_report = classification_report(
        y_test_int, xgb_test_preds, target_names=class_names,
    )
    logger.info("XGBoost classification report:\n%s", xgb_report)

    # Plots
    plot_confusion_matrix(
        y_test_int, xgb_test_preds, class_names,
        "XGBoost Confusion Matrix (Normalized)",
        cfg.PLOTS_DIR / "xgboost_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_test_int, xgb_test_preds, class_names,
        "XGBoost Confusion Matrix (Counts)",
        cfg.PLOTS_DIR / "xgboost_confusion_matrix_counts.png",
        normalize=False,
    )
    plot_per_class_f1(
        xgb_test_metrics,
        "XGBoost Per-Class F1 Score",
        cfg.PLOTS_DIR / "xgboost_per_class_f1.png",
    )
    plot_roc_curves(
        y_test_int, xgb_test_probs, class_names,
        "XGBoost ROC Curves (One-vs-Rest)",
        cfg.PLOTS_DIR / "xgboost_roc_curves.png",
    )

    # SHAP analysis
    logger.info("Running SHAP analysis for XGBoost...")
    run_shap_analysis(
        xgb_model, X_test_xgb, feature_names, class_names, cfg.SHAP_DIR,
    )

    # ==================================================================
    # LSTM Evaluation
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EVALUATING LSTM (v1, seq_len=200)")
    logger.info("=" * 70)

    lstm_model = load_lstm_model(
        cfg.LSTM_MODEL_DIR / f"best_len{cfg.LSTM_BEST_SEQ_LEN}.keras",
    )

    train_encoded = [s["encoded"] for s in train_samples]
    test_encoded = [s["encoded"] for s in test_samples]

    X_train_lstm = pad_sequences(train_encoded, max_len=cfg.LSTM_BEST_SEQ_LEN)
    X_test_lstm = pad_sequences(test_encoded, max_len=cfg.LSTM_BEST_SEQ_LEN)

    logger.info(
        "LSTM padded sequences: train=%s, test=%s",
        X_train_lstm.shape, X_test_lstm.shape,
    )

    # Predictions
    lstm_train_preds, lstm_train_probs = lstm_predict(lstm_model, X_train_lstm)
    lstm_test_preds, lstm_test_probs = lstm_predict(lstm_model, X_test_lstm)

    # Metrics
    lstm_train_metrics = compute_all_metrics(
        y_train_int, lstm_train_preds, lstm_train_probs, class_names,
    )
    lstm_test_metrics = compute_all_metrics(
        y_test_int, lstm_test_preds, lstm_test_probs, class_names,
    )

    logger.info("LSTM train: acc=%.4f, macro-F1=%.4f",
                lstm_train_metrics["accuracy"], lstm_train_metrics["macro_f1"])
    logger.info("LSTM test:  acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
                lstm_test_metrics["accuracy"], lstm_test_metrics["macro_f1"],
                lstm_test_metrics["roc_auc_macro"] or 0)

    # Save metrics
    save_json(
        {"train": lstm_train_metrics, "test": lstm_test_metrics},
        cfg.METRICS_DIR / "lstm_evaluation.json",
    )

    # Classification report (text)
    lstm_report = classification_report(
        y_test_int, lstm_test_preds, target_names=class_names,
    )
    logger.info("LSTM classification report:\n%s", lstm_report)

    # Plots
    plot_confusion_matrix(
        y_test_int, lstm_test_preds, class_names,
        "LSTM Confusion Matrix (Normalized)",
        cfg.PLOTS_DIR / "lstm_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_test_int, lstm_test_preds, class_names,
        "LSTM Confusion Matrix (Counts)",
        cfg.PLOTS_DIR / "lstm_confusion_matrix_counts.png",
        normalize=False,
    )
    plot_per_class_f1(
        lstm_test_metrics,
        "LSTM Per-Class F1 Score",
        cfg.PLOTS_DIR / "lstm_per_class_f1.png",
    )
    plot_roc_curves(
        y_test_int, lstm_test_probs, class_names,
        "LSTM ROC Curves (One-vs-Rest)",
        cfg.PLOTS_DIR / "lstm_roc_curves.png",
    )

    # ==================================================================
    # Ensemble Evaluation
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EVALUATING ENSEMBLE (F1-weighted XGBoost + LSTM)")
    logger.info("=" * 70)

    from src.model_training.ensemble_model import load_model as load_ensemble_model

    ensemble_path = cfg.ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
    if not ensemble_path.exists():
        logger.warning(
            "Ensemble model not found at %s. Run python -m src.model_training.ensemble_model first. "
            "Skipping ensemble evaluation.",
            ensemble_path,
        )
        ens_test_metrics = None
    else:
        ensemble = load_ensemble_model(ensemble_path)

        # Use pre-computed probabilities from both models (faster)
        ens_train_preds, ens_train_probs = ensemble.predict_from_precomputed(
            xgb_train_probs, lstm_train_probs,
        )
        ens_test_preds, ens_test_probs = ensemble.predict_from_precomputed(
            xgb_test_probs, lstm_test_probs,
        )

        ens_train_metrics = compute_all_metrics(
            y_train_int, ens_train_preds, ens_train_probs, class_names,
        )
        ens_test_metrics = compute_all_metrics(
            y_test_int, ens_test_preds, ens_test_probs, class_names,
        )

        logger.info("Ensemble train: acc=%.4f, macro-F1=%.4f",
                     ens_train_metrics["accuracy"], ens_train_metrics["macro_f1"])
        logger.info("Ensemble test:  acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
                     ens_test_metrics["accuracy"], ens_test_metrics["macro_f1"],
                     ens_test_metrics["roc_auc_macro"] or 0)

        save_json(
            {"train": ens_train_metrics, "test": ens_test_metrics},
            cfg.METRICS_DIR / "ensemble_evaluation.json",
        )

        ens_report = classification_report(
            y_test_int, ens_test_preds, target_names=class_names,
        )
        logger.info("Ensemble classification report:\n%s", ens_report)

        # Plots
        plot_confusion_matrix(
            y_test_int, ens_test_preds, class_names,
            "Ensemble Confusion Matrix (Normalized)",
            cfg.PLOTS_DIR / "ensemble_confusion_matrix.png",
        )
        plot_confusion_matrix(
            y_test_int, ens_test_preds, class_names,
            "Ensemble Confusion Matrix (Counts)",
            cfg.PLOTS_DIR / "ensemble_confusion_matrix_counts.png",
            normalize=False,
        )
        plot_per_class_f1(
            ens_test_metrics,
            "Ensemble Per-Class F1 Score",
            cfg.PLOTS_DIR / "ensemble_per_class_f1.png",
        )
        plot_roc_curves(
            y_test_int, ens_test_probs, class_names,
            "Ensemble ROC Curves (One-vs-Rest)",
            cfg.PLOTS_DIR / "ensemble_roc_curves.png",
        )

    # ==================================================================
    # Model Comparison
    # ==================================================================
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    comparison = {
        "xgboost": {
            "accuracy": xgb_test_metrics["accuracy"],
            "macro_f1": xgb_test_metrics["macro_f1"],
            "weighted_f1": xgb_test_metrics["weighted_f1"],
            "roc_auc_macro": xgb_test_metrics["roc_auc_macro"],
        },
        "lstm": {
            "accuracy": lstm_test_metrics["accuracy"],
            "macro_f1": lstm_test_metrics["macro_f1"],
            "weighted_f1": lstm_test_metrics["weighted_f1"],
            "roc_auc_macro": lstm_test_metrics["roc_auc_macro"],
        },
    }
    if ens_test_metrics:
        comparison["ensemble"] = {
            "accuracy": ens_test_metrics["accuracy"],
            "macro_f1": ens_test_metrics["macro_f1"],
            "weighted_f1": ens_test_metrics["weighted_f1"],
            "roc_auc_macro": ens_test_metrics["roc_auc_macro"],
        }
    save_json(comparison, cfg.METRICS_DIR / "model_comparison.json")

    # Three-model comparison plot (if ensemble exists)
    if ens_test_metrics:
        plot_three_model_comparison(
            xgb_test_metrics, lstm_test_metrics, ens_test_metrics,
            cfg.PLOTS_DIR / "model_comparison_f1.png",
        )
    else:
        plot_model_comparison(
            xgb_test_metrics, lstm_test_metrics,
            cfg.PLOTS_DIR / "model_comparison_f1.png",
        )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 6 — Evaluation Summary")
    print(f"{'='*60}")
    print(f"\n{'Model':<12} {'Accuracy':>10} {'Macro-F1':>10} {'ROC-AUC':>10}")
    print(f"{'-'*42}")
    print(f"{'XGBoost':<12} {xgb_test_metrics['accuracy']:>10.4f} "
          f"{xgb_test_metrics['macro_f1']:>10.4f} "
          f"{xgb_test_metrics['roc_auc_macro'] or 0:>10.4f}")
    print(f"{'LSTM':<12} {lstm_test_metrics['accuracy']:>10.4f} "
          f"{lstm_test_metrics['macro_f1']:>10.4f} "
          f"{lstm_test_metrics['roc_auc_macro'] or 0:>10.4f}")
    if ens_test_metrics:
        print(f"{'Ensemble':<12} {ens_test_metrics['accuracy']:>10.4f} "
              f"{ens_test_metrics['macro_f1']:>10.4f} "
              f"{ens_test_metrics['roc_auc_macro'] or 0:>10.4f}")
    print(f"\nPlots saved to:   {cfg.PLOTS_DIR}")
    print(f"Metrics saved to: {cfg.METRICS_DIR}")
    print(f"SHAP saved to:    {cfg.SHAP_DIR}")


if __name__ == "__main__":
    main()
