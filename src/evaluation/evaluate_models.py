"""
Phase 6 — Run full evaluation for Stage-1 binary and Stage-2 family models.

Evaluates:
  - Stage-1: Binary XGBoost (malware vs benign)
  - Stage-2: XGBoost v2, LSTM, and Ensemble (8-class family classification)

Loads cached data, generates predictions, computes metrics, and produces
all plots and SHAP explanations.  Outputs saved under ``results/``.

Usage::

    python -m src.evaluation.evaluate_models
"""

import numpy as np
from sklearn.metrics import classification_report

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
from src.model_training.binary_xgboost_model import (
    BINARY_STATISTICAL_FEATURE_NAMES,
    build_binary_feature_matrix,
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
    # ==================================================================
    # Stage-1: Binary XGBoost (Malware vs Benign)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STAGE-1: BINARY XGBOOST (Malware vs Benign)")
    logger.info("=" * 70)

    # Load binary data and model
    logger.info("Loading binary preprocessed data...")
    bin_train = load_pickle(cfg.BINARY_PREPROCESSED_TRAIN_PATH)
    bin_test = load_pickle(cfg.BINARY_PREPROCESSED_TEST_PATH)
    bin_label_enc = load_pickle(cfg.BINARY_LABEL_ENCODER_PATH)
    bin_tfidf = load_pickle(cfg.BINARY_CACHE_DIR / "tfidf_vectorizer.pkl")

    bin_class_names = list(bin_label_enc.classes_)
    y_bin_train = bin_label_enc.transform([s["label"] for s in bin_train])
    y_bin_test = bin_label_enc.transform([s["label"] for s in bin_test])

    logger.info(
        "Binary dataset: %d train, %d test, classes=%s.",
        len(bin_train), len(bin_test), bin_class_names,
    )

    # Class distribution plot
    cfg.BINARY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(
        y_bin_train, y_bin_test, bin_class_names,
        cfg.BINARY_PLOTS_DIR / "binary_class_distribution.png",
    )

    # Load binary model and features
    bin_model = load_xgb_model(cfg.BINARY_XGBOOST_MODEL_DIR / "best_model.pkl")

    cache_train = cfg.BINARY_FEATURES_DIR / "X_train_binary_xgb.pkl"
    cache_test = cfg.BINARY_FEATURES_DIR / "X_test_binary_xgb.pkl"

    if cache_train.exists() and cache_test.exists():
        logger.info("Loading cached binary feature matrices...")
        X_bin_train = load_pickle(cache_train)
        X_bin_test = load_pickle(cache_test)
    else:
        logger.info("Building binary feature matrices...")
        X_bin_train = build_binary_feature_matrix(bin_train, bin_tfidf)
        X_bin_test = build_binary_feature_matrix(bin_test, bin_tfidf)

    logger.info(
        "Binary feature matrix: train=%s, test=%s",
        X_bin_train.shape, X_bin_test.shape,
    )

    # Binary feature names
    bin_feature_names = (
        bin_tfidf.get_feature_names_out().tolist()
        + list(BINARY_STATISTICAL_FEATURE_NAMES)
        + CATEGORY_FEATURE_NAMES
    )

    # Predictions
    bin_train_preds, bin_train_probs = xgb_predict(bin_model, X_bin_train)
    bin_test_preds, bin_test_probs = xgb_predict(bin_model, X_bin_test)

    # Metrics
    bin_train_metrics = compute_all_metrics(
        y_bin_train, bin_train_preds, bin_train_probs, bin_class_names,
    )
    bin_test_metrics = compute_all_metrics(
        y_bin_test, bin_test_preds, bin_test_probs, bin_class_names,
    )

    logger.info("Binary XGBoost train: acc=%.4f, macro-F1=%.4f",
                bin_train_metrics["accuracy"], bin_train_metrics["macro_f1"])
    logger.info("Binary XGBoost test:  acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
                bin_test_metrics["accuracy"], bin_test_metrics["macro_f1"],
                bin_test_metrics["roc_auc_macro"] or 0)

    # Save metrics
    cfg.BINARY_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {"train": bin_train_metrics, "test": bin_test_metrics},
        cfg.BINARY_METRICS_DIR / "binary_xgboost_evaluation.json",
    )

    # Classification report
    bin_report = classification_report(
        y_bin_test, bin_test_preds, target_names=bin_class_names,
    )
    logger.info("Binary XGBoost classification report:\n%s", bin_report)

    # Plots
    plot_confusion_matrix(
        y_bin_test, bin_test_preds, bin_class_names,
        "Stage-1 Binary XGBoost — Confusion Matrix (Normalized)",
        cfg.BINARY_PLOTS_DIR / "binary_xgboost_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_bin_test, bin_test_preds, bin_class_names,
        "Stage-1 Binary XGBoost — Confusion Matrix (Counts)",
        cfg.BINARY_PLOTS_DIR / "binary_xgboost_confusion_matrix_counts.png",
        normalize=False,
    )
    plot_per_class_f1(
        bin_test_metrics,
        "Stage-1 Binary XGBoost — Per-Class F1",
        cfg.BINARY_PLOTS_DIR / "binary_xgboost_per_class_f1.png",
    )
    plot_roc_curves(
        y_bin_test, bin_test_probs, bin_class_names,
        "Stage-1 Binary XGBoost — ROC Curves",
        cfg.BINARY_PLOTS_DIR / "binary_xgboost_roc_curves.png",
    )

    # SHAP for binary model
    logger.info("Running SHAP analysis for binary XGBoost...")
    bin_shap_dir = cfg.BINARY_RESULTS_DIR / "shap"
    run_shap_analysis(
        bin_model, X_bin_test, bin_feature_names, bin_class_names, bin_shap_dir,
    )

    # ==================================================================
    # Stage-2: Family Classification Models
    # ==================================================================

    # ── Load cached preprocessed data ─────────────────────────────────────
    logger.info("Loading cached family classification data...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)
    vocab = load_json(cfg.VOCABULARY_PATH)
    label_encoder = load_pickle(cfg.CACHE_DIR / "label_encoder.pkl")

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    y_train_int = label_encoder.transform([s["label"] for s in train_samples])
    y_test_int = label_encoder.transform([s["label"] for s in test_samples])

    logger.info(
        "Family data loaded: %d train, %d test, %d classes.",
        len(train_samples), len(test_samples), num_classes,
    )

    # ── Class distribution plot ───────────────────────────────────────────
    logger.info("Plotting family class distribution...")
    plot_class_distribution(
        y_train_int, y_test_int, class_names,
        cfg.PLOTS_DIR / "class_distribution.png",
    )

    # ==================================================================
    # XGBoost v2 Evaluation
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EVALUATING XGBOOST v2 (Stage-2 Family)")
    logger.info("=" * 70)

    # Load v2 model and features
    xgb_model = load_xgb_model(cfg.XGBOOST_MODEL_DIR / "best_model_v2.pkl")
    tfidf_vectorizer = load_pickle(cfg.CACHE_DIR / "tfidf_vectorizer.pkl")

    X_train_xgb = load_pickle(cfg.FEATURES_DIR / "X_train_xgb_v2.pkl")
    X_test_xgb = load_pickle(cfg.FEATURES_DIR / "X_test_xgb_v2.pkl")

    # Feature names: TF-IDF + Stats + Categories
    feature_names = (
        tfidf_vectorizer.get_feature_names_out().tolist()
        + list(STATISTICAL_FEATURE_NAMES)
        + CATEGORY_FEATURE_NAMES
    )
    assert len(feature_names) == X_test_xgb.shape[1], (
        f"Feature name count ({len(feature_names)}) != matrix columns ({X_test_xgb.shape[1]})"
    )

    logger.info(
        "XGBoost v2 feature matrix: train=%s, test=%s",
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

    logger.info("XGBoost v2 train: acc=%.4f, macro-F1=%.4f",
                xgb_train_metrics["accuracy"], xgb_train_metrics["macro_f1"])
    logger.info("XGBoost v2 test:  acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
                xgb_test_metrics["accuracy"], xgb_test_metrics["macro_f1"],
                xgb_test_metrics["roc_auc_macro"] or 0)

    # Save metrics
    save_json(
        {"train": xgb_train_metrics, "test": xgb_test_metrics},
        cfg.METRICS_DIR / "xgboost_evaluation.json",
    )

    # Classification report
    xgb_report = classification_report(
        y_test_int, xgb_test_preds, target_names=class_names,
    )
    logger.info("XGBoost v2 classification report:\n%s", xgb_report)

    # Plots
    plot_confusion_matrix(
        y_test_int, xgb_test_preds, class_names,
        "XGBoost v2 Confusion Matrix (Normalized)",
        cfg.PLOTS_DIR / "xgboost_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_test_int, xgb_test_preds, class_names,
        "XGBoost v2 Confusion Matrix (Counts)",
        cfg.PLOTS_DIR / "xgboost_confusion_matrix_counts.png",
        normalize=False,
    )
    plot_per_class_f1(
        xgb_test_metrics,
        "XGBoost v2 Per-Class F1 Score",
        cfg.PLOTS_DIR / "xgboost_per_class_f1.png",
    )
    plot_roc_curves(
        y_test_int, xgb_test_probs, class_names,
        "XGBoost v2 ROC Curves (One-vs-Rest)",
        cfg.PLOTS_DIR / "xgboost_roc_curves.png",
    )

    # SHAP analysis
    logger.info("Running SHAP analysis for XGBoost v2...")
    run_shap_analysis(
        xgb_model, X_test_xgb, feature_names, class_names, cfg.SHAP_DIR,
    )

    # ==================================================================
    # LSTM Evaluation
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EVALUATING LSTM (Stage-2 Family, seq_len=%d)", cfg.LSTM_BEST_SEQ_LEN)
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

    # Classification report
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
    logger.info("EVALUATING ENSEMBLE (Stage-2 Family)")
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
    print(f"\n{'='*65}")
    print("Phase 6 — Evaluation Summary")
    print(f"{'='*65}")

    print(f"\n--- Stage-1: Binary Detection (Malware vs Benign) ---")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*52}")
    for name in bin_class_names:
        pc = bin_test_metrics["per_class"][name]
        print(f"{name:<12} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
              f"{pc['f1']:>10.4f} {pc['support']:>10d}")
    print(f"\nBinary XGBoost: acc={bin_test_metrics['accuracy']:.4f}, "
          f"macro-F1={bin_test_metrics['macro_f1']:.4f}, "
          f"ROC-AUC={bin_test_metrics['roc_auc_macro'] or 0:.4f}")

    print(f"\n--- Stage-2: Family Classification (8-class) ---")
    print(f"\n{'Model':<12} {'Accuracy':>10} {'Macro-F1':>10} {'ROC-AUC':>10}")
    print(f"{'-'*42}")
    print(f"{'XGBoost v2':<12} {xgb_test_metrics['accuracy']:>10.4f} "
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
    print(f"Binary plots:     {cfg.BINARY_PLOTS_DIR}")
    print(f"Metrics saved to: {cfg.METRICS_DIR}")
    print(f"Binary metrics:   {cfg.BINARY_METRICS_DIR}")
    print(f"SHAP saved to:    {cfg.SHAP_DIR}")


if __name__ == "__main__":
    main()
