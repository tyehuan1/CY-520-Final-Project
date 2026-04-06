"""
Phase 6 — Full evaluation for Stage-1 binary and Stage-2 family models.

Evaluates on the Mal-API-2019 held-out test set:
  - Stage-1: Binary XGBoost (malware vs benign, Olivera-preprocessed)
  - Stage-2: XGBoost, LSTM, and Ensemble (7-class family, no Trojan)

Each stage is evaluated independently on its own test split.  An end-to-end
pipeline evaluation is not performed here because the binary and family test
sets use different preprocessing (Olivera vs full-length); see
``evaluate_generalizability.py`` for the end-to-end pipeline on MalBehavD.

Usage::

    python -m src.evaluation.evaluate_models
"""

import gc

import numpy as np
from sklearn.metrics import classification_report

import config as cfg
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
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

    # Load binary model
    bin_model = load_xgb_model(cfg.BINARY_XGBOOST_MODEL_DIR / "best_model.pkl")

    # Feature names for SHAP
    bin_feature_names = (
        bin_tfidf.get_feature_names_out().tolist()
        + list(BINARY_STATISTICAL_FEATURE_NAMES)
        + CATEGORY_FEATURE_NAMES
    )

    # Build features and predict — train first, then free memory before test
    logger.info("Building binary train features...")
    X_bin_train = build_binary_feature_matrix(bin_train, bin_tfidf)
    bin_train_preds, bin_train_probs = xgb_predict(bin_model, X_bin_train)
    bin_train_metrics = compute_all_metrics(
        y_bin_train, bin_train_preds, bin_train_probs, bin_class_names,
    )
    del X_bin_train, bin_train_preds, bin_train_probs, bin_train
    gc.collect()

    logger.info("Building binary test features...")
    X_bin_test = build_binary_feature_matrix(bin_test, bin_tfidf)
    bin_test_preds, bin_test_probs = xgb_predict(bin_model, X_bin_test)
    bin_test_metrics = compute_all_metrics(
        y_bin_test, bin_test_preds, bin_test_probs, bin_class_names,
    )

    logger.info("Binary train: acc=%.4f, macro-F1=%.4f",
                bin_train_metrics["accuracy"], bin_train_metrics["macro_f1"])
    logger.info("Binary test:  acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
                bin_test_metrics["accuracy"], bin_test_metrics["macro_f1"],
                bin_test_metrics["roc_auc_macro"] or 0)

    cfg.BINARY_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {"train": bin_train_metrics, "test": bin_test_metrics},
        cfg.BINARY_METRICS_DIR / "binary_xgboost_evaluation.json",
    )

    bin_report = classification_report(
        y_bin_test, bin_test_preds, target_names=bin_class_names,
    )
    logger.info("Binary classification report:\n%s", bin_report)

    # Plots
    plot_confusion_matrix(
        y_bin_test, bin_test_preds, bin_class_names,
        "Stage-1 Binary XGBoost — Confusion Matrix",
        cfg.BINARY_PLOTS_DIR / "binary_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_bin_test, bin_test_preds, bin_class_names,
        "Stage-1 Binary XGBoost — Confusion Matrix (Counts)",
        cfg.BINARY_PLOTS_DIR / "binary_confusion_matrix_counts.png",
        normalize=False,
    )
    plot_per_class_f1(
        bin_test_metrics,
        "Stage-1 Binary XGBoost — Per-Class F1",
        cfg.BINARY_PLOTS_DIR / "binary_per_class_f1.png",
    )
    plot_roc_curves(
        y_bin_test, bin_test_probs, bin_class_names,
        "Stage-1 Binary XGBoost — ROC Curves",
        cfg.BINARY_PLOTS_DIR / "binary_roc_curves.png",
    )

    # SHAP
    logger.info("Running SHAP analysis for binary XGBoost...")
    bin_shap_dir = cfg.BINARY_RESULTS_DIR / "shap"
    run_shap_analysis(
        bin_model, X_bin_test, bin_feature_names, bin_class_names, bin_shap_dir,
    )

    # Free binary artifacts before Stage-2
    del bin_model, X_bin_test, bin_test, bin_tfidf
    gc.collect()

    # ==================================================================
    # Stage-2: 7-Class Family Classification (No Trojan)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STAGE-2: 7-CLASS FAMILY CLASSIFICATION (No Trojan)")
    logger.info("=" * 70)

    logger.info("Loading no-Trojan family data...")
    train_samples = load_pickle(cfg.NO_TROJAN_TRAIN_PATH)
    test_samples = load_pickle(cfg.NO_TROJAN_TEST_PATH)
    vocab = load_json(cfg.NO_TROJAN_VOCABULARY_PATH)
    label_encoder = load_pickle(cfg.NO_TROJAN_LABEL_ENCODER_PATH)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    y_train_int = label_encoder.transform([s["label"] for s in train_samples])
    y_test_int = label_encoder.transform([s["label"] for s in test_samples])

    logger.info(
        "Family data: %d train, %d test, %d classes (%s).",
        len(train_samples), len(test_samples), num_classes, class_names,
    )

    # Class distribution
    cfg.NO_TROJAN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(
        y_train_int, y_test_int, class_names,
        cfg.NO_TROJAN_PLOTS_DIR / "class_distribution.png",
    )

    # ── XGBoost ─────────────────────────────────────────────────────────
    logger.info("Evaluating Stage-2 XGBoost...")
    xgb_model = load_xgb_model(cfg.NO_TROJAN_XGBOOST_MODEL_DIR / "best_model.pkl")
    tfidf_vec = load_pickle(cfg.NO_TROJAN_CACHE_DIR / "tfidf_vectorizer.pkl")

    feature_names = (
        tfidf_vec.get_feature_names_out().tolist()
        + list(STATISTICAL_FEATURE_NAMES)
        + CATEGORY_FEATURE_NAMES
    )

    # XGBoost train metrics — load, predict, free
    logger.info("Loading family XGBoost train features...")
    X_train_xgb = load_pickle(cfg.NO_TROJAN_FEATURES_DIR / "X_train_xgb.pkl")
    assert len(feature_names) == X_train_xgb.shape[1]
    xgb_train_preds, xgb_train_probs = xgb_predict(xgb_model, X_train_xgb)
    xgb_train_metrics = compute_all_metrics(
        y_train_int, xgb_train_preds, xgb_train_probs, class_names,
    )
    del X_train_xgb, xgb_train_preds
    gc.collect()

    # XGBoost test metrics
    logger.info("Loading family XGBoost test features...")
    X_test_xgb = load_pickle(cfg.NO_TROJAN_FEATURES_DIR / "X_test_xgb.pkl")
    xgb_test_preds, xgb_test_probs = xgb_predict(xgb_model, X_test_xgb)
    xgb_test_metrics = compute_all_metrics(
        y_test_int, xgb_test_preds, xgb_test_probs, class_names,
    )

    logger.info("XGBoost train: acc=%.4f, macro-F1=%.4f",
                xgb_train_metrics["accuracy"], xgb_train_metrics["macro_f1"])
    logger.info("XGBoost test:  acc=%.4f, macro-F1=%.4f",
                xgb_test_metrics["accuracy"], xgb_test_metrics["macro_f1"])

    cfg.NO_TROJAN_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {"train": xgb_train_metrics, "test": xgb_test_metrics},
        cfg.NO_TROJAN_METRICS_DIR / "xgboost_evaluation.json",
    )

    logger.info("XGBoost report:\n%s",
                classification_report(y_test_int, xgb_test_preds, target_names=class_names))

    plot_confusion_matrix(
        y_test_int, xgb_test_preds, class_names,
        "XGBoost Confusion Matrix (7-class)",
        cfg.NO_TROJAN_PLOTS_DIR / "xgboost_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_test_int, xgb_test_preds, class_names,
        "XGBoost Confusion Matrix (Counts)",
        cfg.NO_TROJAN_PLOTS_DIR / "xgboost_confusion_matrix_counts.png",
        normalize=False,
    )
    plot_per_class_f1(
        xgb_test_metrics, "XGBoost Per-Class F1 (7-class)",
        cfg.NO_TROJAN_PLOTS_DIR / "xgboost_per_class_f1.png",
    )
    plot_roc_curves(
        y_test_int, xgb_test_probs, class_names,
        "XGBoost ROC Curves (7-class)",
        cfg.NO_TROJAN_PLOTS_DIR / "xgboost_roc_curves.png",
    )

    logger.info("Running SHAP analysis for family XGBoost...")
    run_shap_analysis(
        xgb_model, X_test_xgb, feature_names, class_names, cfg.NO_TROJAN_SHAP_DIR,
    )

    # ── LSTM ────────────────────────────────────────────────────────────
    logger.info("Evaluating Stage-2 LSTM (seq_len=%d)...", cfg.LSTM_BEST_SEQ_LEN)
    lstm_model = load_lstm_model(
        cfg.NO_TROJAN_LSTM_MODEL_DIR / f"best_len{cfg.LSTM_BEST_SEQ_LEN}.keras",
    )

    # LSTM train — predict, compute metrics, free
    X_train_lstm = pad_sequences(
        [s["encoded"] for s in train_samples], max_len=cfg.LSTM_BEST_SEQ_LEN,
    )
    lstm_train_preds, lstm_train_probs = lstm_predict(lstm_model, X_train_lstm)
    lstm_train_metrics = compute_all_metrics(
        y_train_int, lstm_train_preds, lstm_train_probs, class_names,
    )
    del X_train_lstm, lstm_train_preds
    gc.collect()

    # LSTM test
    X_test_lstm = pad_sequences(
        [s["encoded"] for s in test_samples], max_len=cfg.LSTM_BEST_SEQ_LEN,
    )
    lstm_test_preds, lstm_test_probs = lstm_predict(lstm_model, X_test_lstm)
    lstm_test_metrics = compute_all_metrics(
        y_test_int, lstm_test_preds, lstm_test_probs, class_names,
    )

    logger.info("LSTM train: acc=%.4f, macro-F1=%.4f",
                lstm_train_metrics["accuracy"], lstm_train_metrics["macro_f1"])
    logger.info("LSTM test:  acc=%.4f, macro-F1=%.4f",
                lstm_test_metrics["accuracy"], lstm_test_metrics["macro_f1"])

    save_json(
        {"train": lstm_train_metrics, "test": lstm_test_metrics},
        cfg.NO_TROJAN_METRICS_DIR / "lstm_evaluation.json",
    )

    logger.info("LSTM report:\n%s",
                classification_report(y_test_int, lstm_test_preds, target_names=class_names))

    plot_confusion_matrix(
        y_test_int, lstm_test_preds, class_names,
        "LSTM Confusion Matrix (7-class)",
        cfg.NO_TROJAN_PLOTS_DIR / "lstm_confusion_matrix.png",
    )
    plot_per_class_f1(
        lstm_test_metrics, "LSTM Per-Class F1 (7-class)",
        cfg.NO_TROJAN_PLOTS_DIR / "lstm_per_class_f1.png",
    )
    plot_roc_curves(
        y_test_int, lstm_test_probs, class_names,
        "LSTM ROC Curves (7-class)",
        cfg.NO_TROJAN_PLOTS_DIR / "lstm_roc_curves.png",
    )

    # ── Ensemble ────────────────────────────────────────────────────────
    logger.info("Evaluating Stage-2 Ensemble...")
    from src.model_training.ensemble_model import (
        EnsembleClassifier,
        load_model as load_ensemble_model,
    )
    # Pickle needs EnsembleClassifier visible on __main__ when loading
    import __main__
    __main__.EnsembleClassifier = EnsembleClassifier

    ensemble_path = cfg.NO_TROJAN_ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
    if not ensemble_path.exists():
        logger.warning("Ensemble not found at %s. Skipping.", ensemble_path)
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
        logger.info("Ensemble test:  acc=%.4f, macro-F1=%.4f",
                     ens_test_metrics["accuracy"], ens_test_metrics["macro_f1"])

        save_json(
            {"train": ens_train_metrics, "test": ens_test_metrics},
            cfg.NO_TROJAN_METRICS_DIR / "ensemble_evaluation.json",
        )

        logger.info("Ensemble report:\n%s",
                     classification_report(y_test_int, ens_test_preds, target_names=class_names))

        plot_confusion_matrix(
            y_test_int, ens_test_preds, class_names,
            "Ensemble Confusion Matrix (7-class)",
            cfg.NO_TROJAN_PLOTS_DIR / "ensemble_confusion_matrix.png",
        )
        plot_per_class_f1(
            ens_test_metrics, "Ensemble Per-Class F1 (7-class)",
            cfg.NO_TROJAN_PLOTS_DIR / "ensemble_per_class_f1.png",
        )

    # ── Model comparison ────────────────────────────────────────────────
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
    save_json(comparison, cfg.NO_TROJAN_METRICS_DIR / "model_comparison.json")

    if ens_test_metrics:
        plot_three_model_comparison(
            xgb_test_metrics, lstm_test_metrics, ens_test_metrics,
            cfg.NO_TROJAN_PLOTS_DIR / "model_comparison_f1.png",
        )

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Phase 6 — Evaluation Summary")
    print(f"{'='*65}")

    print(f"\n--- Stage-1: Binary Detection ---")
    for name in bin_class_names:
        pc = bin_test_metrics["per_class"][name]
        print(f"  {name:<10} P={pc['precision']:.4f}  R={pc['recall']:.4f}  "
              f"F1={pc['f1']:.4f}  (n={pc['support']})")
    print(f"  Overall:   acc={bin_test_metrics['accuracy']:.4f}, "
          f"macro-F1={bin_test_metrics['macro_f1']:.4f}, "
          f"AUC={bin_test_metrics['roc_auc_macro'] or 0:.4f}")

    print(f"\n--- Stage-2: 7-Class Family Classification ---")
    print(f"  {'Model':<12} {'Accuracy':>10} {'Macro-F1':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*42}")
    for name, m in [("XGBoost", xgb_test_metrics), ("LSTM", lstm_test_metrics)]:
        print(f"  {name:<12} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
              f"{m['roc_auc_macro'] or 0:>10.4f}")
    if ens_test_metrics:
        print(f"  {'Ensemble':<12} {ens_test_metrics['accuracy']:>10.4f} "
              f"{ens_test_metrics['macro_f1']:>10.4f} "
              f"{ens_test_metrics['roc_auc_macro'] or 0:>10.4f}")

    print(f"\nResults: {cfg.NO_TROJAN_RESULTS_DIR}")
    print(f"Binary:  {cfg.BINARY_RESULTS_DIR}")


if __name__ == "__main__":
    main()
