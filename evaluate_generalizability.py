"""
Phase 7 — Generalizability evaluation on MalBehavD-V1.

Tests whether models trained on Mal-API-2019 generalize to an independent
dataset (MalBehavD-V1) whose malware samples have been family-labeled via
VirusTotal.

Two evaluation modes:
    1. **Family classification** (8-class): Uses only the VT-labeled malware
       samples.  Same task as the primary evaluation.
    2. **Binary detection** (benign vs malware): Uses the full MalBehavD dataset
       (1285 benign + VT-labeled malware).  Any family prediction counts as
       "malware"; this tests whether the models can distinguish malicious
       behaviour at all on unseen data.

Additionally reports domain-shift diagnostics:
    - Vocabulary overlap / UNK token ratio
    - Sequence length distributions (Mal-API vs MalBehavD)
    - Per-family sample counts on the generalizability set

Usage::

    python evaluate_generalizability.py
"""

import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import config as cfg
from src.evaluation import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_roc_curves,
)
from src.feature_engineering import (
    CATEGORY_FEATURE_NAMES,
    STATISTICAL_FEATURE_NAMES,
    compute_category_features,
    compute_statistical_features,
    tfidf_transform,
)
from src.lstm_model import load_model as load_lstm_model
from src.lstm_model import predict_with_confidence as lstm_predict
from src.preprocessing import (
    compute_unk_ratio,
    pad_sequences,
    preprocess_malbehavd_sequences,
)
from src.utils import get_logger, load_json, load_pickle, save_json
from src.xgboost_model import load_model as load_xgb_model
from src.xgboost_model import predict_with_confidence as xgb_predict

logger = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_xgb_features(samples, tfidf_vectorizer):
    """Build v1 XGBoost feature matrix (TF-IDF + stats + cats)."""
    tfidf = tfidf_transform(samples, tfidf_vectorizer)
    stats = compute_statistical_features(samples)
    cats = compute_category_features(samples)
    return np.hstack([tfidf, stats, cats])


def _plot_sequence_length_comparison(
    train_samples, malbehavd_samples, save_path,
):
    """Side-by-side histogram of sequence lengths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_lens = [len(s["sequence"]) for s in train_samples]
    mb_lens = [len(s["sequence"]) for s in malbehavd_samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    axes[0].hist(train_lens, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Mal-API-2019 (Training)")
    axes[0].set_xlabel("Sequence Length (API calls)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.median(train_lens), color="red", ls="--",
                     label=f"Median: {int(np.median(train_lens))}")
    axes[0].legend()

    axes[1].hist(mb_lens, bins=50, color="darkorange", edgecolor="white")
    axes[1].set_title("MalBehavD-V1 (Generalizability)")
    axes[1].set_xlabel("Sequence Length (API calls)")
    axes[1].axvline(np.median(mb_lens), color="red", ls="--",
                     label=f"Median: {int(np.median(mb_lens))}")
    axes[1].legend()

    fig.suptitle("Sequence Length Distribution Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Sequence length comparison saved to %s", save_path)


def _plot_family_distribution(
    train_labels, malbehavd_labels, class_names, save_path,
):
    """Side-by-side bar chart of family distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_counts = Counter(train_labels)
    mb_counts = Counter(malbehavd_labels)

    x = np.arange(len(class_names))
    width = 0.35

    train_vals = [train_counts.get(name, 0) for name in class_names]
    mb_vals = [mb_counts.get(name, 0) for name in class_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_vals, width, label="Mal-API (Train+Test)",
                    color="steelblue")
    bars2 = ax.bar(x + width / 2, mb_vals, width, label="MalBehavD-V1",
                    color="darkorange")

    ax.set_xlabel("Malware Family")
    ax.set_ylabel("Sample Count")
    ax.set_title("Family Distribution: Training vs Generalizability Set")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()

    # Add count labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h, str(int(h)),
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h, str(int(h)),
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Family distribution comparison saved to %s", save_path)


def _plot_model_comparison_bar(
    metrics_dict, metric_key, title, save_path,
):
    """Grouped bar chart comparing models on a metric across datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(metrics_dict.keys())
    values = [metrics_dict[m].get(metric_key, 0) or 0 for m in model_names]

    colors = ["steelblue", "darkorange", "seagreen"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(model_names, values, color=colors[:len(model_names)])

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _compute_binary_metrics(y_true_binary, y_pred_binary, y_prob_malware):
    """Compute binary (benign vs malware) metrics.

    Args:
        y_true_binary: 0=benign, 1=malware.
        y_pred_binary: 0=benign, 1=malware.
        y_prob_malware: Probability of malware class.

    Returns:
        Dict of metrics.
    """
    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="binary", zero_division=0,
    )
    try:
        auc = roc_auc_score(y_true_binary, y_prob_malware)
    except ValueError:
        auc = None

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc is not None else None,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    plots_dir = cfg.GENERALIZABILITY_PLOTS_DIR
    metrics_dir = cfg.GENERALIZABILITY_METRICS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Load training artefacts ──────────────────────────────────────────
    logger.info("Loading training artefacts...")
    train_samples = load_pickle(cfg.PREPROCESSED_TRAIN_PATH)
    test_samples = load_pickle(cfg.PREPROCESSED_TEST_PATH)
    vocab = load_json(cfg.VOCABULARY_PATH)
    label_encoder = load_pickle(cfg.CACHE_DIR / "label_encoder.pkl")
    tfidf_vectorizer = load_pickle(cfg.CACHE_DIR / "tfidf_vectorizer.pkl")

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    logger.info(
        "Training set: %d train + %d test, %d classes, vocab size %d.",
        len(train_samples), len(test_samples), num_classes, len(vocab),
    )

    # ── Load VT-labeled MalBehavD ────────────────────────────────────────
    logger.info("Loading VT-labeled MalBehavD-V1 samples...")
    malbehavd_data = load_json(cfg.MALBEHAVD_LABELED_PATH)
    all_malbehavd = malbehavd_data["samples"]

    # Separate benign vs family-labeled malware
    benign_samples = [s for s in all_malbehavd if s["label"] == cfg.BENIGN_LABEL]
    malware_samples = [s for s in all_malbehavd if s["label"] != cfg.BENIGN_LABEL]

    malware_labels = Counter(s["label"] for s in malware_samples)
    logger.info(
        "MalBehavD: %d benign, %d malware (%d families), %d dropped.",
        len(benign_samples), len(malware_samples),
        len(malware_labels), len(malbehavd_data.get("dropped_hashes", [])),
    )
    for fam, count in malware_labels.most_common():
        logger.info("  %12s: %d", fam, count)

    # ==================================================================
    # Domain Shift Diagnostics
    # ==================================================================
    logger.info("=" * 70)
    logger.info("DOMAIN SHIFT DIAGNOSTICS")
    logger.info("=" * 70)

    # Preprocess MalBehavD (lowercase + clean + encode with training vocab)
    logger.info("Preprocessing MalBehavD sequences with training vocabulary...")
    malware_processed = preprocess_malbehavd_sequences(malware_samples, vocab)
    all_processed = preprocess_malbehavd_sequences(all_malbehavd, vocab)

    # UNK ratio
    unk_ratio = compute_unk_ratio(malware_processed, vocab)
    logger.info(
        "UNK token ratio on MalBehavD malware: %.2f%% of tokens are unknown.",
        unk_ratio * 100,
    )

    # Compare with Mal-API training UNK ratio (should be ~0 since vocab was
    # built from train, but interesting to quantify)
    train_unk = compute_unk_ratio(train_samples, vocab)
    logger.info("UNK ratio on Mal-API train: %.2f%%", train_unk * 100)

    # Sequence length stats
    train_lens = [len(s["sequence"]) for s in train_samples]
    mb_lens = [len(s["sequence"]) for s in malware_processed]
    diagnostics = {
        "unk_ratio_malbehavd_malware": round(unk_ratio, 4),
        "unk_ratio_malapi_train": round(train_unk, 4),
        "sequence_length_malapi": {
            "mean": round(float(np.mean(train_lens)), 1),
            "median": round(float(np.median(train_lens)), 1),
            "std": round(float(np.std(train_lens)), 1),
            "min": int(np.min(train_lens)),
            "max": int(np.max(train_lens)),
        },
        "sequence_length_malbehavd_malware": {
            "mean": round(float(np.mean(mb_lens)), 1),
            "median": round(float(np.median(mb_lens)), 1),
            "std": round(float(np.std(mb_lens)), 1),
            "min": int(np.min(mb_lens)),
            "max": int(np.max(mb_lens)),
        },
        "vocab_size": len(vocab),
        "malbehavd_malware_count": len(malware_processed),
        "malbehavd_benign_count": len(benign_samples),
        "malbehavd_family_distribution": dict(malware_labels.most_common()),
    }

    logger.info(
        "Sequence lengths — Mal-API: mean=%.0f, median=%.0f | "
        "MalBehavD: mean=%.0f, median=%.0f",
        np.mean(train_lens), np.median(train_lens),
        np.mean(mb_lens), np.median(mb_lens),
    )

    save_json(diagnostics, metrics_dir / "domain_shift_diagnostics.json")

    # Diagnostic plots
    _plot_sequence_length_comparison(
        train_samples, malware_processed,
        plots_dir / "sequence_length_comparison.png",
    )

    all_malapi_labels = (
        [s["label"] for s in train_samples]
        + [s["label"] for s in test_samples]
    )
    _plot_family_distribution(
        all_malapi_labels,
        [s["label"] for s in malware_processed],
        class_names,
        plots_dir / "family_distribution_comparison.png",
    )

    # ==================================================================
    # 8-Class Family Classification (malware-only)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("8-CLASS FAMILY CLASSIFICATION ON MALBEHAVD")
    logger.info("=" * 70)

    # Encode ground-truth labels
    y_true = label_encoder.transform([s["label"] for s in malware_processed])

    # ── XGBoost ──────────────────────────────────────────────────────────
    logger.info("Running XGBoost predictions on MalBehavD...")
    xgb_model = load_xgb_model(cfg.XGBOOST_MODEL_DIR / "best_model.pkl")
    X_mb_xgb = _build_xgb_features(malware_processed, tfidf_vectorizer)
    logger.info("XGBoost feature matrix: %s", X_mb_xgb.shape)

    xgb_preds, xgb_probs = xgb_predict(xgb_model, X_mb_xgb)
    xgb_metrics = compute_all_metrics(y_true, xgb_preds, xgb_probs, class_names)

    logger.info(
        "XGBoost on MalBehavD: acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
        xgb_metrics["accuracy"], xgb_metrics["macro_f1"],
        xgb_metrics["roc_auc_macro"] or 0,
    )
    logger.info(
        "XGBoost classification report:\n%s",
        classification_report(y_true, xgb_preds, target_names=class_names,
                              zero_division=0),
    )

    save_json(xgb_metrics, metrics_dir / "xgboost_family.json")
    plot_confusion_matrix(
        y_true, xgb_preds, class_names,
        "XGBoost on MalBehavD — Confusion Matrix (Normalized)",
        plots_dir / "xgboost_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_true, xgb_preds, class_names,
        "XGBoost on MalBehavD — Confusion Matrix (Counts)",
        plots_dir / "xgboost_confusion_matrix_counts.png",
        normalize=False,
    )
    plot_per_class_f1(
        xgb_metrics,
        "XGBoost on MalBehavD — Per-Class F1",
        plots_dir / "xgboost_per_class_f1.png",
    )
    plot_roc_curves(
        y_true, xgb_probs, class_names,
        "XGBoost on MalBehavD — ROC Curves",
        plots_dir / "xgboost_roc_curves.png",
    )

    # ── LSTM ─────────────────────────────────────────────────────────────
    logger.info("Running LSTM predictions on MalBehavD...")
    lstm_model = load_lstm_model(
        cfg.LSTM_MODEL_DIR / f"best_len{cfg.LSTM_BEST_SEQ_LEN}.keras",
    )

    X_mb_lstm = pad_sequences(
        [s["encoded"] for s in malware_processed],
        max_len=cfg.LSTM_BEST_SEQ_LEN,
    )
    logger.info("LSTM padded sequences: %s", X_mb_lstm.shape)

    lstm_preds, lstm_probs = lstm_predict(lstm_model, X_mb_lstm)
    lstm_metrics = compute_all_metrics(y_true, lstm_preds, lstm_probs, class_names)

    logger.info(
        "LSTM on MalBehavD: acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
        lstm_metrics["accuracy"], lstm_metrics["macro_f1"],
        lstm_metrics["roc_auc_macro"] or 0,
    )
    logger.info(
        "LSTM classification report:\n%s",
        classification_report(y_true, lstm_preds, target_names=class_names,
                              zero_division=0),
    )

    save_json(lstm_metrics, metrics_dir / "lstm_family.json")
    plot_confusion_matrix(
        y_true, lstm_preds, class_names,
        "LSTM on MalBehavD — Confusion Matrix (Normalized)",
        plots_dir / "lstm_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_true, lstm_preds, class_names,
        "LSTM on MalBehavD — Confusion Matrix (Counts)",
        plots_dir / "lstm_confusion_matrix_counts.png",
        normalize=False,
    )
    plot_per_class_f1(
        lstm_metrics,
        "LSTM on MalBehavD — Per-Class F1",
        plots_dir / "lstm_per_class_f1.png",
    )
    plot_roc_curves(
        y_true, lstm_probs, class_names,
        "LSTM on MalBehavD — ROC Curves",
        plots_dir / "lstm_roc_curves.png",
    )

    # ── Ensemble ─────────────────────────────────────────────────────────
    logger.info("Running Ensemble predictions on MalBehavD...")
    from src.ensemble_model import load_model as load_ensemble_model

    ensemble_path = cfg.ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
    if not ensemble_path.exists():
        logger.warning("Ensemble model not found at %s. Skipping.", ensemble_path)
        ens_metrics = None
    else:
        ensemble = load_ensemble_model(ensemble_path)
        ens_preds, ens_probs = ensemble.predict_from_precomputed(
            xgb_probs, lstm_probs,
        )
        ens_metrics = compute_all_metrics(y_true, ens_preds, ens_probs, class_names)

        logger.info(
            "Ensemble on MalBehavD: acc=%.4f, macro-F1=%.4f, ROC-AUC=%.4f",
            ens_metrics["accuracy"], ens_metrics["macro_f1"],
            ens_metrics["roc_auc_macro"] or 0,
        )
        logger.info(
            "Ensemble classification report:\n%s",
            classification_report(y_true, ens_preds, target_names=class_names,
                                  zero_division=0),
        )

        save_json(ens_metrics, metrics_dir / "ensemble_family.json")
        plot_confusion_matrix(
            y_true, ens_preds, class_names,
            "Ensemble on MalBehavD — Confusion Matrix (Normalized)",
            plots_dir / "ensemble_confusion_matrix.png",
        )
        plot_confusion_matrix(
            y_true, ens_preds, class_names,
            "Ensemble on MalBehavD — Confusion Matrix (Counts)",
            plots_dir / "ensemble_confusion_matrix_counts.png",
            normalize=False,
        )
        plot_per_class_f1(
            ens_metrics,
            "Ensemble on MalBehavD — Per-Class F1",
            plots_dir / "ensemble_per_class_f1.png",
        )
        plot_roc_curves(
            y_true, ens_probs, class_names,
            "Ensemble on MalBehavD — ROC Curves",
            plots_dir / "ensemble_roc_curves.png",
        )

    # ── Family classification comparison ─────────────────────────────────
    family_comparison = {
        "xgboost": {
            "accuracy": xgb_metrics["accuracy"],
            "macro_f1": xgb_metrics["macro_f1"],
            "weighted_f1": xgb_metrics["weighted_f1"],
            "roc_auc_macro": xgb_metrics["roc_auc_macro"],
        },
        "lstm": {
            "accuracy": lstm_metrics["accuracy"],
            "macro_f1": lstm_metrics["macro_f1"],
            "weighted_f1": lstm_metrics["weighted_f1"],
            "roc_auc_macro": lstm_metrics["roc_auc_macro"],
        },
    }
    if ens_metrics:
        family_comparison["ensemble"] = {
            "accuracy": ens_metrics["accuracy"],
            "macro_f1": ens_metrics["macro_f1"],
            "weighted_f1": ens_metrics["weighted_f1"],
            "roc_auc_macro": ens_metrics["roc_auc_macro"],
        }
    save_json(family_comparison, metrics_dir / "family_comparison.json")

    _plot_model_comparison_bar(
        family_comparison, "macro_f1",
        "MalBehavD Family Classification — Macro F1",
        plots_dir / "family_comparison_macro_f1.png",
    )

    # ==================================================================
    # Generalization Gap Analysis
    # ==================================================================
    logger.info("=" * 70)
    logger.info("GENERALIZATION GAP (Mal-API test vs MalBehavD)")
    logger.info("=" * 70)

    malapi_metrics = load_json(cfg.METRICS_DIR / "model_comparison.json")

    gap_analysis = {}
    for model_name in ["xgboost", "lstm"]:
        mb_f1 = family_comparison[model_name]["macro_f1"]
        ma_f1 = malapi_metrics[model_name]["macro_f1"]
        gap = ma_f1 - mb_f1
        gap_analysis[model_name] = {
            "malapi_test_macro_f1": ma_f1,
            "malbehavd_macro_f1": mb_f1,
            "generalization_gap": round(gap, 4),
            "relative_drop_pct": round(100 * gap / ma_f1, 1) if ma_f1 > 0 else None,
        }
        logger.info(
            "%s: Mal-API=%.4f, MalBehavD=%.4f, gap=%.4f (%.1f%% drop)",
            model_name.upper(), ma_f1, mb_f1, gap,
            100 * gap / ma_f1 if ma_f1 > 0 else 0,
        )

    if ens_metrics and "ensemble" in malapi_metrics:
        mb_f1 = family_comparison["ensemble"]["macro_f1"]
        ma_f1 = malapi_metrics["ensemble"]["macro_f1"]
        gap = ma_f1 - mb_f1
        gap_analysis["ensemble"] = {
            "malapi_test_macro_f1": ma_f1,
            "malbehavd_macro_f1": mb_f1,
            "generalization_gap": round(gap, 4),
            "relative_drop_pct": round(100 * gap / ma_f1, 1) if ma_f1 > 0 else None,
        }
        logger.info(
            "ENSEMBLE: Mal-API=%.4f, MalBehavD=%.4f, gap=%.4f (%.1f%% drop)",
            ma_f1, mb_f1, gap, 100 * gap / ma_f1 if ma_f1 > 0 else 0,
        )

    save_json(gap_analysis, metrics_dir / "generalization_gap.json")

    # ==================================================================
    # Binary Detection (Benign vs Malware)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("BINARY DETECTION (Benign vs Malware)")
    logger.info("=" * 70)

    # Ground truth: 0=benign, 1=malware
    benign_processed = [s for s in all_processed if s["label"] == cfg.BENIGN_LABEL]
    malware_for_binary = [s for s in all_processed if s["label"] != cfg.BENIGN_LABEL]

    binary_samples = benign_processed + malware_for_binary
    y_binary_true = np.array(
        [0] * len(benign_processed) + [1] * len(malware_for_binary),
    )

    logger.info(
        "Binary set: %d benign + %d malware = %d total.",
        len(benign_processed), len(malware_for_binary), len(binary_samples),
    )

    # XGBoost binary
    logger.info("XGBoost binary detection...")
    X_bin_xgb = _build_xgb_features(binary_samples, tfidf_vectorizer)
    xgb_bin_preds, xgb_bin_probs = xgb_predict(xgb_model, X_bin_xgb)
    # Any non-benign family prediction → malware (1)
    # Since XGBoost always predicts one of the 8 families, every prediction is
    # "malware".  The confidence is 1 - max_prob only when the model is very
    # uncertain.  For a proper binary score, use 1 - P(most-benign-looking class).
    # However, since the model was trained on 8 malware families with no benign
    # class, we treat all predictions as "malware=1" and use the max probability
    # as the malware confidence.
    xgb_bin_pred_binary = np.ones(len(binary_samples), dtype=int)  # always predicts malware
    xgb_bin_prob_malware = xgb_bin_probs.max(axis=1)  # confidence in its prediction

    xgb_binary_metrics = _compute_binary_metrics(
        y_binary_true, xgb_bin_pred_binary, xgb_bin_prob_malware,
    )
    logger.info(
        "XGBoost binary: acc=%.4f, prec=%.4f, rec=%.4f, F1=%.4f, AUC=%.4f",
        xgb_binary_metrics["accuracy"], xgb_binary_metrics["precision"],
        xgb_binary_metrics["recall"], xgb_binary_metrics["f1"],
        xgb_binary_metrics["roc_auc"] or 0,
    )

    # LSTM binary
    logger.info("LSTM binary detection...")
    X_bin_lstm = pad_sequences(
        [s["encoded"] for s in binary_samples],
        max_len=cfg.LSTM_BEST_SEQ_LEN,
    )
    lstm_bin_preds, lstm_bin_probs = lstm_predict(lstm_model, X_bin_lstm)
    lstm_bin_pred_binary = np.ones(len(binary_samples), dtype=int)
    lstm_bin_prob_malware = lstm_bin_probs.max(axis=1)

    lstm_binary_metrics = _compute_binary_metrics(
        y_binary_true, lstm_bin_pred_binary, lstm_bin_prob_malware,
    )
    logger.info(
        "LSTM binary: acc=%.4f, prec=%.4f, rec=%.4f, F1=%.4f, AUC=%.4f",
        lstm_binary_metrics["accuracy"], lstm_binary_metrics["precision"],
        lstm_binary_metrics["recall"], lstm_binary_metrics["f1"],
        lstm_binary_metrics["roc_auc"] or 0,
    )

    # Ensemble binary
    ens_binary_metrics = None
    if ens_metrics:
        logger.info("Ensemble binary detection...")
        ens_bin_preds, ens_bin_probs = ensemble.predict_from_precomputed(
            xgb_bin_probs, lstm_bin_probs,
        )
        ens_bin_pred_binary = np.ones(len(binary_samples), dtype=int)
        ens_bin_prob_malware = ens_bin_probs.max(axis=1)

        ens_binary_metrics = _compute_binary_metrics(
            y_binary_true, ens_bin_pred_binary, ens_bin_prob_malware,
        )
        logger.info(
            "Ensemble binary: acc=%.4f, prec=%.4f, rec=%.4f, F1=%.4f, AUC=%.4f",
            ens_binary_metrics["accuracy"], ens_binary_metrics["precision"],
            ens_binary_metrics["recall"], ens_binary_metrics["f1"],
            ens_binary_metrics["roc_auc"] or 0,
        )

    binary_comparison = {
        "xgboost": xgb_binary_metrics,
        "lstm": lstm_binary_metrics,
    }
    if ens_binary_metrics:
        binary_comparison["ensemble"] = ens_binary_metrics
    save_json(binary_comparison, metrics_dir / "binary_detection.json")

    # Binary detection note
    logger.info(
        "NOTE: Models were trained on 8 malware families (no benign class). "
        "All predictions are inherently 'malware'. Binary recall is always "
        "1.0 for malware. The interesting signal is whether benign samples "
        "get LOW confidence scores (indicating the model 'knows' they're "
        "different). AUC captures this."
    )

    # ==================================================================
    # Per-Class Generalization Gap
    # ==================================================================
    logger.info("=" * 70)
    logger.info("PER-CLASS GENERALIZATION (XGBoost)")
    logger.info("=" * 70)

    xgb_malapi = load_json(cfg.METRICS_DIR / "xgboost_evaluation.json")
    per_class_gap = {}
    for name in class_names:
        ma_f1 = xgb_malapi["test"]["per_class"][name]["f1"]
        mb_f1 = xgb_metrics["per_class"][name]["f1"]
        mb_support = xgb_metrics["per_class"][name]["support"]
        gap = ma_f1 - mb_f1
        per_class_gap[name] = {
            "malapi_f1": ma_f1,
            "malbehavd_f1": mb_f1,
            "gap": round(gap, 4),
            "malbehavd_support": mb_support,
        }
        logger.info(
            "  %12s: Mal-API=%.4f, MalBehavD=%.4f, gap=%+.4f  (n=%d)",
            name, ma_f1, mb_f1, -gap, mb_support,
        )

    save_json(per_class_gap, metrics_dir / "per_class_gap_xgboost.json")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Phase 7 — Generalizability Evaluation Summary")
    print(f"{'='*65}")

    print(f"\nDomain Shift Diagnostics:")
    print(f"  UNK ratio (MalBehavD malware): {unk_ratio*100:.2f}%")
    print(f"  Sequence lengths — Mal-API median: {np.median(train_lens):.0f}, "
          f"MalBehavD median: {np.median(mb_lens):.0f}")

    print(f"\n{'Model':<12} {'MalAPI F1':>10} {'MalBehD F1':>11} {'Gap':>8} {'Drop%':>8}")
    print(f"{'-'*50}")
    for model_name in ["xgboost", "lstm"] + (["ensemble"] if ens_metrics else []):
        if model_name in gap_analysis:
            g = gap_analysis[model_name]
            print(
                f"{model_name:<12} {g['malapi_test_macro_f1']:>10.4f} "
                f"{g['malbehavd_macro_f1']:>11.4f} "
                f"{g['generalization_gap']:>8.4f} "
                f"{g['relative_drop_pct']:>7.1f}%"
            )

    print(f"\nBinary Detection (Benign vs Malware):")
    print(f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'F1':>10} {'AUC':>10}")
    print(f"{'-'*52}")
    for model_name, bm in binary_comparison.items():
        print(
            f"{model_name:<12} {bm['accuracy']:>10.4f} {bm['precision']:>10.4f} "
            f"{bm['f1']:>10.4f} {bm['roc_auc'] or 0:>10.4f}"
        )

    print(f"\nResults saved to: {cfg.GENERALIZABILITY_DIR}")


if __name__ == "__main__":
    main()
