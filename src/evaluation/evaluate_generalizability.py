"""
Phase 7 — Generalizability evaluation on MalBehavD-V1.

Tests whether the two-stage pipeline generalizes to an independent dataset:

1. **Stage-1 binary evaluation**: Binary XGBoost on all MalBehavD samples.
2. **Stage-2 family evaluation**: 7-class models on malware-only samples.
3. **End-to-end pipeline**: Stage-1 (50% threshold) gates Stage-2.
   Scored as an 8-class problem (Benign + 7 families).
4. **Generalization gap**: Compares MalBehavD metrics vs Mal-API test metrics.

Design notes:
  - Stage-1 and Stage-2 use **different preprocessing** pipelines:
    Stage-1 uses Olivera-style (deduplicate + truncate to 100 calls) because
    the binary model was trained on Olivera-format data.
    Stage-2 uses standard cleaning (collapse to max 5 repeats, full length)
    because the family models were trained on full Mal-API sequences.
  - For end-to-end evaluation, each MalBehavD sample is preprocessed both
    ways.  Stage-1 decides routing; Stage-2 uses its own preprocessing.
  - The 50% threshold is used for scoring purposes.  In deployment, a
    three-zone approach (high/uncertain/low) would be used instead.

Usage::

    python -m src.evaluation.evaluate_generalizability
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
from sklearn.preprocessing import label_binarize

import config as cfg
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_roc_curves,
)
from src.model_training.binary_xgboost_model import build_binary_feature_matrix
from src.model_training.feature_engineering import (
    compute_category_features,
    compute_statistical_features,
    tfidf_transform,
)
from src.model_training.lstm_model import load_model as load_lstm_model
from src.model_training.lstm_model import predict_with_confidence as lstm_predict
from src.data_loading.preprocessing import (
    clean_sequence,
    compute_unk_ratio,
    encode_sequence,
    olivera_style_preprocess,
    pad_sequences,
    preprocess_malbehavd_sequences,
)
from src.utils import get_logger, load_json, load_pickle, save_json
from src.model_training.xgboost_model import load_model as load_xgb_model
from src.model_training.xgboost_model import predict_with_confidence as xgb_predict

logger = get_logger(__name__)

# Threshold for Stage-1 → Stage-2 routing.
# Samples with P(malware) >= this value are sent to Stage-2.
# For evaluation scoring we use 0.50 (strict binary decision).
BINARY_THRESHOLD = 0.50


# ── Helpers ──────────────────────────────────────────────────────────────────


def _olivera_preprocess_malbehavd(samples, binary_vocab=None):
    """Olivera-preprocess MalBehavD samples for Stage-1 binary model.

    Steps: lowercase → remove sandbox tokens → deduplicate consecutive
    → truncate to 100 calls.

    The binary model does not use encoded sequences (it uses TF-IDF +
    statistical features), so encoding is optional.

    Args:
        samples: Raw MalBehavD samples with ``sequence`` field.
        binary_vocab: Optional vocab for encoding (not needed for XGBoost).

    Returns:
        New list of sample dicts with Olivera-preprocessed sequences.
    """
    processed = []
    for s in samples:
        new = dict(s)
        lowered = [tok.lower() for tok in s["sequence"]]
        new["sequence"] = olivera_style_preprocess(lowered)
        processed.append(new)
    return processed


def _build_family_xgb_features(samples, tfidf_vectorizer):
    """Build Stage-2 XGBoost feature matrix (TF-IDF + stats + categories)."""
    tfidf = tfidf_transform(samples, tfidf_vectorizer)
    stats = compute_statistical_features(samples)
    cats = compute_category_features(samples)
    return np.hstack([tfidf, stats, cats])


def _plot_sequence_length_comparison(train_samples, mb_samples, save_path):
    """Side-by-side histogram of sequence lengths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_lens = [len(s["sequence"]) for s in train_samples]
    mb_lens = [len(s["sequence"]) for s in mb_samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    axes[0].hist(train_lens, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Mal-API-2019 (Training)")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.median(train_lens), color="red", ls="--",
                     label=f"Median: {int(np.median(train_lens))}")
    axes[0].legend()

    axes[1].hist(mb_lens, bins=50, color="darkorange", edgecolor="white")
    axes[1].set_title("MalBehavD-V1 (Generalizability)")
    axes[1].set_xlabel("Sequence Length")
    axes[1].axvline(np.median(mb_lens), color="red", ls="--",
                     label=f"Median: {int(np.median(mb_lens))}")
    axes[1].legend()

    fig.suptitle("Sequence Length Distribution Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_family_distribution(train_labels, mb_labels, class_names, save_path):
    """Side-by-side bar chart of family distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_counts = Counter(train_labels)
    mb_counts = Counter(mb_labels)

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2,
           [train_counts.get(n, 0) for n in class_names],
           width, label="Mal-API (Train+Test)", color="steelblue")
    ax.bar(x + width / 2,
           [mb_counts.get(n, 0) for n in class_names],
           width, label="MalBehavD-V1", color="darkorange")

    ax.set_xlabel("Malware Family")
    ax.set_ylabel("Sample Count")
    ax.set_title("Family Distribution: Training vs Generalizability Set")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_model_comparison_bar(metrics_dict, metric_key, title, save_path):
    """Grouped bar chart comparing models on a metric."""
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


def _plot_end_to_end_confusion(y_true_labels, y_pred_labels, all_class_names,
                               title, save_path):
    """Plot confusion matrix for end-to-end 8-class predictions using string labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true_labels, y_pred_labels, labels=all_class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    # Handle rows with zero support (no true samples for that class)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=all_class_names, yticklabels=all_class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    plots_dir = cfg.GENERALIZABILITY_PLOTS_DIR
    metrics_dir = cfg.GENERALIZABILITY_METRICS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Load all artifacts
    # ==================================================================
    logger.info("Loading artifacts...")

    # Stage-1 (binary) artifacts
    bin_model = load_xgb_model(cfg.BINARY_XGBOOST_MODEL_DIR / "best_model.pkl")
    bin_tfidf = load_pickle(cfg.BINARY_CACHE_DIR / "tfidf_vectorizer.pkl")
    bin_label_enc = load_pickle(cfg.BINARY_LABEL_ENCODER_PATH)
    # Binary label encoder: classes = ["Benign", "Malware"] (alphabetical)
    malware_idx = list(bin_label_enc.classes_).index(cfg.MALWARE_LABEL)

    # Stage-2 (family) artifacts — no-Trojan 7-class
    family_tfidf = load_pickle(cfg.NO_TROJAN_CACHE_DIR / "tfidf_vectorizer.pkl")
    family_vocab = load_json(cfg.NO_TROJAN_VOCABULARY_PATH)
    family_label_enc = load_pickle(cfg.NO_TROJAN_LABEL_ENCODER_PATH)
    family_class_names = list(family_label_enc.classes_)

    family_xgb = load_xgb_model(cfg.NO_TROJAN_XGBOOST_MODEL_DIR / "best_model.pkl")
    family_lstm = load_lstm_model(
        cfg.NO_TROJAN_LSTM_MODEL_DIR / f"best_len{cfg.LSTM_BEST_SEQ_LEN}.keras",
    )

    from src.model_training.ensemble_model import (
        EnsembleClassifier,
        load_model as load_ensemble,
    )
    import __main__
    __main__.EnsembleClassifier = EnsembleClassifier
    ens_path = cfg.NO_TROJAN_ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
    family_ensemble = load_ensemble(ens_path) if ens_path.exists() else None

    # Training data (for diagnostics)
    train_samples = load_pickle(cfg.NO_TROJAN_TRAIN_PATH)
    test_samples = load_pickle(cfg.NO_TROJAN_TEST_PATH)

    # ==================================================================
    # Load MalBehavD (no-Trojan version)
    # ==================================================================
    logger.info("Loading no-Trojan MalBehavD data...")
    mb_data = load_json(cfg.NO_TROJAN_MALBEHAVD_PATH)
    all_mb_samples = mb_data["samples"]

    benign_samples = [s for s in all_mb_samples if s["label"] == cfg.BENIGN_LABEL]
    malware_samples = [s for s in all_mb_samples if s["label"] != cfg.BENIGN_LABEL]

    logger.info(
        "MalBehavD (no-Trojan): %d benign, %d malware, %d total.",
        len(benign_samples), len(malware_samples), len(all_mb_samples),
    )
    malware_dist = Counter(s["label"] for s in malware_samples)
    for fam, count in malware_dist.most_common():
        logger.info("  %12s: %d", fam, count)

    # ==================================================================
    # Domain Shift Diagnostics
    # ==================================================================
    logger.info("=" * 70)
    logger.info("DOMAIN SHIFT DIAGNOSTICS")
    logger.info("=" * 70)

    # Standard-preprocess malware for diagnostics and Stage-2
    malware_std = preprocess_malbehavd_sequences(malware_samples, family_vocab)
    all_std = preprocess_malbehavd_sequences(all_mb_samples, family_vocab)

    unk_ratio = compute_unk_ratio(malware_std, family_vocab)
    train_unk = compute_unk_ratio(train_samples, family_vocab)

    train_lens = [len(s["sequence"]) for s in train_samples]
    mb_lens = [len(s["sequence"]) for s in malware_std]

    diagnostics = {
        "unk_ratio_malbehavd_malware": round(unk_ratio, 4),
        "unk_ratio_malapi_train": round(train_unk, 4),
        "sequence_length_malapi": {
            "mean": round(float(np.mean(train_lens)), 1),
            "median": round(float(np.median(train_lens)), 1),
        },
        "sequence_length_malbehavd_malware": {
            "mean": round(float(np.mean(mb_lens)), 1),
            "median": round(float(np.median(mb_lens)), 1),
        },
        "vocab_size": len(family_vocab),
        "malbehavd_malware_count": len(malware_std),
        "malbehavd_benign_count": len(benign_samples),
        "malbehavd_family_distribution": dict(malware_dist.most_common()),
    }
    save_json(diagnostics, metrics_dir / "domain_shift_diagnostics.json")

    logger.info("UNK ratio — Mal-API train: %.2f%%, MalBehavD malware: %.2f%%",
                train_unk * 100, unk_ratio * 100)

    _plot_sequence_length_comparison(
        train_samples, malware_std,
        plots_dir / "sequence_length_comparison.png",
    )
    all_malapi_labels = (
        [s["label"] for s in train_samples] + [s["label"] for s in test_samples]
    )
    _plot_family_distribution(
        all_malapi_labels, [s["label"] for s in malware_std],
        family_class_names, plots_dir / "family_distribution_comparison.png",
    )

    # ==================================================================
    # Stage-1: Binary XGBoost on MalBehavD
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STAGE-1: BINARY XGBOOST ON MALBEHAVD")
    logger.info("=" * 70)

    # Olivera-preprocess ALL MalBehavD samples for Stage-1
    all_olivera = _olivera_preprocess_malbehavd(all_mb_samples)

    # Build binary features
    X_bin = build_binary_feature_matrix(all_olivera, bin_tfidf)
    logger.info("Binary feature matrix: %s", X_bin.shape)

    # Predict
    bin_preds_int, bin_probs = xgb_predict(bin_model, X_bin)
    bin_pred_labels = bin_label_enc.inverse_transform(bin_preds_int)
    p_malware = bin_probs[:, malware_idx]

    # Ground truth for binary: Benign=0, any family=1
    y_bin_true = np.array([
        0 if s["label"] == cfg.BENIGN_LABEL else 1 for s in all_mb_samples
    ])
    y_bin_pred = (p_malware >= BINARY_THRESHOLD).astype(int)

    bin_acc = accuracy_score(y_bin_true, y_bin_pred)
    bin_prec, bin_rec, bin_f1, _ = precision_recall_fscore_support(
        y_bin_true, y_bin_pred, average="macro", zero_division=0,
    )
    try:
        bin_auc = roc_auc_score(y_bin_true, p_malware)
    except ValueError:
        bin_auc = None

    stage1_metrics = {
        "threshold": BINARY_THRESHOLD,
        "accuracy": round(float(bin_acc), 4),
        "macro_precision": round(float(bin_prec), 4),
        "macro_recall": round(float(bin_rec), 4),
        "macro_f1": round(float(bin_f1), 4),
        "roc_auc": round(float(bin_auc), 4) if bin_auc else None,
        "n_benign": int(y_bin_true.sum() == 0),
        "n_predicted_malware": int(y_bin_pred.sum()),
        "n_predicted_benign": int((1 - y_bin_pred).sum()),
        "true_benign_passed_to_stage2": int(
            ((y_bin_true == 0) & (y_bin_pred == 1)).sum()
        ),
        "true_malware_blocked": int(
            ((y_bin_true == 1) & (y_bin_pred == 0)).sum()
        ),
    }

    logger.info(
        "Stage-1 binary: acc=%.4f, macro-F1=%.4f, AUC=%.4f",
        bin_acc, bin_f1, bin_auc or 0,
    )
    logger.info(
        "Routing: %d→Stage-2 (%d benign leak), %d→Benign (%d malware blocked).",
        y_bin_pred.sum(),
        stage1_metrics["true_benign_passed_to_stage2"],
        (1 - y_bin_pred).sum(),
        stage1_metrics["true_malware_blocked"],
    )
    save_json(stage1_metrics, metrics_dir / "stage1_binary.json")

    # Binary confusion matrix (Benign vs Malware)
    _plot_end_to_end_confusion(
        ["Benign" if y == 0 else "Malware" for y in y_bin_true],
        ["Benign" if y == 0 else "Malware" for y in y_bin_pred],
        ["Benign", "Malware"],
        f"Stage-1 Binary on MalBehavD (threshold={BINARY_THRESHOLD})",
        plots_dir / "stage1_binary_confusion.png",
    )

    # ==================================================================
    # Stage-2: 7-Class Family Classification (malware-only)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STAGE-2: 7-CLASS FAMILY CLASSIFICATION (malware-only)")
    logger.info("=" * 70)

    # Ground truth (malware only, no Trojan)
    y_family_true = family_label_enc.transform([s["label"] for s in malware_std])

    # ── XGBoost ──
    X_mb_xgb = _build_family_xgb_features(malware_std, family_tfidf)
    xgb_preds, xgb_probs = xgb_predict(family_xgb, X_mb_xgb)
    xgb_metrics = compute_all_metrics(
        y_family_true, xgb_preds, xgb_probs, family_class_names,
    )
    logger.info("XGBoost family: acc=%.4f, macro-F1=%.4f",
                xgb_metrics["accuracy"], xgb_metrics["macro_f1"])
    save_json(xgb_metrics, metrics_dir / "stage2_xgboost_family.json")

    plot_confusion_matrix(
        y_family_true, xgb_preds, family_class_names,
        "Stage-2 XGBoost on MalBehavD (7-class)",
        plots_dir / "stage2_xgboost_confusion.png",
    )
    plot_per_class_f1(
        xgb_metrics, "Stage-2 XGBoost Per-Class F1 (MalBehavD)",
        plots_dir / "stage2_xgboost_per_class_f1.png",
    )

    # ── LSTM ──
    X_mb_lstm = pad_sequences(
        [s["encoded"] for s in malware_std], max_len=cfg.LSTM_BEST_SEQ_LEN,
    )
    lstm_preds, lstm_probs = lstm_predict(family_lstm, X_mb_lstm)
    lstm_metrics = compute_all_metrics(
        y_family_true, lstm_preds, lstm_probs, family_class_names,
    )
    logger.info("LSTM family: acc=%.4f, macro-F1=%.4f",
                lstm_metrics["accuracy"], lstm_metrics["macro_f1"])
    save_json(lstm_metrics, metrics_dir / "stage2_lstm_family.json")

    plot_confusion_matrix(
        y_family_true, lstm_preds, family_class_names,
        "Stage-2 LSTM on MalBehavD (7-class)",
        plots_dir / "stage2_lstm_confusion.png",
    )

    # ── Ensemble ──
    ens_metrics = None
    if family_ensemble:
        ens_preds, ens_probs = family_ensemble.predict_from_precomputed(
            xgb_probs, lstm_probs,
        )
        ens_metrics = compute_all_metrics(
            y_family_true, ens_preds, ens_probs, family_class_names,
        )
        logger.info("Ensemble family: acc=%.4f, macro-F1=%.4f",
                     ens_metrics["accuracy"], ens_metrics["macro_f1"])
        save_json(ens_metrics, metrics_dir / "stage2_ensemble_family.json")

    # Stage-2 comparison
    family_comparison = {
        "xgboost": {"accuracy": xgb_metrics["accuracy"],
                     "macro_f1": xgb_metrics["macro_f1"]},
        "lstm": {"accuracy": lstm_metrics["accuracy"],
                 "macro_f1": lstm_metrics["macro_f1"]},
    }
    if ens_metrics:
        family_comparison["ensemble"] = {
            "accuracy": ens_metrics["accuracy"],
            "macro_f1": ens_metrics["macro_f1"],
        }
    save_json(family_comparison, metrics_dir / "stage2_family_comparison.json")

    _plot_model_comparison_bar(
        family_comparison, "macro_f1",
        "Stage-2 Family Classification on MalBehavD — Macro F1",
        plots_dir / "stage2_family_comparison.png",
    )

    # ==================================================================
    # End-to-End Pipeline Evaluation
    # ==================================================================
    # Design: Stage-1 binary XGBoost with 50% threshold gates Stage-2.
    # Samples below threshold → "Benign".
    # Samples above threshold → Stage-2 family prediction.
    # Scored as 8-class: [Benign] + 7 malware families.
    #
    # Why this design:
    # - Mirrors the actual deployment pipeline
    # - Stage-1 errors propagate: a benign sample passing Stage-1 will
    #   ALWAYS be misclassified (Stage-2 has no Benign class)
    # - A malware sample blocked by Stage-1 is labeled Benign (false neg)
    # - This is the most honest evaluation of end-to-end performance
    # ==================================================================
    logger.info("=" * 70)
    logger.info("END-TO-END PIPELINE (Stage-1 → Stage-2)")
    logger.info("=" * 70)

    # The 8 end-to-end class names: Benign + 7 families
    e2e_class_names = [cfg.BENIGN_LABEL] + family_class_names

    # Ground truth labels (strings) for all samples
    y_true_labels = [s["label"] for s in all_mb_samples]

    # Standard-preprocess ALL samples for Stage-2 (we need this for samples
    # that pass Stage-1, but we preprocess all upfront for simplicity)
    # all_std was already computed above during diagnostics

    # Build Stage-2 features for ALL samples (we'll index into these)
    X_all_xgb = _build_family_xgb_features(all_std, family_tfidf)
    X_all_lstm = pad_sequences(
        [s["encoded"] for s in all_std], max_len=cfg.LSTM_BEST_SEQ_LEN,
    )

    # Stage-1 routing: which samples go to Stage-2?
    passes_stage1 = y_bin_pred == 1  # True = routed to Stage-2

    def _run_end_to_end(model_name, family_preds_all):
        """Combine Stage-1 routing with Stage-2 family predictions.

        Args:
            model_name: For logging.
            family_preds_all: Stage-2 integer predictions for ALL samples
                (only used for samples passing Stage-1).

        Returns:
            List of predicted label strings (length = n_samples).
        """
        family_pred_labels = family_label_enc.inverse_transform(family_preds_all)
        pred_labels = []
        for i in range(len(all_mb_samples)):
            if passes_stage1[i]:
                pred_labels.append(family_pred_labels[i])
            else:
                pred_labels.append(cfg.BENIGN_LABEL)
        return pred_labels

    # Run Stage-2 predictions on ALL samples (even those Stage-1 would block,
    # since we need the predictions indexed by position)
    xgb_all_preds, xgb_all_probs = xgb_predict(family_xgb, X_all_xgb)
    lstm_all_preds, lstm_all_probs = lstm_predict(family_lstm, X_all_lstm)

    e2e_results = {}

    for model_name, family_preds in [
        ("xgboost", xgb_all_preds),
        ("lstm", lstm_all_preds),
    ]:
        pred_labels = _run_end_to_end(model_name, family_preds)

        # Classification report (8-class)
        report_str = classification_report(
            y_true_labels, pred_labels, labels=e2e_class_names,
            zero_division=0,
        )
        report_dict = classification_report(
            y_true_labels, pred_labels, labels=e2e_class_names,
            output_dict=True, zero_division=0,
        )

        acc = accuracy_score(y_true_labels, pred_labels)
        macro_f1 = f1_score(
            y_true_labels, pred_labels, labels=e2e_class_names,
            average="macro", zero_division=0,
        )
        weighted_f1 = f1_score(
            y_true_labels, pred_labels, labels=e2e_class_names,
            average="weighted", zero_division=0,
        )

        e2e_results[model_name] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": {
                name: {
                    "precision": round(report_dict[name]["precision"], 4),
                    "recall": round(report_dict[name]["recall"], 4),
                    "f1": round(report_dict[name]["f1-score"], 4),
                    "support": int(report_dict[name]["support"]),
                }
                for name in e2e_class_names
                if name in report_dict
            },
        }

        logger.info(
            "E2E %s: acc=%.4f, macro-F1=%.4f, weighted-F1=%.4f",
            model_name.upper(), acc, macro_f1, weighted_f1,
        )
        logger.info("E2E %s report:\n%s", model_name.upper(), report_str)

        _plot_end_to_end_confusion(
            y_true_labels, pred_labels, e2e_class_names,
            f"End-to-End Pipeline ({model_name.upper()}) on MalBehavD",
            plots_dir / f"e2e_{model_name}_confusion.png",
        )

    # Ensemble E2E
    if family_ensemble:
        ens_all_preds, ens_all_probs = family_ensemble.predict_from_precomputed(
            xgb_all_probs, lstm_all_probs,
        )
        pred_labels = _run_end_to_end("ensemble", ens_all_preds)

        report_str = classification_report(
            y_true_labels, pred_labels, labels=e2e_class_names,
            zero_division=0,
        )
        report_dict = classification_report(
            y_true_labels, pred_labels, labels=e2e_class_names,
            output_dict=True, zero_division=0,
        )

        acc = accuracy_score(y_true_labels, pred_labels)
        macro_f1 = f1_score(
            y_true_labels, pred_labels, labels=e2e_class_names,
            average="macro", zero_division=0,
        )
        weighted_f1 = f1_score(
            y_true_labels, pred_labels, labels=e2e_class_names,
            average="weighted", zero_division=0,
        )

        e2e_results["ensemble"] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": {
                name: {
                    "precision": round(report_dict[name]["precision"], 4),
                    "recall": round(report_dict[name]["recall"], 4),
                    "f1": round(report_dict[name]["f1-score"], 4),
                    "support": int(report_dict[name]["support"]),
                }
                for name in e2e_class_names
                if name in report_dict
            },
        }

        logger.info("E2E ENSEMBLE: acc=%.4f, macro-F1=%.4f", acc, macro_f1)
        logger.info("E2E Ensemble report:\n%s", report_str)

        _plot_end_to_end_confusion(
            y_true_labels, pred_labels, e2e_class_names,
            "End-to-End Pipeline (Ensemble) on MalBehavD",
            plots_dir / "e2e_ensemble_confusion.png",
        )

    save_json(e2e_results, metrics_dir / "end_to_end_pipeline.json")

    _plot_model_comparison_bar(
        {k: v for k, v in e2e_results.items()},
        "macro_f1",
        "End-to-End Pipeline on MalBehavD — Macro F1 (8-class)",
        plots_dir / "e2e_comparison_macro_f1.png",
    )

    # ==================================================================
    # Generalization Gap Analysis
    # ==================================================================
    logger.info("=" * 70)
    logger.info("GENERALIZATION GAP (Mal-API test vs MalBehavD)")
    logger.info("=" * 70)

    # Load Mal-API test metrics for comparison
    malapi_comparison = load_json(cfg.NO_TROJAN_METRICS_DIR / "model_comparison.json")

    gap_analysis = {}
    for model_name in ["xgboost", "lstm"] + (["ensemble"] if ens_metrics else []):
        mb_f1 = family_comparison[model_name]["macro_f1"]
        ma_f1 = malapi_comparison[model_name]["macro_f1"]
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

    save_json(gap_analysis, metrics_dir / "generalization_gap.json")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*65}")
    print("Phase 7 — Generalizability Evaluation Summary")
    print(f"{'='*65}")

    print(f"\nDomain Shift:")
    print(f"  UNK ratio (MalBehavD malware): {unk_ratio*100:.2f}%")
    print(f"  Sequence lengths — Mal-API median: {np.median(train_lens):.0f}, "
          f"MalBehavD median: {np.median(mb_lens):.0f}")

    print(f"\n--- Stage-1: Binary Detection on MalBehavD ---")
    print(f"  Accuracy: {stage1_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {stage1_metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC:  {stage1_metrics['roc_auc'] or 0:.4f}")
    print(f"  Benign leaked to Stage-2: {stage1_metrics['true_benign_passed_to_stage2']}")
    print(f"  Malware blocked by Stage-1: {stage1_metrics['true_malware_blocked']}")

    print(f"\n--- Stage-2: Family Classification (malware-only) ---")
    print(f"  {'Model':<12} {'MalAPI F1':>10} {'MalBehD F1':>11} {'Gap':>8} {'Drop%':>8}")
    print(f"  {'-'*50}")
    for name in ["xgboost", "lstm"] + (["ensemble"] if ens_metrics else []):
        g = gap_analysis.get(name, {})
        if g:
            print(f"  {name:<12} {g['malapi_test_macro_f1']:>10.4f} "
                  f"{g['malbehavd_macro_f1']:>11.4f} "
                  f"{g['generalization_gap']:>8.4f} "
                  f"{g['relative_drop_pct']:>7.1f}%")

    print(f"\n--- End-to-End Pipeline (8-class: Benign + 7 families) ---")
    print(f"  {'Model':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12}")
    print(f"  {'-'*46}")
    for name, m in e2e_results.items():
        print(f"  {name:<12} {m['accuracy']:>10.4f} "
              f"{m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f}")

    print(f"\nResults saved to: {cfg.GENERALIZABILITY_DIR}")


if __name__ == "__main__":
    main()
