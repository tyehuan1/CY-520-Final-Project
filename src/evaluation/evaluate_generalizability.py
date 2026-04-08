"""
Phase 7 — Generalizability evaluation on MalBehavD-V1 (malware-only).

Tests whether the single-stage 7-class family classification models generalize
to an independent dataset (MalBehavD-V1).  Only malware samples are used
(no binary detection stage).

Evaluations:
1. **Domain shift diagnostics**: Sequence length and UNK ratio comparison.
2. **Family classification**: XGBoost, LSTM, and Ensemble on MalBehavD malware.
3. **Generalization gap**: Compares MalBehavD metrics vs Mal-API test metrics.

Usage::

    python -m src.evaluation.evaluate_generalizability
"""

import numpy as np
from collections import Counter
from sklearn.metrics import classification_report

import config as cfg
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_per_class_f1,
)
from src.model_training.feature_engineering import build_feature_matrix
from src.model_training.lstm_model import load_model as load_lstm_model
from src.model_training.lstm_model import predict_with_confidence as lstm_predict
from src.data_loading.preprocessing import (
    compute_unk_ratio,
    pad_sequences,
    preprocess_malbehavd_sequences,
)
from src.utils import get_logger, load_json, load_pickle, save_json
from src.model_training.xgboost_model import load_model as load_xgb_model
from src.model_training.xgboost_model import predict_with_confidence as xgb_predict

logger = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


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


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    plots_dir = cfg.GENERALIZABILITY_PLOTS_DIR
    metrics_dir = cfg.GENERALIZABILITY_METRICS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Load artifacts
    # ==================================================================
    logger.info("Loading artifacts...")

    # Family model artifacts — no-Trojan 7-class
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
    # Load MalBehavD (no-Trojan, malware only)
    # ==================================================================
    logger.info("Loading no-Trojan MalBehavD data...")
    mb_data = load_json(cfg.NO_TROJAN_MALBEHAVD_PATH)
    all_mb_samples = mb_data["samples"]

    # Filter to malware only (no benign — single-stage model has no benign class)
    malware_samples = [s for s in all_mb_samples if s["label"] != cfg.BENIGN_LABEL]

    logger.info(
        "MalBehavD (no-Trojan, malware-only): %d samples.", len(malware_samples),
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

    # Standard-preprocess malware for diagnostics and evaluation
    malware_std = preprocess_malbehavd_sequences(malware_samples, family_vocab)

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
    # 7-Class Family Classification on MalBehavD (malware-only)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("7-CLASS FAMILY CLASSIFICATION ON MALBEHAVD (malware-only)")
    logger.info("=" * 70)

    # Ground truth
    y_family_true = family_label_enc.transform([s["label"] for s in malware_std])

    # ── XGBoost ──
    X_mb_xgb = build_feature_matrix(malware_std, family_tfidf)
    xgb_preds, xgb_probs = xgb_predict(family_xgb, X_mb_xgb)
    xgb_metrics = compute_all_metrics(
        y_family_true, xgb_preds, xgb_probs, family_class_names,
    )
    logger.info("XGBoost family: acc=%.4f, macro-F1=%.4f",
                xgb_metrics["accuracy"], xgb_metrics["macro_f1"])
    save_json(xgb_metrics, metrics_dir / "xgboost_family.json")

    logger.info("XGBoost report:\n%s",
                classification_report(y_family_true, xgb_preds,
                                       target_names=family_class_names, zero_division=0))

    plot_confusion_matrix(
        y_family_true, xgb_preds, family_class_names,
        "XGBoost on MalBehavD (7-class)",
        plots_dir / "xgboost_confusion.png",
    )
    plot_per_class_f1(
        xgb_metrics, "XGBoost Per-Class F1 (MalBehavD)",
        plots_dir / "xgboost_per_class_f1.png",
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
    save_json(lstm_metrics, metrics_dir / "lstm_family.json")

    plot_confusion_matrix(
        y_family_true, lstm_preds, family_class_names,
        "LSTM on MalBehavD (7-class)",
        plots_dir / "lstm_confusion.png",
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
        save_json(ens_metrics, metrics_dir / "ensemble_family.json")

    # Model comparison
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
    save_json(family_comparison, metrics_dir / "family_comparison.json")

    _plot_model_comparison_bar(
        family_comparison, "macro_f1",
        "Family Classification on MalBehavD — Macro F1",
        plots_dir / "family_comparison.png",
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

    print(f"\n--- Family Classification (malware-only) ---")
    print(f"  {'Model':<12} {'MalAPI F1':>10} {'MalBehD F1':>11} {'Gap':>8} {'Drop%':>8}")
    print(f"  {'-'*50}")
    for name in ["xgboost", "lstm"] + (["ensemble"] if ens_metrics else []):
        g = gap_analysis.get(name, {})
        if g:
            print(f"  {name:<12} {g['malapi_test_macro_f1']:>10.4f} "
                  f"{g['malbehavd_macro_f1']:>11.4f} "
                  f"{g['generalization_gap']:>8.4f} "
                  f"{g['relative_drop_pct']:>7.1f}%")

    print(f"\nResults saved to: {cfg.GENERALIZABILITY_DIR}")


if __name__ == "__main__":
    main()
