"""
Phase 6 — Model evaluation: metrics, confusion matrices, ROC curves, SHAP.

Generates publication-quality plots and comprehensive metrics for both the
XGBoost and LSTM classifiers.  All outputs are saved under ``results/``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import config as cfg
from src.utils import get_logger, save_json

logger = get_logger(__name__)

# Consistent style for all plots
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE_CM = (10, 8)
FIGSIZE_WIDE = (14, 6)
DPI = 150


# ── Metrics ───────────────────────────────────────────────────────────────


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute a comprehensive metrics dictionary for a single model.

    Args:
        y_true: Ground-truth integer labels, shape ``(n,)``.
        y_pred: Predicted integer labels, shape ``(n,)``.
        y_prob: Predicted probabilities, shape ``(n, num_classes)``.
        class_names: Ordered list of class name strings.

    Returns:
        Dict containing accuracy, macro/weighted F1, per-class metrics,
        confusion matrix, and multiclass ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    # Pass labels= so the per-class arrays always have one entry per
    # class_name even when some classes are absent from y_true (e.g.
    # cross-dataset evaluation where the eval set lacks some classes).
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        average=None, zero_division=0,
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    # Multiclass ROC-AUC (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    # label_binarize with 2 classes returns shape (n, 1); expand to (n, 2)
    if y_true_bin.ndim == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    try:
        roc_auc_macro = roc_auc_score(
            y_true_bin, y_prob, average="macro", multi_class="ovr",
        )
        roc_auc_weighted = roc_auc_score(
            y_true_bin, y_prob, average="weighted", multi_class="ovr",
        )
    except ValueError:
        roc_auc_macro = None
        roc_auc_weighted = None

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names))),
    ).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc_macro": round(roc_auc_macro, 4) if roc_auc_macro else None,
        "roc_auc_weighted": round(roc_auc_weighted, 4) if roc_auc_weighted else None,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


# ── Confusion matrix plot ─────────────────────────────────────────────────


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Ordered class name strings.
        title: Plot title.
        save_path: Path to save the PNG file.
        normalize: If True, show row-normalized percentages.
    """
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names))),
    )

    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.divide(
                cm.astype(float), row_sums,
                out=np.zeros_like(cm, dtype=float), where=row_sums != 0,
            )
        fmt = ".1%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=FIGSIZE_CM)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


# ── Per-class F1 bar chart ────────────────────────────────────────────────


def plot_per_class_f1(
    metrics: Dict[str, Any],
    title: str,
    save_path: Path,
) -> None:
    """Plot a horizontal bar chart of per-class F1 scores.

    Args:
        metrics: Output of :func:`compute_all_metrics`.
        title: Plot title.
        save_path: Path to save the PNG file.
    """
    names = list(metrics["per_class"].keys())
    f1_scores = [metrics["per_class"][n]["f1"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, f1_scores, color=sns.color_palette("muted", len(names)))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)

    # Annotate bars
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=9,
        )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Per-class F1 chart saved to %s", save_path)


# ── Model comparison bar chart ────────────────────────────────────────────


def plot_model_comparison(
    xgb_metrics: Dict[str, Any],
    lstm_metrics: Dict[str, Any],
    save_path: Path,
) -> None:
    """Side-by-side per-class F1 comparison between XGBoost and LSTM.

    Args:
        xgb_metrics: XGBoost metrics dict from :func:`compute_all_metrics`.
        lstm_metrics: LSTM metrics dict from :func:`compute_all_metrics`.
        save_path: Path to save the PNG file.
    """
    class_names = list(xgb_metrics["per_class"].keys())
    xgb_f1 = [xgb_metrics["per_class"][n]["f1"] for n in class_names]
    lstm_f1 = [lstm_metrics["per_class"][n]["f1"] for n in class_names]

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(x - width / 2, xgb_f1, width, label="XGBoost", color="#4C72B0")
    ax.bar(x + width / 2, lstm_f1, width, label="LSTM", color="#DD8452")

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1: XGBoost vs LSTM")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", save_path)


def plot_three_model_comparison(
    xgb_metrics: Dict[str, Any],
    lstm_metrics: Dict[str, Any],
    ens_metrics: Dict[str, Any],
    save_path: Path,
) -> None:
    """Per-class F1 comparison across XGBoost, LSTM, and Ensemble.

    Args:
        xgb_metrics: XGBoost metrics dict.
        lstm_metrics: LSTM metrics dict.
        ens_metrics: Ensemble metrics dict.
        save_path: Path to save the PNG file.
    """
    class_names = list(xgb_metrics["per_class"].keys())
    xgb_f1 = [xgb_metrics["per_class"][n]["f1"] for n in class_names]
    lstm_f1 = [lstm_metrics["per_class"][n]["f1"] for n in class_names]
    ens_f1 = [ens_metrics["per_class"][n]["f1"] for n in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(x - width, xgb_f1, width, label="XGBoost", color="#4C72B0")
    ax.bar(x, lstm_f1, width, label="LSTM", color="#DD8452")
    ax.bar(x + width, ens_f1, width, label="Ensemble", color="#55A868")

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1: XGBoost vs LSTM vs Ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Three-model comparison chart saved to %s", save_path)


# ── Multiclass ROC curves ────────────────────────────────────────────────


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
) -> None:
    """Plot one-vs-rest ROC curves for each class.

    Args:
        y_true: Ground-truth integer labels, shape ``(n,)``.
        y_prob: Predicted probabilities, shape ``(n, num_classes)``.
        class_names: Ordered class name strings.
        title: Plot title.
        save_path: Path to save the PNG file.
    """
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    # label_binarize with 2 classes returns shape (n, 1); expand to (n, 2)
    if y_true_bin.ndim == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    colors = sns.color_palette("tab10", len(class_names))

    fig, ax = plt.subplots(figsize=FIGSIZE_CM)

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_val = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved to %s", save_path)


# ── SHAP analysis (XGBoost only) ─────────────────────────────────────────


def run_shap_analysis(
    model: Any,
    X_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    save_dir: Path,
    max_display: int = 20,
) -> None:
    """Run SHAP TreeExplainer on the XGBoost model and save plots.

    Generates:
      - Global summary bar plot (mean |SHAP| across all classes)
      - Per-class summary beeswarm plots

    Args:
        model: Trained XGBoost model.
        X_test: Test feature matrix, shape ``(n_samples, n_features)``.
        feature_names: Feature name strings matching columns of X_test.
        class_names: Ordered class name strings.
        save_dir: Directory to save SHAP plots.
        max_display: Number of top features to display.
    """
    import shap

    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)

    # Normalize SHAP output to list-of-arrays: [array(n, features), ...] per class.
    if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        # Newer SHAP: shape (n_samples, n_features, n_classes)
        shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2:
        # Binary classifier: single (n_samples, n_features) array.
        # Treat as class-1 SHAP; class-0 is the negation.
        shap_values = [-shap_values_raw, shap_values_raw]
    elif isinstance(shap_values_raw, list):
        shap_values = shap_values_raw
    else:
        shap_values = [shap_values_raw]

    # Global bar plot: mean absolute SHAP across all classes
    logger.info("Generating global SHAP summary bar plot...")
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title("Global Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_global_bar.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")
    logger.info("Global SHAP bar plot saved.")

    # Per-class beeswarm plots
    for i, name in enumerate(class_names):
        logger.info("Generating SHAP beeswarm for class: %s", name)
        shap.summary_plot(
            shap_values[i],
            X_test,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title(f"SHAP Values — {name}")
        plt.tight_layout()
        plt.savefig(
            save_dir / f"shap_beeswarm_{name.lower()}.png",
            dpi=DPI, bbox_inches="tight",
        )
        plt.close("all")

    logger.info("All SHAP plots saved to %s", save_dir)


# ── LSTM training history plot ────────────────────────────────────────────


def plot_training_history(
    history_path: Path,
    save_path: Path,
) -> None:
    """Plot LSTM training/validation loss and accuracy curves.

    Args:
        history_path: Path to pickled Keras History.history dict.
        save_path: Path to save the PNG file.
    """
    from src.utils import load_pickle
    history = load_pickle(history_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("LSTM Training Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, history["accuracy"], "b-", label="Train")
    ax2.plot(epochs, history["val_accuracy"], "r-", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("LSTM Training Accuracy")
    ax2.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training history plot saved to %s", save_path)


# ── Class distribution plot ───────────────────────────────────────────────


def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> None:
    """Plot class distribution for train and test sets.

    Args:
        y_train: Integer-encoded training labels.
        y_test: Integer-encoded test labels.
        class_names: Ordered class name strings.
        save_path: Path to save the PNG file.
    """
    train_counts = np.bincount(y_train, minlength=len(class_names))
    test_counts = np.bincount(y_test, minlength=len(class_names))

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(x - width / 2, train_counts, width, label="Train", color="#4C72B0")
    ax.bar(x + width / 2, test_counts, width, label="Test", color="#DD8452")

    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution: Train vs Test")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.legend()

    # Annotate
    for i, (tr, te) in enumerate(zip(train_counts, test_counts)):
        ax.text(i - width / 2, tr + 5, str(tr), ha="center", fontsize=8)
        ax.text(i + width / 2, te + 5, str(te), ha="center", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Class distribution plot saved to %s", save_path)
