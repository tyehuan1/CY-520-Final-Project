"""
Gradio web app for malware family classification.

Accepts a raw API call sequence (space-separated text or .txt file upload),
runs all three V2 models (XGBoost, LSTM, Ensemble), and displays:
  - Side-by-side predicted families and confidence scores
  - Per-family confidence bar chart
  - SHAP waterfall plot for the XGBoost prediction
  - "Further Review Needed" flag when all models are uncertain

Usage::

    python -m src.deployment.app
"""

import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from src.data_loading.preprocessing import clean_sequence, encode_sequence, pad_sequences
from src.model_training.feature_engineering import (
    build_feature_matrix,
    get_feature_names,
)
from src.model_training.lstm_model import (
    load_model as load_lstm_model,
    predict_with_confidence as lstm_predict,
)
from src.model_training.xgboost_model import (
    load_model as load_xgb_model,
    predict_with_confidence as xgb_predict,
)
from src.utils import get_logger, load_json, load_pickle

logger = get_logger(__name__)

# When ALL three models have max confidence below this value,
# the prediction is flagged as needing further review.
# Derived from analysis: at 0.40, accuracy on flagged samples is ~24%
# (effectively random for 8 classes), affecting ~10% of test samples.
REVIEW_THRESHOLD = 0.40


# ── Global model state (loaded once at startup) ─────────────────────────────

_models: Dict = {}


def _load_models() -> None:
    """Load all V2 model artifacts into the global _models dict."""
    if _models:
        return  # already loaded

    logger.info("Loading V2 model artifacts for Gradio app...")

    _models["xgb"] = load_xgb_model(cfg.V2_XGBOOST_MODEL_DIR / "best_model.pkl")
    _models["lstm"] = load_lstm_model(
        cfg.V2_LSTM_MODEL_DIR / f"best_len{cfg.LSTM_V2_BEST_SEQ_LEN}.keras",
    )
    _models["tfidf"] = load_pickle(cfg.V2_TFIDF_PATH)
    _models["label_encoder"] = load_pickle(cfg.V2_LABEL_ENCODER_PATH)
    _models["vocab"] = load_json(cfg.VOCABULARY_PATH)
    _models["class_names"] = list(_models["label_encoder"].classes_)
    _models["feature_names"] = get_feature_names(_models["tfidf"])

    # Load ensemble
    from src.model_training.ensemble_model import (
        EnsembleClassifier,
        load_model as load_ensemble,
    )
    import __main__
    __main__.EnsembleClassifier = EnsembleClassifier

    ens_path = cfg.V2_ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
    if ens_path.exists():
        _models["ensemble"] = load_ensemble(ens_path)
    else:
        _models["ensemble"] = None
        logger.warning("Ensemble model not found at %s", ens_path)

    # Pre-build SHAP explainer (takes ~1s, avoids delay on first request)
    _models["shap_explainer"] = shap.TreeExplainer(_models["xgb"])

    logger.info(
        "All models loaded. Classes: %s", _models["class_names"],
    )


# ── Core prediction logic ───────────────────────────────────────────────────


def _parse_sequence(text: str) -> List[str]:
    """Parse space-separated or newline-separated API call sequence.

    Args:
        text: Raw input text containing API call names.

    Returns:
        List of API call name strings.
    """
    tokens = text.strip().replace("\n", " ").replace(",", " ").split()
    return [t.strip() for t in tokens if t.strip()]


def _predict(sequence: List[str]) -> Dict:
    """Run all three models on a single API call sequence.

    Args:
        sequence: List of API call name strings.

    Returns:
        Dict with predictions, probabilities, and SHAP values.
    """
    _load_models()

    class_names = _models["class_names"]
    vocab = _models["vocab"]
    tfidf = _models["tfidf"]
    le = _models["label_encoder"]

    # Match preprocess_external_samples: lowercase + clean so CamelCase input
    # (e.g. NtClose) resolves to vocab tokens instead of <UNK>.
    sequence = clean_sequence([tok.lower() for tok in sequence])

    # Build sample dict matching the expected format
    sample = {"sequence": sequence}

    # ── XGBoost ──
    X_xgb = build_feature_matrix([sample], tfidf, log_dampen=True)
    xgb_preds, xgb_probs = xgb_predict(_models["xgb"], X_xgb)
    xgb_pred_label = class_names[xgb_preds[0]]
    xgb_confidence = float(xgb_probs[0].max())

    # ── LSTM ──
    encoded = encode_sequence(sequence, vocab)
    X_lstm = pad_sequences([encoded], max_len=cfg.LSTM_V2_BEST_SEQ_LEN)
    lstm_preds, lstm_probs = lstm_predict(_models["lstm"], X_lstm)
    lstm_pred_label = class_names[lstm_preds[0]]
    lstm_confidence = float(lstm_probs[0].max())

    # ── Ensemble ──
    ens_pred_label = "N/A"
    ens_confidence = 0.0
    ens_probs_arr = xgb_probs  # fallback
    if _models["ensemble"] is not None:
        ens_probs_arr = _models["ensemble"]._apply_confidence_gate(
            xgb_probs, lstm_probs,
        )
        ens_pred_idx = int(np.argmax(ens_probs_arr[0]))
        ens_pred_label = class_names[ens_pred_idx]
        ens_confidence = float(ens_probs_arr[0].max())

    # ── SHAP waterfall ──
    shap_values = _models["shap_explainer"].shap_values(X_xgb)
    # shap_values is list of arrays per class, or 3D array
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_for_pred = shap_values[0, :, xgb_preds[0]]
    elif isinstance(shap_values, list):
        shap_for_pred = shap_values[xgb_preds[0]][0]
    else:
        shap_for_pred = shap_values[0]

    # ── Further review flag ──
    needs_review = (
        xgb_confidence < REVIEW_THRESHOLD
        and lstm_confidence < REVIEW_THRESHOLD
        and ens_confidence < REVIEW_THRESHOLD
    )

    return {
        "xgb": {
            "label": xgb_pred_label,
            "confidence": xgb_confidence,
            "probs": {n: float(xgb_probs[0][i]) for i, n in enumerate(class_names)},
        },
        "lstm": {
            "label": lstm_pred_label,
            "confidence": lstm_confidence,
            "probs": {n: float(lstm_probs[0][i]) for i, n in enumerate(class_names)},
        },
        "ensemble": {
            "label": ens_pred_label,
            "confidence": ens_confidence,
            "probs": {n: float(ens_probs_arr[0][i]) for i, n in enumerate(class_names)},
        },
        "needs_review": needs_review,
        "shap_values": shap_for_pred,
        "shap_base_value": float(
            _models["shap_explainer"].expected_value[xgb_preds[0]]
            if isinstance(_models["shap_explainer"].expected_value, (list, np.ndarray))
            else _models["shap_explainer"].expected_value
        ),
        "features": X_xgb[0],
        "xgb_pred_idx": int(xgb_preds[0]),
        "sequence_length": len(sequence),
    }


# ── Plotting helpers ────────────────────────────────────────────────────────


def _plot_confidence_bars(result: Dict) -> plt.Figure:
    """Create a grouped bar chart of per-family confidences for all models.

    Args:
        result: Output dict from _predict.

    Returns:
        Matplotlib Figure.
    """
    class_names = _models["class_names"]
    n_classes = len(class_names)

    xgb_vals = [result["xgb"]["probs"][n] for n in class_names]
    lstm_vals = [result["lstm"]["probs"][n] for n in class_names]
    ens_vals = [result["ensemble"]["probs"][n] for n in class_names]

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, xgb_vals, width, label="XGBoost", color="#4C72B0")
    ax.bar(x, lstm_vals, width, label="LSTM", color="#DD8452")
    ax.bar(x + width, ens_vals, width, label="Ensemble", color="#55A868")

    ax.set_ylabel("Confidence")
    ax.set_title("Per-Family Confidence Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.axhline(y=REVIEW_THRESHOLD, color="red", linestyle="--", alpha=0.5,
               label=f"Review threshold ({REVIEW_THRESHOLD})")
    plt.tight_layout()
    return fig


def _shorten_feature_name(name: str, max_len: int = 35) -> str:
    """Shorten long feature names for display.

    TF-IDF n-gram features like 'ldrgetprocedureaddress ntallocatevirtualmemory'
    are truncated. Bigram features like 'bigram_filesystem_registry' are
    reformatted as 'bi:filesystem>registry'.

    Args:
        name: Raw feature name.
        max_len: Maximum display length.

    Returns:
        Shortened feature name string.
    """
    if name.startswith("bigram_"):
        # "bigram_filesystem_registry" -> "bi:filesystem>registry"
        parts = name.replace("bigram_", "").split("_", 1)
        if len(parts) == 2:
            return f"bi:{parts[0]}>{parts[1]}"
        return name
    if name.startswith("cat_"):
        return name  # already short
    # For TF-IDF n-grams, truncate long names
    if len(name) > max_len:
        return name[:max_len - 2] + ".."
    return name


def _plot_shap_waterfall(result: Dict) -> plt.Figure:
    """Create a SHAP waterfall plot for the XGBoost prediction.

    Uses a horizontal bar chart instead of the built-in SHAP waterfall to
    avoid layout issues with long n-gram feature names.

    Args:
        result: Output dict from _predict.

    Returns:
        Matplotlib Figure.
    """
    class_names = _models["class_names"]
    feature_names = _models["feature_names"]
    pred_class = class_names[result["xgb_pred_idx"]]

    shap_vals = result["shap_values"]
    features = result["features"]
    max_display = 15

    # Get top features by absolute SHAP value
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:max_display]
    # Reverse so largest is at top of horizontal bar chart
    top_idx = top_idx[::-1]

    names = [_shorten_feature_name(feature_names[i]) for i in top_idx]
    values = [shap_vals[i] for i in top_idx]
    feat_vals = [features[i] for i in top_idx]
    colors = ["#ff0051" if v > 0 else "#008bfb" for v in values]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(names)), values, color=colors, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(
        f"SHAP Feature Impact — Predicted: {pred_class} "
        f"({result['xgb']['confidence']:.1%})",
        fontsize=12,
    )

    # Add feature value annotations left-justified inside bars,
    # extending past the bar edge if the bar is too short.
    for i, (bar, fv) in enumerate(zip(bars, feat_vals)):
        w = bar.get_width()
        # Place text just inside the bar's starting edge
        if w >= 0:
            x_pos = 0.002
        else:
            x_pos = -0.002
        ha = "left" if w >= 0 else "right"
        ax.text(x_pos, i, f" val={fv:.4f}", va="center", ha=ha, fontsize=7,
                color="white", fontweight="bold",
                bbox=dict(facecolor="none", edgecolor="none", pad=0))

    # Add vertical line at 0
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff0051", label="Pushes toward prediction"),
        Patch(facecolor="#008bfb", label="Pushes away from prediction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    return fig


# ── Gradio interface ────────────────────────────────────────────────────────


def classify(text_input: str, file_input) -> Tuple:
    """Main Gradio callback: classify an API call sequence.

    Args:
        text_input: Space-separated API call sequence from text box.
        file_input: Uploaded .txt file with API calls.

    Returns:
        Tuple of (summary_text, confidence_chart, shap_chart).
    """
    # Determine input source
    if file_input is not None:
        # Gradio file upload returns a file path string
        with open(file_input, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    elif text_input and text_input.strip():
        raw_text = text_input
    else:
        return (
            "Please enter an API call sequence or upload a .txt file.",
            None,
            None,
        )

    sequence = _parse_sequence(raw_text)
    if len(sequence) < 3:
        return (
            f"Sequence too short ({len(sequence)} calls). "
            "Please provide at least 3 API calls.",
            None,
            None,
        )

    # Run prediction
    result = _predict(sequence)

    # Build summary text
    lines = []
    lines.append(f"Sequence length: {result['sequence_length']} API calls")
    lines.append("")

    header = f"{'Model':<12} {'Prediction':<14} {'Confidence':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for model_key, model_name in [
        ("xgb", "XGBoost"),
        ("lstm", "LSTM"),
        ("ensemble", "Ensemble"),
    ]:
        m = result[model_key]
        lines.append(
            f"{model_name:<12} {m['label']:<14} {m['confidence']:>10.1%}"
        )

    if result["needs_review"]:
        lines.append("")
        lines.append("⚠ FURTHER REVIEW NEEDED")
        lines.append(
            f"All models have max confidence below {REVIEW_THRESHOLD:.0%}. "
            "This sample may not belong to any trained family, or the "
            "API call pattern is ambiguous."
        )

    summary = "\n".join(lines)

    # Generate plots
    confidence_fig = _plot_confidence_bars(result)
    shap_fig = _plot_shap_waterfall(result)

    return summary, confidence_fig, shap_fig


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app.

    Returns:
        Configured Gradio Blocks instance.
    """
    # Pre-load models at startup
    _load_models()

    with gr.Blocks(
        title="Malware Family Classifier",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Malware Family Classifier\n"
            "Classify Windows API call sequences into malware families using "
            "three models: **XGBoost**, **BiLSTM**, and a "
            "**Confidence-Gated Ensemble**.\n\n"
            "Enter a space-separated API call sequence or upload a `.txt` file."
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="API Call Sequence",
                    placeholder=(
                        "e.g.: NtCreateFile NtWriteFile RegSetValueExW "
                        "CreateThread NtClose ..."
                    ),
                    lines=5,
                )
                file_input = gr.File(
                    label="Or upload a .txt file",
                    file_types=[".txt"],
                )
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                summary_output = gr.Textbox(
                    label="Classification Results",
                    lines=12,
                    interactive=False,
                )

        with gr.Row():
            confidence_plot = gr.Plot(label="Per-Family Confidence")
            shap_plot = gr.Plot(label="SHAP Waterfall (XGBoost)")

        submit_btn.click(
            fn=classify,
            inputs=[text_input, file_input],
            outputs=[summary_output, confidence_plot, shap_plot],
        )

        gr.Markdown(
            "---\n"
            f"**Review threshold:** {REVIEW_THRESHOLD:.0%} — when all three "
            "models have max confidence below this level, the prediction is "
            "flagged for further review.\n\n"
            "**Families:** " + ", ".join(_models.get("class_names", [])) + "\n\n"
            "Built for CY-520 — Malware Family Classification Project."
        )

    return app


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """Launch the Gradio app."""
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
