"""
Confidence-gated, per-class F1-weighted ensemble of XGBoost and LSTM.

When XGBoost is confident (max probability >= threshold), its prediction
is used directly.  When XGBoost is uncertain, the ensemble blends both
models' probabilities using per-class F1-derived weights.

This exploits the observation that XGBoost outperforms the LSTM on most
samples, but the LSTM can rescue a few uncertain predictions where
XGBoost's feature engineering misses sequential patterns.

The class exposes the same ``predict_with_confidence`` interface as the
base models, and can process raw API-call sequences end-to-end.

Run directly to build the ensemble from trained base models::

    python -m src.model_training.ensemble_model
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config as cfg
from src.utils import get_logger, load_json, load_pickle

logger = get_logger(__name__)


class EnsembleClassifier:
    """Confidence-gated, F1-weighted ensemble of XGBoost + LSTM.

    When XGBoost's maximum predicted probability is at or above
    ``confidence_threshold``, the ensemble trusts XGBoost alone.
    Otherwise it falls back to a per-class F1-weighted blend of
    both models' probability outputs.

    Args:
        xgb_model: Trained XGBoost classifier.
        lstm_model: Trained Keras LSTM model.
        tfidf_vectorizer: Fitted TF-IDF vectorizer (for XGBoost features).
        label_encoder: Fitted sklearn LabelEncoder.
        vocab: Token-to-index vocabulary dict.
        xgb_class_f1: Per-class F1 scores for XGBoost, keyed by class name.
        lstm_class_f1: Per-class F1 scores for LSTM, keyed by class name.
        seq_len: LSTM input sequence length.
        confidence_threshold: XGBoost max-probability gate.  Samples where
            XGBoost's highest class probability >= this value use XGBoost
            alone; lower-confidence samples use the F1-weighted blend.
    """

    def __init__(
        self,
        xgb_model: Any,
        lstm_model: Any,
        tfidf_vectorizer: Any,
        label_encoder: Any,
        vocab: Dict[str, int],
        xgb_class_f1: Dict[str, float],
        lstm_class_f1: Dict[str, float],
        seq_len: int = cfg.LSTM_BEST_SEQ_LEN,
        confidence_threshold: float = 0.30,
    ) -> None:
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.label_encoder = label_encoder
        self.vocab = vocab
        self.seq_len = seq_len
        self.confidence_threshold = confidence_threshold

        # Compute per-class weights from F1 scores
        class_names = list(label_encoder.classes_)
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.xgb_weights = np.zeros(self.num_classes)
        self.lstm_weights = np.zeros(self.num_classes)

        for i, name in enumerate(class_names):
            xgb_f1 = xgb_class_f1[name]
            lstm_f1 = lstm_class_f1[name]
            total = xgb_f1 + lstm_f1
            if total > 0:
                self.xgb_weights[i] = xgb_f1 / total
                self.lstm_weights[i] = lstm_f1 / total
            else:
                # Fallback to equal weights if both are zero
                self.xgb_weights[i] = 0.5
                self.lstm_weights[i] = 0.5

        logger.info(
            "Confidence gate threshold: %.2f (XGBoost-only when max_prob >= %.2f)",
            self.confidence_threshold, self.confidence_threshold,
        )
        logger.info("Ensemble weights (XGBoost / LSTM) per class:")
        for i, name in enumerate(class_names):
            logger.info(
                "  %12s: XGB=%.3f  LSTM=%.3f",
                name, self.xgb_weights[i], self.lstm_weights[i],
            )

    def _build_xgb_features(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """Build the v1 XGBoost feature matrix from raw samples.

        Args:
            samples: List of dicts with ``sequence`` key (list of API call strings).

        Returns:
            Dense feature matrix of shape ``(n_samples, n_features)``.
        """
        from src.model_training.feature_engineering import (
            compute_category_features,
            compute_statistical_features,
            tfidf_transform,
        )

        tfidf = tfidf_transform(samples, self.tfidf_vectorizer)
        stats = compute_statistical_features(samples)
        cats = compute_category_features(samples)
        return np.hstack([tfidf, stats, cats])

    def _build_lstm_sequences(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """Encode and pad sequences for the LSTM.

        If samples already have an ``encoded`` field (from preprocessing),
        uses it directly.  Otherwise encodes from the raw ``sequence`` field.

        Args:
            samples: List of dicts with ``sequence`` key (list of API call strings).

        Returns:
            Padded integer array of shape ``(n_samples, seq_len)``.
        """
        from src.data_loading.preprocessing import encode_sequence, pad_sequences

        if samples and "encoded" in samples[0]:
            encoded = [s["encoded"] for s in samples]
        else:
            encoded = [encode_sequence(s["sequence"], self.vocab) for s in samples]
        return pad_sequences(encoded, max_len=self.seq_len)

    def predict_with_confidence(
        self,
        samples: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ensemble predictions from raw API-call sequences.

        Args:
            samples: List of dicts with ``sequence`` key.

        Returns:
            Tuple of (predicted_labels, probabilities) where labels has shape
            ``(n,)`` and probabilities has shape ``(n, num_classes)``.
        """
        from src.model_training.lstm_model import predict_with_confidence as lstm_predict
        from src.model_training.xgboost_model import predict_with_confidence as xgb_predict

        # Get probability outputs from both models
        X_xgb = self._build_xgb_features(samples)
        _, xgb_probs = xgb_predict(self.xgb_model, X_xgb)

        X_lstm = self._build_lstm_sequences(samples)
        _, lstm_probs = lstm_predict(self.lstm_model, X_lstm)

        # Apply confidence-gated blending
        ensemble_probs = self._apply_confidence_gate(xgb_probs, lstm_probs)
        predictions = np.argmax(ensemble_probs, axis=1)
        return predictions, ensemble_probs

    def _apply_confidence_gate(
        self,
        xgb_probs: np.ndarray,
        lstm_probs: np.ndarray,
    ) -> np.ndarray:
        """Apply confidence-gated blending to probability arrays.

        For each sample, if XGBoost's maximum class probability is at or
        above ``self.confidence_threshold``, the output row is XGBoost's
        probabilities alone.  Otherwise, the row is the per-class
        F1-weighted blend of both models' probabilities (re-normalized).

        Args:
            xgb_probs: XGBoost probabilities, shape ``(n, num_classes)``.
            lstm_probs: LSTM probabilities, shape ``(n, num_classes)``.

        Returns:
            Blended probability array, shape ``(n, num_classes)``.
        """
        n = xgb_probs.shape[0]
        xgb_max_prob = xgb_probs.max(axis=1)  # (n,)

        # Boolean mask: True = XGBoost is uncertain, use blend
        uncertain = xgb_max_prob < self.confidence_threshold  # (n,)
        n_blended = uncertain.sum()

        logger.info(
            "Confidence gate: %d / %d samples (%.1f%%) fall below threshold %.2f "
            "and will be blended; %d use XGBoost alone.",
            n_blended, n, 100.0 * n_blended / n,
            self.confidence_threshold, n - n_blended,
        )

        # Start with XGBoost probabilities for all samples
        ensemble_probs = xgb_probs.copy()

        # For uncertain samples, apply F1-weighted blend
        if n_blended > 0:
            blended = (
                xgb_probs[uncertain] * self.xgb_weights[np.newaxis, :]
                + lstm_probs[uncertain] * self.lstm_weights[np.newaxis, :]
            )
            row_sums = blended.sum(axis=1, keepdims=True)
            blended = blended / row_sums
            ensemble_probs[uncertain] = blended

        return ensemble_probs

    def predict_from_precomputed(
        self,
        xgb_probs: np.ndarray,
        lstm_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine pre-computed probability outputs from both models.

        Uses confidence gating: trusts XGBoost alone when its max
        probability >= threshold, blends with LSTM otherwise.

        Args:
            xgb_probs: XGBoost probabilities, shape ``(n, num_classes)``.
            lstm_probs: LSTM probabilities, shape ``(n, num_classes)``.

        Returns:
            Tuple of (predicted_labels, ensemble_probabilities).
        """
        ensemble_probs = self._apply_confidence_gate(xgb_probs, lstm_probs)
        predictions = np.argmax(ensemble_probs, axis=1)
        return predictions, ensemble_probs


def save_model(ensemble: EnsembleClassifier, path: Path) -> None:
    """Save the ensemble classifier to disk.

    Saves the ensemble metadata (weights, vocab, seq_len, vectorizer,
    label encoder) and paths to the base models.  The base model files
    must remain at their original locations.

    Args:
        ensemble: Trained EnsembleClassifier instance.
        path: Destination pickle file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ensemble, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Ensemble model saved to %s.", path)


def load_model(path: Path) -> EnsembleClassifier:
    """Load a saved EnsembleClassifier from disk.

    Args:
        path: Path to the pickled ensemble file.

    Returns:
        Loaded EnsembleClassifier.
    """
    with open(path, "rb") as f:
        ensemble = pickle.load(f)
    logger.info("Ensemble model loaded from %s.", path)
    return ensemble


# ── End-to-end build pipeline ────────────────────────────────────────────


def main() -> None:
    """Build the per-class F1-weighted ensemble from trained base models."""
    from src.model_training.lstm_model import load_model as load_lstm_model
    from src.model_training.xgboost_model import load_model as load_xgb_model

    # ── Load no-Trojan base models ──────────────────────────────────────
    logger.info("Loading no-Trojan base models (7-class)...")
    xgb_model = load_xgb_model(cfg.NO_TROJAN_XGBOOST_MODEL_DIR / "best_model.pkl")
    lstm_model = load_lstm_model(
        cfg.NO_TROJAN_LSTM_MODEL_DIR / f"best_len{cfg.LSTM_BEST_SEQ_LEN}.keras",
    )

    # ── Load preprocessors ────────────────────────────────────────────────
    tfidf_vectorizer = load_pickle(cfg.NO_TROJAN_CACHE_DIR / "tfidf_vectorizer.pkl")
    label_encoder = load_pickle(cfg.NO_TROJAN_LABEL_ENCODER_PATH)
    vocab = load_json(cfg.NO_TROJAN_VOCABULARY_PATH)

    # ── Load per-class F1 from evaluation metrics ─────────────────────────
    xgb_eval = load_json(cfg.NO_TROJAN_METRICS_DIR / "xgboost_evaluation.json")
    lstm_eval = load_json(cfg.NO_TROJAN_METRICS_DIR / "lstm_evaluation.json")

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
    save_path = cfg.NO_TROJAN_ENSEMBLE_MODEL_DIR / "ensemble_model.pkl"
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
