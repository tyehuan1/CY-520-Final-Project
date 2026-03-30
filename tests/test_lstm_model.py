"""
Tests for src.lstm_model — architecture, training, saving, loading, prediction.

Uses small synthetic datasets to keep tests fast.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pytest
import tensorflow as tf

import config as cfg
from src.lstm_model import (
    build_lstm_model,
    compute_class_weights,
    load_model,
    predict_with_confidence,
    save_model,
    train_lstm,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_data():
    """Create synthetic sequence data: 100 samples, 3 classes, vocab=50."""
    rng = np.random.RandomState(cfg.RANDOM_SEED)
    n_samples = 100
    vocab_size = 50
    max_seq_len = 30
    n_classes = 3

    X = rng.randint(0, vocab_size, size=(n_samples, max_seq_len)).astype(np.int32)
    y_int = rng.choice(n_classes, size=n_samples)
    y_onehot = tf.keras.utils.to_categorical(y_int, num_classes=n_classes)

    return X, y_int, y_onehot, vocab_size, max_seq_len, n_classes


@pytest.fixture(scope="module")
def trained_model_and_history(synthetic_data, tmp_path_factory):
    """Build and train a small LSTM on synthetic data."""
    X, y_int, y_onehot, vocab_size, max_seq_len, n_classes = synthetic_data
    tmp_path = tmp_path_factory.mktemp("lstm")

    model = build_lstm_model(vocab_size, max_seq_len, n_classes)
    history = train_lstm(
        model,
        X,
        y_onehot,
        y_int,
        model_path=tmp_path / "best_model.keras",
        batch_size=32,
        max_epochs=3,
        validation_split=0.2,
    )
    return model, history


# ── Architecture tests ─────────────────────────────────────────────────────


class TestArchitecture:
    """LSTM model builds with correct structure."""

    def test_model_builds(self, synthetic_data):
        """Model should build without errors."""
        _, _, _, vocab_size, max_seq_len, n_classes = synthetic_data
        model = build_lstm_model(vocab_size, max_seq_len, n_classes)
        assert model is not None

    def test_output_shape(self, synthetic_data):
        """Model output should have shape (batch, n_classes)."""
        X, _, _, vocab_size, max_seq_len, n_classes = synthetic_data
        model = build_lstm_model(vocab_size, max_seq_len, n_classes)
        output = model.predict(X[:5], verbose=0)
        assert output.shape == (5, n_classes)


# ── Training tests ─────────────────────────────────────────────────────────


class TestTraining:
    """LSTM trains without errors on synthetic data."""

    def test_model_trains(self, trained_model_and_history):
        """Training should complete and return a history."""
        model, history = trained_model_and_history
        assert model is not None
        assert history is not None

    def test_history_has_expected_keys(self, trained_model_and_history):
        """History should contain loss, accuracy, val_loss, val_accuracy."""
        _, history = trained_model_and_history
        for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
            assert key in history.history, f"Missing key: {key}"


# ── Prediction tests ──────────────────────────────────────────────────────


class TestPrediction:
    """Predictions have correct shapes and valid probabilities."""

    def test_prediction_shapes(self, trained_model_and_history, synthetic_data):
        """Labels shape (n,), probabilities shape (n, n_classes)."""
        model, _ = trained_model_and_history
        X, _, _, _, _, n_classes = synthetic_data
        preds, probs = predict_with_confidence(model, X)
        assert preds.shape == (len(X),)
        assert probs.shape == (len(X), n_classes)

    def test_probabilities_sum_to_one(self, trained_model_and_history, synthetic_data):
        """Each row of probabilities should sum to ~1.0."""
        model, _ = trained_model_and_history
        X, _, _, _, _, _ = synthetic_data
        _, probs = predict_with_confidence(model, X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_probabilities_nonnegative(self, trained_model_and_history, synthetic_data):
        """All probabilities should be ≥ 0."""
        model, _ = trained_model_and_history
        X, _, _, _, _, _ = synthetic_data
        _, probs = predict_with_confidence(model, X)
        assert np.all(probs >= 0)

    def test_predictions_are_valid_classes(
        self, trained_model_and_history, synthetic_data
    ):
        """All predicted labels should be valid class indices."""
        model, _ = trained_model_and_history
        X, _, _, _, _, n_classes = synthetic_data
        preds, _ = predict_with_confidence(model, X)
        assert np.all(preds >= 0)
        assert np.all(preds < n_classes)


# ── Save / load tests ─────────────────────────────────────────────────────


class TestSaveLoad:
    """Model serialization round-trip."""

    def test_save_load_identical_predictions(
        self, trained_model_and_history, synthetic_data, tmp_path
    ):
        """Loaded model should produce identical predictions."""
        model, _ = trained_model_and_history
        X, _, _, _, _, _ = synthetic_data

        preds_before, probs_before = predict_with_confidence(model, X)

        model_path = tmp_path / "lstm_test_model.keras"
        save_model(model, model_path)
        loaded = load_model(model_path)

        preds_after, probs_after = predict_with_confidence(loaded, X)
        np.testing.assert_array_equal(preds_before, preds_after)
        np.testing.assert_allclose(probs_before, probs_after, atol=1e-5)


# ── Class weight test ─────────────────────────────────────────────────────


class TestClassWeights:
    """Class weight computation produces expected values."""

    def test_balanced_weights(self):
        """Minority class should get higher weight."""
        y = np.array([0, 0, 0, 0, 1, 1, 2])
        weights = compute_class_weights(y)
        assert weights[2] > weights[1]
        assert weights[1] > weights[0]

    def test_equal_classes(self):
        """Equal-sized classes should get equal weight (~1.0)."""
        y = np.array([0, 0, 1, 1, 2, 2])
        weights = compute_class_weights(y)
        for w in weights.values():
            assert abs(w - 1.0) < 0.01
