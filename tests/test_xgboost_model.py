"""
Tests for src.xgboost_model — training, saving, loading, prediction.

Uses small synthetic datasets to keep tests fast.
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

import config as cfg
from src.model_training.xgboost_model import (
    load_model,
    predict_with_confidence,
    save_model,
    train_xgboost,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_data():
    """Create a small synthetic dataset: 100 samples, 3 classes, 20 features."""
    rng = np.random.RandomState(cfg.RANDOM_SEED)
    n_samples = 100
    n_features = 20
    n_classes = 3

    X = rng.randn(n_samples, n_features)
    y_str = rng.choice(["ClassA", "ClassB", "ClassC"], size=n_samples)

    le = LabelEncoder()
    y_int = le.fit_transform(y_str)

    return X, y_int, le, n_classes


@pytest.fixture(scope="module")
def trained_model(synthetic_data):
    """Train a small XGBoost model on synthetic data."""
    X, y, le, _ = synthetic_data
    # Use very small search for speed
    param_dist = {
        "n_estimators": [10, 20],
        "max_depth": [3, 4],
        "learning_rate": [0.1],
    }
    model, params = train_xgboost(
        X, y, le, param_dist=param_dist, n_iter=2, cv_folds=2
    )
    return model


# ── Training tests ─────────────────────────────────────────────────────────


class TestTraining:
    """XGBoost trains without errors on synthetic data."""

    def test_model_trains(self, trained_model):
        """Model object should be returned."""
        assert trained_model is not None

    def test_model_has_classes(self, trained_model, synthetic_data):
        """Model should know about all classes."""
        _, y, _, n_classes = synthetic_data
        assert len(trained_model.classes_) == n_classes


# ── Prediction tests ──────────────────────────────────────────────────────


class TestPrediction:
    """Predictions have correct shapes and valid probabilities."""

    def test_prediction_shapes(self, trained_model, synthetic_data):
        """Labels shape (n,), probabilities shape (n, n_classes)."""
        X, y, _, n_classes = synthetic_data
        preds, probs = predict_with_confidence(trained_model, X)
        assert preds.shape == (len(X),)
        assert probs.shape == (len(X), n_classes)

    def test_probabilities_sum_to_one(self, trained_model, synthetic_data):
        """Each row of probabilities should sum to ~1.0."""
        X, _, _, _ = synthetic_data
        _, probs = predict_with_confidence(trained_model, X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_probabilities_nonnegative(self, trained_model, synthetic_data):
        """All probabilities should be ≥ 0."""
        X, _, _, _ = synthetic_data
        _, probs = predict_with_confidence(trained_model, X)
        assert np.all(probs >= 0)

    def test_predictions_are_valid_classes(self, trained_model, synthetic_data):
        """All predicted labels should be valid class indices."""
        X, _, _, n_classes = synthetic_data
        preds, _ = predict_with_confidence(trained_model, X)
        assert np.all(preds >= 0)
        assert np.all(preds < n_classes)


# ── Save / load tests ─────────────────────────────────────────────────────


class TestSaveLoad:
    """Model serialization round-trip."""

    def test_save_load_identical_predictions(
        self, trained_model, synthetic_data, tmp_path
    ):
        """Loaded model should produce identical predictions."""
        X, _, _, _ = synthetic_data
        preds_before, probs_before = predict_with_confidence(trained_model, X)

        model_path = tmp_path / "xgb_model.pkl"
        save_model(trained_model, model_path)
        loaded = load_model(model_path)

        preds_after, probs_after = predict_with_confidence(loaded, X)
        np.testing.assert_array_equal(preds_before, preds_after)
        np.testing.assert_allclose(probs_before, probs_after, atol=1e-7)


# ── Class weight test ─────────────────────────────────────────────────────


class TestClassWeights:
    """Class weight computation produces expected values."""

    def test_balanced_weights(self):
        """Balanced weights: minority class gets higher weight."""
        from sklearn.utils.class_weight import compute_sample_weight

        y = np.array([0, 0, 0, 0, 1, 1, 2])  # 4:2:1 imbalance
        weights = compute_sample_weight("balanced", y)
        # Class 2 (1 sample) should get highest weight
        assert weights[6] > weights[4]  # class 2 > class 1
        assert weights[4] > weights[0]  # class 1 > class 0
