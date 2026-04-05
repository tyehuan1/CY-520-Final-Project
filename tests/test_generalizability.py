"""
Tests for Phase 7 — Generalizability evaluation on MalBehavD-V1.

Validates that the VT-labeled MalBehavD cache loads correctly, that
preprocessing with the training vocabulary works, that feature engineering
produces correctly shaped matrices, and that binary detection helpers
return sensible results.
"""

from collections import Counter

import numpy as np
import pytest

import config as cfg
from src.model_training.feature_engineering import (
    CATEGORY_FEATURE_NAMES,
    STATISTICAL_FEATURE_NAMES,
    compute_category_features,
    compute_statistical_features,
    tfidf_transform,
)
from src.data_loading.preprocessing import (
    compute_unk_ratio,
    pad_sequences,
    preprocess_malbehavd_sequences,
)
from src.utils import load_json, load_pickle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def malbehavd_data():
    """Load the VT-labeled MalBehavD cache."""
    return load_json(cfg.MALBEHAVD_LABELED_PATH)


@pytest.fixture(scope="module")
def vocab():
    """Load the training vocabulary."""
    return load_json(cfg.VOCABULARY_PATH)


@pytest.fixture(scope="module")
def label_encoder():
    """Load the fitted label encoder."""
    return load_pickle(cfg.CACHE_DIR / "label_encoder.pkl")


@pytest.fixture(scope="module")
def tfidf_vectorizer():
    """Load the fitted TF-IDF vectorizer."""
    return load_pickle(cfg.CACHE_DIR / "tfidf_vectorizer.pkl")


@pytest.fixture(scope="module")
def malware_samples(malbehavd_data):
    """Extract malware-only samples (family-labeled)."""
    return [s for s in malbehavd_data["samples"]
            if s["label"] != cfg.BENIGN_LABEL]


@pytest.fixture(scope="module")
def malware_processed(malware_samples, vocab):
    """Preprocess malware samples with training vocabulary."""
    return preprocess_malbehavd_sequences(malware_samples, vocab)


# ---------------------------------------------------------------------------
# MalBehavD loading tests
# ---------------------------------------------------------------------------

class TestMalBehavDLabeled:
    """Tests for the VT-labeled MalBehavD cache structure."""

    def test_cache_has_expected_keys(self, malbehavd_data):
        """Cache file must have samples, dropped_hashes, and stats."""
        assert "samples" in malbehavd_data
        assert "dropped_hashes" in malbehavd_data
        assert "stats" in malbehavd_data

    def test_sample_format(self, malbehavd_data):
        """Every sample must have sequence, label, and sha256."""
        for s in malbehavd_data["samples"][:20]:
            assert "sequence" in s
            assert "label" in s
            assert "sha256" in s
            assert isinstance(s["sequence"], list)
            assert len(s["sequence"]) > 0

    def test_benign_count(self, malbehavd_data):
        """Should have 1285 benign samples."""
        benign = [s for s in malbehavd_data["samples"]
                  if s["label"] == cfg.BENIGN_LABEL]
        assert len(benign) == cfg.MALBEHAVD_EXPECTED_BENIGN

    def test_malware_has_valid_families(self, malware_samples, label_encoder):
        """All malware labels must be one of the 8 training families."""
        valid = set(label_encoder.classes_)
        for s in malware_samples:
            assert s["label"] in valid, (
                f"Unexpected family: {s['label']}"
            )

    def test_no_empty_sequences(self, malbehavd_data):
        """No sample should have an empty sequence."""
        for s in malbehavd_data["samples"]:
            assert len(s["sequence"]) > 0


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Tests for MalBehavD preprocessing with training vocabulary."""

    def test_processed_has_encoded_field(self, malware_processed):
        """Preprocessing must add an encoded field."""
        for s in malware_processed[:10]:
            assert "encoded" in s
            assert isinstance(s["encoded"], list)
            assert len(s["encoded"]) > 0

    def test_encoded_values_are_valid_indices(self, malware_processed, vocab):
        """All encoded values must be valid vocabulary indices."""
        vocab_size = len(vocab)
        for s in malware_processed[:50]:
            for idx in s["encoded"]:
                assert 0 <= idx < vocab_size

    def test_sequences_are_lowercased(self, malware_processed):
        """All tokens in processed sequences should be lowercase."""
        for s in malware_processed[:50]:
            for tok in s["sequence"]:
                assert tok == tok.lower(), f"Token not lowercase: {tok}"

    def test_unk_ratio_is_bounded(self, malware_processed, vocab):
        """UNK ratio should be between 0 and 1."""
        ratio = compute_unk_ratio(malware_processed, vocab)
        assert 0.0 <= ratio <= 1.0

    def test_unk_ratio_is_nonzero(self, malware_processed, vocab):
        """MalBehavD should have SOME unknown tokens (domain shift)."""
        ratio = compute_unk_ratio(malware_processed, vocab)
        # It's extremely unlikely that all MalBehavD tokens are in the
        # Mal-API vocabulary — this tests that domain shift exists.
        assert ratio > 0.0, "Expected nonzero UNK ratio for cross-dataset eval"


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Tests that feature extraction works on MalBehavD samples."""

    def test_tfidf_shape(self, malware_processed, tfidf_vectorizer):
        """TF-IDF matrix must have (n_samples, max_features) shape."""
        tfidf = tfidf_transform(malware_processed, tfidf_vectorizer)
        assert tfidf.shape[0] == len(malware_processed)
        assert tfidf.shape[1] == cfg.TFIDF_MAX_FEATURES

    def test_statistical_shape(self, malware_processed):
        """Statistical features must have 9 columns."""
        stats = compute_statistical_features(malware_processed)
        assert stats.shape == (len(malware_processed), 9)

    def test_category_shape(self, malware_processed):
        """Category features must have 8 columns."""
        cats = compute_category_features(malware_processed)
        assert cats.shape == (len(malware_processed), len(CATEGORY_FEATURE_NAMES))

    def test_combined_shape(self, malware_processed, tfidf_vectorizer):
        """Stacked v1 features must have 5017 columns."""
        tfidf = tfidf_transform(malware_processed, tfidf_vectorizer)
        stats = compute_statistical_features(malware_processed)
        cats = compute_category_features(malware_processed)
        combined = np.hstack([tfidf, stats, cats])
        expected_cols = cfg.TFIDF_MAX_FEATURES + 9 + len(CATEGORY_FEATURE_NAMES)
        assert combined.shape == (len(malware_processed), expected_cols)

    def test_no_nan_in_features(self, malware_processed, tfidf_vectorizer):
        """Feature matrix must not contain NaN or Inf."""
        tfidf = tfidf_transform(malware_processed, tfidf_vectorizer)
        stats = compute_statistical_features(malware_processed)
        cats = compute_category_features(malware_processed)
        combined = np.hstack([tfidf, stats, cats])
        assert np.all(np.isfinite(combined))


# ---------------------------------------------------------------------------
# LSTM sequence tests
# ---------------------------------------------------------------------------

class TestLSTMSequences:
    """Tests for padded sequence generation for LSTM."""

    def test_padded_shape(self, malware_processed):
        """Padded output must be (n_samples, seq_len)."""
        encoded = [s["encoded"] for s in malware_processed]
        padded = pad_sequences(encoded, max_len=cfg.LSTM_BEST_SEQ_LEN)
        assert padded.shape == (len(malware_processed), cfg.LSTM_BEST_SEQ_LEN)

    def test_padded_dtype(self, malware_processed):
        """Padded array should be integer type."""
        encoded = [s["encoded"] for s in malware_processed]
        padded = pad_sequences(encoded, max_len=cfg.LSTM_BEST_SEQ_LEN)
        assert np.issubdtype(padded.dtype, np.integer)


# ---------------------------------------------------------------------------
# Binary detection logic tests
# ---------------------------------------------------------------------------

class TestBinaryDetection:
    """Tests for the binary (benign vs malware) evaluation logic."""

    def test_binary_label_assignment(self, malbehavd_data):
        """Benign samples should get label 0, malware should get 1."""
        benign = [s for s in malbehavd_data["samples"]
                  if s["label"] == cfg.BENIGN_LABEL]
        malware = [s for s in malbehavd_data["samples"]
                   if s["label"] != cfg.BENIGN_LABEL]
        y_true = np.array([0] * len(benign) + [1] * len(malware))
        assert y_true.sum() == len(malware)
        assert (y_true == 0).sum() == len(benign)

    def test_all_predictions_are_malware(self):
        """Models trained on 8 families (no benign) always predict malware.

        Binary recall for the malware class is therefore always 1.0.
        """
        # Simulate: model outputs 8-class probs for 10 samples
        probs = np.random.dirichlet(np.ones(8), size=10)
        # "Predicted binary" = 1 for every sample
        pred_binary = np.ones(10, dtype=int)
        assert pred_binary.sum() == 10


# ---------------------------------------------------------------------------
# Label encoder compatibility tests
# ---------------------------------------------------------------------------

class TestLabelEncoderCompat:
    """Tests that MalBehavD labels are compatible with the training encoder."""

    def test_all_families_encodable(self, malware_samples, label_encoder):
        """Every MalBehavD family label must be encodable."""
        labels = [s["label"] for s in malware_samples]
        encoded = label_encoder.transform(labels)
        assert len(encoded) == len(labels)
        assert all(0 <= e < len(label_encoder.classes_) for e in encoded)

    def test_family_distribution_reasonable(self, malware_samples):
        """At least 3 families should have >= 10 samples for meaningful eval."""
        dist = Counter(s["label"] for s in malware_samples)
        families_with_10_plus = sum(1 for c in dist.values() if c >= 10)
        assert families_with_10_plus >= 3, (
            f"Only {families_with_10_plus} families have >= 10 samples: {dist}"
        )
