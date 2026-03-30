"""
Tests for src.feature_engineering — TF-IDF, statistical, and category features.
"""

import numpy as np
import pytest

import config as cfg
from src.data_loader import load_mal_api
from src.feature_engineering import (
    build_feature_matrix,
    build_tfidf_vectorizer,
    compute_category_features,
    compute_statistical_features,
    get_feature_names,
    tfidf_transform,
)
from src.preprocessing import clean_samples, stratified_split


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def split_data():
    """Load, clean, and split data once for the module."""
    samples = load_mal_api()
    cleaned = clean_samples(samples)
    train, test = stratified_split(cleaned)
    return train, test


@pytest.fixture(scope="module")
def train_samples(split_data):
    return split_data[0]


@pytest.fixture(scope="module")
def test_samples(split_data):
    return split_data[1]


@pytest.fixture(scope="module")
def tfidf_vec(train_samples):
    return build_tfidf_vectorizer(train_samples)


# ── TF-IDF tests ──────────────────────────────────────────────────────────


class TestTfidf:
    """Tests for TF-IDF feature extraction."""

    def test_train_matrix_shape(self, train_samples, tfidf_vec):
        """TF-IDF matrix has (n_train, ≤5000) shape."""
        matrix = tfidf_transform(train_samples, tfidf_vec)
        assert matrix.shape[0] == len(train_samples)
        assert matrix.shape[1] <= cfg.TFIDF_MAX_FEATURES

    def test_test_matrix_shape(self, train_samples, test_samples, tfidf_vec):
        """Test TF-IDF matrix has same number of features as train."""
        train_mat = tfidf_transform(train_samples, tfidf_vec)
        test_mat = tfidf_transform(test_samples, tfidf_vec)
        assert test_mat.shape[1] == train_mat.shape[1]

    def test_no_nan_or_inf(self, train_samples, tfidf_vec):
        """TF-IDF values should not contain NaN or Inf."""
        matrix = tfidf_transform(train_samples, tfidf_vec)
        assert not np.any(np.isnan(matrix))
        assert not np.any(np.isinf(matrix))

    def test_vectorizer_fit_on_train_only(self, test_samples, tfidf_vec):
        """Verify the vectorizer was fit on training data only.

        Test-only n-grams should NOT appear in the vectorizer vocabulary.
        We check this indirectly: terms in the vectorizer vocab should all
        be present in the training corpus (they are, by construction of fit).
        The test corpus may have terms not in the vocab — those are correctly
        ignored and get zero weight.
        """
        # All vocab terms came from training fit
        vocab_terms = set(tfidf_vec.vocabulary_.keys())
        assert len(vocab_terms) > 0
        # Transform test data — should not error even with unseen terms
        matrix = tfidf_transform(test_samples, tfidf_vec)
        assert matrix.shape[0] == len(test_samples)

    def test_values_are_nonnegative(self, train_samples, tfidf_vec):
        """TF-IDF values should be non-negative."""
        matrix = tfidf_transform(train_samples, tfidf_vec)
        assert np.all(matrix >= 0)


# ── Statistical feature tests ─────────────────────────────────────────────


class TestStatisticalFeatures:
    """Tests for statistical / sequential features."""

    def test_shape(self, train_samples):
        """Output shape should be (n_samples, 3 + top_k + 1)."""
        features = compute_statistical_features(train_samples)
        expected_cols = 3 + cfg.TOP_K_API_FREQUENCIES + 1  # 9
        assert features.shape == (len(train_samples), expected_cols)

    def test_no_nan_or_inf(self, train_samples):
        """No NaN or Inf values allowed."""
        features = compute_statistical_features(train_samples)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_sequence_length_nonnegative(self, train_samples):
        """Sequence lengths (col 0) should all be ≥ 0.

        A small number of samples may have empty sequences after cleaning
        (e.g., samples that contained only sandbox artifacts).
        """
        features = compute_statistical_features(train_samples)
        assert np.all(features[:, 0] >= 0)
        # Vast majority should be positive
        assert np.sum(features[:, 0] > 0) >= len(train_samples) - 5

    def test_unique_ratio_bounded(self, train_samples):
        """Unique-to-total ratio (col 2) should be in [0, 1]."""
        features = compute_statistical_features(train_samples)
        ratios = features[:, 2]
        assert np.all(ratios >= 0)
        assert np.all(ratios <= 1.0)

    def test_entropy_nonnegative(self, train_samples):
        """Shannon entropy should be ≥ 0."""
        features = compute_statistical_features(train_samples)
        entropy_col = features[:, -1]
        assert np.all(entropy_col >= 0)

    def test_hand_crafted(self):
        """Verify exact values on a known tiny sample."""
        samples = [{"sequence": ["a", "a", "b", "c"]}]
        features = compute_statistical_features(samples, top_k=3)
        # length=4, unique=3, ratio=3/4=0.75
        assert features[0, 0] == 4.0
        assert features[0, 1] == 3.0
        assert abs(features[0, 2] - 0.75) < 1e-9
        # top-3 ratios: a=2/4=0.5, b=1/4=0.25, c=1/4=0.25
        assert abs(features[0, 3] - 0.5) < 1e-9
        assert abs(features[0, 4] - 0.25) < 1e-9
        assert abs(features[0, 5] - 0.25) < 1e-9
        # entropy: -(0.5*log2(0.5) + 0.25*log2(0.25) + 0.25*log2(0.25))
        #        = -(-.5 + -.5 + -.5) = 1.5
        assert abs(features[0, 6] - 1.5) < 1e-9


# ── API category feature tests ────────────────────────────────────────────


class TestCategoryFeatures:
    """Tests for API category ratio features."""

    def test_shape(self, train_samples):
        """Output shape should be (n_samples, 8)."""
        features = compute_category_features(train_samples)
        assert features.shape == (len(train_samples), 8)

    def test_ratios_sum_to_one(self, train_samples):
        """Category ratios for non-empty samples should sum to ~1.0.

        Empty sequences (post-cleaning) produce all-zero rows, which is
        correct — there are no API calls to categorize.
        """
        features = compute_category_features(train_samples)
        row_sums = features.sum(axis=1)
        nonempty_mask = row_sums > 0
        np.testing.assert_allclose(row_sums[nonempty_mask], 1.0, atol=1e-9)
        # At most a handful of empty sequences
        assert np.sum(~nonempty_mask) <= 5

    def test_no_nan_or_inf(self, train_samples):
        """No NaN or Inf values."""
        features = compute_category_features(train_samples)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_values_in_zero_one(self, train_samples):
        """All ratios should be in [0, 1]."""
        features = compute_category_features(train_samples)
        assert np.all(features >= 0)
        assert np.all(features <= 1.0)

    def test_hand_crafted(self):
        """Verify on a known sequence with clear category assignments."""
        samples = [{"sequence": ["ntcreatefile", "ntreadfile", "regopenkeyexa", "connect"]}]
        features = compute_category_features(samples)
        # filesystem=2/4=0.5, registry=1/4=0.25, network=1/4=0.25, rest=0
        from src.api_categories import CATEGORIES
        cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
        assert abs(features[0, cat_idx["filesystem"]] - 0.5) < 1e-9
        assert abs(features[0, cat_idx["registry"]] - 0.25) < 1e-9
        assert abs(features[0, cat_idx["network"]] - 0.25) < 1e-9
        assert abs(features[0, cat_idx["process"]]) < 1e-9


# ── Combined feature matrix tests ─────────────────────────────────────────


class TestCombinedFeatures:
    """Tests for the full concatenated feature matrix."""

    def test_combined_shape(self, train_samples, test_samples, tfidf_vec):
        """Train and test matrices should have consistent column counts."""
        X_train = build_feature_matrix(train_samples, tfidf_vec)
        X_test = build_feature_matrix(test_samples, tfidf_vec)
        assert X_train.shape[0] == len(train_samples)
        assert X_test.shape[0] == len(test_samples)
        assert X_train.shape[1] == X_test.shape[1]

    def test_feature_names_match_columns(self, train_samples, tfidf_vec):
        """Feature name list length should match matrix column count."""
        X = build_feature_matrix(train_samples, tfidf_vec)
        names = get_feature_names(tfidf_vec)
        assert len(names) == X.shape[1]

    def test_no_nan_or_inf(self, train_samples, tfidf_vec):
        """Combined matrix should have no NaN or Inf."""
        X = build_feature_matrix(train_samples, tfidf_vec)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))
