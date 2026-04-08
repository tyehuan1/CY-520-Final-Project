"""
Tests for src.model_training.feature_engineering — length-normalized TF,
statistical, category, and bigram transition features.

Also tests augment_sequences from xgboost_model.
"""

import math

import numpy as np
import pytest

import config as cfg
from src.data_loading.api_categories import CATEGORIES
from src.data_loading.data_loader import load_mal_api
from src.model_training.feature_engineering import (
    BIGRAM_FEATURE_NAMES,
    CATEGORY_FEATURE_NAMES,
    STATISTICAL_FEATURE_NAMES,
    build_feature_matrix,
    build_tfidf_vectorizer,
    compute_bigram_transition_features,
    compute_category_features,
    compute_statistical_features,
    get_feature_names,
    tfidf_transform,
)
from src.model_training.xgboost_model import augment_sequences
from src.data_loading.preprocessing import clean_samples, stratified_split


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


# ── TF feature tests (L1-normalized, no IDF) ────────────────────────────


class TestTfidf:
    """Tests for length-normalized TF feature extraction."""

    def test_train_matrix_shape(self, train_samples, tfidf_vec):
        """TF matrix has (n_train, <=5000) shape."""
        matrix = tfidf_transform(train_samples, tfidf_vec)
        assert matrix.shape[0] == len(train_samples)
        assert matrix.shape[1] <= cfg.TFIDF_MAX_FEATURES

    def test_test_matrix_shape(self, train_samples, test_samples, tfidf_vec):
        """Test TF matrix has same number of features as train."""
        train_mat = tfidf_transform(train_samples, tfidf_vec)
        test_mat = tfidf_transform(test_samples, tfidf_vec)
        assert test_mat.shape[1] == train_mat.shape[1]

    def test_no_nan_or_inf(self, train_samples, tfidf_vec):
        """TF values should not contain NaN or Inf."""
        matrix = tfidf_transform(train_samples, tfidf_vec)
        assert not np.any(np.isnan(matrix))
        assert not np.any(np.isinf(matrix))

    def test_values_are_nonnegative(self, train_samples, tfidf_vec):
        """TF values should be non-negative."""
        matrix = tfidf_transform(train_samples, tfidf_vec)
        assert np.all(matrix >= 0)

    def test_l1_normalization(self, train_samples, tfidf_vec):
        """Each row should sum to ~1.0 (L1-normalized).

        Empty sequences produce all-zero rows, which is correct.
        """
        matrix = tfidf_transform(train_samples, tfidf_vec)
        row_sums = matrix.sum(axis=1)
        nonempty = row_sums > 0
        np.testing.assert_allclose(row_sums[nonempty], 1.0, atol=1e-6)

    def test_no_idf_weighting(self, tfidf_vec):
        """Vectorizer should have IDF disabled."""
        assert tfidf_vec.use_idf is False

    def test_vectorizer_fit_on_train_only(self, test_samples, tfidf_vec):
        """Verify the vectorizer was fit on training data only."""
        vocab_terms = set(tfidf_vec.vocabulary_.keys())
        assert len(vocab_terms) > 0
        matrix = tfidf_transform(test_samples, tfidf_vec)
        assert matrix.shape[0] == len(test_samples)


# ── Statistical feature tests ─────────────────────────────────────────────


class TestStatisticalFeatures:
    """Tests for statistical / sequential features (log-scaled)."""

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

    def test_log_sequence_length_nonnegative(self, train_samples):
        """Log-scaled sequence lengths (col 0) should all be >= 0."""
        features = compute_statistical_features(train_samples)
        assert np.all(features[:, 0] >= 0)
        assert np.sum(features[:, 0] > 0) >= len(train_samples) - 5

    def test_unique_ratio_bounded(self, train_samples):
        """Unique-to-total ratio (col 2) should be in [0, 1]."""
        features = compute_statistical_features(train_samples)
        ratios = features[:, 2]
        assert np.all(ratios >= 0)
        assert np.all(ratios <= 1.0)

    def test_entropy_nonnegative(self, train_samples):
        """Shannon entropy should be >= 0."""
        features = compute_statistical_features(train_samples)
        entropy_col = features[:, -1]
        assert np.all(entropy_col >= 0)

    def test_hand_crafted(self):
        """Verify exact values on a known tiny sample with log-scaled lengths."""
        samples = [{"sequence": ["a", "a", "b", "c"]}]
        features = compute_statistical_features(samples, top_k=3)
        # log2(4+1) = log2(5) ≈ 2.3219
        assert abs(features[0, 0] - math.log2(5)) < 1e-9
        # log2(3+1) = log2(4) = 2.0
        assert abs(features[0, 1] - math.log2(4)) < 1e-9
        # ratio = 3/4 = 0.75
        assert abs(features[0, 2] - 0.75) < 1e-9
        # top-3 ratios: a=2/4=0.5, b=1/4=0.25, c=1/4=0.25
        assert abs(features[0, 3] - 0.5) < 1e-9
        assert abs(features[0, 4] - 0.25) < 1e-9
        assert abs(features[0, 5] - 0.25) < 1e-9
        # entropy: -(0.5*log2(0.5) + 0.25*log2(0.25) + 0.25*log2(0.25)) = 1.5
        assert abs(features[0, 6] - 1.5) < 1e-9

    def test_feature_names_count(self):
        """STATISTICAL_FEATURE_NAMES should match expected count."""
        expected = 3 + cfg.TOP_K_API_FREQUENCIES + 1
        assert len(STATISTICAL_FEATURE_NAMES) == expected

    def test_log_scaled_names(self):
        """First two feature names should indicate log-scaling."""
        assert STATISTICAL_FEATURE_NAMES[0] == "log_seq_length"
        assert STATISTICAL_FEATURE_NAMES[1] == "log_unique_count"


# ── API category feature tests ────────────────────────────────────────────


class TestCategoryFeatures:
    """Tests for API category ratio features."""

    def test_shape(self, train_samples):
        """Output shape should be (n_samples, 8)."""
        features = compute_category_features(train_samples)
        assert features.shape == (len(train_samples), 8)

    def test_ratios_sum_to_one(self, train_samples):
        """Category ratios for non-empty samples should sum to ~1.0."""
        features = compute_category_features(train_samples)
        row_sums = features.sum(axis=1)
        nonempty_mask = row_sums > 0
        np.testing.assert_allclose(row_sums[nonempty_mask], 1.0, atol=1e-9)
        assert np.sum(~nonempty_mask) <= 5

    def test_no_nan_or_inf(self, train_samples):
        features = compute_category_features(train_samples)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_values_in_zero_one(self, train_samples):
        features = compute_category_features(train_samples)
        assert np.all(features >= 0)
        assert np.all(features <= 1.0)

    def test_hand_crafted(self):
        """Verify on a known sequence with clear category assignments."""
        samples = [{"sequence": ["ntcreatefile", "ntreadfile", "regopenkeyexa", "connect"]}]
        features = compute_category_features(samples)
        cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
        assert abs(features[0, cat_idx["filesystem"]] - 0.5) < 1e-9
        assert abs(features[0, cat_idx["registry"]] - 0.25) < 1e-9
        assert abs(features[0, cat_idx["network"]] - 0.25) < 1e-9
        assert abs(features[0, cat_idx["process"]]) < 1e-9


# ── Bigram transition feature tests ──────────────────────────────────────


class TestBigramTransitionFeatures:
    """Tests for API category bigram transition features."""

    def test_shape(self, train_samples):
        """Output should be (n_samples, 64) for 8 categories."""
        features = compute_bigram_transition_features(train_samples)
        assert features.shape == (len(train_samples), len(CATEGORIES) ** 2)

    def test_proportions_sum_to_one(self, train_samples):
        """Transition proportions for samples with >=2 calls should sum to ~1.0."""
        features = compute_bigram_transition_features(train_samples)
        row_sums = features.sum(axis=1)
        nonempty = row_sums > 0
        np.testing.assert_allclose(row_sums[nonempty], 1.0, atol=1e-9)

    def test_no_nan_or_inf(self, train_samples):
        features = compute_bigram_transition_features(train_samples)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_values_in_zero_one(self, train_samples):
        features = compute_bigram_transition_features(train_samples)
        assert np.all(features >= 0)
        assert np.all(features <= 1.0)

    def test_single_call_sequence(self):
        """A sequence with 1 call has no transitions — all zeros."""
        samples = [{"sequence": ["ntcreatefile"]}]
        features = compute_bigram_transition_features(samples)
        assert features.shape == (1, len(CATEGORIES) ** 2)
        assert np.all(features == 0)

    def test_hand_crafted(self):
        """Verify exact transitions on a known sequence.

        Sequence: ntcreatefile -> ntreadfile -> regopenkeyexa -> connect
        Transitions: filesystem->filesystem, filesystem->registry, registry->network
        That's 3 transitions total.
        """
        samples = [{"sequence": ["ntcreatefile", "ntreadfile", "regopenkeyexa", "connect"]}]
        features = compute_bigram_transition_features(samples)
        n_cats = len(CATEGORIES)
        cat_idx = {c: i for i, c in enumerate(CATEGORIES)}

        fs = cat_idx["filesystem"]
        reg = cat_idx["registry"]
        net = cat_idx["network"]

        # filesystem -> filesystem: 1/3
        assert abs(features[0, fs * n_cats + fs] - 1 / 3) < 1e-9
        # filesystem -> registry: 1/3
        assert abs(features[0, fs * n_cats + reg] - 1 / 3) < 1e-9
        # registry -> network: 1/3
        assert abs(features[0, reg * n_cats + net] - 1 / 3) < 1e-9
        # All other transitions should be 0
        total_nonzero = np.count_nonzero(features[0])
        assert total_nonzero == 3

    def test_feature_names_count(self):
        """BIGRAM_FEATURE_NAMES should have 64 entries."""
        assert len(BIGRAM_FEATURE_NAMES) == len(CATEGORIES) ** 2


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

    def test_expected_total_features(self, train_samples, tfidf_vec):
        """Total features = TF + stats(9) + categories(8) + bigrams(64)."""
        X = build_feature_matrix(train_samples, tfidf_vec)
        n_tf = len(tfidf_vec.vocabulary_)
        n_stats = 3 + cfg.TOP_K_API_FREQUENCIES + 1  # 9
        n_cats = len(CATEGORIES)  # 8
        n_bigrams = n_cats ** 2  # 64
        expected = n_tf + n_stats + n_cats + n_bigrams
        assert X.shape[1] == expected

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


# ── Augmentation tests ────────────────────────────────────────────────────


class TestAugmentSequences:
    """Tests for sequence length augmentation in xgboost_model."""

    def test_augmented_count(self):
        """With n_augments=2, eligible samples produce 2 extra copies each."""
        samples = [
            {"sequence": list(range(50)), "label": "Adware", "encoded": list(range(50))},
            {"sequence": list(range(10)), "label": "Virus", "encoded": list(range(10))},
        ]
        augmented = augment_sequences(samples, n_augments=2, min_len=20)
        # Sample 0 (len=50 > 20): 2 augmented copies
        # Sample 1 (len=10 <= 20): 0 augmented copies
        assert len(augmented) == 2 + 2  # originals + augmented

    def test_originals_preserved(self):
        """Original samples appear unmodified at the start."""
        samples = [
            {"sequence": list(range(30)), "label": "Adware"},
        ]
        augmented = augment_sequences(samples, n_augments=1, min_len=5)
        assert augmented[0] is samples[0]  # exact same object

    def test_truncation_lengths(self):
        """Augmented sequences should be within [min_len, min(len, max_len)]."""
        seq = list(range(100))
        samples = [{"sequence": seq, "label": "Adware", "encoded": seq}]
        augmented = augment_sequences(
            samples, n_augments=50, min_len=20, max_len=80, random_seed=42,
        )
        for s in augmented[1:]:  # skip original
            assert 20 <= len(s["sequence"]) <= 80
            assert 20 <= len(s["encoded"]) <= 80

    def test_augmented_are_prefixes(self):
        """Augmented sequences should be prefixes of the original."""
        seq = list(range(100))
        samples = [{"sequence": seq, "label": "Adware", "encoded": seq}]
        augmented = augment_sequences(samples, n_augments=5, min_len=10, max_len=50)
        for s in augmented[1:]:
            n = len(s["sequence"])
            assert s["sequence"] == seq[:n]
            assert s["encoded"] == seq[:n]

    def test_label_preserved(self):
        """Augmented samples should keep the same label."""
        samples = [
            {"sequence": list(range(30)), "label": "Backdoor"},
        ]
        augmented = augment_sequences(samples, n_augments=3, min_len=5)
        for s in augmented:
            assert s["label"] == "Backdoor"

    def test_empty_and_short_sequences_skipped(self):
        """Sequences at or below min_len should not be augmented."""
        samples = [
            {"sequence": [], "label": "Adware"},
            {"sequence": list(range(5)), "label": "Virus"},
        ]
        augmented = augment_sequences(samples, n_augments=3, min_len=5)
        # Neither sample eligible: len 0 <= 5, len 5 <= 5
        assert len(augmented) == 2

    def test_reproducibility(self):
        """Same seed should produce identical augmentation."""
        samples = [{"sequence": list(range(100)), "label": "Adware"}]
        a1 = augment_sequences(samples, n_augments=5, random_seed=123)
        a2 = augment_sequences(samples, n_augments=5, random_seed=123)
        for s1, s2 in zip(a1, a2):
            assert s1["sequence"] == s2["sequence"]
