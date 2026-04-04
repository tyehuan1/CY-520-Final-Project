"""
Tests for Olivera-style preprocessing: deduplicate consecutive calls,
truncate to first 100, and compare against actual Olivera data.
"""

import numpy as np
import pytest

import config as cfg
from src.preprocessing import (
    deduplicate_consecutive,
    olivera_preprocess_samples,
    olivera_style_preprocess,
)


# ── Unit tests for deduplicate_consecutive ────────────────────────────────


class TestDeduplicateConsecutive:
    """Tests for strict consecutive deduplication."""

    def test_no_duplicates(self):
        seq = ["a", "b", "c", "d"]
        assert deduplicate_consecutive(seq) == ["a", "b", "c", "d"]

    def test_all_same(self):
        seq = ["a", "a", "a", "a"]
        assert deduplicate_consecutive(seq) == ["a"]

    def test_consecutive_pairs(self):
        seq = ["a", "a", "b", "b", "c", "c"]
        assert deduplicate_consecutive(seq) == ["a", "b", "c"]

    def test_non_consecutive_repeats_kept(self):
        """Non-consecutive duplicates should be preserved."""
        seq = ["a", "b", "a", "b", "a"]
        assert deduplicate_consecutive(seq) == ["a", "b", "a", "b", "a"]

    def test_long_run(self):
        seq = ["x"] * 100
        assert deduplicate_consecutive(seq) == ["x"]

    def test_empty(self):
        assert deduplicate_consecutive([]) == []

    def test_single(self):
        assert deduplicate_consecutive(["a"]) == ["a"]

    def test_mixed_runs(self):
        seq = ["a", "a", "b", "c", "c", "c", "b", "b", "a"]
        assert deduplicate_consecutive(seq) == ["a", "b", "c", "b", "a"]


# ── Unit tests for olivera_style_preprocess ───────────────────────────────


class TestOliveraStylePreprocess:
    """Tests for the full Olivera-style preprocessing pipeline."""

    def test_sandbox_tokens_removed(self):
        seq = ["a", "__exception__", "b", "__anomaly__", "c"]
        result = olivera_style_preprocess(seq, max_calls=100)
        assert "__exception__" not in result
        assert "__anomaly__" not in result
        assert result == ["a", "b", "c"]

    def test_truncation_to_100(self):
        seq = [f"api_{i}" for i in range(200)]
        result = olivera_style_preprocess(seq, max_calls=100)
        assert len(result) == 100

    def test_dedup_then_truncate(self):
        """Deduplication happens before truncation."""
        # 200 tokens but with lots of consecutive duplicates
        seq = []
        for i in range(200):
            seq.extend([f"api_{i}"] * 5)  # 1000 tokens total
        result = olivera_style_preprocess(seq, max_calls=100)
        assert len(result) == 100
        # First 100 should be api_0 through api_99
        assert result[0] == "api_0"
        assert result[99] == "api_99"

    def test_short_sequence_not_padded(self):
        """Sequences shorter than max_calls are NOT padded."""
        seq = ["a", "b", "c"]
        result = olivera_style_preprocess(seq, max_calls=100)
        assert len(result) == 3

    def test_default_max_calls_is_100(self):
        seq = [f"api_{i}" for i in range(200)]
        result = olivera_style_preprocess(seq)
        assert len(result) == cfg.OLIVERA_SEQ_COLUMNS  # 100

    def test_preserves_order(self):
        seq = ["ntclose", "ldrloaddll", "ntcreatefile", "ntclose", "ntwritefile"]
        result = olivera_style_preprocess(seq, max_calls=100)
        assert result == ["ntclose", "ldrloaddll", "ntcreatefile", "ntclose", "ntwritefile"]


# ── Integration tests: Olivera-preprocessed Mal-API ──────────────────────


class TestOliveraPreprocessMalAPI:
    """Test Olivera-style preprocessing on actual Mal-API data."""

    @pytest.fixture(scope="class")
    def malapi_samples(self):
        from src.data_loader import load_mal_api
        return load_mal_api()

    @pytest.fixture(scope="class")
    def olivera_processed(self, malapi_samples):
        return olivera_preprocess_samples(malapi_samples)

    def test_sample_count_preserved(self, malapi_samples, olivera_processed):
        assert len(olivera_processed) == len(malapi_samples)

    def test_labels_preserved(self, malapi_samples, olivera_processed):
        for orig, proc in zip(malapi_samples[:100], olivera_processed[:100]):
            assert proc["label"] == orig["label"]

    def test_max_length_is_100(self, olivera_processed):
        for s in olivera_processed:
            assert len(s["sequence"]) <= cfg.OLIVERA_SEQ_COLUMNS

    def test_no_consecutive_duplicates(self, olivera_processed):
        """No sequence should have consecutive duplicate API calls."""
        for s in olivera_processed[:500]:
            seq = s["sequence"]
            for i in range(len(seq) - 1):
                assert seq[i] != seq[i + 1], (
                    f"Consecutive duplicate '{seq[i]}' at position {i}"
                )

    def test_no_sandbox_tokens(self, olivera_processed):
        for s in olivera_processed[:500]:
            for tok in s["sequence"]:
                assert not tok.startswith("__"), f"Sandbox token '{tok}' not removed"

    def test_majority_reach_100(self, olivera_processed):
        """Most Mal-API samples should reach the 100-call limit."""
        at_100 = sum(1 for s in olivera_processed
                     if len(s["sequence"]) == cfg.OLIVERA_SEQ_COLUMNS)
        ratio = at_100 / len(olivera_processed)
        # From our analysis: 73.7% reach 100
        assert ratio > 0.70, f"Only {ratio:.1%} reached 100 calls"

    def test_length_distribution_comparable_to_olivera(self, olivera_processed):
        """Median length should be close to 100 (Olivera is exactly 100)."""
        lens = [len(s["sequence"]) for s in olivera_processed]
        median = np.median(lens)
        assert median == 100, f"Median is {median}, expected 100"

    def test_family_distribution_unchanged(self, malapi_samples, olivera_processed):
        """Preprocessing should not change the family distribution."""
        from collections import Counter
        orig_dist = Counter(s["label"] for s in malapi_samples)
        proc_dist = Counter(s["label"] for s in olivera_processed)
        assert orig_dist == proc_dist
