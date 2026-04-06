"""
Tests for src.data_loader — Mal-API-2019 and MalbehavD-V1 loading.
"""

from collections import Counter

import pytest

import config as cfg
from src.data_loading.data_loader import load_mal_api, load_malbehavd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mal_api_samples():
    """Load Mal-API-2019 once for the entire test module."""
    return load_mal_api()


@pytest.fixture(scope="module")
def malbehavd_samples():
    """Load MalbehavD-V1 once for the entire test module."""
    return load_malbehavd()


# ---------------------------------------------------------------------------
# Mal-API-2019 tests
# ---------------------------------------------------------------------------

class TestMalApi:
    """Tests for the Mal-API-2019 dataset loader."""

    def test_sample_count(self, mal_api_samples):
        """Dataset must contain exactly 7,107 samples."""
        assert len(mal_api_samples) == cfg.MAL_API_EXPECTED_SAMPLES

    def test_all_families_present(self, mal_api_samples):
        """All 8 raw Mal-API family names must appear in the dataset."""
        labels = {s["label"] for s in mal_api_samples}
        # The raw dataset always has 8 families (including Trojan).
        # cfg.MALWARE_FAMILIES may exclude Trojan for Stage-2 models.
        raw_families = {
            "Adware", "Backdoor", "Downloader", "Dropper",
            "Spyware", "Trojan", "Virus", "Worms",
        }
        assert labels == raw_families

    def test_label_distribution(self, mal_api_samples):
        """Verify per-family counts match the known distribution."""
        expected = {
            "Trojan": 1001,
            "Backdoor": 1001,
            "Downloader": 1001,
            "Worms": 1001,
            "Virus": 1001,
            "Dropper": 891,
            "Spyware": 832,
            "Adware": 379,
        }
        counts = Counter(s["label"] for s in mal_api_samples)
        assert dict(counts) == expected

    def test_sequences_are_nonempty_lists(self, mal_api_samples):
        """Every sequence must be a non-empty list of strings."""
        for i, s in enumerate(mal_api_samples):
            assert isinstance(s["sequence"], list), f"Sample {i}: not a list"
            assert len(s["sequence"]) > 0, f"Sample {i}: empty sequence"

    def test_no_empty_strings_in_sequences(self, mal_api_samples):
        """No sequence should contain empty-string tokens."""
        for i, s in enumerate(mal_api_samples):
            for token in s["sequence"]:
                assert isinstance(token, str), f"Sample {i}: non-str token"
                assert token != "", f"Sample {i}: empty string in sequence"

    def test_output_format_keys(self, mal_api_samples):
        """Each sample dict must have 'sequence' and 'label' keys."""
        for s in mal_api_samples:
            assert "sequence" in s
            assert "label" in s


# ---------------------------------------------------------------------------
# MalbehavD-V1 tests
# ---------------------------------------------------------------------------

class TestMalbehavD:
    """Tests for the MalbehavD-V1 dataset loader."""

    def test_sample_count(self, malbehavd_samples):
        """Dataset must contain exactly 2,570 samples."""
        assert len(malbehavd_samples) == cfg.MALBEHAVD_EXPECTED_SAMPLES

    def test_benign_count(self, malbehavd_samples):
        """Must have exactly 1,285 benign samples."""
        count = sum(1 for s in malbehavd_samples if s["label"] == cfg.BENIGN_LABEL)
        assert count == cfg.MALBEHAVD_EXPECTED_BENIGN

    def test_malware_count(self, malbehavd_samples):
        """Must have exactly 1,285 malware samples."""
        count = sum(1 for s in malbehavd_samples if s["label"] == cfg.MALWARE_LABEL)
        assert count == cfg.MALBEHAVD_EXPECTED_MALWARE

    def test_labels_are_valid(self, malbehavd_samples):
        """Labels must be only 'Benign' or 'Malware'."""
        valid = {cfg.BENIGN_LABEL, cfg.MALWARE_LABEL}
        labels = {s["label"] for s in malbehavd_samples}
        assert labels == valid

    def test_sequences_are_nonempty_lists(self, malbehavd_samples):
        """Every sequence must be a non-empty list of strings."""
        for i, s in enumerate(malbehavd_samples):
            assert isinstance(s["sequence"], list), f"Sample {i}: not a list"
            assert len(s["sequence"]) > 0, f"Sample {i}: empty sequence"

    def test_no_empty_strings_in_sequences(self, malbehavd_samples):
        """No sequence should contain empty-string tokens."""
        for i, s in enumerate(malbehavd_samples):
            for token in s["sequence"]:
                assert isinstance(token, str), f"Sample {i}: non-str token"
                assert token != "", f"Sample {i}: empty string in sequence"

    def test_sha256_present(self, malbehavd_samples):
        """Each MalbehavD sample must include a 'sha256' field."""
        for s in malbehavd_samples:
            assert "sha256" in s
            assert len(s["sha256"]) == 64  # SHA-256 hex digest length


# ---------------------------------------------------------------------------
# Cross-dataset format consistency
# ---------------------------------------------------------------------------

class TestFormatConsistency:
    """Both datasets should produce the same core output format."""

    def test_common_keys(self, mal_api_samples, malbehavd_samples):
        """Both datasets must have 'sequence' and 'label' keys."""
        common_keys = {"sequence", "label"}
        for s in mal_api_samples[:5]:
            assert common_keys.issubset(s.keys())
        for s in malbehavd_samples[:5]:
            assert common_keys.issubset(s.keys())

    def test_sequence_type_consistency(self, mal_api_samples, malbehavd_samples):
        """Sequences from both datasets should be List[str]."""
        s1 = mal_api_samples[0]["sequence"]
        s2 = malbehavd_samples[0]["sequence"]
        assert isinstance(s1, list) and isinstance(s1[0], str)
        assert isinstance(s2, list) and isinstance(s2[0], str)

    def test_label_type_consistency(self, mal_api_samples, malbehavd_samples):
        """Labels from both datasets should be strings."""
        assert isinstance(mal_api_samples[0]["label"], str)
        assert isinstance(malbehavd_samples[0]["label"], str)
