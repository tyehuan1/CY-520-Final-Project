"""
Tests for Olivera dataset loading and Hybrid Analysis labeling pipeline.

Tests cover:
- Data loading and format validation
- API decode map correctness
- Family extraction from HA response structures
- Cache helpers
"""

from collections import Counter
from unittest.mock import patch

import pytest

import config as cfg
from src.data_loader import load_olivera
from src.hybrid_analysis_labeler import (
    extract_family_from_ha_response,
    load_cache,
    save_cache,
)
from src.olivera_api_map import OLIVERA_API_DECODE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_samples():
    """Load Olivera dataset without decoding (integer strings)."""
    return load_olivera()


@pytest.fixture(scope="module")
def decoded_samples():
    """Load Olivera dataset with API call decoding."""
    return load_olivera(api_decode_map=OLIVERA_API_DECODE)


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestOliveraLoader:
    """Tests for the Olivera dataset loader."""

    def test_sample_count(self, raw_samples):
        """Dataset must contain exactly 43,876 samples."""
        assert len(raw_samples) == cfg.OLIVERA_EXPECTED_SAMPLES

    def test_benign_count(self, raw_samples):
        """Should have 1,079 benign samples."""
        benign = sum(1 for s in raw_samples if s["label"] == cfg.BENIGN_LABEL)
        assert benign == cfg.OLIVERA_EXPECTED_BENIGN

    def test_malware_count(self, raw_samples):
        """Should have 42,797 malware samples."""
        malware = sum(1 for s in raw_samples if s["label"] == cfg.MALWARE_LABEL)
        assert malware == cfg.OLIVERA_EXPECTED_MALWARE

    def test_output_format_keys(self, raw_samples):
        """Each sample must have sequence, label, and hash keys."""
        for s in raw_samples[:10]:
            assert "sequence" in s
            assert "label" in s
            assert "hash" in s

    def test_sequence_length(self, raw_samples):
        """Every sequence must have exactly 100 timesteps."""
        for s in raw_samples[:100]:
            assert len(s["sequence"]) == cfg.OLIVERA_SEQ_COLUMNS

    def test_hash_is_md5(self, raw_samples):
        """Hashes should be 32-character hex strings (MD5)."""
        for s in raw_samples[:100]:
            assert len(s["hash"]) == 32
            assert all(c in "0123456789abcdef" for c in s["hash"])

    def test_labels_are_valid(self, raw_samples):
        """Labels must be either Benign or Malware."""
        valid = {cfg.BENIGN_LABEL, cfg.MALWARE_LABEL}
        for s in raw_samples[:100]:
            assert s["label"] in valid

    def test_raw_sequences_are_integer_strings(self, raw_samples):
        """Without decode map, sequences should be string integers."""
        for tok in raw_samples[0]["sequence"]:
            assert tok.isdigit()


# ---------------------------------------------------------------------------
# API decode map tests
# ---------------------------------------------------------------------------

class TestDecodeMap:
    """Tests for the Olivera API decode map."""

    def test_map_size(self):
        """Decode map must cover indices 0 through 306."""
        assert len(OLIVERA_API_DECODE) == 307
        assert 0 in OLIVERA_API_DECODE
        assert 306 in OLIVERA_API_DECODE

    def test_decoded_sequences_are_strings(self, decoded_samples):
        """Decoded sequences should contain API call name strings."""
        first_tok = decoded_samples[0]["sequence"][0]
        assert isinstance(first_tok, str)
        assert not first_tok.isdigit()  # Should be a name, not a number

    def test_all_data_values_in_map(self, raw_samples):
        """Every integer value in the dataset must be in the decode map."""
        for s in raw_samples[:500]:
            for tok in s["sequence"]:
                assert int(tok) in OLIVERA_API_DECODE

    def test_known_mappings(self):
        """Spot-check a few known index-to-name mappings."""
        assert OLIVERA_API_DECODE[0] == "NtOpenThread"
        assert OLIVERA_API_DECODE[306] == "__exception__"
        assert OLIVERA_API_DECODE[215] == "NtClose"


# ---------------------------------------------------------------------------
# HA family extraction tests
# ---------------------------------------------------------------------------

class TestHAFamilyExtraction:
    """Tests for extract_family_from_ha_response."""

    def test_trojan_from_tags(self):
        """Tags containing 'trojan' should classify as Trojan."""
        resp = {
            "vx_family": "Trojan.Emotet",
            "classification_tags": ["trojan", "emotet"],
        }
        family = extract_family_from_ha_response(resp)
        assert family == "Trojan"

    def test_worm_from_tags(self):
        """Tags containing 'worm' should classify as Worms."""
        resp = {
            "vx_family": "Worm.Conficker",
            "classification_tags": ["worm", "conficker"],
        }
        family = extract_family_from_ha_response(resp)
        assert family == "Worms"

    def test_backdoor_from_vx_family(self):
        """vx_family containing 'backdoor' should classify as Backdoor."""
        resp = {
            "vx_family": "Backdoor.Poison",
            "classification_tags": [],
        }
        family = extract_family_from_ha_response(resp)
        assert family == "Backdoor"

    def test_downloader_from_tags(self):
        """Tag 'downloader' should classify as Downloader."""
        resp = {
            "vx_family": "Trojan.Downloader",
            "classification_tags": ["downloader", "trojan"],
        }
        family = extract_family_from_ha_response(resp)
        assert family == "Downloader"

    def test_spyware_from_banker_tag(self):
        """Tag 'banker' should classify as Spyware."""
        resp = {
            "vx_family": "Banker.Zeus",
            "classification_tags": ["banker"],
        }
        family = extract_family_from_ha_response(resp)
        assert family == "Spyware"

    def test_generic_malware_returns_none(self):
        """Generic labels like 'Malware' should return None."""
        resp = {
            "vx_family": "Malware",
            "classification_tags": [],
        }
        family = extract_family_from_ha_response(resp)
        assert family is None

    def test_generic_trojan_returns_none(self):
        """'Trojan.Generic' with no useful tags should return None."""
        resp = {
            "vx_family": "Trojan.Generic",
            "classification_tags": [],
        }
        family = extract_family_from_ha_response(resp)
        assert family is None

    def test_empty_response_returns_none(self):
        """Empty response should return None."""
        assert extract_family_from_ha_response({}) is None
        assert extract_family_from_ha_response(None) is None

    def test_virus_from_vx_family(self):
        """vx_family containing 'virus' should classify as Virus."""
        resp = {
            "vx_family": "Virus.Sality",
            "classification_tags": ["virus"],
        }
        family = extract_family_from_ha_response(resp)
        assert family == "Virus"

    def test_dropper_from_tags(self):
        """Tag 'dropper' should classify as Dropper."""
        resp = {
            "vx_family": "Dropper.Agent",
            "classification_tags": ["dropper"],
        }
        family = extract_family_from_ha_response(resp)
        assert family == "Dropper"

    def test_adware_from_tags(self):
        """Tag 'adware' should classify as Adware."""
        resp = {
            "vx_family": "Adware.BrowseFox",
            "classification_tags": ["adware"],
        }
        family = extract_family_from_ha_response(resp)
        assert family == "Adware"


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestCaching:
    """Tests for HA cache save/load."""

    def test_save_and_load(self, tmp_path):
        """Cache round-trips correctly."""
        cache_path = tmp_path / "test_cache.json"
        original = {"abc123": {"vx_family": "Trojan.Test"}, "def456": None}
        save_cache(original, cache_path)
        loaded = load_cache(cache_path)
        assert loaded == original

    def test_load_missing_file(self, tmp_path):
        """Loading from nonexistent path returns empty dict."""
        cache_path = tmp_path / "nonexistent.json"
        assert load_cache(cache_path) == {}
