"""
Tests for src.virustotal_labeler — label matching, caching, rate limiting.

All tests use mocked VT API responses; no real network calls are made.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config as cfg
from src.virustotal_labeler import (
    extract_family_from_labels,
    extract_family_from_response,
    label_malbehavd_samples,
    load_cache,
    query_virustotal,
    save_cache,
)


# ── Helpers to build mock VT responses ─────────────────────────────────────


def _make_engine(result: str) -> dict:
    """Return a minimal AV engine result dict."""
    return {"category": "malicious", "result": result}


def _make_vt_response(engine_results: dict) -> dict:
    """Wrap engine results into a full VT v3 JSON structure."""
    return {"data": {"attributes": {"last_analysis_results": engine_results}}}


# ── 10+ mocked VT responses for label matching ────────────────────────────

MOCK_RESPONSES = [
    # 1. Clear Trojan match (non-generic labels that carry real Trojan signal)
    (
        {
            "EngineA": _make_engine("Trojan.Emotet"),
            "EngineB": _make_engine("Win32/Trojan.Zbot"),
            "EngineC": _make_engine("Malware.Generic"),
        },
        "Trojan",
    ),
    # 2. Clear Backdoor match
    (
        {
            "EngineA": _make_engine("Backdoor.Win32.Agent"),
            "EngineB": _make_engine("Backdoor/Meterpreter"),
            "EngineC": _make_engine("Backdoor.Generic"),
        },
        "Backdoor",
    ),
    # 3. Clear Downloader match
    (
        {
            "EngineA": _make_engine("Downloader.Upatre"),
            "EngineB": _make_engine("TrojanDownloader.Agent"),
            "EngineC": _make_engine("Downloader.Generic"),
        },
        "Downloader",
    ),
    # 4. Clear Worms match
    (
        {
            "EngineA": _make_engine("Worms.Win32.Conficker"),
            "EngineB": _make_engine("Net-Worms.Agent"),
            "EngineC": _make_engine("Worms/Autorun"),
        },
        "Worms",
    ),
    # 5. Clear Spyware match
    (
        {
            "EngineA": _make_engine("Spyware.Keylogger"),
            "EngineB": _make_engine("Win32.Spyware.Zeus"),
            "EngineC": _make_engine("Spyware.Generic"),
        },
        "Spyware",
    ),
    # 6. Clear Adware match
    (
        {
            "EngineA": _make_engine("Adware.BrowseFox"),
            "EngineB": _make_engine("not-a-virus:Adware.Win32"),
            "EngineC": _make_engine("Adware/Generic"),
        },
        "Adware",
    ),
    # 7. Clear Dropper match
    (
        {
            "EngineA": _make_engine("Dropper.Win32.Agent"),
            "EngineB": _make_engine("TrojanDropper.Generic"),
            "EngineC": _make_engine("Dropper/Agent"),
        },
        "Dropper",
    ),
    # 8. Clear Virus match
    (
        {
            "EngineA": _make_engine("Virus.Win32.Sality"),
            "EngineB": _make_engine("Virus.Ramnit"),
            "EngineC": _make_engine("Win32.Virus.Parite"),
        },
        "Virus",
    ),
    # 9. Mixed weak non-Trojan: Trojan wins (Backdoor has only 1 vote,
    #    below the 5-vote / 30% threshold)
    (
        {
            "EngineA": _make_engine("Trojan.Emotet"),
            "EngineB": _make_engine("Backdoor.Generic"),
            "EngineC": _make_engine("Trojan/Win32.Agent"),
            "EngineD": _make_engine("Trojan.Downloader"),  # Trojan + Downloader
        },
        "Trojan",  # Backdoor=1 < 5 votes, falls back to Trojan
    ),
    # 10. Mixed strong non-Trojan: Worms wins (7 votes >= 5 AND >= 30% of 9)
    (
        {
            "E1": _make_engine("Worm.Python.Generic"),
            "E2": _make_engine("Worm.Agent.Win32"),
            "E3": _make_engine("Worm:Win32/Agent"),
            "E4": _make_engine("HEUR:Worm.Python.Generic"),
            "E5": _make_engine("Net-Worm.Agent"),
            "E6": _make_engine("W32/Agent.worm"),
            "E7": _make_engine("Worm.VBAgent"),
            "T1": _make_engine("Trojan.Siggen"),
            "T2": _make_engine("Trojan.Win32.Agent"),
            "T3": _make_engine("Win32:Troj-gen"),
            "T4": _make_engine("Trojan.Generic.Agent"),
            "T5": _make_engine("Trojan.Win32.Vilsel"),
            "T6": _make_engine("Trojan.Vilsel"),
            "T7": _make_engine("TROJ_VILSEL.SMB"),
            "T8": _make_engine("Trojan.Win32.Copyself"),
            "T9": _make_engine("Trojan/Win32.Vilsel"),
        },
        "Worms",  # Worms=7 >= 5 votes AND 7/9 >= 30% → Worms wins
    ),
    # 11. No match — all labels are generic with no family keyword
    (
        {
            "EngineA": _make_engine("Malware.Generic"),
            "EngineB": _make_engine("Suspicious.Cloud"),
            "EngineC": _make_engine(None),  # some engines return null
        },
        None,
    ),
    # 11. No match — all results are None/undetected
    (
        {
            "EngineA": {"category": "undetected", "result": None},
            "EngineB": {"category": "undetected", "result": None},
        },
        None,
    ),
    # 12. Case-insensitive: "TROJAN" should match "Trojan"
    (
        {
            "EngineA": _make_engine("TROJAN.Emotet"),
            "EngineB": _make_engine("trojan/zbot"),
        },
        "Trojan",
    ),
]


# ── Label extraction tests ─────────────────────────────────────────────────


class TestExtractFamily:
    """Test extract_family_from_response / extract_family_from_labels."""

    @pytest.mark.parametrize(
        "engine_results, expected_family",
        MOCK_RESPONSES,
        ids=[
            "clear_trojan",
            "clear_backdoor",
            "clear_downloader",
            "clear_worms",
            "clear_spyware",
            "clear_adware",
            "clear_dropper",
            "clear_virus",
            "mixed_weak_trojan_wins",
            "mixed_strong_worms_wins",
            "no_match_generic",
            "no_match_undetected",
            "case_insensitive",
        ],
    )
    def test_label_matching(self, engine_results, expected_family):
        """Family extraction returns the correct label for each mock."""
        result = extract_family_from_labels(engine_results)
        assert result == expected_family

    def test_empty_response(self):
        """Completely empty response returns None."""
        assert extract_family_from_response({}) is None

    def test_missing_attributes(self):
        """Response with no attributes returns None."""
        assert extract_family_from_response({"data": {}}) is None

    def test_empty_analysis_results(self):
        """Response with empty analysis_results returns None."""
        resp = _make_vt_response({})
        assert extract_family_from_response(resp) is None

    def test_returned_family_is_title_cased(self):
        """Returned family names are title-cased (e.g. 'Trojan')."""
        engines = {"E1": _make_engine("backdoor.win32.agent")}
        result = extract_family_from_labels(engines)
        assert result == "Backdoor"
        assert result[0].isupper()


# ── Cache tests ────────────────────────────────────────────────────────────


class TestCaching:
    """Cache load/save round-trip."""

    def test_save_and_load(self, tmp_path):
        """Saved cache should be loadable and identical."""
        cache_path = tmp_path / "vt_cache.json"
        original = {"abc123": {"data": "test"}, "def456": None}
        save_cache(original, cache_path)
        loaded = load_cache(cache_path)
        assert loaded == original

    def test_load_missing_file(self, tmp_path):
        """Loading from non-existent path returns empty dict."""
        cache_path = tmp_path / "nonexistent.json"
        assert load_cache(cache_path) == {}

    def test_cache_prevents_requery(self, tmp_path):
        """Second call for the same hash should use cache, not API."""
        cache_path = tmp_path / "vt_cache.json"

        # Pre-populate cache with a known response (non-generic Trojan label)
        vt_resp = _make_vt_response({"E1": _make_engine("Trojan.Emotet")})
        save_cache({"hash_abc": vt_resp}, cache_path)

        # Build a minimal set of mock samples (1 benign + 1 cached malware)
        mock_samples = [
            {"sequence": ["NtClose"], "label": cfg.BENIGN_LABEL, "sha256": "benign_sha"},
            {"sequence": ["NtClose"], "label": cfg.MALWARE_LABEL, "sha256": "hash_abc"},
        ]

        with patch("src.virustotal_labeler.load_malbehavd", return_value=mock_samples):
            with patch("src.virustotal_labeler.query_virustotal") as mock_query:
                labeled, dropped = label_malbehavd_samples(
                    api_key="fake_key",
                    cache_path=cache_path,
                    rate_limit_sleep=0,
                )
                # query_virustotal should NOT have been called (cache hit)
                mock_query.assert_not_called()

        # The malware sample should be labeled as Trojan from cache
        malware_labeled = [s for s in labeled if s["label"] != cfg.BENIGN_LABEL]
        assert len(malware_labeled) == 1
        assert malware_labeled[0]["label"] == "Trojan"


# ── Rate-limiting test ─────────────────────────────────────────────────────


class TestRateLimiting:
    """Verify that sleep is called between API requests."""

    def test_sleep_between_requests(self, tmp_path):
        """Pipeline should sleep between consecutive API calls."""
        cache_path = tmp_path / "vt_cache.json"

        # Two uncached malware samples → should trigger 2 API calls with sleep
        mock_samples = [
            {"sequence": ["NtClose"], "label": cfg.MALWARE_LABEL, "sha256": "hash_1"},
            {"sequence": ["NtClose"], "label": cfg.MALWARE_LABEL, "sha256": "hash_2"},
        ]

        vt_resp = _make_vt_response({"E1": _make_engine("Trojan.Agent")})

        with patch("src.virustotal_labeler.load_malbehavd", return_value=mock_samples):
            with patch(
                "src.virustotal_labeler.query_virustotal", return_value=vt_resp
            ):
                with patch("src.virustotal_labeler.time.sleep") as mock_sleep:
                    labeled, dropped = label_malbehavd_samples(
                        api_key="fake_key",
                        cache_path=cache_path,
                        rate_limit_sleep=15,
                    )
                    # Sleep should be called once (before the 2nd API call)
                    assert mock_sleep.call_count == 1
                    mock_sleep.assert_called_with(15)

    def test_no_sleep_before_first_request(self, tmp_path):
        """No sleep before the very first API call."""
        cache_path = tmp_path / "vt_cache.json"

        mock_samples = [
            {"sequence": ["NtClose"], "label": cfg.MALWARE_LABEL, "sha256": "hash_1"},
        ]

        vt_resp = _make_vt_response({"E1": _make_engine("Trojan.Agent")})

        with patch("src.virustotal_labeler.load_malbehavd", return_value=mock_samples):
            with patch(
                "src.virustotal_labeler.query_virustotal", return_value=vt_resp
            ):
                with patch("src.virustotal_labeler.time.sleep") as mock_sleep:
                    label_malbehavd_samples(
                        api_key="fake_key",
                        cache_path=cache_path,
                        rate_limit_sleep=15,
                    )
                    mock_sleep.assert_not_called()


# ── Dropped samples test ──────────────────────────────────────────────────


class TestDroppedSamples:
    """Unmatched hashes should be dropped and logged."""

    def test_unmatched_hashes_dropped(self, tmp_path):
        """Samples with no family match are in the dropped list."""
        cache_path = tmp_path / "vt_cache.json"

        # One sample returns a generic label with no family keyword
        mock_samples = [
            {"sequence": ["NtClose"], "label": cfg.MALWARE_LABEL, "sha256": "unmatchable"},
        ]
        vt_resp = _make_vt_response({"E1": _make_engine("Malware.Generic")})

        with patch("src.virustotal_labeler.load_malbehavd", return_value=mock_samples):
            with patch(
                "src.virustotal_labeler.query_virustotal", return_value=vt_resp
            ):
                labeled, dropped = label_malbehavd_samples(
                    api_key="fake_key",
                    cache_path=cache_path,
                    rate_limit_sleep=0,
                )

        assert "unmatchable" in dropped
        assert len(labeled) == 0

    def test_none_response_dropped(self, tmp_path):
        """Samples where VT returns None (404/error) are dropped."""
        cache_path = tmp_path / "vt_cache.json"

        mock_samples = [
            {"sequence": ["NtClose"], "label": cfg.MALWARE_LABEL, "sha256": "not_found"},
        ]

        with patch("src.virustotal_labeler.load_malbehavd", return_value=mock_samples):
            with patch(
                "src.virustotal_labeler.query_virustotal", return_value=None
            ):
                labeled, dropped = label_malbehavd_samples(
                    api_key="fake_key",
                    cache_path=cache_path,
                    rate_limit_sleep=0,
                )

        assert "not_found" in dropped
        assert len(labeled) == 0


# ── Output format test ─────────────────────────────────────────────────────


class TestOutputFormat:
    """Labeled output must match the Mal-API-2019 format."""

    def test_output_matches_mal_api_format(self, tmp_path):
        """Labeled samples have 'sequence', 'label', and 'sha256' keys."""
        cache_path = tmp_path / "vt_cache.json"

        mock_samples = [
            {"sequence": ["NtClose", "NtOpen"], "label": cfg.BENIGN_LABEL, "sha256": "benign1"},
            {"sequence": ["LdrLoadDll"], "label": cfg.MALWARE_LABEL, "sha256": "mal1"},
        ]
        vt_resp = _make_vt_response({"E1": _make_engine("Backdoor.Win32.Agent")})

        with patch("src.virustotal_labeler.load_malbehavd", return_value=mock_samples):
            with patch(
                "src.virustotal_labeler.query_virustotal", return_value=vt_resp
            ):
                labeled, dropped = label_malbehavd_samples(
                    api_key="fake_key",
                    cache_path=cache_path,
                    rate_limit_sleep=0,
                )

        assert len(labeled) == 2
        for s in labeled:
            assert "sequence" in s
            assert "label" in s
            assert isinstance(s["sequence"], list)
            assert isinstance(s["label"], str)

        # Benign keeps its label
        benign = [s for s in labeled if s["sha256"] == "benign1"]
        assert benign[0]["label"] == cfg.BENIGN_LABEL

        # Malware gets a family label
        malware = [s for s in labeled if s["sha256"] == "mal1"]
        assert malware[0]["label"] == "Backdoor"
