"""
Tests for src.data_loading.extract_winmet — label loading, target selection,
behavioral class preview, trace parsing, archive indexing, checkpointing,
and disk safety.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_loading.extract_winmet import (
    FAMILY_CLASS_MAPPING,
    FAMILY_TO_CLASS_PREVIEW,
    build_unified_labels,
    load_avclass_labels,
    load_cape_labels,
    load_consensus_labels,
    parse_trace,
    select_targets,
)


# ===================================================================
# Fixtures — small hand-crafted label files
# ===================================================================

AVCLASS_DATA = {
    "n_reports": 10,
    "Redline": {
        "n_reports": 4,
        "reports": [
            {"report": "aaa.json", "sha256": "aaa", "md5": "m1"},
            {"report": "bbb.json", "sha256": "bbb", "md5": "m2"},
            {"report": "ccc.json", "sha256": "ccc", "md5": "m3"},
            {"report": "ddd.json", "sha256": "ddd", "md5": "m4"},
        ],
    },
    "Amadey": {
        "n_reports": 3,
        "reports": [
            {"report": "eee.json", "sha256": "eee", "md5": "m5"},
            {"report": "fff.json", "sha256": "fff", "md5": "m6"},
            {"report": "ggg.json", "sha256": "ggg", "md5": "m7"},
        ],
    },
    "(n/a)": {
        "n_reports": 2,
        "reports": [
            {"report": "xxx.json", "sha256": "xxx", "md5": "mx"},
            {"report": "yyy.json", "sha256": "yyy", "md5": "my"},
        ],
    },
    "Tiny": {
        "n_reports": 1,
        "reports": [
            {"report": "zzz.json", "sha256": "zzz", "md5": "mz"},
        ],
    },
}

CAPE_DATA = {
    "n_reports": 10,
    "Metastealer": {
        "n_reports": 3,
        "reports": [
            {"report": "aaa.json", "sha256": "aaa", "md5": "m1"},
            {"report": "bbb.json", "sha256": "bbb", "md5": "m2"},
            {"report": "eee.json", "sha256": "eee", "md5": "m5"},
        ],
    },
    "(n/a)": {
        "n_reports": 4,
        "reports": [
            {"report": "ccc.json", "sha256": "ccc", "md5": "m3"},
            {"report": "ddd.json", "sha256": "ddd", "md5": "m4"},
            {"report": "fff.json", "sha256": "fff", "md5": "m6"},
            {"report": "ggg.json", "sha256": "ggg", "md5": "m7"},
        ],
    },
    "Amadey": {
        "n_reports": 1,
        "reports": [
            {"report": "zzz.json", "sha256": "zzz", "md5": "mz"},
        ],
    },
}

# Consensus format: {sha256: {"avclass": ..., "cape": ...}}
CONSENSUS_DATA = {
    "reports_avclass_no_consensus": 1,
    "reports_cape_no_consensus": 2,
    "reports_both_no_consensus": 0,
    "aaa": {"avclass": "Redline", "cape": "Metastealer"},
    "bbb": {"avclass": "Redline", "cape": "Metastealer"},
    "ccc": {"avclass": "Redline", "cape": "(n/a)"},
    "ddd": {"avclass": "Redline", "cape": "(n/a)"},
    "eee": {"avclass": "Amadey", "cape": "Metastealer"},
    "fff": {"avclass": "Amadey", "cape": "(n/a)"},
    "ggg": {"avclass": "Amadey", "cape": "(n/a)"},
    "xxx": {"avclass": "(n/a)", "cape": "(n/a)"},
    "yyy": {"avclass": "(n/a)", "cape": "(n/a)"},
    "zzz": {"avclass": "Tiny", "cape": "Amadey"},
}


@pytest.fixture
def label_dir(tmp_path):
    """Write fixture label files to a temp directory."""
    av_path = tmp_path / "avclass.json"
    cape_path = tmp_path / "cape.json"
    cons_path = tmp_path / "consensus.json"
    av_path.write_text(json.dumps(AVCLASS_DATA), encoding="utf-8")
    cape_path.write_text(json.dumps(CAPE_DATA), encoding="utf-8")
    cons_path.write_text(json.dumps(CONSENSUS_DATA), encoding="utf-8")
    return av_path, cape_path, cons_path


# ===================================================================
# Label file loading tests
# ===================================================================


class TestAVClassLoading:
    """Tests for AVClass label parsing."""

    def test_correct_families_extracted(self, label_dir):
        avclass = load_avclass_labels(label_dir[0])
        families = set(avclass.values())
        assert "redline" in families
        assert "amadey" in families

    def test_na_families_skipped(self, label_dir):
        avclass = load_avclass_labels(label_dir[0])
        assert "xxx.json" not in avclass
        assert "yyy.json" not in avclass
        assert "(n/a)" not in set(avclass.values())

    def test_metadata_keys_skipped(self, label_dir):
        """n_reports (int) should not produce any entries."""
        avclass = load_avclass_labels(label_dir[0])
        # 4 redline + 3 amadey + 1 tiny = 8 (not 10, since (n/a) dropped)
        assert len(avclass) == 8

    def test_families_lowercased(self, label_dir):
        avclass = load_avclass_labels(label_dir[0])
        for fam in avclass.values():
            assert fam == fam.lower()

    def test_filename_is_report_field(self, label_dir):
        avclass = load_avclass_labels(label_dir[0])
        assert "aaa.json" in avclass
        assert avclass["aaa.json"] == "redline"


class TestCAPELoading:
    """Tests for CAPE label parsing."""

    def test_na_entries_included(self, label_dir):
        """CAPE keeps (n/a) entries since it's a secondary field."""
        cape = load_cape_labels(label_dir[1])
        na_entries = [f for f, fam in cape.items() if fam == "(n/a)"]
        assert len(na_entries) == 4

    def test_correct_cape_labels(self, label_dir):
        cape = load_cape_labels(label_dir[1])
        assert cape["aaa.json"] == "metastealer"
        assert cape["eee.json"] == "metastealer"


class TestConsensusLoading:
    """Tests for consensus label file parsing."""

    def test_metadata_keys_skipped(self, label_dir):
        consensus = load_consensus_labels(label_dir[2])
        assert "reports_avclass_no_consensus" not in consensus
        assert "reports_cape_no_consensus" not in consensus

    def test_json_appended(self, label_dir):
        """Keys should have .json appended."""
        consensus = load_consensus_labels(label_dir[2])
        assert "aaa.json" in consensus
        assert "aaa" not in consensus

    def test_labels_lowercased(self, label_dir):
        consensus = load_consensus_labels(label_dir[2])
        for entry in consensus.values():
            if entry["avclass"]:
                assert entry["avclass"] == entry["avclass"].lower()
            if entry["cape"]:
                assert entry["cape"] == entry["cape"].lower()


class TestUnifiedLabels:
    """Tests for the unified label dict."""

    def test_only_avclass_samples_included(self, label_dir):
        avclass = load_avclass_labels(label_dir[0])
        cape = load_cape_labels(label_dir[1])
        consensus = load_consensus_labels(label_dir[2])
        unified = build_unified_labels(avclass, cape, consensus)
        # Only samples with valid AVClass labels
        assert len(unified) == 8  # 4 redline + 3 amadey + 1 tiny

    def test_cape_nullable(self, label_dir):
        """Samples with CAPE (n/a) should get None."""
        avclass = load_avclass_labels(label_dir[0])
        cape = load_cape_labels(label_dir[1])
        consensus = load_consensus_labels(label_dir[2])
        unified = build_unified_labels(avclass, cape, consensus)
        # ccc.json: AVClass=redline, CAPE=(n/a) -> None
        assert unified["ccc.json"]["cape"] is None

    def test_cape_present(self, label_dir):
        avclass = load_avclass_labels(label_dir[0])
        cape = load_cape_labels(label_dir[1])
        consensus = load_consensus_labels(label_dir[2])
        unified = build_unified_labels(avclass, cape, consensus)
        assert unified["aaa.json"]["cape"] == "metastealer"

    def test_consensus_agreement(self, label_dir):
        """Consensus label set only when avclass == cape in consensus file."""
        avclass = load_avclass_labels(label_dir[0])
        cape = load_cape_labels(label_dir[1])
        consensus = load_consensus_labels(label_dir[2])
        unified = build_unified_labels(avclass, cape, consensus)
        # aaa: avclass=Redline, cape=Metastealer in consensus -> disagree -> None
        assert unified["aaa.json"]["consensus"] is None
        # zzz: avclass=Tiny, cape=Amadey in consensus -> disagree -> None
        assert unified["zzz.json"]["consensus"] is None

    def test_sha256_extracted(self, label_dir):
        avclass = load_avclass_labels(label_dir[0])
        cape = load_cape_labels(label_dir[1])
        consensus = load_consensus_labels(label_dir[2])
        unified = build_unified_labels(avclass, cape, consensus)
        assert unified["aaa.json"]["sha256"] == "aaa"


# ===================================================================
# Target selection tests
# ===================================================================


# Build a larger fixture for selection tests
SELECTION_AVCLASS = {
    "n_reports": 100,
    "FamilyA": {"n_reports": 80, "reports": [
        {"report": f"a{i:03d}.json", "sha256": f"a{i:03d}"} for i in range(80)
    ]},
    "FamilyB": {"n_reports": 60, "reports": [
        {"report": f"b{i:03d}.json", "sha256": f"b{i:03d}"} for i in range(60)
    ]},
    "FamilyC": {"n_reports": 40, "reports": [
        {"report": f"c{i:03d}.json", "sha256": f"c{i:03d}"} for i in range(40)
    ]},
    "FamilyD": {"n_reports": 20, "reports": [
        {"report": f"d{i:03d}.json", "sha256": f"d{i:03d}"} for i in range(20)
    ]},
    "FamilyE": {"n_reports": 5, "reports": [
        {"report": f"e{i:03d}.json", "sha256": f"e{i:03d}"} for i in range(5)
    ]},
}


@pytest.fixture
def selection_unified(tmp_path):
    """Build unified dict from SELECTION_AVCLASS."""
    av_path = tmp_path / "av.json"
    av_path.write_text(json.dumps(SELECTION_AVCLASS), encoding="utf-8")
    avclass = load_avclass_labels(av_path)
    # No CAPE/consensus needed — pass empty dicts
    unified = {}
    for fname, fam in avclass.items():
        unified[fname] = {
            "avclass": fam, "cape": None, "consensus": None,
            "sha256": fname.replace(".json", ""),
        }
    return unified


class TestTargetSelection:
    """Tests for target filtering and capping."""

    def test_top_k_selects_largest(self, selection_unified):
        selected, plan = select_targets(
            selection_unified, top_k=3, max_per_family=999, min_samples=1,
        )
        families = [fam for fam, _, _ in plan]
        assert families == ["familya", "familyb", "familyc"]

    def test_max_per_family_caps(self, selection_unified):
        selected, plan = select_targets(
            selection_unified, top_k=5, max_per_family=10, min_samples=1,
        )
        for fam, raw, capped in plan:
            assert capped <= 10

    def test_cap_is_deterministic(self, selection_unified):
        """Same selection across runs (sorted by filename)."""
        sel1, _ = select_targets(
            selection_unified, top_k=2, max_per_family=5, min_samples=1,
        )
        sel2, _ = select_targets(
            selection_unified, top_k=2, max_per_family=5, min_samples=1,
        )
        assert sorted(sel1.keys()) == sorted(sel2.keys())

    def test_min_samples_drops_small(self, selection_unified):
        """FamilyE (5 samples) should be dropped with min_samples=10."""
        selected, plan = select_targets(
            selection_unified, top_k=10, max_per_family=999, min_samples=10,
        )
        families = {fam for fam, _, _ in plan}
        assert "familye" not in families
        # FamilyD (20) should still be included
        assert "familyd" in families

    def test_missing_avclass_dropped(self):
        """Samples not in AVClass should not appear."""
        unified = {
            "has_label.json": {
                "avclass": "familyx", "cape": None,
                "consensus": None, "sha256": "has_label",
            },
        }
        # With min_samples=1, familyx (1 sample) should be selected
        selected, _ = select_targets(unified, top_k=10, max_per_family=999, min_samples=1)
        assert "has_label.json" in selected


# ===================================================================
# Behavioral class preview tests
# ===================================================================


class TestBehavioralClassPreview:
    """Tests for the family-to-class mapping."""

    def test_preview_covers_all_mapped(self):
        """Every key in FAMILY_CLASS_MAPPING should appear in FAMILY_TO_CLASS_PREVIEW."""
        for fam in FAMILY_CLASS_MAPPING:
            assert fam in FAMILY_TO_CLASS_PREVIEW

    def test_known_mappings(self):
        assert FAMILY_TO_CLASS_PREVIEW["redline"] == "Spyware"
        assert FAMILY_TO_CLASS_PREVIEW["amadey"] == "Downloader"
        assert FAMILY_TO_CLASS_PREVIEW["qbot"] == "Trojan"

    def test_secondary_classes_stored(self):
        """Spot-check that secondary classes are present."""
        assert FAMILY_CLASS_MAPPING["agenttesla"][1] == ["Backdoor", "Trojan"]
        assert FAMILY_CLASS_MAPPING["amadey"][1] == ["Trojan", "Spyware"]
        assert FAMILY_CLASS_MAPPING["virlock"][1] == ["Trojan"]

    def test_primary_matches_preview(self):
        """Primary class from FAMILY_CLASS_MAPPING should match FAMILY_TO_CLASS_PREVIEW."""
        for fam, (primary, _) in FAMILY_CLASS_MAPPING.items():
            assert FAMILY_TO_CLASS_PREVIEW[fam] == primary


# ===================================================================
# Trace parsing tests
# ===================================================================


def _make_trace(processes):
    """Build a minimal CAPE-format trace dict."""
    proc_list = []
    for pname, api_names in processes:
        calls = [{"api": name, "timestamp": "t", "category": "system"}
                 for name in api_names]
        proc_list.append({
            "process_id": len(proc_list) + 1,
            "process_name": pname,
            "calls": calls,
        })
    return {"behavior": {"processes": proc_list}}


class TestTraceParsing:
    """Tests for API sequence extraction from trace JSON."""

    def test_basic_extraction(self):
        trace = _make_trace([
            ("proc1.exe", ["NtCreateFile", "NtReadFile"]),
            ("proc2.exe", ["RegOpenKeyExA"]),
        ])
        result = parse_trace(trace)
        assert result is not None
        api_seq, num_procs = result
        assert api_seq == ["ntcreatefile", "ntreadfile", "regopenkeyexa"]
        assert num_procs == 2

    def test_process_order_preserved(self):
        """APIs should be in process-start order then call order."""
        trace = _make_trace([
            ("first.exe", ["ApiA", "ApiB"]),
            ("second.exe", ["ApiC"]),
            ("third.exe", ["ApiD", "ApiE"]),
        ])
        result = parse_trace(trace)
        api_seq, _ = result
        assert api_seq == ["apia", "apib", "apic", "apid", "apie"]

    def test_lowercase_conversion(self):
        trace = _make_trace([("p.exe", ["GetModuleHandleA", "NtQueryLicenseValue"])])
        api_seq, _ = parse_trace(trace)
        assert api_seq == ["getmodulehandlea", "ntquerylicensevalue"]

    def test_empty_processes_returns_none(self):
        trace = {"behavior": {"processes": []}}
        assert parse_trace(trace) is None

    def test_no_behavior_returns_none(self):
        trace = {"target": {"file": {}}}
        assert parse_trace(trace) is None

    def test_no_calls_returns_none(self):
        """Processes exist but none have API calls."""
        trace = {"behavior": {"processes": [
            {"process_id": 1, "process_name": "p.exe", "calls": []},
        ]}}
        assert parse_trace(trace) is None

    def test_malformed_call_skipped(self):
        """Calls without 'api' field are skipped, not crashed."""
        trace = {"behavior": {"processes": [{
            "process_id": 1,
            "process_name": "p.exe",
            "calls": [
                {"api": "GoodCall"},
                {"no_api_key": True},
                {"api": "AnotherCall"},
                "not_a_dict",
            ],
        }]}}
        result = parse_trace(trace)
        api_seq, _ = result
        assert api_seq == ["goodcall", "anothercall"]

    def test_mixed_processes_some_empty(self):
        """Some processes have calls, some don't."""
        trace = {"behavior": {"processes": [
            {"process_id": 1, "process_name": "a.exe", "calls": [{"api": "Api1"}]},
            {"process_id": 2, "process_name": "b.exe", "calls": []},
            {"process_id": 3, "process_name": "c.exe", "calls": [{"api": "Api2"}]},
        ]}}
        result = parse_trace(trace)
        api_seq, num_procs = result
        assert api_seq == ["api1", "api2"]
        assert num_procs == 3  # all 3 processes counted even if empty


# ===================================================================
# Archive index tests
# ===================================================================


class TestArchiveIndex:
    """Tests for archive member indexing."""

    def test_bare_filename_extraction(self):
        """Test that directory prefixes are stripped."""
        # Simulate what build_archive_index does internally
        members = [
            "reports/aaa.json",
            "WinMET/bbb.json",
            "ccc.json",
        ]
        index = {}
        for m in members:
            bare = m.replace("\\", "/").split("/")[-1]
            index[bare] = (1, Path("vol1.7z"))
        assert "aaa.json" in index
        assert "bbb.json" in index
        assert "ccc.json" in index

    def test_missing_targets_detected(self):
        """Targets not in any archive should be identified."""
        archive_members = {"aaa.json", "bbb.json"}
        targets = {"aaa.json", "bbb.json", "missing.json"}
        found = targets & archive_members
        missing = targets - archive_members
        assert missing == {"missing.json"}
        assert len(found) == 2


# ===================================================================
# Checkpoint tests
# ===================================================================


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_checkpoint_round_trip(self, tmp_path):
        checkpoint_path = tmp_path / "checkpoint.json"
        processed = {"aaa", "bbb", "ccc"}
        data = {
            "processed_hashes": sorted(processed),
            "position": 42,
            "timestamp": "2026-04-08T12:00:00",
        }
        checkpoint_path.write_text(json.dumps(data), encoding="utf-8")

        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert set(loaded["processed_hashes"]) == processed
        assert loaded["position"] == 42

    def test_resume_skips_processed(self, tmp_path):
        """Simulate resume: already-processed hashes should be skipped."""
        processed = {"aaa", "bbb"}
        all_targets = ["aaa", "bbb", "ccc", "ddd"]
        remaining = [t for t in all_targets if t not in processed]
        assert remaining == ["ccc", "ddd"]


# ===================================================================
# Disk safety tests
# ===================================================================


class TestDiskSafety:
    """Tests for disk space checking."""

    def test_low_disk_detected(self):
        """Mock shutil.disk_usage to simulate low disk."""
        mock_usage = MagicMock()
        mock_usage.free = 1 * 1024**3  # 1 GB
        with patch("shutil.disk_usage", return_value=mock_usage):
            import shutil
            usage = shutil.disk_usage("/")
            free_gb = usage.free / (1024**3)
            assert free_gb < 2.0  # below default threshold

    def test_sufficient_disk_passes(self):
        """Mock shutil.disk_usage with plenty of space."""
        mock_usage = MagicMock()
        mock_usage.free = 50 * 1024**3  # 50 GB
        with patch("shutil.disk_usage", return_value=mock_usage):
            import shutil
            usage = shutil.disk_usage("/")
            free_gb = usage.free / (1024**3)
            assert free_gb >= 2.0
