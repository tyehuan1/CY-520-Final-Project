#!/usr/bin/env python
"""
WinMET Extraction Script — Encrypted 7z Volumes to Parquet.

Converts the WinMET dataset from password-protected 7z archives + separate
label files into a single Parquet file containing API call sequences and
family labels.  Designed for cross-dataset generalizability evaluation:
WinMET is a test set only (models are trained on Mal-API-2019).

Usage::

    python src/data_loading/extract_winmet.py --plan          # show plan
    python src/data_loading/extract_winmet.py --probe         # inspect trace
    python src/data_loading/extract_winmet.py --yes           # full extraction
    python src/data_loading/extract_winmet.py --resume --yes  # resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WINMET_DATA_DIR = PROJECT_ROOT / "data" / "WinMET" / "Data"
WINMET_OUTPUT_DIR = PROJECT_ROOT / "data" / "winmet"

AVCLASS_PATH = WINMET_DATA_DIR / "avclass_report_to_label_mapping.json"
CAPE_PATH = WINMET_DATA_DIR / "cape_report_to_label_mapping.json"
CONSENSUS_PATH = WINMET_DATA_DIR / "reports_consensus_label.json"

ARCHIVE_PATTERN = "WinMET_volume_{vol}.7z"
NUM_VOLUMES = 5
DEFAULT_PASSWORD = "infected"

PLAN_PATH = WINMET_OUTPUT_DIR / ".extraction_plan.json"
CHECKPOINT_PATH = WINMET_OUTPUT_DIR / ".extraction_checkpoint.json"
ARCHIVE_INDEX_PATH = WINMET_OUTPUT_DIR / ".archive_index.json"
LOG_PATH = WINMET_OUTPUT_DIR / ".extraction.log"

# ---------------------------------------------------------------------------
# 7z CLI detection
# ---------------------------------------------------------------------------
_7Z_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def find_7z() -> Optional[str]:
    """Return path to 7z CLI binary, or None."""
    path = shutil.which("7z")
    if path:
        return path
    for candidate in _7Z_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Behavioral class mapping: WinMET family -> Mal-API-2019 class
# ---------------------------------------------------------------------------
# Each entry: (primary_class, [secondary, ...])
# Primary is used for generalizability evaluation.  Secondaries are stored
# in the Parquet output for post-hoc misclassification analysis (i.e., did
# the model confuse the sample with one of its alternative classes?).
#
# Only the primary class is used in FAMILY_TO_CLASS_PREVIEW for the plan
# display.  The full mapping is written to the Parquet file.
FAMILY_CLASS_MAPPING: Dict[str, Tuple[str, List[str]]] = {
    # ── Spyware / Infostealers ──
    "redline":      ("Spyware",     []),
    "berbew":       ("Spyware",     ["Backdoor"]),
    "agenttesla":   ("Spyware",     ["Backdoor", "Trojan"]),
    "noon":         ("Spyware",     ["Trojan"]),
    "strab":        ("Spyware",     ["Trojan"]),
    "snakelogger":  ("Spyware",     ["Trojan"]),
    # Extra spyware families (not in top-25 but useful for broader mapping)
    "lokibot":      ("Spyware",     []),
    "loki":         ("Spyware",     []),
    "azorult":      ("Spyware",     []),
    "formbook":     ("Spyware",     []),
    "vidar":        ("Spyware",     []),
    "raccoon":      ("Spyware",     []),
    "metastealer":  ("Spyware",     []),
    "stealc":       ("Spyware",     []),
    "risepro":      ("Spyware",     []),
    "lumma":        ("Spyware",     []),
    "pony":         ("Spyware",     []),
    "arkei":        ("Spyware",     []),
    # ── Downloaders / Loaders ──
    "amadey":       ("Downloader",  ["Trojan", "Spyware"]),
    "snojan":       ("Downloader",  ["Trojan"]),
    "dofoil":       ("Downloader",  ["Trojan"]),
    "gcleaner":     ("Downloader",  ["Trojan"]),
    "sload":        ("Downloader",  ["Spyware"]),
    "deyma":        ("Downloader",  ["Trojan", "Backdoor"]),
    # Extra downloader families
    "smokeloader":  ("Downloader",  []),
    "guloader":     ("Downloader",  []),
    "privacyloader":("Downloader",  []),
    "gozi":         ("Downloader",  []),
    "icedid":       ("Downloader",  []),
    "pikabot":      ("Downloader",  []),
    "bumblebee":    ("Downloader",  []),
    # ── Trojans ──
    "cosmu":        ("Worms",       ["Trojan"]),
    "taskun":       ("Trojan",      ["Dropper"]),
    "disabler":     ("Trojan",      ["Dropper"]),
    "stop":         ("Trojan",      ["Downloader"]),
    "qbot":         ("Trojan",      ["Spyware", "Worms"]),
    "blihan":       ("Trojan",      ["Spyware"]),
    # Extra trojan families
    "nanocore":     ("Trojan",      []),
    "asyncrat":     ("Trojan",      []),
    "remcos":       ("Trojan",      []),
    "emotet":       ("Trojan",      []),
    "trickbot":     ("Trojan",      []),
    "darkcomet":    ("Trojan",      []),
    "xworm":        ("Trojan",      []),
    "limerat":      ("Trojan",      []),
    "netwire":      ("Trojan",      []),
    # ── Backdoors ──
    "equationdrug": ("Backdoor",    ["Trojan", "Spyware"]),
    "mokes":        ("Backdoor",    ["Spyware"]),
    "bladabindi":   ("Backdoor",    ["Trojan"]),
    "makoob":       ("Backdoor",    ["Trojan", "Spyware"]),
    # Extra backdoor families
    "cobalt":       ("Backdoor",    []),
    "cobaltstrike": ("Backdoor",    []),
    # ── Worms ──
    "vbclone":      ("Worms",       ["Trojan"]),
    "gamarue":      ("Worms",       ["Downloader", "Backdoor"]),
    # ── Adware ──
    "installcore":  ("Adware",      []),
    "firseria":     ("Adware",      []),
    # ── Dropper ──
    "ursnif":       ("Dropper",     []),
    "dridex":       ("Dropper",     []),
    "zloader":      ("Dropper",     []),
    # ── Virus ──
    "virlock":      ("Virus",       ["Trojan"]),
    "virut":        ("Virus",       []),
    "ramnit":       ("Virus",       []),
    "sality":       ("Virus",       []),
    "parite":       ("Virus",       []),
}

# Flat primary-only mapping used for the plan preview display
FAMILY_TO_CLASS_PREVIEW: Dict[str, str] = {
    fam: cls for fam, (cls, _) in FAMILY_CLASS_MAPPING.items()
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("extract_winmet")


def setup_logging() -> None:
    """Configure logging to stdout + file."""
    WINMET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)


# ===================================================================
# Label Loading
# ===================================================================


def load_avclass_labels(path: Path) -> Dict[str, str]:
    """Load AVClass label mapping: {filename.json: family_lowercase}.

    Skips metadata keys and samples labeled ``(n/a)`` / empty.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels: Dict[str, str] = {}
    for key, val in data.items():
        if not isinstance(val, dict) or "reports" not in val:
            continue  # skip metadata like n_reports
        family = key.strip().lower()
        if family in ("(n/a)", "n/a", "none", ""):
            continue
        for report in val["reports"]:
            fname = report["report"]
            labels[fname] = family
    return labels


def load_cape_labels(path: Path) -> Dict[str, str]:
    """Load CAPE label mapping: {filename.json: family_lowercase}.

    Same nested-by-family format as AVClass.  (n/a) entries included
    as-is since CAPE is a secondary field.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels: Dict[str, str] = {}
    for key, val in data.items():
        if not isinstance(val, dict) or "reports" not in val:
            continue
        family = key.strip().lower()
        for report in val["reports"]:
            fname = report["report"]
            labels[fname] = family
    return labels


def load_consensus_labels(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """Load consensus file: {filename.json: {"avclass": ..., "cape": ...}}.

    Keys in the file are bare SHA256; we append ``.json`` to match archive
    member names.  Values are dicts with ``avclass`` and ``cape`` fields.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta_keys = {
        "reports_avclass_no_consensus",
        "reports_cape_no_consensus",
        "reports_both_no_consensus",
    }
    labels: Dict[str, Dict[str, Optional[str]]] = {}
    for key, val in data.items():
        if key in meta_keys:
            continue
        if isinstance(val, dict):
            fname = key + ".json"
            labels[fname] = {
                "avclass": (val.get("avclass") or "").strip().lower() or None,
                "cape": (val.get("cape") or "").strip().lower() or None,
            }
    return labels


def build_unified_labels(
    avclass: Dict[str, str],
    cape: Dict[str, str],
    consensus: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, Dict[str, Any]]:
    """Build per-sample label dict keyed on AVClass membership.

    Only samples with a valid AVClass label are included.

    Returns:
        ``{filename: {"avclass": str, "cape": str|None,
        "consensus": str|None, "sha256": str}}``
    """
    unified: Dict[str, Dict[str, Any]] = {}
    for fname, av_family in avclass.items():
        sha256 = fname.replace(".json", "")

        # CAPE label (nullable)
        cape_label = cape.get(fname)
        if cape_label and cape_label in ("(n/a)", "n/a", "none"):
            cape_label = None

        # Consensus label: if both avclass and cape agree in consensus file
        cons = consensus.get(fname, {})
        cons_av = cons.get("avclass")
        cons_cape = cons.get("cape")
        if cons_av and cons_cape and cons_av == cons_cape:
            consensus_label = cons_av
        else:
            consensus_label = None

        unified[fname] = {
            "avclass": av_family,
            "cape": cape_label,
            "consensus": consensus_label,
            "sha256": sha256,
        }

    return unified


# ===================================================================
# Target Selection
# ===================================================================


def select_targets(
    unified: Dict[str, Dict[str, Any]],
    top_k: int = 25,
    max_per_family: int = 300,
    min_samples: int = 50,
) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, int, int]]]:
    """Apply family filtering and per-family caps.

    Args:
        unified: Full label dict from ``build_unified_labels``.
        top_k: Keep only the top K families by AVClass count.
        max_per_family: Cap each family at N samples.
        min_samples: Drop families with fewer than this many samples.

    Returns:
        (selected_targets, plan_rows) where plan_rows is a list of
        ``(family, raw_count, capped_count)`` tuples sorted descending.
    """
    # Count samples per AVClass family
    family_counts: Counter = Counter()
    family_files: Dict[str, List[str]] = {}
    for fname, info in unified.items():
        fam = info["avclass"]
        family_counts[fam] += 1
        family_files.setdefault(fam, []).append(fname)

    # Sort families by count descending, take top K
    sorted_families = family_counts.most_common()
    selected_families = []
    for fam, count in sorted_families:
        if len(selected_families) >= top_k:
            break
        if count < min_samples:
            continue
        selected_families.append((fam, count))

    # Apply cap and build target set
    selected: Dict[str, Dict[str, Any]] = {}
    plan_rows: List[Tuple[str, int, int]] = []

    for fam, raw_count in selected_families:
        capped = min(raw_count, max_per_family)
        # Deterministic selection: sort by filename, take first N
        files_sorted = sorted(family_files[fam])[:capped]
        for fname in files_sorted:
            selected[fname] = unified[fname]
        plan_rows.append((fam, raw_count, capped))

    return selected, plan_rows


# ===================================================================
# Plan Display
# ===================================================================


def print_plan(
    plan_rows: List[Tuple[str, int, int]],
    selected: Dict[str, Dict[str, Any]],
    top_k: int,
    max_per_family: int,
    min_samples: int,
) -> None:
    """Print the extraction plan table and behavioral class preview."""
    total_raw = sum(r for _, r, _ in plan_rows)
    total_capped = sum(c for _, _, c in plan_rows)

    print()
    print("=" * 65)
    print("  WinMET Extraction Plan")
    print("=" * 65)
    print(f"  Settings: top_k={top_k}, max_per_family={max_per_family}, "
          f"min_samples={min_samples}")
    print()
    print(f"  {'Family':<25} {'Raw Count':>10} {'Capped':>10}")
    print(f"  {'-' * 47}")
    for fam, raw, capped in plan_rows:
        marker = " *" if capped < raw else ""
        print(f"  {fam:<25} {raw:>10,} {capped:>10,}{marker}")
    print(f"  {'-' * 47}")
    print(f"  {'TOTAL':<25} {total_raw:>10,} {total_capped:>10,}")
    print()
    print(f"  Distinct families selected: {len(plan_rows)}")
    print(f"  Total files to extract: {total_capped:,}")
    if len(plan_rows) >= top_k:
        rank_k_count = plan_rows[top_k - 1][1]
        print(f"  Count at rank {top_k}: {rank_k_count:,}")
    time_est = total_capped * 0.2
    print(f"  Estimated time: ~{time_est / 60:.0f} minutes "
          f"(at 0.2s/file)")

    # Behavioral class preview
    print()
    print("  --- Behavioral Class Preview (Mal-API-2019 mapping) ---")
    class_counts: Counter = Counter()
    unmapped_families: List[Tuple[str, int]] = []
    for fam, raw, capped in plan_rows:
        cls = FAMILY_TO_CLASS_PREVIEW.get(fam)
        if cls:
            class_counts[cls] += capped
        else:
            unmapped_families.append((fam, capped))
            class_counts["unmapped"] += capped

    malapi_classes = [
        "Trojan", "Backdoor", "Downloader", "Worms",
        "Spyware", "Adware", "Dropper", "Virus",
    ]
    print(f"  {'Class':<15} {'Projected Samples':>18}")
    print(f"  {'-' * 35}")
    for cls in malapi_classes:
        count = class_counts.get(cls, 0)
        flag = "  << ZERO" if count == 0 else ""
        print(f"  {cls:<15} {count:>18,}{flag}")
    print(f"  {'unmapped':<15} {class_counts.get('unmapped', 0):>18,}")
    print()

    if unmapped_families:
        print(f"  Unmapped families ({len(unmapped_families)}):")
        for fam, cnt in unmapped_families:
            print(f"    {fam:<25} {cnt:>5}")
    print()


def save_plan(
    plan_rows: List[Tuple[str, int, int]],
    selected: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Save extraction plan to disk for resume capability."""
    WINMET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = {
        "settings": {
            "top_k_families": args.top_k_families,
            "max_per_family": args.max_per_family,
            "min_samples_per_family": args.min_samples_per_family,
        },
        "families": [
            {"family": fam, "raw_count": raw, "capped_count": capped}
            for fam, raw, capped in plan_rows
        ],
        "target_filenames": sorted(selected.keys()),
        "total_targets": len(selected),
    }
    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    logger.info("Plan saved to %s", PLAN_PATH)


# ===================================================================
# Archive Indexing
# ===================================================================


def build_archive_index(password: str = DEFAULT_PASSWORD) -> Dict[str, Tuple[int, Path]]:
    """Build {bare_filename: (volume_number, archive_path)} index.

    Lists all members of each 7z volume and maps bare filenames to their
    archive location.  Caches the result to ``ARCHIVE_INDEX_PATH``.
    """
    if ARCHIVE_INDEX_PATH.exists():
        logger.info("Loading cached archive index from %s", ARCHIVE_INDEX_PATH)
        with open(ARCHIVE_INDEX_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Convert back from JSON: {fname: [vol, path_str]}
        return {fname: (info[0], Path(info[1])) for fname, info in raw.items()}

    import py7zr

    logger.info("Building archive index across %d volumes...", NUM_VOLUMES)
    index: Dict[str, Tuple[int, Path]] = {}

    for vol in range(1, NUM_VOLUMES + 1):
        archive_path = WINMET_DATA_DIR / ARCHIVE_PATTERN.format(vol=vol)
        if not archive_path.exists():
            logger.error("Archive not found: %s", archive_path)
            sys.exit(1)

        with py7zr.SevenZipFile(str(archive_path), mode="r", password=password) as archive:
            members = archive.getnames()

        count = 0
        for member in members:
            # Handle possible directory prefixes — extract bare filename
            bare = member.replace("\\", "/").split("/")[-1]
            if bare in index:
                logger.warning("Duplicate member %s (vol %d vs vol %d)",
                               bare, index[bare][0], vol)
            index[bare] = (vol, archive_path)
            count += 1

        logger.info("  Volume %d: %d members indexed.", vol, count)

    logger.info("Total archive index: %d unique files.", len(index))

    # Cache to disk
    WINMET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {fname: [vol, str(path)] for fname, (vol, path) in index.items()}
    with open(ARCHIVE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f)
    logger.info("Archive index cached to %s", ARCHIVE_INDEX_PATH)

    return index


# ===================================================================
# Trace Probe (Phase 5)
# ===================================================================


def extract_one_file_7z(
    archive_path: Path,
    member_name: str,
    z7_path: str,
    password: str,
) -> bytes:
    """Extract a single member from a 7z archive using the 7z CLI.

    Streams the file through a temp file to avoid holding large traces
    entirely in memory via subprocess.communicate().
    """
    cmd = [
        z7_path, "e", "-so",
        f"-p{password}",
        str(archive_path),
        member_name,
    ]
    with tempfile.TemporaryFile() as tmp:
        proc = subprocess.Popen(cmd, stdout=tmp, stderr=subprocess.PIPE)
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)
        tmp.seek(0)
        return tmp.read()


def extract_one_file_py7zr(
    archive_path: Path,
    member_name: str,
    password: str,
) -> bytes:
    """Extract a single member using py7zr (fallback)."""
    import py7zr

    with py7zr.SevenZipFile(str(archive_path), mode="r", password=password) as archive:
        # py7zr extract returns {name: BytesIO} dict
        archive.reset()
        extracted = archive.read(targets=[member_name])
        if not extracted:
            raise FileNotFoundError(f"{member_name} not found in {archive_path}")
        # Return first (only) value
        bio = next(iter(extracted.values()))
        return bio.read()


def extract_one_file(
    archive_path: Path,
    member_name: str,
    z7_path: Optional[str],
    password: str,
) -> bytes:
    """Extract a single file using 7z CLI if available, else py7zr."""
    if z7_path:
        return extract_one_file_7z(archive_path, member_name, z7_path, password)
    return extract_one_file_py7zr(archive_path, member_name, password)


# ===================================================================
# Trace Parsing (Phase 6)
# ===================================================================


def parse_trace(trace: Dict[str, Any]) -> Optional[Tuple[List[str], int]]:
    """Extract API call sequence from a CAPE/MALVADA trace JSON.

    Walks ``behavior.processes`` in order, then each process's ``calls``
    in order, collecting the ``api`` field as a lowercase string.

    Args:
        trace: Parsed JSON dict from a WinMET trace file.

    Returns:
        ``(api_sequence, num_processes)`` or ``None`` if the trace has
        no processes or no API calls.
    """
    behavior = trace.get("behavior")
    if not isinstance(behavior, dict):
        return None

    processes = behavior.get("processes")
    if not isinstance(processes, list) or not processes:
        return None

    api_sequence: List[str] = []
    num_processes = len(processes)

    for proc in processes:
        if not isinstance(proc, dict):
            continue
        calls = proc.get("calls", [])
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            api_name = call.get("api")
            if api_name and isinstance(api_name, str):
                api_sequence.append(api_name.lower())

    if not api_sequence:
        return None

    return api_sequence, num_processes


def _print_json_structure(obj: Any, prefix: str = "", depth: int = 0, max_depth: int = 3) -> None:
    """Recursively print JSON key structure up to max_depth."""
    if depth >= max_depth:
        return

    if isinstance(obj, dict):
        for key in list(obj.keys())[:20]:  # cap keys shown
            val = obj[key]
            type_name = type(val).__name__
            if isinstance(val, str):
                preview = repr(val[:80])
            elif isinstance(val, (int, float, bool)):
                preview = repr(val)
            elif isinstance(val, list):
                preview = f"list[{len(val)}]"
                if val and isinstance(val[0], (str, int, float)):
                    preview += f" first={repr(val[0])}" if len(str(val[0])) < 60 else ""
            elif isinstance(val, dict):
                preview = f"dict[{len(val)} keys]"
            else:
                preview = type_name
            print(f"  {prefix}{key}: {preview}")
            if isinstance(val, dict):
                _print_json_structure(val, prefix + "  ", depth + 1, max_depth)
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                print(f"  {prefix}  [0]:")
                _print_json_structure(val[0], prefix + "    ", depth + 1, max_depth)
        if len(obj) > 20:
            print(f"  {prefix}... ({len(obj) - 20} more keys)")
    elif isinstance(obj, list):
        print(f"  {prefix}(list of {len(obj)})")
        if obj and isinstance(obj[0], dict):
            print(f"  {prefix}[0]:")
            _print_json_structure(obj[0], prefix + "  ", depth + 1, max_depth)


def probe_trace(
    filename: str,
    archive_index: Dict[str, Tuple[int, Path]],
    z7_path: Optional[str],
    password: str,
) -> None:
    """Extract one trace JSON and print its structure for inspection."""
    vol, archive_path = archive_index[filename]
    logger.info("Probing %s from volume %d...", filename, vol)

    raw = extract_one_file(archive_path, filename, z7_path, password)
    trace = json.loads(raw.decode("utf-8", errors="replace"))

    print()
    print("=" * 65)
    print(f"  TRACE PROBE: {filename}")
    print(f"  Volume: {vol}, Size: {len(raw):,} bytes")
    print("=" * 65)
    print()

    print("Top-level structure (depth 3):")
    _print_json_structure(trace, depth=0, max_depth=3)

    # Try to find API calls at common paths
    print()
    print("--- API call search ---")
    api_paths_found = []

    # Path 1: behavior.processes[].calls[]
    behavior = trace.get("behavior", {})
    if isinstance(behavior, dict):
        processes = behavior.get("processes", [])
        if processes:
            api_paths_found.append("behavior.processes")
            p0 = processes[0]
            print(f"  behavior.processes: {len(processes)} processes")
            print(f"  processes[0] keys: {list(p0.keys())[:15]}")
            calls = p0.get("calls", p0.get("api_calls", []))
            call_key = "calls" if "calls" in p0 else "api_calls" if "api_calls" in p0 else None
            if calls:
                print(f"  processes[0].{call_key}: {len(calls)} calls")
                c0 = calls[0]
                print(f"  calls[0] keys: {list(c0.keys())}")
                # Try common API name fields
                for field in ("api", "name", "function", "API"):
                    if field in c0:
                        print(f"  calls[0].{field}: {repr(c0[field])}")
                # Show first 5 API names
                api_field = None
                for field in ("api", "name", "function", "API"):
                    if field in c0:
                        api_field = field
                        break
                if api_field:
                    first_5 = [c.get(api_field, "???") for c in calls[:5]]
                    print(f"  First 5 API calls: {first_5}")
                    total_apis = sum(
                        len(p.get(call_key, []))
                        for p in processes
                    )
                    print(f"  Total API calls across all processes: {total_apis:,}")

    # Path 2: processes[] at top level
    if not api_paths_found:
        processes = trace.get("processes", [])
        if processes:
            api_paths_found.append("processes (top-level)")
            print(f"  Top-level processes: {len(processes)}")

    if not api_paths_found:
        print("  WARNING: No known API call paths found!")
        print("  Top-level keys: ", list(trace.keys()))

    print()


# ===================================================================
# ===================================================================
# Checkpoint / Parquet helpers
# ===================================================================


def _write_checkpoint(processed_hashes: Set[str], position: int) -> None:
    """Persist extraction progress so runs can be resumed."""
    WINMET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_hashes": sorted(processed_hashes),
        "position": position,
    }
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)
    logger.debug("Checkpoint saved: %d hashes, position %d.", len(processed_hashes), position)


def _write_parquet(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write extracted rows to a Parquet file using pyarrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build column arrays
    table = pa.table({
        "sha256": pa.array([r["sha256"] for r in rows], type=pa.string()),
        "family_avclass": pa.array([r["family_avclass"] for r in rows], type=pa.string()),
        "family_cape": pa.array([r["family_cape"] for r in rows], type=pa.string()),
        "family_consensus": pa.array([r["family_consensus"] for r in rows], type=pa.string()),
        "primary_class": pa.array([r["primary_class"] for r in rows], type=pa.string()),
        "secondary_classes": pa.array(
            [",".join(r["secondary_classes"]) if r["secondary_classes"] else "" for r in rows],
            type=pa.string(),
        ),
        "api_sequence": pa.array(
            [" ".join(r["api_sequence"]) for r in rows], type=pa.string()
        ),
        "num_processes": pa.array([r["num_processes"] for r in rows], type=pa.int32()),
        "sequence_length": pa.array([r["sequence_length"] for r in rows], type=pa.int32()),
        "source_volume": pa.array([r["source_volume"] for r in rows], type=pa.int32()),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract WinMET dataset from 7z volumes to Parquet.",
    )
    p.add_argument("--plan", action="store_true",
                   help="Load labels, compute selection, print plan, then exit.")
    p.add_argument("--probe", action="store_true",
                   help="Extract and print one trace JSON structure, then exit.")
    p.add_argument("--top-k-families", type=int, default=25,
                   help="Number of top families to include (default: 25).")
    p.add_argument("--max-per-family", type=int, default=300,
                   help="Max samples per family (default: 300).")
    p.add_argument("--min-samples-per-family", type=int, default=50,
                   help="Min samples for a family to be included (default: 50).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint.")
    p.add_argument("--yes", action="store_true",
                   help="Skip confirmation prompt.")
    p.add_argument("--min-free-gb", type=float, default=2.0,
                   help="Disk safety threshold in GB (default: 2.0).")
    p.add_argument("--password", type=str, default=DEFAULT_PASSWORD,
                   help="Archive password (default: standard research convention).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    # ── Environment checks ──────────────────────────────────────────
    z7_path = find_7z()
    if z7_path:
        logger.info("7z CLI found: %s", z7_path)
    else:
        logger.info("7z CLI not found; will use py7zr fallback.")

    # Verify label files exist
    for path, name in [
        (AVCLASS_PATH, "AVClass"),
        (CAPE_PATH, "CAPE"),
        (CONSENSUS_PATH, "Consensus"),
    ]:
        if not path.exists():
            logger.error("%s label file not found: %s", name, path)
            sys.exit(1)

    # ── Load labels ─────────────────────────────────────────────────
    logger.info("Loading AVClass labels...")
    avclass = load_avclass_labels(AVCLASS_PATH)
    logger.info("  %d samples with valid AVClass labels.", len(avclass))

    logger.info("Loading CAPE labels...")
    cape = load_cape_labels(CAPE_PATH)
    logger.info("  %d samples with CAPE labels.", len(cape))

    logger.info("Loading consensus labels...")
    consensus = load_consensus_labels(CONSENSUS_PATH)
    logger.info("  %d samples in consensus file.", len(consensus))

    logger.info("Building unified label dict (AVClass-primary)...")
    unified = build_unified_labels(avclass, cape, consensus)
    logger.info("  %d samples in unified dict.", len(unified))

    # ── Target selection ────────────────────────────────────────────
    selected, plan_rows = select_targets(
        unified,
        top_k=args.top_k_families,
        max_per_family=args.max_per_family,
        min_samples=args.min_samples_per_family,
    )

    print_plan(
        plan_rows, selected,
        top_k=args.top_k_families,
        max_per_family=args.max_per_family,
        min_samples=args.min_samples_per_family,
    )
    save_plan(plan_rows, selected, args)

    if args.plan:
        return

    # ── Confirmation ────────────────────────────────────────────────
    if not args.yes:
        answer = input("Proceed with extraction? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # ── Phase 4: Archive indexing ───────────────────────────────────
    archive_index = build_archive_index(args.password)
    target_fnames = set(selected.keys())
    found = target_fnames & set(archive_index.keys())
    missing = target_fnames - set(archive_index.keys())

    logger.info("Archive index: %d targets found, %d missing.", len(found), len(missing))
    if missing:
        examples = sorted(missing)[:5]
        logger.warning("Missing target files (first 5): %s", examples)

    if args.probe:
        # ── Phase 5: Probe one trace ────────────────────────────────
        probe_fname = sorted(found)[0]
        probe_trace(probe_fname, archive_index, z7_path, args.password)
        return

    # ── Checkpoint / resume ───────────────────────────────────────
    processed_hashes: Set[str] = set()
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        processed_hashes = set(ckpt.get("processed_hashes", []))
        logger.info("Resumed from checkpoint: %d already processed.", len(processed_hashes))

    # ── Extraction loop ────────────────────────────────────────────
    target_list = sorted(found)  # deterministic order
    remaining = [f for f in target_list if selected[f]["sha256"] not in processed_hashes]
    logger.info("Extracting %d samples (%d already done).",
                len(remaining), len(target_list) - len(remaining))

    # Accumulate rows in chunks to avoid memory exhaustion
    CHUNK_SIZE = 500
    rows: List[Dict[str, Any]] = []
    chunk_files: List[Path] = []
    dropped_parse = 0
    dropped_empty = 0
    total_seq_len = 0
    total_rows = 0
    seq_lens_all: List[int] = []  # lightweight — just ints
    fam_counter: Counter = Counter()
    class_counter: Counter = Counter()
    start_time = time.time()

    # Graceful shutdown handler
    _shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
        logger.warning("Shutdown requested — flushing checkpoint after current sample.")

    signal.signal(signal.SIGINT, _signal_handler)

    for idx, fname in enumerate(remaining):
        if _shutdown_requested:
            logger.info("Graceful shutdown at sample %d.", idx)
            break

        info = selected[fname]
        vol, archive_path = archive_index[fname]

        # Disk safety check every 100 samples
        if idx > 0 and idx % 100 == 0:
            usage = shutil.disk_usage(str(WINMET_OUTPUT_DIR))
            free_gb = usage.free / (1024**3)
            if free_gb < args.min_free_gb:
                logger.error("Disk space low (%.1f GB free). Stopping.", free_gb)
                break

        # Extract and parse
        try:
            raw = extract_one_file(archive_path, fname, z7_path, args.password)
            trace = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as e:
            logger.debug("Parse error for %s: %s", fname, e)
            dropped_parse += 1
            processed_hashes.add(info["sha256"])
            continue

        result = parse_trace(trace)
        if result is None:
            dropped_empty += 1
            processed_hashes.add(info["sha256"])
            continue

        api_sequence, num_processes = result

        # Look up class mapping
        fam = info["avclass"]
        mapping = FAMILY_CLASS_MAPPING.get(fam)
        primary_class = mapping[0] if mapping else None
        secondary_classes = mapping[1] if mapping else []

        rows.append({
            "sha256": info["sha256"],
            "family_avclass": info["avclass"],
            "family_cape": info["cape"],
            "family_consensus": info["consensus"],
            "primary_class": primary_class,
            "secondary_classes": secondary_classes,
            "api_sequence": api_sequence,
            "num_processes": num_processes,
            "sequence_length": len(api_sequence),
            "source_volume": vol,
        })

        processed_hashes.add(info["sha256"])
        total_seq_len += len(api_sequence)
        total_rows += 1
        seq_lens_all.append(len(api_sequence))
        fam_counter[info["avclass"]] += 1
        if primary_class:
            class_counter[primary_class] += 1

        # Progress log every 100 samples
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (len(remaining) - idx - 1) / rate if rate > 0 else 0
            avg_len = total_seq_len / total_rows if total_rows else 0
            logger.info(
                "Progress: %d/%d (%.0f/s, ETA %.0fs) | "
                "%d rows, avg_seq_len=%.0f, dropped=%d+%d",
                idx + 1, len(remaining), rate, eta,
                total_rows, avg_len, dropped_parse, dropped_empty,
            )

        # Flush chunk to disk every CHUNK_SIZE rows to limit memory
        if len(rows) >= CHUNK_SIZE:
            chunk_path = WINMET_OUTPUT_DIR / f".chunk_{len(chunk_files):04d}.parquet"
            _write_parquet(rows, chunk_path)
            chunk_files.append(chunk_path)
            logger.info("Flushed chunk %d (%d rows) to disk.", len(chunk_files), len(rows))
            rows.clear()

        # Checkpoint every 100 samples
        if (idx + 1) % 100 == 0:
            _write_checkpoint(processed_hashes, idx + 1)

    # Final checkpoint
    _write_checkpoint(processed_hashes, len(remaining))

    # Flush remaining rows
    if rows:
        chunk_path = WINMET_OUTPUT_DIR / f".chunk_{len(chunk_files):04d}.parquet"
        _write_parquet(rows, chunk_path)
        chunk_files.append(chunk_path)
        rows.clear()

    # ── Merge chunks into final Parquet ───────────────────────────
    output_path = WINMET_OUTPUT_DIR / "winmet_extracted.parquet"
    if chunk_files:
        import pyarrow.parquet as pq
        tables = [pq.read_table(cf) for cf in chunk_files]
        import pyarrow as pa
        merged = pa.concat_tables(tables)
        pq.write_table(merged, output_path, compression="snappy")
        logger.info("Parquet written: %s (%d rows)", output_path, merged.num_rows)
        # Clean up chunk files
        for cf in chunk_files:
            cf.unlink(missing_ok=True)
    else:
        logger.warning("No rows extracted.")

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  Extraction Complete")
    print("=" * 60)
    print(f"  Samples extracted: {total_rows}")
    print(f"  Dropped (parse error): {dropped_parse}")
    print(f"  Dropped (empty sequence): {dropped_empty}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if seq_lens_all:
        import numpy as np
        arr = np.array(seq_lens_all)
        print(f"  Sequence lengths: "
              f"p25={np.percentile(arr,25):.0f}, "
              f"p50={np.percentile(arr,50):.0f}, "
              f"p75={np.percentile(arr,75):.0f}, "
              f"p90={np.percentile(arr,90):.0f}, "
              f"p99={np.percentile(arr,99):.0f}")
        print(f"  Families: {len(fam_counter)}")
        for fam, cnt in fam_counter.most_common():
            print(f"    {fam:<25} {cnt:>5}")
        print(f"  Behavioral class distribution:")
        for cls in ["Trojan","Backdoor","Downloader","Worms","Spyware","Adware","Dropper","Virus"]:
            print(f"    {cls:<15} {class_counter.get(cls, 0):>5}")
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024**2)
            print(f"  Output file: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
