"""
VirusTotal-based family labeling for MalbehavD-V1 malware samples.

Queries the VirusTotal v3 API to classify each malware sample's SHA256 hash
into one of the 8 target malware families by searching AV engine detection
labels for family-name string matches.

Usage (as a script)::

    python -m src.virustotal_labeler --api-key YOUR_KEY

Set the ``VT_API_KEY`` environment variable or pass ``--api-key`` on the CLI.
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import config as cfg
from src.data_loading.data_loader import load_malbehavd
from src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)

# Pre-compile case-insensitive patterns for each family.
# Each family maps to multiple variant spellings / abbreviations
# observed in real AV engine labels on VirusTotal.
_FAMILY_VARIANT_MAP: Dict[str, List[str]] = {
    "Adware": [r"adware", r"adw", r"browsermodifier", r"clicker"],
    "Backdoor": [r"backdoor", r"back[\-\s]?door", r"bkdr", r"backdr",
                 r"\brat\b", r"remoteadmin"],
    "Downloader": [r"downloader", r"trojan[\-\.]?downloader", r"dldr",
                   r"dwnldr", r"download"],
    "Dropper": [r"dropper", r"trojan[\-\.]?dropper", r"drop",
                r"tr/dropper", r"injector", r"\binject\b"],
    "Spyware": [r"spyware", r"spy(?:agent)?", r"trojan[\-\.]?spy",
                r"trojansspy", r"keylog(?:ger)?", r"infostealer",
                r"stealer", r"banker", r"\bpws\b", r"\bpsw\b",
                r"pwstool", r"pswtool", r"pwstealer"],
    "Trojan": [r"trojan", r"troj", r"trj"],
    "Virus": [r"virus", r"infector", r"infectpe", r"fileinf"],
    "Worms": [r"worm", r"email[\-]?worm", r"net[\-]?worm", r"i[\-]?worm",
              r"w32\.worm", r"win32\.worm"],
}

# Well-known malware family names → our category.
# These are checked AFTER pattern matching as a secondary signal.
# Only includes families with strong consensus in the security community.
_KNOWN_FAMILY_NAMES: Dict[str, str] = {
    # ── Original (from VT label analysis) ──
    "padodor": "Backdoor",     # Well-documented backdoor trojan
    "dorkbot": "Worms",        # IRC-based worm
    "dinwod": "Dropper",       # Known dropper family
    "qukart": "Worms",         # Network worm (also seen as "Quart")
    "quart": "Worms",          # Alias for Qukart
    "soltern": "Worms",        # Email worm
    # ── Added from HA cache analysis ──
    # Backdoors / RATs
    "remcos": "Backdoor",      # Commercial RAT
    "bladabindi": "Backdoor",  # njRAT
    "njrat": "Backdoor",       # Alias for Bladabindi
    # Spyware / info-stealers
    "lokibot": "Spyware",      # Credential-harvesting info-stealer
    "azorult": "Spyware",      # Info-stealer
    "pony": "Spyware",         # Pony/Fareit credential stealer
    # Worms
    "gamarue": "Worms",        # Andromeda — USB/network worm
    # Trojans (generic but well-known family names)
    "razy": "Trojan",          # Browser modifier / crypto stealer
    "strictor": "Trojan",      # Trojan family
    "johnnie": "Trojan",       # Generic trojan variant
    # Droppers
    "msilperseus": "Trojan",   # Generic .NET trojan
    "ursu": "Trojan",          # Generic trojan (BitDefender/GData)
    "istartsur": "Adware",     # iStartSurf browser hijacker
    # Adware
    "graftor": "Adware",       # Adware/PUP family (BitDefender)
}

_FAMILY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (family, re.compile(pattern, re.IGNORECASE))
    for family, patterns in _FAMILY_VARIANT_MAP.items()
    for pattern in patterns
]

# Labels that match "Trojan" generically but carry no real family info.
# These should NOT count as a vote for the "Trojan" family.
# Built from analysis of actual VT labels in the cached responses.
_GENERIC_TROJAN_PATTERN: re.Pattern = re.compile(
    r"(?:"
    # Explicit generic suffixes
    r"trojan[:\.\-_/]?\s*(?:generic|agent|gen\b|horse|heur|malware|multi|"
    r"packed|small|starter|kryptik|crypt|zusy|occamy|casdet|"
    r"tedy|woreflint|convagent|tiggre|swrort|bulz|sabsik|phonzy|razy|"
    r"gencirc|ceeinject|skeeyah|ymacco|redcap|wacatac|bearfoos|mauvaise|"
    r"vilsel|pornoasset|nymaim)"
    r"|"
    # Trojan.Win32.Save.a — generic detection name (249 occurrences)
    r"trojan[:\.\-_/]?\s*(?:win32[:\.])?save\b"
    r"|"
    # Win32:Injector-CVE [Trj] and similar generic injector labels
    r"win32:injector"
    r"|"
    # Trojan.Win32.Inject / Trojan.Inject1 — generic injection labels
    r"trojan[:\.\-_/]?\s*(?:win32[:\.])?inject[0-9]*\b"
    r"|"
    # Trojan/Win32.AGeneric
    r"trojan/win32\.ageneric"
    r"|"
    # Win32:TrojanX-gen, Win32:MalwareX-gen, Win32:Evo-gen style
    r"win32:[a-z]*\-?gen\b"
    r"|"
    # Trojan.Win32.Generic!BT, HEUR:Trojan.Win32.Generic
    r"trojan\.win32\.generic"
    r"|"
    # Trojan ( hex ) format — always generic
    r"trojan\s*\(\s*[0-9a-f]+\s*\)"
    r"|"
    # Trj/Genetic.gen, Trj/CI.A, Trj/GdSda — generic Sophos-style
    r"trj/(?:genetic|ci\.|gdsda)"
    r"|"
    # TROJ_GEN.* — Trend Micro generic
    r"troj_gen\b"
    r"|"
    # ML/PE-A + Troj/* — ML-based generic detections
    r"ml/pe"
    r"|"
    # Win-Trojan/Name.size — generic Korean AV pattern (no family info)
    r"win[\-]?trojan/(?!.*(?:worm|spy|back|down|drop|virus|adw|ransom))"
    r")",
    re.IGNORECASE,
)


# ── Cache helpers ──────────────────────────────────────────────────────────


def load_cache(path: Path = cfg.VT_CACHE_PATH) -> Dict[str, Any]:
    """Load the VT response cache from disk, or return an empty dict."""
    if path.exists():
        try:
            return load_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cache (%s); starting fresh.", exc)
    return {}


def save_cache(cache: Dict[str, Any], path: Path = cfg.VT_CACHE_PATH) -> None:
    """Persist the VT response cache to disk (atomic write).

    Writes to a temporary file first, then renames to the target path.
    This prevents data loss if the process is interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    # Atomic replace — old file survives if write was interrupted
    tmp_path.replace(path)
    logger.debug("Cache saved: %d entries.", len(cache))


# ── VT API ─────────────────────────────────────────────────────────────────


VT_API_KEY = "edfa2cc77b02b053b3d909eab3354b6f8651d133ef5e3d8799a3142d265a301b"


def query_virustotal(sha256: str, api_key: str = VT_API_KEY) -> Optional[Dict[str, Any]]:
    """Query the VirusTotal v3 API for a single file hash.

    Args:
        sha256: SHA-256 hex digest of the file.
        api_key: VirusTotal API key.

    Returns:
        The JSON response dict on success, or ``None`` on HTTP error / timeout.
    """
    url = f"{cfg.VT_API_BASE}/{sha256}"
    headers = {"x-apikey": api_key}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            logger.info("Hash %s not found on VirusTotal (404).", sha256)
            return None
        logger.warning(
            "VT returned status %d for %s: %s",
            resp.status_code,
            sha256,
            resp.text[:200],
        )
        return None
    except requests.RequestException as exc:
        logger.error("Request failed for %s: %s", sha256, exc)
        return None


# ── Label extraction ───────────────────────────────────────────────────────


def extract_family_from_response(vt_response: Dict[str, Any]) -> Optional[str]:
    """Determine the malware family from a VT API response.

    For each AV engine label, finds all matching family names.  Uses three
    rules to avoid the "Trojan-as-prefix" inflation problem:

    1. **Generic Trojan suppression**: Labels like ``Trojan.Generic``,
       ``Win32:BackdoorX-gen [Trj]`` match a generic Trojan pattern.  The
       Trojan vote is suppressed, but any *specific* family keyword in the
       same label (e.g. ``Backdoor``) still counts.
    2. **Specificity preference**: If a label matches both ``Trojan`` and a
       more specific family (e.g. ``Trojan.Downloader``), only the specific
       family is counted.
    3. **Known family names**: Well-known malware names (e.g. Padodor,
       Dorkbot) are mapped to their established family even when AV vendors
       disagree on the prefix.
    4. **Deduplication**: Each label contributes at most one vote per family
       (multiple variant patterns matching the same family count once).

    Args:
        vt_response: Full JSON response from VT v3 ``/files/{hash}`` endpoint.

    Returns:
        Family name string (title-cased, e.g. ``"Trojan"``), or ``None``.
    """
    try:
        results = (
            vt_response
            .get("data", {})
            .get("attributes", {})
            .get("last_analysis_results", {})
        )
    except AttributeError:
        return None

    family_counts: Counter = Counter()

    for engine_info in results.values():
        label = engine_info.get("result")
        if not label:
            continue

        # Check if this is a generic trojan label (e.g., Trojan.Generic,
        # Win32:BackdoorX-gen [Trj]).  Generic labels should NOT contribute
        # a "Trojan" vote, but they may still contain a real family keyword
        # (e.g., "BackdoorX" in Win32:BackdoorX-gen), so we do NOT skip
        # the label entirely — we just suppress the Trojan vote below.
        is_generic_trojan = bool(_GENERIC_TROJAN_PATTERN.search(label))

        # Collect all families that match this label (deduplicated)
        matched_families: set = set()
        for family_name, pattern in _FAMILY_PATTERNS:
            if pattern.search(label):
                matched_families.add(family_name)

        # Check for known malware family names (e.g., Padodor → Backdoor)
        label_lower = label.lower()
        for known_name, known_family in _KNOWN_FAMILY_NAMES.items():
            if known_name in label_lower:
                matched_families.add(known_family)

        if not matched_families:
            continue

        # If the label matched the generic trojan filter, or if a specific
        # family matched alongside Trojan, drop the Trojan vote — it's
        # either noise or a prefix, not real family information.
        if "Trojan" in matched_families:
            if is_generic_trojan or len(matched_families) > 1:
                matched_families.discard("Trojan")

        for family_name in matched_families:
            family_counts[family_name] += 1

    if not family_counts:
        return None

    # Prefer a non-Trojan family over Trojan only when the evidence is
    # strong enough: the non-Trojan candidate must have >= 5 votes AND
    # at least 30 % of the Trojan vote count.  This avoids reclassifying
    # a sample as "Downloader" based on 1 engine vs 13 Trojan engines,
    # while still allowing Worms (7 votes vs 9 Trojan) to win.
    trojan_votes = family_counts.get("Trojan", 0)
    non_trojan = {f: c for f, c in family_counts.items() if f != "Trojan"}

    if non_trojan:
        best_non_trojan = max(non_trojan, key=non_trojan.get)
        best_nt_votes = non_trojan[best_non_trojan]

        if trojan_votes == 0:
            # No Trojan votes at all — take the best non-Trojan directly
            return best_non_trojan

        # Non-Trojan wins only with sufficient evidence
        if best_nt_votes >= 5 and best_nt_votes >= 0.30 * trojan_votes:
            return best_non_trojan

    # Fall back to overall most-common (usually Trojan)
    best_family, _ = family_counts.most_common(1)[0]
    return best_family


def extract_family_from_labels(
    engine_results: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Determine the malware family directly from engine results dict.

    This is a convenience wrapper for use in tests where you have the
    ``last_analysis_results`` dict already extracted.

    Args:
        engine_results: ``{engine_name: {result: ..., ...}, ...}`` dict.

    Returns:
        Family name string or ``None``.
    """
    # Wrap into the expected VT response structure
    wrapped = {"data": {"attributes": {"last_analysis_results": engine_results}}}
    return extract_family_from_response(wrapped)


# ── Main labeling pipeline ─────────────────────────────────────────────────


def label_malbehavd_samples(
    api_key: str = VT_API_KEY,
    cache_path: Path = cfg.VT_CACHE_PATH,
    rate_limit_sleep: float = cfg.VT_RATE_LIMIT_SLEEP,
    max_requests: int = 0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Label all malware samples in MalbehavD-V1 via VirusTotal.

    Benign samples are kept as-is.  For each malware sample, query VT (with
    caching and rate limiting) and attempt to assign one of the 8 target
    families.  Samples that cannot be mapped are dropped.

    Args:
        api_key: VirusTotal API key.
        cache_path: Path for the JSON cache file.
        rate_limit_sleep: Seconds to sleep between API requests.
        max_requests: Stop after this many new API calls (0 = unlimited).
            Cached lookups don't count.  Use this to stay within daily quotas.

    Returns:
        Tuple of (labeled_samples, dropped_hashes).
        Each labeled sample is a dict with keys ``sequence``, ``label``,
        and ``sha256``.
    """
    raw_samples = load_malbehavd()
    cache = load_cache(cache_path)

    labeled: List[Dict[str, Any]] = []
    dropped: List[str] = []
    api_calls_made = 0

    try:
        for i, sample in enumerate(raw_samples):
            # Keep benign samples as-is
            if sample["label"] == cfg.BENIGN_LABEL:
                labeled.append(sample)
                continue

            sha = sample["sha256"]

            # Check cache first
            if sha in cache:
                vt_resp = cache[sha]
            else:
                # Rate-limit before calling the API
                if api_calls_made > 0:
                    logger.debug("Sleeping %ss for rate limit...", rate_limit_sleep)
                    time.sleep(rate_limit_sleep)

                # Check daily quota before making a new request
                if max_requests > 0 and api_calls_made >= max_requests:
                    save_cache(cache, cache_path)
                    logger.info(
                        "Reached max_requests limit (%d). Stopping. "
                        "Re-run to continue from cache.",
                        max_requests,
                    )
                    break

                vt_resp = query_virustotal(sha, api_key)
                api_calls_made += 1

                # Cache even None responses so we don't re-query failures
                cache[sha] = vt_resp

                # Persist cache periodically (every 10 API calls)
                if api_calls_made % 10 == 0:
                    save_cache(cache, cache_path)
                    logger.info(
                        "Progress: %d/%d samples processed, %d API calls made.",
                        i + 1,
                        len(raw_samples),
                        api_calls_made,
                    )

            # Extract family label
            if vt_resp is None:
                family = None
            else:
                family = extract_family_from_response(vt_resp)

            if family is None:
                dropped.append(sha)
                logger.debug("Dropped hash %s — no family match.", sha)
            else:
                labeled.append({
                    "sequence": sample["sequence"],
                    "label": family,
                    "sha256": sha,
                })

    except KeyboardInterrupt:
        logger.info(
            "Interrupted by user after %d API calls. Saving cache...",
            api_calls_made,
        )

    # Final cache save (runs on normal exit, quota limit, AND KeyboardInterrupt)
    save_cache(cache, cache_path)
    logger.info("Cache saved with %d entries.", len(cache))

    logger.info(
        "Labeling complete: %d labeled (%d benign + %d malware), %d dropped.",
        len(labeled),
        sum(1 for s in labeled if s["label"] == cfg.BENIGN_LABEL),
        sum(1 for s in labeled if s["label"] != cfg.BENIGN_LABEL),
        len(dropped),
    )
    return labeled, dropped


# ── CLI entry point ────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for running the labeling pipeline."""
    parser = argparse.ArgumentParser(
        description="Label MalbehavD-V1 malware samples via VirusTotal API."
    )
    parser.add_argument(
        "--api-key",
        default=VT_API_KEY,
        help="VirusTotal API key (defaults to hardcoded key).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=490,
        help="Max new API calls per run (default 490, to stay under 500/day).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.CACHE_DIR / "malbehavd_labeled.json",
        help="Path to save labeled samples as JSON.",
    )
    args = parser.parse_args()

    labeled, dropped = label_malbehavd_samples(
        api_key=args.api_key,
        max_requests=args.max_requests,
    )

    # Save results
    output_data = {
        "samples": labeled,
        "dropped_hashes": dropped,
        "stats": {
            "total_labeled": len(labeled),
            "total_dropped": len(dropped),
            "family_distribution": dict(
                Counter(s["label"] for s in labeled)
            ),
        },
    }
    save_json(output_data, args.output)
    logger.info("Results saved to %s", args.output)

    # Print summary
    print(f"\n{'='*50}")
    print("VirusTotal Labeling Summary")
    print(f"{'='*50}")
    print(f"Total labeled:  {len(labeled)}")
    print(f"Total dropped:  {len(dropped)}")
    print(f"\nFamily distribution:")
    for fam, count in sorted(output_data["stats"]["family_distribution"].items()):
        print(f"  {fam:>12}: {count}")


if __name__ == "__main__":
    main()
