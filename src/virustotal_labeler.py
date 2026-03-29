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
from src.data_loader import load_malbehavd
from src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)

# Pre-compile case-insensitive patterns for each family
_FAMILY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (fam, re.compile(re.escape(fam), re.IGNORECASE))
    for fam in cfg.MALWARE_FAMILIES
]


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
    """Persist the VT response cache to disk."""
    save_json(cache, path)


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

    Searches all AV engine detection labels for case-insensitive matches
    against the 8 target family names.  Returns the family that appears most
    frequently across engines, or ``None`` if no family is matched.

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
        for family_name, pattern in _FAMILY_PATTERNS:
            if pattern.search(label):
                family_counts[family_name] += 1

    if not family_counts:
        return None

    # Return the family with the highest count
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

    # Final cache save
    save_cache(cache, cache_path)

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
