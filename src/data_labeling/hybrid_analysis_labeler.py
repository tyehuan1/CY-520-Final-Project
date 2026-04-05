"""
Hybrid Analysis-based family labeling for Olivera dataset malware samples.

Queries the Hybrid Analysis v2 API to classify each malware sample's MD5 hash
into one of the 8 target malware families.  Uses a two-step lookup:

1. ``GET /search/hash`` — resolves the MD5 to SHA256(s) and report IDs.
2. ``GET /report/{sha256}:{env_id}/summary`` — fetches the full sandbox
   report including ``classification_tags`` and ``vx_family``.

The family extraction rules are copied from the VirusTotal labeler and will
be refined once initial results are available.

Usage (as a script)::

    python -m src.hybrid_analysis_labeler --max-requests 190
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import config as cfg
from src.data_loading.data_loader import load_olivera
from src.data_loading.olivera_api_map import OLIVERA_API_DECODE
from src.utils import get_logger, load_json, save_json
from src.data_labeling.virustotal_labeler import extract_family_from_response

logger = get_logger(__name__)


# ── Hybrid Analysis API ───────────────────────────────────────────────────────

HA_API_KEY = "yrtrflaide8f37ce5r70lpi6e9561a7aszo5l3b256cbab153lpl3ogt24303fde"

_HA_HEADERS_BASE = {
    "User-Agent": "Falcon Sandbox",
    "accept": "application/json",
}


def _ha_headers(api_key: str) -> Dict[str, str]:
    """Build request headers with the given API key."""
    return {**_HA_HEADERS_BASE, "api-key": api_key}


def query_hybrid_analysis(
    md5_hash: str,
    api_key: str = HA_API_KEY,
) -> Optional[Dict[str, Any]]:
    """Look up an MD5 hash on Hybrid Analysis and return the richest report.

    Step 1: ``GET /search/hash`` to resolve MD5 → SHA256 + report IDs.
    Step 2: ``GET /report/{sha256}:{env_id}/summary`` for the first report
    to get ``classification_tags``, ``vx_family``, etc.

    If the hash has no reports, falls back to the ``/overview/{sha256}``
    endpoint which still provides tags and vx_family.

    Args:
        md5_hash: MD5 hex digest of the file.
        api_key: Hybrid Analysis API key.

    Returns:
        A dict with at least ``classification_tags`` and/or ``vx_family``
        on success, or ``None`` if the hash is not found.
    """
    headers = _ha_headers(api_key)

    # Step 1: Resolve MD5 to SHA256 + reports
    try:
        search_resp = requests.get(
            f"{cfg.HA_API_BASE}/search/hash",
            headers=headers,
            params={"hash": md5_hash},
            timeout=30,
        )
    except requests.RequestException as exc:
        logger.error("HA search request failed for %s: %s", md5_hash, exc)
        return None

    if search_resp.status_code == 404:
        logger.debug("Hash %s not found on Hybrid Analysis.", md5_hash)
        return None
    if search_resp.status_code != 200:
        logger.warning(
            "HA search returned status %d for %s: %s",
            search_resp.status_code, md5_hash, search_resp.text[:200],
        )
        return None

    search_data = search_resp.json()
    sha256_list = search_data.get("sha256s", [])
    reports = search_data.get("reports", [])

    if not sha256_list:
        logger.debug("Hash %s found but has no SHA256 mapping.", md5_hash)
        return None

    sha256 = sha256_list[0]

    # Step 2: Try to get full report summary (has richer data)
    if reports:
        env_id = reports[0].get("environment_id", 100)
        try:
            report_resp = requests.get(
                f"{cfg.HA_API_BASE}/report/{sha256}:{env_id}/summary",
                headers=headers,
                timeout=30,
            )
            if report_resp.status_code == 200:
                return report_resp.json()
        except requests.RequestException as exc:
            logger.warning("HA report request failed for %s: %s", sha256, exc)

    # Fallback: overview endpoint (less detailed but still has tags)
    try:
        overview_resp = requests.get(
            f"{cfg.HA_API_BASE}/overview/{sha256}",
            headers=headers,
            timeout=30,
        )
        if overview_resp.status_code == 200:
            return overview_resp.json()
    except requests.RequestException as exc:
        logger.warning("HA overview request failed for %s: %s", sha256, exc)

    return None


def extract_family_from_ha_response(
    ha_response: Dict[str, Any],
) -> Optional[str]:
    """Determine malware family from a Hybrid Analysis report/overview.

    Collects classification signals from:
    - ``classification_tags`` (list of strings like "banker", "trojan")
    - ``vx_family`` (HA's own classification, e.g. "Trojan.Fareit")
    - ``tags`` (alias for classification_tags in some responses)

    These are wrapped into a VT-compatible structure and run through the
    same ``extract_family_from_response`` logic used for VirusTotal.

    Args:
        ha_response: Dict from report/summary or overview endpoint.

    Returns:
        Family name string or ``None``.
    """
    if not ha_response:
        return None

    # Build a VT-like engine_results dict from HA's classification fields
    engine_results: Dict[str, Dict[str, Any]] = {}
    idx = 0

    # 1. vx_family — HA's own family classification (e.g. "Trojan.Fareit")
    vx_family = ha_response.get("vx_family")
    if vx_family:
        engine_results[f"ha_vx_family"] = {"result": vx_family}
        idx += 1

    # 2. classification_tags — list like ["banker", "downloader", "trojan"]
    tags = ha_response.get("classification_tags") or ha_response.get("tags")
    if tags and isinstance(tags, list):
        for tag in tags:
            engine_results[f"ha_tag_{idx}"] = {"result": str(tag)}
            idx += 1

    if not engine_results:
        return None

    # Wrap into VT response format and reuse the extraction logic
    wrapped = {
        "data": {
            "attributes": {
                "last_analysis_results": engine_results,
            },
        },
    }
    return extract_family_from_response(wrapped)


# ── Cache helpers ─────────────────────────────────────────────────────────────


def load_cache(path: Path = cfg.HA_CACHE_PATH) -> Dict[str, Any]:
    """Load the HA response cache from disk, or return an empty dict."""
    if path.exists():
        try:
            return load_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load HA cache (%s); starting fresh.", exc)
    return {}


def save_cache(cache: Dict[str, Any], path: Path = cfg.HA_CACHE_PATH) -> None:
    """Persist the HA response cache to disk (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)
    logger.debug("HA cache saved: %d entries.", len(cache))


# ── Main labeling pipeline ────────────────────────────────────────────────────


def label_olivera_samples(
    api_key: str = HA_API_KEY,
    cache_path: Path = cfg.HA_CACHE_PATH,
    rate_limit_sleep: float = cfg.HA_RATE_LIMIT_SLEEP,
    max_requests: int = 0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Label all malware samples in the Olivera dataset via Hybrid Analysis.

    Benign samples are kept as-is.  For each malware sample, query HA (with
    caching and rate limiting) and attempt to assign one of the 8 target
    families.  Samples that cannot be mapped are dropped.

    Note: each hash lookup makes up to 2 API calls (search + report), but
    ``max_requests`` counts *hash lookups*, not individual HTTP requests.

    Args:
        api_key: Hybrid Analysis API key.
        cache_path: Path for the JSON cache file.
        rate_limit_sleep: Seconds to sleep between hash lookups.
        max_requests: Stop after this many new hash lookups (0 = unlimited).
            Cached lookups don't count.

    Returns:
        Tuple of (labeled_samples, dropped_hashes).
        Each labeled sample is a dict with keys ``sequence``, ``label``,
        and ``hash`` (MD5).
    """
    raw_samples = load_olivera(api_decode_map=OLIVERA_API_DECODE)
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

            md5 = sample["hash"]

            # Check cache first
            if md5 in cache:
                ha_resp = cache[md5]
            else:
                # Rate-limit before calling the API
                if api_calls_made > 0:
                    logger.debug("Sleeping %ss for rate limit...", rate_limit_sleep)
                    time.sleep(rate_limit_sleep)

                # Check quota before making a new request
                if max_requests > 0 and api_calls_made >= max_requests:
                    save_cache(cache, cache_path)
                    logger.info(
                        "Reached max_requests limit (%d). Stopping. "
                        "Re-run to continue from cache.",
                        max_requests,
                    )
                    break

                ha_resp = query_hybrid_analysis(md5, api_key)
                api_calls_made += 1

                # Cache even None responses
                cache[md5] = ha_resp

                # Persist cache periodically
                if api_calls_made % 10 == 0:
                    save_cache(cache, cache_path)
                    logger.info(
                        "Progress: %d/%d samples processed, %d hash lookups made.",
                        i + 1,
                        len(raw_samples),
                        api_calls_made,
                    )

            # Extract family label
            if ha_resp is None:
                family = None
            else:
                family = extract_family_from_ha_response(ha_resp)

            if family is None:
                dropped.append(md5)
                logger.debug("Dropped hash %s — no family match.", md5)
            else:
                labeled.append({
                    "sequence": sample["sequence"],
                    "label": family,
                    "hash": md5,
                })

    except KeyboardInterrupt:
        logger.info(
            "Interrupted by user after %d hash lookups. Saving cache...",
            api_calls_made,
        )

    # Final cache save
    save_cache(cache, cache_path)
    logger.info("HA cache saved with %d entries.", len(cache))

    logger.info(
        "Labeling complete: %d labeled (%d benign + %d malware), %d dropped.",
        len(labeled),
        sum(1 for s in labeled if s["label"] == cfg.BENIGN_LABEL),
        sum(1 for s in labeled if s["label"] != cfg.BENIGN_LABEL),
        len(dropped),
    )
    return labeled, dropped


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for running the labeling pipeline."""
    parser = argparse.ArgumentParser(
        description="Label Olivera malware samples via Hybrid Analysis API."
    )
    parser.add_argument(
        "--api-key",
        default=HA_API_KEY,
        help="Hybrid Analysis API key.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=2000,
        help="Max new hash lookups per run (default 2000, ~10 hrs at 18s/req).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.OLIVERA_LABELED_PATH,
        help="Path to save labeled samples as JSON.",
    )
    args = parser.parse_args()

    labeled, dropped = label_olivera_samples(
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
    print("Hybrid Analysis Labeling Summary")
    print(f"{'='*50}")
    print(f"Total labeled:  {len(labeled)}")
    print(f"Total dropped:  {len(dropped)}")
    print(f"\nFamily distribution:")
    for fam, count in sorted(output_data["stats"]["family_distribution"].items()):
        print(f"  {fam:>12}: {count}")


if __name__ == "__main__":
    main()
