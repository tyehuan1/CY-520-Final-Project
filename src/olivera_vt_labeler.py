"""
VirusTotal-based family labeling for Olivera dataset malware samples.

Queries the VirusTotal v3 API using MD5 hashes (VT accepts MD5, SHA1,
or SHA256 interchangeably on the ``/files/{hash}`` endpoint).

Reuses the same family extraction logic as the MalBehavD VT labeler
(``extract_family_from_response``).

Usage (as a script)::

    python -m src.olivera_vt_labeler --max-requests 490
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as cfg
from src.data_loader import load_olivera
from src.olivera_api_map import OLIVERA_API_DECODE
from src.utils import get_logger, save_json
from src.virustotal_labeler import (
    VT_API_KEY,
    extract_family_from_response,
    load_cache,
    query_virustotal,
    save_cache,
)

logger = get_logger(__name__)


# ── Main labeling pipeline ────────────────────────────────────────────────────


def label_olivera_vt_samples(
    api_key: str = VT_API_KEY,
    cache_path: Path = cfg.OLIVERA_VT_CACHE_PATH,
    rate_limit_sleep: float = cfg.VT_RATE_LIMIT_SLEEP,
    max_requests: int = 0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Label Olivera malware samples via VirusTotal using MD5 hashes.

    Benign samples (malware=0) are kept as-is.  For each malware sample,
    query VT with the MD5 hash (with caching and rate limiting) and
    attempt to assign one of the 8 target families.  Samples that cannot
    be mapped are dropped.

    Args:
        api_key: VirusTotal API key.
        cache_path: Path for the Olivera-specific VT JSON cache file.
        rate_limit_sleep: Seconds to sleep between API requests.
        max_requests: Stop after this many new API calls (0 = unlimited).
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
                vt_resp = cache[md5]
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

                # VT accepts MD5 directly on the /files/{hash} endpoint
                vt_resp = query_virustotal(md5, api_key)
                api_calls_made += 1

                # Cache even None responses so we don't re-query failures
                cache[md5] = vt_resp

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
            "Interrupted by user after %d API calls. Saving cache...",
            api_calls_made,
        )

    # Final cache save
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


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for running the Olivera VT labeling pipeline."""
    parser = argparse.ArgumentParser(
        description="Label Olivera malware samples via VirusTotal API (MD5)."
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
        default=cfg.OLIVERA_VT_LABELED_PATH,
        help="Path to save labeled samples as JSON.",
    )
    args = parser.parse_args()

    labeled, dropped = label_olivera_vt_samples(
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
    cache_size = len(load_cache(cfg.OLIVERA_VT_CACHE_PATH))
    print(f"\n{'='*50}")
    print("Olivera VirusTotal Labeling Summary")
    print(f"{'='*50}")
    print(f"Total labeled:  {len(labeled)}")
    print(f"Total dropped:  {len(dropped)}")
    print(f"Cache entries:  {cache_size}")
    print(f"\nFamily distribution:")
    for fam, count in sorted(output_data["stats"]["family_distribution"].items()):
        print(f"  {fam:>12}: {count}")


if __name__ == "__main__":
    main()
