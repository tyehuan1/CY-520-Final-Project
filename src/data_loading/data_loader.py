"""
Data loading for Mal-API-2019, MalbehavD-V1, and Olivera datasets.

All datasets are loaded into a common format: a list of dicts with keys
``sequence`` (List[str]) and ``label`` (str).
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)

# Type alias for a single sample
Sample = Dict[str, object]  # {"sequence": List[str], "label": str}


def load_mal_api(
    sequences_path: Path = cfg.MAL_API_SEQUENCES_PATH,
    labels_path: Path = cfg.MAL_API_LABELS_PATH,
    max_seq_len: Optional[int] = cfg.MAX_RAW_SEQUENCE_LENGTH,
) -> List[Sample]:
    """Load the Mal-API-2019 dataset.

    Args:
        sequences_path: Path to ``Mal API.txt`` (one sequence per line,
            space-separated API call names).
        labels_path: Path to ``Mal API Labels.csv`` (one label per line,
            no header).
        max_seq_len: If set, truncate each sequence to this many tokens.
            Some samples exceed 1 million API calls and would exhaust memory.

    Returns:
        List of sample dicts with keys ``sequence`` and ``label``.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If the number of sequences and labels do not match.
    """
    logger.info("Loading Mal-API-2019 sequences from %s", sequences_path)
    sequences: List[List[str]] = []
    with open(sequences_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                tokens = stripped.split()
                if max_seq_len is not None and len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                sequences.append(tokens)

    logger.info("Loading Mal-API-2019 labels from %s", labels_path)
    labels: List[str] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                labels.append(stripped)

    if len(sequences) != len(labels):
        raise ValueError(
            f"Sequence count ({len(sequences)}) != label count ({len(labels)})"
        )

    samples: List[Sample] = [
        {"sequence": seq, "label": lbl} for seq, lbl in zip(sequences, labels)
    ]

    logger.info(
        "Loaded %d Mal-API-2019 samples across %d families",
        len(samples),
        len(set(labels)),
    )
    return samples


def load_malbehavd(path: Path = cfg.MALBEHAVD_PATH) -> List[Sample]:
    """Load the MalbehavD-V1 dataset.

    Each row becomes a sample whose ``sequence`` is the ordered list of API
    call names read from the numbered columns (non-empty values only), and
    whose ``label`` is ``"Benign"`` (labels==0) or ``"Malware"`` (labels==1).

    Args:
        path: Path to ``MalBehavD-V1-dataset.csv``.

    Returns:
        List of sample dicts with keys ``sequence``, ``label``, and ``sha256``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    logger.info("Loading MalbehavD-V1 from %s", path)
    samples: List[Sample] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Determine the range of numbered API-call columns.
        # Columns are: sha256, labels, 0, 1, 2, ... (some trailing empties).
        api_start_idx = 2  # first numbered column

        for row in reader:
            sha256 = row[0]
            label_int = int(row[1])
            label = cfg.BENIGN_LABEL if label_int == 0 else cfg.MALWARE_LABEL

            # Collect non-empty API call names from the numbered columns
            sequence: List[str] = [
                cell.strip()
                for cell in row[api_start_idx:]
                if cell.strip()
            ]

            samples.append({
                "sequence": sequence,
                "label": label,
                "sha256": sha256,
            })

    logger.info(
        "Loaded %d MalbehavD-V1 samples (%d benign, %d malware)",
        len(samples),
        sum(1 for s in samples if s["label"] == cfg.BENIGN_LABEL),
        sum(1 for s in samples if s["label"] == cfg.MALWARE_LABEL),
    )
    return samples


def load_olivera(
    path: Path = cfg.OLIVERA_PATH,
    api_decode_map: Optional[Dict[int, str]] = None,
) -> List[Sample]:
    """Load the Olivera dataset.

    The CSV has columns: ``hash``, ``t_0`` through ``t_99``, ``malware``.
    The ``t_X`` columns contain integer-encoded API call identifiers.
    If ``api_decode_map`` is provided, integers are decoded to API call
    name strings; otherwise the raw integer strings are kept as-is.

    Args:
        path: Path to ``Olivera Data.csv``.
        api_decode_map: Optional mapping from integer code to API call name.
            If ``None``, sequences contain the string representation of
            each integer (e.g. ``"112"``, ``"274"``).

    Returns:
        List of sample dicts with keys ``sequence`` (List[str]),
        ``label`` (``"Benign"`` or ``"Malware"``), and ``hash``
        (MD5 hex string).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    logger.info("Loading Olivera dataset from %s", path)
    samples: List[Sample] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            md5_hash = row["hash"].strip()
            malware_flag = int(row["malware"])
            label = cfg.BENIGN_LABEL if malware_flag == 0 else cfg.MALWARE_LABEL

            # Read the 100 timestep columns
            sequence: List[str] = []
            for i in range(cfg.OLIVERA_SEQ_COLUMNS):
                val = int(row[f"t_{i}"])
                if api_decode_map is not None:
                    token = api_decode_map.get(val, str(val))
                else:
                    token = str(val)
                sequence.append(token)

            samples.append({
                "sequence": sequence,
                "label": label,
                "hash": md5_hash,
            })

    benign_count = sum(1 for s in samples if s["label"] == cfg.BENIGN_LABEL)
    malware_count = sum(1 for s in samples if s["label"] == cfg.MALWARE_LABEL)

    logger.info(
        "Loaded %d Olivera samples (%d benign, %d malware)",
        len(samples), benign_count, malware_count,
    )
    return samples
