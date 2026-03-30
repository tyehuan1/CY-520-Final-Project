"""
Data loading for Mal-API-2019 and MalbehavD-V1 datasets.

Both datasets are loaded into a common format: a list of dicts with keys
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
