"""
Central configuration for the malware family classification project.

All file paths, hyperparameters, random seeds, and constants are defined here.
No magic numbers in code — import from this module.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
MAL_API_SEQUENCES_PATH = DATA_DIR / "Mal API.txt"
MAL_API_LABELS_PATH = DATA_DIR / "Mal API Labels.csv"
MALBEHAVD_PATH = DATA_DIR / "MalBehavD-V1-dataset.csv"
OLIVERA_PATH = DATA_DIR / "Olivera Data.csv"

# Cache
CACHE_DIR = PROJECT_ROOT / "cache"
VT_CACHE_PATH = CACHE_DIR / "virustotal_cache.json"
PREPROCESSED_TRAIN_PATH = CACHE_DIR / "preprocessed_train.pkl"
PREPROCESSED_TEST_PATH = CACHE_DIR / "preprocessed_test.pkl"
VOCABULARY_PATH = CACHE_DIR / "vocabulary.json"

# Olivera-limited Mal-API (first 100 non-repeated calls)
MALAPI_OLIVERA_LIMITED_PATH = CACHE_DIR / "malapi_olivera_limited.pkl"
FEATURES_DIR = CACHE_DIR / "features"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost"
LSTM_MODEL_DIR = MODELS_DIR / "lstm"
ENSEMBLE_MODEL_DIR = MODELS_DIR / "ensemble"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
SHAP_DIR = RESULTS_DIR / "shap"

# Phase 7 — Generalizability
GENERALIZABILITY_DIR = RESULTS_DIR / "generalizability"
GENERALIZABILITY_PLOTS_DIR = GENERALIZABILITY_DIR / "plots"
GENERALIZABILITY_METRICS_DIR = GENERALIZABILITY_DIR / "metrics"
MALBEHAVD_LABELED_PATH = CACHE_DIR / "malbehavd_labeled.json"
OLIVERA_LABELED_PATH = CACHE_DIR / "olivera_labeled.json"
OLIVERA_VT_CACHE_PATH = CACHE_DIR / "olivera_vt_cache.json"
OLIVERA_VT_LABELED_PATH = CACHE_DIR / "olivera_vt_labeled.json"
HA_CACHE_PATH = CACHE_DIR / "hybrid_analysis_cache.json"

# =============================================================================
# Random seed
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# Dataset constants
# =============================================================================
MAL_API_EXPECTED_SAMPLES = 7107
MALBEHAVD_EXPECTED_SAMPLES = 2570
MALBEHAVD_EXPECTED_BENIGN = 1285
MALBEHAVD_EXPECTED_MALWARE = 1285

OLIVERA_EXPECTED_SAMPLES = 43876
OLIVERA_EXPECTED_BENIGN = 1079
OLIVERA_EXPECTED_MALWARE = 42797
OLIVERA_SEQ_COLUMNS = 100  # t_0 through t_99

MALWARE_FAMILIES = [
    "Adware",
    "Backdoor",
    "Downloader",
    "Dropper",
    "Spyware",
    "Trojan",
    "Virus",
    "Worms",
]

NUM_CLASSES = len(MALWARE_FAMILIES)

# =============================================================================
# Preprocessing
# =============================================================================
# Cap raw sequences at load time to avoid memory issues.
# The longest LSTM input is 1000; TF-IDF benefits from more context but
# sequences beyond 10k calls add diminishing returns.
MAX_RAW_SEQUENCE_LENGTH = 10_000

TEST_SIZE = 0.20
MAX_CONSECUTIVE_DUPLICATES = 5
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_INDEX = 0
UNK_INDEX = 1

# =============================================================================
# Feature engineering (XGBoost)
# =============================================================================
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 3)
TOP_K_API_FREQUENCIES = 5


# =============================================================================
# XGBoost hyperparameter search space
# =============================================================================
XGBOOST_PARAM_DIST = {
    "n_estimators": [200, 300, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1],
}
XGBOOST_CV_FOLDS = 3
XGBOOST_N_ITER = 40  # Number of random search iterations
XGBOOST_TREE_METHOD = "hist"

# =============================================================================
# LSTM hyperparameters
# =============================================================================
LSTM_EMBEDDING_DIM = 128
LSTM_HIDDEN_UNITS_1 = 128
LSTM_HIDDEN_UNITS_2 = 64
LSTM_SPATIAL_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.1
LSTM_DENSE_UNITS = 64
LSTM_DROPOUT = 0.3
LSTM_INITIAL_LR = 1e-3
LSTM_BATCH_SIZE = 64
LSTM_MAX_EPOCHS = 50
LSTM_VALIDATION_SPLIT = 0.10
LSTM_SEQUENCE_LENGTHS = [200, 300, 400]

# Best LSTM sequence length (selected by test macro-F1 in Phase 5)
LSTM_BEST_SEQ_LEN = 200

# ReduceLROnPlateau
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 3
LR_MIN = 1e-6

# EarlyStopping
EARLY_STOP_PATIENCE = 7

# =============================================================================
# Binary (Stage-1) model — malware vs benign detection
# =============================================================================
BINARY_CACHE_DIR = CACHE_DIR / "binary"
BINARY_PREPROCESSED_TRAIN_PATH = BINARY_CACHE_DIR / "preprocessed_train.pkl"
BINARY_PREPROCESSED_TEST_PATH = BINARY_CACHE_DIR / "preprocessed_test.pkl"
BINARY_VOCABULARY_PATH = BINARY_CACHE_DIR / "vocabulary.json"
BINARY_LABEL_ENCODER_PATH = BINARY_CACHE_DIR / "label_encoder.pkl"
BINARY_FEATURES_DIR = BINARY_CACHE_DIR / "features"

BINARY_XGBOOST_MODEL_DIR = MODELS_DIR / "binary_xgboost"
BINARY_RESULTS_DIR = RESULTS_DIR / "binary"
BINARY_METRICS_DIR = BINARY_RESULTS_DIR / "metrics"
BINARY_PLOTS_DIR = BINARY_RESULTS_DIR / "plots"

BINARY_NUM_CLASSES = 2

# =============================================================================
# VirusTotal
# =============================================================================
VT_API_BASE = "https://www.virustotal.com/api/v3/files"
VT_RATE_LIMIT_SLEEP = 16  # seconds between requests (safe margin under 4 req/min)

# =============================================================================
# Hybrid Analysis (for Olivera dataset labeling)
# =============================================================================
HA_API_BASE = "https://www.hybrid-analysis.com/api/v2"
HA_RATE_LIMIT_SLEEP = 18  # seconds between requests (200 req/hour limit)

# =============================================================================
# Generalizability evaluation
# =============================================================================
BENIGN_LABEL = "Benign"
MALWARE_LABEL = "Malware"
