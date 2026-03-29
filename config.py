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

# Cache
CACHE_DIR = PROJECT_ROOT / "cache"
VT_CACHE_PATH = CACHE_DIR / "virustotal_cache.json"
PREPROCESSED_TRAIN_PATH = CACHE_DIR / "preprocessed_train.pkl"
PREPROCESSED_TEST_PATH = CACHE_DIR / "preprocessed_test.pkl"
VOCABULARY_PATH = CACHE_DIR / "vocabulary.json"
FEATURES_DIR = CACHE_DIR / "features"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost"
LSTM_MODEL_DIR = MODELS_DIR / "lstm"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
SHAP_DIR = RESULTS_DIR / "shap"

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
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma": [0.0, 0.1, 0.2, 0.3],
}
XGBOOST_CV_FOLDS = 5
XGBOOST_N_ITER = 50  # Number of random search iterations
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
LSTM_SEQUENCE_LENGTHS = [200, 500, 1000]

# ReduceLROnPlateau
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 3
LR_MIN = 1e-6

# EarlyStopping
EARLY_STOP_PATIENCE = 7

# =============================================================================
# VirusTotal
# =============================================================================
VT_API_BASE = "https://www.virustotal.com/api/v3/files"
VT_RATE_LIMIT_SLEEP = 15  # seconds between requests (4 req/min)

# =============================================================================
# Generalizability evaluation
# =============================================================================
BENIGN_LABEL = "Benign"
MALWARE_LABEL = "Malware"
