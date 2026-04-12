"""
Central configuration for the malware family classification project.

All file paths, hyperparameters, random seeds, and constants are defined here.
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

# Stage-2 no-Trojan filtered data (7-class family classification)
NO_TROJAN_CACHE_DIR = CACHE_DIR / "no_trojan"
NO_TROJAN_TRAIN_PATH = NO_TROJAN_CACHE_DIR / "preprocessed_train.pkl"
NO_TROJAN_TEST_PATH = NO_TROJAN_CACHE_DIR / "preprocessed_test.pkl"
NO_TROJAN_VOCABULARY_PATH = NO_TROJAN_CACHE_DIR / "vocabulary.json"
NO_TROJAN_LABEL_ENCODER_PATH = NO_TROJAN_CACHE_DIR / "label_encoder.pkl"
NO_TROJAN_FEATURES_DIR = NO_TROJAN_CACHE_DIR / "features"
NO_TROJAN_MALBEHAVD_PATH = NO_TROJAN_CACHE_DIR / "malbehavd_labeled.json"
NO_TROJAN_WINMET_PATH = NO_TROJAN_CACHE_DIR / "winmet_preprocessed.pkl"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost"
LSTM_MODEL_DIR = MODELS_DIR / "lstm"
ENSEMBLE_MODEL_DIR = MODELS_DIR / "ensemble"

# Stage-2 no-Trojan models (7-class)
NO_TROJAN_XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost_no_trojan"
NO_TROJAN_LSTM_MODEL_DIR = MODELS_DIR / "lstm_no_trojan"
NO_TROJAN_ENSEMBLE_MODEL_DIR = MODELS_DIR / "ensemble_no_trojan"

# V2 models — trained on the WITH-Trojan dataset (8 classes), with restored
# (longer) LSTM sequence lengths, log-dampened category/bigram features, no
# augment_sequences truncation, and sliding-window LSTM inference at eval
# time.  Kept entirely separate from the *_no_trojan artifacts above so the
# previous models stay reproducible.
V2_CACHE_DIR = CACHE_DIR / "v2"
V2_FEATURES_DIR = V2_CACHE_DIR / "features"
V2_LABEL_ENCODER_PATH = V2_CACHE_DIR / "label_encoder.pkl"
V2_TFIDF_PATH = V2_CACHE_DIR / "tfidf_vectorizer.pkl"

V2_XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost_v2"
V2_LSTM_MODEL_DIR = MODELS_DIR / "lstm_v2"
V2_ENSEMBLE_MODEL_DIR = MODELS_DIR / "ensemble_v2"
# V2 result paths are defined further down, after RESULTS_DIR.

# Results — organised as results/{model_version}/{dataset}/{artifact_type}
RESULTS_DIR = PROJECT_ROOT / "results"

# Original 8-class (with-Trojan) model
ORIGINAL_RESULTS_DIR = RESULTS_DIR / "original"
PLOTS_DIR = ORIGINAL_RESULTS_DIR / "MalAPI" / "plots"
METRICS_DIR = ORIGINAL_RESULTS_DIR / "MalAPI" / "metrics"
SHAP_DIR = ORIGINAL_RESULTS_DIR / "MalAPI" / "shap"
ORIGINAL_TRAINING_METRICS_DIR = ORIGINAL_RESULTS_DIR / "training" / "metrics"

# V2 results (with-Trojan retraining, log-dampened features)
V2_RESULTS_DIR = RESULTS_DIR / "v2"
V2_METRICS_DIR = V2_RESULTS_DIR / "MalAPI" / "metrics"
V2_PLOTS_DIR = V2_RESULTS_DIR / "MalAPI" / "plots"
V2_MALBEHAVD_METRICS_DIR = V2_RESULTS_DIR / "MalBehavD" / "metrics"
V2_MALBEHAVD_PLOTS_DIR = V2_RESULTS_DIR / "MalBehavD" / "plots"
V2_WINMET_METRICS_DIR = V2_RESULTS_DIR / "WinMET" / "metrics"
V2_WINMET_PLOTS_DIR = V2_RESULTS_DIR / "WinMET" / "plots"
V2_OLIVERA_METRICS_DIR = V2_RESULTS_DIR / "Olivera" / "metrics"
V2_OLIVERA_PLOTS_DIR = V2_RESULTS_DIR / "Olivera" / "plots"
V2_TRAINING_METRICS_DIR = V2_RESULTS_DIR / "training" / "metrics"

# Stage-2 no-Trojan results (7-class)
NO_TROJAN_RESULTS_DIR = RESULTS_DIR / "no_trojan"
NO_TROJAN_PLOTS_DIR = NO_TROJAN_RESULTS_DIR / "MalAPI" / "plots"
NO_TROJAN_METRICS_DIR = NO_TROJAN_RESULTS_DIR / "MalAPI" / "metrics"
NO_TROJAN_SHAP_DIR = NO_TROJAN_RESULTS_DIR / "MalAPI" / "shap"
NO_TROJAN_TRAINING_METRICS_DIR = NO_TROJAN_RESULTS_DIR / "training" / "metrics"

# Cross-dataset generalizability (no-Trojan models evaluated on external data)
GENERALIZABILITY_DIR = NO_TROJAN_RESULTS_DIR  # legacy alias
GENERALIZABILITY_MALBEHAVD_METRICS_DIR = NO_TROJAN_RESULTS_DIR / "MalBehavD" / "metrics"
GENERALIZABILITY_MALBEHAVD_PLOTS_DIR = NO_TROJAN_RESULTS_DIR / "MalBehavD" / "plots"
GENERALIZABILITY_WINMET_METRICS_DIR = NO_TROJAN_RESULTS_DIR / "WinMET" / "metrics"
GENERALIZABILITY_WINMET_PLOTS_DIR = NO_TROJAN_RESULTS_DIR / "WinMET" / "plots"
# Legacy flat aliases (used by older scripts — point to MalBehavD by default)
GENERALIZABILITY_PLOTS_DIR = GENERALIZABILITY_MALBEHAVD_PLOTS_DIR
GENERALIZABILITY_METRICS_DIR = GENERALIZABILITY_MALBEHAVD_METRICS_DIR
MALBEHAVD_LABELED_PATH = CACHE_DIR / "malbehavd_labeled.json"
OLIVERA_LABELED_PATH = CACHE_DIR / "olivera_labeled.json"

# WinMET (cross-dataset generalizability — Mal-API-format outputs)
WINMET_DIR = DATA_DIR / "winmet"
WINMET_PARQUET_PATH = WINMET_DIR / "winmet_extracted.parquet"
WINMET_SEQUENCES_PATH = WINMET_DIR / "winmet_sequences.txt"
WINMET_LABELS_PATH = WINMET_DIR / "winmet_labels.csv"
WINMET_NO_TROJAN_SEQUENCES_PATH = WINMET_DIR / "winmet_sequences_no_trojan.txt"
WINMET_NO_TROJAN_LABELS_PATH = WINMET_DIR / "winmet_labels_no_trojan.csv"
OLIVERA_VT_CACHE_PATH = CACHE_DIR / "olivera_vt_cache.json"
OLIVERA_VT_LABELED_PATH = CACHE_DIR / "olivera_vt_labeled.json"
HA_CACHE_PATH = CACHE_DIR / "hybrid_analysis_cache.json"

# Olivera generalizability experiment — train on Olivera-limited Mal-API,
# test on VT-labeled Olivera.  Both use Cuckoo sandbox, same 307-token vocab
# (case-normalised), fixed 100-call sequences.
OLIVERA_CACHE_DIR = CACHE_DIR / "olivera"
OLIVERA_TRAIN_PATH = OLIVERA_CACHE_DIR / "train.pkl"
OLIVERA_TEST_PATH = OLIVERA_CACHE_DIR / "test.pkl"
OLIVERA_EXT_TEST_PATH = OLIVERA_CACHE_DIR / "olivera_test.pkl"
OLIVERA_VOCABULARY_PATH = OLIVERA_CACHE_DIR / "vocabulary.json"
OLIVERA_LABEL_ENCODER_PATH = OLIVERA_CACHE_DIR / "label_encoder.pkl"
OLIVERA_TFIDF_PATH = OLIVERA_CACHE_DIR / "tfidf_vectorizer.pkl"
OLIVERA_XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost_olivera"
OLIVERA_LSTM_MODEL_DIR = MODELS_DIR / "lstm_olivera"
OLIVERA_RESULTS_DIR = RESULTS_DIR / "olivera"
OLIVERA_METRICS_DIR = OLIVERA_RESULTS_DIR / "metrics"
OLIVERA_PLOTS_DIR = OLIVERA_RESULTS_DIR / "plots"

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

# Olivera experiment — short (100-call) sequences, ~240-token vocab.
# Reduced TF-IDF dimensions (fewer possible n-grams at this length),
# bigrams only (trigrams too sparse at 100 calls).
OLIVERA_TFIDF_MAX_FEATURES = 1000
OLIVERA_TFIDF_NGRAM_RANGE = (1, 2)

# Olivera XGBoost — reuse V2-style regularised search space.  Short
# sequences + small vocab = higher overfit risk, so keep trees shallow.
OLIVERA_XGBOOST_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7],
    "min_child_weight": [3, 5, 7],
    "gamma": [0.0, 0.1, 0.3],
}

# V2 search space — fewer/shallower trees + stronger regularization to
# favour generalizable splits over training-distribution memorization.
XGBOOST_V2_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7],
    "min_child_weight": [3, 5, 7],
    "gamma": [0.0, 0.1, 0.3],
}

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
LSTM_SEQUENCE_LENGTHS = [300, 400, 500]

# Best LSTM sequence length (selected by test macro-F1 after retraining)
LSTM_BEST_SEQ_LEN = 400  # best macro-F1=0.6561 (tested 300, 400, 500)

# V2 (with-Trojan) LSTM sweep — restored to the pre-MalBehav lengths so the
# model can use the longer Mal-API sequences end-to-end.  At eval time
# sequences longer than the chosen length are handled by the
# sliding-window inference path (see ``predict_with_sliding_window``).
LSTM_V2_SEQUENCE_LENGTHS = [200, 500, 1000]
LSTM_V2_BEST_SEQ_LEN = 500  # best test macro-F1=0.5702 (tested 200, 500, 1000)
LSTM_V2_SLIDING_STRIDE = 200  # step size for sliding-window inference

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
VT_RATE_LIMIT_SLEEP = 16  # seconds between requests (safe margin under 4 req/min)

# =============================================================================
# Generalizability evaluation
# =============================================================================
BENIGN_LABEL = "Benign"
MALWARE_LABEL = "Malware"
