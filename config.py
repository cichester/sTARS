import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = BASE_DIR / "dataset" / "Electronics.jsonl"

# Derived Data Paths
PROCESSED_DATA_DIR = DATA_DIR
PARQUET_5CORE_PATH = PROCESSED_DATA_DIR / "electronics_5core.parquet"
FEATURES_PATH = PROCESSED_DATA_DIR / "features_behavioral.parquet"
EMBEDDINGS_DIR = PROCESSED_DATA_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, RESULTS_DIR, BASE_DIR / "src", BASE_DIR / "tests"]:
    os.makedirs(directory, exist_ok=True)

# Hyperparameters - Data Ingestion
MIN_INTERACTIONS = 10  # 5-core threshold (era 5, ma 15M recensioni sono troppe da processare in locale in tempi utili)

# Hyperparameters - Models
EMBEDDING_BATCH_SIZE = 1024 # Aumentato per sfruttare meglio la RTX 5070
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROBERTA_MODEL_NAME = "roberta-base"

# LoRA Fine-Tuning
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "value"]
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 8
FINETUNE_GRAD_ACCUM = 4
FINETUNE_LR = 2e-4
FINETUNE_MAX_SEQ_LEN = 128
TRIPLET_MARGIN = 0.3
NUM_TRIPLETS = 500_000

# Paths RoBERTa
ROBERTA_ADAPTER_DIR = DATA_DIR / "roberta_lora_adapter"
EMBEDDINGS_ROBERTA_DIR = DATA_DIR / "embeddings_roberta"
TRIPLETS_PATH = DATA_DIR / "triplets.parquet"

# Ensure RoBERTa directories exist
os.makedirs(EMBEDDINGS_ROBERTA_DIR, exist_ok=True)

# Anomaly Detection
ISOLATION_FOREST_ESTIMATORS = 200
ISOLATION_FOREST_CONTAMINATION = 0.05
RANDOM_SEED = 42

# Training/Evaluation
TEST_SIZE_RATIO = 0.2
TOP_K_VALUES = [5, 10, 20]

# Google Gemini API configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_CACHE_PATH = DATA_DIR / "genai_reviews_cache.json"
GENAI_BATCH_SIZE = 5
GENAI_RPM_LIMIT = 14

