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

# Ensure directories exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, BASE_DIR / "src", BASE_DIR / "tests"]:
    os.makedirs(directory, exist_ok=True)

# Hyperparameters - Data Ingestion
MIN_INTERACTIONS = 10  # 5-core threshold (era 5, ma 15M recensioni sono troppe da processare in locale in tempi utili)

# Hyperparameters - Models
EMBEDDING_BATCH_SIZE = 1024 # Aumentato per sfruttare meglio la RTX 5070
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROBERTA_MODEL_NAME = "roberta-base"

# Anomaly Detection
ISOLATION_FOREST_ESTIMATORS = 200
ISOLATION_FOREST_CONTAMINATION = 0.05
RANDOM_SEED = 42

# Training/Evaluation
TEST_SIZE_RATIO = 0.2
TOP_K_VALUES = [5, 10, 20]
