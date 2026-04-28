import polars as pl
import numpy as np
import os
import torch
import logging
from config import PARQUET_5CORE_PATH, EMBEDDINGS_DIR, SBERT_MODEL_NAME, EMBEDDING_BATCH_SIZE
from src.preprocessing import clean_text_columns

logger = logging.getLogger(__name__)

def generate_embeddings(batch_texts: list[str], model, device) -> np.ndarray:
    """Codifica un batch di stringhe usando il modello."""
    with torch.no_grad():
        embeddings = model.encode(batch_texts, batch_size=len(batch_texts), device=device, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def build_profiles(df: pl.DataFrame, embeddings: np.ndarray) -> tuple[dict, dict]:
    """Costruisce i profili medi per utente e item."""
    logger.info("Costruzione profili (medie embeddings)...")
    
    users = df["user_id"].to_numpy()
    items = df["parent_asin"].to_numpy()
    
    # Per array in memory cosi grandi NumPy index ops/groupby sono super veloci
    # Ma Polars group_by lo è altrettanto se passiamo le tuple.
    # Un hack performante: creiamo un df polars temporaneo con l'id e row_idx
    import collections
    
    user_idxs = collections.defaultdict(list)
    item_idxs = collections.defaultdict(list)
    
    for i, (u, it) in enumerate(zip(users, items)):
        user_idxs[u].append(i)
        item_idxs[it].append(i)
        
    user_profiles = {}
    for u, idxs in user_idxs.items():
        user_profiles[u] = np.mean(embeddings[idxs], axis=0)
        
    item_profiles = {}
    for it, idxs in item_idxs.items():
        item_profiles[it] = np.mean(embeddings[idxs], axis=0)
        
    logger.info(f"Creati {len(user_profiles)} user profiles e {len(item_profiles)} item profiles.")
    return user_profiles, item_profiles

def run_embedding_pipeline():
    # Model Loading
    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Caricando modello: {SBERT_MODEL_NAME} su {device}")
    model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    
    # Load dataset & apply prep
    logger.info("Caricamento dataset e pulizia testuale...")
    df = pl.read_parquet(PARQUET_5CORE_PATH)
    df = clean_text_columns(df)
    
    # Estrazione array testuale
    texts = df["review_text"].to_list()
    
    logger.info(f"Produzione embeddings per {len(texts)} recensioni in batch da {EMBEDDING_BATCH_SIZE}...")
    all_embeddings = model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, device=device, show_progress_bar=True, convert_to_numpy=True)
    
    # Salvataggio memmap degli embeddings originali (opzionale, ma utile per il caching)
    emb_file = os.path.join(EMBEDDINGS_DIR, "reviews_embeddings.npy")
    np.save(emb_file, all_embeddings)
    logger.info(f"Embeddings salvati in {emb_file}")
    
    # Profili
    user_profiles, item_profiles = build_profiles(df, all_embeddings)
    
    # Salviamo i profili
    np.save(os.path.join(EMBEDDINGS_DIR, "user_profiles.npy"), user_profiles, allow_pickle=True)
    np.save(os.path.join(EMBEDDINGS_DIR, "item_profiles.npy"), item_profiles, allow_pickle=True)
    logger.info("Pipeline completata.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_embedding_pipeline()
