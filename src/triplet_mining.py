import polars as pl
import numpy as np
import logging
import os
from config import DATA_DIR, TRIPLETS_PATH, NUM_TRIPLETS, RANDOM_SEED

logger = logging.getLogger(__name__)

def mine_triplets():
    """Mina triple contrastive (anchor, positive, negative) dal dataset clean."""
    if os.path.exists(TRIPLETS_PATH):
        logger.info(f"File {TRIPLETS_PATH} già esistente. Skip mining.")
        return

    logger.info("Iniziando il mining delle triple contrastive...")
    clean_path = DATA_DIR / "electronics_5core_clean.parquet"
    if not os.path.exists(clean_path):
        logger.error(f"File {clean_path} non trovato!")
        return
        
    df = pl.read_parquet(clean_path)
    # Filtriamo review molto corte per la qualità del dataset
    df = df.filter(pl.col("review_text").str.len_chars() > 20)
    
    # Raggruppiamo recensioni per item (parent_asin)
    logger.info("Raggruppamento per item...")
    df_grouped = df.group_by("parent_asin").agg(pl.col("review_text"))
    
    # Teniamo solo gli item con almeno 2 recensioni (serve per anchor e positive)
    df_grouped = df_grouped.filter(pl.col("review_text").list.len() >= 2)
    
    # Convertiamo a dizionario o lista per campionamento veloce
    items = df_grouped["parent_asin"].to_list()
    reviews_per_item = df_grouped["review_text"].to_list()
    
    np.random.seed(RANDOM_SEED)
    num_items = len(items)
    
    triplets = []
    logger.info(f"Generazione di {NUM_TRIPLETS} triple...")
    
    for _ in range(NUM_TRIPLETS):
        # Scegliamo un item random per l'anchor e il positive
        item_idx = np.random.randint(0, num_items)
        item_reviews = reviews_per_item[item_idx]
        
        # Campioniamo 2 recensioni diverse dallo stesso item
        idx1, idx2 = np.random.choice(len(item_reviews), size=2, replace=False)
        anchor = item_reviews[idx1]
        positive = item_reviews[idx2]
        
        # Scegliamo un item diverso per il negative
        neg_item_idx = np.random.randint(0, num_items)
        while neg_item_idx == item_idx:
            neg_item_idx = np.random.randint(0, num_items)
            
        neg_item_reviews = reviews_per_item[neg_item_idx]
        neg_idx = np.random.randint(0, len(neg_item_reviews))
        negative = neg_item_reviews[neg_idx]
        
        triplets.append({"anchor_text": anchor, "positive_text": positive, "negative_text": negative})
        
        if len(triplets) % 100_000 == 0:
            logger.info(f"Generati {len(triplets)} / {NUM_TRIPLETS}...")

    triplets_df = pl.DataFrame(triplets)
    logger.info(f"Salvataggio triple in {TRIPLETS_PATH}...")
    triplets_df.write_parquet(TRIPLETS_PATH)
    logger.info("Mining completato.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mine_triplets()
