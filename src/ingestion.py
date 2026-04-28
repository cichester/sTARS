import polars as pl
import logging
from config import RAW_DATA_PATH, PARQUET_5CORE_PATH, MIN_INTERACTIONS
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def find_5core_ids(raw_path: str, min_interactions: int) -> tuple[set, set]:
    """
    Legge solo user_id e parent_asin per calcolare iterativamente i core ID.
    Usa Polars per un memory footprint ridotto.
    """
    logger.info("Caricamento selettivo (user_id, parent_asin) dal dataset raw...")
    # Leggiamo solo le due colonne necessarie in memoria. 
    # Con 44M records, 2 colonne stringa pesano circa 1-2 GB
    df = pl.read_ndjson(raw_path)[["user_id", "parent_asin"]]
    
    iteration = 0
    while True:
        iteration += 1
        initial_rows = df.height
        
        # Filtro per users validi
        user_counts = df.group_by("user_id").len()
        valid_users = user_counts.filter(pl.col("len") >= min_interactions).select("user_id")
        df = df.join(valid_users, on="user_id", how="inner")
        
        # Filtro per item validi
        item_counts = df.group_by("parent_asin").len()
        valid_items = item_counts.filter(pl.col("len") >= min_interactions).select("parent_asin")
        df = df.join(valid_items, on="parent_asin", how="inner")
        
        final_rows = df.height
        logger.info(f"Iterazione {iteration}: {initial_rows} -> {final_rows} interazioni residue")
        
        if initial_rows == final_rows:
            break
            
    # Estraiamo i set puri
    final_users = set(df["user_id"].to_list())
    final_items = set(df["parent_asin"].to_list())
    
    logger.info(f"5-core convergenza raggiunta! Utenti: {len(final_users)}, Item: {len(final_items)}")
    return final_users, final_items


def extract_and_save_5core(raw_path: str, out_path: str, valid_users: set, valid_items: set):
    """
    Rilegge il file JSONL riga per riga, scrivendo solo i record validi in un array intermedio
    (o leggendoli a chunk) per materializzare il dataset finale senza Memory OOM.
    Qui usiamo pl.scan_ndjson che fa filter pushdown se possibile, 
    oppure filter dopo il load. Considerando 21GB, se abbiamo abbastanza RAM possiamo leggere tutto
    e filtrare. Se no, usiamo batch/streaming.
    """
    logger.info("Scansione per generare il dataset 5-core...")
    
    # Siccome l'operatore is_in su set molto grandi in memoria distribuita (lazy) 
    # può essere pesante, castiamo i set a Series di polars.
    s_valid_users = pl.Series("user_id", list(valid_users))
    s_valid_items = pl.Series("parent_asin", list(valid_items))
    
    # Read/Scan filter
    lf = pl.scan_ndjson(raw_path)
    
    # Applico filtro (Join e' più efficiente di is_in per grandi volumi)
    lf_valid_users = pl.LazyFrame(s_valid_users)
    lf_valid_items = pl.LazyFrame(s_valid_items)
    
    result_lf = (
        lf.join(lf_valid_users, on="user_id", how="inner")
          .join(lf_valid_items, on="parent_asin", how="inner")
    )
    
    logger.info("Materializzazione in corso (potrebbe richiedere tempo)...")
    # Usa use_pyarrow per migliorare I/O se disponibile o streaming if out of memory
    # ma visto il filtering aggressivo, collect() dovrebbe stare nella RAM (siamo a ~2-5M records)
    final_df = result_lf.collect()
    
    logger.info(f"Dataset materializzato: {final_df.height} records finali.")
    final_df.write_parquet(out_path)
    logger.info(f"Salvato in: {out_path}")


def run_ingestion():
    if os.path.exists(PARQUET_5CORE_PATH):
        logger.info(f"Il file {PARQUET_5CORE_PATH} esiste già. Ingestion skippata.")
        return

    users_5core, items_5core = find_5core_ids(RAW_DATA_PATH, MIN_INTERACTIONS)
    extract_and_save_5core(RAW_DATA_PATH, PARQUET_5CORE_PATH, users_5core, items_5core)

if __name__ == "__main__":
    run_ingestion()
