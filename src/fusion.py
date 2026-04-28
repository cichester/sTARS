import polars as pl
import numpy as np
import os
import logging
from config import PARQUET_5CORE_PATH, FEATURES_PATH, EMBEDDINGS_DIR, TOP_K_VALUES
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def calculate_item_trust_factors(df: pl.DataFrame, features_df: pl.DataFrame) -> dict:
    """
    Calcola il Trust Factor per ogni item.
    Trust Factor = Media dei trust_score degli utenti che hanno recensito l'item.
    """
    logger.info("Calcolo Item Trust Factors...")
    # Join dataset con trust_scores (da features)
    user_trust = features_df.select(["user_id", "trust_score"])
    df_joined = df.join(user_trust, on="user_id", how="left").fill_null(1.0) # Se manca, assumiamo trust 1.0 (Safe)
    
    # Aggregazione per item
    item_trust_df = df_joined.group_by("parent_asin").agg(
        pl.col("trust_score").mean().alias("item_trust_factor")
    )
    
    # Ritorna dict
    items = item_trust_df["parent_asin"].to_numpy()
    trusts = item_trust_df["item_trust_factor"].to_numpy()
    
    return dict(zip(items, trusts))


def generate_ranking(user_profiles: dict, item_profiles: dict, item_trust_factors: dict, top_k: int = 10, use_trust: bool = True) -> dict:
    """
    Per ogni utente, genera la top K di item raccomandati.
    Applica la Late Fusion: Score = Cosine Similarity * Item_Trust_Factor se use_trust è True.
    Calcolo in batch per evitare Out of Memory (OOM) su grandi dataset.
    """
    logger.info(f"Generazione Ranking Top-{top_k} (Trust-Aware: {use_trust})")
    
    users = list(user_profiles.keys())
    items = list(item_profiles.keys())
    
    X_items = np.array([item_profiles[i] for i in items])
    
    if use_trust:
        trust_array = np.array([item_trust_factors.get(i, 1.0) for i in items])
    
    recommendations = {}
    batch_size = 1000
    num_users = len(users)
    
    logger.info("Calcolando Cosine Similarity e Ranking in batch...")
    
    for start_idx in range(0, num_users, batch_size):
        end_idx = min(start_idx + batch_size, num_users)
        batch_users = users[start_idx:end_idx]
        X_users_batch = np.array([user_profiles[u] for u in batch_users])
        
        # Calcolo similarity per il batch corrente
        sim_matrix = cosine_similarity(X_users_batch, X_items)
        sim_matrix = np.maximum(sim_matrix, 0)
        
        # Moltiplicazione Late Fusion
        if use_trust:
            final_scores = sim_matrix * trust_array
        else:
            final_scores = sim_matrix
            
        for row_idx, user in enumerate(batch_users):
            user_scores = final_scores[row_idx]
            
            if len(user_scores) <= top_k:
                top_indices = np.argsort(-user_scores)
            else:
                top_indices = np.argpartition(-user_scores, top_k)[:top_k]
                top_scores = user_scores[top_indices]
                sorted_k = np.argsort(-top_scores)
                top_indices = top_indices[sorted_k]
                
            recommendations[user] = [items[idx] for idx in top_indices]
            
        if (start_idx // batch_size) % 10 == 0 and start_idx > 0:
            logger.info(f"  Elaborati {start_idx}/{num_users} utenti...")
            
    logger.info(f"  Elaborati {num_users}/{num_users} utenti.")
    return recommendations

def run_fusion_pipeline():
    logger.info("Caricando Dati per Fusion Pipeline...")
    
    df = pl.read_parquet(PARQUET_5CORE_PATH)
    features_df = pl.read_parquet(FEATURES_PATH)
    
    user_prof_path = os.path.join(EMBEDDINGS_DIR, "user_profiles.npy")
    item_prof_path = os.path.join(EMBEDDINGS_DIR, "item_profiles.npy")
    
    user_profiles = np.load(user_prof_path, allow_pickle=True).item()
    item_profiles = np.load(item_prof_path, allow_pickle=True).item()
    
    item_trust_factors = calculate_item_trust_factors(df, features_df)
    
    # Baseline ranking (Senza Trust)
    baseline_ranking = generate_ranking(user_profiles, item_profiles, item_trust_factors, top_k=20, use_trust=False)
    
    # Trust-Aware ranking
    trust_ranking = generate_ranking(user_profiles, item_profiles, item_trust_factors, top_k=20, use_trust=True)
    
    # Salviamo risultati
    import pickle
    os.makedirs("data/rankings", exist_ok=True)
    
    with open("data/rankings/baseline.pkl", "wb") as f:
        pickle.dump(baseline_ranking, f)
        
    with open("data/rankings/trust_aware.pkl", "wb") as f:
        pickle.dump(trust_ranking, f)
        
    logger.info("Rankings salvati con successo. Sprint 5 terminato.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_fusion_pipeline()
