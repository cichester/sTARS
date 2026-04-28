import numpy as np
import polars as pl
import logging
from config import PARQUET_5CORE_PATH

logger = logging.getLogger(__name__)

def precision_at_k(recommended_items: list, ground_truth: set, k: int) -> float:
    """Calcola la Precision@K."""
    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / k

def ndcg_at_k(recommended_items: list, ground_truth: set, k: int) -> float:
    """Calcola il Normalized Discounted Cumulative Gain (nDCG@K)."""
    dcg = 0.0
    idcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        rel = 1.0 if item in ground_truth else 0.0
        dcg += rel / np.log2(i + 2)
        
    for i in range(min(len(ground_truth), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def calculate_rank_shift(baseline_ranking: dict, trust_ranking: dict) -> dict:
    """
    Calcola il Rank Shift medio per item.
    Rank Shift negativo = l'item ha perso posizioni nel Trust-Aware.
    """
    item_shifts = {} # dict di liste
    
    for user, base_rec in baseline_ranking.items():
        if user not in trust_ranking:
            continue
            
        trust_rec = trust_ranking[user]
        
        # Mappiamo le posizioni
        base_pos = {item: rank for rank, item in enumerate(base_rec)}
        trust_pos = {item: rank for rank, item in enumerate(trust_rec)}
        
        for item, old_p in base_pos.items():
            new_p = trust_pos.get(item, len(base_rec)) # se sparito, poniamo in fondo
            shift = old_p - new_p # Es: era 1, ora 3. Shift = 1 - 3 = -2 (ha perso posizioni)
            
            if item not in item_shifts:
                item_shifts[item] = []
            item_shifts[item].append(shift)
            
    avg_shifts = {item: np.mean(shifts) for item, shifts in item_shifts.items()}
    return avg_shifts

def evaluate_recommender(predictions: dict, test_set: dict, k_list: list = [5, 10, 20]) -> dict:
    """Valuta le generiche predizioni con nDCG e Precision."""
    metrics = {f"nDCG@{k}": [] for k in k_list}
    metrics.update({f"Precision@{k}": [] for k in k_list})
    
    for user, truth in test_set.items():
        if user not in predictions or not truth:
            continue
            
        recs = predictions[user]
        
        for k in k_list:
            metrics[f"nDCG@{k}"].append(ndcg_at_k(recs, truth, k))
            metrics[f"Precision@{k}"].append(precision_at_k(recs, truth, k))
            
    # Averaging
    return {met_name: np.mean(vals) if vals else 0.0 for met_name, vals in metrics.items()}

def run_evaluation_pipeline():
    logger.info("Avvio valutazione pipeline...")
    # Qui andrebbe il codice per splittare temporale e calcolare tutto live.
    # Abbiamo test unitari solidi per mostrare che la logica funziona. 
    logger.info("Fine evaluation module.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation_pipeline()
