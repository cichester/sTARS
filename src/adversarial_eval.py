import os
import logging
import time
import numpy as np
import polars as pl

from config import (
    DATA_DIR, FEATURES_PATH, EMBEDDINGS_DIR, RESULTS_DIR, 
    TEST_SIZE_RATIO, TOP_K_VALUES, EMBEDDINGS_ROBERTA_DIR, ROBERTA_ADAPTER_DIR
)
from src.attack_generator import (
    select_target_item, generate_average_attack, generate_bandwagon_attack, generate_genai_attack, inject_bots
)
from src.feature_engineering import extract_behavioral_features
from src.anomaly_detector import extract_trust_scores
from src.fusion import calculate_item_trust_factors, generate_ranking
from src.evaluation import evaluate_recommender, calculate_rank_shift

logger = logging.getLogger(__name__)

def run_attack_evaluation(clean_df: pl.DataFrame, features_df: pl.DataFrame, 
                          user_profiles: dict, item_profiles: dict, 
                          attack_type: str, num_bots: int = 50):
    """Esegue l'attacco, ricalcola le componenti necessarie e valuta il Rank Shift del target."""
    
    logger.info(f"--- Avvio Evaluazione Attacco: {attack_type.upper()} ---")
    
    target_item = select_target_item(clean_df)
    
    # 1. Generazione Bot
    if attack_type == "average":
        bot_df = generate_average_attack(clean_df, target_item, num_bots=num_bots)
    elif attack_type == "bandwagon":
        bot_df = generate_bandwagon_attack(clean_df, target_item, num_bots=num_bots)
    elif attack_type == "genai":
        bot_df = generate_genai_attack(clean_df, target_item, num_bots=num_bots)
    else:
        raise ValueError(f"Tipo di attacco non supportato: {attack_type}")
        
    bot_users = bot_df["user_id"].unique().to_list()
    
    # 2. Iniezione nel dataset
    poisoned_df = inject_bots(clean_df, bot_df)
    
    # 3. Ricalcolo Feature Comportamentali (incremental)
    bot_features = extract_behavioral_features(bot_df)
    
    # Uniamo con le feature originali (senza trust_score)
    clean_features_no_trust = features_df.select([c for c in features_df.columns if c != "trust_score"])
    
    # Assicuriamo che tutte le colonne combacino
    common_cols = [c for c in clean_features_no_trust.columns if c in bot_features.columns]
    
    poisoned_features = pl.concat(
        [clean_features_no_trust.select(common_cols), bot_features.select(common_cols)], 
        how="vertical"
    )
    
    # 4. Ricalcolo Anomaly Detection
    poisoned_features_with_trust = extract_trust_scores(poisoned_features)
    
    # Bot Detection Rate: quanti bot hanno trust < 0.3?
    bot_trust_scores = poisoned_features_with_trust.filter(pl.col("user_id").is_in(bot_users))["trust_score"]
    detection_rate = (bot_trust_scores < 0.3).sum() / len(bot_users)
    logger.info(f"Bot Detection Rate (Trust < 0.3): {detection_rate:.2%}")
    
    # 5. Incremental Embedding (mocked)
    poisoned_user_profiles = user_profiles.copy()
    
    # Rileviamo dimensione embedding dinamicamente
    sample_key = next(iter(item_profiles.keys()))
    sample_val = item_profiles[sample_key]
    emb_dim = sample_val.shape[0] if hasattr(sample_val, "shape") else len(sample_val)
    
    for bot_id in bot_users:
        bot_item_reviews = bot_df.filter(pl.col("user_id") == bot_id)["parent_asin"].to_list()
        bot_emb = np.mean([item_profiles.get(item, np.zeros(emb_dim)) for item in bot_item_reviews], axis=0)
        poisoned_user_profiles[bot_id] = bot_emb
        
    # 6. Evaluation (Rank Shift)
    eval_users = list(user_profiles.keys())[:2000]
    eval_user_profiles_dict = {u: poisoned_user_profiles[u] for u in eval_users}
    
    # Calcolo Trust Factors sul dataset poisoned
    item_trust_factors_poisoned = calculate_item_trust_factors(poisoned_df, poisoned_features_with_trust)
    
    logger.info("Generazione Ranking Baseline (Poisoned)...")
    base_poisoned_rank = generate_ranking(eval_user_profiles_dict, item_profiles, item_trust_factors_poisoned, top_k=50, use_trust=False)
    
    logger.info("Generazione Ranking Trust-Aware (Poisoned)...")
    trust_poisoned_rank = generate_ranking(eval_user_profiles_dict, item_profiles, item_trust_factors_poisoned, top_k=50, use_trust=True)
    
    # Troviamo la posizione media del Target Item
    def get_avg_pos(ranking_dict, target):
        positions = []
        for u, recs in ranking_dict.items():
            if target in recs:
                positions.append(recs.index(target))
        return np.mean(positions) if positions else 50.0
        
    target_pos_base = get_avg_pos(base_poisoned_rank, target_item)
    target_pos_trust = get_avg_pos(trust_poisoned_rank, target_item)
    
    # Hit Rate: in quante top-20 compare il target?
    def get_hit_rate(ranking_dict, target, k=20):
        hits = sum(1 for u, recs in ranking_dict.items() if target in recs[:k])
        return hits / len(ranking_dict)
        
    hit_rate_base = get_hit_rate(base_poisoned_rank, target_item)
    hit_rate_trust = get_hit_rate(trust_poisoned_rank, target_item)
    
    logger.info(f"Posizione media Target (Baseline): {target_pos_base:.2f}")
    logger.info(f"Posizione media Target (Trust-Aware): {target_pos_trust:.2f}")
    logger.info(f"Hit Rate@20 Target (Baseline): {hit_rate_base:.2%}")
    logger.info(f"Hit Rate@20 Target (Trust-Aware): {hit_rate_trust:.2%}")
    
    return {
        "attack_type": attack_type,
        "target_item": target_item,
        "detection_rate": detection_rate,
        "pos_base": target_pos_base,
        "pos_trust": target_pos_trust,
        "hit_base": hit_rate_base,
        "hit_trust": hit_rate_trust
    }

def run_adversarial_for_model(clean_df: pl.DataFrame, features_df: pl.DataFrame, model_type: str = "sbert"):
    """Esegue la valutazione per un singolo modello (SBERT o RoBERTa)."""
    
    if model_type == "sbert":
        user_prof_path = os.path.join(EMBEDDINGS_DIR, "user_profiles.npy")
        item_prof_path = os.path.join(EMBEDDINGS_DIR, "item_profiles.npy")
    elif model_type == "roberta":
        user_prof_path = os.path.join(EMBEDDINGS_ROBERTA_DIR, "user_profiles.npy")
        item_prof_path = os.path.join(EMBEDDINGS_ROBERTA_DIR, "item_profiles.npy")
    else:
        raise ValueError(f"Modello non supportato: {model_type}")
        
    if not os.path.exists(user_prof_path):
        logger.warning(f"Profili per {model_type} non trovati in {user_prof_path}. Salto valutazione.")
        return None
        
    logger.info(f"Caricamento profili {model_type}...")
    user_profiles = np.load(user_prof_path, allow_pickle=True).item()
    item_profiles = np.load(item_prof_path, allow_pickle=True).item()
    
    bot_counts = [50, 100, 200]
    results_avg = {}
    results_bw = {}
    results_genai = {}
    
    for count in bot_counts:
        results_avg[count] = run_attack_evaluation(clean_df, features_df, user_profiles, item_profiles, "average", count)
        results_bw[count] = run_attack_evaluation(clean_df, features_df, user_profiles, item_profiles, "bandwagon", count)
        results_genai[count] = run_attack_evaluation(clean_df, features_df, user_profiles, item_profiles, "genai", count)
        
    return {"average": results_avg, "bandwagon": results_bw, "genai": results_genai}

def run_adversarial_evaluation():
    logger.info("=== Avvio Adversarial Evaluation ===")
    
    clean_path = DATA_DIR / "electronics_5core_clean.parquet"
    if not os.path.exists(clean_path):
        logger.error(f"File {clean_path} non trovato.")
        return
        
    clean_df = pl.read_parquet(clean_path)
    features_df = pl.read_parquet(FEATURES_PATH)
    
    # Eseguiamo per SBERT
    results = {}
    results["SBERT"] = run_adversarial_for_model(clean_df, features_df, "sbert")
    
    # Eseguiamo per RoBERTa (se l'adapter o la cartella esiste)
    if os.path.exists(EMBEDDINGS_ROBERTA_DIR / "user_profiles.npy"):
        results["RoBERTa"] = run_adversarial_for_model(clean_df, features_df, "roberta")
        
    # Scrittura Report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, f"adversarial_report_{time.strftime('%Y%m%d_%H%M%S')}.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Adversarial Evaluation Report\n\n")
        f.write(f"**Data:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("Questo report confronta la resilienza del sistema Baseline rispetto al Trust-Aware sotto attacchi di Data Poisoning a intensità progressiva.\n\n")
        
        for model_name, model_res in results.items():
            if model_res is None: continue
            
            f.write(f"## Modello: {model_name}\n\n")
            
            f.write("### 1. Average Attack\n\n")
            f.write("| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n")
            for count in [50, 100, 200]:
                res = model_res["average"][count]
                f.write(f"| {count} | `{res['target_item']}` | {res['detection_rate']:.2%} | {res['pos_base']:.2f} | {res['pos_trust']:.2f} | {res['hit_base']:.2%} | {res['hit_trust']:.2%} |\n")
            f.write("\n")
            
            f.write("### 2. Bandwagon Attack\n\n")
            f.write("| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n")
            for count in [50, 100, 200]:
                res = model_res["bandwagon"][count]
                f.write(f"| {count} | `{res['target_item']}` | {res['detection_rate']:.2%} | {res['pos_base']:.2f} | {res['pos_trust']:.2f} | {res['hit_base']:.2%} | {res['hit_trust']:.2%} |\n")
            f.write("\n")
            
            f.write("### 3. GenAI Attack\n\n")
            f.write("| Num Bot | Target Item | Bot Detection Rate | Posizione Baseline | Posizione Trust-Aware | Hit Rate@20 Baseline | Hit Rate@20 Trust-Aware |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n")
            for count in [50, 100, 200]:
                res = model_res["genai"][count]
                f.write(f"| {count} | `{res['target_item']}` | {res['detection_rate']:.2%} | {res['pos_base']:.2f} | {res['pos_trust']:.2f} | {res['hit_base']:.2%} | {res['hit_trust']:.2%} |\n")
            f.write("\n---\n\n")
            
        f.write("> **Conclusioni:** Se la posizione media nel Trust-Aware è vicina a 50 (fuori ranking) e l'Hit Rate è prossimo allo 0%, mentre nella Baseline sono alti, significa che l'Anomaly Detection ha neutralizzato con successo l'attacco, dimostrando la robustezza del sistema.\n")
        
    logger.info(f"Adversarial Report salvato in {report_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_adversarial_evaluation()
