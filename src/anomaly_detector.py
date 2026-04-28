import polars as pl
import os
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import FEATURES_PATH, ISOLATION_FOREST_ESTIMATORS, ISOLATION_FOREST_CONTAMINATION, RANDOM_SEED

logger = logging.getLogger(__name__)

def extract_trust_scores(features_df: pl.DataFrame) -> pl.DataFrame:
    """
    Applica Isolation Forest alle feature comportamentali per estrarre l'Anomaly Score,
    e lo normalizza tra 0 e 1 per ottenere il Trust Score.
    """
    logger.info("Avvio Isolation Forest basato su feature comportamentali...")
    
    # 1. Preparazione matrice delle feature numeriche
    feature_cols = [c for c in features_df.columns if c != "user_id"]
    X = features_df.select(feature_cols).to_numpy()
    
    # Standard scaling dei dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Addestramento IF
    iso_forest = IsolationForest(
        n_estimators=ISOLATION_FOREST_ESTIMATORS,
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    logger.info("Fit Isolation Forest...")
    iso_forest.fit(X_scaled)
    
    # 3. Decision function: valori negativi = anomalie, valori positivi = normalità
    scores = iso_forest.decision_function(X_scaled)
    
    # 4. Normalizzazione in [0, 1] per ottenere il Trust Score
    # Il min è l'anomalia più forte (score ~0), il max è la massima normalità (score ~1)
    s_min, s_max = np.min(scores), np.max(scores)
    # Evito division by zero nel caso teorico di score costanti
    if s_max - s_min > 0:
        trust_scores = (scores - s_min) / (s_max - s_min)
    else:
        trust_scores = np.ones_like(scores)
        
    logger.info(f"Trust Score calcolato. Min: {np.min(trust_scores):.4f}, Max: {np.max(trust_scores):.4f}")
    
    # 5. Riassegniamo i risultati al dataframe per restituirlo
    enhanced_df = features_df.with_columns(
        pl.Series("trust_score", trust_scores)
    )
    
    return enhanced_df

def run_anomaly_detection():
    if not os.path.exists(FEATURES_PATH):
        logger.error(f"File {FEATURES_PATH} mancante. Eseguire la feature engineering prima.")
        return
        
    logger.info("Caricamento dataset feature...")
    features_df = pl.read_parquet(FEATURES_PATH)
    
    enhanced_df = extract_trust_scores(features_df)
    
    # Sovrascriviamo le feature con l'aggiunta di trust_score
    enhanced_df.write_parquet(FEATURES_PATH)
    logger.info(f"Dataset feature arricchito salvato in {FEATURES_PATH}")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_anomaly_detection()
