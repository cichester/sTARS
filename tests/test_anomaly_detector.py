import pytest
import polars as pl
import numpy as np
from src.anomaly_detector import extract_trust_scores

def test_extract_trust_scores():
    # Creiamo 100 utenti "normali" e 5 utenti "anomali"
    
    np.random.seed(42)
    # Generiamo feature normali: bassi burstiness, rating entropy mista
    normal_data = {
        "user_id": [f"U_N_{i}" for i in range(100)],
        "review_count": np.random.randint(1, 10, 100),
        "burstiness": np.random.randint(1, 3, 100),
        "rating_entropy": np.random.uniform(0.5, 2.0, 100),
        "pct_extreme": np.random.uniform(0.1, 0.4, 100)
    }
    
    # Generiamo feature anomale: burstiness estrema, tutto 5.0 (pct_extreme=1, entropy=0)
    anomaly_data = {
        "user_id": [f"U_A_{i}" for i in range(5)],
        "review_count": np.random.randint(50, 100, 5),
        "burstiness": np.random.randint(45, 90, 5),
        "rating_entropy": np.zeros(5),
        "pct_extreme": np.ones(5)
    }
    
    # Unione dei due dataframe
    df_n = pl.DataFrame(normal_data)
    df_a = pl.DataFrame(anomaly_data)
    df = pl.concat([df_n, df_a])
    
    # Eseguiamo isolation forest
    enhanced_df = extract_trust_scores(df)
    
    assert "trust_score" in enhanced_df.columns
    trust_scores = enhanced_df["trust_score"].to_numpy()
    
    # Verifica limiti
    assert np.all(trust_scores >= 0.0)
    assert np.all(trust_scores <= 1.0)
    
    # Gli utenti anomali (gli ultimi 5) dovrebbero avere un trust score minore rispetto alla media dei normali
    normal_avg_trust = np.mean(enhanced_df.filter(pl.col("user_id").str.starts_with("U_N_"))["trust_score"].to_numpy())
    anomalous_avg_trust = np.mean(enhanced_df.filter(pl.col("user_id").str.starts_with("U_A_"))["trust_score"].to_numpy())
    
    assert anomalous_avg_trust < normal_avg_trust
    
    # Check that maximum element is exactly or close to 1
    assert np.max(trust_scores) == pytest.approx(1.0)
    assert np.min(trust_scores) == pytest.approx(0.0)
