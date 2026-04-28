import polars as pl
import pytest
from src.feature_engineering import extract_behavioral_features
from datetime import datetime, timedelta

def test_extract_behavioral_features():
    # Creiamo un po' di dati fittizi
    # Utente 1: 3 recensioni, tutte 5 stelle nello stesso giorno (alta burstiness, bassa entropia)
    # Utente 2: 2 recensioni, giorni diversi, rating 1 e 5 (alta entropia 1.0)
    
    base_time = int(datetime(2023, 1, 1).timestamp() * 1000)
    day_ms = 86400 * 1000
    
    data = [
        # User 1
        {"user_id": "U1", "parent_asin": "A1", "timestamp": base_time, "rating": 5.0, "verified_purchase": True, "helpful_vote": 2, "text": "great!"},
        {"user_id": "U1", "parent_asin": "A2", "timestamp": base_time + 1000, "rating": 5.0, "verified_purchase": True, "helpful_vote": 1, "text": "awesome!"},
        {"user_id": "U1", "parent_asin": "A3", "timestamp": base_time + 2000, "rating": 5.0, "verified_purchase": False, "helpful_vote": 0, "text": "nice!"},
        
        # User 2
        {"user_id": "U2", "parent_asin": "A1", "timestamp": base_time, "rating": 1.0, "verified_purchase": True, "helpful_vote": 10, "text": "terrible product"},
        {"user_id": "U2", "parent_asin": "A2", "timestamp": base_time + day_ms * 2, "rating": 5.0, "verified_purchase": True, "helpful_vote": 0, "text": "perfect!"},
    ]
    
    df = pl.DataFrame(data)
    features = extract_behavioral_features(df)
    
    assert features.height == 2
    
    # Check U1
    u1 = features.filter(pl.col("user_id") == "U1").to_dicts()[0]
    assert u1["review_count"] == 3
    assert u1["avg_rating"] == 5.0
    assert u1["burstiness"] == 3  # 3 in the same day
    assert u1["rating_entropy"] == 0.0 # Only one rating value (5.0), so p=1.0 -> log2(1)=0
    assert pytest.approx(u1["pct_extreme"]) == 1.0
    assert pytest.approx(u1["pct_verified"]) == 2 / 3
    assert u1["avg_helpful"] == 1.0 # (2+1+0)/3
    assert u1["activity_span_days"] == 0
    assert u1["unique_items_ratio"] == 1.0
    
    # Check U2
    u2 = features.filter(pl.col("user_id") == "U2").to_dicts()[0]
    assert u2["review_count"] == 2
    assert u2["avg_rating"] == 3.0
    assert u2["burstiness"] == 1 # Different days
    assert pytest.approx(u2["rating_entropy"]) == 1.0 # Due valori equidistribuiti: 2 * (0.5 * log2(2)) = 1.0
    assert u2["activity_span_days"] == 2
    assert u2["pct_extreme"] == 1.0 # Tutti e due sono in (1.0, 5.0)
    assert u2["pct_verified"] == 1.0
    assert u2["avg_helpful"] == 5.0
