import pytest
import numpy as np
import polars as pl
from src.fusion import calculate_item_trust_factors, generate_ranking

def test_late_fusion_ranking():
    # Creiamo configurazione mock: 2 User, 3 Item
    
    df = pl.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "parent_asin": ["i1", "i2", "i2", "i3"]
    })
    
    features_df = pl.DataFrame({
        "user_id": ["u1", "u2"],
        "trust_score": [1.0, 0.0] # u1 è legit, u2 è uno spammer
    })
    
    # Simuliando profile vettoriali 2D (ortogonali e paralleli testabili)
    user_profiles = {
        "u1": np.array([1.0, 0.0]),
        "u2": np.array([0.0, 1.0])
    }
    
    item_profiles = {
        "i1": np.array([1.0, 0.0]), # match perfetto con U1 (Sim=1)
        "i2": np.array([0.7, 0.7]), # match parziale
        "i3": np.array([1.0, 1.0])  # spinto fortemente da U2 (che è spammer)
    }
    
    # 1. Test Trust Factors per item
    # i1 = recensito solo da u1 (1.0) -> Trust = 1.0
    # i2 = recensito da u1(1.0) e u2(0.0) -> Trust = 0.5
    # i3 = recensito solo da u2 (0.0) -> Trust = 0.0
    item_trust_factors = calculate_item_trust_factors(df, features_df)
    assert item_trust_factors["i1"] == pytest.approx(1.0)
    assert item_trust_factors["i2"] == pytest.approx(0.5)
    assert item_trust_factors["i3"] == pytest.approx(0.0)
    
    # 2. Test Baseline Ranking (Senza trust)
    baseline_ranking = generate_ranking(user_profiles, item_profiles, item_trust_factors, top_k=3, use_trust=False)
    
    # Per u1 sim con i1(1.0), i2(~0.7), i3(~0.7)
    # Per u2 sim con i3(0.7), i2(0.7)
    # U2 Baseline top 1 sarà I3 perche i3 [1,1] cos sim [0,1] = ~0.707
    
    assert "i1" in baseline_ranking["u1"]
    
    # 3. Test Trust-Aware Ranking
    trust_ranking = generate_ranking(user_profiles, item_profiles, item_trust_factors, top_k=3, use_trust=True)
    
    # Per U2: i3 è fortemente spinto, MA il suo trust_factor è 0.0 (poiché recensito solo dallo spammer u2).
    # Nello score finale i3 deve crollare a 0.0 e non essere il top 1
    # i2 che ha trust 0.5 avrà uno score maggiore per U2 rispetto ad i3.
    assert trust_ranking["u2"][0] == "i2"  # Rank Shift dimostrato! i3 precipita.
    
    # Extra check
    # Assicura che i1 resti forte per u1 dato il trust 1.0
    assert trust_ranking["u1"][0] == "i1"
