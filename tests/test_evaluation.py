import pytest
from src.evaluation import ndcg_at_k, precision_at_k, calculate_rank_shift, evaluate_recommender

def test_ndcg_at_k():
    recs = ["a", "b", "c", "d", "e"]
    truth = {"a", "c"}
    
    # a è in pos 1 (index 0) => rel = 1.0. Gain = 1 / log2(2) = 1.0
    # c è in pos 3 (index 2) => rel = 1.0. Gain = 1 / log2(4) = 0.5
    # DCG = 1.5
    # IDCG (ideal): a, c primi posti => 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63
    # nDCG = 1.5 / 1.6309 = 0.919...
    
    ndcg = ndcg_at_k(recs, truth, 5)
    assert ndcg == pytest.approx(0.919, 0.01)

def test_precision_at_k():
    recs = ["a", "b", "c", "d", "e"]
    truth = {"a", "c"}
    
    p3 = precision_at_k(recs, truth, 3) # a, b, c -> hits=2 -> 2/3 = 0.66
    assert p3 == pytest.approx(0.666, 0.01)
    
    p5 = precision_at_k(recs, truth, 5) # hits=2 -> 2/5 = 0.4
    assert p5 == 0.4

def test_calculate_rank_shift():
    baseline = {
        "u1": ["i1", "i2", "i3"],
        "u2": ["i1", "i4", "i2"]
    }
    
    trust_ranking = {
        "u1": ["i2", "i1", "i3"], # per u1: i1 scende da pos.0 a pos.1 (shift -1)
        "u2": ["i4", "i2", "i1"]  # per u2: i1 scende da pos.0 a pos.2 (shift -2)
    }
    
    shifts = calculate_rank_shift(baseline, trust_ranking)
    
    # media shift per i1: (-1 - 2) / 2 = -1.5
    assert shifts["i1"] == -1.5
    
    # i2 sale da pos 1 a 0 per u1 (shift +1)
    # i2 sale da pos 2 a 1 per u2 (shift +1)
    assert shifts["i2"] == 1.0

def test_evaluate_recommender():
    preds = {
        "u1": ["a", "b", "c"],
        "u2": ["x", "y", "z"]
    }
    test_set = {
        "u1": {"a"}, # 1 hit in pos 0
        "u2": {"z"}  # 1 hit in pos 2
    }
    
    metrics = evaluate_recommender(preds, test_set, k_list=[3])
    
    # Precision@3 = U1(1/3) + U2(1/3) / 2 = 1/3
    assert metrics["Precision@3"] == pytest.approx(0.333, 0.01)
