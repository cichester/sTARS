import pytest
from unittest.mock import patch, MagicMock
from src.adversarial_eval import run_attack_evaluation
import polars as pl

@pytest.fixture
def mock_data():
    clean_df = pl.DataFrame({
        "user_id": ["U1", "U2", "U3"],
        "parent_asin": ["ITEM1", "ITEM2", "ITEM1"],
        "rating": [5.0, 4.0, 5.0],
        "timestamp": [1, 2, 3]
    })
    
    features_df = pl.DataFrame({
        "user_id": ["U1", "U2", "U3"],
        "trust_score": [0.9, 0.8, 0.95],
        "review_count": [1, 1, 1]
    })
    
    user_profiles = {"U1": [0.1], "U2": [0.2], "U3": [0.1]}
    item_profiles = {"ITEM1": [0.1], "ITEM2": [0.2]}
    
    return clean_df, features_df, user_profiles, item_profiles

@patch("src.adversarial_eval.select_target_item")
@patch("src.adversarial_eval.generate_average_attack")
@patch("src.adversarial_eval.extract_behavioral_features")
@patch("src.adversarial_eval.extract_trust_scores")
@patch("src.adversarial_eval.generate_ranking")
def test_run_attack_evaluation(mock_ranking, mock_trust, mock_features, mock_attack, mock_select, mock_data):
    clean_df, features_df, user_profiles, item_profiles = mock_data
    
    mock_select.return_value = "ITEM1"
    
    bot_df = pl.DataFrame({
        "user_id": ["BOT1"],
        "parent_asin": ["ITEM1"],
        "rating": [5.0]
    })
    mock_attack.return_value = bot_df
    
    mock_features.return_value = pl.DataFrame({"user_id": ["BOT1"], "review_count": [1]})
    
    mock_trust.return_value = pl.DataFrame({
        "user_id": ["U1", "U2", "U3", "BOT1"],
        "trust_score": [0.9, 0.8, 0.95, 0.1] # Bot gets low trust
    })
    
    # Mock rankings
    mock_ranking.side_effect = [
        {"U1": ["ITEM1", "ITEM2"]}, # Base rank
        {"U1": ["ITEM2", "ITEM1"]}  # Trust rank (target went down)
    ]
    
    res = run_attack_evaluation(clean_df, features_df, user_profiles, item_profiles, "average", 1)
    
    assert res["attack_type"] == "average"
    assert res["target_item"] == "ITEM1"
    assert res["detection_rate"] == 1.0 # 1 bot with score < 0.3
    assert res["pos_base"] == 0.0 # ITEM1 is at index 0 in base
    assert res["pos_trust"] == 1.0 # ITEM1 is at index 1 in trust
