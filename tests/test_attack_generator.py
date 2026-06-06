import pytest
import polars as pl
from src.attack_generator import select_target_item, generate_average_attack, generate_bandwagon_attack

@pytest.fixture
def mock_clean_df():
    # 2 items, 1 with 10 reviews, 1 with 5 reviews
    records = []
    for i in range(10):
        records.append({
            "rating": 4.0,
            "parent_asin": "ITEM1",
            "user_id": f"U_{i}",
            "timestamp": 1600000000000
        })
    for i in range(5):
        records.append({
            "rating": 3.0,
            "parent_asin": "ITEM2",
            "user_id": f"U_2_{i}",
            "timestamp": 1600000000000
        })
        
    return pl.DataFrame(records)

def test_select_target_item(mock_clean_df):
    target = select_target_item(mock_clean_df)
    # the 50th percentile of [5, 10] is the one at index 1 = ITEM1
    assert target in ["ITEM1", "ITEM2"]

def test_generate_average_attack(mock_clean_df):
    bot_df = generate_average_attack(mock_clean_df, "ITEM1", num_bots=5, filler_size=1)
    
    assert bot_df.height == 10 # 5 bots * (1 target + 1 filler)
    assert bot_df["user_id"].n_unique() == 5
    
    # Check push attack
    target_reviews = bot_df.filter(pl.col("parent_asin") == "ITEM1")
    assert target_reviews.height == 5
    assert (target_reviews["rating"] == 5.0).all()
    
def test_generate_bandwagon_attack(mock_clean_df):
    bot_df = generate_bandwagon_attack(mock_clean_df, "ITEM2", num_bots=5, filler_size=1)
    
    assert bot_df.height == 10
    
    target_reviews = bot_df.filter(pl.col("parent_asin") == "ITEM2")
    assert target_reviews.height == 5
    assert (target_reviews["rating"] == 5.0).all()
