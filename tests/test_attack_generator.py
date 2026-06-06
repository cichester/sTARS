import pytest
from unittest.mock import MagicMock, patch
import polars as pl
from src.attack_generator import select_target_item, generate_average_attack, generate_bandwagon_attack, generate_genai_attack

@pytest.fixture
def mock_clean_df():
    # 3 items: ITEM1 (10 reviews), ITEM2 (5 reviews), ITEM3 (3 reviews)
    records = []
    for i in range(10):
        records.append({
            "rating": 4.0,
            "parent_asin": "ITEM1",
            "user_id": f"U_{i}",
            "timestamp": 1600000000000,
            "text": "Regular user review text"
        })
    for i in range(5):
        records.append({
            "rating": 3.0,
            "parent_asin": "ITEM2",
            "user_id": f"U_2_{i}",
            "timestamp": 1600000000000,
            "text": "Another user review text"
        })
    for i in range(3):
        records.append({
            "rating": 2.0,
            "parent_asin": "ITEM3",
            "user_id": f"U_3_{i}",
            "timestamp": 1600000000000,
            "text": "Yet another review text"
        })
        
    return pl.DataFrame(records)

def test_select_target_item(mock_clean_df):
    target = select_target_item(mock_clean_df)
    assert target in ["ITEM1", "ITEM2", "ITEM3"]

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

def test_generate_genai_attack(mock_clean_df):
    # Mocking GeminiClient per evitare chiamate reali durante i test
    with patch("src.attack_generator.GeminiClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate_reviews.return_value = [
            {"title": "GenAI Title", "text": "GenAI Text content of review."}
        ]
        mock_client_class.return_value = mock_client
        
        bot_df = generate_genai_attack(mock_clean_df, "ITEM1", num_bots=5, filler_size=2)
        
        # 5 bots * (1 target + 2 fillers) = 15 righe
        assert bot_df.height == 15
        assert bot_df["user_id"].n_unique() == 5
        
        # Controlliamo la presenza delle recensioni per il target e i filler
        target_reviews = bot_df.filter(pl.col("parent_asin") == "ITEM1")
        assert target_reviews.height == 5
        assert (target_reviews["rating"] == 5.0).all()
        assert (target_reviews["review_text"] == "GenAI Text content of review.").all()
