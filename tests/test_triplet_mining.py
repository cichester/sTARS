import pytest
import polars as pl
from unittest.mock import patch
import os

from src.triplet_mining import mine_triplets

@pytest.fixture
def mock_dataset(tmp_path):
    df = pl.DataFrame({
        "parent_asin": ["ITEM1", "ITEM1", "ITEM1", "ITEM2", "ITEM2", "ITEM3", "ITEM3", "ITEM3"],
        "review_text": [
            "Good product, highly recommend it",
            "Very nice ITEM1, works well and is cheap",
            "I love ITEM1 it is the best",
            "ITEM2 is okay but could be better designed",
            "Not a fan of ITEM2 to be honest",
            "ITEM3 is the worst thing ever made in the history of things",
            "Terrible ITEM3, do not buy it please",
            "ITEM3 broke after 5 minutes"
        ]
    })
    
    clean_path = tmp_path / "electronics_5core_clean.parquet"
    df.write_parquet(clean_path)
    
    return tmp_path

def test_triplet_mining(mock_dataset):
    tmp_path = mock_dataset
    triplets_path = str(tmp_path / "triplets.parquet")
    
    with patch("src.triplet_mining.DATA_DIR", tmp_path), \
         patch("src.triplet_mining.TRIPLETS_PATH", triplets_path), \
         patch("src.triplet_mining.NUM_TRIPLETS", 10), \
         patch("src.triplet_mining.RANDOM_SEED", 42):
         
        # Ensure triplets don't exist yet
        assert not os.path.exists(triplets_path)
        
        # Run mining
        mine_triplets()
        
        # Check if generated
        assert os.path.exists(triplets_path)
        
        triplets = pl.read_parquet(triplets_path)
        
        assert triplets.height == 10
        assert "anchor_text" in triplets.columns
        assert "positive_text" in triplets.columns
        assert "negative_text" in triplets.columns
        
        for row in triplets.iter_rows(named=True):
            assert row["anchor_text"] != row["positive_text"]
            assert row["anchor_text"] != row["negative_text"]
