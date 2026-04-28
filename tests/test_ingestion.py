import os
import json
import polars as pl
import pytest
from src.ingestion import find_5core_ids, extract_and_save_5core

MOCK_RAW_PATH = "tests/data/mock_electronics.jsonl"
MOCK_OUT_PATH = "tests/data/mock_out.parquet"

@pytest.fixture(scope="session", autouse=True)
def setup_mock_data():
    os.makedirs("tests/data", exist_ok=True)
    # We create a mock where user1 and itemA have >= 3 interactions (we will test MIN=3 to keep it small)
    # user1 has 3 interactions across itemA
    # user2 has 1
    # itemB has 1
    
    data = [
        {"user_id": "u1", "parent_asin": "i1", "text": "A"},
        {"user_id": "u1", "parent_asin": "i1", "text": "B"},
        {"user_id": "u1", "parent_asin": "i2", "text": "C"},  # i2 has only 1 interaction, so u1 will lose an interaction!
        {"user_id": "u2", "parent_asin": "i1", "text": "D"},  # u2 has only 1 interaction -> removed
        
        # Let's make an actual 3-core
        {"user_id": "u_core1", "parent_asin": "i_core1", "text": "A"},
        {"user_id": "u_core1", "parent_asin": "i_core2", "text": "B"},
        {"user_id": "u_core1", "parent_asin": "i_core3", "text": "C"},
        
        {"user_id": "u_core2", "parent_asin": "i_core1", "text": "A"},
        {"user_id": "u_core2", "parent_asin": "i_core2", "text": "B"},
        {"user_id": "u_core2", "parent_asin": "i_core3", "text": "C"},
        
        {"user_id": "u_core3", "parent_asin": "i_core1", "text": "A"},
        {"user_id": "u_core3", "parent_asin": "i_core2", "text": "B"},
        {"user_id": "u_core3", "parent_asin": "i_core3", "text": "C"},
    ]
    with open(MOCK_RAW_PATH, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
            
    yield
    
    # Teardown
    if os.path.exists(MOCK_RAW_PATH):
        os.remove(MOCK_RAW_PATH)
    if os.path.exists(MOCK_OUT_PATH):
        os.remove(MOCK_OUT_PATH)


def test_find_5core_ids():
    # Tested with min_interactions=3
    users, items = find_5core_ids(MOCK_RAW_PATH, min_interactions=3)
    
    assert set(users) == {"u_core1", "u_core2", "u_core3"}
    assert set(items) == {"i_core1", "i_core2", "i_core3"}
    
    assert "u1" not in users
    assert "u2" not in users

def test_extract_and_save_5core():
    # Extract
    users = {"u_core1", "u_core2", "u_core3"}
    items = {"i_core1", "i_core2", "i_core3"}
    
    extract_and_save_5core(MOCK_RAW_PATH, MOCK_OUT_PATH, users, items)
    
    assert os.path.exists(MOCK_OUT_PATH)
    
    df = pl.read_parquet(MOCK_OUT_PATH)
    assert df.height == 9
    assert "review_text" not in df.columns # that comes in preprocessing
    
    all_users = set(df["user_id"].unique().to_list())
    assert all_users == users
