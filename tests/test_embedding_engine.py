import pytest
import numpy as np
import polars as pl
import os
import torch
from unittest.mock import patch
from sentence_transformers import SentenceTransformer
from src.embedding_engine import build_profiles, generate_embeddings, SBERT_MODEL_NAME

def test_build_profiles():
    # 4 recensioni: 2 per U1, 2 per U2
    df = pl.DataFrame({
        "user_id": ["U1", "U1", "U2", "U2"],
        "parent_asin": ["I1", "I2", "I1", "I3"]
    })
    
    # Dummy embeddings 2D
    embeddings = np.array([
        [1.0, 0.0], # u1, i1
        [1.0, 2.0], # u1, i2
        [3.0, 4.0], # u2, i1
        [5.0, 6.0]  # u2, i3
    ])
    
    user_profs, item_profs = build_profiles(df, embeddings)
    
    # U1 media: [1.0, 1.0]
    assert np.allclose(user_profs["U1"], [1.0, 1.0])
    # U2 media: [4.0, 5.0]
    assert np.allclose(user_profs["U2"], [4.0, 5.0])
    
    # I1 media: [2.0, 2.0]
    assert np.allclose(item_profs["I1"], [2.0, 2.0])
    # I2 media: [1.0, 2.0]
    assert np.allclose(item_profs["I2"], [1.0, 2.0])


def test_generate_embeddings_with_model():
    model = SentenceTransformer(SBERT_MODEL_NAME, device='cpu')
    texts = ["This is a test", "Another test string"]
    
    embeddings = generate_embeddings(texts, model, device='cpu')
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384 # MiniLM-L6-v2 embedding size
