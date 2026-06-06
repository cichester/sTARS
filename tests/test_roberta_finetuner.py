import pytest
import sys
from unittest.mock import patch, MagicMock

# Create a mock for config
mock_config = MagicMock()
mock_config.TRIPLETS_PATH = "mock_triplets.parquet"
mock_config.ROBERTA_MODEL_NAME = "mock_roberta"
mock_config.ROBERTA_ADAPTER_DIR = "mock_adapter_dir"
sys.modules['config'] = mock_config

from src.roberta_finetuner import train_roberta_lora
import os

@patch("src.roberta_finetuner.os.path.exists")
def test_train_roberta_lora_adapter_exists(mock_exists):
    mock_exists.return_value = True
    # Should return early if adapter already exists
    assert train_roberta_lora() is None
