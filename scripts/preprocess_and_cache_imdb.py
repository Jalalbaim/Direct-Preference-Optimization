"""Pre-tokenize and cache IMDb dataset for faster SFT training."""
import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig, DataConfig

def preprocess_and_cache_imdb():
    """Pre-tokenize IMDb dataset and save to cache."""
    model_config = ModelConfig()
    data_config = DataConfig()
    
    cache_dir = "./data/imdb_cached"
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Loading IMDb dataset...")
    dataset = load_dataset(data_config.dataset_name)
    train_dataset = dataset[data_config.train_split]
    
    print(f"Loaded {len(train_dataset)} training examples")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize function
    def tokenize_function(examples):
        """Tokenize texts with truncation and padding."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_config.max_seq_length,
            padding="max_length",
        )
    
    # Apply tokenization with caching
    print(f"Tokenizing dataset (max_seq_length={data_config.max_seq_length})...")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=256,
        desc="Tokenizing",
        remove_columns=train_dataset.column_names,  # Remove original text column
    )
    
    # Save to disk (Arrow format for fast loading)
    cache_path = os.path.join(cache_dir, "train")
    print(f"Saving tokenized dataset to {cache_path}...")
    tokenized_dataset.save_to_disk(cache_path)
    
    print(f"✅ Successfully cached {len(tokenized_dataset)} tokenized examples")
    print(f"   Cache location: {cache_path}")
    print(f"   Dataset columns: {tokenized_dataset.column_names}")
    print(f"   Dataset size: {tokenized_dataset.info.dataset_size / (1024**3):.2f} GB")
    
    return tokenized_dataset

if __name__ == "__main__":
    preprocess_and_cache_imdb()
