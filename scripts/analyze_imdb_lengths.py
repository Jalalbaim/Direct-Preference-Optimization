"""Analyze token length distribution in IMDB dataset."""
import os
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig, DataConfig

def analyze_sequence_lengths(num_samples=100):
    """Sample and analyze sequence lengths in IMDB."""
    model_config = ModelConfig()
    data_config = DataConfig()
    
    print("Loading IMDb dataset...")
    dataset = load_dataset(data_config.dataset_name)
    train_dataset = dataset[data_config.train_split]
    
    print(f"Total examples: {len(train_dataset)}")
    
    # Sample randomly
    sample_indices = np.random.choice(len(train_dataset), min(num_samples, len(train_dataset)), replace=False)
    sample_dataset = train_dataset.select(sample_indices)
    
    print(f"\nSampling {len(sample_dataset)} examples for analysis...")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    # Compute token lengths
    token_lengths = []
    
    for example in sample_dataset:
        text = example["text"]
        tokens = tokenizer(text, truncation=False, return_tensors=None)
        token_count = len(tokens["input_ids"])
        token_lengths.append(token_count)
    
    # Statistics
    token_lengths = np.array(token_lengths)
    
    print(f"\n{'='*60}")
    print(f"Token Length Statistics (n={len(token_lengths)})")
    print(f"{'='*60}")
    print(f"Min:     {token_lengths.min()} tokens")
    print(f"Max:     {token_lengths.max()} tokens")
    print(f"Mean:    {token_lengths.mean():.1f} tokens")
    print(f"Median:  {np.median(token_lengths):.1f} tokens")
    print(f"Std:     {token_lengths.std():.1f} tokens")
    print(f"P95:     {np.percentile(token_lengths, 95):.1f} tokens")
    print(f"P99:     {np.percentile(token_lengths, 99):.1f} tokens")
    print(f"{'='*60}\n")
    
    print("Recommended max_seq_length values:")
    print(f"  Conservative (covers ~90%):  {int(np.percentile(token_lengths, 90))}")
    print(f"  Balanced (covers ~95%):      {int(np.percentile(token_lengths, 95))}")
    print(f"  Generous (covers ~99%):      {int(np.percentile(token_lengths, 99))}")
    print(f"\n💡 Tip: Use 512 for standard training, or adjust based on your GPU memory.")
    
    return {
        "min": int(token_lengths.min()),
        "max": int(token_lengths.max()),
        "mean": float(token_lengths.mean()),
        "median": float(np.median(token_lengths)),
        "std": float(token_lengths.std()),
        "p95": float(np.percentile(token_lengths, 95)),
        "p99": float(np.percentile(token_lengths, 99)),
    }

if __name__ == "__main__":
    analyze_sequence_lengths(num_samples=100)
