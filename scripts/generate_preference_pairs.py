"""Generate synthetic preference pairs using trained SFT model and sentiment classifier.

This script should be run AFTER SFT training to generate preference pairs for DPO.
"""
import os
import json
import torch
import numpy as np
from typing import List, Dict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm import tqdm

from src.config import DataConfig, ModelConfig
from src.utils import load_sentiment_classifier, get_sentiment_score

def load_sft_model(sft_path: str = "./results/sft_model"):
    """Load the trained SFT model."""
    print(f"Loading SFT model from {sft_path}")
    model = AutoModelForCausalLM.from_pretrained(sft_path)
    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_completions(
    prompts: List[str],
    model,
    tokenizer,
    num_completions: int = 4,
    device: str = "cuda"
) -> List[List[str]]:
    """Generate multiple completions for each prompt."""
    model = model.to(device)
    model.eval()
    
    result = []
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating completions"):
            batch_prompts = prompts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate multiple completions per prompt
            batch_completions = [[] for _ in range(len(batch_prompts))]
            
            for _ in range(num_completions):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for j, gen in enumerate(generations):
                    batch_completions[j].append(gen)
            
            result.extend(batch_completions)
    
    return result

def create_preference_pairs_from_completions(
    prompts: List[str],
    all_completions: List[List[str]],
    sentiment_model,
    sentiment_tokenizer,
    device: str = "cuda"
) -> List[Dict]:
    """Create preference pairs by ranking completions with sentiment classifier."""
    
    preference_pairs = []
    
    for prompt, completions in tqdm(zip(prompts, all_completions), 
                                     desc="Creating preference pairs",
                                     total=len(prompts)):
        # Score each completion
        scores = get_sentiment_score(completions, sentiment_model, sentiment_tokenizer, device)
        
        # Create all possible preference pairs (6 pairs from 4 completions)
        for i in range(len(completions)):
            for j in range(i+1, len(completions)):
                if scores[i] > scores[j]:
                    chosen = completions[i]
                    rejected = completions[j]
                elif scores[j] > scores[i]:
                    chosen = completions[j]
                    rejected = completions[i]
                else:
                    continue  # Skip ties
                
                preference_pairs.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                })
    
    return preference_pairs

def generate_preference_pairs(config: DataConfig, model_config: ModelConfig, seed: int = 42):
    """Generate preference pairs using trained SFT model.
    
    This should be called AFTER SFT training.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("\n" + "="*60)
    print("STEP 2: Generating Synthetic Preference Pairs")
    print("="*60 + "\n")
    
    # Load SFT model
    if not os.path.exists("./results/sft_model"):
        print("ERROR: SFT model not found at ./results/sft_model")
        print("Please train SFT first: python main.py --stage sft")
        return []
    
    sft_model, sft_tokenizer = load_sft_model()
    device = model_config.device
    
    # Load sentiment classifier
    print("Loading sentiment classifier...")
    sentiment_model, sentiment_tokenizer = load_sentiment_classifier()
    
    # Load IMDb dataset and extract prompts from training set
    print("Loading IMDb training set...")
    dataset = load_dataset(config.dataset_name)
    train_data = dataset[config.train_split]
    
    # Sample prompts
    print(f"Extracting {config.num_train_samples} prompts...")
    prompts = []
    
    for example in train_data:
        text = example['text']
        words = text.split()
        
        if len(words) < 8:
            continue
        
        # Take first 2-8 words as prompt (following Northeastern blog)
        prompt_length = min(np.random.randint(2, 9), len(words))
        prompt = ' '.join(words[:prompt_length])
        prompts.append(prompt)
        
        if len(prompts) >= config.num_train_samples:
            break
    
    print(f"Extracted {len(prompts)} prompts")
    
    # Generate 4 completions per prompt
    print(f"Generating {config.num_train_samples} x 4 completions...")
    all_completions = generate_completions(
        prompts,
        sft_model,
        sft_tokenizer,
        num_completions=4,
        device=device
    )
    
    # Create preference pairs (6 pairs per prompt from 4 completions)
    print("Creating preference pairs from completions...")
    preference_pairs = create_preference_pairs_from_completions(
        prompts,
        all_completions,
        sentiment_model,
        sentiment_tokenizer,
        device
    )
    
    print(f"Created {len(preference_pairs)} preference pairs")
    
    # Save preference pairs
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "preference_pairs.json"), 'w') as f:
        json.dump(preference_pairs, f, indent=2)
    
    print(f"Saved preference pairs to {output_dir}/preference_pairs.json")
    
    # Also save test prompts for evaluation
    print("Extracting test prompts for evaluation...")
    test_data = dataset[config.test_split]
    test_prompts = []
    
    for example in test_data:
        text = example['text']
        words = text.split()
        
        if len(words) < 8:
            continue
        
        prompt_length = min(np.random.randint(2, 9), len(words))
        prompt = ' '.join(words[:prompt_length])
        test_prompts.append(prompt)
        
        if len(test_prompts) >= config.num_test_prompts:
            break
    
    with open(os.path.join(output_dir, "test_prompts.json"), 'w') as f:
        json.dump(test_prompts, f, indent=2)
    
    print(f"Saved {len(test_prompts)} test prompts to {output_dir}/test_prompts.json")
    
    return preference_pairs

if __name__ == "__main__":
    config = DataConfig()
    model_config = ModelConfig()
    generate_preference_pairs(config, model_config)
