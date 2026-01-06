"""Prepare IMDb dataset with preference pairs for DPO training."""
import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm

from src.config import DataConfig, ModelConfig
from src.utils import get_sentiment_score, load_sentiment_classifier, get_device

def prepare_dataset(config: DataConfig, model_config: ModelConfig, seed: int = 42):
    """Prepare IMDb dataset with preference pairs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("Loading IMDb dataset...")
    dataset = load_dataset(config.dataset_name)
    
    # Use train split for SFT training (raw texts)
    train_data = dataset[config.train_split]
    
    # Sample subset if needed
    if config.num_train_samples:
        indices = np.random.choice(len(train_data), config.num_train_samples, replace=False)
        train_data = train_data.select(indices)
    
    print(f"Loaded {len(train_data)} training examples for SFT")
    
    # Load sentiment classifier for preference pair generation
    print("Loading sentiment classifier for preference pair creation...")
    sentiment_model, sentiment_tokenizer = load_sentiment_classifier()
    device = model_config.device
    print(f"Using device: {device}")
    
    # Save raw SFT training texts first
    print("\nSaving raw training texts for SFT...")
    sft_texts = [example['text'] for example in train_data]
    
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "sft_train_texts.json"), 'w') as f:
        json.dump(sft_texts, f, indent=2)
    
    print(f"Saved {len(sft_texts)} SFT training texts to {output_dir}/sft_train_texts.json")
    
    
    # Extract prompts and generate preference pairs for DPO
    print("\nExtracting prompts from reviews for DPO...")
    prompts = []
    labels = []
    
    for example in tqdm(train_data, desc="Extracting prompts"):
        text = example['text']
        label = example['label']  # 0: negative, 1: positive
        
        words = text.split()
        if len(words) < 10:
            continue
        
        # Take first 10-30 words as prompt
        prompt_length = min(np.random.randint(10, 30), len(words))
        prompt = ' '.join(words[:prompt_length])
        prompts.append(prompt)
        labels.append(label)
    
    print(f"Extracted {len(prompts)} prompts for DPO")
    
    # Generate continuations for each prompt
    print("Generating continuations for prompts...")
    from transformers import AutoModelForCausalLM
    
    model_name = model_config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_continuations = []
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating continuations"):
            batch_prompts = prompts[i:i+batch_size]
            
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate 2 continuations per prompt for preference pairs
            for gen_idx in range(2):
                outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=config.generation_max_length,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_continuations.append(generations)
    
    print(f"Generated {len(all_continuations)} continuation sets")
    
    # Create preference pairs using sentiment scores
    print("Creating preference pairs using sentiment classifier...")
    preference_pairs = []
    
    for idx, prompt in tqdm(enumerate(prompts[:config.num_train_samples] if config.num_train_samples else prompts), 
                             desc="Creating preference pairs"):
        if idx >= len(all_continuations[0]):
            break
        
        gen1 = all_continuations[0][idx]
        gen2 = all_continuations[1][idx]
        
        # Calculate sentiment scores
        with torch.no_grad():
            scores = get_sentiment_score([gen1, gen2], sentiment_model, sentiment_tokenizer, device)
        
        # Assign chosen (higher score) and rejected (lower score)
        if scores[0] > scores[1]:
            chosen = gen1
            rejected = gen2
        else:
            chosen = gen2
            rejected = gen1
        
        # Only keep pairs where there's a clear preference
        if abs(scores[0] - scores[1]) > 0.1:
            preference_pairs.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'chosen_sentiment': float(max(scores)),
                'rejected_sentiment': float(min(scores))
            })
    
    print(f"Created {len(preference_pairs)} preference pairs for DPO")
    
    # Save dataset
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "preference_pairs.json"), 'w') as f:
        json.dump(preference_pairs, f, indent=2)
    
    print(f"Saved preference pairs to {output_dir}/preference_pairs.json")
    
    # Also save test prompts for evaluation
    test_data = dataset[config.test_split]
    test_prompts = []
    
    for example in test_data:
        text = example['text']
        words = text.split()
        if len(words) < 10:
            continue
        
        prompt_length = min(np.random.randint(10, 30), len(words))
        prompt = ' '.join(words[:prompt_length])
        test_prompts.append(prompt)
        
        if len(test_prompts) >= config.num_test_prompts:
            break
    
    with open(os.path.join(output_dir, "test_prompts.json"), 'w') as f:
        json.dump(test_prompts, f, indent=2)
    
    print(f"Saved {len(test_prompts)} test prompts to {output_dir}/test_prompts.json")
    
    return preference_pairs, test_prompts

if __name__ == "__main__":
    config = DataConfig()
    model_config = ModelConfig()
    prepare_dataset(config, model_config)
