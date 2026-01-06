"""Direct Preference Optimization (DPO) training on IMDb sentiment generation."""
import os
import json
import torch
import wandb
import numpy as np
from datetime import datetime
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from tqdm import tqdm

from src.config import ModelConfig, DPOConfig as DPOTrainingConfig, DataConfig
from src.utils import get_device

def prepare_dpo_data(data_path: str = "./data/preference_pairs.json"):
    """Prepare preference pairs for DPO training."""
    with open(data_path, 'r') as f:
        pairs = json.load(f)
    
    dpo_data = {
        'prompt': [],
        'chosen': [],
        'rejected': []
    }
    
    for pair in pairs:
        dpo_data['prompt'].append(pair['prompt'])
        dpo_data['chosen'].append(pair['chosen'])
        dpo_data['rejected'].append(pair['rejected'])
    
    return dpo_data

def train_dpo(beta: float = 0.1):
    """Train DPO model with specified beta."""
    model_config = ModelConfig()
    dpo_config = DPOTrainingConfig()
    data_config = DataConfig()
    
    # Initialize Weights & Biases
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dpo_beta{beta}_{timestamp}"
    wandb.init(
        project="dpo_ppo",
        name=run_name,
        config={
            "stage": "dpo",
            "beta": beta,
            "model": model_config.model_name,
            "batch_size": dpo_config.per_device_train_batch_size,
            "learning_rate": dpo_config.learning_rate,
            "num_epochs": dpo_config.num_train_epochs,
        }
    )
    
    device = model_config.device
    print(f"\n{'='*60}")
    print(f"Training DPO with beta={beta}")
    print(f"Using device: {device}")
    print(f"W&B Run: {run_name}")
    print(f"{'='*60}\n")
    
    # Load base model and tokenizer
    print(f"Loading model: {model_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float32,
        device_map=device if device != "cpu" else None
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    print("Preparing DPO data...")
    if os.path.exists("./data/preference_pairs.json"):
        dpo_data = prepare_dpo_data()
    else:
        print("Warning: preference_pairs.json not found. Using dummy data.")
        dpo_data = {
            'prompt': ["This movie is"] * 10,
            'chosen': ["great and wonderful!"] * 10,
            'rejected': ["terrible and awful!"] * 10
        }
    
    # Create dataset
    dataset = Dataset.from_dict(dpo_data)
    
    # Training configuration
    training_args = DPOConfig(
        output_dir=f"./results/dpo_beta_{beta}",
        overwrite_output_dir=True,
        num_train_epochs=dpo_config.num_train_epochs,
        per_device_train_batch_size=dpo_config.per_device_train_batch_size,
        learning_rate=dpo_config.learning_rate,
        max_prompt_length=dpo_config.max_prompt_length,
        max_target_length=dpo_config.max_target_length,
        beta=beta,
        warmup_steps=dpo_config.warmup_steps,
        weight_decay=dpo_config.weight_decay,
        logging_steps=dpo_config.logging_steps,
        save_steps=dpo_config.save_steps,
        gradient_accumulation_steps=dpo_config.gradient_accumulation_steps,
        bf16=device != "cuda",  # Use bfloat16 for non-CUDA devices
        fp16=device == "cuda",  # Use fp16 for CUDA
        remove_unused_columns=False,
        seed=42,
        report_to=["wandb"],
    )
    
    # Initialize DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=None,  # Can add LoRA config for memory efficiency if needed
    )
    
    # Train
    print("Starting DPO training...")
    dpo_trainer.train()
    
    # Save model
    output_dir = f"./results/dpo_beta_{beta}"
    print(f"Saving model to {output_dir}")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"DPO training completed with beta={beta}!")
    wandb.finish()
    return dpo_trainer.model, tokenizer
    
    print(f"DPO training completed for beta={beta}!")
    return dpo_trainer.model, tokenizer

def train_all_betas():
    """Train DPO with multiple beta values."""
    betas = [0.05, 0.1, 1, 5]
    
    print("\nTraining DPO models with different beta values...")
    for beta in betas:
        try:
            train_dpo(beta=beta)
        except Exception as e:
            print(f"Error training DPO with beta={beta}: {e}")
            continue

if __name__ == "__main__":
    train_all_betas()
