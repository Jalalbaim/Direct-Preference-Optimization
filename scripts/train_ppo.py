"""PPO and PPO-GT training on IMDb sentiment generation using TRL."""
import os
import json
import torch
import wandb
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from datasets import Dataset
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm

from src.config import ModelConfig, PPOConfig as PPOTrainingConfig, DataConfig
from src.utils import get_device, load_sentiment_classifier, get_sentiment_score

def prepare_ppo_data(data_path: str = "./data/preference_pairs.json"):
    """Prepare prompts for PPO training."""
    with open(data_path, 'r') as f:
        pairs = json.load(f)
    
    prompts = [pair['prompt'] for pair in pairs]
    return prompts

def create_reward_model(device: str = "cuda"):
    """Create reward model using sentiment classifier."""
    model, tokenizer = load_sentiment_classifier()
    model = model.to(device if device != "mps" else "cpu")
    return model, tokenizer

def compute_rewards(texts: List[str], reward_model, reward_tokenizer, device: str = "cuda") -> np.ndarray:
    """Compute rewards (sentiment scores) for texts."""
    inference_device = "cpu" if device == "mps" else device
    return get_sentiment_score(texts, reward_model, reward_tokenizer, inference_device)

def train_ppo(target_kl: float = 6, use_gth: bool = False):
    """Train PPO model with specified target KL.
    
    Args:
        target_kl: Target KL divergence constraint
        use_gth: If True, use ground truth rewards (PPO-GT). If False, use learned reward model.
    """
    model_config = ModelConfig()
    ppo_config = PPOTrainingConfig()
    data_config = DataConfig()
    
    device = model_config.device
    method = "PPO-GT" if use_gth else "PPO"
    
    # Initialize Weights & Biases
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{method.lower()}_kl{target_kl}_{timestamp}"
    wandb.init(
        project="dpo_ppo",
        name=run_name,
        config={
            "stage": method.lower(),
            "method": method,
            "target_kl": target_kl,
            "use_gth": use_gth,
            "model": model_config.model_name,
            "batch_size": ppo_config.batch_size,
            "learning_rate": ppo_config.learning_rate,
            "ppo_epochs": ppo_config.ppo_epochs,
        }
    )
    
    print(f"\n{'='*60}")
    print(f"Training {method} with target_kl={target_kl}")
    print(f"Using device: {device}")
    print(f"W&B Run: {run_name}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    print(f"Loading model: {model_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float32,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create reward model
    print("Creating reward model...")
    reward_model, reward_tokenizer = create_reward_model(device)
    
    # Prepare data
    print("Preparing PPO data...")
    if os.path.exists("./data/preference_pairs.json"):
        prompts = prepare_ppo_data()
    else:
        print("Warning: preference_pairs.json not found. Using dummy data.")
        prompts = ["This movie is"] * 100
    
    # Create dataset
    dataset = Dataset.from_dict({'prompt': prompts})
    
    # PPO Configuration
    ppo_training_args = PPOConfig(
        output_dir=f"./results/{method.lower().replace('-','_')}_kl_{target_kl}",
        learning_rate=ppo_config.learning_rate,
        target_kl=target_kl,
        mini_batch_size=ppo_config.mini_batch_size,
        batch_size=ppo_config.per_device_train_batch_size,
        ppo_epochs=ppo_config.ppo_epochs,
        gradient_accumulation_steps=ppo_config.gradient_accumulation_steps,
        logging_steps=ppo_config.logging_steps,
        save_steps=ppo_config.save_steps,
        remove_unused_columns=False,
        seed=42,
    )
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_training_args,
        model=model,
        ref_model=None,  # Use KL penalty instead of separate ref model
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=None,
    )
    
    # Training loop
    print(f"Starting {method} training...")
    
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": data_config.generation_max_length,
    }
    
    num_batches = min(len(dataset) // ppo_training_args.batch_size, 10)  # Limit for testing
    
    for epoch in range(ppo_config.num_train_epochs):
        print(f"Epoch {epoch+1}/{ppo_config.num_train_epochs}")
        
        for batch_idx in tqdm(range(num_batches), desc=f"{method} training"):
            # Get batch
            batch_prompts = prompts[batch_idx * ppo_training_args.batch_size:(batch_idx + 1) * ppo_training_args.batch_size]
            
            if len(batch_prompts) == 0:
                break
            
            # Generate responses
            query_tensors = ppo_trainer.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            response_tensors = ppo_trainer.generate(
                query_tensors["input_ids"],
                **generation_kwargs
            )
            
            # Decode to text
            batch_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # Compute rewards
            rewards = compute_rewards(
                [f"{p} {r}" for p, r in zip(batch_prompts, batch_responses)],
                reward_model,
                reward_tokenizer,
                device
            )
            reward_tensors = [torch.tensor(r) for r in rewards]
            
            # Update model
            stats = ppo_trainer.step(query_tensors["input_ids"], response_tensors, reward_tensors)
            
            if (batch_idx + 1) % ppo_training_args.logging_steps == 0:
                print(f"  Batch {batch_idx+1}: loss={stats.get('loss', 'N/A')}")
    
    # Save model
    output_dir = f"./results/{method.lower().replace('-','_')}_kl_{target_kl}"
    print(f"Saving model to {output_dir}")
    ppo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"{method} training completed for target_kl={target_kl}!")
    wandb.finish()
    return ppo_trainer.model, tokenizer

def train_all_target_kls():
    """Train PPO and PPO-GT with multiple target KL values."""
    target_kls = [3, 6, 9, 12]
    
    print("\nTraining PPO models with different target KL values...")
    for target_kl in target_kls:
        try:
            # Train PPO
            train_ppo(target_kl=target_kl, use_gth=False)
        except Exception as e:
            print(f"Error training PPO with target_kl={target_kl}: {e}")
            continue
    
    print("\nTraining PPO-GT models with different target KL values...")
    for target_kl in target_kls:
        try:
            # Train PPO-GT (using ground truth rewards)
            train_ppo(target_kl=target_kl, use_gth=True)
        except Exception as e:
            print(f"Error training PPO-GT with target_kl={target_kl}: {e}")
            continue

if __name__ == "__main__":
    train_all_target_kls()
