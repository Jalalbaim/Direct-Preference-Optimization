"""Supervised Fine-Tuning (SFT) on IMDb dataset using TRL SFTTrainer."""
import os
import torch
import wandb
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

from src.config import ModelConfig, SFTConfig as SFTConfigClass, DataConfig
from src.utils import get_device

def train_sft():
    """Train SFT model on IMDb training set."""
    model_config = ModelConfig()
    sft_config_class = SFTConfigClass()
    data_config = DataConfig()
    
    # Initialize Weights & Biases
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sft_{timestamp}"
    wandb.init(
        project="dpo_ppo",
        name=run_name,
        config={
            "stage": "sft",
            "model": model_config.model_name,
            "batch_size": sft_config_class.per_device_train_batch_size,
            "learning_rate": sft_config_class.learning_rate,
            "num_epochs": sft_config_class.num_train_epochs,
        }
    )
    
    device = model_config.device
    print(f"\n{'='*60}")
    print(f"Training SFT on IMDb")
    print(f"Using device: {device}")
    print(f"W&B Run: {run_name}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    print(f"Loading model: {model_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load IMDb dataset - simple, direct approach
    print("Loading IMDb training set...")
    dataset = load_dataset(data_config.dataset_name)
    train_dataset = dataset[data_config.train_split]
    print(f"Loaded {len(train_dataset)} examples")

    # Device-specific adjustments
    if device == "mps":
        # Conservative settings for Mac MPS
        max_train = int(os.getenv("MPS_MAX_TRAIN", "8000"))
        batch_size = int(os.getenv("MPS_BATCH_SIZE", "4"))
        max_seq = int(os.getenv("MPS_MAX_SEQ_LENGTH", "256"))
        
        train_dataset = train_dataset.select(range(min(max_train, len(train_dataset))))
        sft_config_class.per_device_train_batch_size = batch_size
        data_config.max_seq_length = max_seq
        
        print(f"MPS mode: {len(train_dataset)} examples, batch={batch_size}, max_seq={max_seq}")
    elif device == "cuda":
        # Increase throughput on GPU
        sft_config_class.per_device_train_batch_size = max(8, sft_config_class.per_device_train_batch_size)
        data_config.max_seq_length = 512
        # Enable TF32 for faster matmul
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        print(f"CUDA mode: batch={sft_config_class.per_device_train_batch_size}, max_seq=512, tf32=enabled")
    
    # Formatting function for raw text
    def formatting_func(examples):
        return examples["text"]
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=sft_config_class.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=sft_config_class.num_train_epochs,
        per_device_train_batch_size=sft_config_class.per_device_train_batch_size,
        learning_rate=sft_config_class.learning_rate,
        warmup_steps=sft_config_class.warmup_steps,
        weight_decay=sft_config_class.weight_decay,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=device == "cuda",
        bf16=False,
        gradient_accumulation_steps=sft_config_class.gradient_accumulation_steps,
        dataloader_num_workers=8,
        dataloader_pin_memory=device == "cuda",
        logging_dir="./logs",
        seed=42,
        report_to=["wandb"],
    )
    
    # Initialize SFT trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        max_seq_length=data_config.max_seq_length,
    )
    
    # Train
    print("Starting SFT training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {sft_config_class.output_dir}")
    trainer.model.save_pretrained(sft_config_class.output_dir)
    tokenizer.save_pretrained(sft_config_class.output_dir)
    
    print("✅ SFT training completed!")
    wandb.finish()
    return trainer.model, tokenizer

if __name__ == "__main__":
    train_sft()
