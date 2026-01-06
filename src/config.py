"""Configuration for DPO vs PPO sentiment generation experiments."""
from dataclasses import dataclass, field
from typing import Optional
import torch

def get_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "gpt2-large"
    device: str = field(default_factory=get_device)
    dtype: str = "float32"
    max_seq_length: int = 512

@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "imdb"
    train_split: str = "train"
    test_split: str = "test"
    num_train_samples: Optional[int] = 16000  # Use subset for efficiency
    num_test_prompts: int = 500
    prompt_max_length: int = 50
    generation_max_length: int = 50
    max_seq_length: int = 512

@dataclass
class SFTConfig:
    """Supervised fine-tuning configuration."""
    output_dir: str = "./results/sft_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_steps: int = 500

@dataclass
class DPOConfig:
    """DPO training configuration."""
    output_dir: str = "./results/dpo_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    learning_rate: float = 5e-6
    beta: float = 0.1  # Will be varied: [0.05, 0.1, 1, 5]
    max_prompt_length: int = 50
    max_target_length: int = 50
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_steps: int = 500
    gradient_accumulation_steps: int = 1

@dataclass
class PPOConfig:
    """PPO training configuration."""
    output_dir: str = "./results/ppo_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    learning_rate: float = 1e-5
    reward_model_path: str = "./results/reward_model"
    target_kl: float = 6  # Will be varied: [3, 6, 9, 12]
    max_prompt_length: int = 50
    max_target_length: int = 50
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_steps: int = 500
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    eval_batch_size: int = 64
    num_eval_samples: int = 500
    eval_every_n_steps: int = 100
