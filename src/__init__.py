"""DPO vs PPO Sentiment Generation Experiments."""

__version__ = "1.0.0"
__author__ = "DPO Paper Reproduction"

from .config import ModelConfig, DataConfig, SFTConfig, DPOConfig, PPOConfig, EvalConfig, get_device
from .utils import (
    get_device as get_device_util,
    get_dtype,
    load_sentiment_classifier,
    get_sentiment_score,
    calculate_kl_divergence,
    extract_prompts_from_imdb,
    batch_generate,
)

__all__ = [
    "ModelConfig",
    "DataConfig", 
    "SFTConfig",
    "DPOConfig",
    "PPOConfig",
    "EvalConfig",
    "get_device",
    "get_dtype",
    "load_sentiment_classifier",
    "get_sentiment_score",
    "calculate_kl_divergence",
    "extract_prompts_from_imdb",
    "batch_generate",
]
