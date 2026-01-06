# scripts/train_sentiment_ppo.py
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dpo.models import load_models
from src.dpo.utils import load_yaml_config
from src.ppo.data import PromptDataset, prompt_collate_fn
from src.ppo.utils import ValueHead
from src.ppo.trainer import PPOTrainer

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/sentiment_ppo.yaml")
    args = p.parse_args()

    cfg = load_yaml_config(args.config)

    mb = load_models(cfg["model"]["name"], dtype=cfg["model"]["dtype"])

    # value head
    hidden_size = getattr(mb.policy_model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(mb.policy_model.config, "n_embd")

    value_head = ValueHead(hidden_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert value_head to the same dtype as the policy model
    model_dtype = mb.policy_model.dtype
    value_head.to(device=device, dtype=model_dtype)

    ds = PromptDataset(cfg["data"]["train_prompts_path"])
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=prompt_collate_fn,
    )

    trainer = PPOTrainer(
        model_bundle=mb,
        value_head=value_head,
        train_loader=loader,
        config=cfg,
    )
    trainer.train()

if __name__ == "__main__":
    main()
