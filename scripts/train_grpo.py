import os
import sys

import torch
from torch.utils.data import DataLoader

# pour que "src" soit dans le path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dpo.models import load_models
from src.dpo.data import PromptDataset, prompt_collate_fn
from src.dpo.grpo_trainer import GRPOTrainer
from src.dpo.utils import load_yaml_config


def main():
    config_path = "configs/grpo_sentiment.yaml"
    config = load_yaml_config(config_path)

    model_name = config["model"]["name"]
    dtype = config["model"]["dtype"]

    # Charger les modèles
    mb = load_models(model_name, dtype=dtype)
    tokenizer = mb.tokenizer

    # Dataset de prompts
    prompt_dataset = PromptDataset(config["data"]["prompt_path"])
    max_prompt_length = config["data"]["max_prompt_length"]

    def collate(batch):
        return prompt_collate_fn(
            batch,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
        )

    prompt_loader = DataLoader(
        prompt_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )

    # Trainer GRPO
    trainer = GRPOTrainer(
        model_bundle=mb,
        prompt_loader=prompt_loader,
        config=config,
    )

    print(f"Starting GRPO training: {config['experiment_name']}")
    print(f"Model: {model_name}")
    print(f"GRPO params: group_size={config['grpo']['group_size']}, "
          f"clip_epsilon={config['grpo']['clip_epsilon']}, "
          f"beta={config['grpo']['beta']}")
    
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
