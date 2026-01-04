import os
import sys

import torch
from torch.utils.data import DataLoader

# pour que "src" soit dans le path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.core.models import load_models
from src.core.data import PromptDataset, prompt_collate_fn
from src.ppo.ppo_trainer import PPOTrainer
from src.core.utils import load_yaml_config


def main():
    config_path = "configs/ppo_sentiment.yaml"
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

    # Trainer PPO
    trainer = PPOTrainer(
        model_bundle=mb,
        prompt_loader=prompt_loader,
        config=config,
    )

    print(f"Starting PPO training: {config['experiment_name']}")
    print(f"Model: {model_name}")
    print(f"PPO params: clip_epsilon={config['ppo']['clip_epsilon']}, "
          f"value_coef={config['ppo']['value_coef']}, "
          f"entropy_coef={config['ppo']['entropy_coef']}")
    
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
