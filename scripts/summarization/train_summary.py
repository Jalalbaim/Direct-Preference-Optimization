#IMPORTS -----------------
import os
import sys

import torch
from torch.utils.data import DataLoader

# pour que "src" soit dans le path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.dpo.models import load_models
from src.dpo.data import PreferenceDataset, preference_collate_fn
from src.dpo.trainer import DPOTrainer
from src.dpo.utils import load_yaml_config


#MAIN FUNCTIONS -----------------
def main():
    config_path = "configs/summary.yaml"
    config = load_yaml_config(config_path)

    model_name = config["model"]["name"]
    dtype = config["model"]["dtype"]

    print(f"Model name: {model_name}")
    mb = load_models(model_name, dtype=dtype)
    tokenizer = mb.tokenizer

    train_dataset = PreferenceDataset(config["data"]["train_path"])
    val_dataset = PreferenceDataset(config["data"]["val_path"])

    max_prompt_length = config["data"]["max_prompt_length"]
    max_response_length = config["data"]["max_response_length"]

    def collate(batch):
        return preference_collate_fn(
            batch,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    trainer = DPOTrainer(
        model_bundle=mb,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    trainer.train()


if __name__ == "__main__":
    main()


    