#!/usr/bin/env python3
"""
Script d'entraînement PPO SANS value head.
Utilise directement les rewards du reward model comme avantages.
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

# Ajouter le répertoire racine au path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.core.models import load_models
from src.core.data import PromptDataset, prompt_collate_fn
from src.ppo.ppo_trainer_no_vh import PPOTrainerNoValueHead
from src.core.utils import load_yaml_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train PPO without value head")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_sentiment_no_vh.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Charger la config
    config = load_yaml_config(args.config)
    set_seed(config["training"]["seed"])

    print("="*60)
    print(f"🚀 PPO Training (NO VALUE HEAD)")
    print(f"   Config: {args.config}")
    print(f"   Experiment: {config['experiment_name']}")
    print("="*60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")

    # Charger les modèles
    print("\n📥 Loading models...")
    model_name = config["model"]["name"]
    dtype = config["model"]["dtype"]
    
    mb = load_models(model_name, dtype=dtype, device=device)
    tokenizer = mb.tokenizer
    
    print(f"   ✓ Policy model: {model_name}")
    print(f"   ✓ Params: {mb.policy_model.num_parameters():,}")
    print(f"   ✓ NO VALUE HEAD - Économie de paramètres!")

    # Charger les données
    print("\n📚 Loading data...")
    prompt_dataset = PromptDataset(config["data"]["prompt_path"])
    max_prompt_length = config["data"]["max_prompt_length"]
    
    print(f"   ✓ Prompts: {len(prompt_dataset)}")

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
    
    print(f"   ✓ Batches per epoch: {len(prompt_loader)}")

    # Créer le trainer
    print("\n⚙️  Initializing PPO Trainer (No Value Head)...")
    trainer = PPOTrainerNoValueHead(
        model_bundle=mb,
        prompt_loader=prompt_loader,
        config=config,
    )
    
    print(f"   ✓ Reward model: {config['reward_model']['name']}")
    print(f"   ✓ Save dir: {config['logging']['save_dir']}")
    
    # Entraînement
    print("\n🏋️  Starting training...")
    print("="*60)
    
    import time
    start_time = time.time()
    
    try:
        trainer.train()
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ Training completed!")
        print(f"   Total time: {elapsed/60:.2f} minutes")
        print(f"   Avg time/epoch: {elapsed/config['training']['num_epochs']/60:.2f} min")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
