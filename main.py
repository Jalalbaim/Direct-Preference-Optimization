"""Main runner script for DPO vs PPO sentiment generation experiments."""
import argparse
import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_prepare_dataset():
    """Prepare test prompts (SFT uses full IMDb directly)."""
    print("\n" + "="*70)
    print("STEP 1: Training SFT on Full IMDb Dataset")
    print("="*70)
    from scripts.train_sft import train_sft
    train_sft()

def run_generate_preference_pairs():
    """Generate preference pairs using SFT model."""
    print("\n" + "="*70)
    print("STEP 2: Generating Synthetic Preference Pairs")
    print("="*70)
    from scripts.generate_preference_pairs import generate_preference_pairs
    from src.config import DataConfig, ModelConfig
    
    config = DataConfig()
    model_config = ModelConfig()
    generate_preference_pairs(config, model_config)

def run_dpo_training():
    """Train DPO models."""
    print("\n" + "="*70)
    print("STEP 3: Training DPO Models")
    print("="*70)
    from scripts.train_dpo import train_all_betas
    train_all_betas()

def run_ppo_training():
    """Train PPO and PPO-GT models."""
    print("\n" + "="*70)
    print("STEP 4: Training PPO and PPO-GT Models")
    print("="*70)
    from scripts.train_ppo import train_all_target_kls
    train_all_target_kls()

def run_evaluation():
    """Evaluate all models and create comparison plots."""
    print("\n" + "="*70)
    print("STEP 5: Evaluating All Models and Creating Plots")
    print("="*70)
    from scripts.evaluate import evaluate_all_models
    evaluate_all_models()

def main():
    parser = argparse.ArgumentParser(
        description="DPO vs PPO sentiment generation experiments"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["prepare", "dpo", "ppo", "evaluate", "all"],
        default="all",
        help="Which stage to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Force specific device (cuda for NVIDIA, mps for Mac, cpu for CPU)"
    )
    
    args = parser.parse_args()
    
    # Set device if specified
    if args.device:
        os.environ["TORCH_DEVICE"] = args.device
    
    print("\n" + "="*70)
    print("DPO vs PPO: Sentiment Generation Experiments")
    print("Paper: Direct Preference Optimization")
    print("Dataset: IMDb")
    print("Model: GPT-2-Large")
    print("="*70)
    
    if args.stage == "prepare" or args.stage == "all":
        run_prepare_dataset()
    
    if args.stage == "dpo" or args.stage == "all":
        if args.stage == "all":
            run_generate_preference_pairs()
        run_dpo_training()
    
    if args.stage == "ppo" or args.stage == "all":
        run_ppo_training()
    
    if args.stage == "evaluate" or args.stage == "all":
        run_evaluation()
    
    print("\n" + "="*70)
    print("All stages completed!")
    print("Results saved to ./results/")
    print("Plots saved to ./results/plots/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
