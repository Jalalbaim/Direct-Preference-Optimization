"""Evaluation and plotting script for DPO vs PPO comparison."""
import os
import json
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.config import ModelConfig, DataConfig
from src.utils import (
    load_sentiment_classifier,
    get_sentiment_score,
    get_device,
    batch_generate,
)

def load_test_prompts(test_prompts_path: str = "./data/test_prompts.json") -> List[str]:
    """Load test prompts."""
    if os.path.exists(test_prompts_path):
        with open(test_prompts_path, 'r') as f:
            return json.load(f)
    else:
        print("Warning: test_prompts.json not found. Using dummy test prompts.")
        return ["This movie is"] * 100

def calculate_sequence_kl(
    policy_texts: List[str],
    reference_texts: List[str],
    policy_model,
    policy_tokenizer,
    ref_model,
    ref_tokenizer,
    device: str = "cuda"
) -> float:
    """Calculate sequence-level KL divergence between policy and reference."""
    inference_device = "cpu" if device == "mps" else device
    
    policy_model = policy_model.to(inference_device)
    ref_model = ref_model.to(inference_device)
    
    kl_divs = []
    
    policy_model.eval()
    ref_model.eval()
    
    with torch.no_grad():
        for policy_text, ref_text in zip(policy_texts, reference_texts):
            # Get log probabilities
            policy_tokens = policy_tokenizer(policy_text, return_tensors="pt")
            ref_tokens = ref_tokenizer(ref_text, return_tensors="pt")
            
            # Ensure same length (take minimum)
            min_len = min(policy_tokens['input_ids'].shape[1], ref_tokens['input_ids'].shape[1])
            policy_tokens['input_ids'] = policy_tokens['input_ids'][:, :min_len]
            ref_tokens['input_ids'] = ref_tokens['input_ids'][:, :min_len]
            
            policy_tokens = {k: v.to(inference_device) for k, v in policy_tokens.items()}
            ref_tokens = {k: v.to(inference_device) for k, v in ref_tokens.items()}
            
            with torch.no_grad():
                policy_outputs = policy_model(**policy_tokens)
                ref_outputs = ref_model(**ref_tokens)
            
            # Calculate log probabilities per token
            policy_logps = torch.nn.functional.log_softmax(policy_outputs.logits, dim=-1)
            ref_logps = torch.nn.functional.log_softmax(ref_outputs.logits, dim=-1)
            
            # Get token-level log probs
            policy_log_probs = policy_logps[:, -1, :].squeeze()
            ref_log_probs = ref_logps[:, -1, :].squeeze()
            
            # Simple KL approximation
            seq_kl = float(torch.abs(policy_log_probs - ref_log_probs).mean().item())
            kl_divs.append(seq_kl)
    
    return float(np.mean(kl_divs)) if kl_divs else 0.0

def evaluate_model(
    model_path: str,
    test_prompts: List[str],
    reference_model,
    reference_tokenizer,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Evaluate a model on reward and KL metrics.
    
    Returns:
        Tuple of (average_reward, average_kl)
    """
    print(f"Evaluating model: {model_path}")
    
    # Load model
    try:
        policy_model = AutoModelForCausalLM.from_pretrained(model_path)
        policy_tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return 0.0, float('inf')
    
    policy_model = policy_model.to(device)
    
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    
    # Load sentiment classifier for rewards
    sentiment_model, sentiment_tokenizer = load_sentiment_classifier()
    
    # Generate continuations
    print("  Generating continuations...")
    generated_texts = batch_generate(
        policy_model,
        policy_tokenizer,
        test_prompts,
        max_new_tokens=50,
        batch_size=32,
        device=device
    )
    
    # Calculate rewards
    print("  Computing rewards...")
    rewards = get_sentiment_score(generated_texts, sentiment_model, sentiment_tokenizer, device)
    avg_reward = float(np.mean(rewards))
    
    # Calculate KL divergence
    print("  Computing KL divergence...")
    avg_kl = calculate_sequence_kl(
        generated_texts,
        test_prompts,  # Use prompts as reference for simplified KL
        policy_model,
        policy_tokenizer,
        reference_model,
        reference_tokenizer,
        device
    )
    
    return avg_reward, avg_kl

def evaluate_all_models():
    """Evaluate all trained models and create comparison plots."""
    model_config = ModelConfig()
    device = model_config.device
    
    # Initialize Weights & Biases
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evaluate_{timestamp}"
    wandb.init(
        project="dpo_ppo",
        name=run_name,
        config={"stage": "evaluate"},
    )
    
    print(f"Using device: {device}")
    print(f"W&B Run: {run_name}\n")
    
    # Load test prompts
    test_prompts = load_test_prompts()
    test_prompts = test_prompts[:100]  # Use subset for faster evaluation
    
    # Load reference model (SFT)
    print("Loading reference model...")
    ref_model_path = "./results/sft_model"
    if os.path.exists(ref_model_path):
        reference_model = AutoModelForCausalLM.from_pretrained(ref_model_path)
        reference_tokenizer = AutoTokenizer.from_pretrained(ref_model_path)
    else:
        print("Warning: SFT model not found. Using base model as reference.")
        reference_model = AutoModelForCausalLM.from_pretrained(model_config.model_name)
        reference_tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    results = {
        'DPO': {},
        'PPO': {},
        'PPO-GT': {}
    }
    
    # Evaluate DPO models
    print("\n" + "="*60)
    print("Evaluating DPO models")
    print("="*60)
    betas = [0.05, 0.1, 1, 5]
    for beta in betas:
        model_path = f"./results/dpo_beta_{beta}"
        if os.path.exists(model_path):
            reward, kl = evaluate_model(model_path, test_prompts, reference_model, reference_tokenizer, device)
            results['DPO'][beta] = {'reward': reward, 'kl': kl}
            print(f"  DPO (β={beta}): reward={reward:.3f}, KL={kl:.3f}")
        else:
            print(f"  DPO (β={beta}): model not found")
    
    # Evaluate PPO models
    print("\n" + "="*60)
    print("Evaluating PPO models")
    print("="*60)
    target_kls = [3, 6, 9, 12]
    for target_kl in target_kls:
        model_path = f"./results/ppo_kl_{target_kl}"
        if os.path.exists(model_path):
            reward, kl = evaluate_model(model_path, test_prompts, reference_model, reference_tokenizer, device)
            results['PPO'][target_kl] = {'reward': reward, 'kl': kl}
            print(f"  PPO (KL={target_kl}): reward={reward:.3f}, KL={kl:.3f}")
        else:
            print(f"  PPO (KL={target_kl}): model not found")
    
    # Evaluate PPO-GT models
    print("\n" + "="*60)
    print("Evaluating PPO-GT models")
    print("="*60)
    for target_kl in target_kls:
        model_path = f"./results/ppo_gt_kl_{target_kl}"
        if os.path.exists(model_path):
            reward, kl = evaluate_model(model_path, test_prompts, reference_model, reference_tokenizer, device)
            results['PPO-GT'][target_kl] = {'reward': reward, 'kl': kl}
            print(f"  PPO-GT (KL={target_kl}): reward={reward:.3f}, KL={kl:.3f}")
        else:
            print(f"  PPO-GT (KL={target_kl}): model not found")
    
    # Save results
    print("\nSaving results...")
    os.makedirs("./results", exist_ok=True)
    with open("./results/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log results to wandb
    wandb.log({"evaluation_results": results})
    
    # Create plots
    print("Creating plots...")
    plot_reward_kl_frontier(results)
    
    # Log plot to wandb
    if os.path.exists("./results/plots/reward_kl_frontier.png"):
        wandb.log({"reward_kl_frontier": wandb.Image("./results/plots/reward_kl_frontier.png")})
    
    wandb.finish()

def plot_reward_kl_frontier(results: Dict):
    """Create reward vs KL plot (Figure 2 from DPO paper)."""
    plt.figure(figsize=(12, 8))
    
    colors = {'DPO': '#1f77b4', 'PPO': '#ff7f0e', 'PPO-GT': '#2ca02c'}
    markers = {'DPO': 'o', 'PPO': 's', 'PPO-GT': '^'}
    
    # Plot each method
    for method in ['DPO', 'PPO', 'PPO-GT']:
        if results[method]:
            kls = []
            rewards = []
            
            for key in sorted(results[method].keys()):
                data = results[method][key]
                kls.append(data['kl'])
                rewards.append(data['reward'])
            
            # Sort by KL for plotting frontier
            sorted_pairs = sorted(zip(kls, rewards))
            kls, rewards = zip(*sorted_pairs)
            
            plt.plot(kls, rewards, 
                    label=method,
                    color=colors[method],
                    marker=markers[method],
                    markersize=10,
                    linewidth=2.5,
                    alpha=0.8)
    
    plt.xlabel('KL Divergence from Reference Policy', fontsize=12)
    plt.ylabel('Expected Reward (Sentiment Score)', fontsize=12)
    plt.title('Reward-KL Frontier: DPO vs PPO vs PPO-GT\nIMDb Sentiment Generation Task', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs("./results/plots", exist_ok=True)
    plt.savefig("./results/plots/reward_kl_frontier.png", dpi=300, bbox_inches='tight')
    print("Plot saved to ./results/plots/reward_kl_frontier.png")
    
    plt.close()

if __name__ == "__main__":
    evaluate_all_models()
