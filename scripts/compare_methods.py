"""
Script pour comparer les performances de DPO, PPO et GRPO.
Évalue les modèles entraînés avec chaque méthode.
"""
import os
import sys
import json
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dpo.reward_models import RewardModel
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


class ModelEvaluator:
    """Évalue un modèle sur des prompts de test."""

    def __init__(self, model_path: str, model_name: str =
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Charger le modèle
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Charger les poids entraînés si disponibles
        if os.path.exists(model_path):
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.eval()
        self.reward_model = RewardModel()

    @torch.no_grad()
    def generate_and_evaluate(
        self, prompts: List[str], max_length: int = 128, num_samples: int = 3
    ) -> Dict:
        """Génère des réponses et calcule les rewards moyens."""
        all_rewards = []
        all_responses = []

        for prompt in prompts:
            # Générer plusieurs réponses par prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )     
            # Décoder
            responses = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )           
            # Calculer rewards
            rewards = self.reward_model.compute_rewards(responses)
            all_rewards.extend(rewards.cpu().tolist())
            all_responses.extend(responses)

        return {
            "mean_reward": sum(all_rewards) / len(all_rewards),
            "std_reward": torch.tensor(all_rewards).std().item(),
            "responses": all_responses[:10],  # Garder quelques exemples
        }


def load_test_prompts(num_prompts: int = 20) -> List[str]:
    """Charge des prompts de test."""
    prompts_file = "data/processed/sentiment/prompts.jsonl"
    
    prompts = []
    if os.path.exists(prompts_file):
        with open(prompts_file, "r") as f:
            for i, line in enumerate(f):
                if i >= num_prompts:
                    break
                obj = json.loads(line)
                prompts.append(obj["prompt"])
    else:
        # Prompts par défaut pour le test
        prompts = [
            "The movie was",
            "I think this product is",
            "The customer service was",
            "My experience at the restaurant was",
            "The book I just read was",
        ]
    
    return prompts


def main():
    print("=" * 80)
    print("Comparing DPO, PPO, and GRPO")
    print("=" * 80)
    
    # Chemins des checkpoints
    checkpoints = {
        "DPO": "checkpoints/sentiment_dpo/policy_epoch_1.pt",
        "PPO": "checkpoints/sentiment_ppo/policy_ppo_epoch_1.pt",
        "GRPO": "checkpoints/sentiment_grpo/policy_grpo_epoch_1.pt",
        "Base (no training)": None,  # Modèle de base sans entraînement
    }
    
    # Charger les prompts de test
    test_prompts = load_test_prompts(num_prompts=20)
    print(f"\nEvaluating on {len(test_prompts)} test prompts")
    
    results = {}
    
    for method_name, checkpoint_path in checkpoints.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating {method_name}...")
        print(f"{'=' * 80}")
        
        try:
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                if method_name == "Base (no training)":
                    # Évaluer le modèle de base
                    evaluator = ModelEvaluator("", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                else:
                    print(f"Checkpoint not found: {checkpoint_path}")
                    print("Skipping...")
                    continue
            else:
                evaluator = ModelEvaluator(checkpoint_path)
            
            result = evaluator.generate_and_evaluate(test_prompts, num_samples=3)
            results[method_name] = result
            
            print(f"\n{method_name} Results:")
            print(f"  Mean Reward: {result['mean_reward']:.4f}")
            print(f"  Std Reward:  {result['std_reward']:.4f}")
            print(f"\n  Sample responses:")
            for i, resp in enumerate(result['responses'][:3], 1):
                print(f"    {i}. {resp[:100]}...")
                
        except Exception as e:
            print(f"Error evaluating {method_name}: {e}")
            continue
    
    # Résumé comparatif
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Method':<20} {'Mean Reward':<15} {'Std Reward':<15}")
    print("-" * 50)
    
    for method_name, result in sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
        print(f"{method_name:<20} {result['mean_reward']:<15.4f} {result['std_reward']:<15.4f}")
    
    # Sauvegarder les résultats
    output_file = "comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
