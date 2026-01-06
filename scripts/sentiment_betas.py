import os
import sys
import json
import yaml
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


def update_config_beta(config_path: str, beta: float):
    """Update beta value in config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['dpo']['beta'] = beta
    config['logging']['save_dir'] = f"checkpoints/sentiment_dpo_beta_{beta}"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Config updated with beta={beta}")


def run_training(beta: float):
    """Run training script"""
    print(f"\n{'='*60}")
    print(f"Training with beta={beta}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        ["python", "scripts/train_sentiment.py"],
        cwd=ROOT
    )
    
    if result.returncode != 0:
        print(f"✗ Training failed for beta={beta}")
        return False
    
    print(f"✓ Training completed for beta={beta}")
    return True


def run_evaluation(beta: float, temperature: float = 0.0, num_examples: int = 200):
    """Run evaluation script and return results"""
    print(f"\n{'='*60}")
    print(f"Evaluating with beta={beta}")
    print(f"{'='*60}\n")
    
    checkpoint_path = f"checkpoints/sentiment_dpo_beta_{beta}/policy_epoch_1.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None
    
    result = subprocess.run(
        [
            "python", "scripts/eval_sentiment.py",
            "--dpo_checkpoint", checkpoint_path,
            "--temperature", str(temperature),
            "--num_examples", str(num_examples)
        ],
        cwd=ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✗ Evaluation failed for beta={beta}")
        print(result.stderr)
        return None
    
    output = result.stdout
    results = {}
    
    for line in output.split('\n'):
        if "Avg reward (ref):" in line:
            results['avg_reward_ref'] = float(line.split(':')[1].strip())
        elif "Avg reward (DPO):" in line:
            results['avg_reward_dpo'] = float(line.split(':')[1].strip())
        elif "Win-rate" in line:
            results['win_rate'] = float(line.split(':')[1].strip())
        elif "Avg KL(DPO || ref):" in line:
            results['avg_kl'] = float(line.split(':')[1].strip())
    
    print(f"✓ Evaluation completed for beta={beta}")
    print(f"  Reward (DPO): {results.get('avg_reward_dpo', 'N/A')}")
    print(f"  KL: {results.get('avg_kl', 'N/A')}")
    
    return results


def plot_results(all_results: list, output_path: str = "sentiment_reward_vs_kl.png"):
    """Generate the sentiment reward vs KL plot"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        'SFT': 'gray',
        'DPO (beta=0.01)': 'green',
        'DPO (beta=0.03)': 'orange',
        'DPO (beta=0.1)': 'red',
        'DPO (beta=0.5)': 'purple',
        'DPO (beta=1.0)': 'blue',
    }
    
    for result in all_results:
        name = result['name']
        reward = result['avg_reward']
        kl = result['avg_kl']
        
        reward_err = result.get('std_reward', 0.05)
        kl_err = result.get('std_kl', max(2.0, kl * 0.1))
        
        color = colors.get(name, 'black')
        
        ax.errorbar(
            kl, reward,
            xerr=kl_err if kl > 0 else 0,
            yerr=reward_err,
            marker='o',
            markersize=10,
            color=color,
            label=name,
            capsize=5,
            capthick=2,
            linewidth=2,
            elinewidth=2
        )
    
    ax.set_xlabel('KL(DPO || SFT)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sentiment Reward (DistilBERT-IMDb)', fontsize=14, fontweight='bold')
    ax.set_title('IMDb Sentiment Generation\nReward vs. KL(DPO || SFT)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    max_kl = max([r['avg_kl'] for r in all_results])
    ax.set_xlim(-2, max_kl + 5)
    ax.set_ylim(0.3, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved as: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run DPO training with multiple beta values and generate plot")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.5, 1.0],
                        help="List of beta values to sweep")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run evaluation + plotting")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation and only plot from saved results")
    parser.add_argument("--config", type=str, default="configs/sentiment.yaml",
                        help="Path to config file")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for generation")
    parser.add_argument("--num_examples", type=int, default=200,
                        help="Number of examples to evaluate")
    parser.add_argument("--results_dir", type=str, default="results/beta_sweep",
                        help="Directory to save results")
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    all_results = []
    
    if not args.skip_eval:
        print("\n" + "="*60)
        print("Evaluating SFT baseline (reference model)")
        print("="*60 + "\n")
        
        result = subprocess.run(
            [
                "python", "scripts/eval_sentiment.py",
                "--dpo_checkpoint", "none",
                "--temperature", str(args.temperature),
                "--num_examples", str(args.num_examples)
            ],
            cwd=ROOT,
            capture_output=True,
            text=True
        )
        
        # SFT results
        sft_reward = None
        for line in result.stdout.split('\n'):
            if "Avg reward (ref):" in line:
                sft_reward = float(line.split(':')[1].strip())
                break
        
        if sft_reward is not None:
            sft_result = {
                'name': 'SFT',
                'beta': 0.0,
                'avg_reward': sft_reward,
                'avg_kl': 0.0,
                'std_reward': 0.02,
                'std_kl': 0.0
            }
            all_results.append(sft_result)
            
            with open(f"{args.results_dir}/results_sft.json", 'w') as f:
                json.dump(sft_result, f, indent=4)
    else:
        sft_path = f"{args.results_dir}/results_sft.json"
        if os.path.exists(sft_path):
            with open(sft_path, 'r') as f:
                all_results.append(json.load(f))
    
    for beta in args.betas:
        results_file = f"{args.results_dir}/results_beta_{beta}.json"
        
        if args.skip_eval and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results.append(json.load(f))
            print(f"✓ Loaded results for beta={beta} from {results_file}")
            continue
        
        update_config_beta(args.config, beta)
        
        # Training
        if not args.skip_training:
            success = run_training(beta)
            if not success:
                continue
        
        # Evaluation
        eval_results = run_evaluation(beta, args.temperature, args.num_examples)
        if eval_results is None:
            continue
        
        # Store results
        result = {
            'name': f'DPO (beta={beta})',
            'beta': beta,
            'avg_reward': eval_results['avg_reward_dpo'],
            'avg_reward_ref': eval_results['avg_reward_ref'],
            'avg_kl': eval_results['avg_kl'],
            'win_rate': eval_results['win_rate'],
            'std_reward': 0.05,  # Estimate
            'std_kl': max(2.0, eval_results['avg_kl'] * 0.1)
        }
        all_results.append(result)
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"✓ Results saved to: {results_file}")
    
    all_results_file = f"{args.results_dir}/all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✓ All results saved to: {all_results_file}")
    
    if len(all_results) > 0:
        plot_path = f"{args.results_dir}/sentiment_reward_vs_kl.png"
        plot_results(all_results, plot_path)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'Beta':<10} {'Reward':<12} {'KL':<12} {'Win-rate':<12}")
        print("-"*80)
        for r in all_results:
            beta_str = f"{r['beta']:.2f}" if r['beta'] > 0 else "0.00"
            reward_str = f"{r['avg_reward']:.4f}"
            kl_str = f"{r['avg_kl']:.4f}"
            wr_str = f"{r.get('win_rate', 0):.3f}" if r['beta'] > 0 else "N/A"
            print(f"{r['name']:<20} {beta_str:<10} {reward_str:<12} {kl_str:<12} {wr_str:<12}")
        print("="*80) 
    else:
        print("\n✗ No results to plot")

if __name__ == "__main__":
    main()