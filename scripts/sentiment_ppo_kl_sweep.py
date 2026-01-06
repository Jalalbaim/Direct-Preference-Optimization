# scripts/sentiment_ppo_kl_sweep.py
import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_config", type=str, default="configs/sentiment_ppo.yaml")
    p.add_argument("--kl_coefs", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2])
    p.add_argument("--results_dir", type=str, default="results/ppo_sweep")
    p.add_argument("--num_examples", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    with open(os.path.join(ROOT, args.base_config), "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    all_results = []

    for kl in args.kl_coefs:
        cfg = dict(base_cfg)
        cfg["ppo"] = dict(base_cfg["ppo"])
        cfg["logging"] = dict(base_cfg["logging"])

        cfg["ppo"]["kl_coef"] = float(kl)
        cfg["logging"]["save_dir"] = f"checkpoints/sentiment_ppo_kl_{kl}"

        tmp_cfg_path = os.path.join(args.results_dir, f"tmp_ppo_kl_{kl}.yaml")
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # train
        subprocess.run([sys.executable, "scripts/train_sentiment_ppo.py", "--config", tmp_cfg_path], cwd=ROOT)

        # eval via existing eval_sentiment
        ckpt = os.path.join(ROOT, cfg["logging"]["save_dir"], "policy_epoch_1.pt")
        eval_cmd = [
            sys.executable, "scripts/eval_sentiment.py",
            "--dpo_checkpoint", ckpt,
            "--num_examples", str(args.num_examples),
            "--temperature", str(args.temperature),
        ]
        r = subprocess.run(eval_cmd, cwd=ROOT, capture_output=True, text=True)

        # parse quickly
        out = r.stdout
        parsed = {"kl_coef": kl, "checkpoint": ckpt}
        for line in out.splitlines():
            if "Avg reward (ref):" in line:
                parsed["avg_reward_ref"] = float(line.split(":")[1].strip())
            if "Avg reward (DPO):" in line:
                parsed["avg_reward_ppo"] = float(line.split(":")[1].strip())
            if "Avg KL(DPO || ref):" in line:
                parsed["avg_kl"] = float(line.split(":")[1].strip())
            if "Win-rate" in line:
                parsed["win_rate"] = float(line.split(":")[1].strip())

        all_results.append(parsed)

        with open(os.path.join(args.results_dir, f"results_kl_{kl}.json"), "w") as f:
            json.dump(parsed, f, indent=2)

    with open(os.path.join(args.results_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("Done. Results in:", args.results_dir)

if __name__ == "__main__":
    main()
