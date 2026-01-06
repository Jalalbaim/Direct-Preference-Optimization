# scripts/eval_sentiment_ppo.py
import os
import sys
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_examples", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    cmd = [
        sys.executable, "scripts/eval_sentiment.py",
        "--dpo_checkpoint", args.checkpoint,
        "--num_examples", str(args.num_examples),
        "--temperature", str(args.temperature),
    ]
    subprocess.run(cmd, cwd=ROOT, check=False)

if __name__ == "__main__":
    main()
