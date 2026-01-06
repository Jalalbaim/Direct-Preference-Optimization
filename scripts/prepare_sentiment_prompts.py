# scripts/prepare_sentiment_prompts.py
import os
import json
import argparse
from datasets import load_dataset
from tqdm.auto import tqdm

def write_prompts(ds, out_path: str, num_examples: int, max_prompt_chars: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = min(num_examples, len(ds))
    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(n), desc=f"Writing {out_path}"):
            text = ds[i]["text"]
            prompt = text[:max_prompt_chars].strip()
            if not prompt:
                continue
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_out", type=str, default="data/processed/sentiment/prompts_train.jsonl")
    p.add_argument("--val_out", type=str, default="data/processed/sentiment/prompts_val.jsonl")
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--num_val", type=int, default=500)
    p.add_argument("--max_prompt_chars", type=int, default=300)
    args = p.parse_args()

    imdb = load_dataset("imdb")
    write_prompts(imdb["train"], args.train_out, args.num_train, args.max_prompt_chars)
    write_prompts(imdb["test"], args.val_out, args.num_val, args.max_prompt_chars)
    print("Done.")

if __name__ == "__main__":
    main()
