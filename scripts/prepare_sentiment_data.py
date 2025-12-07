# scripts/prepare_sentiment_data_ollama.py

import os
import sys
import json
import argparse
from typing import List, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ollama


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare preference data for DPO on sentiment generation (IMDb) using Ollama (gemma3:4b)."
    )

    parser.add_argument(
        "--ollama_model",
        type=str,
        default="gemma3:4b",
        help="Ollama model name to use for generation (e.g., gemma3:4b).",
    )
    parser.add_argument(
        "--clf_model_name",
        type=str,
        default="lvwerra/distilbert-imdb",
        help="Sentiment classifier model name (HF).",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="data/processed/sentiment/train_preferences.jsonl",
        help="Where to save the train preferences JSONL.",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="data/processed/sentiment/val_preferences.jsonl",
        help="Where to save the val preferences JSONL.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=2000,
        help="Number of train preference pairs to create.",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=500,
        help="Number of val preference pairs to create.",
    )
    parser.add_argument(
        "--max_prompt_chars",
        type=int,
        default=300,
        help="Max number of characters for the prompt (truncated IMDb review).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens to generate for each completion (num_predict for Ollama).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for the classifier (cuda or cpu).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling.",
    )

    return parser.parse_args()


def load_sentiment_classifier(model_name: str, device: str):
    clf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    clf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf_model.to(device)
    clf_model.eval()
    return clf_tokenizer, clf_model


@torch.no_grad()
def sentiment_score(
    texts: List[str],
    clf_tokenizer,
    clf_model,
    device: str,
) -> List[float]:
    """
    Retourne un score de "positivité" pour chaque texte.
    Pour un modèle binaire IMDb: probabilité de la classe 'positive' (index 1).
    """
    enc = clf_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    outputs = clf_model(**enc)
    logits = outputs.logits   # [B, 2]
    probs = torch.softmax(logits, dim=-1)
    pos_probs = probs[:, 1]   # classe 'positive'
    return pos_probs.cpu().tolist()


def generate_one_completion_ollama(
    prompt: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Génère UNE complétion avec Ollama pour un prompt donné.
    On demande au modèle de continuer le texte/review.
    """

    # Tu peux affiner le prompt ici si tu veux forcer le style
    system_prompt = (
        "You are a helpful assistant that continues movie reviews in a consistent style."
    )
    user_content = (
        f"Continue this movie review in English. "
        f"Keep the same topic and style.\n\nReview start:\n{prompt}\n\nContinuation:"
    )

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        options={
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_new_tokens),
        },
    )

    completion = response["message"]["content"].strip()
    return completion


def generate_two_completions_ollama(
    prompt: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    """
    Génère deux complétions en appelant deux fois Ollama.
    """
    resp_a = generate_one_completion_ollama(
        prompt=prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    resp_b = generate_one_completion_ollama(
        prompt=prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return resp_a, resp_b


def prepare_split(
    split_name: str,
    dataset,
    num_examples: int,
    output_path: str,
    ollama_model: str,
    clf_tokenizer,
    clf_model,
    device: str,
    max_prompt_chars: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_total = min(num_examples, len(dataset))
    print(f"Preparing {split_name} split: {n_total} examples → {output_path}")

    with open(output_path, "w", encoding="utf-8") as f_out:
        pbar = tqdm(range(n_total), desc=f"{split_name} examples")

        for i in pbar:
            row = dataset[i]
            text = row["text"]

            # prompt = début de la review IMDb tronqué
            prompt = text[:max_prompt_chars].strip()
            if not prompt:
                continue

            # Génère deux réponses via Ollama
            try:
                resp_a, resp_b = generate_two_completions_ollama(
                    prompt=prompt,
                    model_name=ollama_model,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                print(f"[{split_name}] Generation error on example {i}: {e}")
                continue

            if not resp_a.strip() or not resp_b.strip():
                continue

            # Score de sentiment
            try:
                scores = sentiment_score(
                    [resp_a, resp_b],
                    clf_tokenizer=clf_tokenizer,
                    clf_model=clf_model,
                    device=device,
                )
            except Exception as e:
                print(f"[{split_name}] Classifier error on example {i}: {e}")
                continue

            score_a, score_b = scores

            # On choisit la complétion la plus positive comme "chosen"
            if score_a > score_b:
                chosen, rejected = resp_a, resp_b
                score_chosen, score_rejected = score_a, score_b
            elif score_b > score_a:
                chosen, rejected = resp_b, resp_a
                score_chosen, score_rejected = score_b, score_a
            else:
                # égalité: on garde quand même, chosen = resp_a
                chosen, rejected = resp_a, resp_b
                score_chosen, score_rejected = score_a, score_b

            example = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "score_chosen": float(score_chosen),
                "score_rejected": float(score_rejected),
            }
            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    device = args.device

    print(f"Using device for classifier: {device}")
    print(f"Using Ollama model: {args.ollama_model}")

    # 1) Chargement IMDb
    imdb = load_dataset("imdb")
    train_ds = imdb["train"]
    test_ds = imdb["test"]

    # 2) Classifieur de sentiment
    print(f"Loading sentiment classifier: {args.clf_model_name}")
    clf_tokenizer, clf_model = load_sentiment_classifier(
        args.clf_model_name,
        device=device,
    )

    # 3) Préparation train
    prepare_split(
        split_name="train",
        dataset=train_ds,
        num_examples=args.num_train,
        output_path=args.train_output,
        ollama_model=args.ollama_model,
        clf_tokenizer=clf_tokenizer,
        clf_model=clf_model,
        device=device,
        max_prompt_chars=args.max_prompt_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # 4) Préparation val
    prepare_split(
        split_name="val",
        dataset=test_ds,
        num_examples=args.num_val,
        output_path=args.val_output,
        ollama_model=args.ollama_model,
        clf_tokenizer=clf_tokenizer,
        clf_model=clf_model,
        device=device,
        max_prompt_chars=args.max_prompt_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("Done.")


if __name__ == "__main__":
    main()
