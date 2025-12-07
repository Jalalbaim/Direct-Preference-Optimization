import os
import sys
import argparse

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dpo.models import load_models, compute_logprobs
from src.dpo.utils import load_yaml_config


def load_sentiment_classifier(model_name: str, device: str):
    clf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=True,  # important pour ton erreur torch
    )
    clf_model.to(device)
    clf_model.eval()
    return clf_tokenizer, clf_model


@torch.no_grad()
def sentiment_score(texts, clf_tokenizer, clf_model, device: str):
    enc = clf_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    outputs = clf_model(**enc)
    probs = torch.softmax(outputs.logits, dim=-1)
    pos_probs = probs[:, 1]  # classe positive
    return pos_probs.cpu()


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8, top_p=0.9, device="cuda"):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    prompt_len = input_ids.shape[1]

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip(), out[0]


def sequence_logprob(model, input_ids, attention_mask, device: str):
    # calcule logp(seq) (somme sur tokens) sous un modèle causal
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [1, L, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_attn = attention_mask[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * shift_attn  # ignore padding
    return token_logps.sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sentiment.yaml")
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--max_prompt_chars", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--ref_model_name", type=str, default=None,
                        help="HF name of the reference model (if None, use config.model.name)")
    parser.add_argument("--dpo_checkpoint", type=str,
                        default="checkpoints/sentiment_dpo/policy_epoch_1.pt")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ref model & policy model DPO
    ref_model_name = args.ref_model_name or config["model"]["name"]
    print(f"Loading reference + policy models from: {ref_model_name}")

    mb = load_models(ref_model_name, dtype=config["model"]["dtype"])
    tokenizer = mb.tokenizer
    ref_model = mb.ref_model
    policy_model = mb.policy_model

    # checkpoint DPO
    if args.dpo_checkpoint and os.path.exists(args.dpo_checkpoint):
        ckpt = torch.load(args.dpo_checkpoint, map_location=device)
        policy_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded DPO checkpoint: {args.dpo_checkpoint}")
    else:
        print("WARNING: DPO checkpoint not found, using base model as policy.")

    ref_model.to(device)
    policy_model.to(device)
    ref_model.eval()
    policy_model.eval()

    # Classifier sentiment
    clf_name = "lvwerra/distilbert-imdb"
    clf_tokenizer, clf_model = load_sentiment_classifier(clf_name, device)

    # IMDb test
    imdb = load_dataset("imdb")
    test_ds = imdb["test"]

    # Evaluation
    rewards_ref = []
    rewards_dpo = []
    kls = []
    wins = 0

    n = min(args.num_examples, len(test_ds))
    print(f"Evaluating on {n} IMDb test examples")

    for i in tqdm(range(n)):
        text = test_ds[i]["text"]
        prompt = text[: args.max_prompt_chars].strip()
        if not prompt:
            continue

        # ref
        resp_ref, full_ids_ref = generate_response(
            ref_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        # DPO
        resp_dpo, full_ids_dpo = generate_response(
            policy_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        # Reward = proba positive
        scores = sentiment_score(
            [resp_ref, resp_dpo],
            clf_tokenizer,
            clf_model,
            device=device,
        )
        r_ref = float(scores[0])
        r_dpo = float(scores[1])

        rewards_ref.append(r_ref)
        rewards_dpo.append(r_dpo)
        if r_dpo > r_ref:
            wins += 1

        # KL approx: logp_dpo(seq_dpo) - logp_ref(seq_dpo)
        attn_dpo = (full_ids_dpo != tokenizer.pad_token_id).long()
        logp_dpo = sequence_logprob(policy_model, full_ids_dpo, attn_dpo, device)
        logp_ref = sequence_logprob(ref_model, full_ids_dpo, attn_dpo, device)
        kl_point = (logp_dpo - logp_ref)  # approx, en nats
        kls.append(kl_point)

    import numpy as np

    avg_r_ref = float(np.mean(rewards_ref))
    avg_r_dpo = float(np.mean(rewards_dpo))
    win_rate = wins / len(rewards_ref) if rewards_ref else 0.0
    avg_kl = float(np.mean(kls)) if kls else 0.0

    print("==== Sentiment DPO Evaluation ====")
    print(f"Avg reward (ref): {avg_r_ref:.4f}")
    print(f"Avg reward (DPO): {avg_r_dpo:.4f}")
    print(f"Win-rate (DPO > ref): {win_rate:.3f}")
    print(f"Approx KL (DPO || ref): {avg_kl:.4f}")


if __name__ == "__main__":
    main()
