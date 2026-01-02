"""
Évaluation améliorée avec calcul correct de la KL divergence.
Calcule la KL uniquement sur la partie générée (réponse), pas le prompt.
"""
import os
import sys
import argparse

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dpo.models import load_models
from src.dpo.utils import load_yaml_config


def load_sentiment_classifier(model_name: str, device: str):
    clf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=True,
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
def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=64,
    temperature=0.8,
    top_p=0.9,
    device="cuda"
):
    """Génère une réponse et retourne (texte, full_ids, prompt_length)."""
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
    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    return response_text, out[0], prompt_len


@torch.no_grad()
def compute_token_logprobs(model, input_ids, attention_mask, device: str):
    """
    Calcule les log-probs pour chaque token de la séquence.
    
    Returns:
        token_logprobs: Tensor [L-1] avec log p(token_i | tokens_{<i})
    """
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [1, L, V]
    
    # Shift pour aligner avec les cibles
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_attn = attention_mask[:, 1:].contiguous()

    # Log-probs
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Masquer le padding
    token_logps = token_logps * shift_attn
    
    return token_logps.squeeze(0)  # [L-1]


@torch.no_grad()
def compute_kl_divergence(
    policy_model,
    ref_model,
    full_ids,
    attention_mask,
    prompt_len,
    device: str
):
    """
    Calcule la KL divergence sur la réponse générée uniquement.
    
    Args:
        policy_model: Modèle policy (DPO/PPO/GRPO)
        ref_model: Modèle de référence
        full_ids: IDs complets (prompt + réponse)
        attention_mask: Masque d'attention
        prompt_len: Longueur du prompt en tokens
        device: Device
        
    Returns:
        kl_total: KL totale sur la réponse
        kl_mean_per_token: KL moyenne par token de réponse
        response_length: Nombre de tokens dans la réponse
    """
    # Calculer les log-probs par token pour chaque modèle
    logps_policy = compute_token_logprobs(
        policy_model, full_ids, attention_mask, device
    )
    logps_ref = compute_token_logprobs(
        ref_model, full_ids, attention_mask, device
    )
    
    # Masque pour la réponse uniquement (exclure le prompt)
    # Note: logps sont décalés de 1, donc prompt_len - 1
    response_mask = torch.zeros_like(logps_policy)
    if prompt_len - 1 < len(response_mask):
        response_mask[prompt_len - 1:] = 1.0
    
    # KL divergence par token: log(p_policy) - log(p_ref)
    kl_per_token = (logps_policy - logps_ref) * response_mask
    
    # KL totale sur la réponse
    kl_total = kl_per_token.sum().item()
    
    # Nombre de tokens dans la réponse
    response_length = response_mask.sum().item()
    
    # KL moyenne par token
    kl_mean_per_token = kl_total / response_length if response_length > 0 else 0.0
    
    return kl_total, kl_mean_per_token, response_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sentiment.yaml")
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--max_prompt_chars", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Nombre de générations par prompt pour réduire variance")
    parser.add_argument("--ref_model_name", type=str, default=None)
    parser.add_argument("--dpo_checkpoint", type=str,
                        default="checkpoints/sentiment_dpo/policy_epoch_1.pt")
    parser.add_argument("--method", type=str, default="dpo",
                        choices=["dpo", "ppo", "grpo"],
                        help="Méthode à évaluer")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")

    # Charger les modèles
    ref_model_name = args.ref_model_name or config["model"]["name"]
    print(f"Loading reference + policy models from: {ref_model_name}")

    mb = load_models(ref_model_name, dtype=config["model"]["dtype"])
    tokenizer = mb.tokenizer
    ref_model = mb.ref_model
    policy_model = mb.policy_model

    # Charger le checkpoint
    checkpoint_map = {
        "dpo": "checkpoints/sentiment_dpo/policy_epoch_1.pt",
        "ppo": "checkpoints/sentiment_ppo/policy_ppo_epoch_1.pt",
        "grpo": "checkpoints/sentiment_grpo/policy_grpo_epoch_1.pt",
    }
    
    checkpoint_path = checkpoint_map.get(args.method, args.dpo_checkpoint)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Gérer les différents formats de checkpoint
        if "model_state_dict" in ckpt:
            policy_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            policy_model.load_state_dict(ckpt, strict=False)
        print(f"Loaded {args.method.upper()} checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found, using base model as policy.")

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

    # Métriques
    rewards_ref = []
    rewards_policy = []
    kls_total = []
    kls_per_token = []
    response_lengths = []
    wins = 0

    n = min(args.num_examples, len(test_ds))
    print(f"Evaluating on {n} IMDb test examples")
    print(f"Generating {args.num_samples} samples per prompt")

    for i in tqdm(range(n)):
        text = test_ds[i]["text"]
        prompt = text[: args.max_prompt_chars].strip()
        if not prompt:
            continue

        # Pour réduire la variance, on génère plusieurs réponses
        for _ in range(args.num_samples):
            # Modèle de référence
            resp_ref, full_ids_ref, prompt_len_ref = generate_response(
                ref_model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )

            # Modèle policy
            resp_policy, full_ids_policy, prompt_len_policy = generate_response(
                policy_model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )

            # Rewards
            scores = sentiment_score(
                [resp_ref, resp_policy],
                clf_tokenizer,
                clf_model,
                device=device,
            )
            r_ref = float(scores[0])
            r_policy = float(scores[1])

            rewards_ref.append(r_ref)
            rewards_policy.append(r_policy)
            if r_policy > r_ref:
                wins += 1

            # KL divergence (sur la réponse uniquement)
            attn_policy = (full_ids_policy != tokenizer.pad_token_id).long()
            
            kl_total, kl_mean, resp_len = compute_kl_divergence(
                policy_model,
                ref_model,
                full_ids_policy,
                attn_policy,
                prompt_len_policy,
                device,
            )
            
            kls_total.append(kl_total)
            kls_per_token.append(kl_mean)
            response_lengths.append(resp_len)

    # Statistiques
    import numpy as np

    avg_r_ref = float(np.mean(rewards_ref))
    avg_r_policy = float(np.mean(rewards_policy))
    win_rate = wins / len(rewards_ref) if rewards_ref else 0.0
    
    avg_kl_total = float(np.mean(kls_total)) if kls_total else 0.0
    avg_kl_per_token = float(np.mean(kls_per_token)) if kls_per_token else 0.0
    avg_response_len = float(np.mean(response_lengths)) if response_lengths else 0.0

    print("\n" + "=" * 60)
    print(f"Sentiment {args.method.upper()} Evaluation ")
    print("=" * 60)
    print(f"\n Reward Metrics:")
    print(f"  Avg reward (ref):      {avg_r_ref:.4f}")
    print(f"  Avg reward ({args.method}):     {avg_r_policy:.4f}")
    print(f"  Improvement:           {avg_r_policy - avg_r_ref:+.4f}")
    print(f"  Win-rate ({args.method} > ref): {win_rate:.3f}")
    
    print(f"\n KL Divergence (response only):")
    print(f"  KL total (avg):        {avg_kl_total:.4f} ")
    print(f"  KL per token (avg):    {avg_kl_per_token:.4f}")
    print(f"  Avg response length:   {avg_response_len:.1f} tokens")
    

if __name__ == "__main__":
    main()
