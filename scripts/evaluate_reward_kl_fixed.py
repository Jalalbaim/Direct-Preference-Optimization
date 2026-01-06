#!/usr/bin/env python3
"""
Évaluation Reward vs KL pour DPO/PPO models.
Charge les prompts depuis le dataset test sauvegardé, sélectionne 50 au hasard,
tronque à 2-8 tokens, puis génère et évalue.

CORRECTION: Calcul correct de la KL divergence au niveau token avec shift et masking.
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ============================================================================
# CONFIGURATION
# ============================================================================

SAVE_BASE_PATH = "drive/MyDrive/dpo_ppo_training"  # Colab path
TEST_DATASET_PATH = f"{SAVE_BASE_PATH}/datasets/test_final"
SENTIMENT_MODEL = "siebert/sentiment-roberta-large-english"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paramètres d'évaluation
NUM_PROMPTS = 50  # 50 prompts au hasard
NUM_COMPLETIONS = 5  # Générations par prompt
MAX_NEW_TOKENS = 50

# Fonction pour trouver le dernier checkpoint
def get_latest_checkpoint(model_dir):
    """Retourne le dernier checkpoint dans un répertoire."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None
    
    checkpoint_dir = model_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return str(model_dir)
    
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return str(model_dir)
    
    checkpoints_sorted = sorted(
        checkpoints,
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
    )
    
    return str(checkpoints_sorted[-1])

# Modèles DPO à évaluer (dernier checkpoint de chaque beta)
DPO_BETAS = [0.05, 0.1, 1.0, 5.0]
DPO_MODEL_PATHS = []

for beta in DPO_BETAS:
    model_dir = f"{SAVE_BASE_PATH}/dpo_beta_{beta}"
    latest_ckpt = get_latest_checkpoint(model_dir)
    if latest_ckpt is None:
        print(f"⚠️ DPO β={beta}: répertoire absent -> skip")
        continue
    if not Path(latest_ckpt).exists():
        print(f"⚠️ DPO β={beta}: chemin introuvable {latest_ckpt} -> skip")
        continue
    DPO_MODEL_PATHS.append((beta, latest_ckpt))
    print(f"DPO β={beta}: {latest_ckpt}")

REF_MODEL_PATH = f"{SAVE_BASE_PATH}/sft_model"

print("="*80)
print("ÉVALUATION REWARD vs KL (KL CORRIGÉE)")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Test dataset: {TEST_DATASET_PATH}")
print(f"Nombre de prompts: {NUM_PROMPTS}")
print(f"Complétions par prompt: {NUM_COMPLETIONS}")
print("="*80 + "\n")

# ============================================================================
# STEP 1. LOAD REFERENCE MODEL
# ============================================================================

print("Chargement du modèle de référence (SFT)...")
ref_tokenizer = AutoTokenizer.from_pretrained(REF_MODEL_PATH)
if ref_tokenizer.pad_token is None:
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
ref_model = AutoModelForCausalLM.from_pretrained(REF_MODEL_PATH).to(DEVICE).eval()
print(f"✅ Référence chargée: {REF_MODEL_PATH}\n")

# ============================================================================
# STEP 2. LOAD SENTIMENT CLASSIFIER & TEST PROMPTS
# ============================================================================

print("Chargement du sentiment classifier...")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    device=0 if DEVICE == "cuda" else -1,
    truncation=True,
    max_length=512
)
print(f"✅ Classifier chargé: {SENTIMENT_MODEL}\n")

print(f"Chargement du dataset test depuis: {TEST_DATASET_PATH}")
test_dataset = load_from_disk(TEST_DATASET_PATH)
print(f"✅ Dataset chargé: {len(test_dataset)} exemples")

random.seed(42)
selected_indices = random.sample(range(len(test_dataset)), NUM_PROMPTS)
selected_samples = test_dataset.select(selected_indices)

prompts = []
for sample in selected_samples:
    text = sample["text"]
    tokens = ref_tokenizer.encode(text, add_special_tokens=False)
    num_tokens = random.randint(2, 8)
    prompt_tokens = tokens[:num_tokens]
    prompt = ref_tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    prompts.append(prompt)

print(f"✅ {len(prompts)} prompts créés (2-8 tokens chacun)")
print(f"\nExemples de prompts:")
for i in range(min(3, len(prompts))):
    print(f"  {i+1}. '{prompts[i]}'")
print()

# ============================================================================
# STEP 3. DEFINE EVALUATION FUNCTIONS
# ============================================================================

def generate_completions(model, tokenizer, prompt, num_completions=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_completions
    )
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    completions = [t[len(prompt):].strip() for t in texts]
    return completions

def sentiment_reward(texts):
    preds = sentiment_pipe(texts, truncation=True, max_length=512)
    probs = [p["score"] if p["label"].upper() == "POSITIVE" else 1 - p["score"] for p in preds]
    return np.mean(probs)

@torch.no_grad()
def compute_kl_divergence(policy_model, policy_tok, ref_model, ref_tok, prompt, completion, debug: bool = False):
    """
    Compute true KL(policy || ref) = Sum [ P_policy(w) * (log P_policy(w) - log P_ref(w)) ]
    """
    full_text = prompt + completion
    enc = policy_tok(full_text, return_tensors="pt").to(DEVICE)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    prompt_len = len(policy_tok.encode(prompt, add_special_tokens=True))
    prompt_ids = policy_tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    prompt_len = prompt_ids.shape[1]
    
    policy_outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
    
    policy_logits = policy_outputs.logits[:, :-1, :].contiguous()
    ref_logits = ref_outputs.logits[:, :-1, :].contiguous()
    shift_attn = attention_mask[:, 1:].contiguous()
    
    policy_probs = torch.softmax(policy_logits, dim=-1)
    policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
    
    kl_per_position = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
    
    completion_mask = torch.zeros_like(shift_attn)
    start_idx = max(prompt_len - 1, 0)
    if start_idx < shift_attn.shape[1]:
        completion_mask[:, start_idx:] = shift_attn[:, start_idx:]
    
    kl_per_position = kl_per_position * completion_mask
    
    num_tokens = completion_mask.sum().item()
    if num_tokens > 0:
        kl_divergence = kl_per_position.sum().item() / num_tokens
    else:
        kl_divergence = 0.0

    if debug:
        seq_len = input_ids.shape[1]
        print(f"[KL DEBUG] prompt_len={prompt_len}, seq_len={seq_len}, num_tokens={num_tokens}, KL={kl_divergence:.6f}")
    
    return kl_divergence

def evaluate_model(policy_path, beta):
    print(f"\n📊 Évaluation: DPO β={beta}")
    print(f"   Chemin: {policy_path}")
    
    policy_tok = AutoTokenizer.from_pretrained(policy_path)
    if policy_tok.pad_token is None:
        policy_tok.pad_token = policy_tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(policy_path).to(DEVICE).eval()

    rewards = []
    kls = []

    for prompt in prompts:
        comps = generate_completions(policy, policy_tok, prompt, NUM_COMPLETIONS)
        r = sentiment_reward([prompt + " " + c for c in comps])
        rewards.append(r)

        kl_samples = []
        for c in comps:
            kl = compute_kl_divergence(policy, policy_tok, ref_model, ref_tokenizer, prompt, c)
            kl = compute_kl_divergence(policy, policy_tok, ref_model, ref_tokenizer, prompt, c, debug=True)
            kl_samples.append(kl)
        kls.append(np.mean(kl_samples))

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "kl_mean": float(np.mean(kls)),
        "kl_std": float(np.std(kls)),
    }

def evaluate_sft_baseline():
    """Évalue le modèle SFT de base (baseline) sur reward et KL (vs référence)."""
    print("\n" + "="*80)
    print("ÉVALUATION DU MODÈLE SFT (BASELINE)")
    print("="*80)
    print(f"   Chemin: {REF_MODEL_PATH}")

    baseline_tok = AutoTokenizer.from_pretrained(REF_MODEL_PATH)
    if baseline_tok.pad_token is None:
        baseline_tok.pad_token = baseline_tok.eos_token
    baseline_model = AutoModelForCausalLM.from_pretrained(REF_MODEL_PATH).to(DEVICE).eval()

    rewards = []
    kls = []

    for prompt in prompts:
        comps = generate_completions(baseline_model, baseline_tok, prompt, NUM_COMPLETIONS)
        r = sentiment_reward([prompt + " " + c for c in comps])
        rewards.append(r)

        # KL vs ref (identique en pratique → proche de 0)
        kl_samples = []
        for c in comps:
            kl = compute_kl_divergence(baseline_model, baseline_tok, ref_model, ref_tokenizer, prompt, c)
            kl_samples.append(kl)
        kls.append(np.mean(kl_samples))

    res = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "kl_mean": float(np.mean(kls)),
        "kl_std": float(np.std(kls)),
    }

    print(f"✅ SFT baseline:")
    print(f"   Reward: {res['reward_mean']:.4f} ± {res['reward_std']:.4f}")
    print(f"   KL: {res['kl_mean']:.4f} ± {res['kl_std']:.4f}")

    return res

# ============================================================================
# STEP 4. EVALUATE ALL MODELS
# ============================================================================

print("\n" + "="*80)
print("ÉVALUATION DES MODÈLES DPO")
print("="*80)

results = []
for beta, path in DPO_MODEL_PATHS:
    if not Path(path).exists():
        print(f"❌ Skip β={beta}: chemin introuvable {path}")
        continue
    try:
        res = evaluate_model(path, beta)
        res["model"] = f"dpo_beta_{beta}"
        res["beta"] = beta
        results.append(res)
        print(f"✅ DPO β={beta}:")
        print(f"   Reward: {res['reward_mean']:.4f} ± {res['reward_std']:.4f}")
        print(f"   KL: {res['kl_mean']:.4f} ± {res['kl_std']:.4f}")
    except Exception as e:
        print(f"❌ Skip β={beta}: {e}")

# Évaluer le modèle SFT baseline et l'ajouter aux résultats
baseline_res = evaluate_sft_baseline()
baseline_res["model"] = "sft_baseline"
baseline_res["beta"] = None
results.append(baseline_res)

# ============================================================================
# STEP 5. PLOT REWARD VS KL
# ============================================================================

print("\n" + "="*80)
print("GÉNÉRATION DU GRAPHIQUE REWARD vs KL")
print("="*80)

if results:
    xs = [r["kl_mean"] for r in results]
    ys = [r["reward_mean"] for r in results]
    labels = [r["model"] for r in results]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, color="royalblue", s=100, alpha=0.7)
    for x, y, lbl in zip(xs, ys, labels):
        plt.text(x, y + 0.01, lbl, fontsize=9, ha="center", va="bottom")
    plt.xlabel("KL(π_θ ∥ π_ref)", fontsize=12)
    plt.ylabel("Sentiment Reward (positive prob)", fontsize=12)
    plt.title("Reward vs KL - DPO Models\n(50 prompts, 2-8 tokens, KL corrigée)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = f"{SAVE_BASE_PATH}/reward_kl_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {save_path}")
    
    plt.show()
else:
    print("❌ Aucun résultat à tracer (aucun checkpoint chargé).")

print("\n" + "="*80)
print("✅ ÉVALUATION TERMINÉE")
print("="*80)
