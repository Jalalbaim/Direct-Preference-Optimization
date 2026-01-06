#!/usr/bin/env python3
"""
Évaluation Reward vs KL pour DPO/PPO models.
Charge les prompts depuis le dataset test sauvegardé, sélectionne 50 au hasard,
tronque à 2-8 tokens, puis génère et évalue.
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
    checkpoint_dir = Path(model_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        # Si pas de checkpoints, utiliser le modèle principal
        return model_dir
    
    # Chercher tous les checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return model_dir
    
    # Trier par numéro de step (extraire le nombre après "checkpoint-")
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
    DPO_MODEL_PATHS.append(latest_ckpt)
    print(f"DPO β={beta}: {latest_ckpt}")

# Modèle de référence (SFT)
REF_MODEL_PATH = f"{SAVE_BASE_PATH}/sft_model"

print("="*80)
print("ÉVALUATION REWARD vs KL")
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

# Sentiment classifier
print("Chargement du sentiment classifier...")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    device=0 if DEVICE == "cuda" else -1,
    truncation=True,
    max_length=512
)
print(f"✅ Classifier chargé: {SENTIMENT_MODEL}\n")

# Load test dataset
print(f"Chargement du dataset test depuis: {TEST_DATASET_PATH}")
test_dataset = load_from_disk(TEST_DATASET_PATH)
print(f"✅ Dataset chargé: {len(test_dataset)} exemples")

# Sélectionner 50 prompts au hasard
random.seed(42)
selected_indices = random.sample(range(len(test_dataset)), NUM_PROMPTS)
selected_samples = test_dataset.select(selected_indices)

# Créer les prompts: sélectionner aléatoirement 2-8 premiers tokens
prompts = []
for sample in selected_samples:
    text = sample["text"]
    tokens = ref_tokenizer.encode(text, add_special_tokens=False)
    
    # Sélectionner aléatoirement entre 2 et 8 tokens
    num_tokens = random.randint(2, 8)
    prompt_tokens = tokens[:num_tokens]
    
    # Décoder pour obtenir le prompt
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
    """Génère plusieurs complétions pour un prompt donné."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Générer plusieurs complétions
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
    # Extraire seulement la complétion (sans le prompt)
    completions = [t[len(prompt):].strip() for t in texts]
    return completions


def sentiment_reward(texts):
    """Calcule la récompense moyenne (probabilité POSITIVE)."""
    preds = sentiment_pipe(texts, truncation=True, max_length=512)
    # Probabilité du label POSITIVE
    probs = [p["score"] if p["label"].upper() == "POSITIVE" else 1 - p["score"] for p in preds]
    return np.mean(probs)


@torch.no_grad()
def logprob_of_completion(model, tokenizer, prompt, completion):
    """Calcule la log-probabilité d'une complétion donnée."""
    # Concat prompt + completion
    full_text = prompt + completion
    enc = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    # Logits
    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [1, T, V]
    log_probs = torch.log_softmax(logits, dim=-1)

    # On veut la log-prob des tokens à partir de la position prompt_len -> fin
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # Cible = tokens décalés de 1
    target_ids = input_ids[:, 1:]                          # [1, T-1]
    log_probs = log_probs[:, :-1, :]                       # aligner
    # positions correspondant à la completion
    mask = torch.zeros_like(target_ids)
    mask[:, prompt_len-1:] = 1                             # prompt_len-1 car shift
    # gather log-probs
    token_logp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    comp_logp = (token_logp * mask).sum() / mask.sum().clamp(min=1.0)       # moyenne par token
    return comp_logp.item()


def evaluate_model(policy_path):
    """Évalue un modèle sur reward et KL."""
    print(f"\n📊 Évaluation: {policy_path}")
    
    # Load policy
    policy_tok = AutoTokenizer.from_pretrained(policy_path)
    if policy_tok.pad_token is None:
        policy_tok.pad_token = policy_tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(policy_path).to(DEVICE).eval()

    rewards = []
    kls = []

    for prompt in prompts:
        # Générer plusieurs complétions
        comps = generate_completions(policy, policy_tok, prompt, NUM_COMPLETIONS)

        # Reward: moyenne proba positive
        r = sentiment_reward([prompt + " " + c for c in comps])
        rewards.append(r)

        # KL: moyenne sur les completions de logp_policy - logp_ref
        kl_samples = []
        for c in comps:
            lp_pol = logprob_of_completion(policy, policy_tok, prompt, c)
            lp_ref = logprob_of_completion(ref_model, ref_tokenizer, prompt, c)
            kl_samples.append(lp_pol - lp_ref)
        kls.append(np.mean(kl_samples))

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "kl_mean": float(np.mean(kls)),
        "kl_std": float(np.std(kls)),
    }

# ============================================================================
# STEP 4. EVALUATE ALL MODELS
# ============================================================================

print("\n" + "="*80)
print("ÉVALUATION DES MODÈLES DPO")
print("="*80)

results = []
for path in DPO_MODEL_PATHS:
    try:
        res = evaluate_model(path)
        res["model"] = path.split("/")[-1]
        results.append(res)
        print(f"✅ {path}:")
        print(f"   Reward: {res['reward_mean']:.4f} ± {res['reward_std']:.4f}")
        print(f"   KL: {res['kl_mean']:.4f} ± {res['kl_std']:.4f}")
    except Exception as e:
        print(f"❌ Skip {path}: {e}")

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
    plt.title("Reward vs KL - DPO Models\n(50 prompts, 2-8 tokens)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder
    save_path = f"{SAVE_BASE_PATH}/reward_kl_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {save_path}")
    
    plt.show()
else:
    print("❌ Aucun résultat à tracer (aucun checkpoint chargé).")

print("\n" + "="*80)
print("✅ ÉVALUATION TERMINÉE")
print("="*80)
