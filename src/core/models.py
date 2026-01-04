from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    policy_model: AutoModelForCausalLM
    ref_model: AutoModelForCausalLM
    device: torch.device


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pour les modèles decoder-only, utiliser left padding
    tokenizer.padding_side = 'left'
    # S'assurer d'avoir un pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_models(model_name: str, dtype: str = "bfloat16", device: str = None) -> ModelBundle:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.bfloat16)

    tokenizer = load_tokenizer(model_name)

    # Pour Google Colab GPU, ne pas utiliser device_map="auto"
    # Charger sur CPU puis déplacer vers le device approprié
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    policy_model = policy_model.to(device)
    
    # Activer gradient checkpointing pour économiser la mémoire
    if hasattr(policy_model, 'gradient_checkpointing_enable'):
        policy_model.gradient_checkpointing_enable()

    # Référence = copie gelée du policy initial
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    ref_model = ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    return ModelBundle(
        tokenizer=tokenizer,
        policy_model=policy_model,
        ref_model=ref_model,
        device=device,
    )


def compute_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calcule le log-prob total des tokens de réponse, pour chaque exemple.

    - input_ids : [B, L]
    - attention_mask : [B, L]
    - response_mask : [B, L] booléen ou 0/1 (1 = token de la réponse)

    Retour : [B] (log-prob sommé sur les tokens de réponse)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    # logits: [B, L, V]
    logits = outputs.logits

    # décaler pour aligner avec les cibles
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_resp_mask = response_mask[:, 1:].contiguous()
    shift_attn_mask = attention_mask[:, 1:].contiguous()

    # shape [B, L-1]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    # gather sur les labels cibles
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # on ignore les tokens qui ne sont pas dans la réponse ou padding
    mask = (shift_resp_mask > 0) & (shift_attn_mask > 0)
    token_logps = token_logps * mask

    # somme sur les positions de réponse
    seq_logps = token_logps.sum(dim=-1)  # [B]

    return seq_logps
