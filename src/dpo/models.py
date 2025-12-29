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
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2") #Se base sur le config.json du modèle utilisé pour summarization

    print("DEBUG PRINT")
    # Gemma est en causal LM, on s'assure d'avoir un pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_models(model_name: str, dtype: str = "bfloat16") -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.bfloat16)

    tokenizer = load_tokenizer(model_name)

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    
    # Activer gradient checkpointing pour économiser la mémoire
    if hasattr(policy_model, 'gradient_checkpointing_enable'):
        policy_model.gradient_checkpointing_enable()

    # Référence = copie gelée du policy initial
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
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
