# src/ppo/utils.py
from dataclasses import dataclass
import torch
import torch.nn as nn

from src.dpo.models import compute_logprobs  # on réutilise ton code existant

def generate_with_model(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, device: str):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    prompt_len = input_ids.shape[1]

    if temperature == 0:
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_ids = out  # [1, L]
    gen_ids = out[:, prompt_len:]
    response = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    return response, full_ids, prompt_len

def response_logprob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_len: int):
    # response_mask: 0 sur prompt, 1 sur réponse
    resp_mask = torch.zeros_like(input_ids)
    resp_mask[:, prompt_len:] = 1
    return compute_logprobs(model, input_ids, attention_mask, resp_mask)  # [B]

class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)

def value_forward(policy_model, value_head: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    out = policy_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states[-1]  # [B, L, H]
    idx = attention_mask.sum(dim=1) - 1  # last token index
    b = torch.arange(hs.size(0), device=hs.device)
    last_h = hs[b, idx]  # [B, H]
    return value_head(last_h)  # [B]

def save_ppo_checkpoint(path: str, policy_model, value_head, policy_optim, value_optim):
    torch.save(
        {
            "model_state_dict": policy_model.state_dict(),       # compatible eval_sentiment.py
            "value_state_dict": value_head.state_dict(),
            "policy_optimizer_state_dict": policy_optim.state_dict(),
            "value_optimizer_state_dict": value_optim.state_dict(),
        },
        path,
    )
