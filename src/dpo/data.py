import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str   # y_w
    rejected: str # y_l


class PreferenceDataset(Dataset):
    """
    Dataset de paires (prompt, chosen, rejected) à partir d'un fichier JSONL.
    Chaque ligne doit contenir au moins:
      { "prompt": "...", "chosen": "...", "rejected": "..." }
    """

    def __init__(self, path: str):
        self.examples: List[PreferenceExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.examples.append(
                    PreferenceExample(
                        prompt=obj["prompt"],
                        chosen=obj["chosen"],
                        rejected=obj["rejected"],
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self.examples[idx]


def preference_collate_fn(
    batch: List[PreferenceExample],
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_length: int,
    max_response_length: int,
) -> Dict[str, Any]:
    """
    Collate pour transformer une liste d'exemples en batch tensors.

    On construit:
      - input_ids_chosen, attention_mask_chosen, response_mask_chosen
      - input_ids_rejected, attention_mask_rejected, response_mask_rejected
    """

    prompts = [ex.prompt for ex in batch]
    chosens = [ex.chosen for ex in batch]
    rejecteds = [ex.rejected for ex in batch]

    # On concatène prompt + réponse. On crée le mask qui marque la réponse.
    chosen_texts = [
        prompt + tokenizer.eos_token + chosen for prompt, chosen in zip(prompts, chosens)
    ]
    rejected_texts = [
        prompt + tokenizer.eos_token + rejected for prompt, rejected in zip(prompts, rejecteds)
    ]

    # Tokenisation
    chosen_enc = tokenizer(
        chosen_texts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length + max_response_length,
        return_tensors="pt",
    )
    rejected_enc = tokenizer(
        rejected_texts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length + max_response_length,
        return_tensors="pt",
    )

    # Pour le mask de réponse: on recalcule la longueur du prompt + eos
    # pour approx marquer la zone réponse.
    # Ici simplifié: on suppose que prompt+eos <= max_prompt_length.
    prompt_enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    )
    prompt_lengths = prompt_enc["attention_mask"].sum(dim=1)  # [B]

    def build_response_mask(input_ids: torch.Tensor, prompt_lengths: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        resp_mask = torch.zeros_like(input_ids)
        for i in range(B):
            start = int(prompt_lengths[i].item())
            # réponse = tokens à partir de start
            resp_mask[i, start:] = 1
        return resp_mask

    response_mask_chosen = build_response_mask(
        chosen_enc["input_ids"], prompt_lengths
    )
    response_mask_rejected = build_response_mask(
        rejected_enc["input_ids"], prompt_lengths
    )

    batch_dict = {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "chosen_response_mask": response_mask_chosen,
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
        "rejected_response_mask": response_mask_rejected,
    }
    return batch_dict
