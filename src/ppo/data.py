# src/ppo/data.py
import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset

@dataclass
class PromptExample:
    prompt: str

class PromptDataset(Dataset):
    def __init__(self, path: str):
        self.examples: List[PromptExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.examples.append(PromptExample(prompt=obj["prompt"]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def prompt_collate_fn(batch):
    return [ex.prompt for ex in batch]
