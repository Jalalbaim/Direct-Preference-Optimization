#python3 scripts/summarization/prepare_summary_data.py

#IMPORT DATA -----------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
from datasets import load_dataset
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.dpo.utils import load_yaml_config

#MAIN FUNCTIONS -----------------
def format_dpo(dataset):
    return dataset.map(
        lambda x: {
            "prompt": x["info"]["post"].strip(),
            "chosen": x["summaries"][x["choice"]]["text"].strip(),
            "rejected": x["summaries"][1 - x["choice"]]["text"].strip(),
        },
        remove_columns=dataset.column_names,
    )


def main():
    config_path = "configs/summary.yaml"
    config = load_yaml_config(config_path)

    #LOAD DATASET WITH HUMAN PREFERENCES -----------------
    ds = load_dataset("openai/summarize_from_feedback", "comparisons")

    #FORMAT DATASET -----------------
    prompt_nb = config["data"]["prompt_nb"]

    rng = random.Random(config["training"]["seed"])

    train_indices = rng.sample(range(len(ds["train"])), prompt_nb)
    val_indices = rng.sample(range(len(ds["validation"])), prompt_nb)

    ds_train = format_dpo(ds["train"].select(train_indices))
    ds_val = format_dpo(ds["validation"].select(val_indices))


    #SAVE DATASET -----------------
    os.makedirs("data/processed/summarization", exist_ok=True)
    ds_train.to_json("data/processed/summarization/train_summarization.jsonl")
    ds_val.to_json("data/processed/summarization/val_summarization.jsonl")

if __name__ == "__main__":
    main()



    