#python3 scripts/summarization/prepare_summary_data.py

#IMPORT DATA -----------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
from datasets import load_dataset
import openai 

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

def generate_win_rate(
    summaries: List[str],
    original_texts: List[str],
    summary_model: str,
    device: str,
    prompt: str,
    temperature:float):
    """
    This function computes the win rate of the chosen summaries over the rejected ones using GPT-4
    """

    win_rate_a = []
    win_rate_b = []

    for summary_a, summary_b, original in zip(summaries_a, summaries_b, original_texts):
        #Prompting GPT-4
        response = openai.ChatCompletion.create(
            model="gtp-5-0314",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )

        choice = response.choices[0].message.content.strip().split("\n")[-1].strip() #based on https://platform.openai.com/docs/api-reference/chat/get

        if choice == "A":
                win_rate_a.append(1)
                win_rate_b.append(0)
        elif choice == "B":
            win_rate_a.append(0)
            win_rate_b.append(1)
        else: #No one wins or the answer is not valid
            win_rate_a.append(0)
            win_rate_b.append(0)



def main():
    config_path = "configs/summary.yaml"
    config = load_yaml_config(config_path)

    #LOAD DATASET WITH HUMAN PREFERENCES -----------------
    ds = load_dataset("openai/summarize_from_feedback", "comparisons")

    #FORMAT DATASET -----------------
    ds = format_dpo(ds["train"].select(range(config["data"]["prompt_nb"])))

    #SAVE DATASET -----------------
    os.makedirs("data/processed/summarization", exist_ok=True)
    ds.to_json("data/processed/summarization/summarization_dpo_data.jsonl")

if __name__ == "__main__":
    main()



    