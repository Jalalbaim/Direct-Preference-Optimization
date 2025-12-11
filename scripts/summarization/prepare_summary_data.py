#IMPORT DATA -----------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.dpo.models import load_models, compute_logprobs
from src.dpo.utils import load_yaml_config

from datasets import load_dataset

#LOAD DATA -----------------
ds = load_dataset("openai/summarize_from_feedback", "comparisons")


#MAIN FUNCTIONS -----------------
def load_summary_classifier(model_name: str, device: str):
    clf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    clf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,  
    )
    clf_model.to(device)
    clf_model.eval()
    return clf_tokenizer, clf_model


def format_dpo(ds):
    return ds.map(lambda x: {
        "prompt": x["info"]["post"],
        "chosen": x["summaries"]["chosen"],
        "rejected": x["summaries"]["rejected"],
    })


@torch.no_grad()
def summary_score(texts, clf_tokenizer, clf_model, device: str):

    text_final = texts.prompt + "\n\nSummary:" + texts.answer

    enc = clf_tokenizer(
        text_final,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    outputs = clf_model(**enc)
    logits = outputs.logits

    #Transformer en proba avec le softmax
    log_probs = torch.log_softmax(logits, dim=-1)
    last_token_log_probs = log_probs[:, -1, :]
    scores = torch.max(last_token_log_probs, dim=-1).values  # Example: max log-prob of last token
    return scores.cpu()

