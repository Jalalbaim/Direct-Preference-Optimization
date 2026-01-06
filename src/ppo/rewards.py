# src/ppo/rewards.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_sentiment_classifier(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
    clf.to(device)
    clf.eval()
    return tok, clf

@torch.no_grad()
def sentiment_reward(texts, clf_tokenizer, clf_model, device: str):
    enc = clf_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    out = clf_model(**enc)
    probs = torch.softmax(out.logits, dim=-1)
    return probs[:, 1].detach()  # proba "positive"
