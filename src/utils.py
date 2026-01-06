"""Utility functions for sentiment generation experiments."""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Sentiment classifier loader
def load_sentiment_classifier(model_name: str = "siebert/sentiment-roberta-large-english", device: Optional[str] = None):
    """Load a sentiment classifier with device awareness (CUDA/MPS/CPU)."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inference_device = "cpu" if device == "mps" else device
    model = model.to(inference_device)
    model.eval()
    return model, tokenizer

def get_device() -> str:
    """Auto-detect the best available device (NVIDIA GPU, Mac MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_dtype(dtype_str: str = "float32"):
    """Get torch dtype from string."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float32

def get_sentiment_score(texts: List[str], model, tokenizer, device: str = "cuda") -> np.ndarray:
    """Get sentiment scores for texts using classifier.
    
    Supports NVIDIA GPU (cuda), Mac MPS, and CPU devices.
    """
    # MPS doesn't support all operations, fall back to CPU for inference
    inference_device = "cpu" if device == "mps" else device
    model = model.to(inference_device)
    
    scores = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(inference_device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Get probability of positive class (class 1)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            positive_prob = probs[0, 1].item()
            scores.append(positive_prob)
    
    return np.array(scores)

def calculate_kl_divergence(
    policy_texts: List[str],
    reference_texts: List[str],
    policy_model,
    policy_tokenizer,
    ref_model,
    ref_tokenizer,
    device: str = "cuda"
) -> float:
    """Calculate KL divergence between policy and reference using log probabilities.
    
    Supports NVIDIA GPU (cuda), Mac MPS, and CPU devices.
    """
    # Use CPU for MPS as it doesn't support all operations
    inference_device = "cpu" if device == "mps" else device
    
    policy_model = policy_model.to(inference_device)
    ref_model = ref_model.to(inference_device)
    
    kl_divs = []
    
    policy_model.eval()
    ref_model.eval()
    
    with torch.no_grad():
        for policy_text, ref_text in zip(policy_texts, reference_texts):
            # Tokenize
            policy_tokens = policy_tokenizer(policy_text, return_tensors="pt")
            ref_tokens = ref_tokenizer(ref_text, return_tensors="pt")
            
            # Ensure same length
            min_len = min(policy_tokens['input_ids'].shape[1], ref_tokens['input_ids'].shape[1])
            policy_tokens['input_ids'] = policy_tokens['input_ids'][:, :min_len]
            ref_tokens['input_ids'] = ref_tokens['input_ids'][:, :min_len]
            
            policy_tokens = {k: v.to(inference_device) for k, v in policy_tokens.items()}
            ref_tokens = {k: v.to(inference_device) for k, v in ref_tokens.items()}
            
            # Get logits
            policy_outputs = policy_model(**policy_tokens, output_hidden_states=False)
            ref_outputs = ref_model(**ref_tokens, output_hidden_states=False)
            
            # Calculate log probabilities
            policy_logps = F.log_softmax(policy_outputs.logits, dim=-1)
            ref_logps = F.log_softmax(ref_outputs.logits, dim=-1)
            
            # Get KL for each token
            token_kl = torch.exp(ref_logps) * (ref_logps - policy_logps)
            seq_kl = token_kl.sum().item()
            kl_divs.append(seq_kl)
    
    return float(np.mean(kl_divs))

def extract_prompts_from_imdb(examples: List[Dict], max_prompts: int = 500) -> List[str]:
    """Extract prompts (prefixes) from IMDb reviews."""
    prompts = []
    for example in examples:
        text = example['text']
        words = text.split()
        # Take first 10-20 words as prompt
        prompt_length = min(np.random.randint(10, 20), len(words))
        prompt = ' '.join(words[:prompt_length])
        prompts.append(prompt)
        
        if len(prompts) >= max_prompts:
            break
    
    return prompts[:max_prompts]

def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    batch_size: int = 32,
    device: str = "cuda"
) -> List[str]:
    """Generate continuations for prompts.
    
    Supports NVIDIA GPU (cuda), Mac MPS, and CPU devices.
    """
    model = model.to(device)
    model.eval()
    
    all_generations = []
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_generations.extend(generations)
    
    return all_generations
