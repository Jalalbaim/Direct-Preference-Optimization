#IMPORT LIBRARIES -----------------

import os
import sys
import argparse

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from accelerate.utils import set_module_tensor_to_device

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
from google.colab import userdata #For Colab 


from src.dpo.models import load_models, compute_logprobs
from src.dpo.utils import load_yaml_config



#FUNCTIONS -----------------
def generate_summary(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8, top_p=0.9, device="cuda"):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    prompt_len = input_ids.shape[1]

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_ids = out[0]
    response_ids = full_ids[prompt_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response, full_ids

def generate_win_rate(
    client,
    summaries_a: list[str],
    summaries_b: list[str],
    original_texts: list[str],
    prompt_template: str, 
    temperature: float):

    win_rate_a = []
    win_rate_b = []

    for sa, sb, txt in tqdm(zip(summaries_a, summaries_b, original_texts), total=len(summaries_a)):
        formatted_prompt = prompt_template.replace("<post>", txt)\
                                           .replace("<Summary A>", sa)\
                                           .replace("<Summary B>", sb)

        response = client.chat.completions.create(
            model="gpt-4.1-mini", 
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=temperature,
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()
        
        lines = content.split("\n")
        choice = "None"
        for line in lines:
            if "Preferred:" in line:
                choice = line.replace("Preferred:", "").strip().upper()
                break

        if "A" in choice:
            win_rate_a.append(1)
            win_rate_b.append(0)
        elif "B" in choice:
            win_rate_a.append(0)
            win_rate_b.append(1)
        else:
            print(f"Warning: GPT-4 gave an unclear answer: {content}")
            win_rate_a.append(0)
            win_rate_b.append(0)

    return win_rate_a, win_rate_b


def sequence_logprob(model, input_ids, attention_mask, device: str):
    # calcule logp(seq) (somme sur tokens) sous un modèle causal
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [1, L, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_attn = attention_mask[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * shift_attn  # ignore padding
    return token_logps.sum().item()




#MAIN FUNCTION -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/summary.yaml")
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--max_prompt_chars", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--ref_model_name", type=str, default=None,
                        help="HF name of the reference model (if None, use config.model.name)")
    parser.add_argument("--dpo_checkpoint", type=str,
                        default="checkpoints/summary_dpo/policy_epoch_1.pt")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ref model & policy model DPO
    ref_model_name = args.ref_model_name or config["model"]["name"]
    print(f"Loading reference + policy models from: {ref_model_name}")

    mb = load_models(ref_model_name, dtype=config["model"]["dtype"])
    tokenizer = mb.tokenizer
    ref_model = mb.ref_model
    policy_model = mb.policy_model

    # checkpoint DPO
    if args.dpo_checkpoint and os.path.exists(args.dpo_checkpoint):
        ckpt = torch.load(args.dpo_checkpoint, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        
        for name, tensor in state_dict.items():
            try:
                set_module_tensor_to_device(policy_model, name, device=0, value=tensor)
            except Exception as e:
                print(f"Skipping key {name} due to: {e}")
                
        print(f"Successfully loaded DPO checkpoint: {args.dpo_checkpoint}")
    else:
        print("WARNING: DPO checkpoint not found, using base model as policy.")

    ref_model.to(device)
    policy_model.to(device)
    ref_model.eval()
    policy_model.eval()
    

    # CarperAI dataset
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    test_ds = dataset["test"].select(range(5))

    # Evaluation
    kls = []

    # OPEN AI KEY
    try:
        #api_key = userdata.get('OPENAI_API_KEY')
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Erreur réelle : {e}")
        print("Error: Could not find OPENAI_API_KEY in Colab Secrets.")
        return

    # Summary data
    summaries_a = []
    summaries_b = []
    original = []

    n = min(args.num_examples, len(test_ds))
    print(f"Evaluating on {n} summary test examples")

    for i in tqdm(range(n)):
        text = test_ds[i]["prompt"]
        prompt = text[: args.max_prompt_chars].strip()
        if not prompt:
            continue

        # ref
        resp_ref, full_ids_ref = generate_summary(
            ref_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        # DPO
        resp_dpo, full_ids_dpo = generate_summary(
            policy_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        summaries_a.append(resp_dpo)
        summaries_b.append(resp_ref)
        original.append(text)

        # KL approx: logp_dpo(seq_dpo) - logp_ref(seq_dpo)
        attn_dpo = (full_ids_dpo != tokenizer.pad_token_id).long()
        logp_dpo = sequence_logprob(policy_model, full_ids_dpo, attn_dpo, device)
        logp_ref = sequence_logprob(ref_model, full_ids_dpo, attn_dpo, device)
        kl_point = (logp_dpo - logp_ref)  # approx, en nats
        kls.append(kl_point)

    import numpy as np
    win_rate_a, win_rate_b = generate_win_rate(
        client, summaries_a, summaries_b, original, prompt_template=config["dpo"]["prompt"], temperature=0.7
    )

    #avg_r_ref = float(np.mean(rewards_ref))
    #avg_r_dpo = float(np.mean(rewards_dpo))
    gpt4_win_rate = sum(win_rate_a) / len(win_rate_a) if win_rate_a else 0.0
    avg_kl = float(np.mean(kls)) if kls else 0.0

    print("==== Sentiment DPO Evaluation ====")
    #print(f"Avg reward (ref): {avg_r_ref:.4f}")
    #print(f"Avg reward (DPO): {avg_r_dpo:.4f}")
    print(f"GPT win-rate (DPO > ref): {gpt4_win_rate:.3f}")
    print(f"Approx KL (DPO || ref): {avg_kl:.4f}")


if __name__ == "__main__":
    main()
