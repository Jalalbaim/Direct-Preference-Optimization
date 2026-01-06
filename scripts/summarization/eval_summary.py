# IMPORT LIBRARIES -----------------
import os
import sys
import argparse
import json

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate.utils import set_module_tensor_to_device

# PATH SETUP -----------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.dpo.models import load_models
from src.dpo.utils import load_yaml_config


# FUNCTIONS -----------------
@torch.no_grad()
def generate_summary(
    model,
    tokenizer,
    prompts,
    max_new_tokens=32,
    temperature=0.8,
    top_p=0.9,
    device="cuda",
):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    ).to(device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    responses = []
    full_ids = []

    for i in range(len(prompts)):
        prompt_len = int(enc["attention_mask"][i].sum())
        gen_ids = out[i][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        responses.append(text)
        full_ids.append(out[i])

    return responses, full_ids


@torch.no_grad()
def judge_pairwise(
    judge_model,
    judge_tokenizer,
    formatted_prompt,
    device,
):
    inputs = judge_tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    outputs = judge_model.generate(
        **inputs,
        max_new_tokens=2,
        do_sample=False,
        temperature=0.0,
        eos_token_id=judge_tokenizer.eos_token_id,
        pad_token_id=judge_tokenizer.eos_token_id,
    )

    gen = outputs[0][inputs["input_ids"].shape[1]:]
    text = judge_tokenizer.decode(gen, skip_special_tokens=True).strip().upper()

    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    return None


@torch.no_grad()
def compute_kl_divergence(policy_model, ref_model, input_ids, attention_mask, device):
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    policy_outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)

    policy_logits = policy_outputs.logits[:, :-1, :]
    ref_logits = ref_outputs.logits[:, :-1, :]
    shift_attn = attention_mask[:, 1:]

    policy_probs = torch.softmax(policy_logits, dim=-1)
    policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

    kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
    kl = (kl_per_token * shift_attn).sum().item()

    return kl


@torch.no_grad()
def evaluate_real_win_rate(
    judge_model,
    judge_tokenizer,
    summaries_a,
    summaries_b,
    originals,
    prompt_template,
    device,
    save_path=None,
):
    wins = losses = abstain = 0
    total = len(summaries_a)

    f_out = open(save_path, "w", encoding="utf-8") if save_path else None

    for sa, sb, txt in tqdm(zip(summaries_a, summaries_b, originals), total=total):
        prompt = (
            prompt_template
            .replace("{post}", txt)
            .replace("{summary_a}", sa)
            .replace("{summary_b}", sb)
        )

        choice = judge_pairwise(judge_model, judge_tokenizer, prompt, device)

        if choice == "A":
            wins += 1
        elif choice == "B":
            losses += 1
        else:
            abstain += 1

        if f_out:
            json.dump(
                {
                    "prompt": txt,
                    "summary_dpo": sa,
                    "summary_ref": sb,
                    "choice": choice,
                },
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")

    if f_out:
        f_out.close()

    return wins, losses, abstain, total


# MAIN -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/summary.yaml")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dpo_checkpoint", default="checkpoints/summary_dpo/policy_epoch_1.pt")
    parser.add_argument("--save_judge_outputs", default="judge_outputs.jsonl")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    mb = load_models(config["model"]["name"], dtype=config["model"]["dtype"])
    tokenizer = mb.tokenizer
    ref_model = mb.ref_model.to(device).eval()
    policy_model = mb.policy_model.to(device).eval()

    if os.path.exists(args.dpo_checkpoint):
        ckpt = torch.load(args.dpo_checkpoint, map_location="cpu")
        for n, t in ckpt["model_state_dict"].items():
            try:
                set_module_tensor_to_device(policy_model, n, device=0, value=t)
            except Exception:
                pass
        print("✅ Loaded DPO checkpoint")

    # Dataset
    ds = load_dataset("CarperAI/openai_summarize_tldr")["test"]
    ds = ds.shuffle(seed=789).select(range(config["testing"]["prompt_nb"]))

    # Judge
    judge_name = "Qwen/Qwen2.5-3B-Instruct"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_name)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()

    summaries_a, summaries_b, originals, kls = [], [], [], []

    for ex in tqdm(ds):
        prompt = ex["prompt"].strip()
        if not prompt:
            continue

        ref, _ = generate_summary(
            ref_model, tokenizer, [prompt],
            args.max_new_tokens, args.temperature, args.top_p, device
        )

        dpo, dpo_ids = generate_summary(
            policy_model, tokenizer, [prompt],
            args.max_new_tokens, args.temperature, args.top_p, device
        )

        summaries_a.append(dpo[0])
        summaries_b.append(ref[0])
        originals.append(prompt)

        attn = (dpo_ids[0] != tokenizer.pad_token_id).long()
        kls.append(compute_kl_divergence(policy_model, ref_model, dpo_ids[0], attn, device))

    wins, losses, abstain, total = evaluate_real_win_rate(
        judge_model,
        judge_tokenizer,
        summaries_a,
        summaries_b,
        originals,
        config["dpo"]["prompt"],
        device,
        args.save_judge_outputs,
    )

    real_win_rate = wins / total
    conditional_win_rate = wins / (wins + losses) if (wins + losses) else 0.0
    decision_rate = (wins + losses) / total

    print("\n==== DPO SUMMARY EVAL ====")
    print(f"Real win rate:        {real_win_rate:.3f}")
    print(f"Conditional win rate:{conditional_win_rate:.3f}")
    print(f"Judge decision rate: {decision_rate:.3f}")
    print(f"Abstentions:         {abstain}/{total}")
    print(f"Avg KL:              {sum(kls)/len(kls):.4f}")

    with open(args.save_judge_outputs, "a", encoding="utf-8") as f:
        json.dump(
            {
                "type": "summary",
                "real_win_rate": real_win_rate,
                "conditional_win_rate": conditional_win_rate,
                "avg_kl_policy_ref": sum(kls)/len(kls),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "dpo_checkpoint": args.dpo_checkpoint,
                "judge_model": judge_name,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")


if __name__ == "__main__":
    main()