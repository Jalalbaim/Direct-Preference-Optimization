# IMPORT LIBRARIES -----------------
import os
import sys
import argparse
import json
import re

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
    max_new_tokens=64,
    temperature=0.8,
    top_p=0.9,
    device="cuda",
):
    responses = []
    full_ids_list = []

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
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

    for i in range(len(prompts)):
        prompt_len = int(enc["attention_mask"][i].sum())
        full_ids = out[i]
        response_ids = full_ids[prompt_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        responses.append(response.strip())
        full_ids_list.append(full_ids)

    return responses, full_ids_list


@torch.no_grad()
def judge_pairwise_chat(
    judge_model,
    judge_tokenizer,
    formatted_prompt,
    device,
    max_new_tokens=128,
):
    messages = [{"role": "user", "content": formatted_prompt}]

    prompt = judge_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = judge_tokenizer(prompt, return_tensors="pt").to(device)

    outputs = judge_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=judge_tokenizer.eos_token_id,
        pad_token_id=judge_tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = judge_tokenizer.decode(generated, skip_special_tokens=True).strip()

    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[:2])


@torch.no_grad()
def generate_win_rate(
    judge_model,
    judge_tokenizer,
    summaries_a,
    summaries_b,
    original_texts,
    prompt_template,
    device,
    save_path=None,
):
    win_rate_a = []
    win_rate_b = []

    if save_path:
        f_out = open(save_path, "w", encoding="utf-8")

    for sa, sb, txt in tqdm(
        zip(summaries_a, summaries_b, original_texts),
        total=len(summaries_a),
    ):
        formatted_prompt = (
            prompt_template
            .replace("{post}", txt)
            .replace("{summary_a}", sa)
            .replace("{summary_b}", sb)
        )

        content = judge_pairwise_chat(
            judge_model,
            judge_tokenizer,
            formatted_prompt,
            device,
        )

        print("\n==== CHECKPOINT JUDGE OUTPUT ====")
        print(content)
        print("==== END ====")

        match = re.search(
            r"^Preferred:\s*([AB])$",
            content.splitlines()[-1],
            re.IGNORECASE,
        )
        choice = match.group(1).upper() if match else "None"

        if choice == "A":
            win_rate_a.append(1)
            win_rate_b.append(0)
        elif choice == "B":
            win_rate_a.append(0)
            win_rate_b.append(1)
        else:
            win_rate_a.append(0)
            win_rate_b.append(0)

        if save_path:
            json.dump(
                {
                    "type": "example",
                    "original_prompt": txt,
                    "summary_dpo": sa,
                    "summary_ref": sb,
                    "formatted_prompt": formatted_prompt,
                    "judge_output": content,
                    "choice": choice,
                },
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")

    if save_path:
        f_out.close()

    return win_rate_a, win_rate_b


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


# MAIN -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/summary.yaml")
    parser.add_argument("--max_prompt_chars", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--ref_model_name", type=str, default=None)
    parser.add_argument("--dpo_checkpoint", type=str, default="checkpoints/summary_dpo/policy_epoch_1.pt")
    parser.add_argument("--save_judge_outputs", type=str, default="judge_outputs.jsonl")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DPO models
    ref_model_name = args.ref_model_name or config["model"]["name"]
    mb = load_models(ref_model_name, dtype=config["model"]["dtype"])
    tokenizer = mb.tokenizer
    ref_model = mb.ref_model
    policy_model = mb.policy_model

    if args.dpo_checkpoint and os.path.exists(args.dpo_checkpoint):
        ckpt = torch.load(args.dpo_checkpoint, map_location="cpu")
        loaded = 0
        for name, tensor in ckpt["model_state_dict"].items():
            try:
                set_module_tensor_to_device(policy_model, name, device=0, value=tensor)
                loaded += 1
            except Exception:
                pass
        print(f"✅ Loaded {loaded} parameters from DPO checkpoint")

    ref_model.to(device).eval()
    policy_model.to(device).eval()

    # Dataset
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    test_ds = (
        dataset["test"]
        .shuffle(seed=789)
        .select(range(config["testing"]["prompt_nb"]))
    )


    # Judge model
    judge_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_name)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()

    summaries_a, summaries_b, originals, kls = [], [], [], []

    for ex in tqdm(test_ds):
        prompt = ex["prompt"][: args.max_prompt_chars].strip()
        if not prompt:
            continue

        ref_resp, _ = generate_summary(
            ref_model, tokenizer, [prompt],
            args.max_new_tokens, args.temperature, args.top_p, device
        )

        dpo_resp, dpo_ids = generate_summary(
            policy_model, tokenizer, [prompt],
            args.max_new_tokens, args.temperature, args.top_p, device
        )

        summaries_a.append(dpo_resp[0])
        summaries_b.append(ref_resp[0])
        originals.append(prompt)

        attn = (dpo_ids[0] != tokenizer.pad_token_id).long()
        kl = compute_kl_divergence(policy_model, ref_model, dpo_ids[0], attn, device)
        kls.append(kl)

    win_a, _ = generate_win_rate(
        judge_model,
        judge_tokenizer,
        summaries_a,
        summaries_b,
        originals,
        prompt_template=config["dpo"]["prompt"],
        device=device,
        save_path=args.save_judge_outputs,
    )

    win_rate = sum(win_a) / len(win_a)
    avg_kl = sum(kls) / len(kls)

    print("\n==== DPO SUMMARY EVAL ====")
    print(f"Win rate (DPO > Ref): {win_rate:.3f}")
    print(f"Avg KL(policy || ref): {avg_kl:.4f}")

    # Append final metrics
    with open(args.save_judge_outputs, "a", encoding="utf-8") as f:
        json.dump(
            {
                "type": "summary",
                "win_rate_dpo_over_ref": win_rate,
                "avg_kl_policy_ref": avg_kl,
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

    print(f"Judge outputs saved to: {args.save_judge_outputs}")


if __name__ == "__main__":
    main()
