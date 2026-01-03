# IMPORT LIBRARIES -----------------
import os
import sys
import argparse
import json

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate.utils import set_module_tensor_to_device

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

        responses.append(response)
        full_ids_list.append(full_ids)

    return responses, full_ids_list

@torch.no_grad()
def generate_win_rate(
    chat_pipeline,
    summaries_a,
    summaries_b,
    original_texts,
    prompt_template,
    temperature,
    save_path=None,
):
    win_rate_a = []
    win_rate_b = []

    # Prepare file for saving outputs if needed
    if save_path:
        f_out = open(save_path, "w", encoding="utf-8")

    for sa, sb, txt in tqdm(
        zip(summaries_a, summaries_b, original_texts),
        total=len(summaries_a),
    ):
        formatted_prompt = (
            prompt_template.replace("<post>", txt)
            .replace("<Summary A>", sa)
            .replace("<Summary B>", sb)
        )

        response = chat_pipeline(formatted_prompt)
        content = response[0]["generated_text"].strip()

        print("CHECKPOINT JUDGE OUTPUT")
        print("Prompt + Summaries:")
        print(formatted_prompt)
        print("----")
        print("Judge response:")
        print(content)
        print("==== End of Judge Output ====")

        content = response[0]["generated_text"].strip().upper()
        if content.startswith("A"):
            choice = "A"
        elif content.startswith("B"):
            choice = "B"
        else:
            choice = "None"  # or skip this example


        print("ANALYSIS OF JUDGE OUTPUT = %s \n" % choice)

        if "A" in choice:
            win_rate_a.append(1)
            win_rate_b.append(0)
        elif "B" in choice:
            win_rate_a.append(0)
            win_rate_b.append(1)
        else:
            win_rate_a.append(0)
            win_rate_b.append(0)

        # Save to file
        if save_path:
            json.dump({
                "original_prompt": txt,
                "summary_dpo": sa,
                "summary_ref": sb,
                "formatted_prompt": formatted_prompt,
                "judge_output": content,
                "choice": choice,
            }, f_out, ensure_ascii=False)
            f_out.write("\n")

    if save_path:
        f_out.close()

    return win_rate_a, win_rate_b

@torch.no_grad()
def compute_kl_divergence(policy_model, ref_model, input_ids, attention_mask, device):
    """
    True KL(policy || ref) summed over sequence tokens
    """
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

# MAIN FUNCTION -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/summary.yaml")
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--max_prompt_chars", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
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
        for name, tensor in ckpt["model_state_dict"].items():
            try:
                set_module_tensor_to_device(policy_model, name, device=0, value=tensor)
            except Exception:
                pass
        print("Loaded DPO checkpoint")

    ref_model.to(device).eval()
    policy_model.to(device).eval()

    # Dataset
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    test_size = config["testing"]["prompt_nb"]
    test_ds = dataset["test"].select(range(test_size))

    # Judge model (TinyLlama)
    judge_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_name)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    chat_pipeline = pipeline(
        "text-generation",
        model=judge_model,
        tokenizer=judge_tokenizer,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    summaries_a, summaries_b, originals, kls = [], [], [], []

    for ex in tqdm(test_ds):
        prompt = ex["prompt"][: args.max_prompt_chars].strip()
        if not prompt:
            continue

        ref_resp, ref_ids = generate_summary(
            ref_model, tokenizer, [prompt],
            args.max_new_tokens, args.temperature, args.top_p, device
        )
        dpo_resp, dpo_ids = generate_summary(
            policy_model, tokenizer, [prompt],
            args.max_new_tokens, args.temperature, args.top_p, device
        )

        ref_ids = ref_ids[0]
        dpo_ids = dpo_ids[0]

        summaries_a.append(dpo_resp[0])
        summaries_b.append(ref_resp[0])
        originals.append(ex["prompt"])

        attn = (dpo_ids != tokenizer.pad_token_id).long()
        kl = compute_kl_divergence(policy_model, ref_model, dpo_ids, attn, device)
        kls.append(kl)

    win_a, _ = generate_win_rate(
        chat_pipeline,
        summaries_a,
        summaries_b,
        originals,
        prompt_template=config["dpo"]["prompt"],
        temperature=0.7,
        save_path=args.save_judge_outputs
    )

    print("\n==== DPO SUMMARY EVAL ====")
    print(f"Win rate (DPO > Ref): {sum(win_a)/len(win_a):.3f}")
    print(f"Avg KL(policy || ref): {sum(kls)/len(kls):.4f}")
    print(f"Judge outputs saved to: {args.save_judge_outputs}")


if __name__ == "__main__":
    main()
