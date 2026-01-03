import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

judge_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(judge_name)
model = AutoModelForCausalLM.from_pretrained(
    judge_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
).eval()

prompt = """
You are a strict evaluator.

Choose which summary is better.

Respond with EXACTLY two lines and nothing else.

Line 1 must start with:
Comparison:

Line 2 must be EXACTLY one of:
Preferred: A
Preferred: B

Post:
{post}

Summary A:
{summary_a}

Summary B:
{summary_b}
"""

messages = [
    {"role": "user", "content": prompt}
]

chat_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = outputs[0][inputs["input_ids"].shape[1]:]
text = tokenizer.decode(generated, skip_special_tokens=True)

print("=== JUDGE OUTPUT ===")
print(text)
