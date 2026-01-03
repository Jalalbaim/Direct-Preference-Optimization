import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
).eval()

prompt = """You are a strict evaluator.

You MUST output two lines.
You MUST output both lines even if obvious.

Use EXACTLY this format:

Comparison: <one sentence>
Preferred: A or B

Post:
I loved the new update, but the battery drains faster and the UI feels cluttered.

Summary A:
The user says they enjoyed the update but complains about battery drain and a cluttered interface.

Summary B:
The user mentions an update and discusses their general feelings.

Now output the two lines.
"""

messages = [{"role": "user", "content": prompt}]

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
        pad_token_id=tokenizer.eos_token_id,
    )

generated = outputs[0][inputs["input_ids"].shape[1]:]
text = tokenizer.decode(generated, skip_special_tokens=True).strip()

# HARD POST-PROCESSING SAFETY
lines = [l for l in text.splitlines() if l.strip()]
text = "\n".join(lines[:2])

print("=== JUDGE OUTPUT ===")
print(text)

# FAIL FAST
assert len(lines) >= 2, "❌ Judge did not produce two lines"
assert re.search(r"^Preferred:\s*[AB]$", lines[1]), \
       f"❌ Bad judge format:\n{text}"
