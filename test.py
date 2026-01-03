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
Which of the following summaries does a better job of summarizing the most important points
in the given forum post, without including unimportant or irrelevant details?

Post:
I loved the new update, but the battery drains faster and the UI feels cluttered.

Summary A:
The user says they enjoyed the update but complains about battery drain and a cluttered interface.

Summary B:
The user mentions an update and discusses their general feelings.

Provide your response in the following format:
Comparison: <one sentence>
Preferred: <"A" or "B">
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
        max_new_tokens=64,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = outputs[0][inputs["input_ids"].shape[1]:]
text = tokenizer.decode(generated, skip_special_tokens=True)

print("=== JUDGE OUTPUT ===")
print(text)
