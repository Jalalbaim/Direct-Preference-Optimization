from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Sample data ---
post = "I went to the store to buy some fruits, but they were out of apples. I bought oranges instead and returned home."
summary_a = "The person wanted apples but bought oranges."
summary_b = "Someone went shopping for apples and oranges."

# --- Load judge model (TinyLlama) ---
judge_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(judge_name)
model = AutoModelForCausalLM.from_pretrained(
    judge_name,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

# --- Create pipeline ---
judge_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=32,  # only need a few tokens for A/B
    do_sample=False,    # deterministic output
)

# --- Construct a short, clear prompt ---
prompt = f"""<|im_start|>user
Post: {post}
Summary A: {summary_a}
Summary B: {summary_b}
Which summary is better? Respond ONLY with the single letter A or B.<|im_end|>
<|im_start|>assistant
"""


# --- Get model output ---
output = judge_pipeline(prompt)
raw_text = output[0]["generated_text"]
print("Judge raw output:\n", raw_text)

# --- Parse output: look for A or B ---
import re
match = re.search(r"\b(A|B)\b", raw_text)
choice = match.group(1) if match else None
print("Parsed choice:", choice)
