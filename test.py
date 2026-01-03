from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------
# Load a small judge model for testing
# -----------------------------
judge_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(judge_name)
model = AutoModelForCausalLM.from_pretrained(
    judge_name,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

# Greedy pipeline for deterministic output
judge_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=16,  # just enough for "A" or "B"
    do_sample=False,    # greedy decoding
)

# -----------------------------
# Sample post and summaries
# -----------------------------
post = """I went to the store to buy some fruits, but they were out of apples. I bought oranges instead and returned home."""
summary_a = "The person wanted apples but bought oranges."
summary_b = "Someone went shopping for apples and oranges."

# -----------------------------
# Build a minimal prompt
# -----------------------------
prompt = f"""
Post:
{post}

Summary A:
{summary_a}

Summary B:
{summary_b}

Which summary is better? ONLY respond with "A" or "B".
"""

# -----------------------------
# Get judge response
# -----------------------------
response = judge_pipeline(prompt)
output = response[0]["generated_text"].strip()

# -----------------------------
# Parse the choice
# -----------------------------
choice = None
if output.upper().startswith("A"):
    choice = "A"
elif output.upper().startswith("B"):
    choice = "B"

print("Judge raw output:", output)
print("Parsed choice:", choice)
