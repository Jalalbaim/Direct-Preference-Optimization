from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

# --- Sample data ---
post = "About 3 weeks ago, I got a job at a grocery store as a stock boy. I was so happy that I could start saving up and buying more expensive things (Like a goddamn"
summary_a = "Got fired from my first job after 3 weeks. Lost my ability to save money and now I have to go back to living paycheck to paycheck."
summary_b = "Got fired from my first job for being too slow. Got another job, but I could only work until 4pm."

# --- Load model ---
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

# --- Create pipeline ---
judge_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,  # Allow more tokens for the explanation
    do_sample=False,
)

# --- Construct a strict prompt with an example ---
prompt = f"""
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise.

Example:
Post: I went to the store to buy apples, but they were out of stock. I bought oranges instead.
Summary A: I bought oranges.
Summary B: I went to the store.
Comparison: Summary A is preferred because it captures the main point (buying oranges).
Preferred: A

Post:
{post}

Summary A:
{summary_a}

Summary B:
{summary_b}

FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""

# --- Get model output ---
output = judge_pipeline(prompt)
raw_text = output[0]["generated_text"]
print("Judge raw output:\n", raw_text)

# --- Parse output: look for "Preferred: A" or "Preferred: B" ---
match = re.search(r"Preferred:\s*(A|B)", raw_text, re.IGNORECASE)
choice = match.group(1) if match else None
