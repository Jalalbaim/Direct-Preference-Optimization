from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Sample data ---
post="About 3 weeks ago, I got a job at a grocery store as a stock boy. I was so happy that I could start saving up and buying more expensive things (Like a goddamn"
summary_a="Got fired from my first job after 3 weeks. Lost my ability to save money and now I have to go back to living paycheck to paycheck."
summary_b="Got fired from my first job for being too slow. Got another job, but I could only work until 4pm."


# --- Load judge model (TinyLlama) ---
judge_name = "gpt2" # Found : https://huggingface.co/docs/transformers/tasks/language_modeling 
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
prompt = f"""
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise.

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

# --- Parse output: look for A or B ---
import re
match = re.search(r"\b(A|B)\b", raw_text)
choice = match.group(1) if match else None
