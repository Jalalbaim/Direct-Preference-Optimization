from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

# --- Sample data ---
post = "About 3 weeks ago, I got a job at a grocery store as a stock boy. I was so happy that I could start saving up and buying more expensive things (Like a goddamn"
summary_a = "Got fired from my first job after 3 weeks. Lost my ability to save money and now I have to go back to living paycheck to paycheck."
summary_b = "Got fired from my first job for being too slow. Got another job, but I could only work until 4pm."

# --- Load model ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    max_new_tokens=12,  # Allow more tokens for the explanation
    do_sample=False,
)

# --- Construct a strict prompt with an example ---
prompt = f"""
<|im_start|>user
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise.

Post:
{post}

Summary A:
{summary_a}

Summary B:
{summary_b}

Provide your response in the following format:
Preferred: <"A" or "B">

After providing your answer, stop generating further text.<|im_end|>
<|im_start|>assistant
"""



# --- Get model output ---
output = judge_pipeline(prompt)
raw_text = output[0]["generated_text"]
print("Judge raw output:\n", raw_text)