"""
Helper script to generate answers for questions using Gemini Flash Lite.
Reads questions.csv, generates responses, and saves back to CSV.
"""

import os
import csv
import time
import threading
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-lite")

INPUT_FILE = "data/questions.csv"
OUTPUT_FILE = "data/questions.csv"

# Gemini 2.0 Flash Lite free tier limits:
#   RPM: 30 requests/minute
#   RPD: 200 requests/day (!)
# Paid tier: 2000 RPM, 4000 RPD
REQUESTS_PER_MINUTE = 30  # Free tier limit
REQUESTS_PER_DAY = 200    # Free tier limit - can only run once/day!


class RateLimiter:
    """Thread-safe rate limiter."""

    def __init__(self, requests_per_minute):
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            wait_time = self.last_request + self.min_interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.time()


rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)  # 1 request every 2 seconds


def generate_answer(question: str, retries: int = 3) -> str:
    """Generate a concise answer with retry on rate limit errors."""
    prompt = f"""Answer this question concisely in 1-2 sentences.

Question: {question}

Answer:"""

    for attempt in range(retries):
        rate_limiter.wait()
        try:
            result = model.generate_content(prompt)
            return result.text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                # Exponential backoff on rate limit
                wait = (2 ** attempt) * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"

    return "ERROR: Max retries exceeded"


def main():
    # Read questions
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} questions", flush=True)

    # Count how many need answers
    to_process = [r for r in rows if not r["response"]]
    print(f"Generating answers for {len(to_process)} questions...", flush=True)
    print(f"Rate: {REQUESTS_PER_MINUTE} RPM (~{len(to_process) * 60 // REQUESTS_PER_MINUTE}s estimated)", flush=True)

    if len(to_process) > REQUESTS_PER_DAY:
        print(f"WARNING: {len(to_process)} requests exceeds daily limit of {REQUESTS_PER_DAY}!", flush=True)
        print("Consider upgrading to paid tier or splitting across days.\n", flush=True)
    else:
        print(f"Daily quota: {len(to_process)}/{REQUESTS_PER_DAY} requests\n", flush=True)

    # Process sequentially with rate limiting
    completed = 0
    for row in rows:
        if row["response"]:  # Skip if already has response
            continue

        row["response"] = generate_answer(row["question"])
        completed += 1

        if completed % 10 == 0:
            print(f"  Completed {completed}/{len(to_process)}", flush=True)

    # Write back to CSV
    fieldnames = ["id", "question", "response", "label"]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Saved {len(rows)} rows to {OUTPUT_FILE}")

    # Show sample
    print("\nSample results:")
    for row in rows[:5]:
        q = row["question"][:40]
        a = row["response"][:60] if row["response"] else "N/A"
        print(f"  Q: {q}...")
        print(f"  A: {a}...\n")


if __name__ == "__main__":
    main()
