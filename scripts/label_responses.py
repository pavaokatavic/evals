"""
Evaluate question/response pairs and label as pass/fail.
Uses OpenAI GPT-5.1 with function calling for reliable structured output.
"""

import os
import csv
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "data/questions_version_2.csv"
OUTPUT_FILE = "data/questions_version_2.csv"
MODEL = "gpt-5.1-2025-11-13"

# Define evaluation tool with strict schema
EVAL_TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "description": "Submit the evaluation result for a question/response pair",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],
                "description": "pass if response is correct and helpful, fail if wrong or unhelpful"
            }
        },
        "required": ["verdict"],
        "additionalProperties": False
    }
}


def evaluate_response(question: str, response: str, retries: int = 3) -> str:
    """Evaluate using function calling for reliable structured output."""

    prompt = f"""Evaluate if this AI response correctly and helpfully answers the question.

Question: {question}
Response: {response}

Criteria:
1. Is the response factually correct?
2. Does it actually answer the question asked?
3. Is it helpful and not misleading?

Use the submit_evaluation tool with verdict="pass" if correct and helpful, or verdict="fail" if wrong/unhelpful."""

    for attempt in range(retries):
        try:
            result = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
                tools=[EVAL_TOOL],
                tool_choice={"type": "function", "name": "submit_evaluation"},
            )

            # Parse function call from output
            for item in result.output:
                if item.type == "function_call" and item.name == "submit_evaluation":
                    args = json.loads(item.arguments)
                    return args.get("verdict", "fail")

            return "fail"  # No function call found

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str:
                wait = (2 ** attempt) * 2
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  Error: {e}", flush=True)
                return "fail"

    return "fail"


def main():
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows", flush=True)

    to_label = [r for r in rows if not r.get("label")]
    print(f"Labeling {len(to_label)} responses with {MODEL} (function calling)...\n", flush=True)

    completed = 0
    pass_count = 0
    fail_count = 0

    for row in rows:
        if row.get("label"):
            if row["label"] == "pass":
                pass_count += 1
            else:
                fail_count += 1
            continue

        label = evaluate_response(row["question"], row["response"])
        row["label"] = label

        if label == "pass":
            pass_count += 1
        else:
            fail_count += 1

        completed += 1
        if completed % 10 == 0:
            print(f"  Completed {completed}/{len(to_label)} (pass: {pass_count}, fail: {fail_count})", flush=True)

    fieldnames = ["id", "question", "response", "label", "category"]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Results:", flush=True)
    print(f"  Pass: {pass_count}", flush=True)
    print(f"  Fail: {fail_count}", flush=True)
    print(f"  Fail rate: {fail_count / len(rows) * 100:.1f}%", flush=True)
    print(f"\nSaved to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
