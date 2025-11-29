"""
Simple demo illustrating Eugene Yan's "Product Evals in Three Simple Steps"
https://eugeneyan.com/writing/product-evals/

Using Gemini 2.0 Flash Lite for cost-effective evaluation

PROS:
- Free tier (200 RPD for Flash Lite)
- Simple API
- Fast inference

CONS:
- Daily quota limits hit quickly
- No strict schema enforcement
- Text parsing is fragile ("pass" vs "Pass" vs "I think it passes")
"""

import os
import random
import concurrent.futures
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# =============================================================================
# STEP 1: Label Some Data (Binary Labels + Organic Failures)
# =============================================================================

# Sample data: AI responses to evaluate (in production, use real samples)
# Key insight: Use BINARY labels (pass/fail), not numeric scales
samples = [
    {"id": 1, "question": "What is 2+2?", "response": "4", "label": "pass"},
    {"id": 2, "question": "Capital of France?", "response": "Paris", "label": "pass"},
    {"id": 3, "question": "What is Python?", "response": "A snake", "label": "fail"},
    {"id": 4, "question": "Explain recursion", "response": "See: recursion", "label": "fail"},
    {"id": 5, "question": "Best practice for passwords?", "response": "Use 'password123'", "label": "fail"},
    {"id": 6, "question": "How to learn coding?", "response": "Practice daily with projects", "label": "pass"},
    {"id": 7, "question": "What is ML?", "response": "Machine Learning uses data to make predictions", "label": "pass"},
    {"id": 8, "question": "Fix my code: x = 1/0", "response": "That looks fine!", "label": "fail"},
]

# Train/Test split (75/25) - use train set to align evaluator prompts
random.seed(42)
random.shuffle(samples)
split = int(len(samples) * 0.75)
train_set, test_set = samples[:split], samples[split:]

# =============================================================================
# STEP 2: Build Single-Dimension LLM Evaluators (not one "God Evaluator")
# =============================================================================

def evaluate_helpfulness(question: str, response: str) -> str:
    """Single dimension: Is the response helpful? Returns 'pass' or 'fail'."""
    prompt = f"""Evaluate if this response is helpful for the question.

Question: {question}
Response: {response}

Reply with exactly one word: 'pass' if helpful, 'fail' if not helpful."""

    result = model.generate_content(prompt)
    # Fragile parsing - model might say "Pass", "PASS", "pass.", "I think it passes"
    return "pass" if "pass" in result.text.lower() else "fail"


def evaluate_accuracy(question: str, response: str) -> str:
    """Single dimension: Is the response accurate?"""
    prompt = f"""Evaluate if this response is factually accurate.

Question: {question}
Response: {response}

Reply with exactly one word: 'pass' if accurate, 'fail' if inaccurate."""

    result = model.generate_content(prompt)
    return "pass" if "pass" in result.text.lower() else "fail"


def compare_responses(question: str, resp_a: str, resp_b: str) -> str:
    """Position bias mitigation: Compare with swapped order."""

    def run_comparison(first, second, first_label, second_label):
        prompt = f"""Which response better answers the question?

Question: {question}
<response_1>{first}</response_1>
<response_2>{second}</response_2>

Reply with exactly: '{first_label}' or '{second_label}'"""
        result = model.generate_content(prompt)
        return first_label if first_label in result.text else second_label

    result1 = run_comparison(resp_a, resp_b, "A", "B")
    result2 = run_comparison(resp_b, resp_a, "B", "A")

    return result1 if result1 == result2 else "tie"

# =============================================================================
# STEP 3: Run Evaluation Harness
# =============================================================================

def run_eval_parallel(samples, evaluator_fn):
    """Run evaluator in parallel for speed."""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(evaluator_fn, s["question"], s["response"]): s
            for s in samples
        }
        for future in concurrent.futures.as_completed(futures):
            sample = futures[future]
            pred = future.result()
            results.append({
                "id": sample["id"],
                "label": sample["label"],
                "prediction": pred,
                "match": sample["label"] == pred
            })

    return results


def calculate_metrics(results):
    """Calculate precision, recall, and Cohen's Kappa."""
    tp = sum(1 for r in results if r["label"] == "pass" and r["prediction"] == "pass")
    fp = sum(1 for r in results if r["label"] == "fail" and r["prediction"] == "pass")
    fn = sum(1 for r in results if r["label"] == "pass" and r["prediction"] == "fail")
    tn = sum(1 for r in results if r["label"] == "fail" and r["prediction"] == "fail")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(results)

    # Cohen's Kappa (human inter-rater kappa is often only 0.2-0.3)
    total = len(results)
    p_observed = accuracy
    p_expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total)
    kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else 0

    return {"precision": precision, "recall": recall, "accuracy": accuracy, "kappa": kappa}


def confidence_interval(accuracy, n, z=1.96):
    """95% confidence interval using standard error."""
    se = (accuracy * (1 - accuracy) / n) ** 0.5
    return (accuracy - z * se, accuracy + z * se)


if __name__ == "__main__":
    print(f"Train: {len(train_set)} samples, Test: {len(test_set)} samples\n")

    print("=== Running Helpfulness Evaluator ===")
    results = run_eval_parallel(test_set, evaluate_helpfulness)

    for r in sorted(results, key=lambda x: x["id"]):
        match = "✓" if r["match"] else "✗"
        print(f"  Sample {r['id']}: label={r['label']}, pred={r['prediction']} {match}")

    metrics = calculate_metrics(results)
    ci = confidence_interval(metrics["accuracy"], len(results))

    print("\n=== Metrics Report ===")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall:    {metrics['recall']:.2f}")
    print(f"  Accuracy:  {metrics['accuracy']:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f})")
    print(f"  Kappa:     {metrics['kappa']:.2f}")

    print("\n=== Position Bias Test ===")
    winner = compare_responses(
        "What is Python?",
        "A programming language for building software",
        "A snake"
    )
    print(f"  Winner (with position swap): {winner}")
