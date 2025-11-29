"""
Verify LLM Evaluator Against Ground Truth Labels

This script implements Eugene Yan's eval verification methodology:
https://eugeneyan.com/writing/product-evals/

Setup:
- Ground truth: GPT-5.1 labels in data/questions_version_2.csv (simulating human labels)
- Evaluator: GPT-5-mini (smaller, cheaper model)
- Goal: Measure how well gpt-5-mini can replicate GPT-5.1's labeling decisions

Key Metrics (from Eugene Yan):
- Cohen's Kappa: 0.4-0.6 = substantial agreement, >0.7 = excellent
- Prioritize RECALL on failures (catching defects is critical)
- Human inter-rater reliability is often only 0.2-0.3

Usage:
    uv run verify_evaluator.py [--sample N] [--train-test]
"""

import os
import csv
import json
import random
import argparse
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Smaller model as evaluator (we're testing if it can match GPT-5.1 labels)
EVALUATOR_MODEL = "gpt-5-mini-2025-08-07"
DATA_FILE = "data/questions_version_2.csv"

# Function calling tool for reliable structured output
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
                "description": "pass if the response correctly answers the question, fail otherwise"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation for the verdict (1-2 sentences)"
            }
        },
        "required": ["verdict", "reasoning"],
        "additionalProperties": False
    }
}


def load_dataset(filepath: str) -> list[dict]:
    """Load labeled dataset."""
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("label")]  # Only rows with labels


def evaluate_response(question: str, response: str, retries: int = 3) -> tuple[str, str]:
    """
    Evaluate a question/response pair using the evaluator model.
    Returns (verdict, reasoning).
    """
    prompt = f"""You are evaluating whether an AI response correctly and helpfully answers a question.

Question: {question}

Response: {response}

Evaluate the response on these criteria:
1. Is the response factually correct?
2. Does it actually answer the question asked?
3. Is it complete and not misleading?

Use the submit_evaluation tool to provide your verdict:
- "pass" if the response is correct and helpful
- "fail" if the response is wrong, incomplete, or misleading"""

    for attempt in range(retries):
        try:
            result = client.responses.create(
                model=EVALUATOR_MODEL,
                input=[{"role": "user", "content": prompt}],
                tools=[EVAL_TOOL],
                tool_choice={"type": "function", "name": "submit_evaluation"},
            )

            for item in result.output:
                if item.type == "function_call" and item.name == "submit_evaluation":
                    args = json.loads(item.arguments)
                    return args.get("verdict", "fail"), args.get("reasoning", "")

            return "fail", "No function call found"

        except Exception as e:
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                return "fail", f"Error: {e}"

    return "fail", "Max retries exceeded"


def calculate_metrics(results: list[dict]) -> dict:
    """
    Calculate evaluation metrics.

    Key insight from Eugene Yan:
    - Prioritize RECALL on failures (we want to catch defects)
    - Cohen's Kappa accounts for chance agreement
    """
    # Confusion matrix
    tp = sum(1 for r in results if r["label"] == "pass" and r["pred"] == "pass")
    fp = sum(1 for r in results if r["label"] == "fail" and r["pred"] == "pass")
    fn = sum(1 for r in results if r["label"] == "pass" and r["pred"] == "fail")
    tn = sum(1 for r in results if r["label"] == "fail" and r["pred"] == "fail")

    total = len(results)

    # Basic metrics
    accuracy = (tp + tn) / total if total > 0 else 0

    # Precision/Recall for PASS predictions
    precision_pass = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pass = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Precision/Recall for FAIL predictions (critical per Eugene Yan)
    precision_fail = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_fail = tn / (tn + fp) if (tn + fp) > 0 else 0  # This is what we care about!

    # F1 scores
    f1_pass = 2 * precision_pass * recall_pass / (precision_pass + recall_pass) if (precision_pass + recall_pass) > 0 else 0
    f1_fail = 2 * precision_fail * recall_fail / (precision_fail + recall_fail) if (precision_fail + recall_fail) > 0 else 0

    # Cohen's Kappa
    # p_observed = accuracy
    # p_expected = probability of agreement by chance
    p_observed = accuracy
    p_yes = ((tp + fp) / total) * ((tp + fn) / total) if total > 0 else 0
    p_no = ((tn + fn) / total) * ((tn + fp) / total) if total > 0 else 0
    p_expected = p_yes + p_no
    kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else 0

    # Confidence interval for accuracy (95%)
    import math
    se = math.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0
    ci_lower = max(0, accuracy - 1.96 * se)
    ci_upper = min(1, accuracy + 1.96 * se)

    return {
        "total": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy,
        "accuracy_ci": (ci_lower, ci_upper),
        "precision_pass": precision_pass,
        "recall_pass": recall_pass,
        "f1_pass": f1_pass,
        "precision_fail": precision_fail,
        "recall_fail": recall_fail,  # Critical metric!
        "f1_fail": f1_fail,
        "kappa": kappa,
    }


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa score."""
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.2:
        return "Slight agreement"
    elif kappa < 0.4:
        return "Fair agreement"
    elif kappa < 0.6:
        return "Moderate/Substantial agreement"  # Eugene's target
    elif kappa < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def main():
    parser = argparse.ArgumentParser(description="Verify LLM evaluator against ground truth")
    parser.add_argument("--sample", type=int, help="Evaluate only N random samples")
    parser.add_argument("--train-test", action="store_true", help="Use 75/25 train/test split")
    parser.add_argument("--category", type=str, help="Filter by category")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM Evaluator Verification")
    print("=" * 70)
    print(f"\nEvaluator Model: {EVALUATOR_MODEL}")
    print(f"Ground Truth: GPT-5.1 labels from {DATA_FILE}")
    print(f"Methodology: Eugene Yan's Product Evals")
    print()

    # Load data
    data = load_dataset(DATA_FILE)
    print(f"Loaded {len(data)} labeled samples")

    # Filter by category if specified
    if args.category:
        data = [d for d in data if d.get("category") == args.category]
        print(f"Filtered to {len(data)} samples in category '{args.category}'")

    # Count label distribution
    pass_count = sum(1 for d in data if d["label"] == "pass")
    fail_count = sum(1 for d in data if d["label"] == "fail")
    print(f"Label distribution: {pass_count} pass, {fail_count} fail ({fail_count/len(data)*100:.1f}% failure rate)")

    # Sample if requested
    if args.sample:
        random.seed(42)
        data = random.sample(data, min(args.sample, len(data)))
        print(f"Sampled {len(data)} items for evaluation")

    # Train/test split if requested
    if args.train_test:
        random.seed(42)
        random.shuffle(data)
        split = int(len(data) * 0.75)
        train_data, test_data = data[:split], data[split:]
        print(f"Split: {len(train_data)} train (for prompt tuning), {len(test_data)} test")
        data = test_data  # Only evaluate on test set

    print(f"\nEvaluating {len(data)} samples...")
    print("-" * 70)

    # Run evaluation
    results = []
    disagreements = []

    for i, row in enumerate(data, 1):
        pred, reasoning = evaluate_response(row["question"], row["response"])

        match = pred == row["label"]
        results.append({
            "id": row["id"],
            "category": row.get("category", ""),
            "label": row["label"],
            "pred": pred,
            "match": match,
            "reasoning": reasoning,
        })

        symbol = "✓" if match else "✗"
        if not match:
            disagreements.append({
                "id": row["id"],
                "question": row["question"][:60] + "..." if len(row["question"]) > 60 else row["question"],
                "label": row["label"],
                "pred": pred,
                "reasoning": reasoning,
            })

        if i % 10 == 0 or i == len(data):
            correct = sum(1 for r in results if r["match"])
            print(f"  Progress: {i}/{len(data)} | Accuracy so far: {correct/i*100:.1f}%", flush=True)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n### Confusion Matrix ###")
    print(f"                  Predicted")
    print(f"                  Pass    Fail")
    print(f"  Actual Pass     {metrics['tp']:4d}    {metrics['fn']:4d}")
    print(f"  Actual Fail     {metrics['fp']:4d}    {metrics['tn']:4d}")

    print("\n### Overall Metrics ###")
    print(f"  Accuracy:     {metrics['accuracy']*100:.1f}% (95% CI: {metrics['accuracy_ci'][0]*100:.1f}%-{metrics['accuracy_ci'][1]*100:.1f}%)")
    print(f"  Cohen's Kappa: {metrics['kappa']:.3f} - {interpret_kappa(metrics['kappa'])}")

    print("\n### Pass Detection ###")
    print(f"  Precision: {metrics['precision_pass']*100:.1f}%")
    print(f"  Recall:    {metrics['recall_pass']*100:.1f}%")
    print(f"  F1 Score:  {metrics['f1_pass']*100:.1f}%")

    print("\n### Fail Detection (Critical per Eugene Yan) ###")
    print(f"  Precision: {metrics['precision_fail']*100:.1f}%")
    print(f"  Recall:    {metrics['recall_fail']*100:.1f}%  <-- Key metric: catching failures")
    print(f"  F1 Score:  {metrics['f1_fail']*100:.1f}%")

    # Per-category breakdown
    categories = defaultdict(list)
    for r in results:
        if r["category"]:
            categories[r["category"]].append(r)

    if categories:
        print("\n### Per-Category Accuracy ###")
        for cat, cat_results in sorted(categories.items()):
            cat_metrics = calculate_metrics(cat_results)
            fail_rate = sum(1 for r in cat_results if r["label"] == "fail") / len(cat_results) * 100
            print(f"  {cat:25s}: {cat_metrics['accuracy']*100:5.1f}% acc, {cat_metrics['kappa']:.2f} kappa, {fail_rate:.0f}% fail rate ({len(cat_results)} samples)")

    # Show disagreements
    if disagreements:
        print(f"\n### Sample Disagreements (showing up to 10) ###")
        for d in disagreements[:10]:
            print(f"\n  ID {d['id']}: Ground truth={d['label']}, Predicted={d['pred']}")
            print(f"  Q: {d['question']}")
            print(f"  Reasoning: {d['reasoning'][:100]}..." if len(d['reasoning']) > 100 else f"  Reasoning: {d['reasoning']}")

    # Eugene Yan's recommendations
    print("\n" + "=" * 70)
    print("INTERPRETATION (Eugene Yan's Guidelines)")
    print("=" * 70)

    if metrics['kappa'] >= 0.6:
        print("  ✓ Cohen's Kappa >= 0.6: Substantial agreement - evaluator is reliable")
    elif metrics['kappa'] >= 0.4:
        print("  ~ Cohen's Kappa 0.4-0.6: Moderate agreement - acceptable for most use cases")
    else:
        print("  ✗ Cohen's Kappa < 0.4: Fair/slight agreement - consider improving prompts")

    if metrics['recall_fail'] >= 0.8:
        print("  ✓ Fail Recall >= 80%: Good at catching failures (critical for trust)")
    elif metrics['recall_fail'] >= 0.6:
        print("  ~ Fail Recall 60-80%: Moderate failure detection - some defects may slip through")
    else:
        print("  ✗ Fail Recall < 60%: Poor failure detection - many defects will be missed")

    print(f"\n  Note: Human inter-rater reliability is often only kappa 0.2-0.3")
    print(f"  Your evaluator: kappa {metrics['kappa']:.3f}")


if __name__ == "__main__":
    main()
