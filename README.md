# LLM Product Evals Demo

Simple demo illustrating [Eugene Yan's "Product Evals in Three Simple Steps"](https://eugeneyan.com/writing/product-evals/).

## Quick Start

```bash
# 1. Generate answers for 200 questions using Gemini Flash Lite
uv run generate_answers.py

# 2. Run the evaluation demo
uv run eval_demo.py
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Add your Gemini API key to `.env`:
   ```
   GOOGLE_API_KEY=your-key-here
   ```

3. Run:
   ```bash
   uv run eval_demo.py
   ```

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Binary labels | `pass`/`fail` only, no numeric scales |
| Train/test split | 75/25 split for prompt alignment vs evaluation |
| Single-dimension evaluators | Separate `evaluate_helpfulness()` and `evaluate_accuracy()` |
| Position bias mitigation | `compare_responses()` runs twice with swapped order |
| Cohen's Kappa | Measures agreement beyond chance (human baseline ~0.2-0.3) |
| Parallel execution | `ThreadPoolExecutor` for speed |
| Confidence intervals | Statistical rigor for accuracy metrics |

## Files

| File | Description |
|------|-------------|
| `questions.csv` | 200 sample questions (generates organic failures) |
| `generate_answers.py` | Uses Gemini Flash Lite to generate answers |
| `eval_demo.py` | Runs evaluators and calculates metrics |

## Article Summary

1. **Label data** - Use binary pass/fail, aim for 50-100 failures in 200+ samples
2. **Align evaluators** - One evaluator per dimension, not a "God Evaluator"
3. **Run harness** - Parallel execution with proper metrics and CI
