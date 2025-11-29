# LLM Provider Comparison for Evaluations

A comparison of three approaches to building LLM-as-judge evaluators, based on Eugene Yan's ["Product Evals in Three Simple Steps"](https://eugeneyan.com/writing/product-evals/).

## Quick Start

```bash
# Run with GPT-5.1 (recommended)
uv run eval_demo_gpt5.py

# Run with GPT-4o-mini
uv run eval_demo_gpt4.py

# Run with Gemini (requires quota)
uv run eval_demo_gemini.py
```

## Summary

| Aspect | Gemini | GPT-4o-mini | GPT-5.1 |
|--------|--------|-------------|---------|
| **Script** | `eval_demo_gemini.py` | `eval_demo_gpt4.py` | `eval_demo_gpt5.py` |
| **Model** | gemini-2.0-flash-lite | gpt-4o-mini | gpt-5.1-2025-11-13 |
| **API** | generate_content | Chat Completions | Responses API |
| **Output Parsing** | Text matching | Text matching | Function calling |
| **Schema Enforcement** | None | None | Strict (enum) |
| **Position Bias** | Variable | High (returns "tie") | Low (consistent) |
| **Cost** | Free (200 RPD) | Pay-per-use | Pay-per-use |
| **Recommended** | Prototyping | Budget-conscious | Production |

## Detailed Comparison

### 1. Gemini 2.0 Flash Lite (`eval_demo_gemini.py`)

**API Pattern:**
```python
import google.generativeai as genai
model = genai.GenerativeModel("gemini-2.0-flash-lite")
result = model.generate_content(prompt)
answer = result.text
```

**Pros:**
- Free tier (200 requests/day)
- Simple API
- Fast inference

**Cons:**
- Daily quota exhausts quickly
- No structured output enforcement
- Fragile text parsing (`"pass" in result.text.lower()`)

**Best for:** Quick prototyping, learning, budget-constrained projects

---

### 2. GPT-4o-mini (`eval_demo_gpt4.py`)

**API Pattern:**
```python
from openai import OpenAI
client = OpenAI()
result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=10
)
answer = result.choices[0].message.content
```

**Pros:**
- Familiar Chat Completions API
- Cost-effective
- No rate limit issues

**Cons:**
- Same fragile text parsing as Gemini
- Position bias observed (comparison test returns "tie")
- No schema enforcement without function calling

**Best for:** Cost-conscious production, when you need more quota than Gemini

---

### 3. GPT-5.1 with Function Calling (`eval_demo_gpt5.py`)

**API Pattern:**
```python
from openai import OpenAI
client = OpenAI()

EVAL_TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "strict": True,  # Key: enforces schema
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],  # Only these values allowed
            }
        },
        "required": ["verdict"],
        "additionalProperties": False
    }
}

result = client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[{"role": "user", "content": prompt}],
    tools=[EVAL_TOOL],
    tool_choice={"type": "function", "name": "submit_evaluation"},
)

# Parse structured output
for item in result.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        verdict = args["verdict"]  # Guaranteed "pass" or "fail"
```

**Pros:**
- **Guaranteed schema compliance** - enum restricts to exact values
- **No text parsing** - structured JSON output
- **Consistent results** - less position bias
- Responses API supports chain-of-thought

**Cons:**
- Pay-per-use only
- More setup (tool definitions)
- Different API from Chat Completions

**Best for:** Production evaluators, high-stakes labeling, reliability-critical pipelines

---

## Key Learnings

### 1. Text Parsing is Fragile

Without function calling, you rely on text matching:
```python
# Can fail on: "Pass", "PASS", "pass.", "I think it passes", "It should pass"
return "pass" if "pass" in result.text.lower() else "fail"
```

Function calling with `strict: True` + `enum` eliminates this entirely.

### 2. Position Bias is Real

When comparing two responses A vs B:
- Run comparison with A first, B second
- Run again with B first, A second
- If answers differ → model has position bias

**Results observed:**
| Model | Position Bias Test |
|-------|-------------------|
| GPT-4o-mini | Returns "tie" (inconsistent) |
| GPT-5.1 | Returns "A" (consistent) |

### 3. API Differences Matter

| Feature | Chat Completions | Responses API |
|---------|-----------------|---------------|
| Input param | `messages` | `input` |
| Token limit | `max_tokens` | `max_completion_tokens` |
| Output | `choices[0].message.content` | `output_text` or `output` |
| Tool calls | `tool_calls` array | `output` with `type: "function_call"` |

### 4. Rate Limits: RPD > RPM

For Gemini free tier, **requests per day (RPD)** is the bottleneck:

| Model | RPM | RPD |
|-------|-----|-----|
| gemini-2.0-flash-lite | 30 | 200 |
| gemini-2.5-flash-lite | 15 | 1,000 |

Plan your batch sizes accordingly.

---

## Recommendation

**For production LLM evaluators:**

1. **Use GPT-5.1 with function calling** for reliability
2. **Define strict schemas** with enums for classification tasks
3. **Force tool calls** with `tool_choice` to guarantee structured output
4. **Test for position bias** in any comparison evaluators

**Migration path:**
```
Gemini (prototype) → GPT-4o-mini (scale) → GPT-5.1 (production)
```

---

## Files

```
eval_demo_gemini.py  # Google Gemini 2.0 Flash Lite
eval_demo_gpt4.py    # OpenAI GPT-4o-mini (Chat Completions)
eval_demo_gpt5.py    # OpenAI GPT-5.1 (Responses API + function calling)
```

All three scripts implement the same evaluation logic:
- Binary pass/fail classification
- Helpfulness and accuracy evaluators
- Position bias test for comparisons
- Metrics: precision, recall, accuracy, Cohen's Kappa
