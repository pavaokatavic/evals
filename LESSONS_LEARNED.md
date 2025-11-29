# LLM API Lessons Learned

Practical insights from building evaluation pipelines across Google Gemini and OpenAI APIs.

## Table of Contents
- [Rate Limits](#rate-limits)
- [OpenAI: Responses API vs Chat Completions](#openai-responses-api-vs-chat-completions)
- [Function Calling for Reliable Output](#function-calling-for-reliable-output)
- [GPT-5.1 Specifics](#gpt-51-specifics)
- [Provider Comparison](#provider-comparison)

---

## Rate Limits

### Google Gemini Free Tier

Each model has **separate quotas**:

| Model | RPM | RPD |
|-------|-----|-----|
| Gemini 2.5 Pro | 2 | 50 |
| Gemini 2.5 Flash | 10 | 250 |
| Gemini 2.5 Flash-Lite | 15 | 1,000 |
| Gemini 2.0 Flash | 15 | 200 |
| Gemini 2.0 Flash-Lite | 30 | 200 |

**Key insight:** RPD (requests per day) is often the bottleneck, not RPM.

### Rate Limiter Implementation

```python
import threading
import time

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
```

### Exponential Backoff on 429 Errors

```python
for attempt in range(retries):
    try:
        result = api_call()
        return result
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            wait = (2 ** attempt) * 5  # 5s, 10s, 20s
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
        else:
            raise
```

---

## OpenAI: Responses API vs Chat Completions

GPT-5.1 works best with the **Responses API**, which supports passing chain-of-thought between turns.

### Chat Completions (Legacy)

```python
from openai import OpenAI
client = OpenAI()

# Old way
result = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
)
print(result.choices[0].message.content)
```

### Responses API (New)

```python
# New way for GPT-5.1
result = client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[{"role": "user", "content": "Hello"}],
    max_completion_tokens=100,  # Note: different parameter name!
)
print(result.output_text)
```

### Key Differences

| Feature | Chat Completions | Responses API |
|---------|-----------------|---------------|
| Token limit param | `max_tokens` | `max_completion_tokens` |
| Input format | `messages` | `input` |
| Output access | `choices[0].message.content` | `output_text` or `output` |
| CoT passing | Not supported | Supported |
| Tool calls | `tool_calls` array | `output` with `type: "function_call"` |

---

## Function Calling for Reliable Output

**Problem:** Parsing free-text responses is unreliable.

```python
# Unreliable - model might say "Pass", "PASS", "pass.", "I think it passes", etc.
text = result.output_text.strip().lower()
return "pass" if "pass" in text else "fail"  # Fragile!
```

**Solution:** Use function calling with strict schema.

### Define Tool with Enum

```python
EVAL_TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "description": "Submit the evaluation result",
    "strict": True,  # Enforces schema compliance
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],  # Only these values allowed
                "description": "pass if correct, fail if wrong"
            }
        },
        "required": ["verdict"],
        "additionalProperties": False  # Required for strict mode
    }
}
```

### Force Tool Call

```python
result = client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[{"role": "user", "content": prompt}],
    tools=[EVAL_TOOL],
    tool_choice={"type": "function", "name": "submit_evaluation"},  # Force this tool
)

# Parse structured output
for item in result.output:
    if item.type == "function_call" and item.name == "submit_evaluation":
        args = json.loads(item.arguments)
        verdict = args["verdict"]  # Guaranteed to be "pass" or "fail"
```

### Benefits

- **Guaranteed schema:** `strict: True` + `enum` = only valid values
- **No parsing errors:** JSON output, not free text
- **Faster:** Model doesn't need to generate explanation text

---

## GPT-5.1 Specifics

### Parameter Changes from GPT-4

```python
# GPT-4 / GPT-4o
client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    max_tokens=100,        # ✓ Works
    temperature=0.7,       # ✓ Works
)

# GPT-5.1
client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[...],
    max_completion_tokens=100,  # ✓ Renamed parameter
    temperature=0.7,            # ⚠️ Only works with reasoning: "none"
)
```

### Reasoning Effort

GPT-5.1 defaults to `reasoning: "none"` for low latency:

```python
# Fast, low-latency (default)
result = client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[...],
    reasoning={"effort": "none"},
)

# For complex tasks (coding, math)
result = client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[...],
    reasoning={"effort": "high"},
)
```

| Setting | Use Case |
|---------|----------|
| `none` | Simple Q&A, classification, evals |
| `low` | Light reasoning tasks |
| `medium` | Balanced |
| `high` | Complex coding, multi-step planning |

### temperature Only Works with reasoning: "none"

```python
# ✓ Works
client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[...],
    reasoning={"effort": "none"},
    temperature=0,
)

# ✗ Error
client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[...],
    reasoning={"effort": "medium"},
    temperature=0,  # Not supported with reasoning!
)
```

---

## Provider Comparison

### Gemini (google-generativeai)

```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-lite")

result = model.generate_content("Hello")
print(result.text)
```

**Pros:**
- Generous free tier (especially Flash-Lite: 1000 RPD)
- Fast inference
- Simple API

**Cons:**
- Daily limits hit quickly
- Less sophisticated function calling
- No strict schema enforcement

### OpenAI (openai)

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

result = client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[{"role": "user", "content": "Hello"}],
)
print(result.output_text)
```

**Pros:**
- Strict function calling schemas
- Responses API with CoT
- Better instruction following

**Cons:**
- Pay-per-use (no free tier)
- API changes between model versions

---

## Quick Reference

### Flush Output in Background Scripts

```python
# Without flush, output buffers and you won't see progress
print(f"Completed {i}/200", flush=True)
```

### Environment Variables

```bash
# .env file
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-proj-...
```

```python
from dotenv import load_dotenv
load_dotenv()

# Access
api_key = os.getenv("GOOGLE_API_KEY")
```

### Minimal Dependencies (pyproject.toml)

```toml
[project]
name = "evals"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "google-generativeai",
    "openai>=1.0",
    "python-dotenv",
]
```

### Run with uv

```bash
# Auto-creates venv and installs deps
uv run script.py
```

---

## Summary

| Lesson | Recommendation |
|--------|----------------|
| Rate limits | Check RPD, not just RPM |
| Structured output | Use function calling + strict schema |
| GPT-5.1 | Use Responses API, not Chat Completions |
| Free tiers | Gemini Flash-Lite has best limits |
| Background scripts | Always `flush=True` for print |
| Reliability | Exponential backoff on 429s |
