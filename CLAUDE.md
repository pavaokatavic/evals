# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM evaluation pipeline implementing Eugene Yan's "Product Evals in Three Simple Steps" methodology.

**Inspiration:** https://eugeneyan.com/writing/product-evals/

Key principles:
- Binary pass/fail labels (not numeric scales)
- Aim for 50-100 failures in 200+ samples for meaningful signal
- One evaluator per dimension, not a "God Evaluator"
- Use smaller models to generate organic failures
- Target Cohen's Kappa 0.4-0.6 (substantial agreement)

## Commands

**Always use Astral UV, never venv directly.**

```bash
# Generate answers for questions using Gemini
uv run scripts/generate_answers.py

# Label responses as pass/fail using GPT-5.1
uv run scripts/label_responses.py

# Run evaluation demo with metrics
uv run eval_demo.py

# Generate harder questions (25%+ failure rate target)
uv run scripts/generate_hard_questions.py

# Add dependencies
uv add package-name
```

## Datasets

| Dataset | Questions | Failures | Fail Rate | Answer Model |
|---------|-----------|----------|-----------|--------------|
| `data/questions.csv` | 200 | 7 | 3.5% | Gemini 2.0 Flash Lite |
| `data/questions_version_2.csv` | 200 | 71 | 35.5% | GPT-3.5-turbo-0125 |

**data/questions_version_2.csv** uses harder question categories designed for higher failure rates:

| Category | Fail Rate | Description |
|----------|-----------|-------------|
| precise_calculations | 60% | Math, code output, conversions |
| temporal_changing | 60% | Current events, changing facts |
| multi_part | 37% | Questions requiring multiple pieces of info |
| ambiguous_debatable | 26% | Contested facts, multiple valid answers |
| edge_cases | 23% | Nuanced topics, exceptions |
| trick_misconceptions | 14% | Common myths (model performed well here) |

## Architecture

### Pipeline Flow

```
data/questions.csv (questions only)
    ↓
scripts/generate_answers.py (Gemini 2.0 Flash Lite)
    ↓
data/questions.csv (questions + responses)
    ↓
scripts/label_responses.py (GPT-5.1 with function calling)
    ↓
data/questions.csv (questions + responses + labels)
    ↓
eval_demo.py (metrics: precision, recall, kappa)
```

### Models

| Provider | Model | Use Case |
|----------|-------|----------|
| Google | gemini-2.0-flash-lite | Answer generation (free tier, 30 RPM / 200 RPD) |
| OpenAI | gpt-5.1-2025-11-13 | Question generation, labeling (Responses API) |
| OpenAI | gpt-3.5-turbo-0125 | Weak answer model for generating failures |

### Key Patterns

**Function calling for reliable output** (see `scripts/label_responses.py`):
- Use `strict: True` + `enum` for guaranteed schema compliance
- Force tool call with `tool_choice={"type": "function", "name": "..."}`

**Rate limiting** (see `scripts/generate_answers.py`):
- Gemini RPD (requests per day) is the bottleneck, not RPM
- Use exponential backoff on 429 errors

## Technical Reference

See `LESSONS_LEARNED.md` for detailed API patterns:
- OpenAI Responses API vs Chat Completions
- GPT-5.1 parameter changes (`max_completion_tokens`, reasoning effort)
- Provider comparison and rate limit tables

## Environment

Required in `.env`:
```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
```
