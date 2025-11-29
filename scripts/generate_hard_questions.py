"""
Generate challenging questions designed to produce higher failure rates.
Uses GPT-5.1 to generate questions, then GPT-3.5-turbo for answers (weaker model = more failures).
"""

import os
import csv
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_FILE = "data/questions_version_2.csv"
QUESTION_MODEL = "gpt-5.1-2025-11-13"  # Strong model for generating tricky questions
ANSWER_MODEL = "gpt-3.5-turbo-0125"    # Weak model for generating answers (more failures)

# Question categories designed to produce failures
QUESTION_CATEGORIES = {
    "ambiguous_debatable": {
        "count": 35,
        "description": "Questions with debatable or ambiguous answers",
        "examples": [
            "What is the longest river in the world?",
            "Who invented the airplane?",
            "What is the oldest civilization?",
        ]
    },
    "trick_misconceptions": {
        "count": 35,
        "description": "Questions that exploit common misconceptions",
        "examples": [
            "What color is the sun when viewed from space?",
            "How many senses do humans have?",
            "What percentage of the brain do humans use?",
        ]
    },
    "precise_calculations": {
        "count": 35,
        "description": "Questions requiring precise calculations or technical details",
        "examples": [
            "What is 17 * 23 + 456 / 12?",
            "What is the output of print([1,2,3][::-1]) in Python?",
            "Convert 0.375 to a fraction in lowest terms.",
        ]
    },
    "temporal_changing": {
        "count": 25,
        "description": "Questions about things that change over time or are time-sensitive",
        "examples": [
            "What is the tallest building in the world?",
            "How many countries are in the United Nations?",
            "What is the world population?",
        ]
    },
    "edge_cases": {
        "count": 35,
        "description": "Questions about edge cases, exceptions, or nuanced topics",
        "examples": [
            "Is a tomato a fruit or vegetable?",
            "How many continents are there?",
            "What is the capital of Myanmar?",
        ]
    },
    "multi_part": {
        "count": 35,
        "description": "Questions requiring multiple pieces of information",
        "examples": [
            "Name all planets in our solar system in order from the sun.",
            "What are the first 10 prime numbers?",
            "List all US states that start with 'M'.",
        ]
    },
}


def generate_questions_for_category(category_name: str, category_info: dict) -> list[str]:
    """Use GPT-5.1 to generate questions for a specific category."""

    prompt = f"""Generate exactly {category_info['count']} unique questions for the category: "{category_name}"

Category description: {category_info['description']}

Example questions in this category:
{chr(10).join(f"- {ex}" for ex in category_info['examples'])}

Requirements:
1. Questions should be challenging and likely to produce incorrect or imprecise answers from a weaker AI model
2. Each question should have a definitive correct answer (even if commonly misunderstood)
3. Questions should be diverse and cover different topics
4. Keep questions concise (1 sentence)
5. Do NOT include the answer in the question

Return ONLY a JSON array of question strings, no other text.
Example format: ["Question 1?", "Question 2?", ...]"""

    try:
        response = client.responses.create(
            model=QUESTION_MODEL,
            input=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        text = ""
        for item in response.output:
            if hasattr(item, 'content'):
                for content in item.content:
                    if hasattr(content, 'text'):
                        text = content.text
                        break

        # Parse JSON array
        # Find the JSON array in the response
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end > start:
            questions = json.loads(text[start:end])
            return questions[:category_info['count']]

        print(f"  Warning: Could not parse JSON for {category_name}")
        return []

    except Exception as e:
        print(f"  Error generating questions for {category_name}: {e}")
        return []


def generate_answer(question: str, retries: int = 3) -> str:
    """Generate answer using GPT-3.5-turbo (weaker model for more failures)."""

    for attempt in range(retries):
        try:
            result = client.chat.completions.create(
                model=ANSWER_MODEL,
                messages=[
                    {"role": "system", "content": "Answer questions concisely in 1-2 sentences."},
                    {"role": "user", "content": question}
                ],
                max_tokens=150,
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                wait = (2 ** attempt) * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"

    return "ERROR: Max retries exceeded"


def main():
    print("=" * 60)
    print("Generating Hard Questions Dataset")
    print("=" * 60)
    print(f"\nTarget: 200 questions designed for ~25% failure rate")
    print(f"Question generator: {QUESTION_MODEL}")
    print(f"Answer generator: {ANSWER_MODEL} (weaker model = more failures)")
    print(f"Output: {OUTPUT_FILE}\n")

    all_questions = []

    # Step 1: Generate questions using GPT-5.1
    print("Step 1: Generating questions with GPT-5.1...")
    print("-" * 40)

    for category_name, category_info in QUESTION_CATEGORIES.items():
        print(f"\n  Category: {category_name} (target: {category_info['count']})")
        questions = generate_questions_for_category(category_name, category_info)
        print(f"  Generated: {len(questions)} questions")

        for q in questions:
            all_questions.append({
                "question": q,
                "category": category_name
            })

    print(f"\n  Total questions generated: {len(all_questions)}")

    # Trim to exactly 200 if we have more
    if len(all_questions) > 200:
        all_questions = all_questions[:200]
        print(f"  Trimmed to: 200 questions")

    # Step 2: Generate answers using GPT-3.5-turbo
    print("\n" + "-" * 40)
    print(f"Step 2: Generating answers with {ANSWER_MODEL}...")
    print("-" * 40 + "\n")

    rows = []
    for i, item in enumerate(all_questions, 1):
        answer = generate_answer(item["question"])

        rows.append({
            "id": i,
            "question": item["question"],
            "response": answer,
            "label": "",  # To be filled by label_responses.py
            "category": item["category"]
        })

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(all_questions)}", flush=True)

    # Step 3: Write to CSV
    print("\n" + "-" * 40)
    print("Step 3: Saving to CSV...")

    fieldnames = ["id", "question", "response", "label", "category"]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} questions to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total questions: {len(rows)}")
    print("\nQuestions per category:")
    category_counts = {}
    for row in rows:
        cat = row["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")

    print(f"\nNext step: Run 'python label_responses.py' with INPUT_FILE='{OUTPUT_FILE}'")
    print("Expected failure rate: ~25% (50+ failures)")

    # Show sample questions
    print("\n" + "-" * 40)
    print("Sample questions:")
    for row in rows[:5]:
        print(f"\n  [{row['category']}]")
        print(f"  Q: {row['question']}")
        ans_preview = row['response'][:80] + "..." if len(row['response']) > 80 else row['response']
        print(f"  A: {ans_preview}")


if __name__ == "__main__":
    main()
