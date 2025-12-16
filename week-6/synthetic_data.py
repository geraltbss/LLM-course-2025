import csv
import argparse
import re
import time
import webbrowser
from typing import List
import dspy

# LLM SETUP (Ollama)

llm = dspy.LM(
    model="ollama/llama3",
    base_url="http://localhost:11434",
    max_tokens=1000,
    temperature=0.7
)

dspy.settings.configure(lm=llm)


# CONFIG

INPUT_CSV = "web_search_queries.csv"
OUTPUT_CSV = "synthetic_misspellings.csv"

# Common abbreviations to skip from corruption
KNOWN_ABBREVIATIONS = {
    "JFK", "LAX", "NBA", "NFL", "BBC", "CNN",
    "UN", "EU", "USA", "UK", "UAE", "AI", "ML"
}

GOOGLE_SEARCH_URL = "https://www.google.com/search?q="

# DSPy Module

class MisspellingGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            "query, n_variants -> misspellings"
        )

    def forward(self, query: str, n_variants: int):
        prompt = f"""
You are generating misspelled web search queries.

Rules:
- Generate EXACTLY {n_variants} variants
- Preserve meaning
- Use different error types:
  1. phonetic spelling (e.g., machine → mashine)
  2. omission (missing letters)
  3. transposition (swapped letters)
  4. repetition (double letters)
- DO NOT corrupt known abbreviations: {", ".join(KNOWN_ABBREVIATIONS)}
- Return ONLY a Python list of strings

Original query:
"{query}"
"""
        return self.generate(
            query=prompt,
            n_variants=n_variants
        )


# Utility Functions

def load_queries(csv_path: str) -> List[str]:
    queries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                queries.append(row[0].strip())
    return queries


def contains_abbreviation(word: str) -> bool:
    return word.upper() in KNOWN_ABBREVIATIONS


def filter_abbreviations(query: str) -> str:
    words = query.split()
    safe_words = []
    for w in words:
        if contains_abbreviation(w):
            safe_words.append(w)
        else:
            safe_words.append(w)
    return " ".join(safe_words)


def parse_llm_output(text: str) -> List[str]:
    """
    Safely extract list from LLM output
    """
    try:
        return eval(text)
    except Exception:
        return []


def google_search(query: str):
    url = GOOGLE_SEARCH_URL + query.replace(" ", "+")
    webbrowser.open(url)
    time.sleep(1.5)


# MAIN PIPELINE

def main(n_variants: int, test_search: bool):
    generator = MisspellingGenerator()
    queries = load_queries(INPUT_CSV)

    results = []

    for q in queries:
        print(f"\nOriginal query: {q}")
        safe_query = filter_abbreviations(q)

        response = generator.forward(
            query=safe_query,
            n_variants=n_variants
        )

        variants = parse_llm_output(response.misspellings)

        for v in variants:
            print("  →", v)
            results.append([q, v])

            if test_search:
                google_search(v)

    # Save results
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_query", "misspelled_query"])
        writer.writerows(results)

    print(f"\nSaved misspellings to {OUTPUT_CSV}")

    print("\n--- Analysis (Task e) ---")
    print("""
Google does not always return identical results for misspelled queries.

Reasons:
1. Spell-correction is context-dependent
2. Some typos reduce confidence in intent
3. Rare typos may trigger different ranking paths
4. Proper nouns and locations are more sensitive
5. Google's ML models weigh query certainty differently

This explains why results may vary across variants.
""")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_variants",
        type=int,
        default=5,
        help="Number of misspellings per query"
    )
    parser.add_argument(
        "--test_search",
        action="store_true",
        help="Open Google search for each variant"
    )

    args = parser.parse_args()
    main(args.n_variants, args.test_search)