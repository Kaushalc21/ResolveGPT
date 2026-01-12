import subprocess
from typing import List

def generate_final_answer(query: str, candidates: List[str]) -> str:
    context = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(candidates)
    )

    prompt = f"""
You are a senior support engineer.

User issue:
{query}

Possible resolutions:
{context}

Task:
- Choose the most relevant resolution
- If multiple apply, combine them
- Be concise and practical
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="ignore",
        capture_output=True
    )

    return result.stdout.strip()
