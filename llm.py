# RAG -> LLM alege 1 title -> tool (full summary) -> print result

from __future__ import annotations
from typing import List, Dict, Any
import os
import json

from openai import OpenAI
import rag
from tools import get_summary_by_title

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY must be set in your environment"

client = OpenAI(api_key=OPENAI_API_KEY)

# prompt pt llm
SYSTEM = """You are an assistant that recommends ONE SINGLE book from the given candidates.
Return STRICTLY JSON in the form:
{"title": "...", "reason": "a short sentence in English"}
If you are not sure, choose the thematically closest title.
"""

def choose_title_with_llm(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    ofera top candidați (title + short_summary) și cere JSON {"title": "...", "reason": "..."}
    """
    if not candidates:
        return {"title": "", "reason": "No candidates found."}

    # construim promptul cu toti candidatii
    lines = ["CANDIDATES:"]
    for c in candidates:
        lines.append(f"- {c['title']}: {c['short_summary']}")
    lines.append(f"\nQUESTION: {query}")
    content = "\n".join(lines)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    msg = resp.choices[0].message.content
    try:
        data = json.loads(msg)
        return {"title": data.get("title", ""), "reason": data.get("reason", "")}
    except Exception:
        # fallback defensiv
        return {"title": candidates[0]["title"], "reason": "Chose the closest match by relevance."}

def recommend(query: str, k: int = 5) -> Dict[str, Any]:
    """
    1) RAG: top-k candidates
    2) LLM: choose 1 title
    3) tool: full summary + author
    """
    candidates = rag.retrieve(query, k=k)
    pick = choose_title_with_llm(query, candidates)
    full = get_summary_by_title(pick["title"])
    return {
        "query": query,
        "choice": pick,
        "full_summary": full,
        "candidates": [{"title": c["title"]} for c in candidates],
    }

if __name__ == "__main__":
    # mic demo
    q = "I want a book about freedom and social control"
    out = recommend(q, k=5)
    print("Question:", out["query"])
    print("Recommendation:", out["choice"]["title"])
    print("Reason:", out["choice"]["reason"])
    if out["full_summary"]["found"]:
        print("Author:", out["full_summary"]["author"])
        print("Full summary:")
        print(out["full_summary"]["summary"])
    else:
        print("Full summary not found in JSON — fallback on RAG short description.")
