# tool ul local care citeste JSON ul cu (title, author, summary)
# și expune get_summary_by_title(title)

from __future__ import annotations
from typing import Optional, Dict, Any
import json
from pathlib import Path

JSON_PATH = Path("data/book_summaries.json")

# json ul e incarca o singura data la import
with JSON_PATH.open("r", encoding="utf-8") as f:
    BOOK_MAP: Dict[str, Dict[str, Any]] = json.load(f)

def get_summary_by_title(title: str) -> Dict[str, Any]:
    """
    returneaza dict cu {found, title, author, summary}
    found=False daca nu exista titlul exact în JSON
    """
    entry = BOOK_MAP.get(title)
    if not entry:
        return {"found": False, "title": title, "author": None, "summary": ""}
    return {
        "found": True,
        "title": title,
        "author": entry.get("author"),
        "summary": entry.get("summary"),
    }

if __name__ == "__main__":
    # test:
    print(get_summary_by_title("1984"))
