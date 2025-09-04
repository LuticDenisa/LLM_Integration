# CLI simplu: intreaba utilizatorul, ruleaza recommend(), afisează frumos

from __future__ import annotations
import sys
from llm import recommend

def main():
    print("Smart Librarian (CLI). Scrie o întrebare sau 'exit':")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q or q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        out = recommend(q, k=5)
        print("\n=== Recomandare ===")
        print(out["choice"]["title"])
        print("Motiv:", out["choice"]["reason"])

        full = out["full_summary"]
        if full.get("found"):
            if full.get("author"):
                print(f"Autor: {full['author']}")
            print("\nRezumat complet:\n" + full.get("summary", ""))
        else:
            print("\nRezumat complet indisponibil in JSON.")

        print("\n(Candidati RAG):", ", ".join([c["title"] for c in out["candidates"]]))
        print("\n-------------------------------------\n")

if __name__ == "__main__":
    sys.exit(main())
