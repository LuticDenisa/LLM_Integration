#!/usr/bin/env python3
"""
Seed a ChromaDB collection from a Markdown file with blocks like:

## Title: 1984
<short description lines here>

Usage:
  python seed_chroma.py --md path/to/book_summaries.md --persist ./chroma_db --collection books_rag

Requirements:
  pip install chromadb openai tiktoken python-dotenv

Env:
  OPENAI_API_KEY must be set
"""

import os
import re
import argparse
from pathlib import Path

def parse_md(md_text: str):
    blocks = re.split(r"\n(?=##\s*Title:\s*)", md_text.strip())
    docs, metas, ids = [], [], []
    for i, blk in enumerate(blocks):
        m = re.match(r"##\s*Title:\s*(.+?)\s*\n(.*)", blk, flags=re.S)
        if not m:
            continue
        title = m.group(1).strip()
        short = m.group(2).strip()
        if not short:
            continue
        docs.append(short)
        metas.append({"title": title})
        safe_id = re.sub(r"\W+", "_", title)
        ids.append(f"doc_{i}_{safe_id}")
    return ids, docs, metas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="Path to Markdown file with book summaries")
    ap.add_argument("--persist", default="./chroma_db", help="Directory for Chroma persistence")
    ap.add_argument("--collection", default="books_rag", help="Chroma collection name")
    ap.add_argument("--test_query", default=None, help="Optional: run a sample query after seeding")
    args = ap.parse_args()

    # Read and parse MD
    md_path = Path(args.md)
    text = md_path.read_text(encoding="utf-8")
    ids, docs, metas = parse_md(text)
    if not ids:
        raise SystemExit("No documents parsed from Markdown. Check the format (## Title: ...)")

    # Setup Chroma with OpenAI embeddings
    import chromadb
    from chromadb.utils import embedding_functions

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Please export it or use a .env file.")

    client = chromadb.PersistentClient(path=args.persist)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )

    col = client.get_or_create_collection(name=args.collection, embedding_function=openai_ef)

    # Upsert
    print(f"Upserting {len(ids)} documents into collection '{args.collection}' at '{args.persist}' ...")
    col.upsert(documents=docs, metadatas=metas, ids=ids)
    print("Done.")

    # Optional test
    if args.test_query:
        print(f"\nTesting query: {args.test_query!r}")
        res = col.query(query_texts=[args.test_query], n_results=5)
        # Print top titles for a quick sanity check
        for rank, md in enumerate(res.get("metadatas", [[]])[0], start=1):
            print(f"{rank}. {md.get('title')}")

if __name__ == "__main__":
    main()
