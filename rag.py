# conectare la Chroma + funcție retrieve(query)
# foloseste colectia: books_rag din ./chroma_db (creata de seed_chroma.py)

from __future__ import annotations
from typing import List, Dict, Any
import os
import chromadb
from chromadb.utils import embedding_functions


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "books_rag"


def get_collection():
    """
    returneaza colectia Chroma creată cu seed_chroma.py.

    IMPORTANT: folosim ACELASI embedding function ca la seed (OpenAI text-embedding-3-small),
    altfel apare mismatch de dimensiune
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    col = client.get_collection(
        COLLECTION_NAME,
        embedding_function=openai_ef,  # <-- cheia!
    )
    return col


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    interogheaza semantic colecyia si intoarce top-k rezultate
    ca lista de dict-uri {title, short_summary, id}.

    Exemplu:
        results = retrieve("libertate și control social", k=3)
    """
    col = get_collection()
    res = col.query(query_texts=[query], n_results=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    out: List[Dict[str, Any]] = []
    for doc, meta, _id in zip(docs, metas, ids):
        out.append({
            "id": _id,
            "title": meta.get("title"),
            "short_summary": doc,
        })
    return out


if __name__ == "__main__":
    # pt un test manual mic: ruleaza `python rag.py` si vezi top-3 pentru o intrebare
    q = "libertate și control social"
    print(f"Query: {q!r}")
    for i, r in enumerate(retrieve(q, k=3), start=1):
        print(f"{i}. {r['title']}\n   {r['short_summary'][:160]}...")

    # alte query-uri de test:
    # python -c "import rag; print([x['title'] for x in rag.retrieve('prietenie și magie', k=5)])"
    # python -c "import rag; print([x['title'] for x in rag.retrieve('război și curaj', k=5)])"
