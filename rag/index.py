"""Эмбеддинги страниц -> data/page_vectors.npy (float16, по строкам data/pages.jsonl).

Из корня репозитория: python -m rag.index
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from rag.parse import DATA, OUT as PAGES

VECTORS = DATA / "page_vectors.npy"
EMB_URL = "https://embeddings.parsers360.ru:10443/embed"


def embed(text):
    for attempt in range(30):
        response = requests.post(
            EMB_URL,
            headers={"Authorization": "Bearer " + os.environ["EMB_KEY"]},
            json={"text": text[:20000]},
            timeout=60,
        )
        if response.status_code in (429, 502, 503):
            time.sleep(float(response.headers.get("Retry-After", 0)) + min(2 ** attempt, 30))
            continue
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)
    raise RuntimeError("Сервис эмбеддингов не ответил: %s" % response.status_code)


def doc_title(doc):
    kind = "промежуточная отчётность за 6 месяцев" if doc["doc_id"].endswith("_6m") else "годовая отчётность"
    return "%s, %s %s" % (doc["organization"], kind, doc["doc_id"].split("_")[1])


def main():
    docs = {d["doc_id"]: d for d in map(json.loads, (DATA / "documents.jsonl").open(encoding="utf-8"))}
    pages = [json.loads(line) for line in PAGES.open(encoding="utf-8")]
    texts = ["%s, стр. %d\n%s" % (doc_title(docs[p["doc_id"]]), p["page"], p["text"]) for p in pages]
    # при обрыве можно перезапустить, продолжит с места остановки
    partial = VECTORS.with_suffix(".partial.npy")
    vectors = list(np.load(partial)) if partial.exists() else []
    started = time.time()
    with ThreadPoolExecutor(2) as pool:
        for start in range(len(vectors), len(texts), 100):
            vectors += pool.map(embed, texts[start:start + 100])
            np.save(partial, np.stack(vectors).astype(np.float16))
            print(len(vectors), "из", len(texts), "за %.0f с" % (time.time() - started), flush=True)
    np.save(VECTORS, np.stack(vectors).astype(np.float16))
    partial.unlink()


if __name__ == "__main__":
    main()
