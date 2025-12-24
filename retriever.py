from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document

from config import settings, get_logger

log = get_logger("retriever")
#vec search

def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

class NumpyRetriever:
    def __init__(self, embeddings: np.ndarray, documents: List[Document]):
        self._emb = _normalize(embeddings.astype(np.float32))
        self._docs = documents

    def similarity_search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Document, float]]:
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = (self._emb @ q)
        idx = np.argsort(-sims)[:max(1, k)]
        return [(self._docs[i], float(sims[i])) for i in idx]

def load_retriever() -> NumpyRetriever:
    index_dir = Path(settings.index_dir)
    emb_path = index_dir / "embeddings.npy"
    docs_path = index_dir / "docs.jsonl"
    if not emb_path.exists() or not docs_path.exists():
        raise RuntimeError("Индекс не найден. Сначала запустите: python ingest.py")

    emb = np.load(emb_path)
    docs: List[Document] = []
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Document(page_content=rec["page_content"], metadata=rec.get("metadata") or {}))

    log.info(f"Loaded index: docs={len(docs)} emb={emb.shape}")
    return NumpyRetriever(embeddings=emb, documents=docs)
