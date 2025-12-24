from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import settings, get_logger

log = get_logger("ingest")
#разбитие на чанки
def load_kb_docs(kb_dir: str):
    kb_path = Path(kb_dir)
    docs = []
    for p in kb_path.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in [".txt", ".md"]:
            continue
        try:
            docs += TextLoader(str(p), encoding="utf-8").load()
        except Exception as e:
            log.warning(f"Skip {p.name}: {e}")
    return docs
#эмбеддинги через OpenAI embeddings
def build_index():
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY пустой. Создайте .env и заполните ключ.")

    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # clean old index files
    for child in index_dir.glob("*"):
        if child.is_file():
            child.unlink()

    log.info(f"Using OpenAI embeddings: {settings.embed_model}")
    log.info("Loading documents from KB...")
    docs = load_kb_docs(settings.kb_dir)
    if not docs:
        raise RuntimeError("KB пустая: добавьте .md/.txt файлы в папку kb/")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    log.info(f"Chunks: {len(chunks)}. Computing embeddings...")

    emb = OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.embed_model)

    texts = [c.page_content for c in chunks]
    vectors = emb.embed_documents(texts)
    vecs = np.array(vectors, dtype=np.float32)
    np.save(index_dir / "embeddings.npy", vecs)

    with (index_dir / "docs.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            rec = {"page_content": c.page_content, "metadata": c.metadata or {}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"Saved index to: {index_dir}")

if __name__ == "__main__":
    build_index()
