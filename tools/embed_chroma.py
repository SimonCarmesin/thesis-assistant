import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
CHUNKS = INDEX / "chunks.jsonl"

PERSIST_DIR = str((INDEX / "chroma").resolve())
COLLECTION = "thesis"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    INDEX.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    # recreate collection (clean rebuild)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION)

    model = SentenceTransformer(MODEL_NAME)

    ids, docs, metas = [], [], []
    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            ids.append(c["id"])
            docs.append(txt)
            metas.append({
                "type": c.get("type",""),
                "section": c.get("section",""),
                "source_path": c.get("source_path",""),
                "heading_path": " > ".join(c.get("heading_path", [])[:6]),
            })

    # batch add with embeddings
    for i in tqdm(range(0, len(ids), 64), desc="Embedding->Chroma"):
        b_ids = ids[i:i+64]
        b_docs = docs[i:i+64]
        b_metas = metas[i:i+64]
        embs = model.encode(b_docs, normalize_embeddings=True, show_progress_bar=False).tolist()
        col.add(ids=b_ids, documents=b_docs, metadatas=b_metas, embeddings=embs)

    print(f"Stored {len(ids)} chunks in Chroma at {PERSIST_DIR} (collection={COLLECTION})")

if __name__ == "__main__":
    main()