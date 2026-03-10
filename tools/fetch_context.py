import json, sqlite3, os, re
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
CHUNKS = INDEX / "chunks.jsonl"
DB = INDEX / "fts.sqlite"

PERSIST_DIR = str((INDEX / "chroma").resolve())
COLLECTION = "thesis"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEXT_EXT = {".java", ".py", ".js", ".ts", ".yml", ".yaml", ".json", ".properties", ".md", ".txt", ".xml", ".sql"}

def load_chunks():
    m = {}
    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            m[c["id"]] = c
    return m

def fts_ids(query: str, k: int = 10):
    safe = sanitize_fts_query(query)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM fts WHERE fts MATCH ? AND (type='thesis' OR type='pdf' OR type='table_summary') LIMIT ?;",
        (safe, k)
    )
    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]

def sanitize_fts_query(q: str) -> str:
    """
    Entfernt FTS5-Syntaxfallen wie ':' und macht aus freiem Text eine sichere AND-Suche.
    """
    q = q.replace(":", " ")          # wichtig: verhindert "no such column"
    q = q.replace('"', " ")          # Quotes nicht ungefiltert übernehmen
    q = re.sub(r"[^\wäöüÄÖÜß ]+", " ", q)  # Sonderzeichen -> Space (Unicode Wortzeichen bleiben)
    terms = [t.strip() for t in q.split() if t.strip()]
    # Optional: sehr kurze Tokens raus, und Query begrenzen
    terms = [t for t in terms if len(t) > 1][:14]
    # AND-Query: Space entspricht in FTS5 im Normalfall AND
    return " ".join(terms) if terms else q

def chroma_ids(query: str, k: int = 14):
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(COLLECTION)

    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["metadatas", "documents", "distances"],
        where={"type": {"$in": ["thesis", "pdf", "table_summary"]}}
    )
    return res["ids"][0]

def simple_code_search(code_root: Path, query: str, max_hits: int = 12):
    """
    Sehr simpel: sucht die Query (case-insensitive) in Text-Dateien und gibt Fundstellen.
    """
    q = query.strip()
    if not q:
        return ""
    q_re = re.compile(re.escape(q), re.IGNORECASE)
    hits = []
    for p in code_root.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in TEXT_EXT:
            continue
        # skip huge binaries
        if p.stat().st_size > 2_000_000:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), start=1):
            if q_re.search(line):
                hits.append(f"{p}:{i}: {line.strip()}")
                if len(hits) >= max_hits:
                    return "\n".join(hits)
    return "\n".join(hits)

def format_block(c):
    hp = " > ".join(c.get("heading_path", [])[:4])
    header = f"[{c['type']}] {c.get('section','')} | {hp} | {c['source_path']}"
    return header + "\n" + (c.get("text") or "").strip()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--chapter_hint", default="")
    ap.add_argument("--code_root", default=str((ROOT / "data").resolve()))
    ap.add_argument("--k_sem", type=int, default=14)
    ap.add_argument("--k_kw", type=int, default=10)
    ap.add_argument("--max_blocks", type=int, default=18)
    ap.add_argument("--code_query", default="")  # optional separate code query
    args = ap.parse_args()

    chunks = load_chunks()

    q = args.query if not args.chapter_hint else f"{args.query} {args.chapter_hint}"

    sem = chroma_ids(q, args.k_sem)
    kw = fts_ids(q, args.k_kw)

    # de-dup
    seen = set()
    ids = []
    for x in sem + kw:
        if x not in seen and x in chunks:
            ids.append(x); seen.add(x)
        if len(ids) >= args.max_blocks:
            break

    # code search
    code_root = Path(args.code_root)
    cq = args.code_query.strip() or args.query.split()[0]
    code_hits = simple_code_search(code_root, cq, max_hits=14)

    print("### CONTEXT BUNDLE (paste into Cursor)\n")
    print(f"USER_QUERY: {args.query}\n")
    if args.chapter_hint:
        print(f"CHAPTER_HINT: {args.chapter_hint}\n")

    print("## Retrieved thesis/code/data chunks\n")
    for cid in ids:
        print("-----")
        print(format_block(chunks[cid]))
        print()

    if code_hits.strip():
        print("## Code hits (simple search)\n")
        print(code_hits)
        print()

    print("### END CONTEXT\n")

if __name__ == "__main__":
    main()
