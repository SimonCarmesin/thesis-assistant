import os, json, sqlite3, re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
CHUNKS = INDEX / "chunks.jsonl"
DB = INDEX / "fts.sqlite"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

PERSIST_DIR = str((INDEX / "chroma").resolve())
COLLECTION = "thesis"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks():
    m = {}
    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            m[c["id"]] = c
    return m

def sanitize_fts_query(q: str) -> str:
    q = q.replace(":", " ")
    q = q.replace('"', " ")
    q = re.sub(r"[^\wäöüÄÖÜß ]+", " ", q)
    terms = [t.strip() for t in q.split() if t.strip()]
    terms = [t for t in terms if len(t) > 1][:14]
    return " ".join(terms) if terms else q

def fts_ids(query: str, k: int = 10):
    safe = sanitize_fts_query(query)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM fts WHERE fts MATCH ? AND (type='thesis' OR type='pdf' OR type='table_summary') LIMIT ?;",
        (safe, k),
    )
    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]

def chroma_ids(query: str, k: int = 14):
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(COLLECTION)

    model = SentenceTransformer(EMB_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["metadatas", "documents", "distances"],
        where={"type": {"$in": ["thesis", "pdf", "table_summary"]}},
    )
    return res["ids"][0]

def simple_code_search(code_root: Path, query: str, max_hits: int = 14):
    TEXT_EXT = {".java", ".py", ".js", ".ts", ".yml", ".yaml", ".json", ".properties", ".md", ".txt"}
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

def build_context(query: str, chapter_hint: str, code_root: Path, code_query: str, k_sem=14, k_kw=10, max_blocks=18):
    chunks = load_chunks()
    q = query if not chapter_hint else f"{query} {chapter_hint}"

    sem = chroma_ids(q, k_sem)
    kw = fts_ids(q, k_kw)

    seen, ids = set(), []
    for x in sem + kw:
        if x not in seen and x in chunks:
            ids.append(x); seen.add(x)
        if len(ids) >= max_blocks:
            break

    context_blocks = [format_block(chunks[cid]) for cid in ids]
    code_hits = simple_code_search(code_root, code_query or query.split()[0], max_hits=16)

    return "\n\n-----\n\n".join(context_blocks), code_hits

def make_prompt(task: str, context_text: str, code_hits: str):
    return f"""Du bist ein wissenschaftlicher Schreibassistent für eine Masterarbeit. 
Du darfst NUR Aussagen machen, die durch den Kontext gedeckt sind. Keine neuen Fakten erfinden.

AUFGABE:
{task}

KONTEXT (Auszüge aus Thesis, Tabellen-Summaries, PDFs):
{context_text}

CODE-HINWEISE (Fundstellen, um Implementierung zu verifizieren):
{code_hits if code_hits.strip() else "(keine)"}

ANWEISUNGEN:
- Schreibe in sachlichem, wissenschaftlichem Stil auf Deutsch.
- Integriere Agentic RAG sauber: Definition, Abgrenzung, Rolle von Tool-Use und warum MCP dazu passt.
- Wenn etwas im Kontext nicht belegt ist, markiere es als 'nicht belegt' statt zu raten.
- Output: Nur der überarbeitete Text (kein Meta-Gelaber).
"""

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Was soll geschrieben werden (z.B. 'Schreibe Kapitel 2.5 neu ...')?")
    ap.add_argument("--query", required=True, help="Retrieval query")
    ap.add_argument("--chapter_hint", default="")
    ap.add_argument("--code_root", default=str((ROOT/"data").resolve()))
    ap.add_argument("--code_query", default="mcp")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY fehlt in .env")

    context_text, code_hits = build_context(
        query=args.query,
        chapter_hint=args.chapter_hint,
        code_root=Path(args.code_root),
        code_query=args.code_query,
    )

    prompt = make_prompt(args.task, context_text, code_hits)

    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "Du bist ein präziser, wissenschaftlicher Schreibassistent."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT / f"generated_{stamp}.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()