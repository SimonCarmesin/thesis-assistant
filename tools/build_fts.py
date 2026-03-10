import json, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
CHUNKS = INDEX / "chunks.jsonl"
DB = INDEX / "fts.sqlite"

def main():
    if DB.exists():
        DB.unlink()

    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("CREATE VIRTUAL TABLE fts USING fts5(id, type, section, source_path, text);")

    n = 0
    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cur.execute(
                "INSERT INTO fts(id, type, section, source_path, text) VALUES(?,?,?,?,?)",
                (c["id"], c["type"], c.get("section",""), c["source_path"], c["text"])
            )
            n += 1

    con.commit()
    con.close()
    print(f"Built FTS index with {n} rows -> {DB}")

if __name__ == "__main__":
    main()