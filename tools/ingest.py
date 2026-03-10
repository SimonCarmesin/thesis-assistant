import os, re, json, hashlib
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from docx import Document
from pypdf import PdfReader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX = ROOT / "index"
INDEX.mkdir(exist_ok=True)

TEXT_EXT = {
    ".md", ".txt", ".json", ".yml", ".yaml", ".xml",
    ".java", ".js", ".ts", ".py", ".properties", ".gradle", ".sql",
}
TABLE_EXT = {".csv", ".xlsx", ".xls"}
DIAGRAM_EXT = {".png", ".jpg", ".jpeg", ".webp", ".svg"}

SKIP_DIRS = {".git", "node_modules", "target", "build", "dist", ".idea", ".vscode", "__pycache__"}

def stable_id(*parts: str) -> str:
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:16]

def clean_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, max_chars: int = 1700) -> List[str]:
    text = clean_ws(text)
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks

def parse_docx(path: Path) -> List[Dict[str, Any]]:
    doc = Document(str(path))
    out = []
    heading_path: List[str] = []
    last_section = ""
    buf = ""

    def flush():
        nonlocal buf, last_section
        if buf.strip():
            for i, ch in enumerate(chunk_text(buf, max_chars=1900)):
                out.append({
                    "id": stable_id(str(path), last_section, str(i), ch[:80]),
                    "type": "thesis",
                    "source_path": str(path),
                    "heading_path": heading_path.copy(),
                    "section": last_section,
                    "text": ch
                })
        buf = ""

    for p in doc.paragraphs:
        style = (p.style.name or "").lower()
        text = (p.text or "").strip()
        if not text:
            continue

        if style.startswith("heading"):
            flush()
            m = re.search(r"heading\s+(\d+)", style)
            level = int(m.group(1)) if m else 1
            while len(heading_path) >= level:
                heading_path.pop()
            heading_path.append(text)
            last_section = " / ".join(heading_path)
        else:
            buf += ("\n\n" + text) if buf else text

    flush()
    return out

def parse_pdf(path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pass
    text = clean_ws("\n".join(pages))
    out = []
    for i, ch in enumerate(chunk_text(text, max_chars=1900)):
        out.append({
            "id": stable_id(str(path), str(i), ch[:80]),
            "type": "pdf",
            "source_path": str(path),
            "heading_path": [],
            "section": path.name,
            "text": ch
        })
    return out

def parse_text_file(path: Path) -> List[Dict[str, Any]]:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    txt = clean_ws(txt)
    if not txt:
        return []
    out = []
    # größere chunks für Code
    max_chars = 2400 if path.suffix.lower() in {".java", ".py", ".js", ".ts", ".yml", ".yaml"} else 1900
    for i, ch in enumerate(chunk_text(txt, max_chars=max_chars)):
        out.append({
            "id": stable_id(str(path), str(i), ch[:80]),
            "type": "code" if path.suffix.lower() in {".java", ".py", ".js", ".ts"} else "text",
            "source_path": str(path),
            "heading_path": [],
            "section": path.name,
            "text": ch
        })
    return out

def summarize_table(path: Path) -> List[Dict[str, Any]]:
    out = []
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception:
        return out

    cols = list(df.columns)
    head = df.head(10).to_string(index=False)
    try:
        desc = df.describe(include="all").transpose().head(25).to_string()
    except Exception:
        desc = ""

    txt = f"""TABLE FILE: {path.name}
PATH: {path}
SHAPE: {df.shape}
COLUMNS: {cols}

HEAD(10):
{head}

DESCRIBE(Top 25):
{desc}
"""
    for i, ch in enumerate(chunk_text(txt, max_chars=1900)):
        out.append({
            "id": stable_id(str(path), "table", str(i), ch[:80]),
            "type": "table_summary",
            "source_path": str(path),
            "heading_path": [],
            "section": path.name,
            "text": ch
        })
    return out

def summarize_diagram(path: Path) -> List[Dict[str, Any]]:
    # Wir OCRen nicht – nur als "Artefakt-Verweis" im Kontext
    txt = f"DIAGRAM FILE: {path.name}\nPATH: {path}\nNOTE: Use file as reference; content not parsed."
    return [{
        "id": stable_id(str(path), "diagram"),
        "type": "diagram_ref",
        "source_path": str(path),
        "heading_path": [],
        "section": path.name,
        "text": txt
    }]

def walk_files(base: Path) -> List[Path]:
    paths = []
    for root, dirs, files in os.walk(base):
        rootp = Path(root)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            paths.append(rootp / f)
    return paths

def main():
    if not DATA.exists():
        raise SystemExit(f"DATA folder not found: {DATA} (put your folders under ./data/...)")

    chunks: List[Dict[str, Any]] = []
    all_paths = walk_files(DATA)

    for p in tqdm(all_paths, desc="Ingest"):
        suf = p.suffix.lower()
        if suf == ".docx":
            chunks.extend(parse_docx(p))
        elif suf == ".pdf":
            chunks.extend(parse_pdf(p))
        elif suf in TEXT_EXT:
            chunks.extend(parse_text_file(p))
        elif suf in TABLE_EXT:
            chunks.extend(summarize_table(p))
        elif suf in DIAGRAM_EXT:
            chunks.extend(summarize_diagram(p))

    out_path = INDEX / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        kept = 0
        for c in chunks:
            c["text"] = clean_ws(c["text"])
            if c["text"]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Wrote {kept} chunks -> {out_path}")

if __name__ == "__main__":
    main()