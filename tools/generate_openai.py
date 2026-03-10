import os
import json
import sqlite3
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# Best-effort: disable Chroma telemetry noise
os.environ["ANONYMIZED_TELEMETRY"] = os.environ.get("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
CHUNKS = INDEX / "chunks.jsonl"
DB = INDEX / "fts.sqlite"

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

PROMPT_DIR = OUT / "prompts"
PROMPT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUT / "run.log"

PERSIST_DIR = str((INDEX / "chroma").resolve())
COLLECTION = "thesis"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FORBIDDEN_CHARS = [":", ";", "–", "—"]


# ----------------------------
# Logging
# ----------------------------
def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("thesis_assistant")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ----------------------------
# History
# ----------------------------
def read_history(hist_file: Path, n: int = 3) -> List[Dict[str, Any]]:
    if not hist_file.exists():
        return []
    items = []
    with hist_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items[-n:]


def append_history(hist_file: Path, entry: Dict[str, Any]) -> None:
    hist_file.parent.mkdir(exist_ok=True)
    with hist_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def summarize_for_history(text: str, max_chars: int = 450) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t[:max_chars] + ("…" if len(t) > max_chars else "")


# ----------------------------
# Global summary
# ----------------------------
def load_global_summary(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


# ----------------------------
# Chunks
# ----------------------------
def load_chunks() -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            m[c["id"]] = c
    return m


def chunk_matches_section(c: Dict[str, Any], section: str) -> bool:
    """
    Heuristik: section string in 'section' oder in heading_path.
    Beispiel: section="2.5" matcht "2 Theoretische... / 2.5 Retrieval..." usw.
    """
    if not section:
        return True
    sec = section.strip().lower()
    s = (c.get("section") or "").lower()
    hp = " > ".join(c.get("heading_path", [])).lower()
    return (sec in s) or (sec in hp)


# ----------------------------
# Retrieval
# ----------------------------
def sanitize_fts_query(q: str) -> str:
    q = q.replace(":", " ")
    q = q.replace('"', " ")
    q = re.sub(r"[^\wäöüÄÖÜß ]+", " ", q)
    terms = [t.strip() for t in q.split() if t.strip()]
    terms = [t for t in terms if len(t) > 1][:16]
    return " ".join(terms) if terms else q


def fts_ids(query: str, k: int, section_filter: str = "") -> List[str]:
    safe = sanitize_fts_query(query)
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # Filter only to thesis/pdf/table_summary; optionally narrow by section
    if section_filter.strip():
        like = f"%{section_filter.strip()}%"
        cur.execute(
            "SELECT id FROM fts "
            "WHERE fts MATCH ? "
            "AND (type='thesis' OR type='pdf' OR type='table_summary') "
            "AND section LIKE ? "
            "LIMIT ?;",
            (safe, like, k),
        )
    else:
        cur.execute(
            "SELECT id FROM fts "
            "WHERE fts MATCH ? "
            "AND (type='thesis' OR type='pdf' OR type='table_summary') "
            "LIMIT ?;",
            (safe, k),
        )

    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]


def chroma_ids(query: str, k: int, section_filter: str = "") -> List[str]:
    """
    Chroma filter unterstützt kein 'contains' auf strings zuverlässig in allen Versionen.
    Daher: Overfetch, dann local filtern über chunks meta.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(COLLECTION)

    model = SentenceTransformer(EMB_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()[0]

    overfetch = max(k * 4, 20)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=overfetch,
        include=["metadatas", "documents", "distances"],
        where={"type": {"$in": ["thesis", "pdf", "table_summary"]}},
    )
    ids = res["ids"][0]

    if not section_filter.strip():
        return ids[:k]

    # filter by section using our chunks.jsonl metadata
    chunks = load_chunks()
    filtered = [cid for cid in ids if cid in chunks and chunk_matches_section(chunks[cid], section_filter)]
    if len(filtered) >= k:
        return filtered[:k]

    # fallback: return what we have + remaining unfiltered
    out = filtered[:]
    for cid in ids:
        if cid not in out:
            out.append(cid)
        if len(out) >= k:
            break
    return out


def simple_code_search(code_root: Path, query: str, max_hits: int = 18) -> str:
    TEXT_EXT = {".java", ".py", ".js", ".ts", ".yml", ".yaml", ".json", ".properties", ".md", ".txt"}
    q = query.strip()
    if not q:
        return ""
    q_re = re.compile(re.escape(q), re.IGNORECASE)
    hits: List[str] = []
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


def format_block(c: Dict[str, Any]) -> str:
    hp = " > ".join(c.get("heading_path", [])[:4])
    header = f"[{c['type']}] {c.get('section','')} | {hp} | {c['source_path']}"
    return header + "\n" + (c.get("text") or "").strip()


def build_context(
    query: str,
    chapter_hint: str,
    code_root: Path,
    code_query: str,
    section_filter: str = "",
    k_sem: int = 14,
    k_kw: int = 10,
    max_blocks: int = 18,
) -> Tuple[str, str, List[str]]:
    """
    Returns:
      context_text, code_hits, used_chunk_ids
    """
    chunks = load_chunks()
    q = query if not chapter_hint else f"{query} {chapter_hint}"

    sem = chroma_ids(q, k_sem, section_filter=section_filter)
    kw = fts_ids(q, k_kw, section_filter=section_filter)

    seen, ids = set(), []
    for x in sem + kw:
        if x not in seen and x in chunks:
            ids.append(x)
            seen.add(x)
        if len(ids) >= max_blocks:
            break

    context_blocks = [format_block(chunks[cid]) for cid in ids]
    code_hits = simple_code_search(code_root, code_query or query.split()[0], max_hits=18)

    return "\n\n-----\n\n".join(context_blocks), code_hits, ids


# ----------------------------
# Output validation + repair
# ----------------------------
def validate_output(text: str) -> List[str]:
    issues: List[str] = []
    for ch in FORBIDDEN_CHARS:
        if ch in text:
            issues.append(f"forbidden_char:{ch}")
    if re.search(r"\s-\s", text):
        issues.append("dash_separator_detected")
    if re.search(r"(?m)^\s*#{1,6}\s+", text):
        issues.append("markdown_headings_detected")
    return sorted(set(issues))


def make_repair_prompt(task: str, original: str, issues: List[str]) -> str:
    issues_str = ", ".join(issues)
    return f"""Du bist ein Lektor für eine deutschsprachige Masterarbeit. Du überarbeitest einen Textabschnitt so, dass er sich nahtlos in die bestehende Arbeit einfügt.

STIL UND ROTER FADEN
- Bleibe konsequent im Stil der bestehenden Arbeit, sachlich und wissenschaftlich, aber natürlich formuliert.
- Achte darauf, dass der Text den roten Faden stärkt und den Abschnitt sinnvoll ersetzt.
- Querverweise auf andere Kapitel sind erwünscht, z.B. (Kap. 2.6.4), (Kap. 5.5), (Kap. 6.1).

ZITIERREGELN
- Vorhandene Quellenangaben im Format [10] dürfen NICHT verändert werden.
- Wenn du neue Literatur einführen möchtest, dann:
  1) Markiere die Stelle im Text mit [NEU1], [NEU2].
  2) Füge am Ende einen Abschnitt "NEUE QUELLEN" an und liste dort jede neue Quelle vollständig im Stil:
     [NEU1] Autor(en), Titel, Venue/Conference/Journal, Jahr. [Online]. Available: URL
  3) Keine URLs im Fließtext, nur in "NEUE QUELLEN".

FORMALREGELN
- Keine Doppelpunkte und keine Semikolons.
- Kein Gedankenstrich und kein Bindestrich als Satztrenner, also weder '–' noch '—' und nicht ' - '.
- Aufzählungen sind erlaubt.
- Keine Markdown Überschriften.

KONKRETE PROBLEME
{issues_str}

AUFGABE
{task}

TEXT
{original}

AUSGABE
- Gib nur den überarbeiteten Text aus.
- Falls du neue Quellen nutzt, hänge "NEUE QUELLEN" an.
"""


# ----------------------------
# Main prompt (strong single pass)
# ----------------------------
def make_prompt(
    task: str,
    global_summary: str,
    history_items: List[Dict[str, Any]],
    section_filter: str,
    context_text: str,
    code_hits: str,
) -> str:
    hist_block = ""
    if history_items:
        lines = []
        for i, h in enumerate(history_items, start=1):
            stamp = h.get("timestamp", "")
            lines.append(f"{i}) {stamp}")
            lines.append(f"   task={h.get('task','')}")
            if h.get("query"):
                lines.append(f"   query={h.get('query')}")
            if h.get("section"):
                lines.append(f"   section={h.get('section')}")
            if h.get("summary"):
                lines.append(f"   last_output_summary={h.get('summary')}")
        hist_block = "\n".join(lines)

    section_line = f"Aktuell zu bearbeitender Abschnitt: {section_filter}" if section_filter.strip() else "Aktuell zu bearbeitender Abschnitt: (nicht gesetzt)"

    return f"""Du bist ein wissenschaftlicher Schreibassistent für eine Masterarbeit.

ZIEL
Du überarbeitest einen Abschnitt so, dass er sich stilistisch und inhaltlich nahtlos in die bestehende Arbeit einfügt.
Der rote Faden muss strikt auf Motivation, Forschungsfrage und Hypothesen ausgerichtet bleiben.
{section_line}

HARTE REGELN
- Nutze ausschließlich Informationen aus dem Kontext. Keine neuen Fakten erfinden.
- Stil wie in der Arbeit. Querverweise auf andere Kapitel sind erlaubt und erwünscht, z.B. (Kap. 2.6.4).
- Quellenangaben im Format [10] dürfen nicht verändert werden.
- Wenn du neue Quellen brauchst, verwende Platzhalter [NEU1], [NEU2] und füge am Ende einen Block "NEUE QUELLEN" an, vollständig formatiert.
- Keine Doppelpunkte und keine Semikolons.
- Kein Gedankenstrich und kein Bindestrich als Satztrenner, also weder '–' noch '—' und nicht ' - '.
- Aufzählungen sind erlaubt, aber keine Markdown Überschriften.
- Ausgabe nur als finaler Text, keine Meta Erklärungen.

GLOBAL SUMMARY
{global_summary if global_summary.strip() else "(keine globale Summary vorhanden)"}

MINI HISTORIE DER LETZTEN RUNS
{hist_block if hist_block.strip() else "(keine Historie vorhanden)"}

AUFGABE
{task}

KONTEXT
{context_text}

CODE HINWEISE
{code_hits if code_hits.strip() else "(keine)"}

AUSGABEFORMAT
- Fließtext mit Absätzen.
- Aufzählungen nur wenn sie wirklich helfen, kurz und präzise.
- Wenn neue Quellen genutzt werden, am Ende Block "NEUE QUELLEN".
"""


def write_debug_files(stamp: str, prompt: str, context_text: str, code_hits: str, used_ids: List[str], logger: logging.Logger):
    prompt_path = PROMPT_DIR / f"prompt_{stamp}.txt"
    ctx_path = PROMPT_DIR / f"context_{stamp}.txt"

    prompt_path.write_text(prompt, encoding="utf-8")
    ctx_path.write_text(
        "### USED CHUNK IDS\n"
        + "\n".join(used_ids)
        + "\n\n### CONTEXT\n"
        + context_text
        + "\n\n### CODE HITS\n"
        + (code_hits or "(keine)"),
        encoding="utf-8",
    )
    logger.info(f"Saved prompt: {prompt_path}")
    logger.info(f"Saved context: {ctx_path}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--chapter_hint", default="")
    ap.add_argument("--section", default="", help="Section focus, e.g. '2.5' or '5.6.5'")
    ap.add_argument("--code_root", default=str((ROOT / "data").resolve()))
    ap.add_argument("--code_query", default="mcp")
    ap.add_argument("--model", default="gpt-4o-mini")

    # history + global summary
    ap.add_argument("--history_file", default=str((OUT / "history.jsonl").resolve()))
    ap.add_argument("--history_n", type=int, default=3)
    ap.add_argument("--global_summary", default=str((INDEX / "global_summary.md").resolve()))

    # retrieval knobs
    ap.add_argument("--k_sem", type=int, default=14)
    ap.add_argument("--k_kw", type=int, default=10)
    ap.add_argument("--max_blocks", type=int, default=18)

    # repair
    ap.add_argument("--max_repairs", type=int, default=1)
    ap.add_argument("--no_repair", action="store_true")

    # logging
    ap.add_argument("--log_level", default="INFO")
    ap.add_argument("--print_prompt", action="store_true", help="Also print final prompt to console (can be long).")

    args = ap.parse_args()
    logger = setup_logger(args.log_level)

    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY fehlt in .env")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    global_summary = load_global_summary(Path(args.global_summary))
    hist_file = Path(args.history_file)
    history_items = read_history(hist_file, n=args.history_n)

    context_text, code_hits, used_ids = build_context(
        query=args.query,
        chapter_hint=args.chapter_hint,
        code_root=Path(args.code_root),
        code_query=args.code_query,
        section_filter=args.section,
        k_sem=args.k_sem,
        k_kw=args.k_kw,
        max_blocks=args.max_blocks,
    )

    prompt = make_prompt(
        task=args.task,
        global_summary=global_summary,
        history_items=history_items,
        section_filter=args.section,
        context_text=context_text,
        code_hits=code_hits,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_debug_files(stamp, prompt, context_text, code_hits, used_ids, logger)

    if args.print_prompt:
        print("\n===== FINAL PROMPT START =====\n")
        print(prompt)
        print("\n===== FINAL PROMPT END =====\n")

    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)

    logger.info(f"Calling model={args.model}")
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "Du bist ein präziser, wissenschaftlicher Schreibassistent."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()

    # Token usage logging (if provided by API)
    usage = getattr(resp, "usage", None)
    if usage:
        try:
            logger.info(f"Tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
        except Exception:
            logger.info(f"Usage present but could not parse: {usage}")

    # Optional repair only if needed
    if not args.no_repair:
        for _ in range(max(0, args.max_repairs)):
            issues = validate_output(text)
            hard = [
                x for x in issues
                if x.startswith("forbidden_char:")
                or x in {"dash_separator_detected", "markdown_headings_detected"}
            ]
            if not hard:
                break

            logger.info(f"Repair pass due to: {hard}")
            repair_prompt = make_repair_prompt(args.task, text, hard)
            rep = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "Du bist ein präziser wissenschaftlicher Lektor."},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0.2,
            )
            text = rep.choices[0].message.content.strip()

            rep_usage = getattr(rep, "usage", None)
            if rep_usage:
                try:
                    logger.info(f"Repair tokens: prompt={rep_usage.prompt_tokens} completion={rep_usage.completion_tokens} total={rep_usage.total_tokens}")
                except Exception:
                    pass

    out_path = OUT / f"generated_{stamp}.md"
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"Wrote output: {out_path}")

    # Save history for next runs
    append_history(hist_file, {
        "timestamp": stamp,
        "task": args.task,
        "query": args.query,
        "chapter_hint": args.chapter_hint,
        "section": args.section,
        "model": args.model,
        "summary": summarize_for_history(text),
        "output_file": str(out_path),
    })
    logger.info(f"History updated: {hist_file}")


if __name__ == "__main__":
    main()