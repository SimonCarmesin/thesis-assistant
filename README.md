# thesis-assistant

Local thesis assistant for a master thesis project.

What it does:
- Ingests thesis text, code, and evaluation artifacts into searchable chunks
- Hybrid retrieval: SQLite FTS5 (keyword) + Chroma (semantic embeddings)
- Creates context bundles you can paste into Cursor for grounded edits
- Optional generates rewritten sections via OpenAI with global summary + mini history + style rules

## Requirements
- Windows + PowerShell
- Python 3.11+

## Setup (Windows / PowerShell)

### 1) Create and activate venv
python -m venv .venv

.\.venv\Scripts\Activate.ps1

### 2) Install dependencies
pip install -r requirements.txt

## Prepare data
Put your files under data/:
- data/thesis/ (Masterarbeit.docx, PDFs)
- data/backend/, data/evaluation/, data/docker/, data/elasticsearch/
- data/tables/ (CSV/XLSX)
- data/diagrams/ (PNGs)

## Build indexes
python tools\ingest.py
python tools\build_fts.py
python tools\embed_chroma.py

## Retrieve context (paste into Cursor)
python tools\fetch_context.py --query "..." --chapter_hint "..." --code_root ".\data" --code_query "mcp"

## Global summary (recommended)
Create index/global_summary.md with motivation, research question, hypotheses, and short chapter overview.

## Generate rewrite (OpenAI)
Create .env in project root:

OPENAI_API_KEY=sk-...

Optional:

OPENAI_BASE_URL=https://api.openai.com/v1

ANONYMIZED_TELEMETRY=False

Run:
python tools\generate_openai.py 
--task "..." 
--query "..." 
--chapter_hint "..." 
--section "2.5" 
--code_root ".\data" 
--code_query "McpEsClient" 
--print_prompt

Outputs are written to out/ (generated text, prompts, logs, history).
