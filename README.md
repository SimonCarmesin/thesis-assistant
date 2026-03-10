# thesis-assistant

Local thesis assistant for a master thesis:
- Ingests thesis + code + tables into chunks
- Hybrid retrieval: SQLite FTS5 (keyword) + Chroma (semantic)
- Builds a context bundle for Cursor
- Optional: generates rewritten sections via OpenAI

## Setup (Windows / PowerShell)

### 1) Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 2) Install dependencies
pip install -r requirements.txt

## Prepare data
Put your files under:
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

## Generate rewrite (OpenAI)
Create a .env file in project root:
OPENAI_API_KEY=sk-...

Then run:
python tools\generate_openai.py 
  --task "Rewrite section ..." 
  --query "..." 
  --chapter_hint "..." 
  --code_root ".\data" 
  --code_query "McpEsClient"

Output is written to ./out/generated_*.md
