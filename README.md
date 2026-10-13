# Sheet Q&A Service

A small Flask service that lets you ask plain-English questions about data sitting in a Google Sheet and get a matched answer back.

## How it actually works

This is embedding-based lookup over a spreadsheet, not retrieval-augmented generation with an LLM. There's no generative model involved and no vector database. When a question comes in:

1. The current sheet is pulled live through a Google Apps Script web app URL.
2. `sentence-transformers` (`all-MiniLM-L6-v2`) encodes the question and every column header, and cosine similarity picks the column the question is most likely asking about.
3. The same embedding comparison runs against the values in the other columns to find the row the question is probably about.
4. The cell at that row/column intersection is returned as the answer, wrapped in a template sentence.

So "what's the status of the Austin show" works by finding the column whose name is closest to "status" and the row whose value is closest to "Austin show," then handing back whatever text is in that cell. It's a reasonable trick for small, simple sheets, and it's genuinely useful when the schema is clean, but it has real limits: a skewed or messy sheet gives bad matches, there's no synthesis across multiple rows, and the thresholds (`COLUMN_THRESHOLD`, `ROW_THRESHOLD` in `sheet_qa.py`) need tuning per sheet.

I built this during an internship at Expo Convention Contractors (Miami) to let non-technical staff query event data without opening a spreadsheet. It worked for that specific, narrow use case.

## Stack

Python, Flask, sentence-transformers, pandas, a static HTML/JS front end.

## Running it

```
pip install -r requirements.txt
export WEB_APP_URL=<your Google Apps Script web app URL>
python app.py
```

Ask a question at `POST /ask` with `{"question": "..."}`, or open the static front end at `/`.

## What I'd do differently now

Swap the column/row-matching heuristic for a proper RAG pipeline (real retrieval over chunked, embedded documents, with an LLM generating the final answer instead of returning a raw cell) if the sheet schema were less predictable or the questions needed synthesis across rows rather than a single lookup.
