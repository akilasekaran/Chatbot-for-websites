# AI RAG Project

This repository contains a refactored retrieval-augmented generation (RAG) example.
The main script is `web_scraper_rag.py` which:
- Loads and parses a webpage.
- Splits the text into chunks.
- Indexes the chunks in a Chroma vectorstore using OpenAI embeddings.
- Runs a retrieval + LLM chain to answer a user query.

## Quick setup

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Recommended environment variables

- `OPENAI_API_KEY` — Your OpenAI API key. The script will prompt for it if not set.
- `USER_AGENT` — Optional. Set a custom user-agent string to identify your web requests (polite crawling).

You can set them in your shell like:

```bash
export OPENAI_API_KEY="sk-..."
export USER_AGENT="my-rag-bot/1.0 (+https://example.com)"
```

## Run the script

Basic usage:

```bash
python web_scraper_rag.py --query "What is Task Decomposition?"
```

Specify a different URL (comma-separated list) and enable history-aware retrieval:

```bash
python web_scraper_rag.py --url "https://example.com/article" --history --query "Summarize the article"
```

## Notes and suggestions

- The first run builds and embeds the vectorstore which may take time and consume OpenAI quota. Consider persisting the vectorstore to disk for faster subsequent runs.
- If you hit OpenAI rate limits or quota errors, try again later or use a smaller dataset (or mocked LLM/embeddings for testing).
- To speed up development iteratively, consider adding a `--limit N` flag (not included) to only index N chunks.

## Troubleshooting

- `openai.RateLimitError` / `insufficient_quota`: check your plan/billing and quota usage on platform.openai.com.
- `USER_AGENT environment variable not set`: not fatal, but set it to be polite to web servers.

If you want, I can add vectorstore persistence (save/load), a `--limit` flag, or a small test harness next.
