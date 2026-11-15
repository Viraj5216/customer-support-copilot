# Customer Support Resolution Copilot

An AI-powered customer support application that answers order questions and resolves customer issues using authenticated order data and company policies.

## Features

- Customer authentication with Argon2-hashed passwords
- Customer-specific order access and ownership checks
- Natural-language order lookup, filtering, and sorting
- Company policy retrieval using ChromaDB and semantic search
- Responses grounded in order and policy data
- LangGraph workflow and Google Gemini language models
- Optional Langfuse tracing and observability

## Requirements

- Python 3.11 or newer
- A Google Gemini API key
- Internet access for Gemini requests and the first policy-indexing run (which downloads the sentence-transformer model)

## Install and run

Clone the repository (replace the placeholder with the GitHub URL after creating the repository):

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer-support-copilot
```

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create your local environment file and set `GEMINI_API_KEY` in `.env`:

```bash
cp .env.example .env
```

Create the local demo users and sample orders, then build the local policy index:

```bash
python -m scripts.seed_demo_users
python -m tools.indexer
```

The seed command writes generated demo login credentials to `data/demo_credentials.txt`. Keep that local file private; it is excluded from Git. Start the app with:

```bash
streamlit run app.py
```

## Optional Langfuse tracing

To enable tracing, set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`. `LANGFUSE_BASE_URL` defaults to the Langfuse US Cloud endpoint and can be changed for another deployment. Leave the keys blank to run without tracing.

## Local data

The application creates its SQLite database, demo credentials, and Chroma vector store locally. These files are ignored by Git. Run the seed and index commands after cloning to create them on your machine.
