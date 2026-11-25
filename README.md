# Movie KG RecSys

**Graph-Based Knowledge Representation for Explainable Movie Recommendations**

Users, movies, genres, and LLM-extracted *themes* (like "redemption" or "found family") live in a Neo4j knowledge graph. The recommender matches a user's theme preferences, and every recommendation is explained by the actual graph path from the user's liked movies to the recommended one.

> Status: Phases 0 and 1 done; Phase 2 mostly done (themes extracted by the LLM for all 2,426 movies of the MovieLens 32M sample and normalized to 84 themes; a few review items are still open). Sections marked "coming" are filled in as the project is built.

## Requirements

- Python 3.10-3.13, with 3.12 recommended. The pinned packages don't support Python 3.14 yet.
- A Docker runtime for Neo4j:
  - macOS: [OrbStack](https://orbstack.dev/) (light, free for personal use), [Docker Desktop](https://www.docker.com/products/docker-desktop/), or Colima (free, command line only; see [Colima on macOS](#colima-on-macos)).
  - Windows: Docker Desktop (WSL 2 backend).
  - Linux: Docker Engine with the Compose plugin.
  - No Docker? See [Neo4j without Docker](#neo4j-without-docker).
- A free TMDB API key (Phase 1) and an LLM API key with a little credit (Phase 2; NanoGPT by default, OpenRouter also works). See [API keys](#api-keys).

## Setup

### 1. Python environment

macOS / Linux:

```bash
cd movie-kg-recsys
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd movie-kg-recsys
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In Command Prompt, activate with `.venv\Scripts\activate.bat`. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Optional extras, needed from Phase 2 (theme normalization) and Phase 7 (embeddings):

```bash
pip install -r requirements-optional.txt
```

### 2. Settings file

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` and fill in the values. Set `NEO4J_PASSWORD` (8+ characters, no `$` or quotes) **before** starting Neo4j for the first time. `.env` is git-ignored; never commit it.

### 3. Start Neo4j

```bash
docker compose up -d
docker compose ps           # the neo4j service should be running
```

Open http://localhost:7474 and log in with user `neo4j` and your `NEO4J_PASSWORD` (connect URL `bolt://localhost:7687`). The first start downloads the image (a few hundred MB) and takes a minute or two.

Stop with `docker compose stop` and start again with `docker compose start`. The database lives in `neo4j_data/`.

#### Colima on macOS

This project was set up with Colima. One-time install:

```bash
brew install colima docker docker-compose docker-credential-helper
colima start --cpu 2 --memory 3 --disk 20 --vm-type vz
```

For `docker compose` (with a space) to work, `~/.docker/config.json` needs `"cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]`, and `"credsStore"` must name a helper that exists (`"osxkeychain"`).

After a reboot, run `colima start` before `docker compose up -d`. `colima stop` shuts the VM down and frees its memory.

### 4. Check the setup

```bash
python config.py
```

Expect every package marked `ok`, your settings shown as `set` (secret values are never printed), and `Neo4j connection OK`. Missing TMDB or LLM keys are fine until Phases 1 and 2.

## API keys

- **TMDB (free):** create an account at themoviedb.org, then go to Settings → API and request a key (developer, personal or educational use). Put the "API Key" (v3) in `TMDB_API_KEY`. The longer "API Read Access Token" works too.
- **LLM provider (NanoGPT by default):** create a key at nano-gpt.com, add a few dollars of credit, and put it in `LLM_API_KEY`. Put a model id from `https://nano-gpt.com/api/v1/models` in `LLM_MODEL`; this project uses `google/gemini-3.5-flash-lite`, about $0.30 / $2.50 per 1M tokens in / out, so extracting themes for about 2,400 movies costs well under a dollar (it cost $0.40 in total here). To use OpenRouter instead, set `LLM_BASE_URL=https://openrouter.ai/api/v1` (or keep the old `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` names). Free OpenRouter `:free` models are capped at 50 requests a day, which is too slow for theme extraction.

## Neo4j without Docker

Only `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in `.env` change.

- **Neo4j Desktop** (free app): create a local DBMS on version 5.x, set a password, and start it. `NEO4J_URI` stays `bolt://localhost:7687`.
- **Neo4j AuraDB Free** (cloud): create a free instance and save the generated credentials. Use its `neo4j+s://...databases.neo4j.io` URI. The free tier allows 200k nodes and 400k relationships; this project needs about 2k nodes and 75k relationships.

## Dataset

The project uses [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) (collected 10/2023), which `01_prepare_data.py` downloads into `data/raw/ml-32m/` (239 MB zip, about 900 MB unzipped; the first download can take a while). It keeps a demo-sized sample: the 2,500 most-rated movies, then 1,000 random users (seed 42), then movies with at least 10 ratings and users with at least 20 inside that sample. That gives about 2,430 movies, 1,000 users, and 126k ratings. Change the size with `--max-movies` and `--users` (set `config.MAX_MOVIES` and `config.SAMPLE_USERS` to change the defaults). Loading the data needs about 2 GB of free RAM. Earlier runs used the much smaller, older `ml-latest-small` (9/2018), which is why some old files in `results/` are named `small_min20` and `small_min10`.

## Project layout

```
movie-kg-recsys/
├── config.py                  # paths, constants, seed themes, .env loading; run it for a setup check
├── docker-compose.yml         # Neo4j Community
├── requirements.txt           # core packages (pinned)
├── requirements-optional.txt  # sentence-transformers, torch, pykeen
├── .env.example               # copy to .env and fill in
├── data/                      # raw/, cache/, processed/ (git-ignored)
├── src/                       # data, TMDB, LLM, themes, graph, recommender, explainer, evaluation
├── scripts/                   # numbered pipeline scripts 01-07, plus the optional GPU scripts
├── app/                       # Streamlit dashboard
├── results/                   # metrics CSVs, charts, survey data, screenshots (report figures)
└── tests/                     # smoke tests (no API calls)
```

## Run order

| Step | Command | Phase |
|---|---|---|
| Setup check | `python config.py` | 0 |
| Prepare data (downloads MovieLens 32M, 239 MB, if needed; samples it; about 10 s after the download) | `python scripts/01_prepare_data.py` | 1 |
| Fetch TMDB metadata (cached; about 7 min the first time, try `--limit 5` first) | `python scripts/02_fetch_tmdb.py` | 1 |
| Extract themes (try `--limit 10` first; it asks y/n before spending, and resumes if stopped) | `python scripts/03_extract_themes.py` | 2 |
| Normalize themes (install `requirements-optional.txt` for the embedding step) | `python scripts/04_normalize_themes.py` | 2 |
| Or run steps 3 and 4 unattended (extraction, a second pass, normalization; keeps the Mac awake; about 27 minutes for 1,129 movies) | `caffeinate -i bash scripts/overnight_extract.sh` | 2 |
| Build the graph | `python scripts/05_build_graph.py --mode full --wipe` | 3 (coming) |
| CLI recommendations | `python -m src.recommender --user 1` | 4 (coming) |
| Dashboard | `streamlit run app/streamlit_app.py` | 5 (coming) |
| Evaluation | `python scripts/06_evaluate_accuracy.py` and `python scripts/07_evaluate_explanations.py` | 6 (coming) |

## Troubleshooting

- **`docker: command not found`:** install a Docker runtime (see Requirements) and open it once so its command-line tools are set up.
- **`Cannot connect to the Docker daemon`:** the runtime isn't running. Run `colima start`, or open OrbStack / Docker Desktop.
- **`error getting credentials ... docker-credential-desktop`:** `~/.docker/config.json` is left over from an old Docker Desktop install. Install `docker-credential-helper` and set `"credsStore": "osxkeychain"`.
- **`Set NEO4J_PASSWORD in .env`:** run `docker compose` from this folder (next to `.env`) and make sure the password is filled in.
- **Login fails after changing `NEO4J_PASSWORD`:** Neo4j keeps the password from its first start. Run `docker compose down`, delete `neo4j_data/`, then `docker compose up -d`. This empties the database; the graph is rebuilt with one command from Phase 3 on.
- **Port 7474 or 7687 already in use:** another Neo4j is running. Stop it, or change the left-hand port numbers in `docker-compose.yml` (and `NEO4J_URI` to match).
- **`pip install` fails on Python 3.14:** use Python 3.12 (`python3.12 -m venv .venv`).

## Credits

- This product uses the TMDB API but is not endorsed or certified by TMDB.
- MovieLens data: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* 5(4), Article 19.
