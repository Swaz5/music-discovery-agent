# music-discovery-agent

A music discovery system that uses Claude as an autonomous reasoning agent. Give it a vibe — "something that sounds like driving through a city at night" — and it searches Last.fm, downloads and analyzes 30-second audio previews from Deezer, queries a ChromaDB knowledge base, and returns recommendations grounded in actual audio feature data.

Built on top of the Anthropic API with tool use. The agent decides which tools to call, in what order, and when it has enough data to stop.

---

## Demo

**Query:** "Something that sounds like driving through a city at night"

**Result:** 7 recommendations in 15 iterations using 129,097 tokens

| # | Artist | Track | Score | Why |
|---|--------|-------|-------|-----|
| 1 | Massive Attack | Teardrop | 0.95 | energy 0.86, tempo 76 BPM, valence 0.46 — forward momentum without anxiety |
| 2 | DJ Shadow | Midnight In A Perfect World | 0.93 | literally designed for midnight drives; dense sample layers |
| 3 | Hird | Keep You Kimi | 0.91 | energy 0.80, danceability 0.91, neutral valence — hidden gem |
| 4 | Sneaker Pimps | 6 Underground | 0.89 | defines the nocturnal trip-hop aesthetic |
| 5 | Phantogram | Black Out Days | 0.88 | high energy, neutral valence — alert but not anxious |
| 6 | Ursula 1000 | Soft Landing | 0.86 | moderate energy 0.50 — cruising, not racing |
| 7 | Marsmobil | Magnetising | 0.82 | high energy 0.90, moderate tempo — pulled forward by invisible forces |

The agent made 14 tool calls to get here: 1 KB search, 3 tag searches, 2 audio feature pulls, 1 similarity expansion, 7 Deezer track lookups. Each recommendation cites the specific audio numbers that justified the pick.

---

## Features

**Vibe-based discovery** — natural language queries interpreted by Claude, not a keyword search. The agent decides which genres and tags to explore, fetches artists, downloads 30-second previews, and runs Librosa audio analysis on the actual audio.

**Taste analysis** — give it a list of artists you like. It fetches all their profiles in parallel, computes aggregate audio statistics (averages, variance, outliers), finds the connective thread, names your taste identity, and identifies blind spots — genres you'd likely enjoy but probably haven't explored.

**Genre bridging** — BFS through the Last.fm artist-similarity graph to find the artists that connect two genres. Builds a transition playlist that moves smoothly from one end to the other.

**Preference learning** — rate artists as liked/disliked. The preference engine tracks audio features and tags, computes sweet spots (dimensions where your liked and disliked artists diverge most), and injects a personalized context block into the agent's system prompt on every iteration.

**SSE streaming** — `POST /discover` with `stream=true` returns a `text/event-stream` that replays the agent's reasoning steps as events: `start` → `reasoning_step` × N → `recommendations` → `done`.

**MCP server** — all 6 agent tools are also exposed via the Model Context Protocol.

---

## Architecture

```
src/
├── agent/
│   ├── discovery_agent.py   # Claude agentic loop — tool dispatch, iteration budget
│   ├── preference_engine.py # Preference persistence + taste profile computation
│   └── tools.py             # Tool registry: schemas + function implementations
├── api/
│   ├── main.py              # FastAPI app, CORS, lifespan
│   └── routes.py            # 9 endpoints + SSE streaming + BFS bridge search
├── data/
│   ├── lastfm_client.py     # Last.fm API: artist search, similar artists, tag queries
│   ├── deezer_client.py     # Deezer API: track search, preview download
│   ├── audio_analyzer.py    # Librosa: energy, tempo, valence, danceability, etc.
│   └── music_service.py     # Unified service — combines Last.fm + Deezer + audio
├── rag/
│   ├── retriever.py         # ChromaDB semantic search over genre knowledge base
│   └── ingest.py            # Ingestion pipeline — runs at container startup
└── mcp/
    └── server.py            # MCP server exposing all 6 agent tools
```

**Request path for `POST /discover`:**

1. FastAPI receives the query and calls `MusicDiscoveryAgent.discover()`
2. Agent calls `client.messages.create()` with tool schemas and a budget-aware system prompt
3. Claude returns `stop_reason="tool_use"` with one or more tool calls
4. Agent dispatches each call through `TOOLS_REGISTRY` — Last.fm lookup, Deezer preview download, Librosa analysis, ChromaDB retrieval
5. Results are appended to the message history and Claude is called again
6. Loop continues until `stop_reason="end_turn"` or the iteration ceiling is hit
7. Agent parses Claude's structured text output with a regex parser and returns recommendations + reasoning trace

**Preference context injection:**

Each iteration, `_build_system_prompt(iterations_used=N)` fills two things into the system prompt: the remaining iteration budget (`{budget}` placeholder) and the current taste profile from `PreferenceEngine.get_recommendation_context()`. Claude sees the user's preferred tags, audio feature sweet spots, and disliked genres before deciding which artists to investigate.

---

## Audio Analysis

Deezer provides 30-second MP3 previews for most tracks. The `get_audio_features` tool:

1. Searches Deezer for the track and downloads the preview to `data/previews/`
2. Loads with `librosa.load()` at 22,050 Hz
3. Computes:

| Feature | Method | Range |
|---------|--------|-------|
| energy | RMS energy (normalized) | 0–1 |
| tempo | BPM via `librosa.beat.beat_track` | continuous |
| danceability | Onset regularity × tempo consistency | 0–1 |
| valence | Spectral ratio of high vs. low frequencies | 0–1 |
| acousticness | Inverse of spectral flatness | 0–1 |
| instrumentalness | ZCR-based voice detection | 0–1 |
| loudness | Mean RMS in dB | negative |

Previews are cached — re-running the same query reuses downloaded files. The `.chroma/` vector store and `data/previews/` are bind-mounted in Docker so data persists across restarts.

---

## Tech Stack

| Layer | Library / Service |
|-------|------------------|
| Agent | Anthropic API — claude-sonnet-4 with tool use |
| API | FastAPI + Uvicorn |
| Music data | Last.fm API, Deezer API |
| Audio analysis | Librosa, soundfile, ffmpeg |
| Vector store | ChromaDB |
| HTTP client | httpx (async, 10s timeouts) |
| Validation | Pydantic v2 |
| Tests | pytest + pytest-asyncio, 203 tests |
| Container | Docker (multi-stage build) |
| Protocol | MCP server |

19 Python files, 5,141 lines of source, 13 modules.

---

## Getting Started

**Prerequisites:** Python 3.11+, a Last.fm API key, a Deezer API key, an Anthropic API key.

```bash
git clone <repo>
cd music-discovery-agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
LASTFM_API_KEY=...
DEEZER_APP_ID=...
DEEZER_APP_SECRET=...
```

Seed the knowledge base (one-time):
```bash
python -m src.rag.ingest
```

Start the server:
```bash
uvicorn src.api.main:app --reload
```

**With Docker:**
```bash
docker-compose up
```

The compose file mounts `.chroma/`, `data/preferences.json`, and `data/previews/` as volumes. `entrypoint.sh` seeds the knowledge base on every container start — idempotent, only fetches what's missing.

**Run tests:**
```bash
pytest tests/ -v --timeout=30
# 203 passed, 3 warnings in 32.15s
```

All 203 tests are fully mocked — no real API calls, no audio downloads.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/discover` | Run discovery agent; `stream=true` for SSE |
| `GET` | `/artist/{name}` | Full artist profile from Last.fm + Deezer |
| `POST` | `/explore` | Genre exploration with KB context |
| `POST` | `/rate` | Save a liked/disliked preference rating |
| `GET` | `/taste-profile` | Current taste profile from all stored ratings |
| `GET` | `/history` | Session query history (in-memory, up to 100) |
| `POST` | `/analyze-taste` | Taste analysis + blind spot discovery |
| `POST` | `/bridge` | Genre bridge: BFS similarity graph + explanation |

Interactive docs at `http://localhost:8000/docs`.

---

## Example: Taste Analysis

**Input:**
```json
POST /analyze-taste
{ "artists": ["Radiohead", "Tame Impala"], "discover_blind_spots": true }
```

**Output:**

> **Taste identity:** Atmospheric Experimentalist

Audio profile averaged across both artists: energy 0.94, danceability 0.87, valence 0.51, tempo 108 BPM, acousticness 0.80.

Common threads: Electronic-organic fusion · Atmospheric production techniques · Rhythmic experimentation within rock frameworks · Emotionally complex but danceable · Studio-as-instrument philosophy

**Blind spots identified:**

| Genre | Why it fits | Start here |
|-------|-------------|------------|
| Ambient Techno | Same electronic textures and hypnotic rhythms, more focus on sonic immersion | Boards of Canada — Roygbiv |
| Dream Pop | Atmospheric production, similar acoustic-electronic blend | Beach House — Space Song |
| Neo-Soul | High danceability + electronic experimentation + organic elements | FKA twigs — Two Weeks |
| Post-Rock | Experimental approach to rock instrumentation, builds atmospheric tension | Godspeed You! Black Emperor — Storm |

**Outlier:** Radiohead — lower energy and danceability than Tame Impala, more emphasis on deconstructed song forms over psychedelic groove.

*Elapsed: 24.2s*

---

## How It Works

### The Agentic Loop

`MusicDiscoveryAgent` runs a multi-turn conversation with Claude. Each turn:

1. Claude receives the full message history plus the current system prompt
2. If it returns `stop_reason="tool_use"`, the agent extracts the tool calls and dispatches them
3. Tool results are appended as `tool_result` content blocks
4. Claude is called again with the updated history

The loop runs until `end_turn` or `_MAX_ITERATIONS=15` is reached. The system prompt includes an explicit budget:

```
You have at most {budget} tool calls available for this query.
- Call get_audio_features for at most 3–4 artists — only the most promising candidates.
- Reserve your last 2–3 iterations to write your final formatted recommendations.
- If you have already made 10 or more tool calls, stop exploring and write recommendations.
```

Without this, Claude spends all 15 iterations on `get_similar_artists` calls (each downloads 8 audio previews in parallel) and never writes the final output.

### Tool Dispatch

6 tools available to the agent:

| Tool | What it does |
|------|-------------|
| `search_artist` | Last.fm artist info: bio, tags, listener count, top tracks |
| `get_similar_artists` | Last.fm similarity graph + Deezer audio features for each result |
| `search_by_tag` | Last.fm tag → top artists |
| `get_audio_features` | Download Deezer preview → Librosa analysis |
| `search_knowledge_base` | ChromaDB semantic search over genre descriptions |
| `search_tracks` | Deezer track search → streaming URL + cover art |

### Genre Bridge: BFS

`POST /bridge` runs breadth-first search through the Last.fm similarity graph:

1. Fetch top-8 seed artists for both genres in parallel
2. Check for direct overlap (hop 0)
3. Expand each seed's similar artists (hop 1: up to 8 seeds; hop 2: up to 6; hop 3: up to 4)
4. Any similar artist whose name appears in genre_b's seed set → path found
5. Fetch tags for all intermediate artists
6. Pass paths + tags + KB context to Claude, which selects the best bridge artists and builds the transition playlist

The graph search and Claude analysis are decoupled — if BFS finds no paths, Claude falls back on its training knowledge grounded by the KB context.

**Jazz → Electronic** (34.3s): BFS found paths through 3 hops. Bridge artists: The Internet, Frank Ocean, Brent Faiyaz, Gorillaz, The Weeknd. Transition playlist runs Miles Davis → Sade → The Internet → Frank Ocean → Brent Faiyaz → The Weeknd → Gorillaz → Daft Punk → The Weeknd.

### RAG Knowledge Base

`python -m src.rag.ingest` pulls genre descriptions from Last.fm, generates embeddings, and stores them in ChromaDB at `.chroma/`. The `search_knowledge_base` tool retrieves relevant passages at query time — giving Claude grounded context about genre history and characteristics before it starts searching for specific artists.

### Preference Engine

`PreferenceEngine` reads/writes `data/preferences.json`. When you rate an artist, it:

1. Fetches the artist's audio profile if not provided
2. Deduplicates by artist name (case-insensitive)
3. Recomputes the taste profile: average audio features for liked vs. disliked artists, tag frequency counts, sweet spots (features where liked and disliked diverge by more than a threshold)
4. Generates a text block injected into the system prompt on every discovery iteration

The recommendation context looks like:

```
## User taste profile
Gravitates toward: trip-hop, atmospheric, electronic
Tends to dislike: pop, mainstream
Preferred audio: energy=0.40, valence=0.25, danceability=0.65
Strongest preferences (sweet spots): energy (δ=−0.55)
Based on 3 ratings (2 liked, 1 disliked). Help them discover new music — don't just confirm existing taste.
```

---

## What I Learned

**Agentic loops need explicit stopping conditions.** The first version hit the 10-iteration ceiling before writing any recommendations — it spent all its budget on `get_similar_artists` calls. Each call downloads and analyzes 8 audio previews in parallel, so one tool call could consume the entire remaining budget before Claude realized it needed to synthesize. The fix was twofold: raise the ceiling to 15 and add explicit budget instructions to the system prompt. Without telling Claude how many calls it has left, it optimizes for thoroughness. With the budget block, it starts writing recommendations by iteration 13-14.

**Testing async multi-turn conversations requires careful mock attribute setup.** The agentic loop isn't complex — it's a while loop that builds a `messages` list. The tricky part is that Anthropic response objects have specific attribute names (`.stop_reason`, `.content[i].type`, `.content[i].input`) and the mock has to match them exactly. `MagicMock` with explicit attribute assignment works, but you have to set `block.type = "tool_use"` before auto-creation kicks in, otherwise `block.type` becomes a child mock that never equals the string `"tool_use"`.

**ChromaDB collection names are strict.** The ingest pipeline initially created a collection called `"music-knowledge-base"`. ChromaDB rejects names with hyphens. Underscores required.

**Docker build-time API keys are a trap.** The obvious approach — `ARG LASTFM_API_KEY` in the Dockerfile + ingestion at build time — bakes the key into the image layer history (`docker history` shows it in plaintext). Moving ingestion to `entrypoint.sh` keeps keys in environment-only scope. The tradeoff is a slightly slower first startup, which is acceptable for local use.

**Last.fm's similarity graph is sparse in places.** For some genre pairs, BFS finds no paths within 3 hops. The bridge endpoint handles this gracefully — Claude draws on its training knowledge and KB context to propose bridge artists even with an empty graph. Output quality for "jazz → electronic" was identical whether BFS found paths or not; the KB context was more useful than the graph data.

**Librosa's valence is a proxy, not ground truth.** Real valence (as Spotify computes it) requires training a model on human-labeled data. The Librosa implementation — spectral ratio of high vs. low frequencies — correlates loosely with perceived positivity but breaks down for tracks with unusual spectral shapes. The system prompt describes it as a rough proxy so Claude treats it accordingly rather than over-weighting it.

---

## License

MIT
