## Project Stats

| Metric | Value |
|--------|-------|
| Python source files | 19 |
| Lines of Python (src/) | 5,141 |
| Source modules | 13 |
| Agent tools | 6 |
| API endpoints | 9 |
| Test files | 8 |
| Tests | `203 passed, 3 warnings in 32.15s` |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check — service liveness |
| `POST` | `/discover` | Run the discovery agent (natural-language vibe → curated recs) |
| `GET`  | `/artist/{name}` | Full artist profile with bio, tags, audio features, similar artists |
| `POST` | `/explore` | Genre exploration with RAG knowledge-base context |
| `POST` | `/rate` | Save a like/dislike preference to shape future recommendations |
| `GET`  | `/taste-profile` | View the current personal taste profile |
| `GET`  | `/history` | Session recommendation history |
| `POST` | `/analyze-taste` | Taste analysis + blind-spot discovery via Claude |
| `POST` | `/bridge` | Genre bridge: BFS similarity graph + Claude explanation |

### Agent Tools

| Tool | Description |
|------|-------------|
| `search_artist` | Look up an artist's profile, tags, and social stats from Last.fm |
| `get_similar_artists` | Find sonically related artists via the Last.fm similarity graph |
| `search_by_tag` | Get top artists for a genre or mood tag |
| `get_audio_features` | Quantified audio analysis — energy, valence, tempo, danceability… |
| `search_knowledge_base` | Semantic search over curated genre guides and artist bios (RAG) |
| `search_tracks` | Search Deezer for specific tracks with preview URLs |