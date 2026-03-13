## TripPlannerBackend Architecture

This document describes the overall architecture of the TripPlannerBackend project: components, data flow, and how the AI travel planning agents interact with local and external services.

---

## 1. High-Level Components

- **FastAPI RAG & Agent Service (`fast_api_embedded_server.py`)**
  - Exposes HTTP APIs:
    - `/embed`, `/search`, `/answer` (RAG over Wikivoyage content in Qdrant)
    - `/search_claude`, `/search_gemini` (single-shot external LLM search)
    - `/agent/plan`, `/agent/plan_claude` (single-turn Agent planning)
    - `/agent/chat`, `/agent/chat_claude` (multi-turn, session-based Agents)
  - Hosts the main `FastAPI` app and includes the Agent router (`agent.router`).

- **Agent Package (`agent/`)**
  - `chain.py`: Agent execution loop
    - Uses **LangChain** Chat models:
      - `ChatOllama` (local LLM via Ollama) for `/agent/plan` and `/agent/chat`
      - `ChatAnthropic` (Claude via Anthropic API) for `/agent/plan_claude` and `/agent/chat_claude`
    - Binds tools and runs a step-by-step tool-calling loop with explicit planning.
  - `tools.py`: Agent tools
    - `search_travel_knowledge`: calls local `/search` (RAG over Qdrant/Wikivoyage)
    - `translink_search_stops`, `translink_get_departures`: query local Translink GTFS data (see below)
  - `gtfs_client.py`: reads local **SEQ GTFS** CSV files under `data/SEQ_GTFS` using pandas, provides:
    - `search_stops(query, limit)`: find stops by name (with lat/lon)
    - `get_departures(stop_id, after_time, limit)`: basic next departures from a stop
  - `session.py`: in-memory session store for multi-turn chat
    - Keeps a short message history per `session_id`
    - Used by `/agent/chat` and `/agent/chat_claude` to maintain conversational context.
  - `router.py`: FastAPI `APIRouter` exposing:
    - `/agent/plan`, `/agent/plan_claude`
    - `/agent/chat`, `/agent/chat_claude`

- **RAG / Vector Store**
  - **Qdrant** (Docker service `qdrant`):
    - Collection: `trip_au`
    - Stores dense embeddings of Wikivoyage-like travel content
  - **Embeddings**:
    - BGE-M3 (via `FlagEmbedding`) or
    - Ollama embeddings (`nomic-embed-text`) via `OLLAMA_URL`
  - Ingestion scripts under `test/`:
    - `ingest_wikivoyage_xml_to_qdrant.py`, `ingest_qdrant.py` etc.

- **External / Local LLMs**
  - **Ollama** (Docker service `ollama`):
    - Provides local LLMs (e.g. `llama3.1`) and optional embedding models
  - **Claude (Anthropic)**:
    - Accessed via HTTPS API from `fast_api_embedded_server.py` and `agent/chain.py`
    - Requires `ANTHROPIC_API_KEY`
  - **Gemini**:
    - Accessed via Google Generative Language API in `/search_gemini`
    - Requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`

- **Translink GTFS Data (`data/SEQ_GTFS/`)**
  - Local snapshot of Translink South East Queensland GTFS:
    - `stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`, `shapes.txt`, `calendar.txt`, `calendar_dates.txt`, etc.
  - Used only by the Agent tools; no external Translink API calls are required.

- **Django App (`TripPlannerBackend/` + `manage.py`)**
  - Standard Django project (admin, Postgres DB) for potential web UI / admin features.
  - Runs in Docker service `web` (port 8000) via `docker-compose.yml`.

- **CLI Client (`cmd_chat.py`)**
  - Terminal-based client for `/agent/chat` and `/agent/chat_claude`
  - Handles:
    - Session persistence via `session_id`
    - Formatted display of agent answers and internal tool-calling steps.

- **Infrastructure / Scripts**
  - `docker-compose.yml`:
    - Services: `db` (Postgres), `web` (Django), `ollama`, `qdrant`
  - `start.sh` / `stop.sh`:
    - `start.sh`: brings up Docker services and starts FastAPI (uvicorn :8001)
    - `stop.sh`: stops FastAPI and brings Docker stack down.

---

## 2. Architecture Diagram

```mermaid
flowchart LR
    subgraph Client
      C1[Web / Mobile frontend\nor API consumer]
      C2[cmd_chat.py\n(CLI client)]
    end

    subgraph Backend["FastAPI RAG & Agent Service\n(fast_api_embedded_server.py)"]
      API[FastAPI app\n/health, /embed, /search,\n/answer, /search_claude,\n/agent/* ...]

      subgraph AgentPkg["agent/ package"]
        R[router.py\n/agent/plan\n/agent/plan_claude\n/agent/chat\n/agent/chat_claude]
        CH[chain.py\nOllama & Claude Agents\n(step-by-step loop)]
        T[tools.py\nsearch_travel_knowledge\ntranslink_search_stops\ntranslink_get_departures]
        SESS[session.py\nin-memory\nsession store]
        GTFS[gtfs_client.py\nread data/SEQ_GTFS\n(stops, trips, times)]
      end
    end

    subgraph VectorDB["Vector Store / RAG"]
      QD[Qdrant\ntrip_au collection]
      EMB[Embeddings\nBGE-M3 or\nOllama embeddings]
    end

    subgraph ExternalLLM["LLMs"]
      OLL[Ollama\nlocal LLMs\nllama3.1, etc.]
      CLAUDE[Claude\n(Anthropic API)]
      GM[Gemini\n(Google API)]
    end

    subgraph Data["Local Data"]
      WK[Wikivoyage-like\ntravel docs\n(ingested to Qdrant)]
      TG[data/SEQ_GTFS\nTranslink GTFS\n(SEQ stops, routes,\ntrips, times)]
    end

    subgraph DjangoStack["Django & DB (optional)"]
      DJ[Django app\nTripPlannerBackend/]
      PG[(Postgres DB\nDocker db)]
    end

    %% Client to Backend
    C1 -->|HTTP JSON\n/embed, /search,\n/agent/*| API
    C2 -->|HTTP JSON\n/agent/chat(_claude)| API

    %% Backend internal wiring
    API --> R
    R --> CH
    CH -->|tool calls| T
    R --> SESS
    T -->|local HTTP| API

    %% Translink GTFS usage
    T --> GTFS
    GTFS --> TG

    %% RAG
    API -->|/search_embed| EMB
    EMB --> QD
    QD --> API
    WK --> QD

    %% LLMs
    CH -->|Ollama Agent\n(ChatOllama)| OLL
    CH -->|Claude Agent\n(ChatAnthropic)| CLAUDE
    API -->|/search_claude| CLAUDE
    API -->|/search_gemini| GM

    %% Django & DB (for web/admin)
    C1 -->|HTTP\n(port 8000)| DJ
    DJ --> PG
```

---

## 3. Agent Execution Flow (Claude or Ollama)

1. **Client call**  
   - For single-turn: `POST /agent/plan` or `POST /agent/plan_claude`  
   - For multi-turn: `POST /agent/chat` or `POST /agent/chat_claude` with optional `session_id`

2. **Router & Session** (`agent/router.py`, `agent/session.py`)  
   - Router parses request into Pydantic models.  
   - For multi-turn, session history is loaded and converted into `previous_messages` for the Agent loop.

3. **Agent loop** (`agent/chain.py`)  
   - Selects LLM:
     - `ChatOllama` for Ollama-based agent
     - `ChatAnthropic` for Claude-based agent
   - Binds tools: `search_travel_knowledge`, `translink_search_stops`, `translink_get_departures`.  
   - Runs a loop:
     - LLM produces a message (may include tool_calls).  
     - When tool_calls exist, the corresponding tool functions are invoked and their results are fed back as `ToolMessage`s.  
     - Continues until LLM returns a message with no tool_calls or `max_iterations` is reached.

4. **Tools** (`agent/tools.py`)  
   - `search_travel_knowledge`:
     - Calls local `POST /search` → FastAPI uses embeddings + Qdrant (`trip_au` collection) → payload snippets returned → tool returns formatted snippets to LLM.
   - `translink_search_stops`:
     - Uses `gtfs_client.search_stops` over local `data/SEQ_GTFS/stops.txt` to find Translink stops by name, returns a small table string.
   - `translink_get_departures`:
     - Uses `gtfs_client.get_departures` over local GTFS (`stop_times`, `trips`, `routes`) to provide upcoming departures for a stop.

5. **Response**  
   - Agent returns final answer + internal step trace to the router.  
   - Router wraps it into JSON response (e.g. `AgentPlanOut` / `ChatOut`) for the client.

---

## 4. Data & Service Startup

- **Docker stack** (`docker-compose.yml`)
  - `db`: Postgres (for Django)
  - `web`: Django (port 8000)
  - `ollama`: local LLM server (port 11434)
  - `qdrant`: vector DB (ports 6333/6334)

- **Helper scripts**
  - `start.sh`:
    - `docker compose up -d` → starts `db`, `web`, `ollama`, `qdrant`
    - Activates `.venv` (if present)
    - Starts FastAPI via `uvicorn fast_api_embedded_server:app --host 0.0.0.0 --port 8001` in the background (PID stored in `.uvicorn.pid`)
  - `stop.sh`:
    - Stops FastAPI by PID or port 8001
    - `docker compose down`

---

## 5. Extending the Architecture

The architecture is designed to follow the Open-Closed Principle:

- To add more travel-related tools (e.g. external APIs, additional GTFS regions, weather, hotel search):
  - Implement new `@tool` functions in `agent/tools.py` (or submodules).
  - Register them in `TOOLS` in `agent/chain.py`.
  - Optionally update `SYSTEM_PROMPT` to describe how/when to use them.

- To add new endpoints:
  - Add Pydantic schemas + route handlers in `agent/router.py` (or new routers).
  - Mount routers in `fast_api_embedded_server.py` via `include_router`, without modifying existing RAG endpoints.

