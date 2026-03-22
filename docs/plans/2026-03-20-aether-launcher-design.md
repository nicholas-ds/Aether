# Aether + Android Launcher: System Design

**Date:** 2026-03-20

---

## Vision

A private, local-first AI system composed of two independently deployable components:

1. **Aether Backend** — Python inference engine + orchestrator, runs on the home PC (migrates to home server later)
2. **Aether Launcher** — Kotlin Android app that replaces the home screen with a chat-forward, voice-enabled, context-aware AI interface

All data stays on the local network. No cloud services. No telemetry.

---

## Guiding Principles

- **Separation of concerns** — each layer owns one job and exposes a clean interface
- **Modular and replaceable** — components can be swapped or upgraded without rippling changes
- **Self-documenting code** — clear naming and focused functions over verbose comments
- **Local-first** — nothing leaves the LAN; designed to run offline from the public internet
- **Built to grow** — SQLite now, PostgreSQL + pgvector when the home server exists; local TTS/STT now, custom models later

---

## System Overview

```
┌─────────────────────────────────────────┐
│         Aether Launcher (Kotlin)        │
│                                         │
│  ┌─────────────┐  ┌───────────────────┐ │
│  │  Chat/Voice │  │  Context UI Layer │ │
│  │     UI      │  │ (time/loc/sensor) │ │
│  └─────────────┘  └───────────────────┘ │
│  ┌───────────┐  ┌────────┐  ┌────────┐  │
│  │ App Drawer│  │  Room  │  │Retrofit│  │
│  │ (Launcher)│  │ SQLite │  │  API   │  │
│  └───────────┘  └────────┘  └────────┘  │
│               WebSocket / HTTP          │
└───────────────────┬─────────────────────┘
                    │ Local WiFi / LAN
┌───────────────────▼─────────────────────┐
│         Aether Backend (Python)         │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ FastAPI  │  │  Router  │  │  LoRA  │ │
│  │ REST/WS  │→ │Orchestr. │→ │Registry│ │
│  └──────────┘  └──────────┘  └────────┘ │
│                      │                  │
│  ┌──────────┐  ┌──────────┐             │
│  │LlamaIndex│  │    HF    │             │
│  │ RAG/Mem  │  │Inference │             │
│  └──────────┘  └──────────┘             │
│                      │                  │
│  ┌──────────────────────────────────┐   │
│  │  SQLite  →  PostgreSQL + pgvector│   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## Component 1: Aether Backend

### Responsibilities

- Serve as the AI brain: inference, orchestration, memory, RAG
- Expose a clean API for the launcher (and any future client)
- Manage model lifecycle, LoRA adapter registry, context-aware prompting

### Stack

| Layer         | Technology                                | Role                                                            |
| ------------- | ----------------------------------------- | --------------------------------------------------------------- |
| API           | FastAPI                                   | REST + WebSocket server                                         |
| Orchestration | Custom router                             | Intent classification, adapter selection, uncertainty detection |
| Inference     | HuggingFace Transformers + PEFT           | Local LLM + LoRA adapter loading                                |
| RAG + Memory  | LlamaIndex (local embeddings only)        | Conversation memory retrieval, document RAG                     |
| Embeddings    | `nomic-embed-text` or `BAAI/bge-small-en` | Local, no cloud                                                 |
| Database      | SQLite (Room-compatible schema)           | Conversations, sessions, context log                            |

### Directory Structure

```
src/
├── api/
│   ├── routes/
│   │   ├── chat.py          # WebSocket + REST chat endpoints
│   │   └── health.py        # Status/health check
│   └── server.py            # FastAPI app factory
├── orchestrator/
│   ├── router.py            # Intent classification + dispatch
│   ├── prompt_builder.py    # Assembles prompt from memory + context + message
│   └── uncertainty.py       # Uncertainty detection logic
├── models/
│   ├── base.py              # HF model loading + streaming inference
│   ├── config.py            # ModelConfig dataclass
│   └── lora_registry.py     # LoRA adapter registry + swap logic
├── memory/
│   ├── retriever.py         # LlamaIndex query interface
│   ├── indexer.py           # Chunking + embedding + indexing
│   └── session.py           # Session lifecycle management
└── db/
    ├── database.py          # SQLite connection + migrations
    └── models.py            # SQLAlchemy/raw schema definitions
```

### Key Design Decisions

- **HuggingFace inference is kept** — Ollama and similar tools don't support hot-swapping LoRA adapters, which is a core Aether capability
- **LlamaIndex is scoped to RAG and memory only** — not used as an agent framework; routing logic stays custom
- **Local embeddings are non-negotiable** — LlamaIndex defaults to OpenAI; explicitly configured to use local models
- **FastAPI over Flask** — async support is needed for streaming WebSocket responses

---

## Component 2: Aether Launcher (Android)

### Responsibilities

- Replace the Android home screen with an AI-native experience
- Enrich every user interaction with ambient context (time, location, phone state)
- Render streamed responses in real-time
- Surface dynamic context UI cards based on conversation content
- Function as a standard Android launcher (app drawer, home intent)

### Stack

| Layer        | Technology                           | Role                              |
| ------------ | ------------------------------------ | --------------------------------- |
| Language     | Kotlin                               | Native Android                    |
| UI           | Jetpack Compose                      | Declarative UI                    |
| Architecture | MVVM + clean architecture            | Clear layer separation            |
| Networking   | Retrofit (REST) + OkHttp (WebSocket) | FastAPI communication             |
| Local DB     | Room (SQLite)                        | Conversation cache, offline state |
| Voice In     | Android SpeechRecognizer             | STT, no external service          |
| Voice Out    | Android TextToSpeech                 | TTS, no external service          |
| DI           | Hilt                                 | Dependency injection              |

### Directory Structure

```
app/src/main/
├── ui/
│   ├── home/          # Main launcher screen (HomeViewModel, HomeScreen)
│   ├── chat/          # Chat + voice interface (ChatViewModel, ChatScreen)
│   ├── context/       # Context cards (ContextCardViewModel, card composables)
│   └── drawer/        # App drawer (DrawerViewModel, DrawerScreen)
├── domain/
│   ├── conversation/  # Conversation model, use cases
│   ├── context/       # Context enrichment (time, location, phone state)
│   └── voice/         # STT + TTS coordination
├── data/
│   ├── api/           # Retrofit client, WebSocket manager, DTOs
│   ├── db/            # Room database, DAOs, entities
│   └── system/        # Android system data (notifications, sensors, location)
└── core/
    ├── launcher/      # HOME intent handling, default launcher logic
    └── di/            # Hilt modules
```

### Key Design Decisions

- **Jetpack Compose over XML layouts** — modern, declarative, composable UI that maps cleanly to dynamic context cards
- **MVVM + clean architecture** — UI knows nothing about data sources; domain layer is pure Kotlin with no Android dependencies
- **Offline-capable** — Room caches recent conversation; launcher remains functional if the backend PC is unreachable
- **Context payload is silent** — ambient data (location, time, battery, notification count) is attached to every request without user intervention
- **WebSocket for streaming** — tokens stream from FastAPI directly into the chat UI bubble in real-time

---

## Data Flow: Message Lifecycle

```
1. User speaks or types
        ↓
2. SpeechRecognizer → text
   ContextCollector → { time_of_day, location, phone_state, notification_count }
        ↓
3. Launcher sends over WebSocket:
   { message: str, context: ContextPayload, session_id: str }
        ↓
4. FastAPI router receives → passes to Orchestrator
        ↓
5. Orchestrator:
   a. LlamaIndex retrieves relevant memory from SQLite
   b. Classifies intent → selects LoRA adapter
   c. PromptBuilder assembles: [system + memory + context + message]
        ↓
6. HF model streams tokens
        ↓
7. FastAPI streams tokens over WebSocket
        ↓
8. Launcher renders tokens in real-time
   TextToSpeech reads response (if voice mode)
   ContextCardEvaluator checks response → surfaces card if relevant
        ↓
9. Conversation persisted: SQLite on both backend and launcher
```

---

## Database Schema (SQLite — Phase 1)

```sql
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    created_at  INTEGER NOT NULL,
    summary     TEXT
);

CREATE TABLE conversations (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    timestamp   INTEGER NOT NULL,
    role        TEXT NOT NULL,   -- 'user' | 'assistant'
    content     TEXT NOT NULL
);

CREATE TABLE context_log (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    timestamp       INTEGER NOT NULL,
    location_lat    REAL,
    location_lng    REAL,
    time_of_day     TEXT,
    phone_state     TEXT
);
```

Designed for direct migration to PostgreSQL. Adding `pgvector` later replaces the LlamaIndex SQLite vector store without changing the conversation schema.

---

## What Comes From aiCompanion

| Element                                            | Status                                         |
| -------------------------------------------------- | ---------------------------------------------- |
| Visual design language (purple/pink/green palette) | Adapted to Kotlin/Compose                      |
| Chat bubble layout (user right, AI left)           | Adapted to Compose                             |
| Robot mascot concept                               | Retained, reimplemented as SVG/vector drawable |
| Two-tier memory (daily active + archive)           | Retained as session-based SQLite pattern       |
| Node.js/Express backend                            | Replaced by FastAPI                            |
| MongoDB                                            | Replaced by SQLite → PostgreSQL                |
| React Native code                                  | Not ported — design patterns only              |

---

## What's Not In Scope (Yet)

- LoRA fine-tuning pipeline (infrastructure exists, training datasets TBD)
- Home server migration (schema is ready, move is mechanical when hardware exists)
- Custom TTS/STT models (Android built-ins first, upgrade later)
- Multi-user support
- External tool integrations (shell, GIMP, Blender)

---

## Migration Path

| Now                           | Later                                |
| ----------------------------- | ------------------------------------ |
| SQLite (backend + launcher)   | PostgreSQL + pgvector on home server |
| Android SpeechRecognizer      | Whisper local model                  |
| Android TextToSpeech          | Local neural TTS (Kokoro, Piper)     |
| Single LoRA adapter at a time | Dynamic multi-adapter routing        |
| Manual model download         | Model registry + versioning          |
