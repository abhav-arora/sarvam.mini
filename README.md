# Streaming Hindi→English S2ST WebSocket Server

[![CI](https://github.com/placeholder-org/sarvam-s2s-server/actions/workflows/ci.yml/badge.svg)](https://github.com/placeholder-org/sarvam-s2s-server/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Inter IIT Tech Meet 15.0 — Sarvam.ai Mini Problem Statement (Part E: Dev Track)**

A low-latency, real-time streaming Speech-to-Speech Translation (S2ST) WebSocket server following §5.1 / §6 specification. Designed for strict causality, robust event-loop isolation, single-shot completion, and zero state leakage across sequential sessions.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              WebSocket Endpoint              │
                      │             ws://0.0.0.0:8000/ws             │
                      └──────────────────────┬───────────────────────┘
                                             │
                             asyncio.Semaphore(1) (Serialised)
                                             │
                        ┌────────────────────┴───────────────────┐
                        │              Session Task              │
                        │                                        │
           Audio Frames │  ┌──────────────────────────────────┐  │ Output Chunks
          (80 ms PCM)   │  │   Reader Loop (Non-blocking)     │  │ (<= 1.0 s PCM)
        ───────────────►│  │   - Parse control & binary frames│  │──────────────►
                        │  │   - Timestamp arrival (perf_cnt) │  │
                        │  │   - Enqueue to Engine            │  │
                        │  └────────────────┬─────────────────┘  │
                        │                   │                    │
                        │  ┌────────────────▼─────────────────┐  │
                        │  │   Engine Interface (Pluggable)   │  │
                        │  │   - EchoEngine (Scheduled)      │  │
                        │  │   - FakeModelEngine (Latency)    │  │
                        │  │   - ModelEngine (Phase 5 Stub)   │  │
                        │  └────────────────┬─────────────────┘  │
                        │                   │                    │
                        │  ┌────────────────▼─────────────────┐  │
                        │  │   Writer Loop & emit() Guard     │  │
                        │  │   - <= 48,000 B, whole samples   │  │
                        │  │   - Single-shot session_complete │  │
                        │  │   - 25s Safety Watchdog          │  │
                        │  └──────────────────────────────────┘  │
                        └────────────────────────────────────────┘
```

### Key Protocol Invariants (§5.1 / §6)
1. **Wire Audio:** Linear PCM, signed 16-bit little-endian (`s16le`), mono, **24,000 Hz** (2 bytes/sample).
2. **Paced Client Framing:** Input audio sent in 80 ms frames (1,920 samples = 3,840 bytes) on an absolute clock schedule.
3. **Server Output Formatting:** Audio emitted in chunks $\le$ 48,000 bytes ($\le$ 1.0 s), always even byte length, with no WAV header.
4. **Single-shot Completion:** Exactly one `session_complete` JSON frame sent at the end; connection closed immediately after.
5. **Session Isolation:** Sequential sessions with a 2-second gap; clean state reset (`engine.reset()`) in a `finally` block.
6. **Watchdog Guard:** 25.0 s watchdog timer after `input_end` guarantees termination within the 30 s PS deadline.

---

## Directory Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI (Ruff lint + Pytest)
├── app/
│   ├── main.py                  # FastAPI server (/health & /ws), lifespan warm-up
│   ├── protocol.py              # Pydantic wire models and protocol constants
│   ├── session.py               # Session state machine, reader & writer tasks
│   ├── audio.py                 # PCM <-> numpy conversion, frame slicing, trimming
│   ├── metrics.py               # Session summary JSON logs, loop-lag canary
│   ├── config.py                # Environment-driven settings (pydantic-settings)
│   └── engine/
│       ├── base.py              # Engine protocol interface
│       ├── echo.py              # Delayed echo engine (arrival-time scheduled)
│       ├── fake_model.py        # Fake model simulating per-step inference compute
│       └── model.py             # Phase 5 real checkpoint adapter stub
├── tools/
│   ├── make_clip.py             # Audio converter & synthetic clip generator
│   ├── client.py                # Paced, harness-like test client with JSON trace
│   ├── analyze.py               # Playback-timeline simulator, rule check & plot
│   ├── wait_health.py           # Cold-start readiness benchmark tool
│   ├── harness_sim.py           # N sequential sessions with 2 s gaps
│   └── stress.py                # Fault-injection (burst, disconnect, concurrent)
├── tests/
│   ├── test_protocol.py         # Protocol serialization & validation tests
│   ├── test_engine_echo.py      # Echo engine timing & order tests
│   ├── test_engine_fake_model.py# Fake model async processing tests
│   └── test_session.py          # WebSocket integration tests via uvicorn
├── clips/                       # Test audio clips (24 kHz mono s16le WAV)
├── out/                         # Generated traces and timeline plots
├── Dockerfile                   # Phase 1 CPU / Echo container
├── Dockerfile.gpu               # Phase 2 CUDA container specification
├── pyproject.toml               # Pinned dependencies & build configuration
└── RESULTS.md                   # Benchmark results and verification checklist
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or `pip`

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/placeholder-org/sarvam-s2s-server.git
cd sarvam-s2s-server

# Install dependencies using uv
uv sync --dev
```

### 2. Run the Server
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 0
```
Server endpoints:
- `GET http://localhost:8000/health`: Health status (`{"status": "ok"}`)
- `WS ws://localhost:8000/ws`: Streaming WebSocket endpoint

### 3. Generate Test Clips
```bash
# Generate synthetic clips (10s, 30s, 60s, paused, noisy, telephone)
uv run python tools/make_clip.py clips/c01.wav --synthetic 10.0
uv run python tools/make_clip.py clips/c02.wav --synthetic 30.0
uv run python tools/make_clip.py clips/c01_pauses.wav --synthetic 15.0 --pauses
```

### 4. Run the Client and Analyze Timeline
```bash
# Run paced client
uv run python tools/client.py clips/c01.wav

# Analyze playback timeline, verify PS rules, and generate plot
uv run python tools/analyze.py out/c01.json --plot
```

---

## Docker Deployment

### Build and Run Local Container
```bash
# Build Phase 1 image
docker build -t sarvam-s2s-server:latest .

# Run container (offline / no network required)
docker run --rm -p 8000:8000 --network none sarvam-s2s-server:latest
```

### Verify Container Health
```bash
uv run python tools/wait_health.py --url http://localhost:8000/health
```

---

## Verification & Benchmarks

All results are recorded in [RESULTS.md](RESULTS.md):

| Metric | Target | Measured | Status |
|---|---|---|---|
| **Cold-start to `/health` 200** | < 300 s | **2.023 s** | PASS |
| **Echo delay accuracy ($D=2.0$s)** | $\pm$ 50 ms | **1.998 s** ($\Delta = -2$ ms) | PASS |
| **First-audio delay** | – | **1.998 s** | PASS |
| **Tail latency** | – | **2.099 s** | PASS |
| **Completion time** | < 30.0 s | **1.995 s** | PASS |
| **Event-loop lag p99** | < 20 ms | **< 1 ms** | PASS |
| **Harness multi-session** | 100% | **3/3 passed** (2.0s gap) | PASS |
| **Burst stress throughput** | Real-time | **168× real time** (0.059 s) | PASS |

### Timeline Plot Showcase
Running `tools/analyze.py --plot` outputs unified timeline figures illustrating frame pacing, input/output activity, and completion boundaries.

---

## Testing & Quality Assurance

```bash
# Run all unit and integration tests
uv run pytest -v

# Run Ruff linter
uv run ruff check .
```

---

## Failure Modes & Engineering Safeguards (§7)

| Risk / Failure Mode | Engineering Safeguard Implemented |
|---|---|
| **Coroutines blocking event loop** | Reader task only enqueues; inference runs asynchronously in a dedicated worker thread. Loop-lag canary monitors jitter. |
| **Socket buffer backup after `input_end`** | Reader loop runs continuously until socket disconnects, reading & discarding silence frames. |
| **State leakage between sessions** | `engine.reset()` and task cleanup in `finally`; `asyncio.Semaphore(1)` guarantees serial sessions. |
| **Oversized audio chunks or odd bytes** | Central `emit()` function enforces $\le$ 48,000 bytes and even sample alignment (`len % 2 == 0`). |
| **Unresponsive model engine** | 25.0-second safety watchdog triggers forced `session_complete` before the 30.0 s penalty deadline. |
| **Network dependence during evaluation** | All dependencies bundled offline; container runs under `--network none`. |

---

## License
MIT License.
