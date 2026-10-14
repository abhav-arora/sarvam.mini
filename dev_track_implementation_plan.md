# Dev Track Implementation Plan: Streaming Hindi→English S2ST WebSocket Server

**Event:** Inter IIT Tech Meet 15.0, Sarvam.ai Mini PS, Part E (Dev track)
**Deliverables:** WebSocket server following §5.1, test client, Dockerfile, a delayed-output demo on an input case, and an answer to "what could make it fall behind, and how would you prevent it?"

> **Scope note.** The base checkpoint's architecture is revealed only after the mid-term. This plan therefore isolates the model behind a single `Engine` interface. The server, container and measurement tooling are built and validated now with a delayed-echo engine. The real model is swapped in later. Statements about how the real model will behave (frame-synchronous stepping, etc.) are assumptions, not facts from the PS.

---

## Table of contents

1. [Requirements that drive the design](#1-requirements-that-drive-the-design)
2. [Technology stack](#2-technology-stack)
3. [Repository layout](#3-repository-layout)
4. [Server architecture](#4-server-architecture)
5. [Measurement tooling and the delay demo](#5-measurement-tooling-and-the-delay-demo)
6. [Docker](#6-docker)
7. [Failure modes and prevention](#7-failure-modes-and-prevention)
8. [Fault-injection and test plan](#8-fault-injection-and-test-plan)
9. [Phased execution plan](#9-phased-execution-plan)
10. [Open questions for the organisers](#10-open-questions-for-the-organisers)
11. [RESULTS.md template](#11-resultsmd-template)
12. [Quick reference: constants and byte math](#12-quick-reference-constants-and-byte-math)

---

## 1. Requirements that drive the design

| PS requirement (§5.1 / §6) | Engineering consequence |
|---|---|
| Client sends 80 ms PCM frames (3,840 B) in real time, frame *k* at t = k × 80 ms | Reader must run continuously. Load is only ~12.5 messages/s, so the risk is **blocking**, not throughput. |
| Frame not deliverable within 1 s of schedule → **session fails** | Reader never awaits inference, disk or a slow `send`. |
| After `input_end`, silent frames continue until `session_complete` | Keep reading (and discarding or consuming) until the socket closes. |
| Server audio frames ≤ 1 s, PCM s16le mono 24 kHz, no header | Chunk output to ≤ 48,000 B, even byte length, never emit a WAV header. |
| Exactly one `session_complete`, nothing after it | Single-shot guard, then close. |
| `session_complete` within 30 s of `input_end` | Watchdog at ~25 s as a safety net. A failed session scores worst on every metric. |
| Latency measured on the **playback timeline** | Send audio as soon as it is ready, in small chunks. Never batch. |
| Accept connection ≤ 10 s; `GET /health` 200 within 300 s; port 8000, path `/ws` | Load and warm up the model **before** reporting healthy. |
| Sessions run sequentially with a 2 s gap | One engine instance. Reset all state between sessions. |
| No network; all weights and dependencies inside the container | Offline env flags, weights baked into the image. |
| H100 80 GB, 16 CPU cores, 128 GB RAM | One event loop, one GPU inference thread, spare cores for the rest. |
| Causality: only audio already received may be used | No look-ahead buffering that reads future frames. Hold-back logic must happen before `send`. |

---

## 2. Technology stack

| Layer | Choice | Rationale / alternative |
|---|---|---|
| Language | Python 3.11 | Matches the PyTorch ecosystem the model will use. |
| Web + WS framework | **FastAPI** (Starlette) on **uvicorn** with `uvloop` | One port serves `GET /health` and `WS /ws`. Alternative: bare `websockets` with a `process_request` hook for `/health`, which has fewer layers but a hand-rolled HTTP path. |
| Message validation | **pydantic v2** | Typed parsing of `session_start` and `input_end`. Malformed input must not crash the handler. |
| Audio handling | `numpy`, `soundfile`, `scipy.signal.resample_poly` | PCM ↔ float conversion, resampling test clips, energy-based activity detection. |
| Concurrency | `asyncio` for I/O; one dedicated thread for model inference (later) | Keeps the event loop free. Torch releases the GIL inside kernels. |
| Config | `pydantic-settings` (env vars) | `ENGINE=echo\|model`, `ECHO_DELAY_MS`, `LOG_LEVEL`, `FINISH_WATCHDOG_S`. |
| Logging / metrics | stdlib `logging` (JSON lines), `time.perf_counter` | Per-frame timestamps and an event-loop-lag canary. |
| Test client | `websockets` (asyncio client) | Closest to a typical harness client. |
| Tests | `pytest`, `pytest-asyncio` | Protocol-rule assertions and fault injection. |
| Analysis / plots | `numpy`, `matplotlib` | Delay demo figures. |
| VAD (analysis only) | Silero VAD, or a simple RMS threshold | Used to find first and last speech in clips. Not in the serving path initially. |
| Packaging | `uv` (or `pip-tools`) with a lockfile | Reproducible pinned builds. |
| Container | Docker: `python:3.11-slim` now, CUDA runtime image later | See [§6](#6-docker). |
| Lint | `ruff` | Optional but cheap. |

**Pin exact versions** of `uvicorn`, `starlette`, and the WebSocket backend (`websockets` or `wsproto`). Behaviours such as ping handling and max message size differ across versions. Verify them on the versions you pin.

---

## 3. Repository layout

```
sarvam-s2s-server/
├── Dockerfile                  # echo / CPU image (Phase 1)
├── Dockerfile.gpu              # CUDA image (after checkpoint release)
├── .dockerignore
├── pyproject.toml
├── uv.lock
├── app/
│   ├── main.py                 # FastAPI app: /health, /ws, startup warm-up
│   ├── protocol.py             # constants (24000, 3840, 48000), pydantic models
│   ├── session.py              # per-connection state machine, reader/writer tasks
│   ├── audio.py                # PCM <-> numpy, frame slicing, trim to valid_samples
│   ├── metrics.py              # timestamps, loop-lag canary, JSONL session log
│   ├── config.py               # env-driven settings
│   └── engine/
│       ├── base.py             # Engine protocol
│       ├── echo.py             # delayed echo (demo + baseline)
│       ├── fake_model.py       # slow fake "model" for threading tests
│       └── model.py            # adapter for the real checkpoint (stub now)
├── tools/
│   ├── make_clip.py            # any audio -> 24 kHz mono s16 test clips
│   ├── client.py               # paced, harness-like client; records arrival times
│   ├── analyze.py              # playback-timeline simulation, metrics, plots
│   ├── harness_sim.py          # N sequential sessions, 2 s gaps, rule checks
│   └── stress.py               # fault injection
├── tests/
│   ├── test_protocol.py
│   ├── test_session.py
│   └── test_engine_echo.py
├── clips/                      # your recorded/test clips (check licensing/consent)
└── RESULTS.md                  # every number + the command that reproduces it
```

---

## 4. Server architecture

### 4.1 Per-connection flow

```
accept
  └─ wait for session_start (JSON text)
       └─ engine.start(session_id)
            ├─ reader task : ws.receive() loop
            │      binary → engine.push(frame, t_arrival)
            │      text   → input_end → engine.end_input(valid_samples)
            ├─ writer task : engine output → ws.send_bytes(chunk ≤ 48,000 B)
            └─ watchdog    : forces completion at input_end + ~25 s
       └─ engine finished → send session_complete once → close
       └─ finally: cancel tasks, engine.reset(), write session summary log
```

### 4.2 Session state machine

```
WAIT_START → STREAMING → INPUT_ENDED → DRAINING → COMPLETE
```

- Unexpected messages in a given state are logged and ignored. They are not fatal.
- A connection closing early at any state goes straight to cleanup.

### 4.3 Implementation rules

1. **One receive loop.** Use `await ws.receive()` and branch on `"bytes"` vs `"text"`. This handles both frame types and the disconnect message in one place.
2. **Reader does minimal work.** Parse, timestamp (`time.perf_counter()`), enqueue. No slow `await` anywhere in it.
3. **Keep reading after `input_end`.** The harness sends silence frames until `session_complete`. An unread socket backs up and fails the session.
4. **Trim padding.** The last real frame is zero-padded. Use `valid_samples` from `input_end` to cut the real audio length, so padding is not treated as speech.
5. **Central `emit()` function.** All outbound audio passes through one function that enforces ≤ 48,000 B and even byte length (whole samples), and raises on violation in tests.
6. **Single-shot `session_complete`.** Guard with a flag (and a lock if multiple tasks may trigger it). Close the socket right after sending.
7. **Watchdog.** If the engine has not finished by `input_end + 25 s`, send `session_complete` anyway. Truncated output scores better than a failed session.
8. **Serialise sessions.** `asyncio.Semaphore(1)` so a stuck handler from a previous session cannot contaminate the next one.
9. **`finally:` always runs.** Cancel remaining tasks, call `engine.reset()`, log the summary.

```python
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(ws, engine)
    tasks = {
        asyncio.create_task(session.read_loop()),
        asyncio.create_task(session.write_loop()),
    }
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        await session.cleanup(tasks)   # cancel rest, reset engine, log summary
```

### 4.4 Engine interface

This is the only part that changes when the real model arrives.

```python
class Engine(Protocol):
    def start(self, session_id: str) -> None: ...
    def push(self, pcm: bytes, t_arrival: float) -> None: ...   # non-blocking
    def end_input(self, valid_samples: int) -> None: ...
    def output(self) -> AsyncIterator[bytes]: ...               # chunks ≤ 1 s, even length
    def reset(self) -> None: ...
```

### 4.5 Echo engine (the delay demo)

- Each pushed frame gets `release_time = t_arrival + D`, where `D` is `ECHO_DELAY_MS`.
- A pump coroutine sleeps until the head frame's release time, then yields it.
- **Schedule by arrival timestamps, not by counting frames.** The echo then stays correct even if the reader jitters.
- Done condition: input ended and queue empty. Expected tail after the speaker stops is about `D`.

```python
def push(self, pcm: bytes, t_arr: float) -> None:      # reader side, non-blocking
    self.q.append((t_arr + self.delay, pcm))
    self.wake.set()

async def output(self):                                 # writer side
    while True:
        release, pcm = await self._next()
        await asyncio.sleep(max(0.0, release - loop.time()))
        yield pcm
```

### 4.6 Real-model adapter (Phase 5, after checkpoint release)

> The following design is an assumption based on typical streaming speech models. Confirm against the released inference code.

- **Threading.** Inference runs in **one dedicated thread** that owns the GPU and all model state (KV cache, codec state). The reader hands frames over through a thread-safe queue. Output returns via `loop.call_soon_threadsafe` into an `asyncio.Queue`.
- **Stepping.** If the model is frame-synchronous: one 80 ms input frame → one step → whatever audio that step yields (possibly silence).
- **Emit per step.** Do not accumulate multiple steps before sending. Batching adds lag directly to your latency score.
- **Finishing rule.** After `end_input`, keep stepping with silence until the model signals end or an idle-silence threshold trips. Tune this to balance tail latency against cut-off words.
- **Warm-up.** Before `/health` returns 200, run a dummy session to trigger CUDA graph capture, `torch.compile`, cuDNN autotune.
- **Testing without a GPU.** The development laptop has no discrete GPU. Test the model path on a Modal H100 container. Keep the echo path as the local CPU baseline, and use `fake_model.py` (emits tone bursts after a configurable per-step compute time) to validate threading before the checkpoint exists.

---

## 5. Measurement tooling and the delay demo

### 5.1 `tools/make_clip.py`

Convert any recording to the wire format: mono, 24 kHz, 16-bit PCM.

```bash
ffmpeg -i input.m4a -ac 1 -ar 24000 -sample_fmt s16 clips/c01.wav
```

Also generate variants: telephone-band (300–3400 Hz band-limit then upsample), noisy (additive noise at a set SNR), and a clip with a long internal pause.

### 5.2 `tools/client.py` (harness-like client)

1. Load a clip, slice into 3,840 B frames, zero-pad the final frame, record `valid_samples`.
2. Send `session_start` JSON.
3. Send frames on an **absolute schedule** so sleep errors do not accumulate:

```python
t0 = time.perf_counter()
for k, frame in enumerate(frames):
    await asyncio.sleep(max(0.0, t0 + k * 0.08 - time.perf_counter()))
    await ws.send(frame)
```

4. Send `input_end` right after the last frame, then continue sending silent frames at the same cadence until `session_complete` arrives.
5. A **separate receive task** records `(t_arrival, nbytes)` for every server frame, relative to `t0`. It must not share a coroutine with the sender.
6. Dump a JSON trace for `analyze.py`.

### 5.3 `tools/analyze.py` (playback-timeline simulator)

Reconstruct what the listener hears using the PS rule: *a frame starts when it arrives or when the previous finishes, whichever is later.*

```python
cursor = 0.0
for t_arr, pcm in chunks:
    start = max(t_arr, cursor)
    cursor = start + len(pcm) / 2 / 24000
    timeline.append((start, cursor, pcm))
```

**Metrics to report**

| Metric | Definition |
|---|---|
| First-audio delay | Playback time of first non-silent output minus time of first non-silent input |
| Steady-state lag | Offset that maximises cross-correlation of input and output amplitude envelopes (for the echo this should equal `D`) |
| Tail latency | End of output playback minus end of valid input (`valid_samples / 24000`) |
| Completion time | `session_complete` arrival minus `input_end` send time |
| Rule checks | Every server frame ≤ 48,000 B and even length; exactly one `session_complete`; nothing after it; completion within 30 s |

**Plot:** input and output waveforms on one time axis with the measured offset annotated. This is the showcase figure.

**Experiments to run**

- Clip lengths: 10 s, 30 s, 60 s.
- A clip with long pauses, and a noisy or telephone-band copy.
- Sweep `D` in {0.5, 1, 2, 4} s and tabulate configured vs measured delay. Report the deviation rather than assuming it is zero.

### 5.4 `tools/harness_sim.py`

Runs N sessions back to back with 2 s gaps and asserts all protocol rules each time. This catches state leaking between sessions.

---

## 6. Docker

### 6.1 Phase 1: echo image (CPU)

- Base: `python:3.11-slim`, pinned by digest.
- Copy the lockfile first and install dependencies, then copy `app/` (layer caching).
- Env: `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`.
- Run as a non-root user. `EXPOSE 8000`.
- **Default entrypoint** starts uvicorn on `0.0.0.0:8000`. The PS requires the default entrypoint to start the server, so do not depend on run-time arguments.
- `HEALTHCHECK` via Python `urllib` (slim has no `curl`). Optional.
- `.dockerignore`: `tools/`, `tests/`, clips, caches, `.git`.

```dockerfile
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev
COPY app/ app/
RUN useradd -m app && chown -R app /srv
USER app
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "0"]
```

> Check the exact uvicorn flag names for your pinned version. The `--ws-ping-interval` option exists in recent uvicorn releases, but confirm before relying on it.

### 6.2 Phase 2: GPU image (after checkpoint release)

- Base: an NVIDIA CUDA runtime image with cuDNN, or the official PyTorch runtime image, matched to the torch wheel you pin. Verify driver compatibility on the Modal H100 and against the organisers' stated hardware.
- Bake weights as a **separate layer** (`COPY weights/ /models/`) so code edits do not rebuild a multi-GB layer.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Nothing may be downloaded at runtime.
- Multi-stage build if anything is compiled. There is no image-size limit per the FAQ, but build time matters when you iterate.
- Local run: `docker run --rm -p 8000:8000 --gpus all <image>`.
- Submission: `.tar` via `docker save`, or a registry digest.

### 6.3 Container acceptance checklist

- [ ] `GET /health` returns 200 within 300 s of start (measure the cold start)
- [ ] WebSocket accepted at `/ws` on port 8000
- [ ] Runs with `--network none` (proves no outbound dependency)
- [ ] `harness_sim.py` passes against the container
- [ ] First session after cold start meets the same timing as later ones (warm-up works)
- [ ] Default entrypoint works with no extra arguments

---

## 7. Failure modes and prevention

This table is the draft answer to "what could make it fall behind, and how would you prevent it?"

| Failure | Symptom | Prevention |
|---|---|---|
| CPU or GPU work inside a coroutine | Event loop stalls, reader misses the 1 s deadline, session fails | Inference in a dedicated thread. Reader only enqueues. Event-loop-lag canary (a task sleeping 10 ms and logging overshoot). |
| Reading and sending in the same loop | A blocked `send` stops reads | Separate reader and writer tasks. |
| Not reading after `input_end` | Socket backs up, harness cannot deliver frames | Reader runs until the socket closes. |
| Model slower than real time (RTF > 1) | Backlog and lag grow without bound | Measure per-step RTF. This is a model/hardware problem. Never drop input silently. Consider smaller or quantised variants, or speculative decoding if allowed. |
| Cold start or first-session compile | First session stalls or times out | Warm-up before `/health` returns 200. |
| State leak across sessions | Degraded output on later sessions | `engine.reset()` in `finally`. Covered by `harness_sim.py`. |
| Ping/pong timeouts | Early close counts as a failure | Raise or disable server ping interval/timeout. Verify the option for the pinned version. |
| Output chunk > 1 s, odd byte length, or WAV header | Rule violation or audible click | Single `emit()` that enforces size and alignment. |
| Double or late `session_complete`, or audio after it | Format-rule failure | Single-shot flag, close immediately after sending. |
| GC pauses, per-frame copies | Jitter | Use `memoryview`/preallocated buffers. Consider `gc.freeze()` after warm-up. At 12.5 frames/s this is minor, so measure before optimising. |
| Stuck previous connection | Next session rejected or contaminated | Per-session timeouts, forced cleanup, serialised connections. |
| End-of-input never detected by the model | No `session_complete`, 30 s limit breached | Watchdog at ~25 s that finalises and sends `session_complete`. |
| Output hold-back that peeks at future audio | Causality violation (disqualification) | Hold-back logic may use only audio already received. Review any "stability" or "agreement" logic for this. |

---

## 8. Fault-injection and test plan

Implemented in `tools/stress.py` and `tests/`.

**Protocol tests**
- Valid session end to end. Assert all rules from §1.
- `input_end` with `valid_samples` = 0, and not a multiple of 1,920.
- Malformed JSON, unknown message type, wrong-size binary frame. The server must not crash, and the session behaviour must be defined.
- Missing `input_end` (client stalls). Watchdog behaviour.

**Concurrency and load**
- Client disconnects mid-session, then a new session runs cleanly.
- Two simultaneous connections. Verify serialisation or rejection.
- Burst: frames sent faster than real time.
- Artificially slow engine (sleep in the engine thread). Reader must still keep up and the event-loop-lag canary must stay low.

**Endurance**
- 60 s sessions.
- 100 back-to-back sessions. Check for memory growth and leaked tasks.

**Pass criteria (set before running, record in RESULTS.md)**
- Zero session failures against the rule checks.
- Event-loop lag p99 below a threshold you define (for example 20 ms) in all scenarios.
- Measured echo delay within a tolerance of configured `D` that you define in advance.

---

## 9. Phased execution plan

| Phase | Goal | Outputs |
|---|---|---|
| **P0: Protocol skeleton** | `/health`, `/ws` accepting `session_start`, echo with fixed delay, `client.py` | Working local echo round trip |
| **P1: Measurement** | `analyze.py`, delay sweep, plots | First `RESULTS.md` entries, showcase figure |
| **P2: Container** | Dockerfile, client against container, `--network none` test, cold-start timing | Verified image |
| **P3: Hardening** | Fault-injection suite, watchdog, loop-lag canary, `harness_sim.py` | Failure-mode evidence for the report |
| **P4: Model hook** | `Engine` adapter, threading model, warm-up, reset. Validate with `fake_model.py` | Threading proven before the checkpoint exists |
| **P5: Real checkpoint** | Integrate, run the official reference client and container check (unlimited attempts), tune chunk size and finishing rule | Passing container check, latency numbers |

---

## 10. Open questions for the organisers

These cannot be resolved from the PS. Send them through the doubt form or Discord.

1. Should `/health` return a non-200 (e.g. 503) while the model loads, or should the port stay closed until ready?
2. Does the harness send WebSocket pings, and does it answer server pings?
3. How does the harness define the end of speech when computing "how soon it finishes after the speaker stops" (trailing-silence threshold)?
4. If the server sends partial output and an early `session_complete`, is that scored only on content loss, or penalised further?
5. Is a session that takes longer than 30 s from `input_end` failed outright, or scored with the late audio?

---

## 11. RESULTS.md template

The mini-PS requires every number to list the command that reproduces it.

```markdown
# RESULTS

| # | Metric | Value | Clip | Config | Command |
|---|--------|-------|------|--------|---------|
| 1 | Measured echo delay | x.xx s | c01.wav | D=2.0 s | `python tools/client.py clips/c01.wav --delay 2000 && python tools/analyze.py out/c01.json` |
| 2 | First-audio delay | ... | | | |
| 3 | Tail latency | ... | | | |
| 4 | Event-loop lag p99 | ... | | | `python tools/stress.py --scenario slow_engine` |
| 5 | Cold start to /health 200 | ... | | | `docker run ... && python tools/wait_health.py` |
```

---

## 12. Quick reference: constants and byte math

| Quantity | Value |
|---|---|
| Sample rate | 24,000 Hz |
| Encoding | pcm_s16le, mono (2 bytes per sample) |
| Input frame | 80 ms = 1,920 samples = **3,840 bytes** |
| Frame rate | 12.5 frames/s |
| Max server frame | 1 s = 24,000 samples = **48,000 bytes** |
| Keepup deadline | 1 s past scheduled delivery |
| Completion deadline | 30 s after `input_end` (plan for ~25 s watchdog) |
| Connection accept deadline | 10 s |
| `/health` deadline | 300 s after start |
| Port / path | 8000 / `/ws` |
| Session gap | 2 s, sequential sessions |

```python
SAMPLE_RATE = 24_000
BYTES_PER_SAMPLE = 2
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000      # 1920
FRAME_BYTES = FRAME_SAMPLES * BYTES_PER_SAMPLE      # 3840
MAX_OUT_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE      # 48000 (1 s)
```
