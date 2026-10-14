"""
Protocol constants and pydantic models for the S2ST WebSocket server.

All wire-format constants derived from §12 of the implementation plan.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Audio constants ──────────────────────────────────────────────────────────

SAMPLE_RATE = 24_000                                        # Hz
BYTES_PER_SAMPLE = 2                                        # pcm_s16le
FRAME_MS = 80                                               # ms per input frame
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000              # 1920
FRAME_BYTES = FRAME_SAMPLES * BYTES_PER_SAMPLE              # 3840
MAX_OUT_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE              # 48000 (1 s)

# ── Timing constants ────────────────────────────────────────────────────────

KEEPUP_DEADLINE_S = 1.0       # frame must arrive within 1 s of schedule
COMPLETION_DEADLINE_S = 30.0  # session_complete must arrive within 30 s of input_end
WATCHDOG_S = 25.0             # default watchdog fires at input_end + 25 s
ACCEPT_DEADLINE_S = 10.0      # connection must be accepted within 10 s
HEALTH_DEADLINE_S = 300.0     # /health must return 200 within 300 s of startup
SESSION_GAP_S = 2.0           # gap between sequential sessions

# ── Protocol message types (text frames) ────────────────────────────────────

MSG_SESSION_START = "session_start"
MSG_INPUT_END = "input_end"
MSG_SESSION_COMPLETE = "session_complete"


# ── Pydantic models for incoming text messages ──────────────────────────────

class SessionStartMessage(BaseModel):
    """Client → Server: signals the start of a new session."""
    type: str = Field(..., pattern=f"^{MSG_SESSION_START}$")
    session_id: str = Field(..., min_length=1)


class InputEndMessage(BaseModel):
    """Client → Server: signals the end of the real audio input."""
    type: str = Field(..., pattern=f"^{MSG_INPUT_END}$")
    valid_samples: int = Field(..., ge=0)


class SessionCompleteMessage(BaseModel):
    """Server → Client: signals the session is finished."""
    type: str = MSG_SESSION_COMPLETE


def parse_text_message(raw: str) -> SessionStartMessage | InputEndMessage | None:
    """
    Parse an incoming text-frame JSON string.

    Returns a typed model on success, None if the message is malformed
    or has an unrecognised type.  Never raises — the server must not crash
    on bad input.
    """
    import json

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    msg_type = data.get("type")
    try:
        if msg_type == MSG_SESSION_START:
            return SessionStartMessage(**data)
        elif msg_type == MSG_INPUT_END:
            return InputEndMessage(**data)
    except Exception:
        return None

    return None
