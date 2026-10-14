"""
test_protocol.py (§8): Protocol model and parsing tests.
"""

from __future__ import annotations

import json

import pytest

from app.protocol import (
    BYTES_PER_SAMPLE,
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    MAX_OUT_BYTES,
    SAMPLE_RATE,
    InputEndMessage,
    SessionStartMessage,
    parse_text_message,
)


class TestConstants:
    """Verify all wire-format constants from §12."""

    def test_sample_rate(self):
        assert SAMPLE_RATE == 24_000

    def test_bytes_per_sample(self):
        assert BYTES_PER_SAMPLE == 2

    def test_frame_ms(self):
        assert FRAME_MS == 80

    def test_frame_samples(self):
        assert FRAME_SAMPLES == 1_920
        assert FRAME_SAMPLES == SAMPLE_RATE * FRAME_MS // 1000

    def test_frame_bytes(self):
        assert FRAME_BYTES == 3_840
        assert FRAME_BYTES == FRAME_SAMPLES * BYTES_PER_SAMPLE

    def test_max_out_bytes(self):
        assert MAX_OUT_BYTES == 48_000
        assert MAX_OUT_BYTES == SAMPLE_RATE * BYTES_PER_SAMPLE


class TestSessionStartMessage:
    def test_valid(self):
        msg = SessionStartMessage(type="session_start", session_id="abc-123")
        assert msg.type == "session_start"
        assert msg.session_id == "abc-123"

    def test_empty_session_id(self):
        with pytest.raises(Exception):
            SessionStartMessage(type="session_start", session_id="")

    def test_wrong_type(self):
        with pytest.raises(Exception):
            SessionStartMessage(type="wrong_type", session_id="abc")


class TestInputEndMessage:
    def test_valid(self):
        msg = InputEndMessage(type="input_end", valid_samples=1920)
        assert msg.valid_samples == 1920

    def test_zero_valid_samples(self):
        msg = InputEndMessage(type="input_end", valid_samples=0)
        assert msg.valid_samples == 0

    def test_negative_valid_samples(self):
        with pytest.raises(Exception):
            InputEndMessage(type="input_end", valid_samples=-1)


class TestParseTextMessage:
    def test_session_start(self):
        raw = json.dumps({"type": "session_start", "session_id": "test-1"})
        msg = parse_text_message(raw)
        assert isinstance(msg, SessionStartMessage)
        assert msg.session_id == "test-1"

    def test_input_end(self):
        raw = json.dumps({"type": "input_end", "valid_samples": 48000})
        msg = parse_text_message(raw)
        assert isinstance(msg, InputEndMessage)
        assert msg.valid_samples == 48000

    def test_malformed_json(self):
        assert parse_text_message("{not json") is None

    def test_not_a_dict(self):
        assert parse_text_message('"just a string"') is None
        assert parse_text_message("[1,2,3]") is None

    def test_unknown_type(self):
        raw = json.dumps({"type": "unknown_message"})
        assert parse_text_message(raw) is None

    def test_missing_fields(self):
        raw = json.dumps({"type": "session_start"})  # missing session_id
        assert parse_text_message(raw) is None

    def test_empty_string(self):
        assert parse_text_message("") is None

    def test_none_input(self):
        assert parse_text_message(None) is None
