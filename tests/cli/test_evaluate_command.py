"""Tests for the standalone evaluate command helpers."""

import json

import numpy as np

from cli.commands.evaluate_command import _json_default


def test_json_default_serializes_numpy_bool() -> None:
    payload = {"ok": np.bool_(True), "nested": [{"flag": np.bool_(False)}]}

    encoded = json.dumps(payload, default=_json_default)

    assert json.loads(encoded) == {"ok": True, "nested": [{"flag": False}]}
