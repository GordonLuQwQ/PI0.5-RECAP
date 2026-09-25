# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packet encoding and trajectory recording for PiperX evaluation."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

BRIDGE_VERSION = "piperx-pi05-http-v1"


MAX_PACKET_BYTES = 32 * 1024 * 1024


def encode_packet(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> bytes:
    """Encode arrays and JSON metadata as a bounded, pickle-free packet."""
    if "_metadata" in arrays:
        raise ValueError("Reserved array name: _metadata")
    header = np.frombuffer(
        json.dumps(metadata, allow_nan=False).encode("utf-8"), dtype=np.uint8
    )
    buffer = io.BytesIO()
    # Uncompressed NPZ avoids encoding a 640x480 camera with Python JSON lists.
    np.savez(buffer, _metadata=header, **arrays)
    body = buffer.getvalue()
    if len(body) > MAX_PACKET_BYTES:
        raise ValueError("Bridge packet exceeds 32 MiB")
    return body


def decode_packet(body: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Decode a packet and reject invalid sizes, metadata, or versions."""
    if not body or len(body) > MAX_PACKET_BYTES:
        raise ValueError("Empty or oversized bridge packet")
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(entries) > 8 or len(set(names)) != len(names):
                raise ValueError("Too many or duplicate packet arrays")
            if sum(entry.file_size for entry in entries) > MAX_PACKET_BYTES:
                raise ValueError("Unpacked bridge packet exceeds 32 MiB")
        with np.load(io.BytesIO(body), allow_pickle=False) as packet:
            header = packet["_metadata"]
            if header.dtype != np.uint8 or header.ndim != 1 or header.size > 65536:
                raise ValueError("Invalid metadata encoding")
            metadata = json.loads(header.tobytes().decode("utf-8"))
            arrays = {key: packet[key] for key in packet.files if key != "_metadata"}
    except (KeyError, OSError, zipfile.BadZipFile, UnicodeError) as exc:
        raise ValueError("Invalid NPZ bridge packet") from exc
    if not isinstance(metadata, dict) or metadata.get("version") != BRIDGE_VERSION:
        raise ValueError("Unsupported bridge protocol version")
    return arrays, metadata


def validate_observation(arrays: dict[str, np.ndarray], cameras: Sequence[str]) -> None:
    """Require measured robot state and the requested RGB cameras."""
    if set(arrays) != {"state", *cameras}:
        raise ValueError(
            f"Expected only state and cameras {list(cameras)}; got {list(arrays)}"
        )
    state = np.asarray(arrays["state"])
    if (
        state.ndim != 2
        or state.shape[0] == 0
        or state.shape[1] != 7
        or state.dtype.kind != "f"
        or not np.isfinite(state).all()
    ):
        raise ValueError(
            "State must be finite floating-point (B, 7): q1..q6 radians + closure"
        )
    if not np.logical_and(state[:, 6] >= 0, state[:, 6] <= 1).all():
        raise ValueError("Measured gripper closure must be in [0, 1]")
    for name in cameras:
        rgb = arrays[name]
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 4
            or rgb.shape[0] != state.shape[0]
            or rgb.shape[-1] != 3
            or not all(8 <= size <= 2048 for size in rgb.shape[1:3])
        ):
            raise ValueError(f"{name} must be uint8 RGB with shape (B, H, W, 3)")


def write_json(path: str | Path, value: Any) -> None:
    """Write finite JSON metadata with UTF-8 text preserved."""
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def new_trace() -> dict[str, list]:
    """Create the per-episode buffers consumed by the simulator."""
    return {
        "state": [],
        "actions_requested": [],
        "actions_applied": [],
        "wall_action_seconds": [],
        "chunk_start_steps": [],
        "requests": [],
    }


def expected_video_frame_count(control_steps: int, video_stride: int) -> int:
    """Return initial frame plus one sampled frame through the final step."""
    return 1 + (control_steps + video_stride - 1) // video_stride


def save_trace(
    root: str | Path, trace: dict[str, list], final_state: np.ndarray
) -> int:
    """Save the episode trace and return the number of clipped action steps."""
    arrays = {
        key: np.asarray(trace[key], dtype=np.float32).reshape(-1, 7)
        for key in ("state", "actions_requested", "actions_applied")
    }
    arrays["wall_action_seconds"] = np.asarray(
        trace["wall_action_seconds"], dtype=np.float64
    )
    arrays["chunk_start_steps"] = np.asarray(trace["chunk_start_steps"], dtype=np.int64)
    arrays["final_state"] = np.asarray(final_state, dtype=np.float32)
    np.savez_compressed(Path(root) / "trajectory.npz", **arrays)
    write_json(Path(root) / "requests.json", trace["requests"])
    requested, applied = arrays["actions_requested"], arrays["actions_applied"]
    return int(np.any(np.abs(requested - applied) > 1e-6, axis=1).sum())
