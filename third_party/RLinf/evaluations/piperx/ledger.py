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

"""Only atomically committed, uniquely identified completed trials enter success rates."""

import fcntl
import html
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

from bridge import expected_video_frame_count
from settings import (
    CAMERAS,
    CHECKPOINT,
    CYLINDER_X_RANGE,
    EXCLUDED_TASK,
    OUTPUT,
    PAIRS,
    PARALLEL_ENVS,
    PIPERX,
    PLACEMENT_RULE,
    RTC_CONFIG,
    SEEDS,
    SUITE,
    TOTAL_TRIALS,
    VIDEO_CAMERAS,
    VIDEO_FAILURE_ONLY,
    VIDEO_STRIDE,
)

OBJECTS = ("red cube", "red cylinder", "blue cube")


def instruction(pair):
    return f"put the {OBJECTS[pair[0]]} on the {OBJECTS[pair[1]]}"


def slug(pair, seed):
    return f"pair{pair[0]}{pair[1]}_seed{seed}"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def evaluation_specification():
    specification = {
        "checkpoint": str(CHECKPOINT),
        "mode": "rtc" if RTC_CONFIG["enabled"] else "chunk",
        "cameras": CAMERAS,
        "pairs": PAIRS,
        "seeds": SEEDS,
        "success_rule": PLACEMENT_RULE,
        "rtc": RTC_CONFIG,
        "suite": SUITE,
        "cylinder_x_range_m": CYLINDER_X_RANGE,
        "reset_rule": "persisted_genesis_state_v1",
        "fast_chunk_step": bool(PIPERX.get("fast_chunk_step", False)),
        "video_stride": VIDEO_STRIDE,
        "video_cameras": VIDEO_CAMERAS,
        "video_failure_only": VIDEO_FAILURE_ONLY,
        "parallel_envs": PARALLEL_ENVS,
        "model": PIPERX["model"],
        "calibration": PIPERX["calibration"],
        "entrypoint": "evaluations/eval_embodied_agent.py",
    }
    return json.loads(json.dumps(specification))


def ensure_manifest(root=OUTPUT):
    root.mkdir(parents=True, exist_ok=True)
    spec = evaluation_specification()
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        assert manifest["specification"] == spec, (
            "Evaluation configuration changed; use a new output directory"
        )
    else:
        # Persist a run ID so completed trials can be counted across restarts.
        manifest = {"signature": str(uuid4()), "specification": spec}
        write_json(path, manifest)
    return manifest["signature"]


def completed_trials(root, digest):
    rows = []
    for pair in PAIRS:
        for seed in SEEDS:
            directory = root / slug(pair, seed)
            path = directory / "report.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            if report.get("status") != "completed":
                continue
            assert report["signature"] == digest
            assert report["pair"] == list(pair) and report["scene_seed"] == seed
            assert report["instruction"] == instruction(pair)
            assert (directory / "trajectory.npz").is_file() and (
                directory / "requests.json"
            ).is_file()
            expected_video_names = (
                set()
                if VIDEO_FAILURE_ONLY and report["task_result"]["success"]
                else set(VIDEO_CAMERAS)
            )
            assert set(report["videos"]) == expected_video_names
            for video in report["videos"].values():
                assert (directory / video["file"]).is_file()
                assert video["frames"] == expected_video_frame_count(
                    report["executed_control_steps"], VIDEO_STRIDE
                )
            rows.append(
                {
                    "directory": directory.name,
                    "pair": list(pair),
                    "scene_seed": seed,
                    "instruction": report["instruction"],
                    "success": bool(report["task_result"]["success"]),
                    "failure": report["task_result"]["failure"],
                    "held_out": report["instruction"] == EXCLUDED_TASK,
                    "executed_control_steps": report["executed_control_steps"],
                }
            )
    return rows


def prepare_trial(root, pair, seed):
    directory = root / slug(pair, seed)
    if directory.exists():
        report_path = directory / "report.json"
        if (
            report_path.exists()
            and json.loads(report_path.read_text()).get("status") == "completed"
        ):
            raise RuntimeError("A completed trial must never be overwritten or rerun")
        archive = root / "incomplete_attempts"
        archive.mkdir(exist_ok=True)
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        shutil.move(str(directory), archive / f"{directory.name}_{suffix}")
    directory.mkdir()
    return directory


def update_summary(root, digest, **extra):
    """Update aggregate files while serializing concurrent simulator writers."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".summary.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _update_summary_unlocked(root, digest, **extra)


def _update_summary_unlocked(root, digest, **extra):
    rows = completed_trials(root, digest)
    groups = []
    for pair in PAIRS:
        subset = [row for row in rows if row["pair"] == list(pair)]
        successes = sum(row["success"] for row in subset)
        groups.append(
            {
                "pair": list(pair),
                "instruction": instruction(pair),
                "held_out": instruction(pair) == EXCLUDED_TASK,
                "completed": len(subset),
                "planned": len(SEEDS),
                "successes": successes,
                "success_rate": successes / len(subset) if subset else None,
                "failure_counts": dict(
                    Counter(row["failure"] for row in subset if not row["success"])
                ),
            }
        )
    result = {
        "status": "completed" if len(rows) == len(PAIRS) * len(SEEDS) else "running",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "signature": digest,
        "completed_trials": len(rows),
        "planned_trials": len(PAIRS) * len(SEEDS),
        "controller": "rtc" if RTC_CONFIG["enabled"] else "chunk",
        "suite": SUITE,
        "cylinder_x_range_m": CYLINDER_X_RANGE,
        "success_rule": PLACEMENT_RULE,
        "groups": groups,
        "trials": rows,
        **extra,
    }
    for key, held_out in (("trained_tasks", False), ("held_out_task", True)):
        subset = [row for row in rows if row["held_out"] == held_out]
        result[key] = {
            "completed": len(subset),
            "successes": sum(row["success"] for row in subset),
            "success_rate": sum(row["success"] for row in subset) / len(subset)
            if subset
            else None,
        }
    write_json(root / "summary.json", result)
    write_json(root / "results.json", rows)
    table = "".join(
        f"<tr><td>{html.escape(g['instruction'])}</td><td>{'held out' if g['held_out'] else 'training task'}</td>"
        f"<td>{g['successes']}/{g['completed']} (target {len(SEEDS)})</td><td>"
        f"{format(g['success_rate'], '.1%') if g['success_rate'] is not None else 'pending'}</td></tr>"
        for g in groups
    )
    trials = "".join(
        f"<li>{r['scene_seed']} · {html.escape(r['instruction'])} · "
        f"{'success' if r['success'] else html.escape(str(r['failure']))} · "
        f'<a href="{r["directory"]}/watch.html">replay</a></li>'
        for r in rows
    )
    (root / "watch.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>RLinf evaluation</title><style>body{font:16px system-ui;margin:30px;background:#101820;color:#eee}"
        f"td,th{{padding:12px}}a{{color:#8cf}}</style><h1>RLinf evaluation · {result['controller']} · resumable</h1>"
        f"<p>{SUITE} suite · completed {len(rows)}/{TOTAL_TRIALS}. Each trial commits its video and trace before its result, so resumed runs do not count it twice.</p>"
        "<p>Success means the released source rests on the requested destination. Simulation control runs at 20 Hz; wall-clock hard real time is not guaranteed.</p>"
        "<table><tr><th>Task</th><th>Split</th><th>Success/completed</th><th>Success rate</th></tr>"
        + table
        + "</table>"
        '<p><a href="summary.json">Complete result record</a></p><ol>'
        + trials
        + "</ol></html>"
    )
    (root.parent / "watch.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><h1>RLinf RTC evaluation</h1>'
        '<p><a href="standard/watch.html">Standard suite</a></p>'
        '<p><a href="reversal/watch.html">Fixed-scene reversal suite</a></p></html>'
    )
    return result
