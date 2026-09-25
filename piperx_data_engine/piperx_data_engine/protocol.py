"""Shared task definition; no simulator or model dependency.

Scene seeds determine geometry only, never the instruction. Evaluation therefore
uses exactly the same scene for opposite instructions and execution variants.
"""

import json
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

OBJECTS = ("red cube", "red cylinder", "blue cube")
PAIRS = tuple(permutations(range(3), 2))
HELD_OUT = (0, 1)
TRAIN_PAIRS = tuple(pair for pair in PAIRS if pair != HELD_OUT)
REVERSAL_PAIRS = ((1, 2), (2, 1))
TASK_SETS = ("benchmark", "all")


def training_pairs(task_set="benchmark"):
    """Keep the five-pair benchmark split distinct from six-pair training."""
    if task_set not in TASK_SETS:
        raise ValueError(f"Unknown task set: {task_set!r}; choose one of {TASK_SETS}")
    return PAIRS if task_set == "all" else TRAIN_PAIRS


@dataclass(frozen=True)
class Protocol:
    version: str = "piperx-stacking-v1"
    table_top: float = 0.02
    object_height: float = 0.04
    object_width: float = 0.04
    x_range: tuple[float, float] = (0.24, 0.40)
    y_range: tuple[float, float] = (-0.16, 0.16)
    min_center_distance: float = 0.10
    control_hz: int = 20
    physics_dt: float = 0.005
    hold_seconds: float = 2.0
    max_seconds: float = 30.0
    stack_xy_tolerance: float = 0.008
    stack_z_tolerance: float = 0.003
    upright_degrees: float = 10.0
    destination_displacement: float = 0.015
    third_displacement: float = 0.005
    third_rotation_degrees: float = 5.0
    max_linear_speed: float = 0.015
    max_angular_speed: float = 0.20
    retreat_distance: float = 0.12


DEFAULT_PROTOCOL = Protocol()


def instruction(pair):
    if tuple(pair) not in PAIRS:
        raise ValueError("Source and destination must be different task objects")
    return f"put the {OBJECTS[pair[0]]} on the {OBJECTS[pair[1]]}"


def sample_scene(seed: int, protocol: Protocol = DEFAULT_PROTOCOL):
    """Randomize upright objects' XY and yaw independently of their roles."""
    rng = np.random.default_rng(seed)
    positions = []
    for _ in OBJECTS:
        for _ in range(10000):
            point = np.array([rng.uniform(*protocol.x_range), rng.uniform(*protocol.y_range)])
            if all(np.linalg.norm(point - p[:2]) >= protocol.min_center_distance for p in positions):
                positions.append(np.r_[point, protocol.table_top + protocol.object_height / 2 + 0.001])
                break
        else:
            raise RuntimeError("Could not sample separated objects in the declared workspace")
    yaw = rng.uniform(-np.pi, np.pi, 3)
    quat = np.zeros((3, 4))
    quat[:, 0], quat[:, 3] = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.asarray(positions), quat


def manifest(protocol: Protocol = DEFAULT_PROTOCOL, *, task_set="benchmark"):
    pairs = training_pairs(task_set)
    return {
        "protocol": asdict(protocol),
        "task_set": task_set,
        "objects": OBJECTS,
        "train_pairs": [instruction(pair) for pair in pairs],
        "held_out_pair": None if task_set == "all" else instruction(HELD_OUT),
        "held_out_reason": (
            "All six ordered pairs are allowed in training; there is no held-out task pair. "
            "Evaluation on new scene seeds measures scene generalization, not unseen-pair generalization."
            if task_set == "all"
            else "Red cube on red cylinder is reserved for compositional generalization. Each object occurs in both roles in the other five training pairs; blue cube on red cylinder covers cube-on-cylinder stacking, and both reversal directions remain seen."
        ),
        "training_scene_seeds": list(range(1000, 2000)),
        "validation_scene_seeds": list(range(8000, 8020)),
        "evaluation_scene_seeds": list(range(10000, 10050)),
        "reversal_scene_seeds": list(range(20000, 20020)),
        "reversal_pairs": [instruction(pair) for pair in REVERSAL_PAIRS],
        "orientation_randomization": "Independent uniform yaw in [-pi, pi]; initial roll and pitch are zero.",
        "evaluation_rule": "All six pairs, same 50 seeds per pair, no rejected or replacement trials.",
        "validation_rule": (
            "All six pairs may be validated on the separate validation scene seeds."
            if task_set == "all"
            else "Only the five training pairs; never select a checkpoint using held-out results."
        ),
        "training_target": (
            "Balanced successful episodes per selected pair; actual count is declared in collection.json."
            if task_set == "all"
            else "Pilot first; proposed main dataset: 200 successful episodes per training pair."
        ),
        "results_status": "NOT RUN: seed lists and episode counts above are protocol, not measured results.",
    }


def write_manifest(path: Path, protocol: Protocol = DEFAULT_PROTOCOL, *, task_set="benchmark"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(protocol, task_set=task_set), indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write the declared stacking protocol and explicit seed lists")
    parser.add_argument("--output", type=Path, default=Path("vla/stacking/protocol.json"))
    parser.add_argument("--task-set", choices=TASK_SETS, default="benchmark")
    args = parser.parse_args()
    write_manifest(args.output, task_set=args.task_set)
    print(args.output)
