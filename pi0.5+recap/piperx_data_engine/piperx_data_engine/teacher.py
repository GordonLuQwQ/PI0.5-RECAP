"""Feedback-checked physical stacking teacher, inspired by ManiSkill motion planning.

Grasp candidates and whole Cartesian paths are tested before execution. Measured
contacts and object poses gate transitions; only env.step applies physical motion.
"""

from dataclasses import dataclass

import numpy as np

from .env import DOWN_QUAT, HOME_TCP
from .planning import CartesianPlanner, compensate_placement, downward_quaternion, grasp_yaws
from .protocol import PAIRS


class TeacherFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class TeacherConfig:
    grasp_above_center: float = 0.014
    travel_clearance: float = 0.075
    arrival_tolerance: float = 0.006
    arrival_timeout: float = 1.0
    grasp_timeout: float = 0.8
    evidence_seconds: float = 0.10
    lift_min_height: float = 0.025
    dropped_distance: float = 0.065
    stability_seconds: float = 4.0


DEFAULT_TEACHER_CONFIG = TeacherConfig()


def demonstration(
    env,
    pairs,
    on_failure=None,
    events=None,
    config=DEFAULT_TEACHER_CONFIG,
):
    """Generate a checked stacking demonstration from the reset home state."""
    pairs = np.asarray(pairs, dtype=int)
    if pairs.shape != (env.num_envs, 2) or any(tuple(p) not in PAIRS for p in pairs):
        raise ValueError("One ordered source/destination pair is required per environment")
    events = [] if events is None else events
    rows = np.arange(env.num_envs)
    source, destination = pairs.T
    active = np.ones(env.num_envs, dtype=bool)
    stopped = env.last_action.copy()
    ticks = 0
    planner = CartesianPlanner(env)

    def emit(indices, event, **details):
        for i in indices:
            events.append({"env": int(i), "step": ticks, "event": event, **details})

    def fail(mask, reason, detail):
        ids = np.flatnonzero(mask & active)
        if not len(ids):
            return
        active[ids] = False
        stopped[ids, :6] = env.state()[ids, :6]
        stopped[ids, 6] = env.last_action[ids, 6]
        emit(ids, reason, detail=detail)
        if on_failure is None:
            raise TeacherFailure(f"{reason}: environments {ids.tolist()}; {detail}")
        on_failure(ids, reason, detail)

    def command(action, phase):
        nonlocal ticks
        if not np.any(active):
            return
        if ticks >= round(env.protocol.max_seconds * env.protocol.control_hz):
            fail(active.copy(), "teacher_timeout", phase)
            return
        action = action.copy()
        action[~active] = stopped[~active]
        ticks += 1
        yield action, phase

    def gate(condition, seconds, phase, reason, evidence=None):
        if not np.any(active):
            return
        required = max(1, round((config.evidence_seconds if evidence is None else evidence) * env.protocol.control_hz))
        streak = np.zeros(env.num_envs, dtype=int)
        for _ in range(max(required, round(seconds * env.protocol.control_hz))):
            good = condition(env.truth(contacts=True))
            streak = np.where(good, streak + 1, 0)
            if np.all((streak >= required) | ~active):
                emit(np.flatnonzero(active), phase + "_confirmed")
                return
            yield from command(env.last_action.copy(), phase)
        fail(streak < required, reason, phase)

    def move(target, orientation, closure, seconds, phase, *, carrying=False, support=None, from_table=False):
        if not np.any(active):
            return
        path, valid = planner.path(target, orientation, seconds, active)
        fail(~valid, "ik_failure", phase)
        clear = planner.collision_free(path, source, carrying=carrying, support=support, allow_table_support=from_table)
        fail(~clear, "path_collision", f"{phase}: {planner.last_collisions}")
        emit(np.flatnonzero(active), phase, planned_steps=len(path) - 1)
        for qpos in path[1:]:
            if not np.any(active):
                return
            if carrying and not from_table:
                truth = env.truth(contacts=True)
                distance = np.linalg.norm(truth["positions"][rows, source] - truth["tcp"], axis=-1)
                fail(distance > config.dropped_distance, "grasped_then_dropped", phase)
            action = np.c_[qpos[:, env.arm_qs], np.full(env.num_envs, closure)]
            yield from command(action, phase)
        yield from gate(
            lambda t: np.linalg.norm(t["tcp"] - target, axis=-1) <= config.arrival_tolerance,
            config.arrival_timeout,
            phase + "_arrival",
            "tracking_timeout",
        )

    def gripper(closure, phase):
        start = env.last_action.copy()
        emit(np.flatnonzero(active), phase)
        for i in range(env.protocol.control_hz):
            action = start.copy()
            action[:, 6] = start[:, 6] + (i + 1) / env.protocol.control_hz * (closure - start[:, 6])
            yield from command(action, phase)

    truth = env.truth(contacts=True)
    travel_height = (
        np.max(truth["positions"][:, :, 2] + env.protocol.object_height / 2, axis=-1) + config.travel_clearance
    )
    grasp = truth["positions"][rows, source].copy()
    grasp[:, 2] += config.grasp_above_center
    above = grasp.copy()
    above[:, 2] = travel_height
    yaws = grasp_yaws(truth["quaternions"][rows, source], source)
    orientation = np.tile(DOWN_QUAT, (env.num_envs, 1)).astype(float)
    selected = np.zeros(env.num_envs, dtype=bool)
    for candidate in range(yaws.shape[1]):
        pending = ~selected
        if not np.any(pending):
            break
        quat = downward_quaternion(yaws[:, candidate])
        approach_path, valid = planner.path(above, quat, 3, pending)
        descend_path, descend_valid = planner.path(
            grasp,
            quat,
            2,
            pending,
            start_qpos=approach_path[-1],
            start_tcp=above,
            start_orientation=quat,
        )
        valid &= descend_valid
        approach_clear = planner.collision_free(approach_path, source)
        descend_clear = planner.collision_free(descend_path, source)
        for i in np.flatnonzero(pending):
            emit(
                [i],
                "grasp_preflight",
                candidate=candidate,
                ik_valid=bool(valid[i]),
                approach_clear=bool(approach_clear[i]),
                descend_clear=bool(descend_clear[i]),
                collisions=planner.last_collisions[i],
            )
        valid &= approach_clear & descend_clear
        chosen = valid & pending
        orientation[chosen] = quat[chosen]
        selected |= chosen
        emit(np.flatnonzero(chosen), "grasp_candidate_selected", candidate=candidate)
    fail(~selected, "no_feasible_grasp", "Neither orthogonal grasp passed IK and collision preflight")

    yield from move(above, orientation, 0, 3, "approach_source")
    yield from move(grasp, orientation, 0, 2, "descend_to_grasp")
    yield from gripper(1, "close_gripper")
    yield from gate(
        lambda t: np.all(t["finger_contacts"][rows, source], axis=-1),
        config.grasp_timeout,
        "grasp_contact",
        "missed_grasp",
    )
    lift = env.truth()["tcp"].copy()
    lift[:, 2] = travel_height
    yield from move(lift, orientation, 1, 2, "lift", carrying=True, from_table=True)
    yield from gate(
        lambda t: (
            ((t["positions"][rows, source, 2] - env.initial["positions"][rows, source, 2]) >= config.lift_min_height)
            & np.all(t["finger_contacts"][rows, source], axis=-1)
        ),
        0.5,
        "lift_check",
        "failed_lift",
    )

    truth = env.truth()
    desired_source = truth["positions"][rows, destination].copy()
    desired_source[:, 2] = truth["positions"][rows, source, 2]
    transfer = compensate_placement(truth["tcp"], truth["positions"][rows, source], desired_source)
    yield from move(transfer, orientation, 1, 3, "transfer", carrying=True)
    truth = env.truth()
    desired_source = truth["positions"][rows, destination].copy()
    desired_source[:, 2] += env.protocol.object_height + 0.001
    place = compensate_placement(truth["tcp"], truth["positions"][rows, source], desired_source)
    yield from move(place, orientation, 1, 2, "lower_to_stack", carrying=True, support=destination)

    for _ in range(3):
        if not np.any(active):
            break
        truth = env.truth(contacts=True)
        touching = truth["object_contacts"][rows, source, destination]
        if np.all(touching | ~active):
            break
        correction = truth["tcp"].copy()
        correction[:, :2] += truth["positions"][rows, destination, :2] - truth["positions"][rows, source, :2]
        correction[~touching, 2] -= 0.0015
        yield from move(correction, orientation, 1, 0.3, "placement_correction", carrying=True, support=destination)
    yield from gate(
        lambda t: t["object_contacts"][rows, source, destination], 0.5, "support_contact", "no_support_contact"
    )
    yield from gripper(0, "release")
    yield from gate(lambda t: ~t["robot_contacts"][rows, source], 0.5, "release_check", "release_failed")
    retreat = env.truth()["tcp"].copy()
    retreat[:, 2] = travel_height
    yield from move(retreat, orientation, 0, 2, "retreat")
    yield from move(np.tile(HOME_TCP, (env.num_envs, 1)), DOWN_QUAT, 0, 3, "return_home")
    for _ in range(round(config.stability_seconds * env.protocol.control_hz)):
        if not np.any(active):
            break
        yield from command(env.last_action.copy(), "stability_hold")
