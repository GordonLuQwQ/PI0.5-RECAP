"""Stateful success evaluator shared by teacher collection and learned-policy runs."""

import numpy as np

from .protocol import DEFAULT_PROTOCOL


def upright_angle(quaternions):
    # Cosine between the object's local +Z and world +Z (wxyz).
    return np.arccos(np.clip(1 - 2 * (quaternions[..., 1] ** 2 + quaternions[..., 2] ** 2), -1, 1))


def rotation_change(quaternions, initial):
    dot = np.abs(np.sum(quaternions * initial, axis=-1))
    return 2 * np.arccos(np.clip(dot, 0, 1))


class SuccessMonitor:
    def __init__(self, pairs, initial, protocol=DEFAULT_PROTOCOL):
        self.pairs = np.asarray(pairs, dtype=int)
        self.initial = initial
        self.protocol = protocol
        self.n = len(pairs)
        self.index = np.arange(self.n)
        self.source, self.destination = self.pairs.T
        self.third = 3 - self.source - self.destination
        self.stable_time = np.zeros(self.n)
        self.longest_stable_time = np.zeros(self.n)
        self.success = np.zeros(self.n, dtype=bool)
        self.max_displacement = np.zeros((self.n, 3))
        self.max_rotation = np.zeros((self.n, 3))
        self.max_lift = np.zeros((self.n, 3))
        self.ever_tipped = np.zeros((self.n, 3), dtype=bool)
        self.reached = np.zeros(self.n, dtype=bool)
        self.ever_closed = np.zeros(self.n, dtype=bool)
        self.last = initial

    def update(self, truth, gripper_closure, dt):
        p, i = self.protocol, self.index
        pos, quat = truth["positions"], truth["quaternions"]
        self.last = truth
        self.max_displacement = np.maximum(
            self.max_displacement, np.linalg.norm(pos - self.initial["positions"], axis=-1)
        )
        self.max_rotation = np.maximum(self.max_rotation, rotation_change(quat, self.initial["quaternions"]))
        self.max_lift = np.maximum(self.max_lift, pos[..., 2] - self.initial["positions"][..., 2])
        tilt = upright_angle(quat)
        self.ever_tipped |= tilt > np.deg2rad(p.upright_degrees)
        source, destination = pos[i, self.source], pos[i, self.destination]
        tcp_distance = np.linalg.norm(pos - truth["tcp"][:, None, :], axis=-1)
        self.reached |= tcp_distance[i, self.source] < 0.04
        self.ever_closed |= np.asarray(gripper_closure) > 0.65
        placed = (
            (np.linalg.norm((source - destination)[:, :2], axis=-1) <= p.stack_xy_tolerance)
            & (np.abs(source[:, 2] - destination[:, 2] - p.object_height) <= p.stack_z_tolerance)
            & (tilt[i, self.source] <= np.deg2rad(p.upright_degrees))
            & (tilt[i, self.destination] <= np.deg2rad(p.upright_degrees))
            & (self.max_displacement[i, self.destination] <= p.destination_displacement)
            & (self.max_displacement[i, self.third] <= p.third_displacement)
            & (self.max_rotation[i, self.third] <= np.deg2rad(p.third_rotation_degrees))
            & np.all(np.linalg.norm(truth["linear_velocity"], axis=-1) <= p.max_linear_speed, axis=-1)
            & np.all(np.linalg.norm(truth["angular_velocity"], axis=-1) <= p.max_angular_speed, axis=-1)
            & (tcp_distance[i, self.source] >= p.retreat_distance)
            & (tcp_distance[i, self.destination] >= p.retreat_distance)
            & ~np.any(truth["robot_contacts"], axis=-1)
            & truth["object_contacts"][i, self.source, self.destination]
        )
        self.stable_time = np.where(placed, self.stable_time + dt, 0)
        self.longest_stable_time = np.maximum(self.longest_stable_time, self.stable_time)
        # A latched flag is convenient for stopping an episode at the first valid success.
        self.success |= self.stable_time + 1e-9 >= p.hold_seconds
        self.success &= self.max_displacement[i, self.third] <= p.third_displacement
        self.success &= self.max_rotation[i, self.third] <= np.deg2rad(p.third_rotation_degrees)
        return self.success.copy()

    def results(self):
        records = []
        p = self.protocol
        for i, (src, dst) in enumerate(self.pairs):
            third = self.third[i]
            failures = []
            if not self.success[i]:
                if self.max_displacement[i, third] > p.third_displacement or self.max_rotation[i, third] > np.deg2rad(
                    p.third_rotation_degrees
                ):
                    failures.append("third_object_disturbed")
                if self.ever_tipped[i, dst]:
                    failures.append("destination_tipped")
                if max(self.max_lift[i, dst], self.max_lift[i, third]) > 0.03:
                    failures.append("wrong_object_lifted")
                delta = self.last["positions"][i, src] - self.last["positions"][i, third]
                if (
                    np.linalg.norm(delta[:2]) < p.stack_xy_tolerance
                    and abs(delta[2] - p.object_height) < p.stack_z_tolerance
                ):
                    failures.append("wrong_destination")
                if not self.reached[i]:
                    failures.append("never_reached")
                elif self.max_lift[i, src] < 0.03:
                    failures.append("missed_grasp" if self.ever_closed[i] else "never_closed_gripper")
                elif self.last["positions"][i, src, 2] < self.initial["positions"][i, src, 2] + 0.025:
                    failures.append("lifted_then_dropped")
                if not failures:
                    failures.append("placement_or_stability_timeout")
            records.append(
                {
                    "success": bool(self.success[i]),
                    "failure": failures[0] if failures else None,
                    "failure_flags": failures,
                    "stable_seconds_at_end": float(self.stable_time[i]),
                    "longest_stable_seconds": float(self.longest_stable_time[i]),
                    "max_object_displacement_m": self.max_displacement[i].tolist(),
                    "max_object_lift_m": self.max_lift[i].tolist(),
                }
            )
        return records
