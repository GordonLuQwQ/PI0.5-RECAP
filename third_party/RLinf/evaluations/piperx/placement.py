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

"""Instantaneous released placement, independent of the collection stability rule."""

import numpy as np
from settings import PLACEMENT_RULE


class PlacementMonitor:
    def __init__(self, pairs, initial, protocol):
        self.pairs = np.asarray(pairs, dtype=int)
        self.initial, self.last, self.protocol = initial, initial, protocol
        self.index = np.arange(len(pairs))
        self.source, self.destination = self.pairs.T
        self.success = np.zeros(len(pairs), dtype=bool)
        self.max_displacement = np.zeros((len(pairs), 3))
        self.max_lift = np.zeros((len(pairs), 3))
        self.reached = np.zeros(len(pairs), dtype=bool)
        self.ever_closed = np.zeros(len(pairs), dtype=bool)
        self.first_placement = [None] * len(pairs)
        self.elapsed = 0.0

    def update(self, truth, gripper_closure, dt):
        self.elapsed += dt
        self.last = truth
        i, src, dst = self.index, self.source, self.destination
        pos = truth["positions"]
        closure = np.asarray(gripper_closure)
        self.max_displacement = np.maximum(
            self.max_displacement,
            np.linalg.norm(pos - self.initial["positions"], axis=-1),
        )
        self.max_lift = np.maximum(
            self.max_lift, pos[..., 2] - self.initial["positions"][..., 2]
        )
        self.reached |= np.linalg.norm(pos[i, src] - truth["tcp"], axis=-1) < 0.04
        self.ever_closed |= closure > 0.65
        delta = pos[i, src] - pos[i, dst]
        xy = np.linalg.norm(delta[:, :2], axis=-1)
        z_error = np.abs(delta[:, 2] - self.protocol.object_height)
        placed = (
            (xy <= PLACEMENT_RULE["stack_xy_tolerance_m"])
            & (z_error <= PLACEMENT_RULE["stack_z_tolerance_m"])
            & truth["object_contacts"][i, src, dst]
            & ~truth["robot_contacts"][i, src]
            & (closure <= PLACEMENT_RULE["max_gripper_closure"])
        )
        for row in np.flatnonzero(placed & ~self.success):
            self.first_placement[row] = {
                "simulation_seconds": float(self.elapsed),
                "xy_error_m": float(xy[row]),
                "z_error_m": float(z_error[row]),
                "gripper_closure": float(closure[row]),
                "source_robot_contact": bool(truth["robot_contacts"][row, src[row]]),
                "source_destination_contact": bool(
                    truth["object_contacts"][row, src[row], dst[row]]
                ),
                "positions": pos[row].tolist(),
                "quaternions": truth["quaternions"][row].tolist(),
                "linear_velocity": truth["linear_velocity"][row].tolist(),
                "angular_velocity": truth["angular_velocity"][row].tolist(),
            }
        # A zero hold time must not turn a false placement predicate into success.
        # Once released placement occurs, subsequent motion is outside this metric.
        self.success |= placed
        return self.success.copy()

    def results(self):
        records = []
        for i, (src, dst) in enumerate(self.pairs):
            failure = None
            if not self.success[i]:
                if not self.reached[i]:
                    failure = "never_reached"
                elif self.max_lift[i, src] < 0.03:
                    failure = (
                        "missed_grasp"
                        if self.ever_closed[i]
                        else "never_closed_gripper"
                    )
                elif (
                    self.last["positions"][i, src, 2]
                    < self.initial["positions"][i, src, 2] + 0.025
                ):
                    failure = "lifted_then_dropped"
                else:
                    failure = "not_released_on_destination_within_episode_budget"
            records.append(
                {
                    "success": bool(self.success[i]),
                    "failure": failure,
                    "failure_flags": [failure] if failure else [],
                    "criterion": PLACEMENT_RULE["name"],
                    "first_placement": self.first_placement[i],
                    "max_object_displacement_m": self.max_displacement[i].tolist(),
                    "max_object_lift_m": self.max_lift[i].tolist(),
                    "third_object_disturbed_diagnostic_only": bool(
                        self.max_displacement[i, 3 - src - dst]
                        > self.protocol.third_displacement
                    ),
                }
            )
        return records
