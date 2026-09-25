"""Online failure watchdog for policy-to-IK corrective demonstrations."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FailureWatchdogConfig:
    """Thresholds for detecting recoverable failures during a rollout."""

    window_seconds: float = 3.0
    reach_distance_m: float = 0.04
    closed_threshold: float = 0.65
    lifted_height_m: float = 0.03
    dropped_height_m: float = 0.025
    progress_distance_m: float = 0.005

    def __post_init__(self):
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0 < self.dropped_height_m < self.lifted_height_m:
            raise ValueError("Require 0 < dropped_height_m < lifted_height_m")
        if not 0 <= self.closed_threshold <= 1:
            raise ValueError("closed_threshold must be in [0, 1]")
        if min(self.reach_distance_m, self.progress_distance_m) <= 0:
            raise ValueError("Distance thresholds must be positive")


class OnlineFailureWatchdog:
    """Latch the first observable failure for one source/destination task.

    The three-second window starts from the event that makes each diagnosis
    meaningful. Reaching arms the close deadline; reaching and closing arm the
    lift deadline. Before reaching, the same window measures lack of approach
    progress instead of elapsed episode time.
    """

    def __init__(self, pair, initial, config=None):
        self.pair = tuple(int(value) for value in pair)
        if len(self.pair) != 2 or self.pair[0] == self.pair[1]:
            raise ValueError("pair must contain distinct source and destination indices")
        self.config = FailureWatchdogConfig() if config is None else config
        self.initial_source_z = float(initial["positions"][0, self.pair[0], 2])
        initial_distance = np.linalg.norm(initial["positions"][0, self.pair[0]] - initial["tcp"][0])
        self.elapsed = 0.0
        self.reached_at = None
        self.closed_at = None
        self.grasp_attempt_at = None
        self.lifted_at = None
        self.max_lift_m = 0.0
        self.best_distance_m = float(initial_distance)
        self.progress_reference_m = float(initial_distance)
        self.last_progress_at = 0.0
        self.trigger = None

    @property
    def reached(self):
        return self.reached_at is not None

    @property
    def ever_closed(self):
        return self.closed_at is not None

    def _latch(self, reason, *, distance_m, closure, current_lift_m):
        if self.trigger is None:
            self.trigger = {
                "reason": reason,
                "simulation_seconds": float(self.elapsed),
                "tcp_source_distance_m": float(distance_m),
                "gripper_closure": float(closure),
                "current_source_lift_m": float(current_lift_m),
                "max_source_lift_m": float(self.max_lift_m),
                "reached_at_seconds": self.reached_at,
                "closed_at_seconds": self.closed_at,
                "lifted_at_seconds": self.lifted_at,
                "watchdog": asdict(self.config),
            }
        return self.trigger

    def update(self, truth, gripper_closure, dt):
        """Consume one measured state and return the first latched event, if any."""
        if self.trigger is not None:
            return self.trigger
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.elapsed += float(dt)
        source = self.pair[0]
        source_position = np.asarray(truth["positions"])[0, source]
        tcp = np.asarray(truth["tcp"])[0]
        closure = float(np.asarray(gripper_closure).reshape(-1)[0])
        distance = float(np.linalg.norm(source_position - tcp))
        current_lift = float(source_position[2] - self.initial_source_z)
        self.max_lift_m = max(self.max_lift_m, current_lift)

        self.best_distance_m = min(self.best_distance_m, distance)
        if self.best_distance_m <= self.progress_reference_m - self.config.progress_distance_m:
            self.progress_reference_m = self.best_distance_m
            self.last_progress_at = self.elapsed

        if self.reached_at is None and distance < self.config.reach_distance_m:
            self.reached_at = float(self.elapsed)
        if self.closed_at is None and closure > self.config.closed_threshold:
            self.closed_at = float(self.elapsed)
        if self.grasp_attempt_at is None and self.reached_at is not None and self.closed_at is not None:
            self.grasp_attempt_at = float(max(self.reached_at, self.closed_at))
        if self.lifted_at is None and self.max_lift_m >= self.config.lifted_height_m:
            self.lifted_at = float(self.elapsed)

        # Once a real lift has happened, crossing back below 2.5 cm is directly
        # observable. Waiting another three seconds would only add bad actions.
        if self.lifted_at is not None and current_lift < self.config.dropped_height_m:
            return self._latch(
                "lifted_then_dropped",
                distance_m=distance,
                closure=closure,
                current_lift_m=current_lift,
            )

        if self.reached_at is None:
            if self.elapsed - self.last_progress_at >= self.config.window_seconds:
                return self._latch(
                    "never_reached_progress_timeout",
                    distance_m=distance,
                    closure=closure,
                    current_lift_m=current_lift,
                )
            return None

        if self.closed_at is None:
            if self.elapsed - self.reached_at >= self.config.window_seconds:
                return self._latch(
                    "never_closed_gripper",
                    distance_m=distance,
                    closure=closure,
                    current_lift_m=current_lift,
                )
            return None

        if self.lifted_at is None and self.elapsed - self.grasp_attempt_at >= self.config.window_seconds:
            return self._latch(
                "missed_grasp",
                distance_m=distance,
                closure=closure,
                current_lift_m=current_lift,
            )
        return None
