"""Genesis Cartesian trajectory planning and dry-run collision checks.

Inspired by ManiSkill's separation of grasp candidates, dry-run planning and PD
execution. A separate Genesis scene holds hypothetical poses: dry runs never
change the physical scene that produces the demonstrations. No SAPIEN dependency.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .env import StackingEnv, array


@dataclass(frozen=True)
class PlannerConfig:
    joint_speed: float = 1.25  # rad/s
    joint_acceleration: float = 5.0  # rad/s^2
    collision_resolution: float = 0.04  # rad between tested configurations
    ik_position_tolerance: float = 0.002
    ik_rotation_tolerance: float = 0.02


DEFAULT_PLANNER_CONFIG = PlannerConfig()


def downward_quaternion(yaw):
    yaw = np.asarray(yaw)
    return np.stack([np.zeros_like(yaw), np.cos(yaw / 2), np.sin(yaw / 2), np.zeros_like(yaw)], axis=-1)


def grasp_yaws(object_quaternions, source_indices):
    """Two orthogonal box face grasps; a cylinder has no distinguished yaw."""
    yaw = 2 * np.arctan2(object_quaternions[:, 3], object_quaternions[:, 0])
    yaw = np.where(np.asarray(source_indices) == 1, 0, yaw)
    candidates = np.stack([yaw, yaw + np.pi / 2], axis=-1)
    candidates = (candidates + np.pi / 2) % np.pi - np.pi / 2
    # Home closes along world X. Prefer the smaller rotation when both are valid.
    return np.take_along_axis(candidates, np.argsort(np.abs(candidates), axis=-1), axis=-1)


def compensate_placement(tcp, source_position, desired_source_position):
    """Preserve the measured TCP-to-object offset instead of assuming a perfect grasp."""
    return np.asarray(tcp) + np.asarray(desired_source_position) - np.asarray(source_position)


def _rot(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[..., [1, 2, 3, 0]])


def retime_joint_path(path, arm_qs, control_hz, config=DEFAULT_PLANNER_CONFIG):
    """Bound finite differences of commanded joints, including stationary endpoints.

    Linear resampling preserves the checked joint segments. Recheck acceleration
    after resampling: at polyline corners it does not scale with duration squared.
    These are command limits, not a guarantee about measured physical acceleration.
    """
    path = np.asarray(path)
    dt = 1 / control_hz
    duration_scale, result = 1.0, path
    for _ in range(30):
        joints = result[:, :, arm_qs]
        extended = np.concatenate([joints[:1], joints, joints[-1:]], axis=0)
        velocity = np.max(np.abs(np.diff(extended, axis=0))) / dt
        acceleration = np.max(np.abs(np.diff(extended, n=2, axis=0))) / dt**2
        if velocity <= config.joint_speed * (1 + 1e-6) and acceleration <= config.joint_acceleration * (1 + 1e-6):
            return result
        duration_scale *= 1.01 * max(
            1.0, velocity / config.joint_speed, np.sqrt(acceleration / config.joint_acceleration)
        )
        length = int(np.ceil((len(path) - 1) * duration_scale)) + 1
        indices = np.linspace(0, len(path) - 1, length)
        lower = np.floor(indices).astype(int)
        upper = np.minimum(lower + 1, len(path) - 1)
        fraction = (indices - lower)[:, None, None]
        result = path[lower] * (1 - fraction) + path[upper] * fraction
    raise RuntimeError("Could not retime the joint path within command limits")


class CartesianPlanner:
    def __init__(self, env, config=DEFAULT_PLANNER_CONFIG):
        self.env, self.config = env, config
        if env._planning_env is None:
            env._planning_env = StackingEnv(env.gs, num_envs=env.num_envs, urdf=env.urdf, protocol=env.protocol)
        self.mirror = env._planning_env

    def path(
        self, target, orientation, seconds, active=None, *, start_qpos=None, start_tcp=None, start_orientation=None
    ):
        """Generate the entire joint path before any execution; retain the IK branch."""
        env, cfg = self.env, self.config
        active = np.ones(env.num_envs, dtype=bool) if active is None else np.asarray(active)
        truth = env.truth()
        start = truth["tcp"] if start_tcp is None else np.asarray(start_tcp)
        initial_rotation = truth["tcp_quaternion"] if start_orientation is None else np.asarray(start_orientation)
        target = np.broadcast_to(target, start.shape).copy()
        orientation = np.broadcast_to(orientation, initial_rotation.shape).copy()
        target[~active], orientation[~active] = start[~active], initial_rotation[~active]
        orientation[np.sum(initial_rotation * orientation, axis=-1) < 0] *= -1
        previous = array(env.robot.get_qpos()).copy() if start_qpos is None else np.asarray(start_qpos).copy()
        path = [previous.copy()]
        valid = np.ones(env.num_envs, dtype=bool)
        steps = max(2, round(seconds * env.protocol.control_hz))
        for tick in range(1, steps + 1):
            t = tick / steps
            blend = t**3 * (10 - 15 * t + 6 * t * t)
            pos = start + blend * (target - start)
            quat = initial_rotation + blend * (orientation - initial_rotation)
            quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
            qpos, error = env.ik(pos, quat, init_qpos=previous)
            valid &= np.isfinite(qpos).all(axis=-1) & np.isfinite(error).all(axis=-1)
            valid &= np.linalg.norm(error[:, :3], axis=-1) <= cfg.ik_position_tolerance
            valid &= np.linalg.norm(error[:, 3:], axis=-1) <= cfg.ik_rotation_tolerance
            valid &= np.max(np.abs(qpos[:, env.arm_qs] - previous[:, env.arm_qs]), axis=-1) <= 0.25
            qpos[~valid | ~active] = previous[~valid | ~active]
            path.append(qpos.copy())
            previous = qpos
        path = retime_joint_path(path, env.arm_qs, env.protocol.control_hz, cfg)
        return path, valid | ~active

    def collision_free(self, path, source, *, carrying=False, support=None, allow_table_support=False):
        """Check the full robot plus a hypothetical carried payload at <=0.04 rad spacing.

        Intentional finger/source contacts are allowed. A carried object may touch
        the table during initial lift, or its destination during final placement.
        Every other robot/object contact is rejected. The checker uses the current
        Genesis collision geometry, including its URDF neutral-pair filtering.
        """
        env, mirror = self.env, self.mirror
        truth = env.truth()
        rows = np.arange(env.num_envs)
        source = np.asarray(source, dtype=int)
        for j, obj in enumerate(mirror.objects):
            obj.set_pos(truth["positions"][:, j], relative=False)
            obj.set_quat(truth["quaternions"][:, j], relative=False)
        mirror.robot.set_qpos(array(env.robot.get_qpos()))
        link_position = array(env.ee.get_pos(relative=False))
        link_rotation = _rot(truth["tcp_quaternion"])
        relative_pos = link_rotation.inv().apply(truth["positions"][rows, source] - link_position)
        relative_rot = link_rotation.inv() * _rot(truth["quaternions"][rows, source])
        valid = np.ones(env.num_envs, dtype=bool)
        self.last_collisions = [[] for _ in range(env.num_envs)]
        fingers = [mirror.robot.get_link(n).idx for n in ("Link7", "Link8")]
        object_links = np.array([o.base_link.idx for o in mirror.objects])
        src_links = object_links[source]
        table_link = mirror.table.base_link.idx
        base_link = mirror.robot.base_link.idx
        previous = path[0]
        for goal in path:
            samples = max(
                1,
                int(
                    np.ceil(
                        np.max(np.abs(goal[:, env.arm_qs] - previous[:, env.arm_qs])) / self.config.collision_resolution
                    )
                ),
            )
            for fraction in np.linspace(0, 1, samples + 1)[1:]:
                qpos = previous + fraction * (goal - previous)
                mirror.robot.set_qpos(qpos, zero_velocity=False)
                if carrying:
                    ee_pos = array(mirror.ee.get_pos(relative=False))
                    ee_rot = _rot(array(mirror.ee.get_quat(relative=False)))
                    carried_pos = ee_pos + ee_rot.apply(relative_pos)
                    carried_quat = (ee_rot * relative_rot).as_quat()[:, [3, 0, 1, 2]]
                    for j, obj in enumerate(mirror.objects):
                        ids = np.flatnonzero(source == j)
                        if len(ids):
                            obj.set_pos(carried_pos[ids], envs_idx=ids, relative=False)
                            obj.set_quat(carried_quat[ids], envs_idx=ids, relative=False)
                # Same collision kernel used by Genesis's native RRT planner, only on the mirror scene.
                mirror.scene.rigid_solver._kernel_detect_collision()
                contacts = mirror.robot.get_contacts()
                a, b, mask = (array(contacts[k]) for k in ("link_a", "link_b", "valid_mask"))
                allowed = (np.isin(a, fingers) & (b == src_links[:, None])) | (
                    np.isin(b, fingers) & (a == src_links[:, None])
                )
                allowed |= ((a == base_link) & (b == table_link)) | ((b == base_link) & (a == table_link))
                forbidden = mask & ~allowed
                for i in np.flatnonzero(np.any(forbidden, axis=-1)):
                    if not self.last_collisions[i]:
                        self.last_collisions[i] = np.stack([a[i][forbidden[i]], b[i][forbidden[i]]], axis=-1).tolist()
                valid &= ~np.any(forbidden, axis=-1)
                if carrying:
                    for j, obj in enumerate(mirror.objects):
                        contacts = obj.get_contacts()
                        a, b, mask = (array(contacts[k]) for k in ("link_a", "link_b", "valid_mask"))
                        other = np.where(a == obj.base_link.idx, b, a)
                        allowed = np.isin(other, fingers)
                        if allow_table_support:
                            allowed |= other == table_link
                        if support is not None:
                            allowed |= other == object_links[np.asarray(support)][:, None]
                        valid &= (source != j) | ~np.any(mask & ~allowed, axis=-1)
            previous = goal
        return valid
