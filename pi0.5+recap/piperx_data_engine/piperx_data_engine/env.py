"""Physical, batched single-arm task. Ground-truth object poses stay out of policy observations."""

import os
from pathlib import Path

import numpy as np

from .protocol import DEFAULT_PROTOCOL, OBJECTS, sample_scene

BUNDLED_URDF = Path(__file__).resolve().parent / "assets/piper_x_description/urdf/piper_x_description.urdf"
DEFAULT_URDF = Path(os.environ.get("PIPERX_URDF", BUNDLED_URDF)).expanduser()
TCP_LOCAL = (0.0, 0.0, 0.125)
DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)  # wxyz; local +Z points towards the table
HOME_TCP = (0.25, 0.0, 0.20)
# Camera Z coordinates below are heights relative to the tabletop.
# Fixed third-person view from the front of the table, tilted down about 54 degrees.
# Lower the camera and tilt it up slightly to expose the gripper below the wrist.
THIRD_PERSON_CAMERA = {
    "res": (640, 480),
    "pos": (1.5, 0.0, 0.4),
    "lookat": (0.30, 0.0, 0.2),
    "up": (0.0, 0.0, 1.0),
    "fov": 26,
    "near": 0.01,
    "far": 4,
}


def array(value):
    return value.detach().cpu().numpy()


class StackingEnv:
    def __init__(
        self,
        gs,
        num_envs=1,
        cameras=False,
        show_viewer=False,
        urdf=DEFAULT_URDF,
        protocol=DEFAULT_PROTOCOL,
    ):
        self.gs, self.num_envs, self.protocol = gs, num_envs, protocol
        self.urdf = Path(urdf).resolve()
        self._planning_env = None
        table_camera = dict(THIRD_PERSON_CAMERA)
        for key in ("pos", "lookat"):
            x, y, height = table_camera[key]
            table_camera[key] = (x, y, protocol.table_top + height)
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not Path(urdf).is_file():
            raise FileNotFoundError(urdf)
        steps = 1 / protocol.control_hz / protocol.physics_dt
        if abs(steps - round(steps)) > 1e-6:
            raise ValueError("The control period must be an integer number of physics steps")
        self.substeps = round(steps)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=protocol.physics_dt, gravity=(0, 0, -9.81)),
            rigid_options=gs.options.RigidOptions(enable_joint_limit=True, enable_self_collision=True),
            viewer_options=gs.options.ViewerOptions(
                res=table_camera["res"],
                camera_pos=table_camera["pos"],
                camera_lookat=table_camera["lookat"],
                camera_up=table_camera["up"],
                camera_fov=table_camera["fov"],
            ),
            vis_options=gs.options.VisOptions(split_envs=True, rendered_envs_idx=list(range(num_envs))),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )
        self.table = self.scene.add_entity(
            gs.morphs.Box(pos=(0.25, 0, protocol.table_top - 0.025), size=(0.90, 0.80, 0.05), fixed=True),
            material=gs.materials.Rigid(friction=0.8),
            surface=gs.surfaces.Rough(color=(0.65, 0.60, 0.51)),
            name="table",
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=str(Path(urdf).resolve()), fixed=True, pos=(0, 0, protocol.table_top + 0.002)),
            material=gs.materials.Rigid(friction=1.0),
            name="piperx",
        )
        self.objects = []
        for i, name in enumerate(OBJECTS):
            kwargs = {
                "pos": (
                    0.24 + 0.10 * i,
                    0,
                    protocol.table_top + protocol.object_height / 2 + 0.001,
                )
            }
            if "cylinder" in name:
                morph = gs.morphs.Cylinder(radius=protocol.object_width / 2, height=protocol.object_height, **kwargs)
            else:
                morph = gs.morphs.Box(
                    size=(protocol.object_width, protocol.object_width, protocol.object_height), **kwargs
                )
            self.objects.append(
                self.scene.add_entity(
                    morph,
                    material=gs.materials.Rigid(rho=600, friction=0.8),
                    surface=gs.surfaces.Rough(
                        color=(0.85, 0.04, 0.04) if name.startswith("red") else (0.03, 0.12, 0.85)
                    ),
                    name=name.replace(" ", "_"),
                )
            )
        self.cameras = {}
        self.camera_mounts = {}
        self.ee = self.robot.get_link("Link6")
        if cameras:
            wrist = self.scene.add_camera(res=(224, 224), fov=65, near=0.005, far=3, GUI=False)
            from genesis.utils.geom import pos_lookat_up_to_T

            offset = pos_lookat_up_to_T(
                np.array((0.029, -0.069, 0.022)), np.array((0.0, 0.0, 0.20)), np.array((0.0, -1.0, 0.0))
            )
            wrist.attach(self.ee, offset)
            self.cameras["wrist"] = wrist
            self.camera_mounts["wrist"] = {"attached_link": "Link6", "camera_to_link_opengl": offset.tolist()}
            self.cameras["third_person"] = self.scene.add_camera(**table_camera, GUI=False)
        self.scene.build(n_envs=num_envs, env_spacing=(1.2, 1.0))
        self.arm_dofs = [self.robot.get_joint(f"joint{i}").dofs_idx_local[0] for i in range(1, 7)]
        self.arm_qs = [self.robot.get_joint(f"joint{i}").qs_idx_local[0] for i in range(1, 7)]
        self.finger_dofs = [self.robot.get_joint(f"joint{i}").dofs_idx_local[0] for i in (7, 8)]
        self.finger_qs = [self.robot.get_joint(f"joint{i}").qs_idx_local[0] for i in (7, 8)]
        self.robot.set_dofs_kp([1500, 1500, 1200, 600, 500, 300], self.arm_dofs)
        self.robot.set_dofs_kv([75, 75, 60, 30, 25, 15], self.arm_dofs)
        self.robot.set_dofs_force_range([-100] * 6, [100] * 6, self.arm_dofs)
        self.robot.set_dofs_kp([1000, 1000], self.finger_dofs)
        self.robot.set_dofs_kv([20, 20], self.finger_dofs)
        self.robot.set_dofs_force_range([-20, -20], [20, 20], self.finger_dofs)
        home_seed = array(self.robot.get_qpos()).copy()
        home_seed[:, self.arm_qs] = [0, 1.358, -1.2181, 1.52381, 0, -1.5708]
        self.home_qpos, err = self.ik(np.tile(HOME_TCP, (num_envs, 1)), init_qpos=home_seed)
        if np.any(np.linalg.norm(err[:, :3], axis=1) > 0.002) or np.any(np.linalg.norm(err[:, 3:], axis=1) > 0.02):
            self.scene.destroy()
            raise RuntimeError(f"Home IK failed: {err.tolist()}")
        self.home_qpos[:, self.finger_qs] = [0.05, -0.05]
        self.last_action = np.concatenate([self.home_qpos[:, self.arm_qs], np.zeros((num_envs, 1))], axis=1)

    def ik(self, positions, quaternions=None, init_qpos=None):
        if quaternions is None:
            quaternions = np.tile(DOWN_QUAT, (self.num_envs, 1))
        qpos, error = self.robot.inverse_kinematics(
            link=self.ee,
            pos=positions,
            quat=quaternions,
            local_point=TCP_LOCAL,
            init_qpos=init_qpos,
            dofs_idx_local=self.arm_dofs,
            respect_joint_limit=True,
            max_samples=25,
            max_solver_iters=50,
            pos_tol=0.001,
            rot_tol=0.01,
            seed=0,
            return_error=True,
        )
        return array(qpos), array(error)

    def reset(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError("One scene seed is required per environment")
        samples = [sample_scene(int(seed), self.protocol) for seed in seeds]
        positions, quaternions = np.stack([s[0] for s in samples]), np.stack([s[1] for s in samples])
        self.robot.set_qpos(self.home_qpos, zero_velocity=True)
        for i, obj in enumerate(self.objects):
            obj.set_pos(positions[:, i], relative=False, zero_velocity=True)
            obj.set_quat(quaternions[:, i], relative=False, zero_velocity=True)
        self.last_action = np.concatenate([self.home_qpos[:, self.arm_qs], np.zeros((self.num_envs, 1))], axis=1)
        for _ in range(self.protocol.control_hz):
            self.step(self.last_action)
        self.initial = self.truth()
        return self.initial

    def state(self):
        joints = array(self.robot.get_dofs_position(self.arm_dofs))
        fingers = array(self.robot.get_dofs_position(self.finger_dofs))
        closure = 1 - np.clip((fingers[:, 0] - fingers[:, 1]) / 0.10, 0, 1)
        return np.concatenate([joints, closure[:, None]], axis=1).astype(np.float32)

    def observe(self, camera_names=None):
        """Read measured state and selected RGB views; default preserves all cameras."""
        observation = {"state": self.state()}
        names = tuple(self.cameras) if camera_names is None else tuple(camera_names)
        unknown = set(names) - self.cameras.keys()
        if unknown:
            raise ValueError(f"Unknown observation cameras: {sorted(unknown)}")
        for name in names:
            camera = self.cameras[name]
            if name in self.camera_mounts:
                camera.move_to_attach()
            rgb = np.asarray(camera.render(rgb=True)[0])
            if rgb.ndim == 3:
                rgb = rgb[None]
            width, height = camera.res
            if rgb.shape != (self.num_envs, height, width, 3):
                raise RuntimeError(f"Unexpected {name} image batch shape: {rgb.shape}")
            observation[name] = rgb.astype(np.uint8)
        return observation

    def camera_metadata(self):
        """Intrinsics, current camera poses and rigid mounts for attached cameras."""
        result = {}
        for name, camera in self.cameras.items():
            if name in self.camera_mounts:
                camera.move_to_attach()
            result[name] = {
                "resolution_wh": list(camera.res),
                "fov_degrees": camera.fov,
                "intrinsics": camera.intrinsics.tolist(),
                "initial_camera_to_world_opengl": camera.transform.tolist(),
                **self.camera_mounts.get(name, {"attached_link": None}),
            }
        return result

    def step(self, action, on_physics_step=None):
        """Absolute q1..q6 radians + closure (0=open, 1=closed), at 20 Hz.

        Targets interpolate from the previous command at physics frequency. This
        controller is identical for collection, playback, and policy evaluation.
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_envs, 7) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite action shape {(self.num_envs, 7)}")
        lower, upper = self.robot.get_dofs_limit(self.arm_dofs)
        action = action.copy()
        action[:, :6] = np.clip(action[:, :6], array(lower), array(upper))
        action[:, 6] = np.clip(action[:, 6], 0, 1)
        for i in range(self.substeps):
            target = self.last_action + ((i + 1) / self.substeps) * (action - self.last_action)
            self.robot.control_dofs_position(target[:, :6], self.arm_dofs)
            opening = 0.05 * (1 - target[:, 6])
            self.robot.control_dofs_position(np.stack([opening, -opening], axis=-1), self.finger_dofs)
            self.scene.step()
            if on_physics_step is not None:
                on_physics_step()
        self.last_action = action.copy()
        return action

    def truth(self, contacts=False):
        """Privileged channels for the scripted teacher / evaluator, never policy input."""
        from genesis.utils.geom import transform_by_trans_quat

        pos = np.stack([array(o.get_pos(relative=False)) for o in self.objects], axis=1)
        quat = np.stack([array(o.get_quat(relative=False)) for o in self.objects], axis=1)
        tcp = array(
            transform_by_trans_quat(
                self.gs.tensor(TCP_LOCAL), self.ee.get_pos(relative=False), self.ee.get_quat(relative=False)
            )
        )
        truth = {
            "positions": pos,
            "quaternions": quat,
            "tcp": tcp,
            "tcp_quaternion": array(self.ee.get_quat(relative=False)),
            "linear_velocity": np.stack([array(o.get_vel()) for o in self.objects], axis=1),
            "angular_velocity": np.stack([array(o.get_ang()) for o in self.objects], axis=1),
        }
        if contacts:
            truth["robot_contacts"] = np.zeros((self.num_envs, 3), dtype=bool)
            truth["finger_contacts"] = np.zeros((self.num_envs, 3, 2), dtype=bool)
            truth["object_contacts"] = np.zeros((self.num_envs, 3, 3), dtype=bool)
            for i, obj in enumerate(self.objects):
                info = obj.get_contacts(self.robot)
                truth["robot_contacts"][:, i] = array(info["valid_mask"].any(dim=-1))
                for finger, name in enumerate(("Link7", "Link8")):
                    link = self.robot.get_link(name).idx
                    mask = info["valid_mask"] & ((info["link_a"] == link) | (info["link_b"] == link))
                    truth["finger_contacts"][:, i, finger] = array(mask.any(dim=-1))
                for j in range(i):
                    info = obj.get_contacts(self.objects[j])
                    value = array(info["valid_mask"].any(dim=-1))
                    truth["object_contacts"][:, i, j] = value
                    truth["object_contacts"][:, j, i] = value
        return truth

    def close(self):
        if self._planning_env is not None:
            self._planning_env.close()
        self.scene.destroy()
