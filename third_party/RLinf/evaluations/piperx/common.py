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

"""Record selected rollout video and restore the dataset camera calibration."""

import json
import subprocess
from pathlib import Path

import numpy as np
from settings import (
    CALIBRATION,
    CAMERAS,
    VIDEO_CAMERAS,
    VIDEO_FAILURE_ONLY,
    VIDEO_STRIDE,
)


def count_video_frames(path: Path) -> tuple[int, float]:
    """Decode a video with bounded runtime and no interactive input."""
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
        "-progress",
        "pipe:1",
        "-nostats",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Video validation failed for {path}: {error.stderr.strip()}"
        ) from error
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if values.get("progress") != "end":
        raise RuntimeError(f"Video validation did not finish: {path}")
    return int(values["frame"]), int(values["out_time_us"]) / 1_000_000


def restore_cameras(env):
    calibration = json.loads(CALIBRATION.read_text())
    for name in CAMERAS:
        saved, camera = calibration[name], env.cameras[name]
        assert (
            list(camera.res) == saved["resolution_wh"]
            and camera.fov == saved["fov_degrees"]
        )
        np.testing.assert_allclose(
            camera.intrinsics, saved["intrinsics"], atol=1e-6, rtol=0
        )
        if saved["attached_link"] is None:
            transform = np.asarray(saved["initial_camera_to_world_opengl"]).reshape(
                -1, 4, 4
            )[0]
            camera.set_pose(transform=transform)
            np.testing.assert_allclose(
                np.asarray(camera.transform).reshape(-1, 4, 4)[0],
                transform,
                atol=1e-6,
                rtol=0,
            )
        else:
            np.testing.assert_allclose(
                env.camera_mounts[name]["camera_to_link_opengl"],
                saved["camera_to_link_opengl"],
                atol=1e-8,
                rtol=0,
            )
    return {
        name: value for name, value in env.camera_metadata().items() if name in CAMERAS
    }


class RecordedEnv:
    def __init__(self, env, directory):
        self.env, self.directory, self.frames = env, directory, 0
        self.control_steps = 0
        self.last_captured_step = -1
        self.latest_observation = None
        self.latest_observation_step = -1
        self.writers = {}
        self.frame_buffers = (
            {name: [] for name in VIDEO_CAMERAS} if VIDEO_FAILURE_ONLY else None
        )
        if self.frame_buffers is None:
            self.writers = {name: self._open_writer(name) for name in VIDEO_CAMERAS}
        self.capture()

    def _open_writer(self, name):
        import imageio.v2 as imageio

        return imageio.get_writer(
            self.directory / f"{name}.mp4",
            fps=self.env.protocol.control_hz / VIDEO_STRIDE,
            codec="libx264",
            pixelformat="yuv420p",
            quality=7,
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart", "-threads", "2"],
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observe(self):
        observation = self.env.observe(camera_names=CAMERAS)
        self.latest_observation = observation
        self.latest_observation_step = self.control_steps
        return observation

    def current_observation(self):
        if self.latest_observation_step != self.control_steps:
            return self.observe()
        return self.latest_observation

    def capture(self):
        observation = self.current_observation()
        if self.frame_buffers is None:
            for name, writer in self.writers.items():
                writer.append_data(observation[name][0])
        else:
            for name, frames in self.frame_buffers.items():
                frames.append(observation[name][0].copy())
        self.frames += 1
        self.last_captured_step = self.control_steps

    def step(self, *args, **kwargs):
        result = self.env.step(*args, **kwargs)
        self.control_steps += 1
        if self.control_steps % VIDEO_STRIDE == 0:
            self.capture()
        return result

    def finish(self, keep_video=True):
        if keep_video and self.last_captured_step != self.control_steps:
            self.capture()
        if self.frame_buffers is not None and keep_video:
            self.writers = {name: self._open_writer(name) for name in VIDEO_CAMERAS}
            for name, writer in self.writers.items():
                for frame in self.frame_buffers[name]:
                    writer.append_data(frame)
        for writer in self.writers.values():
            writer.close()
        if not keep_video:
            if self.frame_buffers is None:
                for name in VIDEO_CAMERAS:
                    (self.directory / f"{name}.mp4").unlink()
            else:
                self.frame_buffers.clear()
            return {}
        if self.frame_buffers is not None:
            self.frame_buffers.clear()
        videos = {}
        for name in VIDEO_CAMERAS:
            frames, seconds = count_video_frames(self.directory / f"{name}.mp4")
            assert frames == self.frames
            videos[name] = {
                "file": f"{name}.mp4",
                "frames": frames,
                "seconds": seconds,
                "fps": self.env.protocol.control_hz / VIDEO_STRIDE,
                "control_step_stride": VIDEO_STRIDE,
            }
        return videos
