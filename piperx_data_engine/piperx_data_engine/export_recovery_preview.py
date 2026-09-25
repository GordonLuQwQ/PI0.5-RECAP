"""Encode viewable MP4s from raw policy-prefix/IK-correction archives."""

import argparse
import html
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np


def encode_video(frames, path, fps, overwrite=False):
    """Encode one RGB array atomically and verify its decoded frame count."""
    expected_frames = len(frames)
    if path.is_file() and not overwrite:
        decoded_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
        if decoded_frames == expected_frames:
            return "existing"
        raise RuntimeError(f"Existing video has {decoded_frames} frames instead of {expected_frames}: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    temporary.unlink(missing_ok=True)
    writer = imageio.get_writer(
        temporary,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=7,
        macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart", "-threads", "2"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    decoded_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(temporary))
    if decoded_frames != expected_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Encoded video has {decoded_frames} frames instead of {expected_frames}: {temporary}")
    os.replace(temporary, path)
    return "encoded"


def write_index(output, source, records, camera_names, fps, target_reached):
    cards = []
    for episode_index, record in enumerate(records):
        stem = Path(record["archive"]).stem
        videos = "".join(
            f"<figure><figcaption>{html.escape(camera)}</figcaption>"
            f'<video controls preload="metadata" src="{stem}/{html.escape(camera)}.mp4"></video></figure>'
            for camera in camera_names
        )
        cards.append(
            f'<section id="{stem}"><h2>{episode_index:02d} · {html.escape(record["instruction"])}</h2>'
            f"<p>seed={record['scene_seed']} · policy={record['policy_steps']} frames · "
            f"IK={record['ik_steps']} frames · recovery={html.escape(str(record.get('recovery_mode')))}</p>"
            f'<div class="views">{videos}</div></section>'
        )
    status = "complete" if target_reached else "incomplete collection preview"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>IK correction videos</title>
<style>
body{{font:15px system-ui,sans-serif;margin:24px;background:#10151d;color:#edf2f7}}
a{{color:#8ecbff}} section{{padding:18px 0;border-top:1px solid #384453}}
.views{{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}}
figure{{margin:0}} figcaption{{margin-bottom:6px;font-weight:600}}
video{{display:block;max-width:min(100%,640px);max-height:480px;background:#000}}
</style></head><body>
<h1>IK correction videos</h1>
<p>{len(records)} episodes · {fps:g} FPS · {html.escape(status)}</p>
<p>Source: {html.escape(str(source))}</p>
{"".join(cards)}
</body></html>"""
    (output / "watch.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw correction collection")
    parser.add_argument("--output", type=Path, help="Defaults to INPUT/preview_videos")
    parser.add_argument("--cameras", nargs="+", help="Defaults to collection.json cameras")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.input.resolve()
    output = (args.output or source / "preview_videos").resolve()
    collection_path, summary_path = source / "collection.json", source / "summary.json"
    if not collection_path.is_file() or not summary_path.is_file():
        parser.error("Input is not a raw correction collection")
    collection = json.loads(collection_path.read_text())
    summary = json.loads(summary_path.read_text())
    camera_names = tuple(args.cameras or collection["cameras"])
    if not camera_names:
        parser.error("At least one camera is required")
    fps = float(collection["control_hz"])
    records = [json.loads(path.read_text()) for path in sorted(source.glob("episode_*.json"))]
    if not records:
        parser.error("No saved correction episodes were found")

    output.mkdir(parents=True, exist_ok=True)
    index_records = []
    for episode_index, record in enumerate(records, start=1):
        archive_path = source / record["archive"]
        if not archive_path.is_file() or not record.get("saved"):
            raise RuntimeError(f"Incomplete saved episode: {archive_path}")
        statuses = {}
        with np.load(archive_path, allow_pickle=False) as archive:
            for camera in camera_names:
                if camera not in archive:
                    raise KeyError(f"{archive_path} does not contain camera {camera!r}")
                frames = archive[camera]
                if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
                    raise ValueError(f"Unexpected {camera} array in {archive_path}: {frames.shape}, {frames.dtype}")
                if len(frames) != record["frames"]:
                    raise ValueError(f"Frame count mismatch in {archive_path}")
                video_path = output / archive_path.stem / f"{camera}.mp4"
                statuses[camera] = encode_video(frames, video_path, fps, args.overwrite)
                del frames
        index_records.append(
            {
                "episode": episode_index,
                "archive": record["archive"],
                "instruction": record["instruction"],
                "frames": record["frames"],
                "videos": statuses,
            }
        )
        print(
            json.dumps(
                {
                    "episode": episode_index,
                    "total": len(records),
                    "instruction": record["instruction"],
                    "frames": record["frames"],
                    "videos": statuses,
                }
            ),
            flush=True,
        )

    write_index(output, source, records, camera_names, fps, bool(summary.get("target_reached")))
    (output / "video_index.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "fps": fps,
                "cameras": camera_names,
                "target_reached": bool(summary.get("target_reached")),
                "episodes": index_records,
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "watch.html")


if __name__ == "__main__":
    main()
