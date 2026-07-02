from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fused_batch_gen import get_features
from models.fused_model import MultiStageModel


LOGGER = logging.getLogger("monactseg.infer_from_features")

DEFAULT_DEMO_VIDEO = REPO_ROOT / "demo" / "demo.mp4"
DEFAULT_RGB_FEATURE = REPO_ROOT / "demo" / "features" / "rgb" / "demo.npy"
DEFAULT_SKELETON_FEATURE = REPO_ROOT / "demo" / "features" / "skeleton" / "demo.npy"
DEFAULT_SEGMENTATION_CKPT = REPO_ROOT / "checkpoints" / "segmentation" / "monactseg_fused_attention_gate.pth"
DEFAULT_ACTIONS_FILE = REPO_ROOT / "checkpoints" / "segmentation" / "monactseg_actions.txt"

MONKEY_BONES = [
    (0, 2),
    (1, 2),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (4, 8),
    (8, 9),
    (9, 10),
    (4, 11),
    (11, 12),
    (12, 13),
    (11, 14),
    (14, 15),
    (11, 16),
]
JOINT_COLORS = [
    (0, 0, 255),
    (0, 128, 255),
    (0, 200, 255),
    (0, 255, 255),
    (0, 220, 120),
    (80, 255, 80),
    (200, 180, 0),
    (255, 120, 0),
    (120, 255, 120),
    (120, 220, 60),
    (60, 180, 0),
    (0, 80, 255),
    (255, 120, 255),
    (255, 0, 180),
    (0, 255, 180),
    (60, 220, 120),
    (180, 180, 180),
]


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_actions(actions_file: Path) -> list[str]:
    ensure_exists(actions_file, "actions file")
    return [line.strip() for line in actions_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def default_run_dir(video_path: Path) -> Path:
    return REPO_ROOT / "outputs" / f"{video_path.stem}_from_features"


def probe_video_stream(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_tag_string,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"Unable to read video stream info: {video_path}")
    return streams[0]


def reference_encode_args(stream_info: dict[str, Any]) -> list[str]:
    codec_name = str(stream_info.get("codec_name") or "").lower()
    codec_tag = str(stream_info.get("codec_tag_string") or "").lower()
    profile = str(stream_info.get("profile") or "").lower()
    pix_fmt = str(stream_info.get("pix_fmt") or "yuv420p")

    args = ["-an"]
    if codec_name == "h264":
        args.extend(["-c:v", "libx264", "-preset", "slow", "-crf", "30"])
        if profile in {"baseline", "main", "high", "high10", "high422", "high444"}:
            args.extend(["-profile:v", profile])
        args.extend(["-pix_fmt", pix_fmt])
        if codec_tag:
            args.extend(["-tag:v", codec_tag])
    else:
        args.extend(["-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", pix_fmt])
    args.extend(["-movflags", "+faststart"])
    return args


def run_subprocess(cmd: list[str]) -> None:
    LOGGER.info("Running: %s", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def smooth_boundary(boundary_probs: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return boundary_probs
    boundary_probs = gaussian_filter1d(boundary_probs, sigma=sigma)
    max_val = boundary_probs.max()
    if max_val > 0:
        boundary_probs = boundary_probs / max_val
    return boundary_probs


def find_boundary_peaks(boundary_probs: np.ndarray, threshold: float, min_distance: int) -> list[int]:
    candidates: list[int] = []
    for idx in range(1, boundary_probs.shape[0] - 1):
        if (
            boundary_probs[idx] > threshold
            and boundary_probs[idx] > boundary_probs[idx - 1]
            and boundary_probs[idx] >= boundary_probs[idx + 1]
        ):
            candidates.append(idx)
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda idx: boundary_probs[idx], reverse=True)
    selected: list[int] = []
    for idx in candidates:
        if all(abs(idx - chosen) >= min_distance for chosen in selected):
            selected.append(idx)
    return sorted(selected)


def postprocess_segments(
    class_probs: np.ndarray,
    boundary_probs: np.ndarray,
    threshold: float,
    min_distance: int,
) -> np.ndarray:
    num_classes, num_frames = class_probs.shape
    peaks = find_boundary_peaks(boundary_probs, threshold=threshold, min_distance=min_distance)
    boundaries = [0] + peaks + [num_frames]

    class_ids = np.argmax(class_probs, axis=0)
    output = np.zeros(num_frames, dtype=np.int64)
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if end <= start:
            continue
        segment_labels = class_ids[start:end]
        counts = np.bincount(segment_labels, minlength=num_classes)
        tied = np.where(counts == counts.max())[0]
        if len(tied) == 1:
            chosen = tied[0]
        else:
            chosen = tied[int(np.argmax(class_probs[tied, start:end].sum(axis=1)))]
        output[start:end] = chosen
    return output


def build_model(num_classes: int, checkpoint_path: Path, device: torch.device) -> MultiStageModel:
    model = MultiStageModel(
        dil=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        num_layers_RF=10,
        num_R=3,
        num_f_maps=64,
        skeleton_features_dim=6,
        rgb_features_dim=2048,
        num_classes=num_classes,
        num_layers_PG=10,
        feedback_features=False,
    )
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_feature_inputs(
    skeleton_path: Path,
    rgb_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    ensure_exists(skeleton_path, "skeleton feature")
    ensure_exists(rgb_path, "rgb feature")

    skeleton_raw = np.load(skeleton_path, allow_pickle=False).astype(np.float32, copy=False)
    rgb_raw = np.load(rgb_path, allow_pickle=False).astype(np.float32, copy=False)
    if skeleton_raw.ndim != 4 or skeleton_raw.shape[0] != 2 or skeleton_raw.shape[2] != 17:
        raise ValueError(f"Expected skeleton feature with shape (2, T, 17, 1), got {skeleton_raw.shape}")
    if rgb_raw.ndim != 2:
        raise ValueError(f"Expected RGB feature with shape (2048, T), got {rgb_raw.shape}")

    skeleton_model_features = get_features(skeleton_raw).astype(np.float32, copy=False)
    lengths = {
        "skeleton_raw": int(skeleton_raw.shape[1]),
        "skeleton_model": int(skeleton_model_features.shape[1]),
        "rgb": int(rgb_raw.shape[1]),
    }
    effective_len = min(lengths.values())
    if effective_len <= 0:
        raise ValueError(f"Empty feature sequence detected: {lengths}")

    return (
        skeleton_raw[:, :effective_len, :, :],
        skeleton_model_features[:, :effective_len, :, :],
        rgb_raw[:, :effective_len],
        lengths,
    )


def predict_labels(
    *,
    model: MultiStageModel,
    skeleton_features: np.ndarray,
    rgb_features: np.ndarray,
    actions: list[str],
    device: torch.device,
    use_postprocess: bool,
    boundary_smooth: float,
    boundary_thresh: float,
    boundary_min_dist: int,
) -> dict[str, np.ndarray | list[str]]:
    num_classes = len(actions)
    ske_in = torch.from_numpy(skeleton_features).unsqueeze(0).to(device=device, dtype=torch.float32)
    rgb_in = torch.from_numpy(rgb_features).unsqueeze(0).to(device=device, dtype=torch.float32)
    mask = torch.ones((1, num_classes, ske_in.shape[2]), device=device, dtype=torch.float32)

    with torch.no_grad():
        fused_preds, _, _, _, _ = model(ske_in, rgb_in, mask)
        logits = fused_preds[-1].squeeze(0)
        class_probs = F.softmax(logits, dim=0).cpu().numpy()

    raw_indices = np.argmax(class_probs, axis=0).astype(np.int64, copy=False)
    boundary_pred = 1.0 - np.sum(class_probs[:, 1:] * class_probs[:, :-1], axis=0)
    boundary_probs = np.zeros(class_probs.shape[1], dtype=np.float32)
    boundary_probs[1:] = boundary_pred
    boundary_probs = smooth_boundary(boundary_probs, boundary_smooth)

    if use_postprocess:
        pred_indices = postprocess_segments(
            class_probs,
            boundary_probs,
            threshold=boundary_thresh,
            min_distance=boundary_min_dist,
        )
    else:
        pred_indices = raw_indices

    idx_to_action = {idx: action for idx, action in enumerate(actions)}
    return {
        "pred_labels": [idx_to_action[int(idx)] for idx in pred_indices],
    }


def segment_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    if not labels:
        return []

    segments: list[tuple[str, int, int]] = []
    start = 0
    current = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] == current:
            continue
        segments.append((current, start, idx - 1))
        start = idx
        current = labels[idx]

    segments.append((current, start, len(labels) - 1))
    return segments


def write_prediction_txt(path: Path, labels: Iterable[str]) -> None:
    path.write_text("### Frame level recognition: ###\n" + " ".join(labels), encoding="utf-8")


def action_colors(actions: list[str]) -> dict[str, tuple[int, int, int]]:
    palette = [
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (230, 126, 34),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (149, 165, 166),
    ]
    return {action: palette[idx % len(palette)] for idx, action in enumerate(actions)}


def parse_fps(fps_value: str) -> float:
    text = str(fps_value)
    if "/" in text:
        num, den = text.split("/", 1)
        den_value = float(den)
        return float(num) / den_value if den_value != 0 else 0.0
    return float(text)


def joint_is_valid(x: float, y: float) -> bool:
    return 0.0 < x <= 1.0 and 0.0 < y <= 1.0


def draw_skeleton_overlay(
    frame: np.ndarray,
    skeleton_frame: np.ndarray,
) -> None:
    height, width = frame.shape[:2]
    xy = skeleton_frame[:, :, 0]

    for start, end in MONKEY_BONES:
        x1, y1 = float(xy[0, start]), float(xy[1, start])
        x2, y2 = float(xy[0, end]), float(xy[1, end])
        if not (joint_is_valid(x1, y1) and joint_is_valid(x2, y2)):
            continue
        pt1 = (int(round(x1 * width)), int(round(y1 * height)))
        pt2 = (int(round(x2 * width)), int(round(y2 * height)))
        color = JOINT_COLORS[end % len(JOINT_COLORS)]
        cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)

    for joint_idx, color in enumerate(JOINT_COLORS):
        x, y = float(xy[0, joint_idx]), float(xy[1, joint_idx])
        if not joint_is_valid(x, y):
            continue
        px = int(round(x * width))
        py = int(round(y * height))
        cv2.circle(frame, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)


def write_prediction_video(
    video_path: Path,
    output_path: Path,
    labels: list[str],
    skeleton_raw: np.ndarray,
    actions: list[str],
    reference_stream: dict[str, Any],
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for visualization: {video_path}")

    temp_video = output_path.with_suffix(".tmp_render.avi")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video size for visualization: {(width, height)}")

        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create temporary render video: {temp_video}")

        colors = action_colors(actions)
        total = max(len(labels), 1)
        segments = segment_spans(labels)
        timeline_top = max(height - 44, 0)
        timeline_bottom = max(height - 20, 0)
        timeline_width = max(width - 40, 1)
        timeline_left = 20
        timeline_right = timeline_left + timeline_width
        actual_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            label_idx = min(actual_frames, total - 1)
            current_label = labels[label_idx]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (min(width, 640), min(height, 126)), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, max(height - 64, 0)), (width, height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            skeleton_idx = min(label_idx, skeleton_raw.shape[1] - 1)
            draw_skeleton_overlay(frame, skeleton_raw[:, skeleton_idx, :, :])

            cv2.putText(
                frame,
                f"Pred: {current_label}",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                colors[current_label],
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Frame {actual_frames + 1}/{total}",
                (24, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Time {actual_frames / fps:.2f}s",
                (220, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for label, start_frame, end_frame in segments:
                seg_left = timeline_left + int(timeline_width * (start_frame / total))
                seg_right = timeline_left + int(timeline_width * ((end_frame + 1) / total))
                seg_right = max(seg_right, seg_left + 1)
                cv2.rectangle(
                    frame,
                    (seg_left, timeline_top),
                    (min(seg_right, timeline_right), timeline_bottom),
                    colors[label],
                    -1,
                )

            marker_x = timeline_left + int(timeline_width * (label_idx / total))
            cv2.rectangle(frame, (timeline_left, timeline_top), (timeline_right, timeline_bottom), (255, 255, 255), 1)
            cv2.line(
                frame,
                (marker_x, max(timeline_top - 4, 0)),
                (marker_x, min(timeline_bottom + 4, height - 1)),
                (255, 255, 255),
                2,
            )

            writer.write(frame)
            actual_frames += 1

        writer.release()
        run_subprocess(["ffmpeg", "-y", "-i", str(temp_video), *reference_encode_args(reference_stream), str(output_path)])
    finally:
        cap.release()
        if temp_video.exists():
            temp_video.unlink()


def parse_args() -> argparse.Namespace:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Inference from precomputed RGB and skeleton features, with skeleton overlay rendering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-path", type=Path, default=DEFAULT_DEMO_VIDEO)
    parser.add_argument("--rgb-feature", type=Path, default=DEFAULT_RGB_FEATURE)
    parser.add_argument("--skeleton-feature", type=Path, default=DEFAULT_SKELETON_FEATURE)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: outputs/<video_stem>_from_features")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SEGMENTATION_CKPT)
    parser.add_argument("--actions-file", type=Path, default=DEFAULT_ACTIONS_FILE)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--boundary-smooth", type=float, default=0.0)
    parser.add_argument("--boundary-thresh", type=float, default=0.5)
    parser.add_argument("--boundary-min-dist", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.video_path = args.video_path.resolve()
    args.rgb_feature = args.rgb_feature.resolve()
    args.skeleton_feature = args.skeleton_feature.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.actions_file = args.actions_file.resolve()

    ensure_exists(args.video_path, "input video")
    ensure_exists(args.rgb_feature, "rgb feature")
    ensure_exists(args.skeleton_feature, "skeleton feature")
    ensure_exists(args.checkpoint, "segmentation checkpoint")
    ensure_exists(args.actions_file, "actions file")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_run_dir(args.video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    inferred_rgb_fps = args.rgb_feature.with_name(f"{args.rgb_feature.stem}_fps.npy")
    inferred_timestamps = args.rgb_feature.with_name(f"{args.rgb_feature.stem}_timestamps_ms.npy")
    video_stream = probe_video_stream(args.video_path)
    video_fps = parse_fps(video_stream.get("avg_frame_rate", "30/1"))

    skeleton_raw, skeleton_model_features, rgb_features, _feature_lengths = load_feature_inputs(
        args.skeleton_feature,
        args.rgb_feature,
    )

    if inferred_rgb_fps.exists():
        rgb_fps = float(np.load(inferred_rgb_fps, allow_pickle=False))
    else:
        rgb_fps = video_fps if video_fps > 0 else 30.0

    if inferred_timestamps.exists():
        timestamps_ms = np.load(inferred_timestamps, allow_pickle=False).astype(np.float32, copy=False)
    else:
        timestamps_ms = np.arange(skeleton_raw.shape[1], dtype=np.float32) * (1000.0 / rgb_fps if rgb_fps > 0 else 1.0)

    effective_len = min(
        skeleton_raw.shape[1],
        skeleton_model_features.shape[1],
        rgb_features.shape[1],
        int(timestamps_ms.shape[0]) if timestamps_ms.ndim > 0 else skeleton_raw.shape[1],
    )
    skeleton_raw = skeleton_raw[:, :effective_len, :, :]
    skeleton_model_features = skeleton_model_features[:, :effective_len, :, :]
    rgb_features = rgb_features[:, :effective_len]
    timestamps_ms = timestamps_ms[:effective_len]

    actions = load_actions(args.actions_file)
    device = torch.device(args.device)
    model = build_model(len(actions), args.checkpoint, device)
    prediction = predict_labels(
        model=model,
        skeleton_features=skeleton_model_features,
        rgb_features=rgb_features,
        actions=actions,
        device=device,
        use_postprocess=not args.disable_postprocess,
        boundary_smooth=args.boundary_smooth,
        boundary_thresh=args.boundary_thresh,
        boundary_min_dist=args.boundary_min_dist,
    )

    frame_labels = list(prediction["pred_labels"])

    pred_txt = output_dir / f"{args.video_path.stem}_prediction.txt"
    video_out = output_dir / f"{args.video_path.stem}_out.mp4"

    write_prediction_txt(pred_txt, frame_labels)
    if not args.skip_video:
        write_prediction_video(
            args.video_path,
            video_out,
            frame_labels,
            skeleton_raw,
            actions,
            video_stream,
        )

    LOGGER.info("Prediction text saved to: %s", pred_txt)
    if not args.skip_video:
        LOGGER.info("Prediction video saved to: %s", video_out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    raise SystemExit(main())
