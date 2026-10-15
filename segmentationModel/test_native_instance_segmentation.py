"""Run a short visual smoke test of YOLO26n-seg on a video in ``raws/``.

From the project root, run:

    python segmentationModel/test_native_instance_segmentation.py

Pass a different video path as the first argument. The script writes an
annotated MP4 under ``segmentationModel/`` and prints per-frame detections.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from nativeInstanceSegmentation import segment_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "raws" / "57583_000082_Sideline.mp4"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "test_output.mp4"
COCO_SKELETON_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def _instance_color(instance_key: int) -> tuple[int, int, int]:
    """Return a repeatable, reasonably bright BGR color for an instance."""
    hue = ((instance_key * 0.61803398875) % 1.0)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def _infer_pose_for_mask(
    pose_model: YOLO,
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Infer one COCO skeleton from a mask-isolated instance crop."""
    height, width = frame.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None

    # Crop bounds come from the instance mask, then add modest context for joints
    # close to the silhouette edge. Pixels from other instances are removed.
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    pad_x = max(4, int((x_max - x_min) * 0.12))
    pad_y = max(4, int((y_max - y_min) * 0.12))
    crop_x1, crop_y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
    crop_x2, crop_y2 = min(width, x_max + pad_x), min(height, y_max + pad_y)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
    crop[~crop_mask] = 0
    device_arg: str | int = 0 if device == "cuda" else device
    results = pose_model.predict(crop, verbose=False, conf=0.1, device=device_arg)
    if not results or results[0].keypoints is None:
        return None

    keypoints = results[0].keypoints
    xy = keypoints.xy.detach().cpu().numpy()
    confidences = keypoints.conf
    if confidences is None or len(xy) == 0:
        return None
    confidences_np = confidences.detach().cpu().numpy()
    best_index = int(np.argmax(np.nanmean(confidences_np, axis=1)))
    points = xy[best_index].astype(np.float32)
    points[:, 0] += crop_x1
    points[:, 1] += crop_y1
    return points, confidences_np[best_index].astype(np.float32)


def _draw_skeleton(
    frame: np.ndarray,
    points: np.ndarray,
    confidences: np.ndarray,
    color: tuple[int, int, int],
    *,
    confidence_threshold: float = 0.2,
) -> None:
    """Draw COCO joints and limbs in place."""
    visible = confidences >= confidence_threshold
    for first, second in COCO_SKELETON_EDGES:
        if visible[first] and visible[second]:
            p1 = tuple(np.rint(points[first]).astype(int))
            p2 = tuple(np.rint(points[second]).astype(int))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for point, is_visible in zip(points, visible):
        if is_visible:
            x, y = np.rint(point).astype(int)
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"Video to process (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument(
        "--pose-model",
        default=str(PROJECT_ROOT / "yolov8n-pose.pt"),
        help="YOLO pose checkpoint used on each mask-isolated person crop",
    )
    parser.add_argument(
        "--pose-class-id",
        type=int,
        default=0,
        help="Segmentation class to run pose on (COCO person is 0; use -1 for all)",
    )
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or cuda")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Maximum frames to process; use 0 to process the full video",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not video_path.is_file():
        parser.error(f"Video not found: {video_path}")
    if args.max_frames < 0:
        parser.error("--max-frames must be 0 or greater")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        parser.error(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    predictions = segment_video(
        video_path,
        model_path=args.model,
        device=args.device,
    )
    processed_frames = 0
    total_detections = 0

    try:
        pose_model = YOLO(args.pose_model)
        for frame_result in predictions:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Video ended before segmentation results")

            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output_path}")

            annotated = frame.copy()
            ids = frame_result.track_ids
            pose_count = 0
            for index, mask in enumerate(frame_result.masks):
                mask = mask.astype(np.uint8) * 255
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)

                track_id = int(ids[index]) if ids is not None else None
                color = _instance_color(track_id if track_id is not None else index + 1)
                class_id = int(frame_result.class_ids[index])

                overlay = annotated.copy()
                overlay[mask] = color
                annotated = cv2.addWeighted(annotated, 0.65, overlay, 0.35, 0)

                if args.pose_class_id == -1 or class_id == args.pose_class_id:
                    pose = _infer_pose_for_mask(
                        pose_model,
                        frame,
                        mask,
                        device=args.device,
                    )
                    if pose is not None:
                        _draw_skeleton(annotated, *pose, color)
                        pose_count += 1

                ys, xs = np.nonzero(mask)
                if len(xs):
                    x, y = int(xs.min()), int(ys.min())
                    label = f"class={class_id}"
                    label += f" id={track_id}" if track_id is not None else " id=?"
                    cv2.putText(
                        annotated,
                        label,
                        (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            writer.write(annotated)
            detections = len(frame_result.class_ids)
            total_detections += detections
            processed_frames += 1
            print(
                f"frame {frame_result.frame_index}: {detections} instances, "
                f"{pose_count} skeletons; "
                f"track IDs={frame_result.track_ids}"
            )

            if args.max_frames and processed_frames >= args.max_frames:
                break
    finally:
        predictions.close()
        capture.release()
        if writer is not None:
            writer.release()

    if processed_frames == 0:
        raise RuntimeError(f"No frames were processed from {video_path}")

    print(
        f"Processed {processed_frames} frame(s), found {total_detections} "
        f"instance detections. Annotated video: {output_path}"
    )


if __name__ == "__main__":
    main()
