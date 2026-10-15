"""Video instance segmentation and tracking with YOLO26n-seg.

Install the optional vision dependencies with ``pip install -e '.[vision]'``.
The returned masks are binary arrays aligned with the detections for each
frame. Tracker IDs are ``None`` when the tracker did not assign IDs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameInstances:
    """Instance segmentation results for one decoded video frame."""

    frame_index: int
    masks: np.ndarray  # (N, H, W), bool
    class_ids: np.ndarray  # (N,), int
    confidences: np.ndarray  # (N,), float
    track_ids: np.ndarray | None  # (N,), int; None until tracker IDs exist


def segment_video(
    video_path: str | Path,
    *,
    model_path: str = "yolo26n-seg.pt",
    device: str = "cpu",
    tracker: str = "bytetrack.yaml",
    confidence: float = 0.25,
    iou: float = 0.5,
) -> Iterator[FrameInstances]:
    """Yield instance masks and persistent track IDs for each video frame.

    ``model_path`` may be the default pretrained YOLO26n-seg checkpoint or a
    local segmentation checkpoint. Frames are decoded with OpenCV and passed
    to Ultralytics one at a time to avoid holding a full video in memory.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ModuleNotFoundError(
            "Missing vision dependencies. Install with `pip install -e '.[vision]'`."
        ) from exc

    source = Path(video_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {source}")

    frame_index = 0

    try:
        model = YOLO(model_path)
        device_arg: Any = 0 if device == "cuda" else device
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=confidence,
                iou=iou,
                device=device_arg,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            masks_obj = result.masks

            if boxes is None or len(boxes) == 0:
                masks = np.empty((0, frame.shape[0], frame.shape[1]), dtype=bool)
                class_ids = np.empty((0,), dtype=np.int64)
                confidences = np.empty((0,), dtype=np.float32)
                track_ids = np.empty((0,), dtype=np.int64)
            else:
                if masks_obj is None:
                    raise RuntimeError("The selected model returned boxes without instance masks")
                masks = masks_obj.data.detach().cpu().numpy().astype(bool)
                class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
                confidences = boxes.conf.detach().cpu().numpy().astype(np.float32)
                ids = boxes.id
                track_ids = (
                    ids.detach().cpu().numpy().astype(np.int64) if ids is not None else None
                )

            yield FrameInstances(
                frame_index=frame_index,
                masks=masks,
                class_ids=class_ids,
                confidences=confidences,
                track_ids=track_ids,
            )
            frame_index += 1
    finally:
        capture.release()
