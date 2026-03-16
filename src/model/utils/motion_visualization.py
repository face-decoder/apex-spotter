import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.face.modules import FaceLandmark

matplotlib.use("Agg")


def _render_figure_to_array(fig: plt.Figure) -> np.ndarray:
    """Rasterize a Matplotlib figure to a BGR NumPy array (OpenCV format)."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img_bgr


def _create_base_plot(magnitude_signal: np.ndarray,
                      plot_height: int,
                      plot_width: int,
                      apex_indices: Optional[List[int]] = None,
                      phases: Optional[Dict[int, dict]] = None,
                      title: str = "Optical Flow Magnitude") -> tuple[np.ndarray, plt.Figure, plt.Axes]:
    dpi = 100
    fig_w = plot_width / dpi
    fig_h = plot_height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.plot(magnitude_signal, color="#3b82f6", linewidth=1.2, label="Signal")

    signal_arr = np.asarray(magnitude_signal, dtype=np.float64)
    mean_val = float(np.mean(signal_arr))
    std_val = float(np.std(signal_arr))
    threshold = mean_val + std_val
    ax.axhline(
        y=threshold,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label=f"Mean+1σ ({threshold:.4f})",
    )

    if phases is not None:
        first_phase = True
        for _apex_idx, phase in phases.items():
            ax.axvspan(
                phase["start"],
                phase["end"],
                color="orange",
                alpha=0.3,
                label="Apex Phase" if first_phase else "",
            )
            first_phase = False

    # Apex markers
    if apex_indices is not None and len(apex_indices) > 0:
        apex_vals = [float(magnitude_signal[i]) for i in apex_indices]
        ax.scatter(
            apex_indices,
            apex_vals,
            color="#ef4444",
            s=40,
            zorder=5,
            label="Apex",
        )

    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    base_img = _render_figure_to_array(fig)
    return base_img, fig, ax


def _frame_to_pixel_x(frame_idx: int,
                      total_frames: int,
                      ax: plt.Axes,
                      fig: plt.Figure,
                      img_width: int) -> int:
    """Convert a frame index to the pixel x-coordinate on the rasterized image."""
    
    display_coords = ax.transData.transform((frame_idx, 0))
    
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    
    norm_x = display_coords[0] / fig_w_px

    return int(norm_x * img_width)



def create_motion_comparison_video(video_path: str,
                                   magnitude_signal: np.ndarray,
                                   output_path: str,
                                   apex_indices: Optional[List[int]] = None,
                                   phases: Optional[Dict[int, dict]] = None,
                                   crop_size: Tuple[int, int] = (160, 160),
                                   plot_width: int = 480,
                                   title: str = "Optical Flow Magnitude") -> None:

    """Generate a synchronized side-by-side comparison video.

    Left panel shows the **cropped face** (tracked via ``FaceLandmark``)
    and the right panel shows the smoothed magnitude signal with apex
    phase regions, apex markers, and an auto-computed mean+1σ threshold
    line — adapted from ``ApexPhaseVisualizer.plot_phases()``.

    Args:
        video_path (str): Path to the input facial video.
        magnitude_signal (np.ndarray): 1-D array of (smoothed) optical flow magnitude values, one per frame.
        output_path (str): Destination path for the output MP4 video.
        apex_indices (list[int], optional): Frame indices of detected apex points (plotted as markers).
        phases (dict[int, dict], optional): Apex phase dict ``{apex_idx: {"start": int, "end": int}}``.
        crop_size (tuple[int, int]): Output size ``(height, width)`` of the cropped face panel.
        plot_width (int): Width of the right panel (the magnitude plot).
        title (str): Title string for the magnitude plot.

    Raises:
        ValueError: If the video cannot be opened, if the signal length doesn't match the video frame count, or if the output video cannot be created.
    """
    magnitude_signal = np.asarray(magnitude_signal, dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(magnitude_signal) == total_frames - 1:
        magnitude_signal = np.concatenate(([magnitude_signal[0]], magnitude_signal))
    elif len(magnitude_signal) == total_frames + 1:
        magnitude_signal = magnitude_signal[:total_frames]
    elif len(magnitude_signal) != total_frames:
        cap.release()
        raise ValueError(
            f"Signal length ({len(magnitude_signal)}) does not match "
            f"video frame count ({total_frames})."
        )

    panel_h, panel_w = crop_size
    combined_w = panel_w + plot_width

    face_lm = FaceLandmark()

    base_plot_img, fig, ax = _create_base_plot(magnitude_signal,
                                               plot_height=panel_h,
                                               plot_width=plot_width,
                                               apex_indices=apex_indices,
                                               phases=phases,
                                               title=title)

    base_plot_img = cv2.resize(base_plot_img, (plot_width, panel_h))

    pixel_xs = []
    for i in range(total_frames):
        px = _frame_to_pixel_x(i, total_frames, ax, fig, plot_width)
        pixel_xs.append(px)

    plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, panel_h))

    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")

    ema_alpha = 0.15
    smooth_bbox = None
    crop_margin = 0.05
    effective_indices = face_lm.FACE_OVAL + face_lm.STABLE_POINTS

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]

        try:
            lm_result = face_lm.detect(frame)
            if not lm_result.face_landmarks:
                raise ValueError("No face detected")

            face_landmarks = lm_result.face_landmarks[0]
            xs = [face_landmarks[i].x * fw for i in effective_indices]
            ys = [face_landmarks[i].y * fh for i in effective_indices]

            raw_x_min, raw_x_max = min(xs), max(xs)
            raw_y_min, raw_y_max = min(ys), max(ys)

            dx = (raw_x_max - raw_x_min) * crop_margin
            dy = (raw_y_max - raw_y_min) * crop_margin
            raw_bbox = (
                raw_x_min - dx,
                raw_y_min - dy,
                raw_x_max + dx,
                raw_y_max + dy,
            )

            if smooth_bbox is None:
                smooth_bbox = raw_bbox
            else:
                smooth_bbox = tuple(
                    ema_alpha * r + (1.0 - ema_alpha) * s
                    for r, s in zip(raw_bbox, smooth_bbox)
                )

            sx_min = int(max(0, smooth_bbox[0]))
            sy_min = int(max(0, smooth_bbox[1]))
            sx_max = int(min(fw, smooth_bbox[2]))
            sy_max = int(min(fh, smooth_bbox[3]))

            face_crop = frame[sy_min:sy_max, sx_min:sx_max]
            left = cv2.resize(face_crop, (panel_w, panel_h))

        except (ValueError, Exception):
            left = cv2.resize(frame, (panel_w, panel_h))

        right = base_plot_img.copy()
        x = pixel_xs[frame_idx]
        cv2.line(right, (x, 0), (x, panel_h), color=(0, 0, 255), thickness=2)

        combined = np.hstack([left, right])
        writer.write(combined)

    cap.release()
    writer.release()
