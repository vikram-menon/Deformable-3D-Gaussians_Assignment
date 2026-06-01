#
# Bonus trajectory diagnostics for trained Deformable 3D Gaussians models.
#

from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv
import html
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, DeformModel
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import geom_transform_points
from utils.rigid_utils import from_homogenous, to_homogenous
from utils.system_utils import searchForMaxIteration


PAPER_CITATIONS = {
    "Deformable 3D Gaussians": {
        "citation": "Yang et al., Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction, arXiv:2309.13101, 2023.",
        "url": "https://arxiv.org/abs/2309.13101",
        "used_for": "Sampling canonical Gaussians through the learned deformation field to form trajectories.",
    },
    "TAP-Vid": {
        "citation": "Doersch et al., TAP-Vid: A Benchmark for Tracking Any Point in a Video, NeurIPS Datasets and Benchmarks, 2022.",
        "url": "https://arxiv.org/abs/2211.03726",
        "used_for": "Track-level statistics such as trajectory length, diameter, and clustering-style motion summaries.",
    },
    "TAPVid-3D": {
        "citation": "Koppula et al., TAPVid-3D: A Benchmark for Tracking Any Point in 3D, arXiv:2407.05921, 2024.",
        "url": "https://arxiv.org/abs/2407.05921",
        "used_for": "3D point-trajectory evaluation framing and the APD/OA/AJ terminology that requires ground truth.",
    },
    "TAPIR": {
        "citation": "Doersch et al., TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement, ICCV, 2023.",
        "url": "https://arxiv.org/abs/2306.08637",
        "used_for": "Long-range point-tracking motivation and trajectory quality context.",
    },
    "CoTracker3": {
        "citation": "Karaev et al., CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos, arXiv:2410.11831, 2024.",
        "url": "https://arxiv.org/abs/2410.11831",
        "used_for": "Current point-tracking context for visible and occluded tracks.",
    },
    "4D-GS": {
        "citation": "Wu et al., 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering, CVPR, 2024.",
        "url": "https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html",
        "used_for": "Gaussian deformation-field and spatial-temporal coherence motivation.",
    },
    "MoSca": {
        "citation": "Lei et al., MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds, arXiv:2405.17421, 2024.",
        "url": "https://arxiv.org/abs/2405.17421",
        "used_for": "Motion scaffold and coherent deformation-field framing for dynamic Gaussian reconstruction.",
    },
    "MonST3R": {
        "citation": "Zhang et al., MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion, ICLR, 2025.",
        "url": "https://arxiv.org/abs/2410.03825",
        "used_for": "Recent geometry-first dynamic 3D reconstruction context.",
    },
    "Dynamic Point Maps": {
        "citation": "Sucar et al., Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction, arXiv:2503.16318, 2025.",
        "url": "https://arxiv.org/abs/2503.16318",
        "used_for": "Current dynamic 3D point-map, scene-flow, and object-tracking context.",
    },
}

TRACK_COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 183, 3),
    (17, 138, 178),
    (6, 214, 160),
]


def finite_float(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def summarize(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p99": None, "min": None, "max": None}
    return {
        "count": int(values.numel()),
        "mean": finite_float(values.mean()),
        "median": finite_float(values.median()),
        "p90": finite_float(torch.quantile(values, 0.90)),
        "p99": finite_float(torch.quantile(values, 0.99)),
        "min": finite_float(values.min()),
        "max": finite_float(values.max()),
    }


def topk_records(values, gaussian_ids, k=10):
    values = torch.as_tensor(values).flatten()
    if values.numel() == 0:
        return []
    k = min(k, values.numel())
    top_values, top_idx = torch.topk(values, k=k, largest=True)
    return [
        {"rank": rank + 1, "gaussian_id": int(gaussian_ids[int(idx)]), "value": finite_float(value)}
        for rank, (idx, value) in enumerate(zip(top_idx.tolist(), top_values.tolist()))
    ]


def summarize_group(mask, gaussian_ids, opacities, labels, metrics, mean_stretch, max_stretch):
    mask = torch.as_tensor(mask, dtype=torch.bool).flatten()
    gaussian_ids = torch.as_tensor(gaussian_ids).flatten()
    opacities = torch.as_tensor(opacities).flatten()
    labels = torch.as_tensor(labels).flatten()
    count = int(mask.sum())
    if count == 0:
        return {
            "count": 0,
            "fraction": 0.0,
            "opacity": summarize([]),
            "motion": {},
            "smoothness": {},
            "local_coherence": {},
            "cluster_counts": [],
        }

    group_ids = gaussian_ids[mask]
    group_labels = labels[mask] if labels.numel() else torch.empty(0, dtype=torch.long)
    cluster_counts = []
    if group_labels.numel():
        counts = torch.bincount(group_labels)
        cluster_counts = [{"cluster": idx, "count": int(count)} for idx, count in enumerate(counts.tolist()) if count > 0]

    return {
        "count": count,
        "fraction": finite_float(count / max(mask.numel(), 1)),
        "opacity": summarize(opacities[mask]),
        "motion": {
            "displacement": summarize(metrics["displacement"][mask]),
            "path_length": summarize(metrics["path_length"][mask]),
            "diameter": summarize(metrics["diameter"][mask]),
            "mean_speed": summarize(metrics["mean_speed"][mask]),
            "peak_speed": summarize(metrics["peak_speed"][mask]),
            "path_length_outliers": topk_records(metrics["path_length"][mask], group_ids),
            "peak_speed_outliers": topk_records(metrics["peak_speed"][mask], group_ids),
        },
        "smoothness": {
            "mean_acceleration": summarize(metrics["mean_acceleration"][mask]),
            "mean_jerk": summarize(metrics["mean_jerk"][mask]),
            "normalized_jerk": summarize(metrics["normalized_jerk"][mask]),
            "normalized_jerk_outliers": topk_records(metrics["normalized_jerk"][mask], group_ids),
        },
        "local_coherence": {
            "mean_local_stretch": summarize(mean_stretch[mask]) if mean_stretch.numel() else summarize([]),
            "max_local_stretch": summarize(max_stretch[mask]) if max_stretch.numel() else summarize([]),
            "local_stretch_outliers": topk_records(mean_stretch[mask], group_ids) if mean_stretch.numel() else [],
        },
        "cluster_counts": cluster_counts,
    }


def make_histogram_png(values, path, title, bins=48, size=(900, 520)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    width, height = size
    margin_l, margin_r, margin_t, margin_b = 72, 24, 52, 70
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin_l, 18), title, fill=(20, 20, 20))

    if values.size == 0:
        draw.text((margin_l, height // 2), "No finite values", fill=(20, 20, 20))
        img.save(path)
        return

    counts, edges = np.histogram(values, bins=bins)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, margin_t + plot_h
    draw.line((x0, margin_t, x0, y0), fill=(40, 40, 40), width=2)
    draw.line((x0, y0, width - margin_r, y0), fill=(40, 40, 40), width=2)

    max_count = max(int(counts.max()), 1)
    bar_w = plot_w / len(counts)
    for i, count in enumerate(counts):
        x_left = margin_l + i * bar_w
        x_right = margin_l + (i + 1) * bar_w - 1
        bar_h = plot_h * (count / max_count)
        draw.rectangle((x_left, y0 - bar_h, x_right, y0), fill=(70, 118, 184))

    draw.text((margin_l, height - 48), f"min {edges[0]:.4g}", fill=(20, 20, 20))
    draw.text((width // 2 - 80, height - 48), f"median {np.median(values):.4g}", fill=(20, 20, 20))
    draw.text((width - margin_r - 160, height - 48), f"max {edges[-1]:.4g}", fill=(20, 20, 20))
    draw.text((margin_l, height - 26), f"n={values.size}", fill=(20, 20, 20))
    img.save(path)


def make_bar_png(values, path, title, xlabel="", size=(900, 520)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    width, height = size
    margin_l, margin_r, margin_t, margin_b = 72, 24, 52, 78
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin_l, 18), title, fill=(20, 20, 20))

    if values.size == 0:
        draw.text((margin_l, height // 2), "No finite values", fill=(20, 20, 20))
        img.save(path)
        return

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, margin_t + plot_h
    draw.line((x0, margin_t, x0, y0), fill=(40, 40, 40), width=2)
    draw.line((x0, y0, width - margin_r, y0), fill=(40, 40, 40), width=2)

    max_value = max(float(values.max()), 1e-12)
    gap = 5
    bar_w = max(5, (plot_w - gap * (values.size - 1)) / values.size)
    for idx, value in enumerate(values):
        x_left = margin_l + idx * (bar_w + gap)
        x_right = x_left + bar_w
        bar_h = plot_h * (float(value) / max_value)
        draw.rectangle((x_left, y0 - bar_h, x_right, y0), fill=(70, 118, 184))
        if idx < 10 or idx == values.size - 1:
            draw.text((x_left, y0 + 8), str(idx + 1), fill=(20, 20, 20))

    draw.text((margin_l, height - 34), xlabel, fill=(20, 20, 20))
    draw.text((width - margin_r - 180, height - 34), f"max {max_value:.4g}", fill=(20, 20, 20))
    img.save(path)


def blend_color(color, alpha, background=(245, 245, 245)):
    alpha = max(0.0, min(1.0, float(alpha)))
    return tuple(int(background[i] * (1.0 - alpha) + color[i] * alpha) for i in range(3))


def tensor_image_to_uint8(image_tensor):
    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image[:3].permute(1, 2, 0)
    image = (image.numpy() * 255.0).round().astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    return image


def select_track_indices(moving_mask, labels, path_length, track_count):
    moving_indices = torch.nonzero(moving_mask, as_tuple=False).flatten()
    if moving_indices.numel() == 0 or track_count <= 0:
        return torch.empty(0, dtype=torch.long)

    labels = torch.as_tensor(labels).flatten()
    path_length = torch.as_tensor(path_length).flatten()
    selected = []
    if labels.numel():
        moving_labels = labels[moving_indices]
        clusters = torch.unique(moving_labels).tolist()
        per_cluster = max(1, int(math.ceil(track_count / max(len(clusters), 1))))
        for cluster in clusters:
            cluster_indices = moving_indices[moving_labels == cluster]
            if cluster_indices.numel() == 0:
                continue
            order = torch.argsort(path_length[cluster_indices], descending=True)
            selected.extend(cluster_indices[order[:per_cluster]].tolist())

    if len(selected) < track_count:
        already = set(int(idx) for idx in selected)
        order = torch.argsort(path_length[moving_indices], descending=True)
        for idx in moving_indices[order].tolist():
            if int(idx) not in already:
                selected.append(int(idx))
            if len(selected) >= track_count:
                break

    return torch.as_tensor(selected[:track_count], dtype=torch.long)


def select_viewer_point_indices(moving_mask, path_length, opacities, point_count, mode="moving"):
    total = int(path_length.numel())
    if total == 0:
        return torch.empty(0, dtype=torch.long)

    moving_mask = torch.as_tensor(moving_mask, dtype=torch.bool).flatten()
    path_length = torch.as_tensor(path_length).flatten()
    opacities = torch.as_tensor(opacities).flatten()
    moving_indices = torch.nonzero(moving_mask, as_tuple=False).flatten()
    static_indices = torch.nonzero(~moving_mask, as_tuple=False).flatten()

    if mode == "moving":
        if point_count < 0 or point_count >= moving_indices.numel():
            return moving_indices
        if point_count == 0 or moving_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        order = torch.argsort(path_length[moving_indices], descending=True)
        return moving_indices[order[:point_count]]

    if mode == "all":
        if point_count < 0 or point_count >= total:
            return torch.arange(total, dtype=torch.long)
        if point_count == 0:
            return torch.empty(0, dtype=torch.long)
        order = torch.argsort(opacities, descending=True)
        return order[:point_count]

    if point_count <= 0:
        return torch.empty(0, dtype=torch.long)
    if point_count >= total:
        return torch.arange(total, dtype=torch.long)

    moving_count = min(moving_indices.numel(), max(point_count // 2, int(point_count * 0.65)))
    static_count = min(static_indices.numel(), point_count - moving_count)
    if static_count + moving_count < point_count:
        moving_count = min(moving_indices.numel(), point_count - static_count)

    selected = []
    if moving_count > 0:
        order = torch.argsort(path_length[moving_indices], descending=True)
        selected.extend(moving_indices[order[:moving_count]].tolist())
    if static_count > 0:
        order = torch.argsort(opacities[static_indices], descending=True)
        selected.extend(static_indices[order[:static_count]].tolist())

    if len(selected) < point_count:
        selected_set = set(int(idx) for idx in selected)
        order = torch.argsort(opacities, descending=True)
        for idx in order.tolist():
            if int(idx) not in selected_set:
                selected.append(int(idx))
            if len(selected) >= point_count:
                break

    return torch.as_tensor(selected[:point_count], dtype=torch.long)


def project_tracks_to_camera(track_subset, camera):
    projected = []
    with torch.no_grad():
        for frame_tracks in track_subset:
            projected.append(project_points(frame_tracks, camera))
    return torch.stack(projected, dim=0)


def make_paths_png(track_subset, labels, path, title, axes=(0, 1), size=(900, 620)):
    tracks_np = torch.as_tensor(track_subset).detach().cpu().numpy()
    width, height = size
    margin_l, margin_r, margin_t, margin_b = 72, 30, 54, 68
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin_l, 18), title, fill=(20, 20, 20))

    if tracks_np.size == 0 or tracks_np.shape[1] == 0:
        draw.text((margin_l, height // 2), "No moving tracks selected", fill=(20, 20, 20))
        img.save(path)
        return

    xy = tracks_np[:, :, list(axes)]
    finite = np.isfinite(xy).all(axis=-1)
    valid = xy[finite]
    if valid.size == 0:
        draw.text((margin_l, height // 2), "No finite track coordinates", fill=(20, 20, 20))
        img.save(path)
        return

    min_xy = valid.min(axis=0)
    max_xy = valid.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-8)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline=(210, 210, 210))
    for track_idx in range(xy.shape[1]):
        color = TRACK_COLORS[int(labels[track_idx]) % len(TRACK_COLORS)] if len(labels) else TRACK_COLORS[0]
        points = []
        for time_idx in range(xy.shape[0]):
            if not finite[time_idx, track_idx]:
                continue
            px = margin_l + (xy[time_idx, track_idx, 0] - min_xy[0]) / span[0] * plot_w
            py = margin_t + plot_h - (xy[time_idx, track_idx, 1] - min_xy[1]) / span[1] * plot_h
            points.append((float(px), float(py)))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
            draw.ellipse((points[-1][0] - 2, points[-1][1] - 2, points[-1][0] + 2, points[-1][1] + 2), fill=color)

    axis_names = ["X", "Y", "Z"]
    draw.text((margin_l, height - 42), f"{axis_names[axes[0]]} vs {axis_names[axes[1]]}; colors indicate motion clusters", fill=(20, 20, 20))
    draw.text((margin_l, height - 22), f"tracks={xy.shape[1]}, frames={xy.shape[0]}", fill=(20, 20, 20))
    img.save(path)


def camera_time(camera):
    fid = getattr(camera, "fid", None)
    if fid is None:
        return 0.0
    if torch.is_tensor(fid):
        return float(fid.detach().cpu().flatten()[0])
    return float(fid)


def select_camera_sequences(cameras, count):
    groups = {}
    for camera in cameras:
        key = getattr(camera, "colmap_id", None)
        if key is None:
            key = camera.image_name.rsplit("_", 1)[0]
        groups.setdefault(key, []).append(camera)

    sequences = []
    for key in sorted(groups.keys(), key=lambda value: str(value)):
        sequence = sorted(groups[key], key=lambda camera: (camera_time(camera), camera.image_name))
        if sequence:
            sequences.append((key, sequence))
        if len(sequences) >= count:
            break
    return sequences


def nearest_camera_for_time(camera_sequence, time_value):
    return min(camera_sequence, key=lambda camera: abs(camera_time(camera) - float(time_value)))


def render_track_overlay_videos(track_subset, selected_ids, selected_labels, camera_sequences, times, output_dir, trail_length, stride, fps=18):
    videos = []
    if imageio is None or track_subset.numel() == 0 or not camera_sequences:
        return videos

    output_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(stride))
    trail_length = max(1, int(trail_length))
    times_np = torch.as_tensor(times).detach().cpu().numpy()

    for camera_key, camera_sequence in camera_sequences:
        display_name = f"cam_{camera_key}" if isinstance(camera_key, int) else str(camera_key)
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in display_name)
        path = output_dir / f"track_overlay_{safe_name}.mp4"
        frames = []
        for time_idx in range(0, track_subset.shape[0], stride):
            frame_camera = nearest_camera_for_time(camera_sequence, times_np[time_idx])
            projected = project_tracks_to_camera(track_subset[max(0, time_idx - trail_length + 1):time_idx + 1], frame_camera).numpy()
            background = tensor_image_to_uint8(frame_camera.original_image)
            img = Image.fromarray(background.copy())
            draw = ImageDraw.Draw(img)
            for track_idx in range(projected.shape[1]):
                color = TRACK_COLORS[int(selected_labels[track_idx]) % len(TRACK_COLORS)] if len(selected_labels) else TRACK_COLORS[0]
                points = []
                for hist_idx in range(projected.shape[0]):
                    x, y = projected[hist_idx, track_idx]
                    if np.isfinite(x) and np.isfinite(y) and -50 <= x <= frame_camera.image_width + 50 and -50 <= y <= frame_camera.image_height + 50:
                        points.append((float(x), float(y)))
                if len(points) >= 2:
                    for segment_idx in range(1, len(points)):
                        alpha = segment_idx / max(len(points) - 1, 1)
                        draw.line((points[segment_idx - 1], points[segment_idx]), fill=blend_color(color, 0.25 + 0.75 * alpha), width=2)
                if points:
                    x, y = points[-1]
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.rectangle((12, 12, 352, 48), fill=(255, 255, 255), outline=(180, 180, 180))
            draw.text((22, 22), f"Moving tracks, frame {time_idx}, {frame_camera.image_name}", fill=(20, 20, 20))
            frames.append(np.asarray(img))
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        videos.append({"camera": display_name, "path": str(path), "track_count": int(track_subset.shape[1])})
    return videos


def export_track_samples(path, selected_ids, selected_labels, track_subset, path_lengths, projected_tracks, camera_names, viewer_cloud_subset=None, viewer_cloud_groups=None):
    np.savez_compressed(
        path,
        gaussian_ids=np.asarray(selected_ids, dtype=np.int64),
        clusters=np.asarray(selected_labels, dtype=np.int64),
        tracks_3d=track_subset.detach().cpu().numpy(),
        path_length=np.asarray(path_lengths, dtype=np.float32),
        projected_tracks=np.asarray(projected_tracks, dtype=np.float32),
        camera_names=np.asarray(camera_names, dtype=str),
        viewer_cloud_3d=torch.as_tensor(viewer_cloud_subset).detach().cpu().numpy() if viewer_cloud_subset is not None else np.empty((0, 0, 3), dtype=np.float32),
        viewer_cloud_is_moving=torch.as_tensor(viewer_cloud_groups).detach().cpu().numpy() if viewer_cloud_groups is not None else np.empty((0,), dtype=np.int64),
    )


def make_track_viewer_data(track_subset, selected_ids, selected_labels, path_lengths, cloud_subset=None, cloud_groups=None, max_frames=160, max_tracks=2400, max_cloud_points=6000):
    tracks = torch.as_tensor(track_subset).detach().cpu()
    cloud = torch.as_tensor(cloud_subset).detach().cpu() if cloud_subset is not None else torch.empty(0, 0, 3)
    cloud_groups_tensor = torch.as_tensor(cloud_groups).detach().cpu().flatten() if cloud_groups is not None else torch.empty(0, dtype=torch.long)
    source_track_count = int(tracks.shape[1]) if tracks.ndim >= 2 else 0
    source_cloud_count = int(cloud.shape[1]) if cloud.ndim >= 2 else 0
    viewer_track_stride = 1
    viewer_cloud_stride = 1
    if tracks.numel() == 0 or tracks.shape[1] == 0:
        return {"frames": [], "cloud_frames": [], "cloud_groups": [], "gaussian_ids": [], "clusters": [], "path_lengths": [], "bounds": None, "source_track_count": 0, "source_cloud_count": source_cloud_count, "viewer_track_stride": 1, "viewer_cloud_stride": viewer_cloud_stride}

    if max_tracks > 0 and tracks.shape[1] > max_tracks:
        viewer_track_stride = int(math.ceil(tracks.shape[1] / max_tracks))
        track_idx = torch.arange(0, tracks.shape[1], viewer_track_stride)[:max_tracks]
        tracks = tracks[:, track_idx, :]
        selected_ids = [selected_ids[int(idx)] for idx in track_idx.tolist()]
        selected_labels = [selected_labels[int(idx)] for idx in track_idx.tolist()]
        path_lengths = [path_lengths[int(idx)] for idx in track_idx.tolist()]

    if max_cloud_points > 0 and cloud.numel() and cloud.shape[1] > max_cloud_points:
        viewer_cloud_stride = int(math.ceil(cloud.shape[1] / max_cloud_points))
        cloud_idx = torch.arange(0, cloud.shape[1], viewer_cloud_stride)[:max_cloud_points]
        cloud = cloud[:, cloud_idx, :]
        if cloud_groups_tensor.numel():
            cloud_groups_tensor = cloud_groups_tensor[cloud_idx]

    frame_count = tracks.shape[0]
    if frame_count > max_frames:
        frame_idx = torch.linspace(0, frame_count - 1, max_frames).long()
        tracks = tracks[frame_idx]
        if cloud.numel():
            cloud = cloud[frame_idx]

    values = tracks.numpy().astype(np.float32)
    cloud_values = cloud.numpy().astype(np.float32) if cloud.numel() else np.empty((0, 0, 3), dtype=np.float32)
    finite = np.isfinite(values).all(axis=-1)
    cloud_finite = np.isfinite(cloud_values).all(axis=-1) if cloud_values.size else np.empty((0, 0), dtype=bool)
    valid_parts = [values[finite]]
    if cloud_values.size:
        valid_parts.append(cloud_values[cloud_finite])
    valid = np.concatenate([part.reshape(-1, 3) for part in valid_parts if part.size], axis=0)
    if valid.size == 0:
        bounds = None
        center = np.zeros(3, dtype=np.float32)
        scale = 1.0
    else:
        mins = valid.min(axis=0)
        maxs = valid.max(axis=0)
        center = (mins + maxs) * 0.5
        scale = float(max(maxs - mins)) or 1.0
        bounds = {"min": [finite_float(v) for v in mins], "max": [finite_float(v) for v in maxs]}

    normalized = (values - center.reshape(1, 1, 3)) / scale
    normalized_cloud = (cloud_values - center.reshape(1, 1, 3)) / scale if cloud_values.size else cloud_values
    rounded = np.round(normalized, 5)
    rounded_cloud = np.round(normalized_cloud, 5) if normalized_cloud.size else normalized_cloud
    return {
        "frames": rounded.tolist(),
        "cloud_frames": rounded_cloud.tolist(),
        "cloud_groups": [int(v) for v in cloud_groups_tensor.tolist()] if cloud_groups_tensor.numel() else [],
        "gaussian_ids": [int(v) for v in selected_ids],
        "clusters": [int(v) for v in selected_labels],
        "path_lengths": [finite_float(v) for v in path_lengths],
        "bounds": bounds,
        "source_track_count": source_track_count,
        "source_cloud_count": source_cloud_count,
        "viewer_track_stride": viewer_track_stride,
        "viewer_cloud_stride": viewer_cloud_stride,
    }


def ensure_source_path(args):
    if args.source_path:
        return args

    cfg_path = Path(args.model_path) / "cfg_args"
    if not cfg_path.exists():
        raise ValueError("source_path was not provided and cfg_args was not found in the model directory")

    cfg_args = eval(cfg_path.read_text(), {"Namespace": Namespace})
    if not getattr(cfg_args, "source_path", None):
        raise ValueError("cfg_args exists, but it does not contain source_path")

    args.source_path = os.path.abspath(cfg_args.source_path)
    return args


def select_gaussians(gaussians, opacity_min, sample_gaussians):
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().flatten()
    selected = torch.nonzero(opacity >= opacity_min, as_tuple=False).flatten()
    if selected.numel() == 0:
        selected = torch.arange(xyz.shape[0], device=xyz.device)

    if sample_gaussians > 0 and selected.numel() > sample_gaussians:
        positions = torch.linspace(0, selected.numel() - 1, sample_gaussians, device=xyz.device).long()
        selected = selected[positions]

    return selected, xyz[selected], opacity[selected]


def deformed_positions(xyz, d_xyz):
    if torch.is_tensor(d_xyz) and d_xyz.ndim == 3:
        return from_homogenous(torch.bmm(d_xyz, to_homogenous(xyz).unsqueeze(-1)).squeeze(-1))
    return xyz + d_xyz


def sample_tracks(deform, selected_xyz, num_times):
    times = torch.linspace(0.0, 1.0, num_times, device=selected_xyz.device)
    tracks = []
    deform.deform.eval()
    with torch.no_grad():
        for fid in tqdm(times, desc="Sampling deformation field"):
            time_input = fid.reshape(1, 1).expand(selected_xyz.shape[0], -1)
            d_xyz, _, _ = deform.step(selected_xyz, time_input)
            tracks.append(deformed_positions(selected_xyz, d_xyz).detach().cpu())
    return torch.stack(tracks, dim=0), times.detach().cpu()


def compute_motion_metrics(tracks):
    eps = 1e-8
    num_times = tracks.shape[0]
    dt = 1.0 / max(num_times - 1, 1)
    deltas = tracks[1:] - tracks[:-1]
    step_dist = torch.linalg.norm(deltas, dim=-1)
    velocity = deltas / dt
    speed = torch.linalg.norm(velocity, dim=-1)
    acceleration = (velocity[1:] - velocity[:-1]) / dt if num_times >= 3 else torch.empty(0, tracks.shape[1], 3)
    accel_norm = torch.linalg.norm(acceleration, dim=-1) if acceleration.numel() else torch.empty(0, tracks.shape[1])
    jerk = (acceleration[1:] - acceleration[:-1]) / dt if num_times >= 4 else torch.empty(0, tracks.shape[1], 3)
    jerk_norm = torch.linalg.norm(jerk, dim=-1) if jerk.numel() else torch.empty(0, tracks.shape[1])

    displacement = torch.linalg.norm(tracks[-1] - tracks[0], dim=-1)
    path_length = step_dist.sum(dim=0)
    mean_speed = speed.mean(dim=0) if speed.numel() else torch.zeros(tracks.shape[1])
    peak_speed = speed.max(dim=0).values if speed.numel() else torch.zeros(tracks.shape[1])

    centered = tracks - tracks.mean(dim=0, keepdim=True)
    diameter = 2.0 * torch.linalg.norm(centered, dim=-1).max(dim=0).values
    mean_accel = accel_norm.mean(dim=0) if accel_norm.numel() else torch.zeros(tracks.shape[1])
    mean_jerk = jerk_norm.mean(dim=0) if jerk_norm.numel() else torch.zeros(tracks.shape[1])
    normalized_jerk = mean_jerk / (mean_speed + eps)

    return {
        "displacement": displacement,
        "path_length": path_length,
        "diameter": diameter,
        "mean_speed": mean_speed,
        "peak_speed": peak_speed,
        "mean_acceleration": mean_accel,
        "mean_jerk": mean_jerk,
        "normalized_jerk": normalized_jerk,
        "all_speed": speed.flatten(),
        "all_acceleration": accel_norm.flatten(),
    }


def compute_knn(xyz, k, chunk_size=1024):
    xyz = xyz.float()
    n = xyz.shape[0]
    if n <= 1 or k <= 0:
        return torch.empty(n, 0, dtype=torch.long), torch.empty(n, 0)
    k = min(k, n - 1)
    indices = []
    distances = []
    for start in tqdm(range(0, n, chunk_size), desc="Building canonical kNN"):
        end = min(start + chunk_size, n)
        dist = torch.cdist(xyz[start:end], xyz)
        values, idx = torch.topk(dist, k=k + 1, largest=False)
        indices.append(idx[:, 1:].cpu())
        distances.append(values[:, 1:].cpu())
    return torch.cat(indices, dim=0), torch.cat(distances, dim=0)


def compute_local_coherence(tracks, knn_idx, base_dist, chunk_size=1024):
    n = tracks.shape[1]
    if knn_idx.numel() == 0:
        empty = torch.empty(n)
        return empty, empty

    eps = 1e-8
    mean_stretch = []
    max_stretch = []
    for start in tqdm(range(0, n, chunk_size), desc="Computing local stretch"):
        end = min(start + chunk_size, n)
        idx = knn_idx[start:end]
        base = base_dist[start:end].clamp_min(eps)
        anchor = tracks[:, start:end, None, :]
        neighbor = tracks[:, idx, :]
        dist_t = torch.linalg.norm(anchor - neighbor, dim=-1)
        stretch = torch.abs(dist_t - base.unsqueeze(0)) / base.unsqueeze(0)
        mean_stretch.append(stretch.mean(dim=(0, 2)))
        max_stretch.append(stretch.amax(dim=(0, 2)))
    return torch.cat(mean_stretch), torch.cat(max_stretch)


def compute_motion_diversity(tracks, path_length, max_clusters=8):
    n = tracks.shape[1]
    if n == 0:
        return {"effective_pca_rank": None, "pca_explained_variance": [], "cluster_counts": []}, torch.empty(0)

    motion = tracks - tracks[0:1]
    features = motion.permute(1, 0, 2).reshape(n, -1).double()
    features = features / (path_length.double().reshape(-1, 1) + 1e-8)
    features = features - features.mean(dim=0, keepdim=True)

    cov = features.T.matmul(features) / max(n - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.flip(eigvals.clamp_min(0), dims=(0,))
    eigvecs = torch.flip(eigvecs, dims=(1,))
    total = eigvals.sum()
    explained = eigvals / total if total > 0 else eigvals
    positive = explained[explained > 0]
    effective_rank = torch.exp(-(positive * torch.log(positive)).sum()) if positive.numel() else torch.tensor(0.0)

    dims = min(6, eigvecs.shape[1], n)
    reduced = features.matmul(eigvecs[:, :dims]).float() if dims > 0 else torch.zeros(n, 1)
    cluster_count = min(max_clusters, n)
    labels = kmeans(reduced, cluster_count) if cluster_count > 1 else torch.zeros(n, dtype=torch.long)
    counts = torch.bincount(labels, minlength=cluster_count)

    return {
        "effective_pca_rank": finite_float(effective_rank),
        "pca_explained_variance": [finite_float(v) for v in explained[:20].tolist()],
        "cluster_counts": [int(v) for v in counts.tolist()],
        "active_clusters": int((counts >= max(5, int(0.01 * n))).sum()),
    }, labels


def kmeans(features, cluster_count, iterations=25):
    n = features.shape[0]
    init_idx = torch.linspace(0, n - 1, cluster_count).long()
    centers = features[init_idx].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iterations):
        dist = torch.cdist(features, centers)
        labels = dist.argmin(dim=1)
        for idx in range(cluster_count):
            mask = labels == idx
            if mask.any():
                centers[idx] = features[mask].mean(dim=0)
    return labels


def compute_motion_split(path_length, tracks, motion_threshold):
    if path_length.numel() == 0:
        empty_mask = torch.zeros_like(path_length, dtype=torch.bool)
        return {
            "threshold": None,
            "threshold_source": "empty",
            "static_fraction": None,
            "moving_fraction": None,
            "moving_bbox_min": None,
            "moving_bbox_max": None,
        }, empty_mask

    if motion_threshold is not None and motion_threshold >= 0:
        threshold = float(motion_threshold)
        threshold_source = "user"
    else:
        threshold = max(float(torch.quantile(path_length, 0.10)), float(path_length.median()) * 1.5, 1e-6)
        threshold_source = "adaptive"

    moving_mask = path_length > threshold
    static_fraction = 1.0 - float(moving_mask.float().mean())
    if moving_mask.any():
        dynamic_tracks = tracks[:, moving_mask, :].reshape(-1, 3)
        bbox_min = [finite_float(v) for v in dynamic_tracks.min(dim=0).values.tolist()]
        bbox_max = [finite_float(v) for v in dynamic_tracks.max(dim=0).values.tolist()]
    else:
        bbox_min = None
        bbox_max = None
    return {
        "threshold": finite_float(threshold),
        "threshold_source": threshold_source,
        "static_fraction": finite_float(static_fraction),
        "moving_fraction": finite_float(1.0 - static_fraction),
        "moving_bbox_min": bbox_min,
        "moving_bbox_max": bbox_max,
    }, moving_mask


def project_points(points, camera):
    device = camera.full_proj_transform.device
    points = points.to(device)
    projected = geom_transform_points(points, camera.full_proj_transform)
    x = (projected[:, 0] + 1.0) * 0.5 * camera.image_width
    # The dataset images loaded for trajectory overlays use top-left image
    # coordinates after the repository's camera transform, so applying the
    # usual NDC Y flip mirrors tracks vertically over the video frame.
    y = (projected[:, 1] + 1.0) * 0.5 * camera.image_height
    return torch.stack((x, y), dim=-1).detach().cpu()


def compute_projection_summary(tracks, cameras, max_cameras=4):
    selected_cameras = (cameras.getTestCameras() + cameras.getTrainCameras())[:max_cameras]
    summaries = []
    with torch.no_grad():
        for camera in selected_cameras:
            first = project_points(tracks[0], camera)
            last = project_points(tracks[-1], camera)
            displacement = torch.linalg.norm(last - first, dim=-1)
            summaries.append({
                "camera": camera.image_name,
                "width": int(camera.image_width),
                "height": int(camera.image_height),
                "screen_displacement_pixels": summarize(displacement),
            })
    return summaries


def write_summary_csv(path, gaussian_ids, opacities, labels, moving_mask, metrics, mean_stretch, max_stretch):
    fieldnames = [
        "gaussian_id", "motion_group", "opacity", "cluster", "displacement", "path_length", "diameter",
        "mean_speed", "peak_speed", "mean_acceleration", "mean_jerk", "normalized_jerk",
        "mean_local_stretch", "max_local_stretch",
    ]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, gaussian_id in enumerate(gaussian_ids):
            writer.writerow({
                "gaussian_id": int(gaussian_id),
                "motion_group": "moving" if bool(moving_mask[idx]) else "static",
                "opacity": finite_float(opacities[idx]),
                "cluster": int(labels[idx]) if labels.numel() else -1,
                "displacement": finite_float(metrics["displacement"][idx]),
                "path_length": finite_float(metrics["path_length"][idx]),
                "diameter": finite_float(metrics["diameter"][idx]),
                "mean_speed": finite_float(metrics["mean_speed"][idx]),
                "peak_speed": finite_float(metrics["peak_speed"][idx]),
                "mean_acceleration": finite_float(metrics["mean_acceleration"][idx]),
                "mean_jerk": finite_float(metrics["mean_jerk"][idx]),
                "normalized_jerk": finite_float(metrics["normalized_jerk"][idx]),
                "mean_local_stretch": finite_float(mean_stretch[idx]) if mean_stretch.numel() else None,
                "max_local_stretch": finite_float(max_stretch[idx]) if max_stretch.numel() else None,
            })


def write_group_summary_csv(path, group_stats):
    fieldnames = [
        "motion_group", "count", "fraction", "path_length_mean", "path_length_median",
        "path_length_p90", "displacement_mean", "displacement_median", "mean_speed_mean",
        "peak_speed_p90", "mean_acceleration_mean", "mean_acceleration_median",
        "normalized_jerk_mean", "normalized_jerk_median", "mean_local_stretch_mean",
        "mean_local_stretch_median", "mean_local_stretch_p90",
    ]

    def get(stats, *keys):
        value = stats
        for key in keys:
            value = value.get(key, {})
        return value if value != {} else None

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for name in ("static", "moving"):
            stats = group_stats[name]
            writer.writerow({
                "motion_group": name,
                "count": stats["count"],
                "fraction": stats["fraction"],
                "path_length_mean": get(stats, "motion", "path_length", "mean"),
                "path_length_median": get(stats, "motion", "path_length", "median"),
                "path_length_p90": get(stats, "motion", "path_length", "p90"),
                "displacement_mean": get(stats, "motion", "displacement", "mean"),
                "displacement_median": get(stats, "motion", "displacement", "median"),
                "mean_speed_mean": get(stats, "motion", "mean_speed", "mean"),
                "peak_speed_p90": get(stats, "motion", "peak_speed", "p90"),
                "mean_acceleration_mean": get(stats, "smoothness", "mean_acceleration", "mean"),
                "mean_acceleration_median": get(stats, "smoothness", "mean_acceleration", "median"),
                "normalized_jerk_mean": get(stats, "smoothness", "normalized_jerk", "mean"),
                "normalized_jerk_median": get(stats, "smoothness", "normalized_jerk", "median"),
                "mean_local_stretch_mean": get(stats, "local_coherence", "mean_local_stretch", "mean"),
                "mean_local_stretch_median": get(stats, "local_coherence", "mean_local_stretch", "median"),
                "mean_local_stretch_p90": get(stats, "local_coherence", "mean_local_stretch", "p90"),
            })


def fmt_metric(value, precision=4):
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
        return f"{value:.{precision}g}"
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def pct(value):
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def stat(stats, *keys):
    value = stats
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def relpath(path, base_dir):
    if not path:
        return ""
    return html.escape(os.path.relpath(path, base_dir).replace(os.sep, "/"))


def compute_narrative(data):
    groups = data["metrics"]["motion_group_stats"]
    diversity = data["metrics"]["motion_diversity"]
    coherence = data["metrics"]["local_coherence"]
    moving = groups["moving"]
    static = groups["static"]

    moving_fraction = moving["fraction"] or 0.0
    moving_median_path = stat(moving, "motion", "path_length", "median") or 0.0
    static_median_path = stat(static, "motion", "path_length", "median") or 0.0
    moving_p90_stretch = stat(moving, "local_coherence", "mean_local_stretch", "p90") or 0.0
    global_stretch_p99 = stat(coherence, "mean_local_stretch", "p99") or 0.0
    pca_first = (diversity.get("pca_explained_variance") or [0.0])[0] or 0.0
    effective_rank = diversity.get("effective_pca_rank") or 0.0

    if moving_median_path > max(static_median_path * 50.0, 0.05) and pca_first >= 0.6 and moving_p90_stretch < 10.0:
        verdict = "Good diagnostic result with coherent moving foreground motion."
    elif moving_median_path > max(static_median_path * 10.0, 0.02):
        verdict = "Mixed but useful result: motion separation is clear, with quality caveats."
    else:
        verdict = "Concerning result: moving/static separation is weak or motion is unstable."

    takeaways = [
        f"Moving Gaussians are {pct(moving_fraction)} of the sample, with median path length {fmt_metric(moving_median_path)} versus {fmt_metric(static_median_path)} for static Gaussians.",
        f"The first PCA component explains {pct(pca_first)} of normalized trajectory variation; effective rank is {fmt_metric(effective_rank)}, indicating {'low-dimensional coherent motion' if pca_first >= 0.6 else 'more distributed motion'}.",
        f"Typical moving-neighborhood stretch has P90 {fmt_metric(moving_p90_stretch)}, while global P99 stretch is {fmt_metric(global_stretch_p99)}, so outliers should be discussed separately from typical behavior.",
        f"The motion split threshold is {fmt_metric(data['metrics']['motion_split']['threshold'])} path length ({data['metrics']['motion_split']['threshold_source']}).",
    ]
    if data["metrics"].get("track_visualizations", {}).get("videos"):
        takeaways.append("Projected moving-Gaussian track videos are included as visual evidence for trajectory direction and coherence.")
    else:
        takeaways.append("No overlay video was produced; use the 3D trajectory plots and exported selected tracks for visual evidence.")

    return {"verdict": verdict, "takeaways": takeaways}


def table_rows(rows):
    return "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )


def metric_card(title, value, caption):
    return (
        '<div class="metric-card">'
        f"<div class=\"metric-title\">{html.escape(title)}</div>"
        f"<div class=\"metric-value\">{html.escape(value)}</div>"
        f"<div class=\"metric-caption\">{html.escape(caption)}</div>"
        "</div>"
    )


def image_panel(title, src, caption):
    return (
        '<figure class="panel">'
        f"<h3>{html.escape(title)}</h3>"
        f"<img src=\"{src}\" alt=\"{html.escape(title)}\">"
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def video_panel(title, src, caption):
    return (
        '<figure class="panel">'
        f"<h3>{html.escape(title)}</h3>"
        f"<video src=\"{src}\" controls muted preload=\"metadata\"></video>"
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def write_dashboard(path, data, artifact_paths):
    base_dir = path.parent
    narrative = data["narrative"]
    metrics = data["metrics"]
    groups = metrics["motion_group_stats"]
    moving = groups["moving"]
    static = groups["static"]
    diversity = metrics["motion_diversity"]

    summary_cards = [
        metric_card("Sampled Gaussians", f"{data['sampled_gaussians']} / {data['total_gaussians']}", "Opacity-filtered sample used for trajectory diagnostics."),
        metric_card("Moving Fraction", pct(metrics["motion_split"]["moving_fraction"]), "Path-length based diagnostic group, not semantic labels."),
        metric_card("Moving Median Path", fmt_metric(stat(moving, "motion", "path_length", "median")), "Typical trajectory length among moving Gaussians."),
        metric_card("PCA First Component", pct((diversity.get("pca_explained_variance") or [None])[0]), "Share of normalized motion explained by the dominant mode."),
    ]

    comparison_rows = []
    for name, stats in (("Global", None), ("Static", static), ("Moving", moving)):
        source = metrics if stats is None else stats
        prefix = ("motion",) if stats is None else ("motion",)
        smooth_prefix = ("smoothness",) if stats is None else ("smoothness",)
        local_prefix = ("local_coherence",) if stats is None else ("local_coherence",)
        comparison_rows.append([
            name,
            data["sampled_gaussians"] if stats is None else stats["count"],
            "100.0%" if stats is None else pct(stats["fraction"]),
            fmt_metric(stat(source, *prefix, "path_length", "median")),
            fmt_metric(stat(source, *prefix, "path_length", "p90")),
            fmt_metric(stat(source, *smooth_prefix, "mean_acceleration", "median")),
            fmt_metric(stat(source, *local_prefix, "mean_local_stretch", "median")),
            fmt_metric(stat(source, *local_prefix, "mean_local_stretch", "p99")),
        ])

    cluster_rows = [
        [entry["cluster"], entry["count"]]
        for entry in moving.get("cluster_counts", [])
    ] or [["n/a", "No moving clusters"]]

    jitter_rows = [
        [item["rank"], item["gaussian_id"], fmt_metric(item["value"])]
        for item in stat(moving, "smoothness", "normalized_jerk_outliers") or []
    ]
    stretch_rows = [
        [item["rank"], item["gaussian_id"], fmt_metric(item["value"])]
        for item in stat(moving, "local_coherence", "local_stretch_outliers") or []
    ]

    plot_paths = artifact_paths.get("plots", {})
    videos = artifact_paths.get("videos", [])
    track_viewer_data = artifact_paths.get("track_viewer_data", {"frames": []})
    track_viewer_json = json.dumps(track_viewer_data, separators=(",", ":"))
    plot_panels = [
        image_panel("Static path length", relpath(plot_paths.get("static_path_length"), base_dir), "Near-static trajectories should concentrate close to zero."),
        image_panel("Moving path length", relpath(plot_paths.get("moving_path_length"), base_dir), "Moving trajectories are evaluated against other moving trajectories."),
        image_panel("Moving acceleration", relpath(plot_paths.get("moving_acceleration"), base_dir), "Shows typical and tail temporal roughness for moving Gaussians."),
        image_panel("Moving local stretch", relpath(plot_paths.get("moving_local_stretch"), base_dir), "Neighborhood coherence among moving Gaussians."),
        image_panel("PCA explained variance", relpath(plot_paths.get("pca_explained_variance"), base_dir), "Ordered spectrum of normalized trajectory variation."),
        image_panel("Moving paths XY", relpath(plot_paths.get("moving_paths_xy"), base_dir), "Orthographic trajectory projection; colors indicate clusters."),
        image_panel("Moving paths XZ", relpath(plot_paths.get("moving_paths_xz"), base_dir), "Side-view trajectory projection."),
        image_panel("Moving paths YZ", relpath(plot_paths.get("moving_paths_yz"), base_dir), "Alternative side-view trajectory projection."),
    ]
    video_panels = [
        video_panel(
            f"Track overlay: {video['camera']}",
            relpath(video["path"], base_dir),
            f"Projected trails for {video['track_count']} selected moving Gaussians; colors indicate motion clusters.",
        )
        for video in videos
    ]

    css = """
    :root { color-scheme: light; --ink:#202124; --muted:#5f6368; --line:#d8dde6; --panel:#ffffff; --soft:#f5f7fb; --accent:#315f9f; --good:#1b7f4b; --warn:#9a6700; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, Helvetica, sans-serif; color:var(--ink); background:#f0f3f8; line-height:1.45; }
    header { background:#fff; border-bottom:1px solid var(--line); padding:32px 40px 24px; }
    main { max-width:1240px; margin:0 auto; padding:28px 28px 48px; }
    h1 { margin:0 0 8px; font-size:32px; letter-spacing:0; }
    h2 { margin:32px 0 14px; font-size:22px; border-bottom:1px solid var(--line); padding-bottom:8px; }
    h3 { margin:0 0 10px; font-size:16px; }
    p { margin:8px 0; }
    .verdict { margin-top:14px; padding:14px 16px; border-left:5px solid var(--accent); background:#eef4ff; font-size:18px; font-weight:700; }
    .meta, .caption, figcaption { color:var(--muted); font-size:13px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:14px; }
    .metric-card, .section-card, .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin:0; }
    .metric-title { color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.04em; }
    .metric-value { font-size:28px; font-weight:700; margin:6px 0; }
    .metric-caption { color:var(--muted); font-size:13px; }
    .takeaways { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px 20px; }
    .takeaways li { margin:8px 0; }
    table { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
    th { background:var(--soft); font-size:13px; color:#30343b; }
    td { font-size:14px; }
    img, video { width:100%; display:block; border:1px solid var(--line); background:#fff; }
    .plot-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(360px, 1fr)); gap:16px; }
    .viewer-shell { background:#111827; border:1px solid #273244; border-radius:8px; overflow:hidden; }
    .viewer-toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:12px; background:#182235; color:#eef2f7; }
    .viewer-toolbar button { appearance:none; border:1px solid #52627a; background:#24324a; color:#eef2f7; border-radius:6px; padding:7px 10px; cursor:pointer; }
    .viewer-toolbar button:hover { background:#31425f; }
    .viewer-toolbar input[type="range"] { flex:1; min-width:180px; }
    .viewer-toolbar label { color:#c9d4e5; font-size:13px; }
    #trackViewer, #pathViewer { width:100%; height:620px; display:block; background:#0b1020; touch-action:none; }
    code { background:#eef1f6; padding:2px 5px; border-radius:4px; }
    footer { color:var(--muted); font-size:13px; margin-top:32px; }
    @media (max-width: 720px) { header { padding:24px 20px; } main { padding:20px 14px 40px; } .plot-grid { grid-template-columns:1fr; } }
    """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trajectory Analysis Dashboard</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>Trajectory Analysis Dashboard</h1>
    <div class="meta">Model: <code>{html.escape(str(data['model_path']))}</code> - Iteration {data['loaded_iteration']} - {data['num_times']} time samples</div>
    <div class="verdict">{html.escape(narrative['verdict'])}</div>
  </header>
  <main>
    <section>
      <h2>Executive Summary</h2>
      <div class="grid">{''.join(summary_cards)}</div>
      <ul class="takeaways">{''.join(f"<li>{html.escape(item)}</li>" for item in narrative['takeaways'])}</ul>
    </section>

    <section>
      <h2>Static vs Moving Split</h2>
      <p>The split uses path length so static Gaussians are compared with static Gaussians and moving Gaussians are compared with moving Gaussians. This avoids compressing the moving distribution with near-zero background tracks.</p>
      <table>
        <thead><tr><th>Group</th><th>Count</th><th>Fraction</th><th>Median path</th><th>P90 path</th><th>Median accel.</th><th>Median stretch</th><th>P99 stretch</th></tr></thead>
        <tbody>{table_rows(comparison_rows)}</tbody>
      </table>
    </section>

    <section>
      <h2>Motion Quality</h2>
      <p>Path length and displacement measure how much each Gaussian travels through the learned deformation field. The moving subset is the right population for judging performer motion.</p>
      <div class="plot-grid">{''.join(plot_panels[:2])}</div>
    </section>

    <section>
      <h2>Temporal Smoothness</h2>
      <p>Acceleration and normalized jerk identify abrupt frame-to-frame changes. High values are most useful when interpreted as outlier diagnostics rather than average quality alone.</p>
      <div class="plot-grid">{plot_panels[2]}</div>
      <h3>Top moving normalized-jerk outliers</h3>
      <table><thead><tr><th>Rank</th><th>Gaussian ID</th><th>Normalized jerk</th></tr></thead><tbody>{table_rows(jitter_rows)}</tbody></table>
    </section>

    <section>
      <h2>Local Coherence</h2>
      <p>Local stretch compares distances between nearby canonical Gaussians over time. Low typical stretch means neighborhoods move coherently; extreme outliers can indicate tearing, expansion, or near-degenerate neighbor distances.</p>
      <div class="plot-grid">{plot_panels[3]}</div>
      <h3>Top moving local-stretch outliers</h3>
      <table><thead><tr><th>Rank</th><th>Gaussian ID</th><th>Mean local stretch</th></tr></thead><tbody>{table_rows(stretch_rows)}</tbody></table>
    </section>

    <section>
      <h2>Motion Diversity</h2>
      <p>PCA is computed on normalized trajectories. A concentrated spectrum indicates that learned motion is dominated by a small number of coherent modes.</p>
      <div class="plot-grid">{plot_panels[4]}</div>
      <h3>Moving cluster counts</h3>
      <table><thead><tr><th>Cluster</th><th>Moving Gaussians</th></tr></thead><tbody>{table_rows(cluster_rows)}</tbody></table>
    </section>

    <section>
      <h2>Trajectory Visualizations</h2>
      <p>The 3D viewer shows an animated downsampled Gaussian point cloud plus selected moving-Gaussian trails before camera projection. This is the best view for checking whether the learned dancer-shaped Gaussian motion and the path geometry agree. Drag to rotate; scroll to zoom.</p>
      <div class="viewer-shell">
        <canvas id="trackViewer" width="1200" height="620"></canvas>
        <div class="viewer-toolbar">
          <button id="viewerPlay" type="button">Pause</button>
          <button data-view="front" type="button">Front</button>
          <button data-view="side" type="button">Side</button>
          <button data-view="top" type="button">Top</button>
          <label for="viewerFrame">Frame</label>
          <input id="viewerFrame" type="range" min="0" max="0" value="0">
          <span id="viewerReadout">0 / 0</span>
        </div>
      </div>
      <p class="caption">Gray/cyan points are the animated Gaussian point cloud. Colored trails are selected moving Gaussians; colors indicate motion clusters.</p>
      <h3>Interactive moving paths</h3>
      <p class="caption">This 3D path graph draws dashboard-sampled moving-Gaussian trajectories with adaptive decimation for browser responsiveness. Full selected tracks are still exported in <code>track_samples.npz</code>. Drag to rotate; scroll to zoom.</p>
      <div class="viewer-shell">
        <canvas id="pathViewer" width="1200" height="620"></canvas>
        <div class="viewer-toolbar">
          <button data-path-view="front" type="button">Front</button>
          <button data-path-view="side" type="button">Side</button>
          <button data-path-view="top" type="button">Top</button>
          <span id="pathReadout">0 tracks</span>
        </div>
      </div>
      <div class="plot-grid">{''.join(video_panels) if video_panels else '<div class="section-card">No MP4 overlay videos were generated. Check whether imageio is installed or use the 3D trajectory plots below.</div>'}</div>
      <div class="plot-grid">{''.join(plot_panels[5:])}</div>
    </section>

    <section>
      <h2>Methods and Caveats</h2>
      <div class="section-card">
        <p>Trajectories are generated by sampling canonical Gaussians through the trained deformation field over normalized time. Metrics are no-ground-truth diagnostics.</p>
        <p>No APD, OA, AJ, or 3D-AJ accuracy is reported because this scene does not include annotated 3D point tracks or visibility labels.</p>
        <p>The static/moving split is based on path length and is not semantic segmentation. Local-stretch outliers should be interpreted alongside typical median/P90 values.</p>
        <p>Generated files include <code>trajectory_metrics.json</code>, <code>trajectory_summary.csv</code>, <code>trajectory_group_summary.csv</code>, <code>track_samples.npz</code>, plots, and videos.</p>
      </div>
    </section>

    <footer>Dashboard generated by <code>trajectory_metrics.py</code>. All paths are relative to this output directory.</footer>
  </main>
  <script>
  const TRACK_VIEWER_DATA = {track_viewer_json};
  (() => {{
    const canvas = document.getElementById('trackViewer');
    const frameSlider = document.getElementById('viewerFrame');
    const readout = document.getElementById('viewerReadout');
    const playButton = document.getElementById('viewerPlay');
    if (!canvas || !TRACK_VIEWER_DATA.frames || TRACK_VIEWER_DATA.frames.length === 0) {{
      if (canvas) {{
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0b1020';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#dbe4f0';
        ctx.font = '20px Arial';
        ctx.fillText('No 3D tracks available in this run.', 32, 48);
      }}
      return;
    }}

    const frames = TRACK_VIEWER_DATA.frames;
    const cloudFrames = TRACK_VIEWER_DATA.cloud_frames || [];
    const pointFrames = cloudFrames.length ? cloudFrames : frames;
    const sourceCloudCount = TRACK_VIEWER_DATA.source_cloud_count || (pointFrames[0] ? pointFrames[0].length : 0);
    const gl = canvas.getContext('webgl2', {{ antialias: true }}) || canvas.getContext('webgl', {{ antialias: true }});
    if (!gl) {{
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#0b1020';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#dbe4f0';
      ctx.font = '20px Arial';
      ctx.fillText('WebGL is unavailable; cannot render all moving points efficiently.', 32, 48);
      return;
    }}

    function compileShader(type, source) {{
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
        throw new Error(gl.getShaderInfoLog(shader) || 'shader compile failed');
      }}
      return shader;
    }}

    const vertexSource = `
      attribute vec3 aPosition;
      uniform float uYaw;
      uniform float uPitch;
      uniform float uZoom;
      uniform float uAspect;
      uniform float uPointSize;
      void main() {{
        float cy = cos(uYaw);
        float sy = sin(uYaw);
        float cp = cos(uPitch);
        float sp = sin(uPitch);
        float x1 = cy * aPosition.x + sy * aPosition.z;
        float z1 = -sy * aPosition.x + cy * aPosition.z;
        float y1 = cp * aPosition.y - sp * z1;
        float z2 = sp * aPosition.y + cp * z1;
        float scale = 1.35 * uZoom / max(0.25, 1.8 + z2);
        gl_Position = vec4(x1 * scale * uAspect, y1 * scale, 0.0, 1.0);
        gl_PointSize = uPointSize;
      }}
    `;
    const fragmentSource = `
      precision mediump float;
      uniform vec4 uColor;
      void main() {{
        vec2 uv = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(uv, uv);
        if (r2 > 1.0) discard;
        gl_FragColor = uColor;
      }}
    `;
    const program = gl.createProgram();
    gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
      throw new Error(gl.getProgramInfoLog(program) || 'program link failed');
    }}
    gl.useProgram(program);

    const positionLocation = gl.getAttribLocation(program, 'aPosition');
    const yawLocation = gl.getUniformLocation(program, 'uYaw');
    const pitchLocation = gl.getUniformLocation(program, 'uPitch');
    const zoomLocation = gl.getUniformLocation(program, 'uZoom');
    const aspectLocation = gl.getUniformLocation(program, 'uAspect');
    const pointSizeLocation = gl.getUniformLocation(program, 'uPointSize');
    const colorLocation = gl.getUniformLocation(program, 'uColor');
    const pointBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);

    function flattenPoints(points) {{
      const out = new Float32Array(points.length * 3);
      for (let i = 0; i < points.length; i++) {{
        out[i * 3] = points[i][0];
        out[i * 3 + 1] = points[i][1];
        out[i * 3 + 2] = points[i][2];
      }}
      return out;
    }}

    const gpuFrames = pointFrames.map(flattenPoints);
    let frame = 0;
    let yaw = -0.75;
    let pitch = 0.35;
    let zoom = 1.15;
    let pointSize = 4.0;
    let playing = true;
    let lastTick = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    frameSlider.max = String(pointFrames.length - 1);
    frameSlider.value = '0';

    function resizeCanvas() {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(640, Math.floor(rect.width * dpr));
      canvas.height = Math.max(360, Math.floor(rect.height * dpr));
      gl.viewport(0, 0, canvas.width, canvas.height);
      draw();
    }}

    function draw() {{
      const points = gpuFrames[frame] || new Float32Array();
      gl.clearColor(0.03, 0.06, 0.12, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, points, gl.DYNAMIC_DRAW);
      gl.uniform1f(yawLocation, yaw);
      gl.uniform1f(pitchLocation, pitch);
      gl.uniform1f(zoomLocation, zoom);
      gl.uniform1f(aspectLocation, canvas.height / Math.max(canvas.width, 1));
      gl.uniform1f(pointSizeLocation, pointSize * (window.devicePixelRatio || 1));
      gl.uniform4f(colorLocation, 0.40, 0.91, 0.98, 0.86);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.POINTS, 0, points.length / 3);
      readout.textContent = `${{frame + 1}} / ${{pointFrames.length}} | WebGL points ${{points.length / 3}}/${{sourceCloudCount}}`;
      frameSlider.value = String(frame);
    }}

    function tick(ts) {{
      if (playing && ts - lastTick > 65) {{
        frame = (frame + 1) % pointFrames.length;
        lastTick = ts;
        draw();
      }}
      requestAnimationFrame(tick);
    }}

    canvas.addEventListener('pointerdown', (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener('pointermove', (event) => {{
      if (!dragging) return;
      yaw += (event.clientX - lastX) * 0.008;
      pitch += (event.clientY - lastY) * 0.008;
      pitch = Math.max(-1.35, Math.min(1.35, pitch));
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    }});
    canvas.addEventListener('pointerup', () => {{ dragging = false; }});
    canvas.addEventListener('wheel', (event) => {{
      event.preventDefault();
      if (event.shiftKey) {{
        pointSize *= event.deltaY > 0 ? 0.88 : 1.12;
        pointSize = Math.max(1.0, Math.min(18.0, pointSize));
      }} else {{
        zoom *= event.deltaY > 0 ? 0.9 : 1.12;
        zoom = Math.max(0.2, Math.min(40.0, zoom));
      }}
      draw();
    }}, {{ passive: false }});
    frameSlider.addEventListener('input', () => {{
      frame = Number(frameSlider.value);
      playing = false;
      playButton.textContent = 'Play';
      draw();
    }});
    playButton.addEventListener('click', () => {{
      playing = !playing;
      playButton.textContent = playing ? 'Pause' : 'Play';
    }});
    document.querySelectorAll('[data-view]').forEach((button) => {{
      button.addEventListener('click', () => {{
        const view = button.getAttribute('data-view');
        if (view === 'front') {{ yaw = 0; pitch = 0; }}
        if (view === 'side') {{ yaw = Math.PI / 2; pitch = 0; }}
        if (view === 'top') {{ yaw = 0; pitch = Math.PI / 2 - 0.02; }}
        draw();
      }});
    }});
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    requestAnimationFrame(tick);
  }})();
  (() => {{
    const canvas = document.getElementById('pathViewer');
    const readout = document.getElementById('pathReadout');
    if (!canvas || !TRACK_VIEWER_DATA.frames || TRACK_VIEWER_DATA.frames.length === 0) {{
      if (canvas) {{
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0b1020';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#dbe4f0';
        ctx.font = '20px Arial';
        ctx.fillText('No moving paths available in this run.', 32, 48);
      }}
      return;
    }}

    const ctx = canvas.getContext('2d');
    const frames = TRACK_VIEWER_DATA.frames;
    const clusters = TRACK_VIEWER_DATA.clusters || [];
    const palette = ['#e63946', '#1d3557', '#2a9d8f', '#f4a261', '#8338ec', '#ffb703', '#118ab2', '#06d6a0'];
    const trackCount = frames[0] ? frames[0].length : 0;
    const sourceTrackCount = TRACK_VIEWER_DATA.source_track_count || trackCount;
    const maxPathTracks = 1400;
    const maxPathFrames = 90;
    const trackStep = Math.max(1, Math.ceil(trackCount / maxPathTracks));
    const frameStep = Math.max(1, Math.ceil(frames.length / maxPathFrames));
    let yaw = -0.75;
    let pitch = 0.35;
    let zoom = 1.35;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function resizeCanvas() {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(640, Math.floor(rect.width * dpr));
      canvas.height = Math.max(360, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}

    function project(point) {{
      const x = point[0], y = point[1], z = point[2];
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * z;
      const z1 = -sy * x + cy * z;
      const y1 = cp * y - sp * z1;
      const z2 = sp * y + cp * z1;
      const scale = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.72 * zoom / Math.max(0.25, 1.6 + z2);
      return [canvas.clientWidth * 0.5 + x1 * scale, canvas.clientHeight * 0.53 - y1 * scale, z2];
    }}

    function drawAxes() {{
      const axes = [
        [['X', '#ef4444'], [0.55, 0, 0]],
        [['Y', '#22c55e'], [0, 0.55, 0]],
        [['Z', '#38bdf8'], [0, 0, 0.55]],
      ];
      const origin = project([0, 0, 0]);
      ctx.lineWidth = 2;
      axes.forEach(([meta, end]) => {{
        const projected = project(end);
        ctx.strokeStyle = meta[1];
        ctx.beginPath();
        ctx.moveTo(origin[0], origin[1]);
        ctx.lineTo(projected[0], projected[1]);
        ctx.stroke();
        ctx.fillStyle = meta[1];
        ctx.fillText(meta[0], projected[0] + 6, projected[1] - 6);
      }});
    }}

    function draw() {{
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      ctx.clearRect(0, 0, width, height);
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, '#07111f');
      gradient.addColorStop(1, '#101827');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      ctx.font = '13px Arial';
      drawAxes();

      let drawnTracks = 0;
      let drawnSegments = 0;
      ctx.lineWidth = 1.6;
      for (let i = 0; i < trackCount; i += trackStep) {{
        const color = palette[Math.abs(clusters[i] || 0) % palette.length];
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.52;
        ctx.beginPath();
        let started = false;
        for (let t = 0; t < frames.length; t += frameStep) {{
          const p = project(frames[t][i]);
          if (!started) {{
            ctx.moveTo(p[0], p[1]);
            started = true;
          }} else {{
            ctx.lineTo(p[0], p[1]);
            drawnSegments += 1;
          }}
        }}
        ctx.stroke();
        const end = project(frames[frames.length - 1][i]);
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(end[0], end[1], 2.6, 0, Math.PI * 2);
        ctx.fill();
        drawnTracks += 1;
      }}
      ctx.globalAlpha = 1;
      if (readout) {{
        readout.textContent = 'drawn tracks ' + drawnTracks + '/' + sourceTrackCount + ' | frame step ' + frameStep + ' | segments ' + drawnSegments;
      }}
    }}

    canvas.addEventListener('pointerdown', (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener('pointermove', (event) => {{
      if (!dragging) return;
      yaw += (event.clientX - lastX) * 0.008;
      pitch += (event.clientY - lastY) * 0.008;
      pitch = Math.max(-1.35, Math.min(1.35, pitch));
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    }});
    canvas.addEventListener('pointerup', () => {{ dragging = false; }});
    canvas.addEventListener('wheel', (event) => {{
      event.preventDefault();
      zoom *= event.deltaY > 0 ? 0.9 : 1.12;
      zoom = Math.max(0.2, Math.min(32.0, zoom));
      draw();
    }}, {{ passive: false }});
    document.querySelectorAll('[data-path-view]').forEach((button) => {{
      button.addEventListener('click', () => {{
        const view = button.getAttribute('data-path-view');
        if (view === 'front') {{ yaw = 0; pitch = 0; }}
        if (view === 'side') {{ yaw = Math.PI / 2; pitch = 0; }}
        if (view === 'top') {{ yaw = 0; pitch = Math.PI / 2 - 0.02; }}
        draw();
      }});
    }});
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  }})();
  </script>
</body>
</html>
"""
    path.write_text(html_doc)


def write_report(path, data):
    motion = data["metrics"]["motion"]
    smoothness = data["metrics"]["smoothness"]
    coherence = data["metrics"]["local_coherence"]
    motion_split = data["metrics"]["motion_split"]
    group_stats = data["metrics"]["motion_group_stats"]
    diversity = data["metrics"]["motion_diversity"]
    narrative = data.get("narrative", compute_narrative(data))

    def fmt(value, precision=6):
        if value is None:
            return "n/a"
        return f"{value:.{precision}g}"

    def fmt_fixed(value, precision=3):
        if value is None:
            return "n/a"
        return f"{value:.{precision}f}"

    lines = [
        "# Bonus Trajectory Metrics Report",
        "",
        "This report analyzes learned Gaussian trajectories by sampling the trained deformation field over normalized time. It is a no-ground-truth diagnostic: APD, OA, AJ, and 3D-AJ are not reported as accuracy metrics because this PKU scene does not include annotated 3D tracks or visibility labels.",
        "",
        "## Executive Summary",
        "",
        f"**Verdict:** {narrative['verdict']}",
        "",
    ]
    lines.extend([f"- {takeaway}" for takeaway in narrative["takeaways"]])
    lines.extend([
        "",
        "## Key Metrics",
        "",
        f"- Sampled Gaussians: {data['sampled_gaussians']} of {data['total_gaussians']} total.",
        f"- Time samples: {data['num_times']}.",
        f"- Mean path length: {fmt(motion['path_length']['mean'])}.",
        f"- Median path length: {fmt(motion['path_length']['median'])}.",
        f"- P90 path length: {fmt(motion['path_length']['p90'])}.",
        f"- Mean speed: {fmt(motion['mean_speed']['mean'])}.",
        f"- Mean acceleration: {fmt(smoothness['mean_acceleration']['mean'])}.",
        f"- Mean normalized jerk: {fmt(smoothness['normalized_jerk']['mean'])}.",
        f"- Mean local stretch: {fmt(coherence['mean_local_stretch']['mean'])}.",
        f"- P90 local stretch: {fmt(coherence['mean_local_stretch']['p90'])}.",
        f"- Motion split threshold: {fmt(motion_split['threshold'])} ({motion_split['threshold_source']}).",
        f"- Static fraction: {fmt_fixed(motion_split['static_fraction'])}.",
        f"- Moving fraction: {fmt_fixed(motion_split['moving_fraction'])}.",
        f"- Effective PCA rank: {fmt_fixed(diversity['effective_pca_rank'])}.",
        f"- Active motion clusters: {diversity['active_clusters']}.",
        "",
        "Primary dashboard artifact: `trajectory_dashboard.html`.",
        "",
        "## Static vs Moving Gaussians",
        "",
        "| Group | Count | Fraction | Median path | P90 path | Median acceleration | Median local stretch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name in ("static", "moving"):
        stats = group_stats[name]
        lines.append(
            f"| {name} | {stats['count']} | {fmt_fixed(stats['fraction'])} | "
            f"{fmt(stats['motion']['path_length']['median'])} | {fmt(stats['motion']['path_length']['p90'])} | "
            f"{fmt(stats['smoothness']['mean_acceleration']['median'])} | "
            f"{fmt(stats['local_coherence']['mean_local_stretch']['median'])} |"
        )
    lines.extend([
        "",
        "## Assessment",
        "",
        "The static and moving groups are summarized separately so that near-static background Gaussians do not compress the moving trajectory distribution. Large path length and diameter within the moving group identify Gaussians carrying the visible dancer motion. High normalized jerk marks temporally rough tracks and is useful for finding deformation-field jitter. Local stretch measures whether nearby canonical Gaussians remain coherent over time; large values indicate local tearing, excessive expansion, or inconsistent motion among neighbors.",
        "",
        "The static/moving split is a motion-magnitude diagnostic, not a semantic segmentation. A healthy dynamic reconstruction should contain a static component for background and a concentrated moving component for the performer.",
        "",
        "## Papers Used",
        "",
    ])
    for name, citation in PAPER_CITATIONS.items():
        lines.append(f"- {name}: {citation['citation']} {citation['url']}")
        lines.append(f"  Used for: {citation['used_for']}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = ArgumentParser(description="Sample Deformable 3D Gaussian trajectories and compute no-ground-truth diagnostics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_times", default=250, type=int)
    parser.add_argument("--sample_gaussians", default=20000, type=int)
    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument("--knn_k", default=8, type=int)
    parser.add_argument("--motion_threshold", default=-1.0, type=float, help="Path-length threshold for moving Gaussians; negative uses adaptive threshold")
    parser.add_argument("--track_count", default=128, type=int, help="Moving Gaussian tracks to export and visualize")
    parser.add_argument("--viewer_point_count", default=-1, type=int, help="Point count shown in the dashboard point viewer; -1 keeps all points from --viewer_cloud_mode")
    parser.add_argument("--viewer_cloud_mode", default="moving", choices=["moving", "mixed", "all"], help="Which Gaussians are embedded in the dashboard point viewer")
    parser.add_argument("--dashboard_max_tracks", default=2400, type=int, help="Maximum moving tracks embedded in the HTML dashboard viewer; full selected tracks are still exported")
    parser.add_argument("--dashboard_max_cloud_points", default=-1, type=int, help="Maximum Gaussian cloud points embedded in the HTML dashboard viewer; -1 keeps all viewer points")
    parser.add_argument("--track_camera_count", default=2, type=int, help="Camera views used for track-overlay videos")
    parser.add_argument("--track_stride", default=2, type=int, help="Temporal stride for track-overlay videos")
    parser.add_argument("--track_fps", default=18, type=int, help="Frames per second for generated track-overlay videos")
    parser.add_argument("--trail_length", default=32, type=int, help="Number of sampled frames retained in each rendered trail")
    parser.add_argument("--skip_dashboard", action="store_true")
    parser.add_argument("--skip_track_video", action="store_true")
    parser.add_argument("--skip_track_export", action="store_true")
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--skip_projection", action="store_true")
    args = get_combined_args(parser)
    args = ensure_source_path(args)
    dataset = model.extract(args)
    _ = pipeline.extract(args)

    if args.num_times < 2:
        raise ValueError("--num_times must be at least 2")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path) / "trajectory_metrics"
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(torch.device("cuda:0"))
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    loaded_iteration = scene.loaded_iter if scene.loaded_iter is not None else searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.load_weights(args.model_path, args.iteration)

    selected, selected_xyz, selected_opacity = select_gaussians(gaussians, args.opacity_min, args.sample_gaussians)
    tracks, times = sample_tracks(deform, selected_xyz, args.num_times)

    motion_metrics = compute_motion_metrics(tracks)
    knn_idx, base_dist = compute_knn(tracks[0], args.knn_k)
    mean_stretch, max_stretch = compute_local_coherence(tracks, knn_idx, base_dist)
    diversity, labels = compute_motion_diversity(tracks, motion_metrics["path_length"])
    motion_split, moving_mask = compute_motion_split(motion_metrics["path_length"], tracks, args.motion_threshold)
    projection_summary = [] if args.skip_projection else compute_projection_summary(tracks, scene)

    selected_cpu = selected.detach().cpu()
    opacity_cpu = selected_opacity.detach().cpu()
    static_mask = ~moving_mask
    group_stats = {
        "static": summarize_group(static_mask, selected_cpu, opacity_cpu, labels, motion_metrics, mean_stretch, max_stretch),
        "moving": summarize_group(moving_mask, selected_cpu, opacity_cpu, labels, motion_metrics, mean_stretch, max_stretch),
    }

    selected_track_idx = select_track_indices(moving_mask, labels, motion_metrics["path_length"], args.track_count)
    track_subset = tracks[:, selected_track_idx, :] if selected_track_idx.numel() else torch.empty(tracks.shape[0], 0, 3)
    selected_track_ids = selected_cpu[selected_track_idx].tolist() if selected_track_idx.numel() else []
    selected_track_labels = labels[selected_track_idx].tolist() if selected_track_idx.numel() and labels.numel() else [-1] * len(selected_track_ids)
    selected_track_lengths = motion_metrics["path_length"][selected_track_idx].tolist() if selected_track_idx.numel() else []
    viewer_point_idx = select_viewer_point_indices(moving_mask, motion_metrics["path_length"], opacity_cpu, args.viewer_point_count, args.viewer_cloud_mode)
    viewer_cloud_subset = tracks[:, viewer_point_idx, :] if viewer_point_idx.numel() else torch.empty(tracks.shape[0], 0, 3)
    viewer_cloud_groups = moving_mask[viewer_point_idx].long() if viewer_point_idx.numel() else torch.empty(0, dtype=torch.long)

    all_video_cameras = scene.getTestCameras() + scene.getTrainCameras()
    video_camera_sequences = [] if args.skip_track_video else select_camera_sequences(all_video_cameras, max(0, args.track_camera_count))
    video_camera_names = [
        f"cam_{camera_key}" if isinstance(camera_key, int) else str(camera_key)
        for camera_key, _ in video_camera_sequences
    ]
    projected_track_sets = []
    for _, camera_sequence in video_camera_sequences:
        projected_frames = []
        for time_idx, time_value in enumerate(times.tolist()):
            frame_camera = nearest_camera_for_time(camera_sequence, time_value)
            projected_frames.append(project_points(track_subset[time_idx], frame_camera).numpy() if track_subset.numel() else np.empty((0, 2), dtype=np.float32))
        projected_track_sets.append(np.stack(projected_frames, axis=0) if projected_frames else np.empty((tracks.shape[0], 0, 2), dtype=np.float32))

    track_videos = [] if args.skip_track_video else render_track_overlay_videos(
        track_subset,
        selected_track_ids,
        selected_track_labels,
        video_camera_sequences,
        times,
        videos_dir,
        args.trail_length,
        args.track_stride,
        args.track_fps,
    )

    results = {
        "model_path": args.model_path,
        "source_path": args.source_path,
        "loaded_iteration": int(loaded_iteration),
        "total_gaussians": int(gaussians.get_xyz.shape[0]),
        "sampled_gaussians": int(selected_cpu.numel()),
        "opacity_min": float(args.opacity_min),
        "motion_threshold": finite_float(args.motion_threshold),
        "track_count": int(args.track_count),
        "viewer_point_count": int(args.viewer_point_count),
        "viewer_cloud_mode": args.viewer_cloud_mode,
        "dashboard_max_tracks": int(args.dashboard_max_tracks),
        "dashboard_max_cloud_points": int(args.dashboard_max_cloud_points),
        "track_camera_count": int(args.track_camera_count),
        "track_stride": int(args.track_stride),
        "track_fps": int(args.track_fps),
        "trail_length": int(args.trail_length),
        "num_times": int(args.num_times),
        "times": [finite_float(v) for v in times.tolist()],
        "papers": PAPER_CITATIONS,
        "metrics": {
            "motion": {
                "displacement": summarize(motion_metrics["displacement"]),
                "path_length": summarize(motion_metrics["path_length"]),
                "diameter": summarize(motion_metrics["diameter"]),
                "mean_speed": summarize(motion_metrics["mean_speed"]),
                "peak_speed": summarize(motion_metrics["peak_speed"]),
                "path_length_outliers": topk_records(motion_metrics["path_length"], selected_cpu),
                "peak_speed_outliers": topk_records(motion_metrics["peak_speed"], selected_cpu),
            },
            "smoothness": {
                "mean_acceleration": summarize(motion_metrics["mean_acceleration"]),
                "mean_jerk": summarize(motion_metrics["mean_jerk"]),
                "normalized_jerk": summarize(motion_metrics["normalized_jerk"]),
                "normalized_jerk_outliers": topk_records(motion_metrics["normalized_jerk"], selected_cpu),
            },
            "local_coherence": {
                "knn_k": int(args.knn_k),
                "mean_local_stretch": summarize(mean_stretch),
                "max_local_stretch": summarize(max_stretch),
                "local_stretch_outliers": topk_records(mean_stretch, selected_cpu),
            },
            "motion_diversity": diversity,
            "motion_split": motion_split,
            "motion_group_stats": group_stats,
            "track_visualizations": {
                "selected_gaussian_ids": [int(v) for v in selected_track_ids],
                "selected_clusters": [int(v) for v in selected_track_labels],
                "selected_path_lengths": [finite_float(v) for v in selected_track_lengths],
                "videos": track_videos,
            },
            "projection_summary": projection_summary,
        },
    }
    results["narrative"] = compute_narrative(results)

    artifact_paths = {
        "plots": {
            "path_length": plots_dir / "path_length.png",
            "mean_speed": plots_dir / "mean_speed.png",
            "acceleration": plots_dir / "acceleration.png",
            "local_stretch": plots_dir / "local_stretch.png",
            "static_path_length": plots_dir / "static_path_length.png",
            "moving_path_length": plots_dir / "moving_path_length.png",
            "static_acceleration": plots_dir / "static_acceleration.png",
            "moving_acceleration": plots_dir / "moving_acceleration.png",
            "static_local_stretch": plots_dir / "static_local_stretch.png",
            "moving_local_stretch": plots_dir / "moving_local_stretch.png",
            "pca_explained_variance": plots_dir / "pca_explained_variance.png",
            "moving_paths_xy": plots_dir / "moving_paths_xy.png",
            "moving_paths_xz": plots_dir / "moving_paths_xz.png",
            "moving_paths_yz": plots_dir / "moving_paths_yz.png",
        },
        "videos": track_videos,
        "track_viewer_data": make_track_viewer_data(
            track_subset,
            selected_track_ids,
            selected_track_labels,
            selected_track_lengths,
            viewer_cloud_subset,
            viewer_cloud_groups,
            max_tracks=args.dashboard_max_tracks,
            max_cloud_points=args.dashboard_max_cloud_points,
        ),
    }

    (output_dir / "trajectory_metrics.json").write_text(json.dumps(results, indent=2))
    write_summary_csv(output_dir / "trajectory_summary.csv", selected_cpu.tolist(), opacity_cpu.tolist(), labels, moving_mask, motion_metrics, mean_stretch, max_stretch)
    write_group_summary_csv(output_dir / "trajectory_group_summary.csv", group_stats)
    write_report(output_dir / "bonus_trajectory_report.md", results)
    if not args.skip_track_export:
        projected_export = np.stack(projected_track_sets, axis=0) if projected_track_sets else np.empty((0, tracks.shape[0], int(track_subset.shape[1]), 2), dtype=np.float32)
        export_track_samples(
            output_dir / "track_samples.npz",
            selected_track_ids,
            selected_track_labels,
            track_subset,
            selected_track_lengths,
            projected_export,
            video_camera_names,
            viewer_cloud_subset,
            viewer_cloud_groups,
        )

    make_histogram_png(motion_metrics["path_length"].numpy(), artifact_paths["plots"]["path_length"], "Path length")
    make_histogram_png(motion_metrics["mean_speed"].numpy(), artifact_paths["plots"]["mean_speed"], "Mean speed")
    make_histogram_png(motion_metrics["all_acceleration"].numpy(), artifact_paths["plots"]["acceleration"], "Acceleration")
    make_histogram_png(mean_stretch.numpy(), artifact_paths["plots"]["local_stretch"], "Mean local stretch")
    make_histogram_png(motion_metrics["path_length"][static_mask].numpy(), artifact_paths["plots"]["static_path_length"], "Static path length")
    make_histogram_png(motion_metrics["path_length"][moving_mask].numpy(), artifact_paths["plots"]["moving_path_length"], "Moving path length")
    make_histogram_png(motion_metrics["mean_acceleration"][static_mask].numpy(), artifact_paths["plots"]["static_acceleration"], "Static mean acceleration")
    make_histogram_png(motion_metrics["mean_acceleration"][moving_mask].numpy(), artifact_paths["plots"]["moving_acceleration"], "Moving mean acceleration")
    make_histogram_png(mean_stretch[static_mask].numpy(), artifact_paths["plots"]["static_local_stretch"], "Static mean local stretch")
    make_histogram_png(mean_stretch[moving_mask].numpy(), artifact_paths["plots"]["moving_local_stretch"], "Moving mean local stretch")
    make_bar_png(np.asarray(diversity["pca_explained_variance"], dtype=np.float64), artifact_paths["plots"]["pca_explained_variance"], "PCA explained variance", "principal component index")
    make_paths_png(track_subset, selected_track_labels, artifact_paths["plots"]["moving_paths_xy"], "Moving Gaussian paths: XY", axes=(0, 1))
    make_paths_png(track_subset, selected_track_labels, artifact_paths["plots"]["moving_paths_xz"], "Moving Gaussian paths: XZ", axes=(0, 2))
    make_paths_png(track_subset, selected_track_labels, artifact_paths["plots"]["moving_paths_yz"], "Moving Gaussian paths: YZ", axes=(1, 2))
    if not args.skip_dashboard:
        write_dashboard(output_dir / "trajectory_dashboard.html", results, artifact_paths)

    print(f"Wrote trajectory metrics to {output_dir}")
    print(f"Sampled {selected_cpu.numel()} Gaussians at iteration {loaded_iteration}")


if __name__ == "__main__":
    main()
