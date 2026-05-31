#
# Bonus trajectory diagnostics for trained Deformable 3D Gaussians models.
#

from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

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


def compute_static_dynamic(path_length, tracks):
    if path_length.numel() == 0:
        return {"threshold": None, "static_fraction": None, "dynamic_fraction": None, "dynamic_bbox_min": None, "dynamic_bbox_max": None}
    threshold = max(float(torch.quantile(path_length, 0.10)), float(path_length.median()) * 0.05, 1e-6)
    dynamic_mask = path_length > threshold
    static_fraction = 1.0 - float(dynamic_mask.float().mean())
    if dynamic_mask.any():
        dynamic_tracks = tracks[:, dynamic_mask, :].reshape(-1, 3)
        bbox_min = [finite_float(v) for v in dynamic_tracks.min(dim=0).values.tolist()]
        bbox_max = [finite_float(v) for v in dynamic_tracks.max(dim=0).values.tolist()]
    else:
        bbox_min = None
        bbox_max = None
    return {
        "threshold": finite_float(threshold),
        "static_fraction": finite_float(static_fraction),
        "dynamic_fraction": finite_float(1.0 - static_fraction),
        "dynamic_bbox_min": bbox_min,
        "dynamic_bbox_max": bbox_max,
    }


def project_points(points, camera):
    device = camera.full_proj_transform.device
    points = points.to(device)
    projected = geom_transform_points(points, camera.full_proj_transform)
    x = (projected[:, 0] + 1.0) * 0.5 * camera.image_width
    y = (1.0 - projected[:, 1]) * 0.5 * camera.image_height
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


def write_summary_csv(path, gaussian_ids, opacities, labels, metrics, mean_stretch, max_stretch):
    fieldnames = [
        "gaussian_id", "opacity", "cluster", "displacement", "path_length", "diameter",
        "mean_speed", "peak_speed", "mean_acceleration", "mean_jerk", "normalized_jerk",
        "mean_local_stretch", "max_local_stretch",
    ]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, gaussian_id in enumerate(gaussian_ids):
            writer.writerow({
                "gaussian_id": int(gaussian_id),
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


def write_report(path, data):
    motion = data["metrics"]["motion"]
    smoothness = data["metrics"]["smoothness"]
    coherence = data["metrics"]["local_coherence"]
    static_dynamic = data["metrics"]["static_dynamic"]
    diversity = data["metrics"]["motion_diversity"]

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
        f"- Static fraction: {fmt_fixed(static_dynamic['static_fraction'])}.",
        f"- Dynamic fraction: {fmt_fixed(static_dynamic['dynamic_fraction'])}.",
        f"- Effective PCA rank: {fmt_fixed(diversity['effective_pca_rank'])}.",
        f"- Active motion clusters: {diversity['active_clusters']}.",
        "",
        "## Assessment",
        "",
        "Large path length and diameter identify Gaussians carrying the visible dancer motion. High normalized jerk marks temporally rough tracks and is useful for finding deformation-field jitter. Local stretch measures whether nearby canonical Gaussians remain coherent over time; large values indicate local tearing, excessive expansion, or inconsistent motion among neighbors.",
        "",
        "The static/dynamic split is adaptive and should be interpreted as a scene diagnostic, not a semantic segmentation. A healthy dynamic reconstruction should contain a static component for background and a concentrated dynamic component for the performer.",
        "",
        "## Papers Used",
        "",
    ]
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
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    static_dynamic = compute_static_dynamic(motion_metrics["path_length"], tracks)
    projection_summary = [] if args.skip_projection else compute_projection_summary(tracks, scene)

    selected_cpu = selected.detach().cpu()
    opacity_cpu = selected_opacity.detach().cpu()

    results = {
        "model_path": args.model_path,
        "source_path": args.source_path,
        "loaded_iteration": int(loaded_iteration),
        "total_gaussians": int(gaussians.get_xyz.shape[0]),
        "sampled_gaussians": int(selected_cpu.numel()),
        "opacity_min": float(args.opacity_min),
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
            "static_dynamic": static_dynamic,
            "projection_summary": projection_summary,
        },
    }

    (output_dir / "trajectory_metrics.json").write_text(json.dumps(results, indent=2))
    write_summary_csv(output_dir / "trajectory_summary.csv", selected_cpu.tolist(), opacity_cpu.tolist(), labels, motion_metrics, mean_stretch, max_stretch)
    write_report(output_dir / "bonus_trajectory_report.md", results)

    make_histogram_png(motion_metrics["path_length"].numpy(), plots_dir / "path_length.png", "Path length")
    make_histogram_png(motion_metrics["mean_speed"].numpy(), plots_dir / "mean_speed.png", "Mean speed")
    make_histogram_png(motion_metrics["all_acceleration"].numpy(), plots_dir / "acceleration.png", "Acceleration")
    make_histogram_png(mean_stretch.numpy(), plots_dir / "local_stretch.png", "Mean local stretch")
    make_histogram_png(np.asarray(diversity["pca_explained_variance"], dtype=np.float64), plots_dir / "pca_spectrum.png", "PCA explained variance")

    print(f"Wrote trajectory metrics to {output_dir}")
    print(f"Sampled {selected_cpu.numel()} Gaussians at iteration {loaded_iteration}")


if __name__ == "__main__":
    main()
