#
# No-ground-truth deformation-field benchmark for trajectory analytics outputs.
#

from argparse import ArgumentParser
from pathlib import Path
import csv
import html
import json
import math


FACTORS = {
    "smoothness": [
        ("mean_acceleration_median", "lower"),
        ("mean_jerk_median", "lower"),
        ("normalized_jerk_median", "lower"),
    ],
    "local_coherence": [
        ("mean_local_stretch_median", "lower"),
        ("mean_local_stretch_p90", "lower"),
        ("mean_local_stretch_p99", "lower"),
    ],
    "motion_separation": [
        ("static_path_length_median", "lower"),
        ("moving_path_length_median", "higher"),
        ("motion_separation_ratio", "higher"),
    ],
    "motion_diversity": [
        ("effective_pca_rank", "higher"),
        ("active_motion_clusters", "higher"),
        ("pca_first_component_dominance", "lower"),
    ],
    "stability": [
        ("path_length_outlier_max", "lower"),
        ("peak_speed_outlier_max", "lower"),
        ("normalized_jerk_outlier_max", "lower"),
        ("max_local_stretch_p99", "lower"),
    ],
}

DIAGNOSTIC_FACTORS = {
    "smoothness": "Temporal smoothness",
    "local_coherence": "Local coherence",
    "motion_separation": "Motion separation",
    "motion_diversity": "Motion diversity",
    "stability": "Stability / outlier control",
}


def finite_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def percentile(values, q):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def median(values):
    return percentile(values, 0.5)


def max_outlier(outliers):
    values = [finite_float(item.get("value")) for item in outliers or []]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def get_stat(summary, name):
    if not isinstance(summary, dict):
        return None
    return finite_float(summary.get(name))


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def resolve_metrics_path(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return input_path / "trajectory_metrics.json"
    return input_path


def find_summary_csv(metrics_path):
    candidate = metrics_path.parent / "trajectory_summary.csv"
    return candidate if candidate.exists() else None


def read_csv_rows(path):
    if path is None:
        return []
    with open(path, "r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def csv_medians_by_motion_group(rows, threshold):
    path_lengths = []
    static_values = []
    moving_values = []

    for row in rows:
        path_length = finite_float(row.get("path_length"))
        if path_length is None:
            continue
        path_lengths.append(path_length)
        group = row.get("motion_group")
        if group == "static":
            static_values.append(path_length)
        elif group in ("moving", "dynamic"):
            moving_values.append(path_length)

    if (not static_values or not moving_values) and threshold is not None:
        static_values = [v for v in path_lengths if v <= threshold]
        moving_values = [v for v in path_lengths if v > threshold]

    return median(static_values), median(moving_values)


def extract_group_medians(metrics, rows):
    group_stats = metrics.get("motion_group_stats") or {}
    static_stats = group_stats.get("static", {})
    moving_stats = group_stats.get("moving") or group_stats.get("dynamic") or {}

    static_median = get_stat(static_stats.get("motion", {}).get("path_length"), "median")
    moving_median = get_stat(moving_stats.get("motion", {}).get("path_length"), "median")
    if static_median is not None and moving_median is not None:
        return static_median, moving_median

    split = metrics.get("motion_split") or metrics.get("static_dynamic") or {}
    threshold = finite_float(split.get("threshold"))
    csv_static, csv_moving = csv_medians_by_motion_group(rows, threshold)
    return (
        static_median if static_median is not None else csv_static,
        moving_median if moving_median is not None else csv_moving,
    )


def extract_raw_scores(metrics_path):
    data = load_json(metrics_path)
    metrics = data.get("metrics", {})
    rows = read_csv_rows(find_summary_csv(metrics_path))

    motion = metrics.get("motion", {})
    smoothness = metrics.get("smoothness", {})
    local = metrics.get("local_coherence", {})
    diversity = metrics.get("motion_diversity", {})

    static_median, moving_median = extract_group_medians(metrics, rows)
    ratio = None
    if static_median is not None and moving_median is not None:
        ratio = moving_median / max(static_median, 1e-8)

    pca_variance = diversity.get("pca_explained_variance") or []
    first_pca = finite_float(pca_variance[0]) if pca_variance else None

    raw = {
        "mean_acceleration_median": get_stat(smoothness.get("mean_acceleration"), "median"),
        "mean_jerk_median": get_stat(smoothness.get("mean_jerk"), "median"),
        "normalized_jerk_median": get_stat(smoothness.get("normalized_jerk"), "median"),
        "mean_local_stretch_median": get_stat(local.get("mean_local_stretch"), "median"),
        "mean_local_stretch_p90": get_stat(local.get("mean_local_stretch"), "p90"),
        "mean_local_stretch_p99": get_stat(local.get("mean_local_stretch"), "p99"),
        "static_path_length_median": static_median,
        "moving_path_length_median": moving_median,
        "motion_separation_ratio": ratio,
        "effective_pca_rank": finite_float(diversity.get("effective_pca_rank")),
        "active_motion_clusters": finite_float(diversity.get("active_clusters")),
        "pca_first_component_dominance": first_pca,
        "path_length_outlier_max": max_outlier(motion.get("path_length_outliers")),
        "peak_speed_outlier_max": max_outlier(motion.get("peak_speed_outliers")),
        "normalized_jerk_outlier_max": max_outlier(smoothness.get("normalized_jerk_outliers")),
        "max_local_stretch_p99": get_stat(local.get("max_local_stretch"), "p99"),
    }

    return {
        "name": metrics_path.parent.name,
        "metrics_path": str(metrics_path),
        "model_path": data.get("model_path"),
        "loaded_iteration": data.get("loaded_iteration"),
        "raw_metrics": raw,
    }


def normalize(value, p5, p95, direction):
    value = finite_float(value)
    if value is None or p5 is None or p95 is None:
        return None
    if math.isclose(p5, p95):
        return 50.0
    if direction == "lower":
        score = 100.0 * (p95 - value) / (p95 - p5)
    elif direction == "higher":
        score = 100.0 * (value - p5) / (p95 - p5)
    else:
        raise ValueError(f"Unknown score direction: {direction}")
    return max(0.0, min(100.0, score))


def clamp_score(value):
    value = finite_float(value)
    if value is None:
        return None
    return max(0.0, min(100.0, value))


def bounded_lower_score(value, good, poor):
    value = finite_float(value)
    if value is None:
        return None
    if math.isclose(good, poor):
        return 50.0
    return clamp_score(100.0 * (poor - value) / (poor - good))


def bounded_higher_score(value, poor, good):
    value = finite_float(value)
    if value is None:
        return None
    if math.isclose(good, poor):
        return 50.0
    return clamp_score(100.0 * (value - poor) / (good - poor))


def safe_ratio(numerator, denominator):
    numerator = finite_float(numerator)
    denominator = finite_float(denominator)
    if numerator is None or denominator is None:
        return None
    return numerator / max(denominator, 1e-8)


def average(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def diagnostic_component(label, score, raw, reasoning):
    return {
        "label": label,
        "score": score,
        "raw": raw,
        "reasoning": reasoning,
    }


def score_run_diagnostic(run):
    raw = run["raw_metrics"]
    moving_path = raw.get("moving_path_length_median")
    static_path = raw.get("static_path_length_median")
    normalized_jerk = raw.get("normalized_jerk_median")
    jerk_tail_ratio = safe_ratio(raw.get("normalized_jerk_outlier_max"), normalized_jerk)
    path_tail_ratio = safe_ratio(raw.get("path_length_outlier_max"), moving_path)

    smoothness_score = average([
        bounded_lower_score(normalized_jerk, 250.0, 2500.0),
        bounded_lower_score(jerk_tail_ratio, 5.0, 50.0),
    ])
    local_score = average([
        bounded_lower_score(raw.get("mean_local_stretch_median"), 0.05, 0.50),
        bounded_lower_score(raw.get("mean_local_stretch_p90"), 0.15, 1.25),
        bounded_lower_score(raw.get("mean_local_stretch_p99"), 0.50, 4.00),
        bounded_lower_score(raw.get("max_local_stretch_p99"), 2.00, 20.00),
    ])
    separation_score = average([
        bounded_lower_score(safe_ratio(static_path, moving_path), 0.03, 0.50),
        bounded_higher_score(raw.get("motion_separation_ratio"), 2.0, 20.0),
    ])
    diversity_score = average([
        bounded_higher_score(raw.get("effective_pca_rank"), 1.0, 4.0),
        bounded_higher_score(raw.get("active_motion_clusters"), 1.0, 4.0),
        bounded_lower_score(raw.get("pca_first_component_dominance"), 0.45, 0.95),
    ])
    stability_score = average([
        bounded_lower_score(path_tail_ratio, 3.0, 25.0),
        bounded_lower_score(jerk_tail_ratio, 5.0, 50.0),
        bounded_lower_score(raw.get("max_local_stretch_p99"), 2.00, 20.00),
    ])

    factors = {
        "smoothness": diagnostic_component(
            DIAGNOSTIC_FACTORS["smoothness"],
            smoothness_score,
            {
                "normalized_jerk_median": normalized_jerk,
                "normalized_jerk_outlier_to_median": jerk_tail_ratio,
            },
            "Rewards low median normalized jerk and avoids large jerk tails among sampled trajectories.",
        ),
        "local_coherence": diagnostic_component(
            DIAGNOSTIC_FACTORS["local_coherence"],
            local_score,
            {
                "mean_local_stretch_median": raw.get("mean_local_stretch_median"),
                "mean_local_stretch_p90": raw.get("mean_local_stretch_p90"),
                "mean_local_stretch_p99": raw.get("mean_local_stretch_p99"),
                "max_local_stretch_p99": raw.get("max_local_stretch_p99"),
            },
            "Rewards neighborhoods that keep canonical neighbor distances stable through time.",
        ),
        "motion_separation": diagnostic_component(
            DIAGNOSTIC_FACTORS["motion_separation"],
            separation_score,
            {
                "static_to_moving_path_median": safe_ratio(static_path, moving_path),
                "moving_to_static_path_median": raw.get("motion_separation_ratio"),
            },
            "Rewards a clean path-length split where static Gaussians stay near-static and moving Gaussians carry the scene motion.",
        ),
        "motion_diversity": diagnostic_component(
            DIAGNOSTIC_FACTORS["motion_diversity"],
            diversity_score,
            {
                "effective_pca_rank": raw.get("effective_pca_rank"),
                "active_motion_clusters": raw.get("active_motion_clusters"),
                "pca_first_component_dominance": raw.get("pca_first_component_dominance"),
            },
            "Rewards motion that is not collapsed into one dominant mode while still using coherent PCA and cluster structure.",
        ),
        "stability": diagnostic_component(
            DIAGNOSTIC_FACTORS["stability"],
            stability_score,
            {
                "path_length_outlier_to_moving_median": path_tail_ratio,
                "normalized_jerk_outlier_to_median": jerk_tail_ratio,
                "max_local_stretch_p99": raw.get("max_local_stretch_p99"),
            },
            "Penalizes extreme trajectory-length, jerk, and local-stretch tails relative to the run's own typical moving behavior.",
        ),
    }
    scores = [item["score"] for item in factors.values()]
    return {
        "score_name": "Diagnostic Deformation Quality Score",
        "score_abbreviation": "DQS",
        "dqs": average(scores),
        "factor_scores": {name: item["score"] for name, item in factors.items()},
        "components": factors,
        "method": "single_run_scale_aware_diagnostic",
        "caveat": "This is a no-ground-truth deformation-field diagnostic. It is intended for manual comparison between runs produced with the same trajectory analytics settings, not as APD/OA/AJ tracking accuracy.",
    }


def build_reference_ranges(runs):
    metric_names = sorted({metric for factor in FACTORS.values() for metric, _ in factor})
    ranges = {}
    for metric_name in metric_names:
        values = [run["raw_metrics"].get(metric_name) for run in runs]
        ranges[metric_name] = {
            "p5": percentile(values, 0.05),
            "p95": percentile(values, 0.95),
            "count": len([v for v in values if v is not None and math.isfinite(v)]),
        }
    return ranges


def score_run(run, ranges):
    components = {}
    factors = {}
    for factor_name, factor_metrics in FACTORS.items():
        factor_scores = []
        for metric_name, direction in factor_metrics:
            ref = ranges[metric_name]
            score = normalize(run["raw_metrics"].get(metric_name), ref["p5"], ref["p95"], direction)
            components[metric_name] = {
                "raw": run["raw_metrics"].get(metric_name),
                "score": score,
                "direction": direction,
                "reference_p5": ref["p5"],
                "reference_p95": ref["p95"],
            }
            factor_scores.append(score)
        factors[factor_name] = average(factor_scores)
    return {
        **run,
        "factor_scores": factors,
        "component_scores": components,
        "dqs": average(list(factors.values())),
    }


def format_score(value):
    return "n/a" if value is None else f"{value:.2f}"


def format_raw(value):
    return "n/a" if value is None else f"{value:.6g}"


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_report(path, payload):
    runs = sorted(payload["runs"], key=lambda item: item["dqs"] if item["dqs"] is not None else -1.0, reverse=True)
    warnings = payload.get("warnings", [])

    lines = [
        "# Deformation Field Benchmark",
        "",
        "This is a no-ground-truth benchmark of the learned deformation field only. It uses sampled Gaussian trajectories and does not include rendering metrics, PSNR, SSIM, LPIPS, or ground-truth tracking accuracy.",
        "",
        "Each factor is weighted equally: Smoothness, Local Coherence, Motion Separation, Motion Diversity, and Stability / Outlier Control.",
        "",
        "## Results",
        "",
        "| Rank | Run | DQS | Smoothness | Local coherence | Motion separation | Motion diversity | Stability |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, run in enumerate(runs, start=1):
        factors = run["factor_scores"]
        lines.append(
            f"| {rank} | {run['name']} | {format_score(run['dqs'])} | "
            f"{format_score(factors.get('smoothness'))} | "
            f"{format_score(factors.get('local_coherence'))} | "
            f"{format_score(factors.get('motion_separation'))} | "
            f"{format_score(factors.get('motion_diversity'))} | "
            f"{format_score(factors.get('stability'))} |"
        )

    lines.extend([
        "",
        "## Standalone Diagnostic Score",
        "",
        "The standalone DQS is computed for each run without looking at other benchmark inputs. It uses fixed, scale-aware diagnostic transforms so that you can run two benchmarks independently and manually compare their DQS values.",
        "",
        "| Run | Diagnostic DQS | Smoothness | Local coherence | Motion separation | Motion diversity | Stability |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for run in runs:
        diagnostic = run.get("diagnostic_score", {})
        factors = diagnostic.get("factor_scores", {})
        lines.append(
            f"| {run['name']} | {format_score(diagnostic.get('dqs'))} | "
            f"{format_score(factors.get('smoothness'))} | "
            f"{format_score(factors.get('local_coherence'))} | "
            f"{format_score(factors.get('motion_separation'))} | "
            f"{format_score(factors.get('motion_diversity'))} | "
            f"{format_score(factors.get('stability'))} |"
        )

    lines.extend([
        "",
        "### Diagnostic Reasoning",
        "",
    ])
    for run in runs:
        diagnostic = run.get("diagnostic_score", {})
        lines.append(f"#### {run['name']}")
        lines.append("")
        lines.append(diagnostic.get("caveat", "No diagnostic caveat recorded."))
        lines.append("")
        for component in diagnostic.get("components", {}).values():
            raw_bits = ", ".join(f"{key}={format_raw(value)}" for key, value in component.get("raw", {}).items())
            lines.append(f"- {component['label']}: {format_score(component.get('score'))}. {component.get('reasoning')} Raw: {raw_bits}.")
        lines.append("")

    lines.extend([
        "",
        "## Scoring",
        "",
        "Relative scores are normalized against the provided benchmark set using p5/p95 ranges, clamped to 0-100. Lower-is-better metrics use `100 * (p95 - value) / (p95 - p5)`. Higher-is-better metrics use `100 * (value - p5) / (p95 - p5)`.",
        "",
        "A single run or a constant benchmark range receives neutral 50-point relative component scores because there is no relative reference spread. Use the standalone diagnostic DQS when you want one comparable score per independently generated trajectory dashboard.",
        "",
        "## Component Definitions",
        "",
        "- Smoothness: median mean acceleration, median mean jerk, median normalized jerk.",
        "- Local Coherence: median, P90, and P99 mean local stretch.",
        "- Motion Separation: static median path length, moving median path length, moving/static median path ratio.",
        "- Motion Diversity: effective PCA rank, active motion clusters, first PCA component dominance.",
        "- Stability / Outlier Control: path-length outlier max, peak-speed outlier max, normalized-jerk outlier max, P99 max local stretch.",
    ])

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])

    path.write_text("\n".join(lines) + "\n")


def dashboard_metric_card(title, value, caption):
    return (
        '<div class="metric-card">'
        f"<div class=\"metric-title\">{html.escape(title)}</div>"
        f"<div class=\"metric-value\">{html.escape(value)}</div>"
        f"<div class=\"metric-caption\">{html.escape(caption)}</div>"
        "</div>"
    )


def dashboard_table_rows(rows):
    return "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )


def build_dashboard_section(run):
    diagnostic = run.get("diagnostic_score", {})
    components = diagnostic.get("components", {})
    factor_cards = [
        dashboard_metric_card(
            component.get("label", name.replace("_", " ").title()),
            format_score(component.get("score")),
            component.get("reasoning", ""),
        )
        for name, component in components.items()
    ]
    reasoning_rows = []
    for component in components.values():
        raw = component.get("raw", {})
        raw_text = ", ".join(f"{key}: {format_raw(value)}" for key, value in raw.items())
        reasoning_rows.append([
            component.get("label", "n/a"),
            format_score(component.get("score")),
            raw_text,
        ])

    return f"""
    <section id="deformationBenchmark">
      <h2>Deformation Benchmark</h2>
      <p>This score is generated by <code>deformation_benchmark.py</code> from the same trajectory metrics used by this dashboard. It is a no-ground-truth deformation-field diagnostic, so compare it only between runs produced with the same analytics settings.</p>
      <div class="grid">
        {dashboard_metric_card("Diagnostic DQS", format_score(diagnostic.get("dqs")), "Single-run deformation quality score, 0-100. Higher is better.")}
        {''.join(factor_cards)}
      </div>
      <h3>Benchmark reasoning</h3>
      <table>
        <thead><tr><th>Factor</th><th>Score</th><th>Evidence</th></tr></thead>
        <tbody>{dashboard_table_rows(reasoning_rows)}</tbody>
      </table>
      <p class="caption">{html.escape(diagnostic.get("caveat", ""))}</p>
    </section>
"""


def update_dashboard(metrics_path, run):
    dashboard_path = metrics_path.parent / "trajectory_dashboard.html"
    if not dashboard_path.exists():
        return False

    document = dashboard_path.read_text()
    section = build_dashboard_section(run)
    start = document.find('    <section id="deformationBenchmark">')
    if start != -1:
        end = document.find("\n    <section>", start + 1)
        if end == -1:
            end = document.find("\n    <footer>", start + 1)
        if end == -1:
            return False
        document = document[:start] + section.rstrip() + document[end:]
    else:
        marker = "\n\n    <section>\n      <h2>Static vs Moving Split</h2>"
        if marker in document:
            document = document.replace(marker, "\n" + section + marker, 1)
        else:
            main_end = document.find("\n    <footer>")
            if main_end == -1:
                return False
            document = document[:main_end] + "\n" + section + document[main_end:]

    dashboard_path.write_text(document)
    return True


def main():
    parser = ArgumentParser(description="Score no-ground-truth deformation-field quality from trajectory analytics outputs.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Trajectory output directories or trajectory_metrics.json files. Provide multiple runs for meaningful relative scores.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for deformation_benchmark.json and deformation_benchmark_report.md. Defaults to the first input directory.",
    )
    args = parser.parse_args()

    metric_paths = [resolve_metrics_path(path) for path in args.inputs]
    missing = [str(path) for path in metric_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing trajectory_metrics.json file(s): " + ", ".join(missing))

    runs = [extract_raw_scores(path) for path in metric_paths]
    ranges = build_reference_ranges(runs)
    scored_runs = []
    for run in runs:
        scored = score_run(run, ranges)
        scored["diagnostic_score"] = score_run_diagnostic(run)
        scored_runs.append(scored)

    warnings = []
    if len(scored_runs) < 2:
        warnings.append("Only one run was provided, so every component score uses the neutral 50 fallback. Provide multiple runs/checkpoints/scenes for a meaningful relative benchmark.")
    for metric_name, ref in ranges.items():
        if ref["count"] == 0:
            warnings.append(f"No finite values were found for component metric `{metric_name}`.")
        elif math.isclose(ref["p5"], ref["p95"]):
            warnings.append(f"Component metric `{metric_name}` has no reference spread; neutral 50 scores were used for that component.")

    output_dir = Path(args.output_dir) if args.output_dir else metric_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "benchmark": "no_ground_truth_deformation_field_quality",
        "score": "DQS",
        "factor_weights": {name: 1.0 / len(FACTORS) for name in FACTORS},
        "factors": FACTORS,
        "reference_ranges": ranges,
        "warnings": warnings,
        "runs": scored_runs,
    }

    write_json(output_dir / "deformation_benchmark.json", payload)
    write_report(output_dir / "deformation_benchmark_report.md", payload)
    updated_dashboards = 0
    for path, run in zip(metric_paths, scored_runs):
        if update_dashboard(path, run):
            updated_dashboards += 1
    print(f"Wrote deformation benchmark to {output_dir}")
    if updated_dashboards:
        print(f"Updated {updated_dashboards} trajectory dashboard(s) with benchmark scores")


if __name__ == "__main__":
    main()
