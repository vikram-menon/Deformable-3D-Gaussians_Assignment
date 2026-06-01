#
# Orchestrate trajectory analytics plus deformation benchmark into one dashboard.
#

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys


def contains_flag(args, long_name, short_name=None):
    names = {long_name}
    if short_name:
        names.add(short_name)
    return any(arg in names or any(arg.startswith(name + "=") for name in names) for arg in args)


def append_if_present(args, name, value, short_name=None):
    if value in (None, ""):
        return
    if contains_flag(args, name, short_name):
        return
    args.extend([name, str(value)])


def main():
    parser = ArgumentParser(
        description=(
            "Run trajectory_metrics.py and deformation_benchmark.py, then embed the "
            "benchmark score in the generated trajectory_dashboard.html."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for both child scripts.")
    parser.add_argument("--model_path", "-m", default="", help="Forwarded to trajectory_metrics.py.")
    parser.add_argument("--output_dir", default="", help="Forwarded to trajectory_metrics.py; defaults to <model_path>/trajectory_metrics.")
    parser.add_argument(
        "--benchmark_output_dir",
        default="",
        help="Directory for deformation_benchmark.json/report. Defaults to the trajectory output directory.",
    )
    args, trajectory_args = parser.parse_known_args()

    if contains_flag(trajectory_args, "--skip_dashboard"):
        raise ValueError("Do not pass --skip_dashboard when using the orchestrator; it needs trajectory_dashboard.html.")

    append_if_present(trajectory_args, "--model_path", args.model_path, "-m")
    append_if_present(trajectory_args, "--output_dir", args.output_dir)

    model_path = args.model_path
    if not model_path:
        for idx, item in enumerate(trajectory_args):
            if item in ("--model_path", "-m") and idx + 1 < len(trajectory_args):
                model_path = trajectory_args[idx + 1]
                break
            if item.startswith("--model_path="):
                model_path = item.split("=", 1)[1]
                break

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        for idx, item in enumerate(trajectory_args):
            if item == "--output_dir" and idx + 1 < len(trajectory_args):
                output_dir = Path(trajectory_args[idx + 1])
                break
            if item.startswith("--output_dir="):
                output_dir = Path(item.split("=", 1)[1])
                break
    if output_dir is None:
        if not model_path:
            raise ValueError("Provide --model_path or --output_dir so the orchestrator can locate trajectory_metrics.json.")
        output_dir = Path(model_path) / "trajectory_metrics"

    script_dir = Path(__file__).resolve().parent
    trajectory_script = script_dir / "trajectory_metrics.py"
    benchmark_script = script_dir / "deformation_benchmark.py"

    trajectory_cmd = [args.python, str(trajectory_script), *trajectory_args]
    print("Running trajectory metrics:")
    print(" ".join(trajectory_cmd))
    subprocess.run(trajectory_cmd, check=True)

    metrics_path = output_dir / "trajectory_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected trajectory metrics at {metrics_path}")

    benchmark_output = Path(args.benchmark_output_dir) if args.benchmark_output_dir else output_dir
    benchmark_cmd = [
        args.python,
        str(benchmark_script),
        str(metrics_path),
        "--output-dir",
        str(benchmark_output),
    ]
    print("Running deformation benchmark:")
    print(" ".join(benchmark_cmd))
    subprocess.run(benchmark_cmd, check=True)

    dashboard_path = output_dir / "trajectory_dashboard.html"
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Expected dashboard at {dashboard_path}")
    print(f"Wrote combined dashboard to {dashboard_path}")


if __name__ == "__main__":
    main()
