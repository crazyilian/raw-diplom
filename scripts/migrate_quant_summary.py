from pathlib import Path

from industrial_ad.experiments import load_detector_from_run
from industrial_ad.quantization import _build_summary
from industrial_ad.utils import dump_json, load_json
import tqdm


def migrate_summary(run_dir: Path) -> None:
    config = load_json(run_dir / "config.json")
    old_summary = load_json(run_dir / "summary.json")
    detector, _ = load_detector_from_run(run_dir)
    summary = _build_summary(
        run_dir,
        config,
        detector,
        old_summary["best_metrics"],
        old_summary["quantization_time_seconds"],
        old_summary["evaluation_time_seconds"],
    )
    if "benchmark" in old_summary:
        summary["benchmark"] = old_summary["benchmark"]
    dump_json(run_dir / "summary.json", summary)


with tqdm.tqdm(sorted(Path("runs").glob("*-quant/*"))) as pbar:
    for run_dir in pbar:
        if not run_dir.is_dir():
            continue
        pbar.set_description(f"{run_dir}")
        migrate_summary(run_dir)

