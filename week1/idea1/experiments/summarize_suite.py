"""
Aggregate run_suite outputs into per-variant mean/std summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


NUMERIC_FIELDS = [
    "training_seconds",
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "time_limit_rate",
    "avg_steps",
    "avg_reward",
    "avg_final_distance",
    "inference_ms_mean",
    "avg_lidar_dropout_ratio",
    "avg_rgb_occlusion_ratio",
    "avg_imu_dropout_dims",
]


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    summaries: List[Dict[str, object]] = []
    for variant, variant_rows in sorted(grouped.items()):
        out: Dict[str, object] = {
            "variant": variant,
            "num_runs": len(variant_rows),
            "difficulty": variant_rows[0].get("difficulty", ""),
            "degradation_level": _to_float(variant_rows[0].get("degradation_level", "nan")),
        }
        for field in NUMERIC_FIELDS:
            values = np.array([_to_float(r.get(field, "nan")) for r in variant_rows], dtype=np.float64)
            out[f"{field}_mean"] = float(np.nanmean(values))
            out[f"{field}_std"] = float(np.nanstd(values))
        summaries.append(out)
    return summaries


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    headers = sorted(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_compact_table(rows: List[Dict[str, object]]) -> None:
    print("\nVariant summary (mean ± std):")
    for r in rows:
        print(
            f"- {r['variant']}: "
            f"reward={r['avg_reward_mean']:.3f}±{r['avg_reward_std']:.3f}, "
            f"success={r['success_rate_mean']:.3f}±{r['success_rate_std']:.3f}, "
            f"collision={r['collision_rate_mean']:.3f}±{r['collision_rate_std']:.3f}, "
            f"time_limit={r['time_limit_rate_mean']:.3f}±{r['time_limit_rate_std']:.3f}, "
            f"train_s={r['training_seconds_mean']:.2f}±{r['training_seconds_std']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize run_suite results by variant")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to suite_summary.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for summary outputs (default: same as input CSV directory)",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_csv.parent
    rows = load_rows(input_csv)
    summary_rows = summarize(rows)

    csv_out = output_dir / "suite_summary_by_variant.csv"
    json_out = output_dir / "suite_summary_by_variant.json"
    save_csv(csv_out, summary_rows)
    save_json(
        json_out,
        {
            "input_csv": str(input_csv),
            "num_rows": len(rows),
            "num_variants": len(summary_rows),
            "summary": summary_rows,
        },
    )

    print(f"input_csv={input_csv}")
    print(f"output_csv={csv_out}")
    print(f"output_json={json_out}")
    print_compact_table(summary_rows)


if __name__ == "__main__":
    main()
