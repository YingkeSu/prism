"""
Run a reproducible comparison suite for Idea1 experiments.

Compares:
1) Ours (dynamic reliability-aware fusion)
2) Ablation (no reliability)
3) Baseline (fixed equal weights)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder
from utils.experiment_pipeline import (
    create_sac_model,
    evaluate_model,
    save_json,
    save_results_csv,
    train_model,
)


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def flatten_row(run: Dict[str, object]) -> Dict[str, object]:
    metrics = run["metrics"]
    assert isinstance(metrics, dict)
    return {
        "run_name": run["run_name"],
        "variant": run["variant"],
        "seed": run["seed"],
        "timesteps": run["timesteps"],
        "training_seconds": round(float(run["training_seconds"]), 3),
        "avg_reward": round(float(metrics["avg_reward"]), 6),
        "success_rate": round(float(metrics["success_rate"]), 6),
        "collision_rate": round(float(metrics["collision_rate"]), 6),
        "out_of_bounds_rate": round(float(metrics["out_of_bounds_rate"]), 6),
        "time_limit_rate": round(float(metrics["time_limit_rate"]), 6),
        "avg_steps": round(float(metrics["avg_steps"]), 6),
        "avg_final_distance": round(float(metrics["avg_final_distance"]), 6),
        "inference_ms_mean": round(float(metrics["inference_ms_mean"]), 6),
        "avg_lidar_dropout_ratio": round(float(metrics["avg_lidar_dropout_ratio"]), 6),
        "avg_rgb_occlusion_ratio": round(float(metrics["avg_rgb_occlusion_ratio"]), 6),
        "avg_imu_dropout_dims": round(float(metrics["avg_imu_dropout_dims"]), 6),
        "degradation_level": round(float(run["degradation_level"]), 3),
        "difficulty": run["difficulty"],
    }


def run_single(
    *,
    variant: str,
    seed: int,
    total_timesteps: int,
    eval_episodes: int,
    degradation_level: float,
    difficulty: str,
    max_steps: int,
    imu_history_len: int,
    share_features_extractor: bool,
    output_dir: Path,
    save_model: bool,
) -> Dict[str, object]:
    use_reliability_modulation = True
    if variant == "ours_dynamic":
        use_reliability = True
        fixed_weights: Optional[Sequence[float]] = None
    elif variant == "no_reliability":
        use_reliability = False
        fixed_weights = None
    elif variant == "fixed_equal":
        use_reliability = True
        fixed_weights = (1 / 3, 1 / 3, 1 / 3)
    elif variant == "no_norm_modulation":
        use_reliability = True
        fixed_weights = None
        use_reliability_modulation = False
    else:
        raise ValueError(f"Unknown variant: {variant}")

    run_name = f"{variant}_seed{seed}"
    print("\n" + "=" * 80)
    print(f"Run: {run_name}")
    print("=" * 80)

    train_env = UAVMultimodalEnv(
        max_steps=max(1, int(max_steps)),
        degradation_level=degradation_level,
        difficulty=difficulty,
    )
    model = create_sac_model(
        train_env,
        GAPEncoder,
        use_reliability=use_reliability,
        use_reliability_modulation=use_reliability_modulation,
        fixed_weights=fixed_weights,
        debug_mode=False,
        imu_history_len=imu_history_len,
        share_features_extractor=share_features_extractor,
        seed=seed,
        tensorboard_log=None,
    )
    training_seconds = train_model(
        model,
        total_timesteps=total_timesteps,
        log_interval=None,
        tb_log_name=run_name,
        callback=None,
    )
    print(f"training_seconds={training_seconds:.2f}")

    if save_model:
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{run_name}.zip"
        model.save(str(model_path))
        print(f"model_saved={model_path}")

    eval_env = UAVMultimodalEnv(
        max_steps=max(1, int(max_steps)),
        degradation_level=degradation_level,
        difficulty=difficulty,
    )
    metrics = evaluate_model(
        model,
        eval_env,
        num_episodes=eval_episodes,
        deterministic=True,
        seed=seed,
    )
    print(
        "metrics:"
        f" success_rate={metrics['success_rate']:.3f},"
        f" collision_rate={metrics['collision_rate']:.3f},"
        f" avg_reward={metrics['avg_reward']:.3f}"
    )

    run_result: Dict[str, object] = {
        "run_name": run_name,
        "variant": variant,
        "seed": seed,
        "degradation_level": degradation_level,
        "difficulty": difficulty,
        "timesteps": total_timesteps,
        "training_seconds": training_seconds,
        "metrics": metrics,
    }
    save_json(output_dir / f"{run_name}.json", run_result)
    return run_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Idea1 comparison suite")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--degradation-level", type=float, default=0.0)
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--imu-history-len", type=int, default=16)
    parser.add_argument(
        "--no-share-features-extractor",
        action="store_true",
        help="Use separate actor/critic feature extractors (slower).",
    )
    parser.add_argument(
        "--include-norm-ablation",
        action="store_true",
        help="Include no_norm_modulation ablation variant",
    )
    parser.add_argument("--output-dir", type=str, default="logs/experiments/suite")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    seeds = parse_seeds(args.seeds)
    variants = ["ours_dynamic", "no_reliability", "fixed_equal"]
    if args.include_norm_ablation:
        variants.append("no_norm_modulation")

    all_runs: List[Dict[str, object]] = []
    flat_rows: List[Dict[str, object]] = []
    for seed in seeds:
        for variant in variants:
            run = run_single(
                variant=variant,
                seed=seed,
                total_timesteps=args.timesteps,
                eval_episodes=args.eval_episodes,
                degradation_level=float(max(0.0, min(1.0, args.degradation_level))),
                difficulty=args.difficulty,
                max_steps=max(1, int(args.max_steps)),
                imu_history_len=max(1, int(args.imu_history_len)),
                share_features_extractor=not args.no_share_features_extractor,
                output_dir=output_dir,
                save_model=args.save_model,
            )
            all_runs.append(run)
            flat_rows.append(flatten_row(run))

    suite_payload = {
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "degradation_level": float(max(0.0, min(1.0, args.degradation_level))),
        "difficulty": args.difficulty,
        "max_steps": max(1, int(args.max_steps)),
        "seeds": seeds,
        "variants": variants,
        "runs": all_runs,
    }
    save_json(output_dir / "suite_results.json", suite_payload)
    save_results_csv(output_dir / "suite_summary.csv", flat_rows)

    print("\n" + "=" * 80)
    print("Suite complete")
    print("=" * 80)
    print(f"json: {output_dir / 'suite_results.json'}")
    print(f"csv:  {output_dir / 'suite_summary.csv'}")


if __name__ == "__main__":
    main()
