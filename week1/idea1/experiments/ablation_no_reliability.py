"""
Ablation experiment: disable reliability-aware fusion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder
from utils.experiment_pipeline import create_sac_model, evaluate_model, save_json, train_model


def run_ablation_no_reliability(
    *,
    total_timesteps: int,
    eval_episodes: int,
    seed: int,
    degradation_level: float,
    difficulty: str,
    max_steps: int,
    imu_history_len: int,
    share_features_extractor: bool,
    output_dir: Path,
    save_model: bool,
) -> None:
    run_name = f"ablation_no_reliability_seed{seed}"
    print("=" * 70)
    print(f"Running {run_name}")
    print("=" * 70)
    print(f"timesteps={total_timesteps}, eval_episodes={eval_episodes}")

    train_env = UAVMultimodalEnv(
        max_steps=max(1, int(max_steps)),
        degradation_level=degradation_level,
        difficulty=difficulty,
    )
    model = create_sac_model(
        train_env,
        GAPEncoder,
        use_reliability=False,
        debug_mode=False,
        imu_history_len=imu_history_len,
        share_features_extractor=share_features_extractor,
        seed=seed,
        tensorboard_log=None,
    )

    train_seconds = train_model(
        model,
        total_timesteps=total_timesteps,
        log_interval=None,
        tb_log_name=run_name,
        callback=None,
    )
    print(f"training_time={train_seconds:.2f}s")

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

    result = {
        "experiment": "ablation_no_reliability",
        "run_name": run_name,
        "seed": seed,
        "timesteps": total_timesteps,
        "degradation_level": degradation_level,
        "difficulty": difficulty,
        "training_seconds": train_seconds,
        "metrics": metrics,
    }
    result_path = output_dir / f"{run_name}.json"
    save_json(result_path, result)
    print(f"result_saved={result_path}")
    print(f"success_rate={metrics['success_rate']:.3f}, avg_reward={metrics['avg_reward']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-reliability ablation experiment")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degradation-level", type=float, default=0.0)
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--imu-history-len", type=int, default=16)
    parser.add_argument(
        "--no-share-features-extractor",
        action="store_true",
        help="Use separate actor/critic feature extractors (slower).",
    )
    parser.add_argument("--output-dir", type=str, default="logs/experiments")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    run_ablation_no_reliability(
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        degradation_level=float(max(0.0, min(1.0, args.degradation_level))),
        difficulty=args.difficulty,
        max_steps=max(1, int(args.max_steps)),
        imu_history_len=max(1, int(args.imu_history_len)),
        share_features_extractor=not args.no_share_features_extractor,
        output_dir=Path(args.output_dir),
        save_model=args.save_model,
    )


if __name__ == "__main__":
    main()
