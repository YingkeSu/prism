"""
Check whether reliability/quality scores respond to sensor degradation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks.reliability_predictor import ReliabilityPredictor


def _to_batch_tensors(
    obs: Dict[str, np.ndarray],
    imu_seq_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lidar = torch.from_numpy(obs["lidar"]).float().unsqueeze(0)  # (1, N, 3)
    rgb = torch.from_numpy(obs["rgb"]).float().permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    imu_seq = torch.from_numpy(imu_seq_np).float().unsqueeze(0)  # (1, 100, 6)
    return lidar, rgb, imu_seq


def _corr(levels: List[float], values: List[float]) -> float:
    if len(levels) < 2:
        return 0.0
    x = np.asarray(levels, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if float(np.std(y)) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check reliability response to degradation")
    parser.add_argument("--model", type=str, default=None, help="Optional SAC model path")
    parser.add_argument("--levels", type=str, default="0.0,0.2,0.5,0.8")
    parser.add_argument("--samples-per-level", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="logs/experiments/reliability_response.json")
    args = parser.parse_args()

    levels = [float(x.strip()) for x in args.levels.split(",") if x.strip()]
    samples = int(max(1, args.samples_per_level))

    predictor = None
    if args.model:
        from stable_baselines3 import SAC

        model = SAC.load(args.model)
        extractor = model.policy.actor.features_extractor
        if not hasattr(extractor, "reliability_fusion"):
            raise RuntimeError("Loaded model extractor has no reliability_fusion module")
        predictor = extractor.reliability_fusion.reliability_estimator.eval()
        print(f"Using reliability estimator from model: {args.model}")
    else:
        predictor = ReliabilityPredictor().eval()
        print("Using fresh ReliabilityPredictor (untrained)")

    rows = []
    lidar_quality_values = []
    rgb_quality_values = []
    imu_quality_values = []
    r_lidar_values = []
    r_rgb_values = []
    r_imu_values = []

    with torch.no_grad():
        for level in levels:
            env = UAVMultimodalEnv(max_steps=200, degradation_level=level)
            obs, _ = env.reset(seed=args.seed)
            imu_history = [obs["imu"].copy() for _ in range(100)]

            lidar_q = []
            rgb_q = []
            imu_q = []
            r_lidar = []
            r_rgb = []
            r_imu = []

            for _ in range(samples):
                imu_seq_np = np.stack(imu_history[-100:], axis=0)
                lidar_t, rgb_t, imu_seq_t = _to_batch_tensors(obs, imu_seq_np)

                # Deterministic quality proxies from estimator internals
                lidar_metrics = predictor.lidar_estimator.compute_traditional_metrics(lidar_t)
                imu_metrics = predictor.imu_estimator.compute_traditional_metrics(imu_seq_t)
                rgb_metrics = predictor.rgb_estimator(rgb_t)
                rgb_proxy = (
                    rgb_metrics["sharpness"] +
                    rgb_metrics["contrast"] +
                    rgb_metrics["brightness"] +
                    rgb_metrics["texture"]
                ) / 4.0

                pred = predictor(lidar_t, rgb_t, imu_seq_t)
                lidar_q.append(float(lidar_metrics["snr"].mean().item()))
                rgb_q.append(float(rgb_proxy.mean().item()))
                imu_q.append(float(imu_metrics["consistency"].mean().item()))
                r_lidar.append(float(pred["r_lidar"].mean().item()))
                r_rgb.append(float(pred["r_rgb"].mean().item()))
                r_imu.append(float(pred["r_imu"].mean().item()))

                action = env.action_space.sample()
                obs, _, done, truncated, _ = env.step(action)
                imu_history.append(obs["imu"].copy())
                if done or truncated:
                    obs, _ = env.reset()
                    imu_history = [obs["imu"].copy() for _ in range(100)]

            row = {
                "degradation_level": level,
                "lidar_quality_proxy": float(np.mean(lidar_q)),
                "rgb_quality_proxy": float(np.mean(rgb_q)),
                "imu_quality_proxy": float(np.mean(imu_q)),
                "r_lidar": float(np.mean(r_lidar)),
                "r_rgb": float(np.mean(r_rgb)),
                "r_imu": float(np.mean(r_imu)),
            }
            rows.append(row)
            lidar_quality_values.append(row["lidar_quality_proxy"])
            rgb_quality_values.append(row["rgb_quality_proxy"])
            imu_quality_values.append(row["imu_quality_proxy"])
            r_lidar_values.append(row["r_lidar"])
            r_rgb_values.append(row["r_rgb"])
            r_imu_values.append(row["r_imu"])

    summary = {
        "levels": levels,
        "samples_per_level": samples,
        "source_model": args.model,
        "rows": rows,
        "correlation_with_degradation": {
            "lidar_quality_proxy": _corr(levels, lidar_quality_values),
            "rgb_quality_proxy": _corr(levels, rgb_quality_values),
            "imu_quality_proxy": _corr(levels, imu_quality_values),
            "r_lidar": _corr(levels, r_lidar_values),
            "r_rgb": _corr(levels, r_rgb_values),
            "r_imu": _corr(levels, r_imu_values),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDegradation response summary:")
    for row in rows:
        print(
            f"level={row['degradation_level']:.2f} | "
            f"lidar_q={row['lidar_quality_proxy']:.3f} rgb_q={row['rgb_quality_proxy']:.3f} imu_q={row['imu_quality_proxy']:.3f} | "
            f"r_lidar={row['r_lidar']:.3f} r_rgb={row['r_rgb']:.3f} r_imu={row['r_imu']:.3f}"
        )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
