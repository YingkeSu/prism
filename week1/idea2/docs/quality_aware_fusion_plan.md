# Quality-Aware Fusion Plan (Idea2 / ULW)

**Created**: 2026-02-04

This document turns `质量感知注意力的动态融合.md` into an implementable plan aligned with the current `week1/idea2` repository layout.

## 0. Repo Reality Check (as-is)

- Existing reliability estimators:
  - `networks/reliability_estimators/lidar_snr_estimator.py` → `LiDARSNREstimator` (outputs: `snr`, `density`, `uniformity`, `features` (B, 64))
  - `networks/reliability_estimators/image_quality_estimator.py` → `ImageQualityEstimator` (outputs: multiple metrics + `overall_quality`, `features` (B, 256))
- Existing dynamic weighting:
  - `networks/dynamic_weighting_layer.py` → `DynamicWeightingLayer` (inputs: (B, D) x3; outputs weights + attention tensors)
- Note (blocker): current repo does NOT contain `envs/uav_multimodal_env.py` or `networks/uav_multimodal_extractor.py`, but several scripts import them (e.g. `tests/integration/test_basic_setup.py`).

## 1. Deliverables (exact)

New modules (per target structure in the technical note):

1. `networks/temporal_quality_predictor.py`
2. `networks/quality_aware_attention.py`
3. `networks/quality_aware_fusion.py`

Planned tests (minimal, aligned to existing style):

- Add module-local `test_*()` functions + `if __name__ == "__main__":` blocks (matches existing `networks/*.py` pattern).
- Optionally later: create `tests/unit/` pytest-style tests (mentioned in `docs/CODE_STYLE.md`), but current repo tests are script-style.

## 2. Module Specs (interfaces + shapes)

### 2.1 TemporalQualityPredictor

Goal: predict next-step modality quality from a fixed-length history.

Inputs:
- `historical_quality`: `torch.Tensor` of shape `(B, T, Q)`
  - `T = seq_len` (default 10)
  - `Q = 3` (lidar, rgb, imu) using a single scalar quality per modality (normalized to [0, 1])

Outputs:
- `pred_quality`: `torch.Tensor` of shape `(B, Q)` in [0, 1]
  - recommended key format: `{'q_lidar': (B,1), 'q_rgb': (B,1), 'q_imu': (B,1)}` to mirror other estimators

Notes (implementation-facing):
- Keep LSTM `batch_first=True` for consistency with `DynamicWeightingLayer`.
- A single shared LSTM + small per-modality heads is sufficient.

### 2.2 QualityAwareAttention

Goal: produce a gate that suppresses low-quality modalities so they cannot dominate fusion.

Inputs:
- `quality_now`: `torch.Tensor` of shape `(B, Q)` in [0, 1]
- `quality_pred`: optional `torch.Tensor` of shape `(B, Q)` in [0, 1]

Outputs:
- `gate`: `torch.Tensor` of shape `(B, Q)` in [0, 1]

Simplest valid rule (learnable):
- MLP over concatenated qualities → sigmoid → per-modality gate.

### 2.3 QualityAwareDynamicFusion (quality_aware_fusion.py)

Goal: orchestrate reliability estimation, temporal prediction, quality-aware gating, and dynamic weighting.

Inputs (minimal, decoupled from SB3 extractor until those files exist):
- `lidar_points`: `(B, N, 3)`
- `rgb_image`: `(B, 3, H, W)`
- `imu_data`: `(B, 6)` (or a project-defined IMU dim)
- `historical_quality`: optional `(B, T, Q)`

Intermediate representations:
- `lidar_rel = LiDARSNREstimator(lidar_points)` → includes `features` `(B, 64)` and quality proxy `snr` `(B,1)`
- `rgb_rel = ImageQualityEstimator(rgb_image)` → includes `features` `(B, 256)` and `overall_quality` `(B,1)`
- `imu_quality`: simplest scalar quality `(B,1)` computed via a lightweight IMU consistency check (define in this module unless a checker already exists)

Feature alignment (required because modalities differ in feature dim):
- Project to a common `feature_dim` D (default 256):
  - `lidar_proj: (B,64) -> (B,D)`
  - `rgb_proj: (B,256) -> (B,D)` (identity if D=256)
  - `imu_proj: (B,imu_dim) -> (B,D)`

Fusion:
- `raw_weights = DynamicWeightingLayer(D)(lidar_feat, rgb_feat, imu_feat)` → yields `(B,1)` per modality that sums to 1.
- `gate = QualityAwareAttention(quality_now, quality_pred)` → `(B,Q)`.
- `gated_weights = normalize(raw_weights * gate)` → ensures sum to 1 while suppressing low-quality modalities.

Outputs:
- `weights`: `Dict[str, torch.Tensor]` with `w_lidar`, `w_rgb`, `w_imu` each `(B,1)`
- `debug`: optional extra tensors (`quality_now`, `quality_pred`, `gate`, `attention_weights`) for logging/visualization

## 3. Implementation Plan (ordered, minimal)

1. Define a single-scalar quality per modality (Q=3) and how to extract it:
   - lidar: use `snr` from `LiDARSNREstimator`
   - rgb: use `overall_quality` from `ImageQualityEstimator`
   - imu: define a simple consistency-based quality score (bounded to [0,1])
2. Implement `TemporalQualityPredictor` with LSTM over `(B,T,Q)` → `(B,Q)`.
3. Implement `QualityAwareAttention` producing `gate (B,Q)` from current/predicted qualities.
4. Implement `QualityAwareDynamicFusion` that:
   - calls reliability estimators
   - projects features to common D
   - calls `DynamicWeightingLayer`
   - applies gating + renormalization
5. Add module-local `test_*()` functions that only validate:
   - tensor shapes
   - value ranges ([0,1])
   - gated weights sum to 1

## 4. Acceptance Criteria

- Forward pass runs on random tensors with shapes documented above.
- Output weights are in [0, 1] and sum to 1 per batch element.
- If one modality quality is forced near 0, its gated weight decreases (relative to others) without producing NaNs.

## 5. Known Risks / Blockers

- Missing `envs/uav_multimodal_env.py` and `networks/uav_multimodal_extractor.py` prevents end-to-end SB3 integration tests from running as-is.
- The technical note code blocks in `质量感知注意力的动态融合.md` are conceptual and not directly runnable (e.g., `nn.Linear(..., activation='sigmoid')`). Implementation should follow PyTorch APIs.
