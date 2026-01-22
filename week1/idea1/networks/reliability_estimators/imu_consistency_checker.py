"""
IMU Data Consistency Checker

Checks IMU data quality:
- Drift (accelerometer/gyroscope)
- Velocity anomaly detection
- Orientation consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class IMUConsistencyChecker(nn.Module):
    """
    IMU Data Consistency Checker

    Checks IMU data quality:
    - Drift (accelerometer/gyroscope)
    - Velocity anomaly detection
    - Orientation consistency

    Args:
        imu_dim: IMU data dimension (default 6: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        window_size: Sliding window size (default 100)
        feature_dim: Feature dimension (default 64)

    Input:
        imu_sequence: (B, T, 6) IMU sequence
                       B: batch_size
                       T: sequence_length
                       6: (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

    Output:
        Dict: {
            'drift_score': (B, 1),       # Drift score [0, 1], 0=no drift
            'velocity_anomaly': (B, 1),   # Velocity anomaly [0, 1], 0=no anomaly
            'consistency': (B, 1),        # Consistency [0, 1], 1=fully consistent
            'features': (B, feature_dim)  # Extracted features
        }
    """

    def __init__(self, imu_dim: int = 6, window_size: int = 100, feature_dim: int = 128):
        super().__init__()
        self.imu_dim = imu_dim
        self.window_size = window_size
        self.feature_dim = feature_dim

        # Drift analyzer (using 1D CNN for time series)
        self.drift_analyzer = nn.Sequential(
            nn.Conv1d(imu_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        # Velocity anomaly detector
        self.velocity_anomaly_detector = nn.Sequential(
            nn.Conv1d(imu_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        # Drift prediction head
        self.drift_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Velocity anomaly prediction head
        self.anomaly_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Time series feature extraction (LSTM)
        self.lstm = nn.LSTM(imu_dim, 64, batch_first=True, bidirectional=True)
        self.lstm_fc = nn.Linear(128, feature_dim)

    def forward(self, imu_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            imu_sequence: (B, T, 6) IMU sequence

        Returns:
            Dict: Consistency metrics
        """
        B, T, _ = imu_sequence.shape

        # Transpose: (B, T, 6) -> (B, 6, T) for Conv1d
        x = imu_sequence.transpose(1, 2)

        # Pad/truncate to window_size
        if T < self.window_size:
            padding = torch.zeros(B, 6, self.window_size - T)
            x = torch.cat([padding, x], dim=2)
        else:
            x = x[:, :, -self.window_size:]

        # Analyze drift
        drift_features = self.drift_analyzer(x)  # (B, 128, 1) -> (B, 128, 1) -> (B, 128)
        drift_features = drift_features.squeeze(-1)  # (B, 128)
        drift_score = self.drift_head(drift_features)  # (B, 1)

        # Analyze velocity anomaly
        anomaly_features = self.velocity_anomaly_detector(x)  # (B, 128, 1) -> (B, 128, 1) -> (B, 128)
        anomaly_features = anomaly_features.squeeze(-1)  # (B, 128)
        velocity_anomaly = self.anomaly_head(anomaly_features)  # (B, 1)

        # Time series feature extraction (LSTM)
        lstm_out, (h_n, c_n) = self.lstm(imu_sequence)
        # lstm_out: (B, T, 128)

        # Use last timestep features
        lstm_features = lstm_out[:, -1, :]  # (B, 128)
        features = self.lstm_fc(lstm_features)  # (B, feature_dim)

        # Compute consistency: 1 - drift - anomaly
        consistency = 1.0 - 0.5 * drift_score - 0.5 * velocity_anomaly
        consistency = torch.clamp(consistency, 0.0, 1.0)

        return {
            'drift_score': drift_score,
            'velocity_anomaly': velocity_anomaly,
            'consistency': consistency,
            'features': features
        }

    def compute_traditional_metrics(self, imu_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute traditional IMU consistency metrics (for validation)

        Args:
            imu_sequence: (B, T, 6)

        Returns:
            Dict: Traditional metrics
        """
        # Accelerometer mean (should be close to [0, 0, -9.81] when stationary)
        acc_mean = imu_sequence[:, :, :3].mean(dim=1)  # (B, 3)
        acc_std = imu_sequence[:, :, :3].std(dim=1)  # (B, 3)

        # Gyroscope mean (should be close to [0, 0, 0] when stationary)
        gyro_mean = imu_sequence[:, :, 3:].mean(dim=1)  # (B, 3)
        gyro_std = imu_sequence[:, :, 3:].std(dim=1)  # (B, 3)

        # Drift score (higher std = more severe drift)
        acc_drift = acc_std.mean(dim=1, keepdim=True)  # (B, 1)
        gyro_drift = gyro_std.mean(dim=1, keepdim=True)  # (B, 1)
        total_drift = (acc_drift + gyro_drift) / 2.0  # (B, 1)

        # Normalize to [0, 1] (using sigmoid to reverse)
        consistency = torch.sigmoid(-total_drift * 10.0)

        return {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'gyro_mean': gyro_mean,
            'gyro_std': gyro_std,
            'drift': total_drift,
            'consistency': consistency
        }


def test_imu_consistency_checker():
    """Test IMU Consistency Checker"""
    model = IMUConsistencyChecker(imu_dim=6, window_size=100)

    # Create test data
    batch_size = 4
    seq_len = 100
    imu_sequence = torch.randn(batch_size, seq_len, 6)

    # Simulate different quality
    # High quality: low noise
    imu_sequence[0] = torch.randn(seq_len, 6) * 0.01
    # Medium quality
    imu_sequence[1] = torch.randn(seq_len, 6) * 0.1
    # Low quality: high noise
    imu_sequence[2] = torch.randn(seq_len, 6) * 1.0
    # Has drift
    imu_sequence[3] = torch.randn(seq_len, 6) * 0.1
    imu_sequence[3, :, :3] += torch.linspace(0, 1, seq_len).unsqueeze(1)  # Accelerometer drift

    # Forward pass
    output = model(imu_sequence)

    # Verify output dimensions
    assert output['drift_score'].shape == (batch_size, 1)
    assert output['velocity_anomaly'].shape == (batch_size, 1)
    assert output['consistency'].shape == (batch_size, 1)
    assert output['features'].shape == (batch_size, 64)

    # Verify output range
    assert torch.all(output['drift_score'] >= 0) and torch.all(output['drift_score'] <= 1)
    assert torch.all(output['velocity_anomaly'] >= 0) and torch.all(output['velocity_anomaly'] <= 1)
    assert torch.all(output['consistency'] >= 0) and torch.all(output['consistency'] <= 1)

    # High quality IMU should have higher consistency
    assert output['consistency'][0] > output['consistency'][2], "Low noise IMU should have higher consistency"
    # No drift IMU should have higher consistency
    assert output['consistency'][0] > output['consistency'][3], "No drift IMU should have higher consistency"

    print("✅ IMU Consistency Checker test passed")
    print(f"High quality IMU consistency: {output['consistency'][0].item():.3f}")
    print(f"Low quality IMU consistency: {output['consistency'][2].item():.3f}")
    print(f"Drift IMU consistency: {output['consistency'][3].item():.3f}")


if __name__ == "__main__":
    test_imu_consistency_checker()
