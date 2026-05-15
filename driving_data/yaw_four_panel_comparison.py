import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, signal

# Constants
INPUT_CSV = 'driving_data_0_imu.csv'
IMU_HEADING_CSV = 'imu_heading_data.csv'
OUTPUT_FIGURE = 'yaw_four_panel_comparison.png'
BUTTERWORTH_ORDER = 2
COMPLEMENTARY_CUTOFF_HZ = 0.1

# Calibration parameters from circle_data/circle_calibration.py (Tesla units)
HARD_IRON_OFFSET = np.array([0.00001978, 0.00001289])
SOFT_IRON_MATRIX = np.array([[1.00017403, -0.00836799],
                             [-0.00836799, 0.99996603]])

# Reading driving data from csv file
df = pd.read_csv(INPUT_CSV)

mag_x = df['mag_x'].values
mag_y = df['mag_y'].values
gyro_z = df['gyro_z'].values
timestamps = df['t'].values
timestamps = timestamps - timestamps[0]

# Reading IMU heading data
df_imu_heading = pd.read_csv(IMU_HEADING_CSV)
imu_yaw = df_imu_heading['imu_yaw'].values
imu_time = df_imu_heading['time'].values
imu_time = imu_time - imu_time[0]

# Gyro bias correction
gyro_bias = np.mean(gyro_z)
gyro_z_corrected = gyro_z - gyro_bias

# Magnetometer calibration
mag_raw = np.column_stack([mag_x, mag_y])
mag_centered = mag_raw - HARD_IRON_OFFSET
mag_calibrated = (SOFT_IRON_MATRIX @ mag_centered.T).T

# Calculating calibrated yaw from magnetometer
yaw_mag = np.arctan2(mag_calibrated[:, 1], mag_calibrated[:, 0])
yaw_mag_unwrapped = np.unwrap(yaw_mag)

# Integrating gyro to get yaw
yaw_gyro = integrate.cumulative_trapezoid(gyro_z_corrected, timestamps, initial=0)
yaw_gyro = yaw_gyro - yaw_gyro[0] + yaw_mag_unwrapped[0]

# Complementary filter
fs = 1 / np.mean(np.diff(timestamps))
nyq = 0.5 * fs

# Low pass filter for magnetometer
lpf_norm = COMPLEMENTARY_CUTOFF_HZ / nyq
b_lpf, a_lpf = signal.butter(BUTTERWORTH_ORDER, lpf_norm, btype='low')
yaw_mag_lpf = signal.filtfilt(b_lpf, a_lpf, yaw_mag_unwrapped)

# High pass filter for gyroscope
hpf_norm = COMPLEMENTARY_CUTOFF_HZ / nyq
b_hpf, a_hpf = signal.butter(BUTTERWORTH_ORDER, hpf_norm, btype='high')
yaw_gyro_hpf = signal.filtfilt(b_hpf, a_hpf, yaw_gyro)

# Fused yaw
yaw_fused = yaw_mag_lpf + yaw_gyro_hpf

# Aligning IMU yaw with fused yaw
imu_yaw_aligned = imu_yaw - imu_yaw[0] + yaw_fused[0]

print(f"cutoff frequency: {COMPLEMENTARY_CUTOFF_HZ} Hz")

# Creating 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# Subplot 1: low pass filter (magnetometer)
axes[0].plot(timestamps, yaw_mag_lpf, 'b-', linewidth=1.5)
axes[0].set_ylabel('yaw (radians)')
axes[0].set_title('low-pass filter (magnetometer)')
axes[0].grid(True, alpha=0.3)

# Subplot 2: high pass filter (gyroscope)
axes[1].plot(timestamps, yaw_gyro_hpf, 'r-', linewidth=1.5)
axes[1].set_ylabel('yaw (radians)')
axes[1].set_title('high-pass filter (gyroscope)')
axes[1].grid(True, alpha=0.3)

# Subplot 3: complementary filter output
axes[2].plot(timestamps, yaw_fused, 'g-', linewidth=2)
axes[2].set_ylabel('yaw (radians)')
axes[2].set_title('complementary filter output')
axes[2].grid(True, alpha=0.3)

# Subplot 4: IMU heading estimate
axes[3].plot(imu_time, imu_yaw_aligned, 'purple', linewidth=1.5)
axes[3].set_xlabel('time (s)')
axes[3].set_ylabel('yaw (radians)')
axes[3].set_title('IMU heading estimate')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')

print(f"\nplot saved as '{OUTPUT_FIGURE}'")
