import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
INPUT_CSV = 'driving_data_0_imu.csv'
OUTPUT_FIGURE = 'yaw_raw_vs_calibrated.png'

# Calibration parameters from circle_data/circle_calibration.py (Tesla units)
HARD_IRON_OFFSET = np.array([0.00001978, 0.00001289])
SOFT_IRON_MATRIX = np.array([[1.00017403, -0.00836799],
                             [-0.00836799, 0.99996603]])

# Reading driving data from csv
df = pd.read_csv(INPUT_CSV)

# Extracting magnetometer data
mag_x = df['mag_x'].values
mag_y = df['mag_y'].values
timestamps = df['t'].values

# Applying calibration
mag_raw = np.column_stack([mag_x, mag_y])
mag_centered = mag_raw - HARD_IRON_OFFSET
mag_calibrated = (SOFT_IRON_MATRIX @ mag_centered.T).T

# Calculating yaw angles in radians
yaw_raw = np.arctan2(mag_y, mag_x)
yaw_calibrated = np.arctan2(mag_calibrated[:, 1], mag_calibrated[:, 0])

# Unwrapping angles for better plots
yaw_raw_unwrapped = np.unwrap(yaw_raw)
yaw_calibrated_unwrapped = np.unwrap(yaw_calibrated)

# Normalising timestamps to start from 0
timestamps = timestamps - timestamps[0]

print(f"raw yaw range: {yaw_raw_unwrapped.min():.2f} to {yaw_raw_unwrapped.max():.2f} radians")
print(f"calibrated yaw range: {yaw_calibrated_unwrapped.min():.2f} to {yaw_calibrated_unwrapped.max():.2f} radians")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(timestamps, yaw_raw_unwrapped, 'b-', linewidth=1, label='raw magnetometer yaw')
plt.plot(timestamps, yaw_calibrated_unwrapped, 'r-', linewidth=1, label='calibrated magnetometer yaw')
plt.xlabel('time (s)')
plt.ylabel('yaw (rad)')
plt.title('magnetometer yaw comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')

print(f"\nplot saved as '{OUTPUT_FIGURE}'")
