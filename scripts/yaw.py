"""Yaw estimation from magnetometer + gyro + onboard quaternion.

Produces:
  build/<dataset>/yaw.csv                       — all yaw streams for downstream stages
  build/<dataset>/yaw_raw_vs_calibrated.png     — raw mag yaw vs calibrated
  build/<dataset>/yaw_gyro_vs_magnetometer.png  — gyro-integrated vs mag yaw
  build/<dataset>/yaw_complementary_filter.png  — LPF mag + HPF gyro + fused
  build/<dataset>/yaw_four_panel.png            — LPF / HPF / fused / quaternion

Reads calibrated inputs from config/calibration.json (so this must run AFTER
scripts/calibration.py).

For verification, also computes yaw a second time using the previously
hardcoded constants and the old mean-over-drive gyro bias, and reports the
divergence. The VN-100's onboard quaternion (preserved in imu.csv by the
converter) acts as an independent reference: a well-calibrated fused yaw
should track it closely.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, signal
from scipy.spatial.transform import Rotation

BUILD_DIR = Path("build")
CALIBRATION_JSON = Path("config") / "calibration.json"

COMPLEMENTARY_CUTOFF_HZ = 0.1
BUTTERWORTH_ORDER = 2

OLD_HARD_IRON_OFFSET_TESLA = np.array([1.978e-5, 1.289e-5])
OLD_SOFT_IRON_MATRIX = np.array([[1.00017403, -0.00836799],
                                 [-0.00836799, 0.99996603]])


def load_calibration(path: Path = CALIBRATION_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def calibrate_mag(mx: np.ndarray, my: np.ndarray, hard_iron: np.ndarray, soft_iron: np.ndarray) -> np.ndarray:
    raw = np.column_stack([mx, my])
    return (soft_iron @ (raw - hard_iron).T).T


def yaw_from_mag(mag_xy: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(mag_xy[:, 1], mag_xy[:, 0]))


def yaw_from_gyro(gyro_z: np.ndarray, t: np.ndarray, bias: float) -> np.ndarray:
    return integrate.cumulative_trapezoid(gyro_z - bias, t, initial=0)


def complementary_filter(yaw_mag: np.ndarray, yaw_gyro: np.ndarray, t: np.ndarray, cutoff_hz: float):
    fs = 1.0 / np.mean(np.diff(t))
    nyq = 0.5 * fs
    b_lpf, a_lpf = signal.butter(BUTTERWORTH_ORDER, cutoff_hz / nyq, btype="low")
    b_hpf, a_hpf = signal.butter(BUTTERWORTH_ORDER, cutoff_hz / nyq, btype="high")
    yaw_mag_lpf = signal.filtfilt(b_lpf, a_lpf, yaw_mag)
    yaw_gyro_hpf = signal.filtfilt(b_hpf, a_hpf, yaw_gyro)
    return yaw_mag_lpf + yaw_gyro_hpf, yaw_mag_lpf, yaw_gyro_hpf


def yaw_from_quaternion(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.unwrap(Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)[:, 2])


def compute_yaw(t, mx, my, gz, hard_iron, soft_iron, gyro_bias_z, cutoff_hz):
    mag_cal = calibrate_mag(mx, my, hard_iron, soft_iron)
    yaw_mag = yaw_from_mag(mag_cal)
    yaw_gyro = yaw_from_gyro(gz, t, gyro_bias_z)
    yaw_gyro = yaw_gyro - yaw_gyro[0] + yaw_mag[0]
    yaw_fused, yaw_mag_lpf, yaw_gyro_hpf = complementary_filter(yaw_mag, yaw_gyro, t, cutoff_hz)
    return {
        "yaw_mag": yaw_mag,
        "yaw_gyro": yaw_gyro,
        "yaw_fused": yaw_fused,
        "yaw_mag_lpf": yaw_mag_lpf,
        "yaw_gyro_hpf": yaw_gyro_hpf,
        "mag_cal": mag_cal,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="Dataset name (subdirectory of build/)")
    parser.add_argument("--build", type=Path, default=BUILD_DIR)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_JSON)
    args = parser.parse_args()

    out_dir = args.build / args.dataset
    imu = pd.read_csv(out_dir / "imu.csv")
    cal = load_calibration(args.calibration)

    t_abs = imu["t"].to_numpy()
    t = t_abs - t_abs[0]
    mx, my = imu["mag_x"].to_numpy(), imu["mag_y"].to_numpy()
    gz = imu["gyro_z"].to_numpy()
    quat = imu[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy()

    # NEW: calibration from config/calibration.json + idle_car gyro bias
    hard_iron_new = np.array(cal["magnetometer"]["hard_iron_offset_tesla_xy"])
    soft_iron_new = np.array(cal["magnetometer"]["soft_iron_matrix_2d"])
    gyro_bias_z_new = float(cal["gyroscope"]["bias_rad_s"][2])
    new = compute_yaw(t, mx, my, gz, hard_iron_new, soft_iron_new, gyro_bias_z_new, COMPLEMENTARY_CUTOFF_HZ)

    # OLD: hardcoded constants + mean(gyro_z) over the entire drive (the old bug)
    gyro_bias_z_old = float(np.mean(gz))
    old = compute_yaw(t, mx, my, gz, OLD_HARD_IRON_OFFSET_TESLA, OLD_SOFT_IRON_MATRIX, gyro_bias_z_old, COMPLEMENTARY_CUTOFF_HZ)

    # Raw magnetometer yaw (no calibration), purely for the first plot
    yaw_mag_raw = np.unwrap(np.arctan2(my, mx))

    # Independent reference: onboard quaternion
    yaw_quat = yaw_from_quaternion(quat)
    yaw_quat_aligned = yaw_quat - yaw_quat[0] + new["yaw_mag"][0]

    # ---- Numeric comparison report ----
    def rad_deg(x):
        return f"{x:.4f} rad ({np.degrees(x):+.2f}°)"

    width = 78
    print("=" * width)
    print(f"YAW stage on '{args.dataset}'")
    print("=" * width)
    print(f"  samples: {len(t)}  duration: {t[-1]:.2f} s")
    print()
    print("  yaw_mag (calibrated magnetometer):  NEW vs OLD calibration matrices")
    diff = new["yaw_mag"] - old["yaw_mag"]
    print(f"    max|diff|: {rad_deg(np.max(np.abs(diff)))}")
    print(f"    RMS|diff|: {rad_deg(np.sqrt(np.mean(diff**2)))}")
    print()
    print("  yaw_gyro (integrated):              NEW (idle_car bias) vs OLD (mean over drive)")
    print(f"    new gyro_z bias (rad/s): {gyro_bias_z_new:+.4e}  source: idle_car")
    print(f"    old gyro_z bias (rad/s): {gyro_bias_z_old:+.4e}  source: mean(gyro_z) over drive")
    drift = (new["yaw_gyro"][-1] - new["yaw_mag"][0]) - (old["yaw_gyro"][-1] - old["yaw_mag"][0])
    print(f"    drift difference at end of run: {rad_deg(drift)}")
    print(f"    (this is the integrated effect of the bias-correction fix)")
    print()
    print("  yaw_fused (complementary filter):   NEW vs OLD")
    diff = new["yaw_fused"] - old["yaw_fused"]
    print(f"    max|diff|: {rad_deg(np.max(np.abs(diff)))}")
    print(f"    RMS|diff|: {rad_deg(np.sqrt(np.mean(diff**2)))}")
    print()
    # The VN-100's onboard quaternion uses the OPPOSITE sign convention to our
    # magnetometer yaw (atan2(my, mx) decreases when the body turns CCW; the
    # quaternion's z-Euler angle increases with CW NED yaw). The old pipeline
    # had the same convention mismatch -- our yaw_quat values are bit-identical
    # to the previously committed driving_data/imu_heading_data.csv.
    yaw_quat_signed = -(yaw_quat - yaw_quat[0]) + new["yaw_mag"][0]
    diff_signed = new["yaw_fused"] - yaw_quat_signed
    print("  yaw_fused (NEW) vs onboard quaternion (sign-corrected for opposite convention):")
    print(f"    max|diff|: {rad_deg(np.max(np.abs(diff_signed)))}")
    print(f"    RMS|diff|: {rad_deg(np.sqrt(np.mean(diff_signed**2)))}")
    print(f"    (raw quat goes from {yaw_quat[0]:+.2f} to {yaw_quat[-1]:+.2f} rad; mag yaw goes from "
          f"{new['yaw_mag'][0]:+.2f} to {new['yaw_mag'][-1]:+.2f}; sign-flipping for comparison only)")
    print()

    # ---- Save yaw streams ----
    yaw_df = pd.DataFrame({
        "t": t_abs,
        "yaw_mag_raw": yaw_mag_raw,
        "yaw_mag_cal": new["yaw_mag"],
        "yaw_gyro": new["yaw_gyro"],
        "yaw_mag_lpf": new["yaw_mag_lpf"],
        "yaw_gyro_hpf": new["yaw_gyro_hpf"],
        "yaw_fused": new["yaw_fused"],
        "yaw_quat": yaw_quat,
    })
    yaw_csv_path = out_dir / "yaw.csv"
    yaw_df.to_csv(yaw_csv_path, index=False)
    print(f"  yaw streams: {yaw_csv_path}")

    # ---- Plot 1: raw vs calibrated magnetometer yaw ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, yaw_mag_raw, "b-", linewidth=1, label="raw magnetometer yaw")
    ax.plot(t, new["yaw_mag"], "r-", linewidth=1, label="calibrated magnetometer yaw")
    ax.set_xlabel("time (s)"); ax.set_ylabel("yaw (rad)")
    ax.set_title(f"{args.dataset}: magnetometer yaw comparison")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "yaw_raw_vs_calibrated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: gyro vs magnetometer yaw ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, new["yaw_mag"], "r-", linewidth=1, label="calibrated magnetometer yaw")
    ax.plot(t, new["yaw_gyro"], "b-", linewidth=1, label="gyro yaw (integrated, idle_car bias)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("yaw (rad)")
    ax.set_title(f"{args.dataset}: gyro vs magnetometer yaw")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "yaw_gyro_vs_magnetometer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 3: complementary filter (filtered components + fused result) ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(t, new["yaw_mag_lpf"], "b-", linewidth=1.5, alpha=0.7,
                 label=f"low-pass filter (magnetometer) @ {COMPLEMENTARY_CUTOFF_HZ} Hz")
    axes[0].plot(t, new["yaw_gyro_hpf"], "r-", linewidth=1.5, alpha=0.7,
                 label=f"high-pass filter (gyroscope) @ {COMPLEMENTARY_CUTOFF_HZ} Hz")
    axes[0].set_ylabel("yaw (rad)")
    axes[0].set_title(f"filtered components (cutoff = {COMPLEMENTARY_CUTOFF_HZ} Hz)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, new["yaw_mag"], "b-", linewidth=1, alpha=0.4, label="calibrated magnetometer yaw")
    axes[1].plot(t, new["yaw_gyro"], "r-", linewidth=1, alpha=0.4, label="integrated gyro yaw")
    axes[1].plot(t, new["yaw_fused"], "g-", linewidth=2, label="fused yaw (complementary filter)")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("yaw (rad)")
    axes[1].set_title("complementary filter result")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "yaw_complementary_filter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 4: four-panel comparison ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    axes[0].plot(t, new["yaw_mag_lpf"], "b-", linewidth=1.5)
    axes[0].set_ylabel("yaw (rad)"); axes[0].set_title("low-pass filter (magnetometer)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, new["yaw_gyro_hpf"], "r-", linewidth=1.5)
    axes[1].set_ylabel("yaw (rad)"); axes[1].set_title("high-pass filter (gyroscope)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, new["yaw_fused"], "g-", linewidth=2)
    axes[2].set_ylabel("yaw (rad)"); axes[2].set_title("complementary filter output")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(t, yaw_quat_aligned, color="purple", linewidth=1.5)
    axes[3].set_xlabel("time (s)"); axes[3].set_ylabel("yaw (rad)")
    axes[3].set_title("IMU onboard quaternion yaw (independent reference)")
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "yaw_four_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  plots:       {out_dir}/yaw_*.png  (4 figures)")


if __name__ == "__main__":
    main()
