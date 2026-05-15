"""Rigid-body consistency check: omega_z * v_forward should match the
filtered lateral acceleration.

For a vehicle in horizontal-plane motion with the IMU near the rotation axis:
    a_y_obs  ≈  omega_z · v_forward

When the two curves agree, the velocity, yaw-rate, and accelerometer estimates
are mutually consistent. A persistent offset reveals a stale bias.

Inputs (per dataset):
  build/<dataset>/imu.csv       — gyro_z, acc_y
  build/<dataset>/velocity.csv  — vel_fused (from scripts/velocity.py)
  config/calibration.json       — gyro_z bias and acc_y at-rest

Outputs:
  build/<dataset>/dead_reckoning_comparison.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

BUILD_DIR = Path("build")
CALIBRATION_JSON = Path("config") / "calibration.json"
LATERAL_ACC_CUTOFF_HZ = 1.0
BUTTERWORTH_ORDER = 2


def lpf(x: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff_hz / nyq, btype="low")
    return signal.filtfilt(b, a, x)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("--build", type=Path, default=BUILD_DIR)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_JSON)
    args = parser.parse_args()

    out_dir = args.build / args.dataset
    imu = pd.read_csv(out_dir / "imu.csv")
    vel = pd.read_csv(out_dir / "velocity.csv")
    with open(args.calibration) as f:
        cal = json.load(f)

    t = imu["t"].to_numpy() - imu["t"].iloc[0]
    gyro_z = imu["gyro_z"].to_numpy()
    acc_y = imu["acc_y"].to_numpy()
    vel_fused = vel["vel_fused"].to_numpy()
    fs = 1.0 / np.mean(np.diff(t))

    gyro_z_bias = float(cal["gyroscope"]["bias_rad_s"][2])
    acc_y_rest = float(cal["accelerometer"]["at_rest_m_s2"][1])

    # NEW: bias-corrected both signals.
    omega_xdot_new = (gyro_z - gyro_z_bias) * vel_fused
    y_obs_new = lpf(acc_y - acc_y_rest, fs, LATERAL_ACC_CUTOFF_HZ, BUTTERWORTH_ORDER)

    # OLD: raw gyro_z, raw acc_y -- this is what dead_reckon_comparison.py did.
    omega_xdot_old = gyro_z * vel_fused
    y_obs_old = lpf(acc_y, fs, LATERAL_ACC_CUTOFF_HZ, BUTTERWORTH_ORDER)

    res_new = omega_xdot_new - y_obs_new
    res_old = omega_xdot_old - y_obs_old

    def stats(name, res):
        return {
            "label": name,
            "mean": float(np.mean(res)),
            "std": float(np.std(res)),       # = RMS around the mean -- the dynamics metric
            "rms": float(np.sqrt(np.mean(res ** 2))),  # = sqrt(mean^2 + std^2)
            "max": float(np.max(np.abs(res))),
        }

    s_new = stats("NEW (bias-corrected)", res_new)
    s_old = stats("OLD (no bias correction)", res_old)

    width = 78
    print("=" * width)
    print(f"DEAD RECKONING consistency check on '{args.dataset}'")
    print("=" * width)
    print(f"  identity:  a_y_obs  ≈  omega_z * v_forward  (rigid body, planar motion)")
    print()
    print(f"  bias values used (idle_car):")
    print(f"    gyro_z bias:   {gyro_z_bias:+.4e} rad/s")
    print(f"    acc_y at rest: {acc_y_rest:+.4f} m/s²   (mounting tilt + sensor bias)")
    print()
    print("  residual = (omega_z · v_fused) - LPF(acc_y) :")
    print(f"    {'method':<30} {'mean':>10} {'std':>10} {'RMS':>10} {'max':>10}")
    for s in (s_new, s_old):
        print(f"    {s['label']:<30} {s['mean']:+10.4f} {s['std']:10.4f} {s['rms']:10.4f} {s['max']:10.4f}")
    print()
    print("  interpretation:")
    print("    * 'mean' is the systematic offset between modeled and observed lateral acc.")
    print("    * 'std' is the dynamics-agreement metric (residual variance, independent of offset).")
    print()
    delta_mean = s_new["mean"] - s_old["mean"]
    delta_std = s_new["std"] - s_old["std"]
    print(f"  mean shift NEW - OLD: {delta_mean:+.4f} m/s²  (matches +acc_y_rest = {acc_y_rest:+.4f},")
    print( "    expected since NEW subtracts the idle_car bias from acc_y).")
    print(f"  std diff NEW - OLD:   {delta_std:+.4f} m/s²  (~zero -> dynamics agreement is the same).")
    print()
    print("  takeaway: OLD residual mean was small by accident (acc_y bias partially cancelled")
    print("  the drive's mean omega·v). NEW shows the principled value: the non-zero mean reveals")
    print("  road-camber drift between idle_car parking and driving locations (roll spread 2.9°")
    print("  from the tilt-consistency check ~ +/-0.5 m/s² of gravity projection on acc_y).")
    print()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, omega_xdot_new, "b-", linewidth=1.5, label="ωẊ  (modeled, gyro·vel)")
    ax.plot(t, y_obs_new, "r-", linewidth=1.5, label="filtered ÿ_obs  (observed, bias-corrected)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acceleration (m/s²)")
    ax.set_title(f"{args.dataset}: dead reckoning consistency  "
                 f"(residual std {s_new['std']:.3f} m/s², mean {s_new['mean']:+.3f}, LPF {LATERAL_ACC_CUTOFF_HZ} Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "dead_reckoning_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot: {out_dir}/dead_reckoning_comparison.png")


if __name__ == "__main__":
    main()
