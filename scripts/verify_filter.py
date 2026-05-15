"""Verify a dataset's yaw filters against expected behavior.

Runs scripts/yaw.py (and scripts/velocity.py if --motion) on a dataset and
reports the std (centered) of every yaw stream. The interpretation depends
on the dataset:

  STATIONARY datasets (idle_car, engine_on): true yaw is constant, so each
  filter output's std measures the filter's own residual uncertainty. Smaller
  std = the filter is better-calibrated against the noise. Useful for:
    * detecting if a tuning change broke the filter (std should remain small)
    * comparing two filters on the same data (RTS vs forward Kalman, etc.)

  MOVING datasets: the std reflects real heading change AND filter noise. Not
  directly informative without ground truth -- use GPS course where available.

Usage:
  scripts/verify_filter.py idle_car
  scripts/verify_filter.py engine_on
  scripts/verify_filter.py driving_data
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--skip-rerun", action="store_true",
                        help="Use existing build/<dataset>/yaw.csv; do not regenerate")
    args = parser.parse_args()

    out_dir = args.build / args.dataset
    yaw_csv = out_dir / "yaw.csv"

    if not args.skip_rerun:
        # bag_to_csv if needed
        if not (out_dir / "imu.csv").exists():
            print(f"converting bag for {args.dataset}...")
            subprocess.run([sys.executable, "scripts/bag_to_csv.py",
                            f"data/{args.dataset}"], check=True)
        print(f"running yaw on {args.dataset}...")
        subprocess.run([sys.executable, "scripts/yaw.py", args.dataset],
                        check=True, capture_output=True)

    y = pd.read_csv(yaw_csv)
    duration_s = float(y["t"].iloc[-1] - y["t"].iloc[0])

    # Heuristic: stationary if the cumulative GPS path length is short. Start-to-end
    # displacement isn't enough (a round trip can return to origin with significant
    # path length in between).
    gps = pd.read_csv(out_dir / "gps.csv")
    seg = np.sqrt(np.diff(gps["utm_easting"]) ** 2 + np.diff(gps["utm_northing"]) ** 2)
    total_path_m = float(seg.sum())
    is_stationary = total_path_m < 50.0

    width = 78
    print("=" * width)
    print(f"FILTER VERIFICATION  on '{args.dataset}'")
    print("=" * width)
    print(f"  duration:           {duration_s:.1f} s")
    print(f"  GPS path length:    {total_path_m:.2f} m  -> "
          f"{'STATIONARY' if is_stationary else 'MOVING'}")
    print()

    if is_stationary:
        print("  Interpretation: true yaw is constant; each filter's std (after centering")
        print("  to its own mean) is the filter's residual uncertainty. Smaller = better-")
        print("  tuned. This isolates the *filter's* behavior from real motion.")
    else:
        print("  Interpretation: dataset contains motion. yaw std is a mix of real")
        print("  heading change and filter behavior; not directly informative.")
        print("  Use scripts/trajectory.py for end-to-end accuracy against GPS ground truth.")
    print()

    streams = ["yaw_mag_cal", "yaw_fused", "yaw_quat", "yaw_gps", "yaw_kalman", "yaw_rts"]
    print(f"  {'yaw stream':<14} {'std (deg)':>12} {'range (deg)':>14}  {'comment':<40}")
    comments = {
        "yaw_mag_cal": "raw calibrated magnetometer (no filter)",
        "yaw_fused":   "LPF(mag) + HPF(gyro) at 0.1 Hz",
        "yaw_quat":    "VN-100 onboard quaternion",
        "yaw_gps":     "LPF(GPS course) + HPF(gyro) at 0.5 Hz",
        "yaw_kalman":  "forward Kalman (mag + GPS)",
        "yaw_rts":     "Kalman + RTS backward smoother",
    }
    best_stream, best_std = None, float("inf")
    for col in streams:
        deg = np.degrees(y[col].values - y[col].mean())
        std_deg = float(deg.std())
        rng_deg = float(deg.max() - deg.min())
        flag = ""
        if is_stationary and col != "yaw_mag_cal":
            if std_deg < best_std:
                best_std = std_deg
                best_stream = col
        print(f"  {col:<14} {std_deg:>12.4f} {rng_deg:>14.4f}  {comments[col]:<40}")
    print()

    if is_stationary and best_stream is not None:
        print(f"  tightest filter on this stationary data: {best_stream} "
              f"(std {best_std:.4f}°)")
        # Sanity checks
        ok = True
        if y["yaw_kalman"].std() > y["yaw_mag_cal"].std():
            print(f"  WARNING: yaw_kalman std exceeds the raw magnetometer it filters -- check tuning.")
            ok = False
        if y["yaw_rts"].std() > y["yaw_kalman"].std():
            print(f"  NOTE: yaw_rts std > yaw_kalman std -- the smoother is hurting, not helping,")
            print(f"         on this stationary segment. Indicates model-assumption violation or")
            print(f"         noise-parameter mismatch.")
            ok = False
        if ok:
            print(f"  no inconsistencies detected.")


if __name__ == "__main__":
    main()
