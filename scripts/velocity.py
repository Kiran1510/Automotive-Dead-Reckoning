"""Forward velocity estimation: GPS speed + IMU integration + complementary filter.

GPS speed can be derived three ways; all three are computed every run for
comparison, and the chosen one feeds into the complementary filter.

  --gps-method utm           (default) Pythagorean on UTM easting/northing.
                              Consistent with the trajectory coordinate frame.
  --gps-method haversine      Great-circle distance on lat/lon.
  --gps-method pythagorean    Equirectangular flat-Earth on lat/lon (cos·lat).

On driving_data, max pairwise speed difference is ~0.024 m/s (a thousand
times below GPS noise), so the choice is a matter of frame consistency and
code simplicity rather than accuracy.

Outputs:
  build/<dataset>/velocity.csv             — t, vel_imu_raw, vel_gps_*, vel_fused
  build/<dataset>/gps_distance_methods.csv — the three speeds + pairwise diffs
  build/<dataset>/velocity_three_panel.png — old fig_4 layout (raw / HPF / fused)
  build/<dataset>/gps_distance_methods.png — overlay of all three GPS speeds
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, signal

BUILD_DIR = Path("build")
CALIBRATION_JSON = Path("config") / "calibration.json"

EARTH_RADIUS_M = 6_371_000
COMPLEMENTARY_CUTOFF_HZ = 0.10
BUTTERWORTH_ORDER = 2
STATIONARY_WINDOW_S = 10.0  # only used for OLD-method comparison


def _speed_from_distances(d: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pad a per-segment distance array with leading 0 and divide by dt."""
    dt = np.diff(t)
    speed = np.zeros(len(t))
    nonzero = dt > 0
    speed[1:][nonzero] = d[nonzero] / dt[nonzero]
    return speed


def gps_speed_haversine(lat_deg: np.ndarray, lon_deg: np.ndarray, t: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    dlat, dlon = np.diff(lat), np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    d = 2 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return _speed_from_distances(d, t)


def gps_speed_pythagorean_latlon(lat_deg: np.ndarray, lon_deg: np.ndarray, t: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    mid_lat = 0.5 * (lat[:-1] + lat[1:])
    dx = EARTH_RADIUS_M * np.cos(mid_lat) * np.diff(lon)
    dy = EARTH_RADIUS_M * np.diff(lat)
    d = np.sqrt(dx ** 2 + dy ** 2)
    return _speed_from_distances(d, t)


def gps_speed_pythagorean_utm(easting: np.ndarray, northing: np.ndarray, t: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diff(easting) ** 2 + np.diff(northing) ** 2)
    return _speed_from_distances(d, t)


def complementary_velocity(vel_gps_on_imu_t: np.ndarray, vel_imu: np.ndarray,
                           t_imu: np.ndarray, cutoff_hz: float):
    fs = 1.0 / np.mean(np.diff(t_imu))
    nyq = 0.5 * fs
    b_lpf, a_lpf = signal.butter(BUTTERWORTH_ORDER, cutoff_hz / nyq, btype="low")
    b_hpf, a_hpf = signal.butter(BUTTERWORTH_ORDER, cutoff_hz / nyq, btype="high")
    vel_gps_lpf = signal.filtfilt(b_lpf, a_lpf, vel_gps_on_imu_t)
    vel_imu_hpf = signal.filtfilt(b_hpf, a_hpf, vel_imu)
    vel_fused = vel_gps_lpf + vel_imu_hpf
    vel_fused = np.maximum(vel_fused, 0.0)
    return vel_fused, vel_gps_lpf, vel_imu_hpf


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("--build", type=Path, default=BUILD_DIR)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_JSON)
    parser.add_argument("--gps-method", choices=["utm", "haversine", "pythagorean"], default="utm",
                        help="GPS distance method to feed the complementary filter")
    args = parser.parse_args()

    out_dir = args.build / args.dataset
    imu = pd.read_csv(out_dir / "imu.csv")
    gps = pd.read_csv(out_dir / "gps.csv")
    with open(args.calibration) as f:
        cal = json.load(f)

    t_imu = imu["t"].to_numpy() - imu["t"].iloc[0]
    t_gps = gps["t"].to_numpy() - gps["t"].iloc[0]
    acc_x = imu["acc_x"].to_numpy()
    lat = gps["latitude"].to_numpy()
    lon = gps["longitude"].to_numpy()
    ue = gps["utm_easting"].to_numpy()
    un = gps["utm_northing"].to_numpy()

    # ---- IMU velocity, NEW (idle_car at-rest as bias) and OLD (first 10s of drive) ----
    acc_x_bias_new = float(cal["accelerometer"]["at_rest_m_s2"][0])
    vel_imu_new = integrate.cumulative_trapezoid(acc_x - acc_x_bias_new, t_imu, initial=0)

    mask_old = t_imu < STATIONARY_WINDOW_S
    acc_x_bias_old = float(np.mean(acc_x[mask_old]))
    vel_imu_old = integrate.cumulative_trapezoid(acc_x - acc_x_bias_old, t_imu, initial=0)

    # ---- Three GPS-speed methods (computed for every run) ----
    v_hav = gps_speed_haversine(lat, lon, t_gps)
    v_pyth_ll = gps_speed_pythagorean_latlon(lat, lon, t_gps)
    v_pyth_utm = gps_speed_pythagorean_utm(ue, un, t_gps)
    methods = {"utm": v_pyth_utm, "haversine": v_hav, "pythagorean": v_pyth_ll}
    v_gps_chosen = methods[args.gps_method]

    # ---- Interpolate GPS speed onto IMU timeline (NEW uses chosen, OLD uses Haversine) ----
    v_gps_on_imu_new = np.interp(t_imu, t_gps, v_gps_chosen)
    v_gps_on_imu_old = np.interp(t_imu, t_gps, v_hav)

    # ---- Complementary filter ----
    vel_fused_new, v_gps_lpf, v_imu_hpf = complementary_velocity(
        v_gps_on_imu_new, vel_imu_new, t_imu, COMPLEMENTARY_CUTOFF_HZ)
    vel_fused_old, _, _ = complementary_velocity(
        v_gps_on_imu_old, vel_imu_old, t_imu, COMPLEMENTARY_CUTOFF_HZ)

    # ---- Stdout report ----
    width = 78
    print("=" * width)
    print(f"VELOCITY stage on '{args.dataset}'  (gps-method: {args.gps_method})")
    print("=" * width)
    print(f"  IMU: {len(t_imu)} samples, {t_imu[-1]:.1f} s")
    print(f"  GPS: {len(t_gps)} samples, {t_gps[-1]:.1f} s")
    print()
    print("  acc_x bias:")
    print(f"    NEW (idle_car at-rest):   {acc_x_bias_new:+.4f} m/s^2")
    print(f"    OLD (first 10s of drive): {acc_x_bias_old:+.4f} m/s^2  (gyro std 30x idle, not actually stationary)")
    print(f"    diff:                     {acc_x_bias_new - acc_x_bias_old:+.4f} m/s^2")
    print()

    print("  GPS distance method comparison (the 'comparative study'):")
    print(f"    {'method':<26} {'mean (m/s)':>12} {'max (m/s)':>11} {'total dist (km)':>17}")
    dt_gps = np.diff(t_gps)
    for name, v in [("Haversine (lat/lon)", v_hav),
                    ("Pythagorean (lat/lon)", v_pyth_ll),
                    ("Pythagorean (UTM)", v_pyth_utm)]:
        total_km = float(np.sum(v[1:] * dt_gps)) / 1000.0
        print(f"    {name:<26} {v.mean():12.4f} {v.max():11.4f} {total_km:17.4f}")
    print()
    print("  pairwise speed deltas:")
    pairs = [("Haversine - Pyth(lat/lon)", v_hav - v_pyth_ll),
             ("Haversine - Pyth(UTM)",     v_hav - v_pyth_utm),
             ("Pyth(lat/lon) - Pyth(UTM)", v_pyth_ll - v_pyth_utm)]
    print(f"    {'comparison':<28} {'max|diff| (m/s)':>16} {'RMS (m/s)':>11}")
    for name, d in pairs:
        print(f"    {name:<28} {np.max(np.abs(d)):16.6f} {np.sqrt(np.mean(d**2)):11.6f}")
    print()

    print("  vel_fused NEW (chosen + idle_car bias) vs OLD (Haversine + first-10s bias):")
    diff = vel_fused_new - vel_fused_old
    print(f"    max|diff|: {np.max(np.abs(diff)):.4f} m/s")
    print(f"    RMS|diff|: {np.sqrt(np.mean(diff ** 2)):.4f} m/s")
    print()

    # ---- Save velocity.csv ----
    vel_df = pd.DataFrame({
        "t": imu["t"],
        "vel_imu_raw": vel_imu_new,
        "vel_imu_hpf": v_imu_hpf,
        "vel_gps_interp": v_gps_on_imu_new,
        "vel_gps_lpf": v_gps_lpf,
        "vel_fused": vel_fused_new,
    })
    vel_df.to_csv(out_dir / "velocity.csv", index=False)

    # ---- Save GPS-method comparison CSV ----
    methods_df = pd.DataFrame({
        "t": gps["t"],
        "v_haversine": v_hav,
        "v_pythagorean_latlon": v_pyth_ll,
        "v_pythagorean_utm": v_pyth_utm,
        "diff_hav_pyth_latlon": v_hav - v_pyth_ll,
        "diff_hav_pyth_utm": v_hav - v_pyth_utm,
        "diff_pyth_latlon_pyth_utm": v_pyth_ll - v_pyth_utm,
    })
    methods_df.to_csv(out_dir / "gps_distance_methods.csv", index=False)
    print(f"  velocity:            {out_dir}/velocity.csv")
    print(f"  GPS method study:    {out_dir}/gps_distance_methods.csv")

    # ---- Plot 1: three-panel velocity (matches old fig_4.png) ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(t_imu, vel_imu_new, "b-", linewidth=1, alpha=0.7, label="IMU velocity (raw, drifts)")
    axes[0].plot(t_imu, v_gps_on_imu_new, "r-", linewidth=1, label=f"GPS velocity ({args.gps_method})")
    axes[0].set_ylabel("velocity (m/s)"); axes[0].set_title("before adjustment")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_imu, v_imu_hpf, "b-", linewidth=1.5,
                 label=f"IMU velocity (HPF {COMPLEMENTARY_CUTOFF_HZ:.2f} Hz)")
    axes[1].plot(t_imu, v_gps_on_imu_new, "r-", linewidth=1, alpha=0.7, label="GPS velocity")
    axes[1].set_ylabel("velocity (m/s)")
    axes[1].set_title(f"IMU with HPF ({COMPLEMENTARY_CUTOFF_HZ:.2f} Hz)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_imu, vel_fused_new, "g-", linewidth=2,
                 label=f"fused velocity (complementary, {COMPLEMENTARY_CUTOFF_HZ:.2f} Hz)")
    axes[2].plot(t_imu, v_gps_on_imu_new, "r-", linewidth=1, alpha=0.7, label="GPS velocity")
    axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("velocity (m/s)")
    axes[2].set_title(f"complementary filter (GPS + IMU, cutoff = {COMPLEMENTARY_CUTOFF_HZ:.2f} Hz)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "velocity_three_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: GPS distance method comparison ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t_gps, v_hav, "r-", linewidth=1.0, alpha=0.7, label="Haversine")
    axes[0].plot(t_gps, v_pyth_ll, "b--", linewidth=1.0, alpha=0.7, label="Pythagorean (lat/lon)")
    axes[0].plot(t_gps, v_pyth_utm, "g:", linewidth=1.5, label="Pythagorean (UTM)")
    axes[0].set_ylabel("GPS speed (m/s)")
    axes[0].set_title("GPS speed: three distance methods compared")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_gps, v_hav - v_pyth_ll, "b-", linewidth=0.5, alpha=0.7,
                 label="Haversine - Pythagorean(lat/lon)")
    axes[1].plot(t_gps, v_hav - v_pyth_utm, "g-", linewidth=0.5, alpha=0.7,
                 label="Haversine - Pythagorean(UTM)")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("speed delta (m/s)")
    axes[1].set_title("pairwise differences")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "gps_distance_methods.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  plots:               {out_dir}/velocity_three_panel.png, gps_distance_methods.png")


if __name__ == "__main__":
    main()
