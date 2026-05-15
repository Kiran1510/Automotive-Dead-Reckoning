"""2D dead-reckoning trajectory in the NE (easting/northing) frame.

Projects the fused forward velocity onto the fused heading, integrates to
get (xe, xn), and aligns the IMU trajectory to the GPS UTM trajectory by
matching the initial heading.

Inputs:
  build/<dataset>/yaw.csv       — yaw_fused (heading)
  build/<dataset>/velocity.csv  — vel_fused (forward speed)
  build/<dataset>/gps.csv       — utm_easting, utm_northing (ground truth)

Outputs:
  build/<dataset>/trajectory.csv             — t, xe, xn, gps_e, gps_n (ground truth interp)
  build/<dataset>/trajectory_imu_vs_gps.png  — overlay of dead-reckoned vs GPS

The initial-heading alignment uses the FIRST_HEADING_SECONDS leading window
of GPS motion (vs the old single 2-point estimate) when the car is moving;
this is more robust to GPS noise at low speed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

BUILD_DIR = Path("build")
FIRST_HEADING_SECONDS = 30.0  # window over which to estimate GPS heading for alignment
MIN_SPEED_M_S = 1.0           # ignore stationary samples when estimating GPS heading


def estimate_initial_gps_heading(utm_e: np.ndarray, utm_n: np.ndarray, t: np.ndarray,
                                 window_s: float, min_speed: float) -> tuple[float, str]:
    """Estimate initial GPS heading as a least-squares line fit over the leading window of motion.

    Falls back to the original 2-point estimate if no moving samples in the window.
    Returns (heading_rad, method_name).
    """
    mask = (t - t[0]) <= window_s
    e = utm_e[mask] - utm_e[0]
    n = utm_n[mask] - utm_n[0]
    # Reject samples too close to origin (car not yet moving)
    dist = np.sqrt(e ** 2 + n ** 2)
    moving = dist > min_speed
    if moving.sum() >= 3:
        # Line through origin: minimize sum( (n - tan(theta) * e)^2 ) -- use atan2 of mean direction.
        # More robust: weight by distance, take direction of weighted centroid.
        we = float(np.sum(e[moving] * dist[moving]))
        wn = float(np.sum(n[moving] * dist[moving]))
        return float(np.arctan2(wn, we)), f"weighted-centroid over {window_s:.0f}s ({moving.sum()} samples)"
    # Fallback: original 2-point estimate.
    return float(np.arctan2(utm_n[1] - utm_n[0], utm_e[1] - utm_e[0])), "2-point (legacy fallback)"


def rotate2d(x: np.ndarray, y: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(theta), np.sin(theta)
    return x * c - y * s, x * s + y * c


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("--build", type=Path, default=BUILD_DIR)
    parser.add_argument("--yaw-source", choices=["fused", "quat"], default="fused",
                        help="Heading source: 'fused' = complementary mag/gyro (default), "
                             "'quat' = VN-100 onboard quaternion (independent reference)")
    args = parser.parse_args()

    out_dir = args.build / args.dataset
    yaw = pd.read_csv(out_dir / "yaw.csv")
    vel = pd.read_csv(out_dir / "velocity.csv")
    gps = pd.read_csv(out_dir / "gps.csv")

    t_imu = vel["t"].to_numpy() - vel["t"].iloc[0]
    vel_fused = vel["vel_fused"].to_numpy()
    utm_e = gps["utm_easting"].to_numpy()
    utm_n = gps["utm_northing"].to_numpy()
    t_gps = gps["t"].to_numpy() - gps["t"].iloc[0]

    # Pick heading source. yaw_quat is from the VN-100's onboard sensor fusion;
    # we negate it because it uses NED-style CW-positive convention vs yaw_fused's
    # math CCW-positive. After this flip the two yaw streams are directly comparable.
    if args.yaw_source == "fused":
        yaw_heading = yaw["yaw_fused"].to_numpy()
    else:
        yaw_heading = -yaw["yaw_quat"].to_numpy()

    # Project fused velocity onto easting/northing.
    # The yaw signal increases with CW heading change (opposite of math convention),
    # so the north component uses -sin(yaw) instead of +sin(yaw). The old NE_mapping.py
    # used +sin(yaw) which produced a vertically-reflected trajectory; its 2-point
    # initial alignment partially cancelled the reflection by coincidence, making the
    # bug look like alignment noise rather than a sign error.
    ve = vel_fused * np.cos(yaw_heading)
    vn = -vel_fused * np.sin(yaw_heading)
    xe = integrate.cumulative_trapezoid(ve, t_imu, initial=0.0)
    xn = integrate.cumulative_trapezoid(vn, t_imu, initial=0.0)

    # GPS ground truth, zeroed at the start.
    gps_e = utm_e - utm_e[0]
    gps_n = utm_n - utm_n[0]

    # -- Alignment: NEW uses a window over moving GPS, OLD uses the original 2-point estimate.
    gps_heading_new, method = estimate_initial_gps_heading(utm_e, utm_n, t_gps,
                                                            FIRST_HEADING_SECONDS, MIN_SPEED_M_S)
    gps_heading_old = float(np.arctan2(utm_n[1] - utm_n[0], utm_e[1] - utm_e[0]))
    imu_heading_initial = float(yaw_heading[0])
    rot_new = gps_heading_new - imu_heading_initial
    rot_old = gps_heading_old - imu_heading_initial

    xe_new, xn_new = rotate2d(xe, xn, rot_new)
    xe_old, xn_old = rotate2d(xe, xn, rot_old)

    # Interpolate the dead-reckoned trajectory onto GPS times so we can compute pointwise error.
    xe_at_gps = np.interp(t_gps, t_imu, xe_new)
    xn_at_gps = np.interp(t_gps, t_imu, xn_new)
    err_m = np.sqrt((xe_at_gps - gps_e) ** 2 + (xn_at_gps - gps_n) ** 2)

    xe_at_gps_old = np.interp(t_gps, t_imu, xe_old)
    xn_at_gps_old = np.interp(t_gps, t_imu, xn_old)
    err_m_old = np.sqrt((xe_at_gps_old - gps_e) ** 2 + (xn_at_gps_old - gps_n) ** 2)

    width = 78
    print("=" * width)
    print(f"TRAJECTORY stage on '{args.dataset}'")
    print("=" * width)
    print(f"  IMU samples: {len(t_imu)}, GPS samples: {len(t_gps)}")
    print()
    print("  initial-heading alignment:")
    print(f"    NEW ({method}): {np.degrees(gps_heading_new):+.2f}°")
    print(f"    OLD (2-point estimate):                       {np.degrees(gps_heading_old):+.2f}°")
    print(f"    IMU initial yaw_fused:                        {np.degrees(imu_heading_initial):+.2f}°")
    print(f"    rotation applied (NEW): {np.degrees(rot_new):+.2f}°  (OLD: {np.degrees(rot_old):+.2f}°)")
    print()
    print("  dead-reckoned position error vs GPS UTM ground truth:")
    print(f"    {'method':<22} {'mean (m)':>10} {'median (m)':>12} {'max (m)':>10} {'final (m)':>11}")
    for name, err in (("NEW alignment", err_m), ("OLD (2-point) alignment", err_m_old)):
        print(f"    {name:<22} {err.mean():10.2f} {np.median(err):12.2f} {err.max():10.2f} {err[-1]:11.2f}")
    print()

    traj_df = pd.DataFrame({
        "t": vel["t"],
        "xe": xe_new,
        "xn": xn_new,
    })
    traj_df.to_csv(out_dir / "trajectory.csv", index=False)
    print(f"  trajectory: {out_dir}/trajectory.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(xe_new, xn_new, "b-", linewidth=2, label="IMU dead-reckoned (new alignment)")
    ax.plot(gps_e, gps_n, "r-", linewidth=2, label="GPS UTM (ground truth)")
    ax.scatter([0], [0], color="black", s=60, zorder=5, label="start")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title(f"{args.dataset}: estimated trajectory vs GPS  "
                 f"(mean err {err_m.mean():.1f} m, max {err_m.max():.0f} m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_dir / "trajectory_imu_vs_gps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot:       {out_dir}/trajectory_imu_vs_gps.png")


if __name__ == "__main__":
    main()
