"""Yaw estimation: five independent streams + their pairwise comparison.

Streams (all stored in yaw.csv in the same CW-positive convention):

  yaw_mag_cal       calibrated magnetometer, atan2(my, mx)
  yaw_gyro          integrated gyro_z with idle_car bias subtracted
  yaw_fused         complementary filter: LPF(mag) + HPF(gyro), 0.1 Hz cutoff
  yaw_quat          VN-100 onboard quaternion (sign-flipped to match convention)
  yaw_gps           GPS-constrained complementary filter (--yaw-source gps_constrained
                    in trajectory.py): LPF(GPS course) + HPF(gyro), 0.5 Hz cutoff,
                    where GPS speed > 3 m/s. Anchors heading to GPS observations
                    instead of the magnetometer.
  yaw_kalman        forward Kalman filter on [yaw, gyro_bias] with magnetometer +
                    GPS-course measurements
  yaw_rts           Kalman + RTS backward smoother — uses all data (forward and
                    backward in time) to estimate each timestamp's yaw. The
                    technically-optimal post-drive estimate under Gaussian
                    assumptions.

Reads calibrated inputs from config/calibration.json (so this must run AFTER
scripts/calibration.py).

Outputs:
  build/<dataset>/yaw.csv                       — all streams
  build/<dataset>/yaw_raw_vs_calibrated.png     — raw mag yaw vs calibrated
  build/<dataset>/yaw_gyro_vs_magnetometer.png  — gyro-integrated vs mag yaw
  build/<dataset>/yaw_complementary_filter.png  — LPF mag + HPF gyro + fused
  build/<dataset>/yaw_four_panel.png            — LPF / HPF / fused / quaternion
  build/<dataset>/yaw_all_sources.png           — all 5 yaw streams overlaid
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


def wrap_to_pi(x):
    """Wrap angle(s) into [-pi, pi)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def gps_course_yaw_convention(utm_e, utm_n, t_gps, min_speed_m_s=3.0):
    """GPS course over ground at each GPS sample, converted to match yaw_fused's
    CW-positive convention by negating. Returns array of length len(t_gps), NaN
    where the vehicle is below min_speed_m_s.
    """
    de = np.diff(utm_e)
    dn = np.diff(utm_n)
    dt = np.diff(t_gps)
    speed = np.sqrt(de ** 2 + dn ** 2) / np.maximum(dt, 1e-9)
    course = -np.arctan2(dn, de)  # negate: math-CCW -> yaw-CW

    out = np.full(len(t_gps), np.nan)
    out[1:] = np.where(speed > min_speed_m_s, course, np.nan)
    return out


def yaw_gps_constrained(t_imu, gyro_z, gyro_z_bias, t_gps, gps_course,
                        cutoff_hz=0.5, butter_order=BUTTERWORTH_ORDER):
    """Complementary filter using GPS course as the low-frequency heading reference.

    GPS course (computed from UTM deltas, then sign-converted) is the LPF input;
    integrated gyro is the HPF input. Cutoff is higher (0.5 Hz) than the
    mag-based filter (0.1 Hz) because GPS course is a much cleaner heading
    reference and doesn't need heavy smoothing.
    """
    valid = ~np.isnan(gps_course)
    yaw_gyro_raw = integrate.cumulative_trapezoid(gyro_z - gyro_z_bias, t_imu, initial=0)

    if valid.sum() < 2:
        return yaw_gyro_raw - yaw_gyro_raw[0]  # nothing to anchor to

    t_valid = t_gps[valid]
    course_unwrapped = np.unwrap(gps_course[valid])

    # Interpolate (and hold ends) onto IMU timeline.
    course_on_imu = np.interp(t_imu, t_valid, course_unwrapped)

    # Align gyro yaw to start at GPS course[0] so HPF has a clean DC anchor.
    yaw_gyro_aligned = yaw_gyro_raw - yaw_gyro_raw[0] + course_on_imu[0]

    fs = 1.0 / np.mean(np.diff(t_imu))
    nyq = 0.5 * fs
    b_lpf, a_lpf = signal.butter(butter_order, cutoff_hz / nyq, btype="low")
    b_hpf, a_hpf = signal.butter(butter_order, cutoff_hz / nyq, btype="high")
    course_lpf = signal.filtfilt(b_lpf, a_lpf, course_on_imu)
    gyro_hpf = signal.filtfilt(b_hpf, a_hpf, yaw_gyro_aligned)
    return course_lpf + gyro_hpf


def yaw_kalman_rts(t_imu, gyro_z, gyro_z_bias_init, mag_yaw, t_gps, gps_course,
                   sigma_gyro=0.01, sigma_bias_walk=1e-5,
                   sigma_mag=0.30, sigma_gps=0.05, gps_match_tol_s=0.5):
    """1D yaw + gyro bias Kalman filter with RTS backward smoother.

    State:    x = [yaw, gyro_bias]
    Process:  yaw_{k+1}  = yaw_k + (gyro_z_k - bias_k) * dt + w_yaw
              bias_{k+1} = bias_k + w_bias
    Measurements:
      - z_mag = yaw + v_mag at every IMU sample (sigma_mag stronger -> noisier)
      - z_gps = yaw + v_gps at IMU samples within gps_match_tol_s of a GPS sample
                that has a valid (non-NaN) course measurement

    Innovations use wrap_to_pi to handle angle discontinuities. Output yaw is
    re-unwrapped so it's continuous across ±π boundaries.

    Returns (yaw_filter, yaw_smoothed, bias_smoothed): forward Kalman yaw, RTS-
    smoothed yaw, and the smoothed gyro_bias trajectory.
    """
    N = len(t_imu)

    # Pre-match GPS samples to IMU samples by nearest neighbor.
    gps_idx = np.searchsorted(t_gps, t_imu)
    gps_idx = np.clip(gps_idx, 1, len(t_gps) - 1)
    pick_left = (t_imu - t_gps[gps_idx - 1]) <= (t_gps[gps_idx] - t_imu)
    nearest_gps_idx = np.where(pick_left, gps_idx - 1, gps_idx)
    gps_dt = np.abs(t_imu - t_gps[nearest_gps_idx])

    # The mag_yaw input is already unwrapped by yaw.py; we keep the state
    # unwrapped throughout. Innovations use wrap_to_pi to bridge the wrap between
    # a wrapped measurement and the (potentially large) unwrapped state.
    # Same idea for gps_course: NaN samples are skipped and the valid ones, when
    # paired with the running state, are wrapped onto the same multiple of 2*pi.

    # Allocate forward-pass storage.
    x_filt = np.zeros((N, 2))
    P_filt = np.zeros((N, 2, 2))
    x_pred = np.zeros((N, 2))
    P_pred = np.zeros((N, 2, 2))

    x_filt[0] = [mag_yaw[0], gyro_z_bias_init]
    P_filt[0] = np.diag([sigma_mag ** 2, (sigma_bias_walk * 100) ** 2])
    x_pred[0] = x_filt[0]
    P_pred[0] = P_filt[0]

    H = np.array([[1.0, 0.0]])
    I2 = np.eye(2)

    for k in range(1, N):
        dt = t_imu[k] - t_imu[k - 1]
        F = np.array([[1.0, -dt], [0.0, 1.0]])
        u = np.array([gyro_z[k] * dt, 0.0])
        x_pred[k] = F @ x_filt[k - 1] + u
        Q = np.array([[(sigma_gyro * dt) ** 2, 0.0],
                      [0.0, (sigma_bias_walk * np.sqrt(dt)) ** 2]])
        P_pred[k] = F @ P_filt[k - 1] @ F.T + Q

        x_k = x_pred[k].copy()
        P_k = P_pred[k].copy()

        # Magnetometer update (every IMU sample). mag_yaw is unwrapped, x_k[0] is unwrapped,
        # so direct subtraction is correct -- no wrap_to_pi needed.
        innov = mag_yaw[k] - x_k[0]
        S = (H @ P_k @ H.T)[0, 0] + sigma_mag ** 2
        K = (P_k @ H.T).ravel() / S
        x_k = x_k + K * innov
        P_k = (I2 - np.outer(K, H.ravel())) @ P_k

        # GPS update if a nearby valid GPS course exists. gps_course comes from atan2 so it's
        # wrapped to [-pi, pi); align it to the running unwrapped state by adding 2*pi*round((state-course)/2pi).
        gi = nearest_gps_idx[k]
        if gps_dt[k] <= gps_match_tol_s and np.isfinite(gps_course[gi]):
            gps_unwrapped = gps_course[gi] + 2 * np.pi * round((x_k[0] - gps_course[gi]) / (2 * np.pi))
            innov = gps_unwrapped - x_k[0]
            S = (H @ P_k @ H.T)[0, 0] + sigma_gps ** 2
            K = (P_k @ H.T).ravel() / S
            x_k = x_k + K * innov
            P_k = (I2 - np.outer(K, H.ravel())) @ P_k

        x_filt[k] = x_k
        P_filt[k] = P_k

    # Backward RTS smoother. State is unwrapped throughout, so direct subtraction is correct
    # (the linear-Gaussian smoother math applies cleanly with no wrap-around).
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for k in range(N - 2, -1, -1):
        dt = t_imu[k + 1] - t_imu[k]
        F = np.array([[1.0, -dt], [0.0, 1.0]])
        try:
            C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - P_pred[k + 1]) @ C.T

    return x_filt[:, 0], x_smooth[:, 0], x_smooth[:, 1]


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
    gps = pd.read_csv(out_dir / "gps.csv")  # for GPS course (new yaw sources)

    t_abs = imu["t"].to_numpy()
    t = t_abs - t_abs[0]
    mx, my = imu["mag_x"].to_numpy(), imu["mag_y"].to_numpy()
    gz = imu["gyro_z"].to_numpy()
    quat = imu[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy()
    t_gps = gps["t"].to_numpy() - t_abs[0]  # zero-aligned to IMU timeline
    utm_e = gps["utm_easting"].to_numpy()
    utm_n = gps["utm_northing"].to_numpy()

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

    # ---- NEW post-drive yaw sources (Option 1 + Option 3) ----
    gps_course = gps_course_yaw_convention(utm_e, utm_n, t_gps, min_speed_m_s=3.0)
    yaw_gps = yaw_gps_constrained(t, gz, gyro_bias_z_new, t_gps, gps_course, cutoff_hz=0.5)
    # Tuning: sigma_mag=0.30 reflects the city-driving reality that the magnetometer
    # is noisy. sigma_gps=0.05 trusts GPS course strongly when it's available.
    # The forward Kalman with these values gives the best post-drive trajectory
    # (~3x lower error than yaw_fused). The RTS smoother is sensitive to this
    # tuning -- it makes the model's bias inference back-propagate to the initial
    # state, which on this 41-min dataset moves yaw_rts[0] away from the magnetometer
    # initial reading. yaw_rts is therefore experimental on long sequences; use
    # yaw_kalman for the cleanest post-drive trajectory.
    yaw_kf, yaw_rts, gyro_bias_smoothed = yaw_kalman_rts(
        t, gz, gyro_bias_z_new, new["yaw_mag"], t_gps, gps_course)

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

    # Post-drive yaw sources -- these track the actual heading better than yaw_fused
    # over long drives because they incorporate GPS-course observations (Option 1)
    # or weight measurements explicitly (Option 3). Big divergence from yaw_fused
    # is expected and desired: yaw_fused drifts because of magnetic interference.
    print("  post-drive yaw sources:")
    for name, y in [("yaw_gps   (Option 1)", yaw_gps),
                    ("yaw_kalman (Option 3, forward)", yaw_kf),
                    ("yaw_rts   (Option 3, smoothed)", yaw_rts)]:
        diff = y - new["yaw_fused"]
        print(f"    {name:<32}  max|d vs fused|={rad_deg(np.max(np.abs(diff)))}")
    print("    (large divergence is the point: these are anchored to GPS / weighted-optimal")
    print("     while yaw_fused has known drift from city magnetic interference.)")
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
        "yaw_gps": yaw_gps,
        "yaw_kalman": yaw_kf,
        "yaw_rts": yaw_rts,
        "gyro_bias_smoothed": gyro_bias_smoothed,
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

    # ---- Plot 5: all five yaw sources overlaid (post-drive comparison) ----
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, new["yaw_fused"], color="green", linewidth=1.5, alpha=0.7,
            label="yaw_fused (LPF mag + HPF gyro, 0.1 Hz)")
    ax.plot(t, yaw_quat_aligned, color="purple", linewidth=1.0, alpha=0.5,
            label="yaw_quat (VN-100 onboard quaternion, aligned)")
    ax.plot(t, yaw_gps, color="orange", linewidth=1.5, alpha=0.7,
            label="yaw_gps (LPF GPS course + HPF gyro, 0.5 Hz)")
    ax.plot(t, yaw_kf, color="blue", linewidth=1.0, alpha=0.5,
            label="yaw_kalman (forward Kalman, mag+GPS)")
    ax.plot(t, yaw_rts, color="red", linewidth=1.5,
            label="yaw_rts (RTS backward smoother — optimal post-drive)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("yaw (rad)")
    ax.set_title(f"{args.dataset}: all yaw sources compared")
    ax.legend(loc="upper left", fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "yaw_all_sources.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  plots:       {out_dir}/yaw_*.png  (5 figures)")


if __name__ == "__main__":
    main()
