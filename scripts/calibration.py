"""Produce config/calibration.json from the calibration bags.

Sources:
  - build/circle_data/imu.csv  → hard-iron offset + soft-iron correction matrix
                                  via Fitzgibbon-Pilu-Fisher direct ellipse fit
                                  (more robust than the original midpoint-of-min/max)
  - build/idle_car/imu.csv     → gyro bias + accelerometer at-rest reading
                                  (truly stationary, no engine vibration)
  - build/engine_on/imu.csv    → diagnostic engine-induced magnetic offset
                                  (mean of engine_on - mean of idle_car)

Compares the new values against the hardcoded constants previously copy-pasted
into every driving_data/ script. Reports residual radius std after correction
as a quality metric (smaller = better fit).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BUILD_DIR = Path("build")
CONFIG_DIR = Path("config")
OUTPUT_FIGURE = BUILD_DIR / "circle_data" / "magnetometer_calibration.png"
CALIBRATION_JSON = CONFIG_DIR / "calibration.json"

GRAVITY_M_S2 = 9.80665
TESLA_TO_MILLIGAUSS = 1e7

# Previously hardcoded values (copy-pasted across driving_data/*.py) for comparison.
OLD_HARD_IRON_OFFSET_TESLA = np.array([1.978e-5, 1.289e-5])
OLD_SOFT_IRON_MATRIX = np.array([[1.00017403, -0.00836799],
                                 [-0.00836799, 0.99996603]])


def fit_ellipse_fpf(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fitzgibbon-Pilu-Fisher (1999) direct least-squares ellipse fit.

    Returns (center, semi_axes, theta) where center=(cx, cy), semi_axes=(a, b)
    are the semi-axis lengths, and theta is the rotation in radians.
    Algebraically guaranteed to return an ellipse, not a hyperbola or parabola.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    # The ellipse constraint matrix C = [[0,0,2],[0,-1,0],[2,0,0]] has inverse:
    C_inv = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
    eigvals, eigvecs = np.linalg.eig(C_inv @ M)
    # Pick the eigenvector that yields an ellipse (4AC - B^2 > 0).
    cond = 4 * eigvecs[0] * eigvecs[2] - eigvecs[1] ** 2
    idx = int(np.argmax(cond))
    a1 = eigvecs[:, idx].real
    a2 = T @ a1
    A, B, C, D, E, F = a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]

    # Geometric parameters of A x^2 + B xy + C y^2 + D x + E y + F = 0.
    denom = B * B - 4 * A * C
    cx = (2 * C * D - B * E) / denom
    cy = (2 * A * E - B * D) / denom
    num = 2 * (A * E * E + C * D * D - B * D * E + denom * F)
    s1 = A + C
    s2 = np.sqrt((A - C) ** 2 + B * B)
    sa = float(-np.sqrt(num * (s1 + s2)) / denom)
    sb = float(-np.sqrt(num * (s1 - s2)) / denom)
    theta = 0.5 * np.arctan2(B, A - C)
    return np.array([cx, cy]), np.array([sa, sb]), float(theta)


def soft_iron_matrix_from_ellipse(semi_axes: np.ndarray, theta: float) -> tuple[np.ndarray, float]:
    """Build a 2x2 soft-iron correction matrix that maps the fitted ellipse to a circle.

    Returns (matrix, target_radius). Apply as: corrected = matrix @ (raw - hard_iron).
    """
    a, b = float(semi_axes[0]), float(semi_axes[1])
    target_radius = 0.5 * (a + b)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    S = np.diag([target_radius / a, target_radius / b])
    return R @ S @ R.T, target_radius


def radius_residual_std(mag_xy: np.ndarray, hard_iron: np.ndarray, soft_iron: np.ndarray) -> float:
    """After applying the correction, points should lie on a circle. Returns radius std (Tesla)."""
    centered = mag_xy - hard_iron
    corrected = (soft_iron @ centered.T).T
    radii = np.linalg.norm(corrected, axis=1)
    return float(radii.std())


def fmt_vec(v, fmt="{:+.4e}"):
    return "[" + ", ".join(fmt.format(x) for x in np.asarray(v).ravel()) + "]"


def fmt_mat(m, fmt="{:+.6f}"):
    return "[" + ", ".join("[" + ", ".join(fmt.format(x) for x in row) + "]" for row in m) + "]"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, default=BUILD_DIR, help="Build directory containing per-dataset CSVs")
    parser.add_argument("--out-json", type=Path, default=CALIBRATION_JSON, help="Where to write calibration.json")
    parser.add_argument("--out-figure", type=Path, default=OUTPUT_FIGURE, help="Magnetometer calibration plot path")
    args = parser.parse_args()

    circle = pd.read_csv(args.build / "circle_data" / "imu.csv")
    idle = pd.read_csv(args.build / "idle_car" / "imu.csv")
    engine = pd.read_csv(args.build / "engine_on" / "imu.csv")

    mag_xy = circle[["mag_x", "mag_y"]].to_numpy()

    # --- New magnetometer calibration (FPF ellipse fit) ---
    center, semi_axes, theta = fit_ellipse_fpf(mag_xy[:, 0], mag_xy[:, 1])
    new_hard_iron = center
    new_soft_iron, target_radius = soft_iron_matrix_from_ellipse(semi_axes, theta)
    new_residual = radius_residual_std(mag_xy, new_hard_iron, new_soft_iron)

    # --- Quality of the OLD hardcoded calibration on the same data ---
    old_residual = radius_residual_std(mag_xy, OLD_HARD_IRON_OFFSET_TESLA, OLD_SOFT_IRON_MATRIX)

    # --- Biases from idle_car (truly stationary) ---
    gyro_bias = idle[["gyro_x", "gyro_y", "gyro_z"]].mean().to_numpy()
    acc_at_rest = idle[["acc_x", "acc_y", "acc_z"]].mean().to_numpy()
    # If the IMU were perfectly level, at rest it would read (0, 0, -g). The deviation
    # from that is bias + mounting tilt; subtracting it from raw acc removes both.
    acc_bias = acc_at_rest - np.array([0.0, 0.0, -GRAVITY_M_S2])

    # --- Diagnostic: engine-induced magnetic offset (engine_on minus idle_car) ---
    mag_idle = idle[["mag_x", "mag_y", "mag_z"]].mean().to_numpy()
    mag_engine = engine[["mag_x", "mag_y", "mag_z"]].mean().to_numpy()
    engine_mag_offset = mag_engine - mag_idle

    # --- Print the comparison report ---
    width = 78
    print("=" * width)
    print("MAGNETOMETER CALIBRATION  (source: circle_data, 2D, Tesla)")
    print("=" * width)
    print(f"  hard-iron offset:")
    print(f"    old (midpoint of min/max): {fmt_vec(OLD_HARD_IRON_OFFSET_TESLA)}")
    print(f"    new (FPF ellipse center):  {fmt_vec(new_hard_iron)}")
    print(f"    diff:                      {fmt_vec(new_hard_iron - OLD_HARD_IRON_OFFSET_TESLA)}")
    print()
    print(f"  soft-iron correction matrix:")
    print(f"    old: {fmt_mat(OLD_SOFT_IRON_MATRIX)}")
    print(f"    new: {fmt_mat(new_soft_iron)}")
    max_diff = float(np.max(np.abs(new_soft_iron - OLD_SOFT_IRON_MATRIX)))
    print(f"    max element-wise diff: {max_diff:.3e}")
    print()
    print(f"  ellipse: semi-axes [{semi_axes[0] * TESLA_TO_MILLIGAUSS:.2f}, "
          f"{semi_axes[1] * TESLA_TO_MILLIGAUSS:.2f}] mG, "
          f"rotation {np.degrees(theta):+.2f} deg, "
          f"target radius {target_radius * TESLA_TO_MILLIGAUSS:.2f} mG")
    print()
    print(f"  quality (radius std after correction, smaller is better):")
    print(f"    OLD calibration on circle_data:  {old_residual * TESLA_TO_MILLIGAUSS:.4f} mG  "
          f"({old_residual:.3e} T)")
    print(f"    NEW calibration on circle_data:  {new_residual * TESLA_TO_MILLIGAUSS:.4f} mG  "
          f"({new_residual:.3e} T)")
    improvement = (old_residual - new_residual) / old_residual * 100
    print(f"    improvement:                     {improvement:+.2f}%")
    print()

    print("=" * width)
    print("GYRO BIAS  (source: idle_car, mean of stationary readings)")
    print("=" * width)
    print(f"  bias (rad/s): {fmt_vec(gyro_bias)}")
    print(f"  previously: mean(gyro_z) over the entire drive (contaminated by real rotation)")
    print()

    print("=" * width)
    print("ACCELEROMETER AT REST  (source: idle_car)")
    print("=" * width)
    print(f"  raw acc at rest (m/s^2):     {fmt_vec(acc_at_rest)}")
    print(f"  bias = raw - (0,0,-g):       {fmt_vec(acc_bias)}")
    print(f"  (includes mounting tilt; magnitude {np.linalg.norm(acc_at_rest):.4f} m/s^2 "
          f"vs g = {GRAVITY_M_S2:.4f})")
    print()

    print("=" * width)
    print("DIAGNOSTIC: engine-induced magnetic offset  (engine_on - idle_car)")
    print("=" * width)
    print(f"  Tesla:     {fmt_vec(engine_mag_offset)}")
    print(f"  milliGauss: {fmt_vec(engine_mag_offset * TESLA_TO_MILLIGAUSS, fmt='{:+.2f}')}")
    print()

    # --- Write JSON ---
    config = {
        "version": 1,
        "magnetometer": {
            "hard_iron_offset_tesla_xy": new_hard_iron.tolist(),
            "soft_iron_matrix_2d": new_soft_iron.tolist(),
            "source": "circle_data",
            "method": "Fitzgibbon-Pilu-Fisher direct ellipse fit",
            "ellipse_semi_axes_tesla": semi_axes.tolist(),
            "ellipse_rotation_rad": float(theta),
            "target_radius_tesla": float(target_radius),
            "residual_radius_std_tesla": float(new_residual),
        },
        "gyroscope": {
            "bias_rad_s": gyro_bias.tolist(),
            "source": "idle_car",
            "samples": int(len(idle)),
            "duration_s": float(idle["t"].iloc[-1] - idle["t"].iloc[0]),
        },
        "accelerometer": {
            "at_rest_m_s2": acc_at_rest.tolist(),
            "bias_m_s2": acc_bias.tolist(),
            "gravity_assumed_m_s2": GRAVITY_M_S2,
            "source": "idle_car",
        },
        "diagnostics": {
            "engine_induced_mag_offset_tesla": engine_mag_offset.tolist(),
            "engine_on_mean_tesla": mag_engine.tolist(),
            "idle_car_mean_tesla": mag_idle.tolist(),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(config, f, indent=2)
    print(f"calibration written to {args.out_json}")

    # --- Reproduce the magnetometer calibration plot ---
    args.out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    mag_mG = mag_xy * TESLA_TO_MILLIGAUSS
    corrected = ((new_soft_iron @ (mag_xy - new_hard_iron).T).T) * TESLA_TO_MILLIGAUSS
    ax.scatter(mag_mG[:, 0], mag_mG[:, 1], c="red", alpha=0.5, s=10,
               label="raw data (including car interference)", edgecolors="none")
    ax.scatter(corrected[:, 0], corrected[:, 1], c="blue", alpha=0.5, s=10,
               label="corrected data", edgecolors="none")
    target_r_mG = target_radius * TESLA_TO_MILLIGAUSS
    circle_artist = plt.Circle((0, 0), target_r_mG, fill=False, color="green", linestyle=":",
                               linewidth=2, alpha=0.6, label=f"target circle (r={target_r_mG:.0f} mG)")
    ax.add_patch(circle_artist)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("magnetometer X (mG)", fontsize=14)
    ax.set_ylabel("magnetometer Y (mG)", fontsize=14)
    ax.set_title(f"magnetometer calibration: hard + soft iron correction "
                 f"(residual {new_residual * TESLA_TO_MILLIGAUSS:.3f} mG)", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(args.out_figure, dpi=300, bbox_inches="tight")
    print(f"figure written to {args.out_figure}")


if __name__ == "__main__":
    main()
