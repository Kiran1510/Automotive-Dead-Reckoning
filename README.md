# Automotive Dead Reckoning

[![CI](https://github.com/kiransairam/Automotive-Dead-Reckoning/actions/workflows/ci.yml/badge.svg)](https://github.com/kiransairam/Automotive-Dead-Reckoning/actions/workflows/ci.yml)

Vehicle navigation system implementing multi-sensor fusion with GPS and IMU for accurate position estimation in automotive applications.

## Overview

This project integrates VectorNav VN-100 IMU and BU-353N GPS data to implement 2-D X and Y coordinate dead reckoning for vehicle navigation. The system fuses two of these sensor streams (40hz rate for IMU and 1hz rate for GPS) using complementary filtering to achieve positioning during GPS signal degradation.

## Key Features

- **Multi-Sensor Fusion**: Integration of GPS, IMU (accelerometer, gyroscope, magnetometer) for comprehensive state estimation
- **Dead Reckoning Implementation**: Position estimation using velocity integration and heading calculations
- **Complementary Filtering**: Sensor fusion techniques combining high-frequency IMU data with GPS corrections
- **Kalman Filtering**: Optimal state estimation for improved accuracy
- **Real-world Testing**: Validation using circular and driving trajectory datasets

## Technical Approach

### Sensors Used
- **VectorNav VN-100**: 9-axis IMU at 40 Hz (acceleration, angular velocity, magnetic field, and onboard quaternion)
- **BU-353N GPS**: position, UTM coordinates, and velocity at 1 Hz with ~2.5 m accuracy

### Methods
- Hard- and soft-iron magnetometer calibration via Fitzgibbon-Pilu-Fisher direct ellipse fit on the `circle_data` recording
- Gyroscope and accelerometer bias estimation from a truly stationary `idle_car` window, with three-window tilt-consistency verification
- Magnetometer / gyroscope complementary fusion for heading estimation (0.10 Hz Butterworth, order 2)
- Velocity estimation: 1 Hz GPS speed (UTM by default; Haversine and lat/lon-Pythagorean available as alternatives) fused with 40 Hz integrated accelerometer via the same complementary filter
- Dead reckoning by projecting fused speed onto fused heading, comparing against ground-truth GPS UTM trajectory

## Results

Robust tracking for upto 30 seconds before needing GPS corrections. Useful in GPS denied areas such as tunnels, underpasses, and occluded city blocks.

## Pipeline

The analysis runs as a sequence of stages, each reading from a known location and emitting predictable artifacts.

```
data/<dataset>/*.mcap                              raw ROS2 bag (drop new datasets here)
        │
        ▼   scripts/bag_to_csv.py
build/<dataset>/{imu.csv, gps.csv}                 flat CSVs (quaternion preserved)
        │
        ▼   scripts/calibration.py
config/calibration.json                            hard/soft iron, biases, mount tilt
        │
        ▼   scripts/yaw.py <dataset>
build/<dataset>/yaw.csv + four yaw plots
        │
        ▼   scripts/velocity.py <dataset>
build/<dataset>/velocity.csv + 3-panel plot + GPS distance comparative study
        │
        ▼   scripts/dead_reckon.py <dataset>
build/<dataset>/dead_reckoning_comparison.png      ωẊ vs lateral-acc consistency
        │
        ▼   scripts/trajectory.py <dataset>
build/<dataset>/trajectory_imu_vs_gps.png          final 2D dead-reckoned path vs GPS
```

The bag converter automatically loads the custom `vn_interface/Vectornav` and `gps_interface/Customgps` message types embedded in the bag itself — no ROS environment required.

## Repository Structure
```
├── data/                              # raw .mcap bags (gitignored; drop new datasets here)
│   ├── circle_data/                   # car driving in circles for hard/soft iron calibration
│   ├── driving_data/                  # the main 41-minute Boston drive
│   ├── engine_on/                     # stationary, engine on (diagnostic for engine magnetic offset)
│   └── idle_car/                      # stationary, engine off (gyro & acc bias source)
├── build/                             # generated CSVs and plots (gitignored)
├── config/
│   └── calibration.json               # produced by scripts/calibration.py
├── scripts/
│   ├── bag_to_csv.py                  # .mcap → imu.csv + gps.csv (preserves quaternion)
│   ├── inspect_bag.py                 # dump schema of any bag
│   ├── calibration.py                 # hard/soft iron + gyro/acc bias + tilt analysis
│   ├── yaw.py                         # mag/gyro/quaternion yaw + complementary filter
│   ├── velocity.py                    # GPS speed (UTM/Haversine/Pyth) + IMU + complementary
│   ├── dead_reckon.py                 # ωẊ vs lateral-acc rigid-body consistency check
│   ├── trajectory.py                  # final 2D dead-reckoned path vs GPS truth
│   └── slice_bag.py                   # utility: slice the leading N seconds of any bag
├── tests/fixtures/
│   ├── driving_tiny/                  # 5-sec slice of driving_data for CI
│   └── calibration.json               # pinned calibration snapshot for CI
├── .github/workflows/ci.yml           # runs the full pipeline on every push/PR
├── circle_data/, driving_data/        # original lab outputs (renamed-and-cleaned-up by the refactor branch, preserved for reference)
├── requirements.txt
├── Lab5 Report.pdf
└── README.md
```

## Running

```bash
# one-time setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# convert every bag to CSV
for d in data/*/; do .venv/bin/python scripts/bag_to_csv.py "$d"; done

# produce calibration constants (writes config/calibration.json)
.venv/bin/python scripts/calibration.py

# run the analysis stages on a dataset (default: driving_data)
.venv/bin/python scripts/yaw.py driving_data
.venv/bin/python scripts/velocity.py driving_data       # default: --gps-method utm
.venv/bin/python scripts/dead_reckon.py driving_data    # ωẊ vs lateral-acc consistency
.venv/bin/python scripts/trajectory.py driving_data     # final 2D path vs GPS truth

# Optional: comparative study against alternative GPS distance methods
.venv/bin/python scripts/velocity.py driving_data --gps-method haversine
.venv/bin/python scripts/velocity.py driving_data --gps-method pythagorean

# Yaw source for trajectory (lower trajectory error = better fusion)
.venv/bin/python scripts/trajectory.py driving_data --yaw-source quat     # VN-100 onboard quaternion
.venv/bin/python scripts/trajectory.py driving_data --yaw-source gps      # LPF(GPS course) + HPF(gyro)
.venv/bin/python scripts/trajectory.py driving_data --yaw-source kalman   # Kalman (mag + GPS) — best post-drive
.venv/bin/python scripts/trajectory.py driving_data --yaw-source rts      # Kalman + RTS smoother (experimental)
```

### Yaw source comparison on the driving_data bag

`yaw_fused` (the default complementary filter on mag + gyro) drifts ~50° RMS over 41 minutes because the magnetometer is noisy in urban environments. Post-drive analysis benefits from anchoring yaw to GPS course. Trajectory error vs GPS UTM ground truth:

| `--yaw-source` | mean (m) | max (m) | final (m) | Notes |
|---|---|---|---|---|
| `fused` (default) | 763 | 2454 | 263 | LPF(magnetometer) + HPF(gyro) |
| `quat` | 508 | 1316 | 1294 | VN-100 onboard sensor fusion |
| `gps` | 2279 | 4387 | **58** | GPS course / gyro complementary — excellent endpoint, wanders mid-drive |
| **`kalman`** | **260** | **425** | 325 | Kalman with mag + GPS observations — best for post-drive |
| `rts` | 1710 | 3136 | 209 | Kalman + RTS smoother (experimental — sensitive to noise-parameter tuning on long sequences) |

Every analysis script prints a side-by-side new-vs-old numeric comparison so any refactor is held to the "match or exceed previous outputs" standard.

## Continuous integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the full pipeline on every push and pull request against a committed 5-second fixture bag (`tests/fixtures/driving_tiny/`). It exercises bag conversion, both default and alternative GPS-distance methods, every analysis stage, and asserts that all expected output files exist with the canonical CSV schemas preserved. Runs in under a minute.

## Documentation

**[Lab Report (PDF)](Lab5%20Report.pdf)** - Complete analysis of sensor fusion implementation, methodology, and results

## Technologies

- Python (NumPy, Pandas, SciPy, Matplotlib)
- ROS2 Jazzy (sensor recording, `.mcap` storage); `rosbags` for offline decoding without a ROS install
- Sensor fusion algorithms (complementary filter)
- GPS/IMU integration with UTM projection for the trajectory frame

## Applications

This dead reckoning implementation is relevant for:
- Autonomous vehicle navigation
- GPS-denied environment positioning
- Sensor fusion in automotive systems
- Localization for mobile robotics

---

## Author

**Kiran Sairam Bethi Balagangadaran**  
MS Robotics, Northeastern University

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dr Kris Dorsey
- Northeastern University EECE5554 Course Staff
- ROS2 Community
- Open-source sensor driver contributors
