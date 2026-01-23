# Automotive Dead Reckoning

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
- **VectorNav VN-100**: 9-axis IMU providing acceleration, angular velocity, and magnetic field measurements
- **BU-353N GPS**: Position and velocity data with ~2.5m accuracy

### Methods
- Quaternion-based orientation tracking from IMU
- Magnetometer/gyroscope fusion for heading estimation
- Velocity estimation from accelerometer data with bias compensation
- GPS/IMU complementary filter for position correction
- Dead reckoning comparison against ground truth GPS

## Results

Robust tracking for upto 30 seconds before needing GPS corrections. Useful in GPS denied areas such as tunnels, underpasses, and occluded city blocks.

## Repository Structure
```
├── circle_data/          # Circular trajectory test data
    ├── *.py             # Calibration scripts
    ├── *.csv            # GPS and IMU sensor logs
    └── *.png            # Result plots
├── driving_data/         # Real-world driving dataset
│   ├── *.py             # Analysis and visualization scripts
│   ├── *.csv            # GPS and IMU sensor logs
│   └── *.png            # Result plots
├── Lab5 Report.pdf      # Detailed technical report
└── README.md
```

## Documentation

**[Lab Report (PDF)](Lab5%20Report.pdf)** - Complete analysis of sensor fusion implementation, methodology, and results

## Technologies

- Python (NumPy, Pandas, Matplotlib)
- ROS2 (for sensor data collection)
- Sensor fusion algorithms (Complementary filter, Kalman filter)
- GPS/IMU integration techniques

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
