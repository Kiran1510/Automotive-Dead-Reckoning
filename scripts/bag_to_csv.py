"""Convert a ROS2 .mcap bag (VN-100 IMU + custom GPS) into flat CSVs.

Usage:
    python scripts/bag_to_csv.py <bag_directory> [--out <output_directory>]

Defaults output to build/<bag_basename>/. Writes imu.csv and gps.csv with
predictable column names so downstream analysis scripts don't have to care
about the underlying ROS message layout.
"""

import argparse
from pathlib import Path

import pandas as pd
from rosbags.highlevel import AnyReader

IMU_TOPIC = "/imu"
GPS_TOPIC = "/gps"


def stamp_seconds(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def extract_imu(reader) -> pd.DataFrame:
    connections = [c for c in reader.connections if c.topic == IMU_TOPIC]
    if not connections:
        return pd.DataFrame()

    rows = []
    for conn, _, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, conn.msgtype)
        imu = msg.imu
        mag = msg.mag_field.magnetic_field
        q = imu.orientation
        gyro = imu.angular_velocity
        acc = imu.linear_acceleration
        rows.append({
            "t": stamp_seconds(msg.header),
            "frame_id": msg.header.frame_id,
            "quat_x": q.x, "quat_y": q.y, "quat_z": q.z, "quat_w": q.w,
            "gyro_x": gyro.x, "gyro_y": gyro.y, "gyro_z": gyro.z,
            "acc_x": acc.x, "acc_y": acc.y, "acc_z": acc.z,
            "mag_x": mag.x, "mag_y": mag.y, "mag_z": mag.z,
            "raw_vnymr": msg.raw_string,
        })
    return pd.DataFrame(rows)


def extract_gps(reader) -> pd.DataFrame:
    connections = [c for c in reader.connections if c.topic == GPS_TOPIC]
    if not connections:
        return pd.DataFrame()

    rows = []
    for conn, _, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, conn.msgtype)
        rows.append({
            "t": stamp_seconds(msg.header),
            "frame_id": msg.header.frame_id,
            "latitude": msg.latitude,
            "longitude": msg.longitude,
            "altitude": msg.altitude,
            "utm_easting": msg.utm_easting,
            "utm_northing": msg.utm_northing,
            "zone": msg.zone,
            "letter": msg.letter,
            "hdop": msg.hdop,
            "raw_gpgga": msg.gpgga_read,
        })
    return pd.DataFrame(rows)


def convert(bagdir: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    with AnyReader([bagdir]) as reader:
        imu_df = extract_imu(reader)
        gps_df = extract_gps(reader)

    if not imu_df.empty:
        imu_path = outdir / "imu.csv"
        imu_df.to_csv(imu_path, index=False)
        duration = imu_df["t"].iloc[-1] - imu_df["t"].iloc[0]
        rate = (len(imu_df) - 1) / duration if duration > 0 else float("nan")
        print(f"  imu.csv:  {len(imu_df):>6} rows, {duration:6.2f} s, {rate:5.1f} Hz -> {imu_path}")
    else:
        print("  imu.csv: (no /imu messages found)")

    if not gps_df.empty:
        gps_path = outdir / "gps.csv"
        gps_df.to_csv(gps_path, index=False)
        duration = gps_df["t"].iloc[-1] - gps_df["t"].iloc[0]
        rate = (len(gps_df) - 1) / duration if duration > 0 else float("nan")
        print(f"  gps.csv:  {len(gps_df):>6} rows, {duration:6.2f} s, {rate:5.1f} Hz -> {gps_path}")
    else:
        print("  gps.csv: (no /gps messages found)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag", type=Path, help="Path to bag directory (containing .mcap + metadata.yaml)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: build/<bag_basename>)")
    args = parser.parse_args()

    bagdir = args.bag.resolve()
    if not (bagdir / "metadata.yaml").exists():
        raise SystemExit(f"error: {bagdir} does not look like a ROS2 bag (no metadata.yaml)")

    outdir = args.out.resolve() if args.out else Path("build") / bagdir.name
    print(f"converting {bagdir.name}")
    convert(bagdir, outdir)


if __name__ == "__main__":
    main()
