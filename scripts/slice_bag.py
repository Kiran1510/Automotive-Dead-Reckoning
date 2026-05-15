"""Slice the leading N seconds of a ROS2 .mcap bag into a new bag.

Used to produce tiny fixtures for CI from the full recording bags.

Usage:
    python scripts/slice_bag.py <src_bag_dir> <duration_s> <out_bag_dir>
"""

import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import StoragePlugin, Writer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="Source bag directory")
    parser.add_argument("duration", type=float, help="Seconds to keep from start")
    parser.add_argument("dst", type=Path, help="Output bag directory (will be created)")
    args = parser.parse_args()

    if args.dst.exists():
        raise SystemExit(f"error: {args.dst} exists -- remove it first")

    duration_ns = int(args.duration * 1e9)
    kept = 0

    with AnyReader([args.src]) as reader:
        with Writer(args.dst, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
            conn_map = {}
            for src_conn in reader.connections:
                new_conn = writer.add_connection(
                    topic=src_conn.topic,
                    msgtype=src_conn.msgtype,
                    typestore=reader.typestore,
                )
                conn_map[src_conn.id] = new_conn

            start_ts = None
            for src_conn, ts, raw in reader.messages():
                if start_ts is None:
                    start_ts = ts
                if ts - start_ts > duration_ns:
                    break
                writer.write(conn_map[src_conn.id], ts, raw)
                kept += 1

    print(f"wrote {kept} messages over {args.duration}s -> {args.dst}")


if __name__ == "__main__":
    main()
