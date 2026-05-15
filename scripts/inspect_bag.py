"""One-off introspection: dump the schema and first message of /imu and /gps topics."""

import sys
from pathlib import Path
from pprint import pprint

from rosbags.highlevel import AnyReader


def describe(obj, indent=0):
    pad = "  " * indent
    if hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            val = getattr(obj, slot)
            type_name = type(val).__name__
            if hasattr(val, "__slots__"):
                print(f"{pad}{slot}: {type_name}")
                describe(val, indent + 1)
            else:
                preview = repr(val)
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                print(f"{pad}{slot}: {type_name} = {preview}")
    else:
        print(f"{pad}{obj!r}")


def main(bagdir: Path):
    print(f"=== inspecting {bagdir} ===\n")
    with AnyReader([bagdir]) as reader:
        for conn in reader.connections:
            print(f"--- topic {conn.topic} (type {conn.msgtype}) ---")
            for _, _, rawdata in reader.messages(connections=[conn]):
                msg = reader.deserialize(rawdata, conn.msgtype)
                describe(msg)
                break
            print()


if __name__ == "__main__":
    bagdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/idle_car")
    main(bagdir)
