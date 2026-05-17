"""Overlay GPS + IMU dead-reckoned trajectories on real-world map tiles.

Reads:
  build/<dataset>/gps.csv         — ground-truth lat/lon
  build/<dataset>/trajectory.csv  — dead-reckoned xe/xn (UTM offsets from start)
  build/<dataset>/gps.csv         — first row's utm_easting / utm_northing / zone / letter
                                     to convert the trajectory back to lat/lon

Writes three artifacts side-by-side in build/<dataset>/:
  trajectory_on_map.html   interactive Folium / OpenStreetMap map (open in browser,
                            pan + zoom, hover for tooltips)
  trajectory_on_map.kml    KML export for Google My Maps, Google Earth, phone apps
  trajectory_on_map.png    static PNG rendered over OpenStreetMap tiles

Colour convention (matches the rest of the pipeline):
  red  = GPS UTM ground truth
  blue = IMU dead-reckoned (whichever yaw source was last used by trajectory.py)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import utm
import folium
from staticmap import StaticMap, Line, CircleMarker


def utm_offsets_to_latlon(xe, xn, start_e, start_n, zone_number, zone_letter):
    """Convert (xe, xn) UTM offsets from a start point into absolute lat/lon."""
    abs_e = start_e + np.asarray(xe)
    abs_n = start_n + np.asarray(xn)
    lats = np.empty_like(abs_e, dtype=float)
    lons = np.empty_like(abs_e, dtype=float)
    for i in range(len(abs_e)):
        lats[i], lons[i] = utm.to_latlon(abs_e[i], abs_n[i], zone_number, zone_letter)
    return lats, lons


def write_kml(path: Path, dataset: str, gps_latlon, imu_latlon):
    """Write a KML file with the two trajectories as separate LineStrings.

    KML colour format is aabbggrr (alpha, blue, green, red). The GPS trace is
    red, the IMU trace is blue, matching the matplotlib plots.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
        f'<name>{dataset} trajectories</name>',
        '<Style id="gps_style"><LineStyle><color>ff0000ff</color><width>4</width></LineStyle></Style>',
        '<Style id="imu_style"><LineStyle><color>ffff0000</color><width>3</width></LineStyle></Style>',

        '<Placemark><name>GPS (ground truth)</name><styleUrl>#gps_style</styleUrl>',
        '<LineString><tessellate>1</tessellate><coordinates>',
    ]
    for lat, lon in gps_latlon:
        lines.append(f'{lon},{lat},0')
    lines += [
        '</coordinates></LineString></Placemark>',

        '<Placemark><name>IMU dead-reckoned</name><styleUrl>#imu_style</styleUrl>',
        '<LineString><tessellate>1</tessellate><coordinates>',
    ]
    for lat, lon in imu_latlon:
        lines.append(f'{lon},{lat},0')
    lines += [
        '</coordinates></LineString></Placemark>',
        '</Document></kml>',
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--png-size", type=int, nargs=2, default=[1400, 1100],
                        help="Width and height in pixels for the PNG output")
    args = parser.parse_args()

    out_dir = args.build / args.dataset
    gps = pd.read_csv(out_dir / "gps.csv")
    traj = pd.read_csv(out_dir / "trajectory.csv")

    # GPS ground truth (already lat/lon).
    gps_latlon = list(zip(gps["latitude"].to_numpy(), gps["longitude"].to_numpy()))

    # Convert IMU trajectory UTM offsets back to lat/lon using the start GPS sample.
    start_e = float(gps["utm_easting"].iloc[0])
    start_n = float(gps["utm_northing"].iloc[0])
    zone_number = int(gps["zone"].iloc[0])
    zone_letter = str(gps["letter"].iloc[0])
    imu_lat, imu_lon = utm_offsets_to_latlon(
        traj["xe"].to_numpy(), traj["xn"].to_numpy(),
        start_e, start_n, zone_number, zone_letter,
    )
    imu_latlon = list(zip(imu_lat.tolist(), imu_lon.tolist()))

    # Stride the IMU trace down to ~2000 points for plotting (40 Hz x 41 min = 98k points
    # is way more than any map renderer benefits from).
    stride = max(1, len(imu_latlon) // 2000)
    imu_latlon_thin = imu_latlon[::stride]

    print(f"  {args.dataset}: {len(gps_latlon)} GPS points, "
          f"{len(imu_latlon_thin)} IMU points (downsampled from {len(imu_latlon)})")

    # ---- 1. Folium interactive HTML ----
    center_lat = float(np.median(gps["latitude"]))
    center_lon = float(np.median(gps["longitude"]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                      tiles="OpenStreetMap")
    folium.PolyLine(gps_latlon, color="red", weight=4, opacity=0.85,
                    tooltip="GPS (ground truth)").add_to(fmap)
    folium.PolyLine(imu_latlon_thin, color="blue", weight=3, opacity=0.85,
                    tooltip="IMU dead-reckoned").add_to(fmap)
    folium.Marker(gps_latlon[0], popup="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps_latlon[-1], popup="GPS end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    folium.Marker(imu_latlon[-1], popup="IMU end",
                  icon=folium.Icon(color="blue", icon="stop")).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    html_path = out_dir / "trajectory_on_map.html"
    fmap.save(str(html_path))
    print(f"  HTML: {html_path}  (open in browser; pan/zoom, hover for legend)")

    # ---- 2. KML export ----
    kml_path = out_dir / "trajectory_on_map.kml"
    write_kml(kml_path, args.dataset, gps_latlon, imu_latlon_thin)
    print(f"  KML:  {kml_path}  (load into Google Earth / My Maps / phone GPS apps)")

    # ---- 3. Static PNG over OSM tiles ----
    m = StaticMap(args.png_size[0], args.png_size[1], padding_x=40, padding_y=40)
    # staticmap takes (lon, lat) order, opposite of folium
    m.add_line(Line([(lon, lat) for lat, lon in gps_latlon], "red", 5))
    m.add_line(Line([(lon, lat) for lat, lon in imu_latlon_thin], "blue", 3))
    # Start / end markers
    m.add_marker(CircleMarker((gps_latlon[0][1], gps_latlon[0][0]), "green", 12))
    m.add_marker(CircleMarker((gps_latlon[-1][1], gps_latlon[-1][0]), "red", 12))
    m.add_marker(CircleMarker((imu_latlon[-1][1], imu_latlon[-1][0]), "blue", 12))
    img = m.render()
    png_path = out_dir / "trajectory_on_map.png"
    img.save(str(png_path))
    print(f"  PNG:  {png_path}")


if __name__ == "__main__":
    main()
