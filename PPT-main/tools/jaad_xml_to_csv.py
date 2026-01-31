#!/usr/bin/env python3
"""Convert JAAD annotation XML files into CSV for prepare_jaad_pie.py.

Expected JAAD XML schema (from annotations/*.xml):
- <annotations> root
- <meta><task><name>video_XXXX</name></task></meta> for scene id
- <meta><task><original_size><width>...</width><height>...</height></original_size></task></meta>
- <track label="pedestrian"> ... <box frame="..." xtl="..." ytl="..." xbr="..." ybr="..."> ...
    <attribute name="id">track_id</attribute>
  </box>

The script computes bbox centers, normalizes by image width/height by default,
and writes CSV with columns: scene_id, frame_id, track_id, x, y.
"""

import argparse
import csv
import os
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JAAD XML annotations to CSV.")
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="One or more JAAD XML files or directories containing XML files.",
    )
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--label_filter",
        default="pedestrian,ped,people",
        help="Comma-separated track labels to keep.",
    )
    parser.add_argument("--width", type=float, default=None, help="Override image width.")
    parser.add_argument("--height", type=float, default=None, help="Override image height.")
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable image-size normalization.",
    )
    return parser.parse_args()


def iter_xml_files(input_paths):
    xml_files = []
    for path in input_paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(".xml"):
                    xml_files.append(os.path.join(path, name))
        else:
            xml_files.append(path)
    return xml_files


def get_scene_id(root, fallback_name):
    name = root.findtext("./meta/task/name")
    if name:
        return name.strip()
    return os.path.splitext(os.path.basename(fallback_name))[0]


def get_image_size(root, fallback_width, fallback_height):
    width_text = root.findtext("./meta/task/original_size/width")
    height_text = root.findtext("./meta/task/original_size/height")
    width = float(width_text) if width_text else fallback_width
    height = float(height_text) if height_text else fallback_height
    if width is None or height is None:
        raise ValueError("Image width/height not found in XML; provide --width/--height.")
    return width, height


def extract_track_id(box, track, track_index, scene_id):
    for attr in box.findall("attribute"):
        if attr.get("name") == "id" and attr.text:
            return attr.text.strip()
    if track.get("id"):
        return track.get("id")
    return f"{scene_id}_track_{track_index}"


def parse_xml(xml_path, label_filter, fallback_width, fallback_height, normalize):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    scene_id = get_scene_id(root, xml_path)
    width, height = get_image_size(root, fallback_width, fallback_height)

    records = []
    for track_index, track in enumerate(root.findall("track")):
        label = (track.get("label") or "").strip()
        if label_filter and label not in label_filter:
            continue
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = box.get("frame")
            xtl = box.get("xtl")
            ytl = box.get("ytl")
            xbr = box.get("xbr")
            ybr = box.get("ybr")
            if frame is None or xtl is None or ytl is None or xbr is None or ybr is None:
                continue

            x_center = (float(xtl) + float(xbr)) / 2.0
            y_center = (float(ytl) + float(ybr)) / 2.0
            if normalize:
                x_center /= width
                y_center /= height

            track_id = extract_track_id(box, track, track_index, scene_id)
            records.append((scene_id, int(frame), track_id, x_center, y_center))

    return records


def main():
    args = parse_args()
    label_filter = {label.strip() for label in args.label_filter.split(",") if label.strip()}
    normalize = not args.no_normalize

    xml_files = iter_xml_files(args.input_paths)
    if not xml_files:
        raise FileNotFoundError("No XML files found in input paths.")

    records = []
    for xml_path in xml_files:
        records.extend(
            parse_xml(
                xml_path,
                label_filter,
                args.width,
                args.height,
                normalize,
            )
        )

    records.sort(key=lambda r: (r[0], r[1], r[2]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "frame_id", "track_id", "x", "y"])
        writer.writerows(records)

    print(f"Wrote {len(records)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
