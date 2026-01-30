#!/usr/bin/env python3
"""
JAAD/PIE preprocessing template.

This script converts per-frame pedestrian annotations into the payload expected by
`JAADPIEDataset` in dataset_loader.py. It assumes a CSV input with at least:

  scene_id, frame_id, track_id, x, y

Where x/y are trajectory coordinates (e.g., bbox center in pixels).
The script builds fixed-length sequences and outputs a pickle containing:

  {
    "trajectories": np.ndarray [num_samples, num_agents, seq_len, 2],
    "masks": np.ndarray [num_samples, num_agents, num_agents],
    "seq_start_end": list[list[tuple[int, int]]],
    "initial_pos": np.ndarray [num_samples, num_agents, 2],
  }

You can adapt `load_records` if your JAAD/PIE export differs (JSON, MAT, etc.).
"""

import argparse
import csv
import os
import pickle
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare JAAD/PIE trajectories for PPT.")
    parser.add_argument("--input_csv", required=True, help="CSV with scene/frame/track/x/y columns.")
    parser.add_argument("--output_dir", required=True, help="Output directory for train/test.pkl.")
    parser.add_argument("--split", default="train", choices=["train", "test", "val"])
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride.")
    parser.add_argument("--min_agents", type=int, default=1, help="Minimum agents per sample.")

    parser.add_argument("--scene_col", default="scene_id")
    parser.add_argument("--frame_col", default="frame_id")
    parser.add_argument("--track_col", default="track_id")
    parser.add_argument("--x_col", default="x")
    parser.add_argument("--y_col", default="y")
    parser.add_argument("--scale", type=float, default=1.0, help="Optional coordinate scaling.")

    return parser.parse_args()


def load_records(csv_path, scene_col, frame_col, track_col, x_col, y_col, scale):
    scenes = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = row[scene_col]
            frame_id = int(row[frame_col])
            track_id = row[track_col]
            x = float(row[x_col]) * scale
            y = float(row[y_col]) * scale
            scenes[scene_id][frame_id].append((track_id, x, y))
    return scenes


def build_sequences(scene_frames, seq_len, obs_len, stride, min_agents):
    trajectories = []
    masks = []
    seq_start_end = []
    initial_pos = []

    frames_sorted = sorted(scene_frames.keys())
    if not frames_sorted:
        return trajectories, masks, seq_start_end, initial_pos

    frame_to_agents = {}
    for frame_id in frames_sorted:
        frame_to_agents[frame_id] = {track_id: (x, y) for track_id, x, y in scene_frames[frame_id]}

    for start_idx in range(0, len(frames_sorted) - seq_len + 1, stride):
        window_frames = frames_sorted[start_idx:start_idx + seq_len]
        if window_frames[-1] - window_frames[0] != seq_len - 1:
            continue

        common_agents = None
        for frame_id in window_frames:
            agents = set(frame_to_agents[frame_id].keys())
            common_agents = agents if common_agents is None else common_agents & agents
        if not common_agents or len(common_agents) < min_agents:
            continue

        agent_ids = sorted(common_agents)
        sample = np.zeros((len(agent_ids), seq_len, 2), dtype=np.float32)
        for t, frame_id in enumerate(window_frames):
            for i, agent_id in enumerate(agent_ids):
                sample[i, t] = frame_to_agents[frame_id][agent_id]

        trajectories.append(sample)
        mask = np.ones((len(agent_ids), len(agent_ids)), dtype=np.float32)
        masks.append(mask)
        seq_start_end.append([(0, len(agent_ids))])
        initial_pos.append(sample[:, obs_len - 1])

    return trajectories, masks, seq_start_end, initial_pos


def main():
    args = parse_args()
    seq_len = args.obs_len + args.pred_len

    scenes = load_records(
        args.input_csv,
        args.scene_col,
        args.frame_col,
        args.track_col,
        args.x_col,
        args.y_col,
        args.scale,
    )

    all_traj = []
    all_masks = []
    all_seq_start_end = []
    all_initial_pos = []

    for _, scene_frames in scenes.items():
        traj, masks, seq_start_end, initial_pos = build_sequences(
            scene_frames, seq_len, args.obs_len, args.stride, args.min_agents
        )
        all_traj.extend(traj)
        all_masks.extend(masks)
        all_seq_start_end.extend(seq_start_end)
        all_initial_pos.extend(initial_pos)

    output = {
        "trajectories": np.asarray(all_traj, dtype=object),
        "masks": np.asarray(all_masks, dtype=object),
        "seq_start_end": all_seq_start_end,
        "initial_pos": np.asarray(all_initial_pos, dtype=object),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.split}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved {len(all_traj)} samples to {output_path}")


if __name__ == "__main__":
    main()
