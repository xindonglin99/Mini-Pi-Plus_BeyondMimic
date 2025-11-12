from __future__ import annotations
import os
import argparse
import pickle
import csv
import pandas as pd
from typing import Any, List
import numpy as np

# Default dirs (relative to repo root)
PKL_DIR = "source/motion/hightorque/pi_plus/pkl"
CSV_DIR = "source/motion/hightorque/pi_plus/csv"


def safe_load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


PI_PLUS_JOINTS = [
    "l_hip_pitch",
    "l_hip_roll",
    "l_thigh",
    "l_calf",
    "l_ankle_pitch",
    "l_ankle_roll",
    "r_hip_pitch",
    "r_hip_roll",
    "r_thigh",
    "r_calf",
    "r_ankle_pitch",
    "r_ankle_roll",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_upper_arm",
    "l_elbow",
    "l_wrist",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_upper_arm",
    "r_elbow",
    "r_wrist",
]


def make_header(joint_names: List[str]) -> List[str]:
    header = [
        "root pos x",
        "root pos y",
        "root pos z",
        "root rot x",
        "root rot y",
        "root rot z",
        "root rot w",
    ]
    header.extend(joint_names)
    return header


def write_csv_from_pkl(pkl_path: str, csv_path: str, joint_names: List[str]) -> None:
    obj = safe_load_pkl(pkl_path)

    root_pos = obj.get("root_pos")
    root_rot = obj.get("root_rot")
    dof_pos = obj.get("dof_pos")

    if root_pos is None or root_rot is None or dof_pos is None:
        raise ValueError(f"Missing required keys in pkl: {pkl_path}")

    # Ensure matching lengths
    n = min(root_pos.shape[0], root_rot.shape[0], dof_pos.shape[0])
    if n != root_pos.shape[0] or n != dof_pos.shape[0] or n != root_rot.shape[0]:
        print(f"[warning] unequal lengths in {pkl_path}, trimming to {n} frames")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    header = make_header(joint_names)

    # Build a pandas DataFrame for robust column ordering and writing
    # Prepare columns
    data = {}
    data["root pos x"] = root_pos[:n, 0]
    data["root pos y"] = root_pos[:n, 1]
    data["root pos z"] = root_pos[:n, 2]
    # root_rot assumed to be [x,y,z,w] in PKL; keep same ordering in CSV
    data["root rot x"] = root_rot[:n, 0]
    data["root rot y"] = root_rot[:n, 1]
    data["root rot z"] = root_rot[:n, 2]
    data["root rot w"] = root_rot[:n, 3]

    dp = dof_pos[:n]
    # Ensure dp has correct shape (n, len(joint_names)) by trimming or padding
    if dp.shape[1] != len(joint_names):
        if dp.shape[1] > len(joint_names):
            dp = dp[:, : len(joint_names)]
        else:
            pad = np.zeros((dp.shape[0], len(joint_names) - dp.shape[1]), dtype=float)
            dp = np.concatenate([dp, pad], axis=1)

    for i, name in enumerate(joint_names):
        data[name] = dp[:, i]

    df = pd.DataFrame(data, columns=header)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Wrote CSV: {csv_path} (frames={n})")


def convert_all(pkl_dir: str = PKL_DIR, csv_dir: str = CSV_DIR, joint_names: List[str] = PI_PLUS_JOINTS, dry_run: bool = False) -> None:
    if not os.path.isdir(pkl_dir):
        print(f"No pkl dir: {pkl_dir}")
        return
    os.makedirs(csv_dir, exist_ok=True)
    files = sorted([fn for fn in os.listdir(pkl_dir) if fn.endswith(".pkl")])
    if not files:
        print("No .pkl files found to convert.")
        return
    for fn in files:
        pkl_path = os.path.join(pkl_dir, fn)
        out_fn = os.path.splitext(fn)[0] + ".csv"
        csv_path = os.path.join(csv_dir, out_fn)
        print(f"Converting: {pkl_path} -> {csv_path}")
        if dry_run:
            continue
        write_csv_from_pkl(pkl_path, csv_path, joint_names)


def main():
    parser = argparse.ArgumentParser(description="Convert PKL motion files to CSV matching sample CSV layout (pi_plus).")
    parser.add_argument("--pkl-dir", type=str, default=PKL_DIR, help="Directory with .pkl files")
    parser.add_argument("--csv-dir", type=str, default=CSV_DIR, help="Directory to write .csv files")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files; only print actions")
    parser.add_argument("--joint-names", type=str, help="Optional comma-separated joint names to use instead of default pi_plus list")
    args = parser.parse_args()

    joint_names = PI_PLUS_JOINTS
    if args.joint_names:
        joint_names = [s.strip() for s in args.joint_names.split(",")]

    convert_all(args.pkl_dir, args.csv_dir, joint_names, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
