#!/usr/bin/env python3
"""Inspect sample .pkl and .npz files and print readable summaries.

Example:
    python scripts/inspect_data.py
    python scripts/inspect_data.py --pkl-dir source/motion/hightorque/pi_plus/pkl --npz-dir source/motion/hightorque/pi_plus/npz

This prints a concise tree for the .pkl (showing dict keys and array shapes) and
the contents of a sample .npz (keys, shapes, dtypes).
"""

from __future__ import annotations
import os
import argparse
import pickle
import csv
from typing import Any, Tuple
import numpy as np


PKL_DIR = "source/motion/hightorque/pi_plus/pkl"
NPZ_DIR = "source/motion/hightorque/pi_plus/npz"
CSV_DIR = "source/motion/hightorque/pi_plus/csv"


def safe_load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def inspect_obj(obj: Any, name: str = "root", max_items=8, indent: int = 0) -> None:
    prefix = " " * indent
    t = type(obj)
    if isinstance(obj, dict):
        print(f"{prefix}{name}: dict ({len(obj)} keys)")
        for i, k in enumerate(sorted(obj.keys())):
            if i >= max_items:
                print(f"{prefix}  ... ({len(obj)-max_items} more keys)")
                break
            inspect_obj(obj[k], name=str(k), max_items=max_items, indent=indent + 2)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {t.__name__} (len={len(obj)})")
        for i, v in enumerate(obj[:max_items]):
            inspect_obj(v, name=f"{name}[{i}]", max_items=max_items, indent=indent + 2)
        if len(obj) > max_items:
            print(f"{prefix}  ... ({len(obj)-max_items} more items)")
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        try:
            s = repr(obj)
            if len(s) > 120:
                s = s[:120] + "..."
        except Exception:
            s = f"<{t.__name__}>"
        print(f"{prefix}{name}: {t.__name__} {s}")


def inspect_sample_files(pkl_dir: str = PKL_DIR, npz_dir: str = NPZ_DIR, csv_dir: str = CSV_DIR) -> Tuple[str, str, str]:
    print("Searching sample files...")
    sample_pkl = None
    sample_npz = None
    sample_csv = None
    if os.path.isdir(pkl_dir):
        for fn in sorted(os.listdir(pkl_dir)):
            if fn.endswith(".pkl"):
                sample_pkl = os.path.join(pkl_dir, fn)
                break
    if os.path.isdir(npz_dir):
        for fn in sorted(os.listdir(npz_dir)):
            if fn.endswith(".npz"):
                sample_npz = os.path.join(npz_dir, fn)
                break
    if os.path.isdir(csv_dir):
        for fn in sorted(os.listdir(csv_dir)):
            if fn.endswith(".csv"):
                sample_csv = os.path.join(csv_dir, fn)
                break

    print(f"Found sample pkl: {sample_pkl}")
    print(f"Found sample npz: {sample_npz}")
    print(f"Found sample csv: {sample_csv}")
    if sample_pkl:
        print("\n--- Inspecting sample .pkl ---")
        obj = safe_load_pkl(sample_pkl)
        inspect_obj(obj, "pkl_root")
    if sample_npz:
        print("\n--- Inspecting sample .npz ---")
        try:
            with np.load(sample_npz, allow_pickle=True) as z:
                for k in sorted(z.files):
                    v = z[k]
                    if isinstance(v, np.ndarray):
                        print(f"npz -> {k}: ndarray shape={v.shape} dtype={v.dtype}")
                    else:
                        print(f"npz -> {k}: {type(v).__name__} repr={repr(v)[:120]}")
        except Exception as e:
            print(f"Failed to read npz {sample_npz}: {e}")

    if sample_csv:
        print("\n--- Inspecting sample .csv ---")
        try:
            with open(sample_csv, "r", newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
                try:
                    first = next(reader)
                except StopIteration:
                    first = None
            # count rows
            with open(sample_csv, "r", newline='') as f:
                total_lines = sum(1 for _ in f)
            rows = max(0, total_lines - (1 if header else 0))
            print(f"csv -> {os.path.basename(sample_csv)}: columns={len(header)} rows={rows}")
            if header:
                print(f"csv header: {', '.join(h.strip() for h in header)}")
            if first is not None:
                print(f"csv first row (truncated): {', '.join(first[:10])}{('...' if len(first)>10 else '')}")
        except Exception as e:
            print(f"Failed to read csv {sample_csv}: {e}")

    return sample_pkl, sample_npz, sample_csv


def main():
    parser = argparse.ArgumentParser(description="Inspect sample .pkl and .npz files under default directories.")
    parser.add_argument("--pkl-dir", type=str, default=PKL_DIR, help="Directory with .pkl files")
    parser.add_argument("--npz-dir", type=str, default=NPZ_DIR, help="Directory with .npz files")
    parser.add_argument("--csv-dir", type=str, default=CSV_DIR, help="Directory with .csv files")
    parser.add_argument("--max-items", type=int, default=8, help="Max items to show in lists/dicts")
    args = parser.parse_args()

    # expose max-items to inspect_obj by temporarily wrapping (simple approach)
    global inspect_obj

    def inspect_obj_wrapped(obj: Any, name: str = "root", max_items_local=args.max_items, indent: int = 0) -> None:
        return inspect_obj.__wrapped__(obj, name=name, max_items=max_items_local, indent=indent) if hasattr(inspect_obj, "__wrapped__") else inspect_obj(obj, name=name, max_items=max_items_local, indent=indent)

    # call inspection (includes csv)
    inspect_sample_files(args.pkl_dir, args.npz_dir, args.csv_dir)


if __name__ == "__main__":
    main()
