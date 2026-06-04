#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evenet_rinv.datasets.svj_rinv_generation import (
    DEFAULT_RINV_MAX,
    DEFAULT_RINV_MIN,
    SVJRinvScaler,
    build_evenet_generation_arrays,
    build_visible_tokens,
    split_indices,
    validate_svj_rinv_dataframe,
)


def save_npz(path: Path, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: value[indices] for key, value in arrays.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SVJ rinv CSV to EveNet TruthGeneration NPZ files.")
    parser.add_argument("--csv", required=True, help="Input CSV with SVJ visible objects and rinv.")
    parser.add_argument("--out-dir", required=True, help="Directory for rinv_generation_{split}.npz files.")
    parser.add_argument("--scaler-path", default=None, help="Where to save train-fitted visible scaler JSON.")
    parser.add_argument("--report-path", default=None, help="Where to save the CSV validation/conversion report JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for sanity runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--rinv-min", type=float, default=DEFAULT_RINV_MIN)
    parser.add_argument("--rinv-max", type=float, default=DEFAULT_RINV_MAX)
    parser.add_argument("--fail-on-invalid", action="store_true", help="Report invalid rows instead of dropping them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.csv, nrows=args.limit)
    df, report = validate_svj_rinv_dataframe(
        raw,
        csv_path=args.csv,
        drop_invalid=not args.fail_on_invalid,
        rinv_min=args.rinv_min,
        rinv_max=args.rinv_max,
    )
    splits = split_indices(
        len(df),
        (args.train_frac, args.val_frac, args.test_frac),
        args.seed,
    )

    train_visible, train_visible_mask = build_visible_tokens(df.iloc[splits["train"]].reset_index(drop=True))
    scaler = SVJRinvScaler.fit(train_visible, train_visible_mask)
    scaler_path = Path(args.scaler_path) if args.scaler_path else Path(args.out_dir) / "rinv_generation_scaler.json"
    scaler.save(scaler_path)

    arrays = build_evenet_generation_arrays(
        df,
        normalize_visible=False,
        rinv_min=args.rinv_min,
        rinv_max=args.rinv_max,
    )
    out_dir = Path(args.out_dir)
    for split_name, indices in splits.items():
        save_npz(out_dir / f"rinv_generation_{split_name}.npz", arrays, indices)

    summary = {
        "data_check": asdict(report),
        "scaler_path": str(scaler_path),
        "npz_files": {
            split_name: str(out_dir / f"rinv_generation_{split_name}.npz")
            for split_name in splits
        },
        "splits": {split_name: int(len(indices)) for split_name, indices in splits.items()},
        "visible_feature_order": scaler.feature_names,
        "condition_feature_order": ["MET", "met_phi"],
        "invisible_target": (
            f"rinv_scaled = 2 * (rinv - {args.rinv_min}) / "
            f"({args.rinv_max} - {args.rinv_min}) - 1"
        ),
        "rinv_inverse_transform": (
            f"rinv = 0.5 * (rinv_scaled + 1) * "
            f"({args.rinv_max} - {args.rinv_min}) + {args.rinv_min}"
        ),
        "evenet_keys": sorted(arrays.keys()),
    }

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report.to_text())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
