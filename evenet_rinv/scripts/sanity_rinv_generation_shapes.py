#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise SystemExit("sanity_rinv_generation_shapes.py requires torch in the active Python environment") from exc

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "repos" / "Core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from evenet.network.heads.generation.generation_head import EventGenerationHead
from evenet.utilities.diffusion_sampler import get_logsnr_alpha_sigma
from evenet_rinv.datasets.svj_rinv_generation import REQUIRED_COLUMNS, SVJRinvGenerationDataset


def make_synthetic_csv(path: Path, n: int) -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "ph_pt": rng.uniform(20.0, 500.0, n),
            "ph_eta": rng.uniform(-2.4, 2.4, n),
            "ph_phi": rng.uniform(-np.pi, np.pi, n),
            "ph_m": np.zeros(n),
            "lead_jet_pt": rng.uniform(30.0, 700.0, n),
            "lead_jet_eta": rng.uniform(-2.4, 2.4, n),
            "lead_jet_phi": rng.uniform(-np.pi, np.pi, n),
            "lead_jet_m": rng.uniform(5.0, 80.0, n),
            "sub_jet_pt": rng.uniform(20.0, 400.0, n),
            "sub_jet_eta": rng.uniform(-2.4, 2.4, n),
            "sub_jet_phi": rng.uniform(-np.pi, np.pi, n),
            "sub_jet_m": rng.uniform(5.0, 70.0, n),
            "MET": rng.uniform(0.0, 800.0, n),
            "met_phi": rng.uniform(-np.pi, np.pi, n),
            "rinv": rng.uniform(-1.0, 2.0, n),
        }
    )
    df.loc[:, REQUIRED_COLUMNS].to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shape sanity test for SVJ rinv EventGenerationHead path.")
    parser.add_argument("--csv", default=None, help="Optional SVJ rinv CSV. Synthetic data is used if omitted.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(args.csv) if args.csv else Path(tmp) / "synthetic_svj_rinv.csv"
        if args.csv is None:
            make_synthetic_csv(csv_path, max(args.batch_size * 2, 16))

        dataset = SVJRinvGenerationDataset(csv_path, fit_scaler=True)
        print(dataset.report.to_text())
        batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))

        visible_tokens = batch["visible_tokens"].float()
        rinv_clean = batch["rinv_clean"].float()
        x_mask = batch["rinv_mask"].float()
        label = batch["label"].long()

        time = torch.rand((rinv_clean.shape[0],), dtype=rinv_clean.dtype)
        noise = torch.randn_like(rinv_clean)
        _, alpha, sigma = get_logsnr_alpha_sigma(time)
        alpha = alpha.view(-1, 1, 1)
        sigma = sigma.view(-1, 1, 1)
        x_noised = alpha * rinv_clean + sigma * noise

        global_cond = batch["conditions"].float().unsqueeze(1)
        global_cond_mask = torch.ones((visible_tokens.shape[0], 1, 1), dtype=visible_tokens.dtype)

        head = EventGenerationHead(
            input_dim=1,
            projection_dim=args.projection_dim,
            num_global_cond=global_cond.shape[-1],
            num_classes=1,
            output_dim=1,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=0.0,
            layer_scale=True,
            layer_scale_init=1.0e-5,
            drop_probability=0.0,
            feature_drop=0.0,
            position_encode=False,
            max_position_length=4,
        )
        head.eval()
        with torch.no_grad():
            model_output = head(
                x=x_noised,
                global_cond=global_cond,
                global_cond_mask=global_cond_mask,
                num_x=None,
                x_mask=x_mask,
                time=time,
                label=label,
            )

        print(f"visible_tokens.shape = {tuple(visible_tokens.shape)}")
        print(f"rinv_clean.shape = {tuple(rinv_clean.shape)}")
        print(f"x_noised.shape = {tuple(x_noised.shape)}")
        print(f"x_mask.shape = {tuple(x_mask.shape)}")
        print(f"global_cond.shape = {tuple(global_cond.shape)}")
        print(f"model_output.shape = {tuple(model_output.shape)}")


if __name__ == "__main__":
    main()
