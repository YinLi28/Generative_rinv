from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


REQUIRED_COLUMNS = [
    "ph_pt",
    "ph_phi",
    "ph_eta",
    "ph_m",
    "lead_jet_pt",
    "lead_jet_eta",
    "lead_jet_phi",
    "lead_jet_m",
    "sub_jet_pt",
    "sub_jet_eta",
    "sub_jet_phi",
    "sub_jet_m",
    "MET",
    "met_phi",
    "rinv",
]

PT_AND_MET_COLUMNS = ["ph_pt", "lead_jet_pt", "sub_jet_pt", "MET"]

VISIBLE_FEATURE_NAMES = [
    "energy",
    "pt",
    "eta",
    "phi",
    "btag_score",
    "is_lepton",
    "charge",
]

NUM_VISIBLE_SLOTS = 18
NUM_REAL_VISIBLE_OBJECTS = 3
CONTINUOUS_VISIBLE_INDICES = [0, 1, 2, 3]
DEFAULT_RINV_MIN = 0.0
DEFAULT_RINV_MAX = 1.0


@dataclass
class DataCheckReport:
    csv: str | None
    input_rows: int
    used_rows: int
    dropped_rows: int
    missing_columns: list[str]
    nonfinite_rows: int
    rinv_out_of_range_rows: int
    negative_pt_or_met_rows: int
    rinv_min: float
    rinv_max: float

    def to_text(self) -> str:
        return (
            "SVJ rinv CSV check: "
            f"input={self.input_rows}, used={self.used_rows}, dropped={self.dropped_rows}, "
            f"missing={self.missing_columns or 'none'}, nonfinite={self.nonfinite_rows}, "
            f"rinv_range=[{self.rinv_min}, {self.rinv_max}], "
            f"rinv_out_of_range={self.rinv_out_of_range_rows}, "
            f"negative_pt_or_met={self.negative_pt_or_met_rows}"
        )


@dataclass
class SVJRinvScaler:
    visible_mean: list[float]
    visible_std: list[float]
    visible_norm_mask: list[bool]
    feature_names: list[str]

    @classmethod
    def fit(cls, visible_tokens: np.ndarray, visible_mask: np.ndarray | None = None) -> "SVJRinvScaler":
        mean = np.zeros(visible_tokens.shape[-1], dtype=np.float64)
        std = np.ones(visible_tokens.shape[-1], dtype=np.float64)
        norm_mask = np.zeros(visible_tokens.shape[-1], dtype=bool)
        norm_mask[CONTINUOUS_VISIBLE_INDICES] = True

        if visible_mask is None:
            flattened = visible_tokens.reshape(-1, visible_tokens.shape[-1])
        else:
            mask = np.asarray(visible_mask).reshape(-1).astype(bool)
            flattened = visible_tokens.reshape(-1, visible_tokens.shape[-1])[mask]
        mean[norm_mask] = flattened[:, norm_mask].mean(axis=0)
        std[norm_mask] = flattened[:, norm_mask].std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(
            visible_mean=mean.astype(float).tolist(),
            visible_std=std.astype(float).tolist(),
            visible_norm_mask=norm_mask.tolist(),
            feature_names=list(VISIBLE_FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SVJRinvScaler":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def transform_visible(self, visible_tokens: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.visible_mean, dtype=np.float32)
        std = np.asarray(self.visible_std, dtype=np.float32)
        mask = np.asarray(self.visible_norm_mask, dtype=bool)
        out = visible_tokens.astype(np.float32, copy=True)
        out[..., mask] = (out[..., mask] - mean[mask]) / std[mask]
        return out

    def inverse_transform_visible(self, visible_tokens: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.visible_mean, dtype=np.float32)
        std = np.asarray(self.visible_std, dtype=np.float32)
        mask = np.asarray(self.visible_norm_mask, dtype=bool)
        out = visible_tokens.astype(np.float32, copy=True)
        out[..., mask] = out[..., mask] * std[mask] + mean[mask]
        return out


def object_energy(pt: np.ndarray, eta: np.ndarray, mass: np.ndarray | float) -> np.ndarray:
    momentum = pt * np.cosh(eta)
    return np.sqrt(np.maximum(momentum * momentum + np.asarray(mass) * np.asarray(mass), 0.0))


def validate_svj_rinv_dataframe(
    df: pd.DataFrame,
    *,
    csv_path: str | Path | None = None,
    drop_invalid: bool = True,
    rinv_min: float = DEFAULT_RINV_MIN,
    rinv_max: float = DEFAULT_RINV_MAX,
) -> tuple[pd.DataFrame, DataCheckReport]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        report = DataCheckReport(
            csv=str(csv_path) if csv_path is not None else None,
            input_rows=len(df),
            used_rows=0,
            dropped_rows=len(df),
            missing_columns=missing,
            nonfinite_rows=0,
            rinv_out_of_range_rows=0,
            negative_pt_or_met_rows=0,
            rinv_min=float(rinv_min),
            rinv_max=float(rinv_max),
        )
        raise ValueError(report.to_text())

    numeric = df.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    rinv_mask = (numeric["rinv"] >= rinv_min) & (numeric["rinv"] <= rinv_max)
    pt_met_mask = (numeric.loc[:, PT_AND_MET_COLUMNS] >= 0.0).all(axis=1)
    valid_mask = finite_mask & rinv_mask.to_numpy() & pt_met_mask.to_numpy()

    report = DataCheckReport(
        csv=str(csv_path) if csv_path is not None else None,
        input_rows=len(df),
        used_rows=int(valid_mask.sum()),
        dropped_rows=int((~valid_mask).sum()),
        missing_columns=[],
        nonfinite_rows=int((~finite_mask).sum()),
        rinv_out_of_range_rows=int((finite_mask & ~rinv_mask.to_numpy()).sum()),
        negative_pt_or_met_rows=int((finite_mask & ~pt_met_mask.to_numpy()).sum()),
        rinv_min=float(rinv_min),
        rinv_max=float(rinv_max),
    )

    if report.dropped_rows and not drop_invalid:
        raise ValueError(report.to_text())

    clean = numeric.loc[valid_mask, REQUIRED_COLUMNS].reset_index(drop=True)
    return clean, report


def scale_rinv_to_unit_interval(
    rinv: np.ndarray,
    *,
    rinv_min: float = DEFAULT_RINV_MIN,
    rinv_max: float = DEFAULT_RINV_MAX,
) -> np.ndarray:
    if rinv_max <= rinv_min:
        raise ValueError(f"rinv_max must be larger than rinv_min, got {rinv_min}, {rinv_max}")
    return (2.0 * (rinv - rinv_min) / (rinv_max - rinv_min)) - 1.0


def unscale_rinv_from_unit_interval(
    rinv_scaled: np.ndarray,
    *,
    rinv_min: float = DEFAULT_RINV_MIN,
    rinv_max: float = DEFAULT_RINV_MAX,
) -> np.ndarray:
    if rinv_max <= rinv_min:
        raise ValueError(f"rinv_max must be larger than rinv_min, got {rinv_min}, {rinv_max}")
    return ((rinv_scaled + 1.0) * 0.5 * (rinv_max - rinv_min)) + rinv_min


def build_visible_tokens(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    tokens = np.zeros((n, NUM_VISIBLE_SLOTS, len(VISIBLE_FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((n, NUM_VISIBLE_SLOTS, 1), dtype=np.float32)
    mask[:, :NUM_REAL_VISIBLE_OBJECTS, 0] = 1.0

    specs = [
        ("ph", "ph_pt", "ph_eta", "ph_phi", "ph_m", 1.0),
        ("lead", "lead_jet_pt", "lead_jet_eta", "lead_jet_phi", "lead_jet_m", 0.0),
        ("sub", "sub_jet_pt", "sub_jet_eta", "sub_jet_phi", "sub_jet_m", 0.0),
    ]
    for token_idx, (_, pt_col, eta_col, phi_col, mass_col, is_lepton) in enumerate(specs):
        pt = df[pt_col].to_numpy(np.float32)
        eta = df[eta_col].to_numpy(np.float32)
        phi = df[phi_col].to_numpy(np.float32)
        mass = df[mass_col].to_numpy(np.float32)
        tokens[:, token_idx, 0] = object_energy(pt, eta, mass)
        tokens[:, token_idx, 1] = pt
        tokens[:, token_idx, 2] = eta
        tokens[:, token_idx, 3] = phi
        tokens[:, token_idx, 4] = 0.0
        tokens[:, token_idx, 5] = is_lepton
        tokens[:, token_idx, 6] = 0.0

    return tokens, mask


def build_evenet_generation_arrays(
    df: pd.DataFrame,
    *,
    normalize_visible: bool = False,
    scaler: SVJRinvScaler | None = None,
    rinv_min: float = DEFAULT_RINV_MIN,
    rinv_max: float = DEFAULT_RINV_MAX,
) -> dict[str, np.ndarray]:
    visible_tokens, visible_mask = build_visible_tokens(df)
    if normalize_visible:
        if scaler is None:
            raise ValueError("normalize_visible=True requires a fitted scaler")
        visible_tokens = scaler.transform_visible(visible_tokens)

    rinv = df["rinv"].to_numpy(np.float32).reshape(-1, 1, 1)
    rinv_scaled = scale_rinv_to_unit_interval(rinv, rinv_min=rinv_min, rinv_max=rinv_max)
    n = len(df)
    conditions = np.stack(
        [
            df["MET"].to_numpy(np.float32),
            df["met_phi"].to_numpy(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "x": visible_tokens.astype(np.float32),
        "x_mask": visible_mask.squeeze(-1).astype(bool),
        "conditions": conditions,
        "conditions_mask": np.ones((n, 1), dtype=bool),
        "num_vectors": np.full(n, NUM_REAL_VISIBLE_OBJECTS, dtype=np.float32),
        "num_sequential_vectors": np.full(n, NUM_REAL_VISIBLE_OBJECTS, dtype=np.float32),
        "x_invisible": rinv_scaled.astype(np.float32),
        "x_invisible_mask": np.ones((n, 1), dtype=bool),
        "classification": np.zeros(n, dtype=np.int64),
        "event_weight": np.ones(n, dtype=np.float32),
        "EXTRA/rinv": rinv.reshape(-1).astype(np.float32),
        "EXTRA/rinv_scaled": rinv_scaled.reshape(-1).astype(np.float32),
    }


class SVJRinvGenerationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        *,
        scaler_path: str | Path | None = None,
        fit_scaler: bool = False,
        drop_invalid: bool = True,
        normalize_visible: bool = True,
        save_report_path: str | Path | None = None,
        rinv_min: float = DEFAULT_RINV_MIN,
        rinv_max: float = DEFAULT_RINV_MAX,
    ):
        self.csv_path = Path(csv_path)
        raw = pd.read_csv(self.csv_path)
        self.dataframe, self.report = validate_svj_rinv_dataframe(
            raw,
            csv_path=self.csv_path,
            drop_invalid=drop_invalid,
            rinv_min=rinv_min,
            rinv_max=rinv_max,
        )
        if save_report_path is not None:
            Path(save_report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_report_path).write_text(json.dumps(asdict(self.report), indent=2), encoding="utf-8")

        visible_tokens, visible_mask = build_visible_tokens(self.dataframe)
        self.scaler: SVJRinvScaler | None = None
        if normalize_visible:
            if fit_scaler:
                self.scaler = SVJRinvScaler.fit(visible_tokens, visible_mask)
                if scaler_path is not None:
                    self.scaler.save(scaler_path)
            elif scaler_path is not None:
                self.scaler = SVJRinvScaler.load(scaler_path)
            else:
                self.scaler = SVJRinvScaler.fit(visible_tokens, visible_mask)
            visible_tokens = self.scaler.transform_visible(visible_tokens)

        rinv = self.dataframe["rinv"].to_numpy(np.float32).reshape(-1, 1, 1)
        self.visible_tokens = visible_tokens.astype(np.float32)
        self.visible_mask = visible_mask.astype(np.float32)
        self.rinv_clean = scale_rinv_to_unit_interval(
            rinv,
            rinv_min=rinv_min,
            rinv_max=rinv_max,
        ).astype(np.float32)
        self.rinv_mask = np.ones_like(self.rinv_clean, dtype=np.float32)
        self.label = np.zeros((len(self.dataframe), 1), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if torch is None:
            raise RuntimeError("SVJRinvGenerationDataset requires torch for __getitem__")
        visible_tokens = torch.from_numpy(self.visible_tokens[index])
        visible_mask = torch.from_numpy(self.visible_mask[index])
        rinv_clean = torch.from_numpy(self.rinv_clean[index])
        rinv_mask = torch.from_numpy(self.rinv_mask[index])
        label = torch.from_numpy(self.label[index])
        return {
            "visible_tokens": visible_tokens,
            "visible_mask": visible_mask,
            "rinv_clean": rinv_clean,
            "rinv_mask": rinv_mask,
            "label": label,
            "x": visible_tokens,
            "x_mask": visible_mask.squeeze(-1).bool(),
            "x_invisible": rinv_clean,
            "x_invisible_mask": rinv_mask.squeeze(-1).bool(),
            "conditions": torch.tensor(
                [
                    self.dataframe["MET"].iloc[index],
                    self.dataframe["met_phi"].iloc[index],
                ],
                dtype=torch.float32,
            ),
            "conditions_mask": torch.ones(1, dtype=torch.bool),
            "classification": label.squeeze(-1).long(),
            "num_vectors": torch.tensor(float(NUM_REAL_VISIBLE_OBJECTS), dtype=torch.float32),
            "num_sequential_vectors": torch.tensor(float(NUM_REAL_VISIBLE_OBJECTS), dtype=torch.float32),
        }


def split_indices(n: int, fractions: Iterable[float], seed: int) -> dict[str, np.ndarray]:
    train_frac, val_frac, test_frac = tuple(fractions)
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {total}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
