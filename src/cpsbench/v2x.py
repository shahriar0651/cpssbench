"""V2X / MisbehaviorX window dataset.

Automatic download is not available. If curated CSVs already exist under
``<root>/misbehaviorx/{ambient,attacks}/generated``, they are converted to the
same window format as the CAN datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .specs import DatasetSpec


def prepare_v2x_split(
    spec: DatasetSpec,
    split_path: Path,
    scaler_path: Path,
    fit_scaler: bool,
) -> None:
    split_path = Path(split_path)
    generated = split_path / "generated"
    csvs = sorted(generated.glob("*.csv"))
    if not csvs and not list(generated.glob("sig_*.npy")):
        raise FileNotFoundError(
            f"MisbehaviorX files were not found in {generated}. "
            "Automatic download is not implemented; place curated CSVs there first."
        )

    if fit_scaler and csvs and not scaler_path.exists():
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        overall_min = None
        overall_max = None
        for csv_path in csvs:
            frame = pd.read_csv(csv_path, index_col=0)
            current_min = frame[list(spec.features)].min(axis=0)
            current_max = frame[list(spec.features)].max(axis=0)
            overall_min = current_min if overall_min is None else np.minimum(overall_min, current_min)
            overall_max = current_max if overall_max is None else np.maximum(overall_max, current_max)
        pd.DataFrame({"Min": overall_min, "Max": overall_max}).to_csv(scaler_path)

    if list(generated.glob("sig_*.npy")):
        return
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing scaler {scaler_path}. Prepare the train/ambient split first."
        )

    for csv_path in csvs:
        frame = pd.read_csv(csv_path, index_col=0)
        if not fit_scaler and "attack_name" in frame.columns:
            frame = frame[frame["attack_name"] != "No Attack"]
        signals = frame[list(spec.features)]
        attributes = frame[list(spec.attributes)].copy()
        if spec.filling == "forward":
            signals = signals.ffill().bfill()
        if "attack_name" in attributes.columns:
            attributes["attack_name"], categories = pd.factorize(attributes["attack_name"])
            mapping = {str(category): int(code) for code, category in enumerate(categories)}
            with open(split_path / "file_index_dict.json", "w", encoding="utf-8") as handle:
                json.dump(mapping, handle, indent=2)
        np.save(generated / f"sig_{csv_path.stem}_1.npy", signals.to_numpy())
        np.save(generated / f"att_{csv_path.stem}_1.npy", attributes.to_numpy())


class V2XWindowDataset(Dataset):
    """Windowed V2X dataset. Returns ``(window, label)`` like MNIST."""

    def __init__(
        self,
        spec: DatasetSpec,
        data_dir: str | Path,
        scaler_path: str | Path,
        return_meta: bool = False,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self.data_dir = Path(data_dir)
        self.return_meta = return_meta
        self.features = list(spec.features)
        generated = self.data_dir / "generated"
        self.sig_files = sorted(generated.glob("sig_*.npy"))
        self.att_files = sorted(generated.glob("att_*.npy"))
        if not self.sig_files or len(self.sig_files) != len(self.att_files):
            raise FileNotFoundError(f"Expected paired npy files in {generated}.")

        table = pd.read_csv(scaler_path, index_col=0)
        self.min_vals = table["Min"].loc[self.features].to_numpy(dtype=np.float32)
        maxs = table["Max"].loc[self.features].to_numpy(dtype=np.float32)
        span = np.where(maxs - self.min_vals == 0, 1.0, maxs - self.min_vals)
        self.max_vals = self.min_vals + span

        with open(self.data_dir / "file_index_dict.json", encoding="utf-8") as handle:
            mapping = json.load(handle)
        self.file_names = {int(value): key for key, value in mapping.items()}
        self.index_map = self._index_windows()
        if verbose:
            print(f"{spec.name}: {len(self.index_map)} windows from {self.data_dir}")

    def _index_windows(self) -> list[tuple[int, int]]:
        index_map: list[tuple[int, int]] = []
        for file_idx, (sig_file, att_file) in enumerate(zip(self.sig_files, self.att_files)):
            sig = np.load(sig_file, mmap_mode="r")
            att = pd.DataFrame(np.load(att_file, mmap_mode="r"), columns=list(self.spec.attributes))
            groups = att[["id", "time_chunk", "attack_name"]].drop_duplicates()
            for _, row in groups.iterrows():
                mask = (
                    (att["id"] == row["id"])
                    & (att["time_chunk"] == row["time_chunk"])
                    & (att["attack_name"] == row["attack_name"])
                )
                indices = att.index[mask]
                if len(indices) == 0 or indices[-1] - indices[0] + 1 != len(indices):
                    continue
                n_windows = (len(indices) - self.spec.window_size) // self.spec.step_size + 1
                start = int(indices[0])
                for offset in range(max(n_windows, 0)):
                    index_map.append((file_idx, start + offset * self.spec.step_size))
        return index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        file_idx, start = self.index_map[idx]
        stop = start + self.spec.window_size
        sig = np.load(self.sig_files[file_idx], mmap_mode="r")
        att = np.load(self.att_files[file_idx], mmap_mode="r")
        window = (sig[start:stop] - self.min_vals) / (self.max_vals - self.min_vals)
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        label = int(np.sum(att[start:stop, self.spec.label_index]) > 0.0)
        if not self.return_meta:
            return tensor, label
        file_id = int(att[start, 3])
        meta = {"idx": int(start), "file": self.file_names.get(file_id, str(file_id))}
        return tensor, label, meta
