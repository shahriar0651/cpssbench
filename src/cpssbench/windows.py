"""Windowed PyTorch datasets over preprocessed signal arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .specs import DatasetSpec


class WindowDataset(Dataset):
    """Sliding-window CAN dataset.

    ``__getitem__`` returns ``(window, label)`` like MNIST. Pass
    ``return_meta=True`` to also get ``{"file", "idx"}``.
    """

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
        self.scaler_path = Path(scaler_path)
        self.return_meta = return_meta
        self.features = list(spec.features)
        self.sampling_periods = np.asarray(spec.sampling_period_factors, dtype=int) * int(spec.sampling_period)
        if len(self.sampling_periods) != len(self.features):
            raise ValueError(
                f"{spec.name} has {len(self.features)} features but "
                f"{len(self.sampling_periods)} sampling-period factors."
            )
        self.max_sampling_period = int(self.sampling_periods.max())

        generated = self.data_dir / "generated"
        self.sig_files = sorted(generated.glob("sig_*.npy"))
        self.att_files = sorted(generated.glob("att_*.npy"))
        if not self.sig_files or len(self.sig_files) != len(self.att_files):
            raise FileNotFoundError(
                f"Expected paired sig_*.npy and att_*.npy files in {generated}."
            )

        self.min_vals, self.max_vals = self._load_scaler()
        self.file_names = self._load_file_names()
        self.index_map = self._index_windows()
        if verbose:
            print(f"{spec.name}: {len(self.index_map)} windows from {self.data_dir}")

    def _load_file_names(self) -> dict[int, str]:
        with open(self.data_dir / "file_index_dict.json", encoding="utf-8") as handle:
            mapping = json.load(handle)
        return {int(value): key for key, value in mapping.items()}

    def _load_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        table = pd.read_csv(self.scaler_path, index_col=0)
        mins = table["Min"].loc[self.features].to_numpy(dtype=np.float32)
        maxs = table["Max"].loc[self.features].to_numpy(dtype=np.float32)
        span = np.where(maxs - mins == 0, 1.0, maxs - mins)
        return mins, mins + span

    def _index_windows(self) -> list[tuple[int, int]]:
        index_map: list[tuple[int, int]] = []
        span = self.spec.window_size * self.max_sampling_period
        for file_idx, (sig_file, att_file) in enumerate(zip(self.sig_files, self.att_files)):
            sig = np.load(sig_file, mmap_mode="r")
            att = np.load(att_file, mmap_mode="r")
            if sig.shape[1] != len(self.features) or att.shape[1] != len(self.spec.attributes):
                raise ValueError(
                    f"Unexpected array shape in {sig_file.name}: signals {sig.shape}, attributes {att.shape}."
                )
            if sig.shape[0] != att.shape[0]:
                raise ValueError(f"Signal/attribute length mismatch in {sig_file.name}.")
            n_windows = (sig.shape[0] - span) // self.spec.step_size + 1
            for offset in range(max(n_windows, 0)):
                index_map.append((file_idx, offset * self.spec.step_size))
        return index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        file_idx, start = self.index_map[idx]
        span = self.spec.window_size * self.max_sampling_period
        sig = np.load(self.sig_files[file_idx], mmap_mode="r")
        att = np.load(self.att_files[file_idx], mmap_mode="r")
        stop = start + span
        if stop > sig.shape[0]:
            raise IndexError(f"Window {idx} exceeds {self.sig_files[file_idx].name}.")

        rows = np.arange(self.spec.window_size)[:, None] * self.sampling_periods
        window = sig[start:stop][rows, np.arange(len(self.features))]
        window = (window - self.min_vals) / (self.max_vals - self.min_vals)
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        att_window = att[start:stop]
        label = int(np.sum(att_window[:, self.spec.label_index]) > 0.0)
        if not self.return_meta:
            return tensor, label

        file_id = int(att_window[0, 3])
        meta = {"idx": int(start), "file": self.file_names.get(file_id, str(file_id))}
        return tensor, label, meta
