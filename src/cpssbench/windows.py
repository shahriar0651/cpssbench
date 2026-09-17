"""Windowed PyTorch datasets over preprocessed signal arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import windows_dir
from .specs import DatasetSpec, normalize_filling


class WindowDataset(Dataset):
    """Sliding-window CAN dataset.

    Default ``__getitem__`` returns ``(x, y)`` like MNIST, where ``x`` has shape
    ``(C, T, F)`` (channel-first; ``C=1``).

    With ``return_mask=True`` (CAN MAE convention):

    * ``x`` — min-max scaled to roughly ``[0, 1]``; **originally missing cells are ``0``**
    * ``y`` — attack label (``0`` / ``1``)
    * ``mask`` — same shape as ``x``; **``1`` = real bus observation**, **``0`` = intermittent gap**

    The observation mask is recorded *before* gap filling, so ``filling="forward"``
    still yields a correct mask of where values were originally absent.
    A ``DataLoader`` then stacks to ``x, mask: (N, C, T, F)`` and ``y: (N,)``.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        data_dir: str | Path,
        scaler_path: str | Path,
        return_meta: bool = False,
        return_mask: bool = False,
        verbose: bool = False,
        windows_path: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self.data_dir = Path(data_dir)
        self.scaler_path = Path(scaler_path)
        self.return_meta = return_meta
        self.return_mask = return_mask
        self.features = list(spec.features)
        self.filling = normalize_filling(spec.filling)
        self.sampling_periods = np.asarray(spec.sampling_period_factors, dtype=int) * int(spec.sampling_period)
        if len(self.sampling_periods) != len(self.features):
            raise ValueError(
                f"{spec.name} has {len(self.features)} features but "
                f"{len(self.sampling_periods)} sampling-period factors."
            )
        self.max_sampling_period = int(self.sampling_periods.max())

        self.windows_path = Path(windows_path) if windows_path is not None else self._resolve_windows_path()
        self.sig_files = sorted(self.windows_path.glob("sig_*.npy"))
        self.att_files = sorted(self.windows_path.glob("att_*.npy"))
        self.msk_files = sorted(self.windows_path.glob("msk_*.npy"))
        if not self.sig_files or len(self.sig_files) != len(self.att_files):
            raise FileNotFoundError(
                f"Expected paired sig_*.npy and att_*.npy files in {self.windows_path}."
            )
        if self.msk_files and len(self.msk_files) != len(self.sig_files):
            raise FileNotFoundError(
                f"Mismatched msk_*.npy count in {self.windows_path} "
                f"({len(self.msk_files)} masks vs {len(self.sig_files)} signals)."
            )
        if self.return_mask and not self.msk_files:
            raise FileNotFoundError(
                f"return_mask=True requires msk_*.npy observation masks in {self.windows_path}. "
                "Re-run preprocessing so masks are written before gap filling."
            )

        self.min_vals, self.max_vals = self._load_scaler()
        self.file_names = self._load_file_names()
        self.index_map = self._index_windows()
        if verbose:
            print(
                f"{spec.name}: {len(self.index_map)} windows from {self.windows_path} "
                f"(filling={self.filling}, masks={'yes' if self.msk_files else 'no'})"
            )

    def _resolve_windows_path(self) -> Path:
        preferred = windows_dir(self.data_dir, self.filling)
        if list(preferred.glob("sig_*.npy")):
            return preferred
        # Older layouts wrote forward-filled arrays directly under generated/ (no masks).
        legacy = self.data_dir / "generated"
        if self.filling == "forward" and list(legacy.glob("sig_*.npy")):
            return legacy
        return preferred

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
            if self.msk_files:
                msk = np.load(self.msk_files[file_idx], mmap_mode="r")
                if msk.shape != sig.shape:
                    raise ValueError(
                        f"Mask shape {msk.shape} != signal shape {sig.shape} in {self.msk_files[file_idx].name}."
                    )
            n_windows = (sig.shape[0] - span) // self.spec.step_size + 1
            for offset in range(max(n_windows, 0)):
                index_map.append((file_idx, offset * self.spec.step_size))
        return index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def _gather_window(self, array: np.ndarray, start: int, stop: int) -> np.ndarray:
        rows = np.arange(self.spec.window_size)[:, None] * self.sampling_periods
        return np.asarray(array[start:stop][rows, np.arange(len(self.features))], dtype=np.float32)

    def __getitem__(self, idx: int):
        file_idx, start = self.index_map[idx]
        span = self.spec.window_size * self.max_sampling_period
        sig = np.load(self.sig_files[file_idx], mmap_mode="r")
        att = np.load(self.att_files[file_idx], mmap_mode="r")
        stop = start + span
        if stop > sig.shape[0]:
            raise IndexError(f"Window {idx} exceeds {self.sig_files[file_idx].name}.")

        window = self._gather_window(sig, start, stop)
        if self.msk_files:
            msk = np.load(self.msk_files[file_idx], mmap_mode="r")
            obs = self._gather_window(msk, start, stop)
            # Stored masks are already 0/1; tolerate bool caches.
            obs = (obs > 0).astype(np.float32)
        else:
            obs = (~np.isnan(window)).astype(np.float32)

        window = (window - self.min_vals) / (self.max_vals - self.min_vals)
        # Dense IDS path keeps filled values. MAE path zeros original gaps.
        if self.return_mask:
            window = np.nan_to_num(window, nan=0.0) * obs
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (C, T, F)
        mask_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (C, T, F)

        att_window = att[start:stop]
        label = int(np.sum(att_window[:, self.spec.label_index]) > 0.0)

        extras: list = []
        if self.return_mask:
            extras.append(mask_tensor)
        if self.return_meta:
            file_id = int(att_window[0, 3])
            meta = {
                "idx": int(start),
                "file": self.file_names.get(file_id, str(file_id)),
                "obs_mask": mask_tensor,
                "filling": self.filling,
            }
            extras.append(meta)
        if extras:
            return (tensor, label, *extras)
        return tensor, label
