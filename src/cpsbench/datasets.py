"""Public dataset classes.

Usage matches torchvision:

    from cpsbench import SynCAN

    train = SynCAN(root="./data", split="train", download=True)
    window, label = train[0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from .download import dataset_root, ensure_downloaded, split_dir
from .preprocess import prepare_can_split
from .specs import DatasetSpec, get_spec
from .v2x import V2XWindowDataset, prepare_v2x_split
from .windows import WindowDataset


class VehicularDataset(Dataset):
    """Base class for datasets that download, preprocess, and yield windows."""

    spec_name: str = ""

    def __init__(
        self,
        root: str | Path = "./data",
        split: str = "train",
        download: bool = True,
        transform=None,
        target_transform=None,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        sampling_period: Optional[int] = None,
        return_meta: bool = False,
        data_dir: Optional[str | Path] = None,
        scaler_dir: Optional[str | Path] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        spec = get_spec(self.spec_name).with_overrides(window_size, step_size, sampling_period)
        if spec.status == "planned":
            raise NotImplementedError(f"{spec.name} is not implemented yet. {spec.notes}")

        self.spec = spec
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta

        if data_dir is None and download:
            ensure_downloaded(spec, self.root)

        self.data_dir = Path(data_dir) if data_dir is not None else split_dir(self.root, spec.name, split)
        if scaler_dir is not None:
            self.scaler_path = Path(scaler_dir) / f"min_max_values_{spec.name}.csv"
        else:
            self.scaler_path = dataset_root(self.root, spec.name) / "scaler" / f"min_max_values_{spec.name}.csv"

        fit_scaler = split.strip().lower() in {"train", "training", "ambient"}
        if spec.family == "can":
            prepare_can_split(spec, self.data_dir, self.scaler_path, fit_scaler, n_jobs=n_jobs)
            self._base = WindowDataset(spec, self.data_dir, self.scaler_path, return_meta, verbose)
        elif spec.family == "v2x":
            prepare_v2x_split(spec, self.data_dir, self.scaler_path, fit_scaler)
            self._base = V2XWindowDataset(spec, self.data_dir, self.scaler_path, return_meta, verbose)
        else:
            raise ValueError(f"Unsupported dataset family: {spec.family}")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        item = self._base[idx]
        window, label = item[0], item[1]
        if self.transform is not None:
            window = self.transform(window)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_meta:
            return window, label, item[2]
        return window, label

    @property
    def classes(self) -> tuple[str, ...]:
        return self.spec.classes

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.spec.input_shape

    @property
    def num_signals(self) -> int:
        return self.spec.num_signals

    @property
    def window_size(self) -> int:
        return self.spec.window_size

    @property
    def channels(self) -> int:
        return self.spec.channels

    @property
    def features(self) -> tuple[str, ...]:
        return self.spec.features


class SynCAN(VehicularDataset):
    """SynCAN intrusion dataset. ``split`` is ``'train'`` or ``'test'``."""

    spec_name = "syncan"


class ROAD(VehicularDataset):
    """ROAD CAN intrusion dataset. ``split`` is ``'train'`` or ``'test'``."""

    spec_name = "road"


class MisbehaviorX(VehicularDataset):
    """V2X misbehavior dataset. Requires a local copy; download is manual."""

    spec_name = "misbehaviorx"
