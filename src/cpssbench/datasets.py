"""Public dataset classes.

Usage matches torchvision:

    from cpssbench import SynCAN

    train = SynCAN(root="./data", split="train", download=True)
    window, label = train[0]

CAN-only: observation mask of intermittent bus gaps (works with forward fill):

    train = SynCAN(root="./data", split="train", filling="forward", return_mask=True)
    x, y, mask = train[0]  # x/mask: (C, T, F); gaps in x are 0; mask is 1=observed / 0=gap
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
        filling: Optional[str] = None,
        return_meta: bool = False,
        return_mask: bool = False,
        data_dir: Optional[str | Path] = None,
        scaler_dir: Optional[str | Path] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        base = get_spec(self.spec_name)
        if filling is not None and base.family != "can":
            raise ValueError(
                f"filling=... is only supported for CAN datasets (got family={base.family!r}). "
                "Intermittent missing samples are a CAN-bus property."
            )
        if return_mask and base.family != "can":
            raise ValueError(
                f"return_mask=True is only supported for CAN datasets (got family={base.family!r})."
            )

        spec = base.with_overrides(window_size, step_size, sampling_period, filling=filling)
        if spec.status == "planned":
            raise NotImplementedError(f"{spec.name} is not implemented yet. {spec.notes}")

        self.spec = spec
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta
        self.return_mask = return_mask

        if data_dir is None and download:
            ensure_downloaded(spec, self.root)

        self.data_dir = Path(data_dir) if data_dir is not None else split_dir(self.root, spec.name, split)
        if scaler_dir is not None:
            self.scaler_path = Path(scaler_dir) / f"min_max_values_{spec.name}.csv"
        else:
            self.scaler_path = dataset_root(self.root, spec.name) / "scaler" / f"min_max_values_{spec.name}.csv"

        fit_scaler = split.strip().lower() in {"train", "training", "ambient"}
        if spec.family == "can":
            windows_path = prepare_can_split(spec, self.data_dir, self.scaler_path, fit_scaler, n_jobs=n_jobs)
            self._base = WindowDataset(
                spec,
                self.data_dir,
                self.scaler_path,
                return_meta=return_meta,
                return_mask=return_mask,
                verbose=verbose,
                windows_path=windows_path,
            )
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
        if len(item) > 2:
            return (window, label, *item[2:])
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

    @property
    def filling(self) -> str:
        """CAN gap-filling mode. Meaningful for ``family='can'`` only."""
        return self.spec.filling


class SynCAN(VehicularDataset):
    """SynCAN intrusion dataset. ``split`` is ``'train'`` or ``'test'``."""

    spec_name = "syncan"


class ROAD(VehicularDataset):
    """ROAD CAN intrusion dataset. ``split`` is ``'train'`` or ``'test'``."""

    spec_name = "road"


class MisbehaviorX(VehicularDataset):
    """V2X misbehavior dataset. Requires a local copy; download is manual."""

    spec_name = "misbehaviorx"
