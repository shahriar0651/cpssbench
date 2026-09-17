"""cpssbench: Cyber-Physical Systems Security Bench, loaded like MNIST.

Example::

    from cpssbench import SynCAN
    from torch.utils.data import DataLoader

    train = SynCAN(root="./data", split="train", download=True)
    window, label = train[0]
    loader = DataLoader(train, batch_size=64, shuffle=True)
"""

from __future__ import annotations

from typing import Optional

from .datasets import MisbehaviorX, ROAD, SynCAN, VehicularDataset
from .specs import FILLING_MODES, DatasetSpec, get_spec, list_specs, normalize_filling

__all__ = [
    "SynCAN",
    "ROAD",
    "MisbehaviorX",
    "VehicularDataset",
    "load",
    "list_datasets",
    "describe",
    "get_spec",
    "FILLING_MODES",
    "normalize_filling",
]

__version__ = "0.2.0"

_CLASSES = {
    "syncan": SynCAN,
    "road": ROAD,
    "misbehaviorx": MisbehaviorX,
    "vasp": MisbehaviorX,
    "veremi": MisbehaviorX,
}


def list_datasets(status: Optional[str] = None) -> list[dict]:
    """Return name, status, family, and shape for every registered dataset."""
    rows = []
    for spec in list_specs(status):
        rows.append(
            {
                "name": spec.name,
                "status": spec.status,
                "family": spec.family,
                "input_shape": spec.input_shape if spec.features else None,
                "downloadable": spec.downloadable,
                "description": spec.description,
            }
        )
    return rows


def describe(name: str) -> DatasetSpec:
    return get_spec(name)


def load(name: str, **kwargs) -> VehicularDataset:
    """Load a dataset by name.

    ``kwargs`` are forwarded to the dataset class (``root``, ``split``,
    ``download``, ``window_size``, ...).
    """
    key = name.strip().lower()
    if key == "x-canids":
        spec = get_spec(key)
        raise NotImplementedError(f"{spec.name} is not implemented yet. {spec.notes}")
    if key not in _CLASSES:
        known = ", ".join(sorted(set(_CLASSES) | {"x-canids"}))
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}")
    return _CLASSES[key](**kwargs)
