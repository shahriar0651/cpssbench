# cpssbench

Cyber-Physical Systems Security Bench. It downloads raw traces, builds windowed tensors, and returns a PyTorch dataset with the same contract as MNIST: `(window, label)`.

```python
from cpssbench import SynCAN
from torch.utils.data import DataLoader

train = SynCAN(root="./data", split="train", download=True)
test = SynCAN(root="./data", split="test", download=True)

window, label = train[0]          # window: (1, time, signals), label: 0 benign / 1 attack
loader = DataLoader(train, batch_size=64, shuffle=True)
```

Or load by name:

```python
import cpssbench

dataset = cpssbench.load("road", root="./data", split="test", download=True)
print(dataset.input_shape)   # (channels, window, signals)
```

## Install

```bash
pip install cpssbench
```

From a local clone, for development:

```bash
pip install -e .
```

Then open `examples/train_like_mnist.ipynb` to plot a sample grid and train a small network.

Python 3.10+. SynCAN also needs `git` on `PATH`. ROAD is fetched from Zenodo with the standard library, so `wget` is not required.

## Datasets

| Name | Status | What you get |
| --- | --- | --- |
| `syncan` | ready, auto-download | Synthetic CAN intrusion traces |
| `road` | ready, auto-download | ROAD dynamometer CAN traces |
| `misbehaviorx` | loader ready, manual files | V2X misbehavior (also accepted as `vasp`) |
| `x-canids` | registered, not implemented | Raises a clear error until a loader is added |

```bash
python -m cpssbench list
python -m cpssbench info syncan
python -m cpssbench download syncan --root ./data --split train
```

Downloaded files land in `<root>/<name>/{ambient,attacks}` plus a fitted min/max scaler under `<root>/<name>/scaler`. Later calls reuse those files.

## Overrides

Windowing defaults live in the library so a new project does not need the old Hydra YAML. Override them per call:

```python
from cpssbench import ROAD

dataset = ROAD(root="./data", split="train", download=True, window_size=50, step_size=5)
```

Pass `return_meta=True` if you also need the source file and row index: `(window, label, {"file", "idx"})`.

### CAN gap filling (SynCAN / ROAD only)

CAN frames are intermittent, so each timestamp only observes the transmitting ID's signals. Choose how to treat those gaps when building the cached signal arrays:

| `filling` | Behavior |
| --- | --- |
| `"forward"` (default) | Forward-fill then back-fill → dense windows |
| `"none"` / `"nan"` | Leave missing samples as `NaN` in the cache |
| `"zero"` | Replace missing samples with `0.0` in the cache |

Observation masks are always saved **before** filling. With `return_mask=True` you get the MAE-style triple even while using forward fill:

| Tensor | Per-sample shape | Batched (`DataLoader`) | Meaning |
| --- | --- | --- | --- |
| `x` | `(C, T, F)` | `(N, C, T, F)` | Min–max scaled; **gaps forced to `0`** |
| `mask` | `(C, T, F)` | `(N, C, T, F)` | **`1` = real observation**, **`0` = intermittent gap** |
| `y` | scalar | `(N,)` | Attack label |

(`C=1` channel axis matches the existing MNIST-style layout; squeeze `C` if you prefer `(N, T, F)`.)

```python
from cpssbench import ROAD
from torch.utils.data import DataLoader

ds = ROAD(root="./data", split="train", filling="forward", return_mask=True)
x, y, mask = ds[0]
# Supervise only real bus samples, e.g.:
# loss = (((recon - x) ** 2) * mask).sum() / mask.sum().clamp_min(1)

loader = DataLoader(ds, batch_size=64, shuffle=True)
x, y, mask = next(iter(loader))  # x, mask: (N, 1, T, F)
```

Without `return_mask`, `__getitem__` stays `(x, y)` and keeps the filled dense values (classic IDS path). Each filling mode is cached under `generated/<filling>/` with matching `msk_*.npy` files. V2X / MisbehaviorX do not accept `filling` or `return_mask`.

## Layout

Each sample is a min-max scaled window with a channel axis, so the same convolutional IDS can run on every dataset. Shape is always `(channels, window_size, num_signals)`. Label `0` is benign and `1` is attack (any attack flag inside the window).

## Adding a dataset

1. Add a `DatasetSpec` in `src/cpssbench/specs.py`.
2. Add a downloader in `src/cpssbench/download.py` if the files can be fetched automatically.
3. Register the class in `src/cpssbench/datasets.py` and `_CLASSES` in `__init__.py`.

The IDS experiments that consume this package live in the sibling [RobIDS](https://github.com/shahriar0651/robids) repo.
