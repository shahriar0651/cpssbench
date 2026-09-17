# cpssbench

Cyber-Physical Systems Security Bench. It downloads (or loads) raw traces, builds windowed tensors, and returns a PyTorch dataset with the same contract as MNIST: `(window, label)`.

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

## Dataset families

Different modalities have different semantics and failure modes. Use the family guide for parameters and pitfalls:

| Family | Guide | What it covers |
| --- | --- | --- |
| **CAN** | [docs/CAN.md](docs/CAN.md) | SynCAN, ROAD — intermittent bus gaps, `filling`, observation masks for Transformer-style training |
| **V2X** | [docs/V2X.md](docs/V2X.md) | MisbehaviorX — cooperative misbehavior traces (manual files; no CAN mask API) |

Index: [docs/README.md](docs/README.md).

| Name | Family | Status | What you get |
| --- | --- | --- | --- |
| `syncan` | CAN | ready, auto-download | Synthetic CAN intrusion traces |
| `road` | CAN | ready, auto-download | ROAD dynamometer CAN traces |
| `misbehaviorx` | V2X | loader ready, manual files | V2X misbehavior (also `vasp` / `veremi`) |
| `x-canids` | CAN | registered, not implemented | Raises until a loader is added |

```bash
python -m cpssbench list
python -m cpssbench info syncan
python -m cpssbench download syncan --root ./data --split train
python -m cpssbench download road --root ./data --split train --filling forward
```

Downloaded files land in `<root>/<name>/{ambient,attacks}` plus a fitted min/max scaler under `<root>/<name>/scaler`. Later calls reuse those files.

## Shared contract

Each sample is a min-max scaled window with a channel axis so the same convolutional IDS can run across families. Shape is always `(channels, window_size, num_signals)` with `channels=1`. Label `0` is benign and `1` is attack (any attack flag inside the window).

Common overrides:

```python
from cpssbench import ROAD

dataset = ROAD(root="./data", split="train", download=True, window_size=50, step_size=5)
```

Pass `return_meta=True` for source file and row index: `(window, label, {"file", "idx", ...})`.

### CAN only: gaps and masks

CAN frames are intermittent. You can still load dense windows, or also request an observation mask for Transformer-style (and similar) training that needs masking information (full detail in [docs/CAN.md](docs/CAN.md)):

```python
from cpssbench import ROAD
from torch.utils.data import DataLoader

ds = ROAD(root="./data", split="train", filling="forward", return_mask=True)
x, y, mask = ds[0]
# x, mask: (1, T, F) — gaps in x are 0; mask is 1=observed / 0=gap

loader = DataLoader(ds, batch_size=64, shuffle=True)
x, y, mask = next(iter(loader))  # (N, 1, T, F)
```

| `filling` | Behavior |
| --- | --- |
| `"forward"` (default) | Dense forward/back-filled cache |
| `"none"` / `"nan"` | Keep NaNs in the signal cache |
| `"zero"` | Zero-fill the signal cache |

V2X does **not** accept `filling` or `return_mask`.

## Adding a dataset

1. Add a `DatasetSpec` in `src/cpssbench/specs.py` (set `family` to `can` or `v2x`).
2. Add a downloader in `src/cpssbench/download.py` if the files can be fetched automatically.
3. Register the class in `src/cpssbench/datasets.py` and `_CLASSES` in `__init__.py`.
4. Document family-specific semantics in `docs/CAN.md` or `docs/V2X.md`.

The IDS experiments that consume this package live in the sibling [RobIDS](https://github.com/shahriar0651/robids) repo.
