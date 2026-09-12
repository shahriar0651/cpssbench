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

That works after the package is published on PyPI. Until then, install this repository directly:

```bash
pip install "git+https://github.com/shahriar0651/cpssbench.git"
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

## Layout

Each sample is a min-max scaled window with a channel axis, so the same convolutional IDS can run on every dataset. Shape is always `(channels, window_size, num_signals)`. Label `0` is benign and `1` is attack (any attack flag inside the window).

## Adding a dataset

1. Add a `DatasetSpec` in `src/cpssbench/specs.py`.
2. Add a downloader in `src/cpssbench/download.py` if the files can be fetched automatically.
3. Register the class in `src/cpssbench/datasets.py` and `_CLASSES` in `__init__.py`.

The IDS experiments that consume this package live in the sibling [RobIDS](https://github.com/shahriar0651/robids) repo.
