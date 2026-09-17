# CAN datasets

In-vehicle **Controller Area Network (CAN)** intrusion traces. Each sample is a sliding window over signal columns from multiple CAN IDs, returned as a channel-first tensor like MNIST: `(C, T, F)`.

Supported loaders:

| Name | Class | Status | Source |
| --- | --- | --- | --- |
| `syncan` | `SynCAN` | ready, auto-download | [SynCAN](https://github.com/etas/SynCAN) |
| `road` | `ROAD` | ready, auto-download | [ROAD (Zenodo)](https://zenodo.org/records/10462796) |
| `x-canids` | — | planned | registered; loader not implemented yet |

```python
from cpssbench import SynCAN, ROAD

train = SynCAN(root="./data", split="train", download=True)
road = ROAD(root="./data", split="train", download=True, window_size=50, step_size=5)
```

CLI:

```bash
python -m cpssbench download syncan --root ./data --split train
python -m cpssbench download road --root ./data --split train --filling forward
```

---

## Semantic meaning

CAN is a shared serial bus. ECUs broadcast frames identified by an **ID**; each frame carries a small payload that we decode into named **signals**.

In cpssbench, a window is a 2D grid:

- **Time axis `T`** — successive bus events in the sliding window (after optional period subsampling per signal)
- **Feature axis `F`** — fixed set of signal columns (e.g. `ID_2_Sig_1`, …)
- **Channel `C=1`** — MNIST-style channel for CNN IDS models

Label `y`:

- `0` = benign (no attack flag in the window)
- `1` = attack (any attack flag inside the window)

Attributes kept alongside signals include time, ID, label, and source file (see `return_meta=True`).

---

## Unique challenge: intermittent / missing data

Unlike a regularly sampled sensor matrix, **most cells in a CAN window were never transmitted**.

At any bus timestamp, only the ECU that sent the current frame updates its signals. All other ID/signal columns are absent for that row. After pivoting raw frames into a dense signal table, those absences appear as **NaNs**.

Historically many IDS pipelines **forward-fill** those gaps so every `(t, f)` has a number. That is convenient for dense CNNs, but it:

- invents values the bus never sent
- hides the natural sparsity / schedule of the network
- discards information that **Transformer-style models** often need: which tokens or cells were actually observed

cpssbench therefore treats intermittency as a first-class signal for CAN research. You can still load dense, forward-filled windows for classic IDS, **or** load the same data with an **observation mask** so downstream models know what was real vs. filled.

---

## Parameters that matter for CAN

| Parameter | Default | Role |
| --- | --- | --- |
| `filling` | `"forward"` | How to write cached signal arrays from sparse pivots |
| `return_mask` | `False` | If `True`, return `(x, y, mask)` so callers get masking information with the window |
| `window_size` | dataset-specific | Length of the time axis `T` |
| `step_size` | dataset-specific | Hop between windows |
| `sampling_period` / factors | dataset-specific | Per-signal temporal stride (SynCAN especially) |
| `return_meta` | `False` | Also return `{"file", "idx", "obs_mask", "filling"}` |
| `split` | `"train"` / `"test"` | Ambient vs attack splits (mapped to on-disk folders) |

`filling` and `return_mask` are **CAN-only**. Passing them to V2X raises `ValueError`.

### `filling` modes

Computed when writing `generated/<filling>/sig_*.npy`. Observation masks are always written **before** this step as `msk_*.npy`.

| `filling` | Aliases | Behavior |
| --- | --- | --- |
| `"forward"` | `"ffill"` | Forward-fill then back-fill → dense cache (classic IDS) |
| `"none"` | `"nan"`, `"null"` | Leave gaps as NaN in the signal cache |
| `"zero"` | `"0"` | Replace gaps with `0.0` in the signal cache |

Each mode is cached separately under:

```text
<root>/<name>/{ambient,attacks}/generated/<filling>/
  sig_*.npy   # signals after chosen filling
  msk_*.npy   # 1 = real observation, 0 = gap (pre-fill)
  att_*.npy   # attributes / labels
```

Switching `filling` does not wipe the download; it builds another cache folder.

---

## Loading with an observation mask

Besides the default dense `(window, label)` path, CAN loaders can return **masked inputs**: the window plus a binary map of which cells were real bus observations. That is useful for **Transformer-based** (and other attention) training pipelines that need masking information—for attention masks, loss weighting, or token validity—without requiring a separate preprocessing stage.

### Motivation

With `return_mask=True` the caller can:

1. Feed values that mark **unobserved** cells clearly (zeros in `x` at gaps)
2. Keep a parallel `mask` tensor (`1` = observed, `0` = gap)
3. Use that mask in the model or objective (ignore gaps, restrict attention, etc.)

You may still use `filling="forward"` for a dense on-disk cache; the **mask is recorded before filling**, so it reflects true intermittency.

### How to load

```python
from cpssbench import ROAD
from torch.utils.data import DataLoader

ds = ROAD(
    root="./data",
    split="train",
    download=True,
    filling="forward",   # dense cache is fine
    return_mask=True,    # also return observation mask
)

x, y, mask = ds[0]
```

### Tensor convention

| Tensor | Per sample | `DataLoader` batch | Meaning |
| --- | --- | --- | --- |
| `x` | `(C, T, F)` | `(N, C, T, F)` | Min–max scaled ≈ `[0, 1]`; **originally missing cells forced to `0`** |
| `mask` | `(C, T, F)` | `(N, C, T, F)` | **`1` = real bus observation**, **`0` = intermittent gap** (`float32`) |
| `y` | scalar | `(N,)` | Attack label (`0` / `1`) |

- `C = 1` matches the existing MNIST-style layout. Squeeze if you prefer `(N, T, F)`: `x.squeeze(1)`, `mask.squeeze(1)`.
- Mask values are recorded **before** filling, so `filling="forward"` still yields a correct gap map.
- When `return_mask=True`, returned `x` zeros gap cells even if the on-disk `sig_*.npy` was forward-filled. Observed cells keep their scaled values.
- When `return_mask=False`, `__getitem__` is `(x, y)` and keeps dense filled values (classic IDS path).

### Using the mask in training

Typical pattern: only score positions that were actually observed (no ground truth exists at intermittent gaps):

```python
# pred, x, mask: (N, 1, T, F)
loss = (((pred - x) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
```

The same `mask` can drive attention / key-padding style logic in Transformer encoders. `x` is already `0` at gaps when `return_mask=True`; `x * mask` is equivalent for those cells.

### Without a mask (classic IDS)

```python
ds = ROAD(root="./data", split="train", filling="forward")  # return_mask=False
x, y = ds[0]  # dense window, no mask tensor
```

### `return_meta`

```python
x, y, meta = ROAD(..., return_meta=True)[0]
# meta["file"], meta["idx"], meta["obs_mask"], meta["filling"]

x, y, mask, meta = ROAD(..., return_mask=True, return_meta=True)[0]
```

---

## Windowing defaults

| Dataset | `window_size` | `step_size` | Notes |
| --- | --- | --- | --- |
| SynCAN | 50 | 10 | Per-signal `sampling_period_factors` (IDs tick at different rates) |
| ROAD | 50 | 5 | Uniform period factors of `1` |

Override per call: `window_size=...`, `step_size=...`, `sampling_period=...`.

---

## On-disk layout

```text
<root>/syncan/   or   <root>/road/
  ambient/                 # train / benign
  attacks/                 # test / attack traces
  scaler/min_max_values_*.csv
```

Scaler min/max are fit on the train/ambient split and reused for test.

---

## Related family docs

- [V2X](V2X.md) — cooperative vehicle messages (no CAN-style intermittency API)
- Main [README](../README.md)
