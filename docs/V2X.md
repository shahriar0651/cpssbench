# V2X datasets

**Vehicle-to-everything (V2X)** misbehavior / intrusion traces. Samples are sliding windows over kinematic and relative-motion features from cooperative awareness messages, returned in the same MNIST-style layout as CAN: `(C, T, F)`.

Supported loaders:

| Name | Class | Aliases | Status |
| --- | --- | --- | --- |
| `misbehaviorx` | `MisbehaviorX` | `vasp`, `veremi` | loader ready; **manual** data placement |

```python
from cpssbench import MisbehaviorX

# Automatic download is not wired yet — place curated files first (see below).
ds = MisbehaviorX(root="./data", split="train", download=False)
window, label = ds[0]
```

### Getting MisbehaviorX today (manual)

Official release:

- **[MisbehaviorX on IEEE DataPort](https://ieee-dataport.org/documents/misbehaviorx-comprehensive-v2x-misbehavior-detection-dataset-enabled-v2x-application)**  
  DOI: [10.21227/s44z-8616](https://dx.doi.org/10.21227/s44z-8616)  
  Archive: `MisbehaviorX.zip` (~992 MB), format `*.csv`

The DataPort page documents raw BSM-style columns (`rv_id`, `hv_id`, `rv_pos_x`, `rv_speed`, `attack_type`, …) under folders **`ambients`** and **`attacks`**. cpssbench does **not** read that raw schema directly. It expects **curated** window-ready CSVs (relative-motion features such as `speed_x`, `del_pos_x`, …) under a slightly different tree (see [On-disk layout](#on-disk-layout-manual)).

**Steps for a user right now**

1. Open the [IEEE DataPort MisbehaviorX page](https://ieee-dataport.org/documents/misbehaviorx-comprehensive-v2x-misbehavior-detection-dataset-enabled-v2x-application).
2. Sign in (or create an account). Dataset files require an **IEEE DataPort subscription** (or institutional access)—anonymous download is not available.
3. Download `MisbehaviorX.zip` and unpack it. You should see something like `ambients/` and `attacks/` with per-scenario CSV traces.
4. Produce or obtain the **curated** feature tables that match cpssbench’s `MISBEHAVIORX_FEATURES` / `MISBEHAVIORX_ATTRIBUTES` (relative kinematics + `id`, `time_chunk`, `attack_gt`, `attack_name`). Raw DataPort CSVs must be transformed; dropping them unchanged into `generated/` will fail column checks.
5. Place curated files here:

   ```text
   <root>/misbehaviorx/ambient/generated/*.csv
   <root>/misbehaviorx/attacks/generated/*.csv
   ```

6. Load ambient/train first so the min–max scaler is fit, then attacks/test:

   ```python
   train = MisbehaviorX(root="./data", split="train", download=False)
   test = MisbehaviorX(root="./data", split="test", download=False)
   ```

### Why automatic download is not wired yet

| Blocker | Detail |
| --- | --- |
| Auth wall | IEEE DataPort file links need a logged-in subscriber session; a plain `urllib` fetch (as used for ROAD/Zenodo) cannot pull `MisbehaviorX.zip` anonymously. |
| Schema gap | Published CSVs are **raw BSM logs**. cpssbench windows use **engineered** features (`del_pos_*`, `del_speed_*`, …) plus `time_chunk` / `attack_name` attributes. |
| Layout naming | Release uses `ambients/`; cpssbench uses `ambient/` and nests curated tables under `generated/`. |
| Size / policy | ~1 GB zip and DataPort terms make “silent auto-download” a poor default without explicit user credentials. |

### Steps to accomplish auto-setup in cpssbench

Work items to make `MisbehaviorX(..., download=True)` (or a documented `python -m cpssbench download misbehaviorx`) real:

1. **Access path**
   - Prefer a redistributable mirror (Zenodo / institutional open URL) if licensing allows; **or**
   - Document/env-based DataPort credentials / browser cookie / AWS keys from the DataPort profile, and fetch `MisbehaviorX.zip` only when the user opts in.
2. **`download_misbehaviorx()` in `download.py`**
   - Download + unzip.
   - Normalize `ambients` → `ambient`, keep `attacks`.
   - Leave raw traces under e.g. `_raw/` for auditability.
3. **Curate raw → loader CSVs**
   - Implement (or import from RobIDS / VehiGAN prep) a transform that builds:
     - features: `speed_x`, `del_pos_x`, `speed_y`, `del_pos_y`, `accel_x`, `del_speed_x`, `accel_y`, `del_speed_y`, `del_heading_x`, `yaw_rate_x`, `del_heading_y`, `yaw_rate_y`
     - attributes: `id`, `time_chunk`, `attack_gt`, `attack_name`
   - Map `attack_type` / Genuine vs attack labels into `attack_gt` / `attack_name`.
   - Write results to `ambient/generated/` and `attacks/generated/`.
4. **Wire `ensure_downloaded`**
   - Set `downloadable=True` on the MisbehaviorX `DatasetSpec` once steps 1–3 work.
   - Register `download_misbehaviorx` next to SynCAN/ROAD.
5. **Tests**
   - Tiny fixture CSV (curated schema) in `tests/` so CI does not need DataPort.
   - Optional smoke test behind an env flag when credentials exist.
6. **Docs / CLI**
   - `python -m cpssbench download misbehaviorx --root ./data`
   - Keep this page’s manual path as fallback when auth fails.

Until that lands, use the manual path above. Related paper context: VehiGAN / VASP-enabled MisbehaviorX (ICDCS 2024); dataset authors include Shahriar et al. on IEEE DataPort.

---

## Semantic meaning

V2X security datasets target **on-board / roadside message integrity**, not in-vehicle CAN frames. Typical features describe how a remote vehicle moves and how that motion relates to the ego vehicle, for example:

- speeds and accelerations (`speed_x`, `accel_y`, …)
- relative position / speed / heading deltas (`del_pos_x`, `del_speed_y`, `del_heading_x`, …)
- yaw rates

A window is a short trajectory snippet for one entity (and time chunk). Label `y`:

- `0` = benign
- `1` = attack / misbehavior (any attack flag inside the window)

This family is about **behavioral misbehavior** in cooperative messages (Sybil-like, position forging, etc.), not ECU bus scheduling.

Upstream simulation stack (from the DataPort abstract): **VASP** (V2X Application Spoofing Platform) on **Veins** / OMNeT++ / SUMO, Boston traffic network; ambient BSMs plus multi-attack malicious runs.

---

## Unique challenges (and what is *not* a challenge here)

| Topic | V2X in cpssbench |
| --- | --- |
| Missing / intermittent cells | **Not** the CAN-bus gap problem. Curated traces are dense feature matrices. |
| `filling=` / `return_mask=` | **Not supported.** Passing them raises `ValueError`. |
| Download | **Manual** via IEEE DataPort + curation (see above). Auto-download blocked by auth + schema gap. |
| Window grouping | Windows are indexed per `(id, time_chunk, attack_name)` contiguous segments, not a simple global CAN stride alone. |

Residual NaNs in curated CSVs (if any) are forward/back-filled during preprocessing so loaders yield dense tensors for CNN-style IDS. That is an implementation convenience, not an exposed research knob like CAN masking.

If you need observation masks for Transformer-style training, use the [CAN family](CAN.md) instead.

---

## Parameters that matter for V2X

| Parameter | Default | Role |
| --- | --- | --- |
| `window_size` | `10` | Time length `T` of each window |
| `step_size` | `10` | Hop within a contiguous entity segment |
| `return_meta` | `False` | Also return `{"file", "idx"}` |
| `split` | `"train"` / `"test"` | Ambient vs attacks folders |
| `download` | `True` | Accepted, but auto-fetch raises until files exist locally |

**Do not pass** `filling` or `return_mask` — those exist only for CAN.

```python
ds = MisbehaviorX(
    root="./data",
    split="train",
    download=False,
    window_size=10,
    step_size=10,
    return_meta=True,
)
window, label, meta = ds[0]
```

---

## Tensor layout

| Tensor | Per sample | Batched | Meaning |
| --- | --- | --- | --- |
| `window` / `x` | `(C, T, F)` | `(N, C, T, F)` | Min–max scaled features; dense |
| `label` / `y` | scalar | `(N,)` | `0` benign / `1` attack |

`C = 1`. Feature count `F` is the MisbehaviorX feature set (12 signals by default in the current spec).

There is **no** observation-mask return path for V2X.

---

## On-disk layout (manual)

```text
<root>/misbehaviorx/
  ambient/generated/*.csv      # curated features — or already-built sig_*.npy / att_*.npy
  attacks/generated/*.csv
  scaler/min_max_values_misbehaviorx.csv   # fit from ambient/train
```

If `sig_*.npy` / `att_*.npy` are already present under `generated/`, they are reused. Otherwise curated CSVs are converted on first load. Fit the scaler on the ambient/train split first.

**Not sufficient alone:** unpacking DataPort `ambients/*.csv` / `attacks/*.csv` raw BSM logs into those folders without the feature transform.

---

## Citation / links

- Dataset: [MisbehaviorX (IEEE DataPort)](https://ieee-dataport.org/documents/misbehaviorx-comprehensive-v2x-misbehavior-detection-dataset-enabled-v2x-application), DOI [10.21227/s44z-8616](https://dx.doi.org/10.21227/s44z-8616)
- Simulation tooling context: VASP / Veins-based V2X spoofing (see DataPort abstract and linked VehiGAN ICDCS work)

---

## Related family docs

- [CAN](CAN.md) — in-vehicle bus traces, intermittent gaps, `filling` + `return_mask` for masked / Transformer-style training
- Main [README](../README.md)

Open engineering items for auto-download / curation: [TODO.md](../TODO.md).
