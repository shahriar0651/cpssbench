# TODO

## V2X / MisbehaviorX (blocked for later)

Auto-download and end-to-end setup are **not** finished. Users must manually place curated files today. Details: [docs/V2X.md](docs/V2X.md).

Upstream release: [MisbehaviorX on IEEE DataPort](https://ieee-dataport.org/documents/misbehaviorx-comprehensive-v2x-misbehavior-detection-dataset-enabled-v2x-application) (DOI [10.21227/s44z-8616](https://dx.doi.org/10.21227/s44z-8616)).

- [ ] **DataPort / auth download** — IEEE DataPort requires login + subscription; cannot use anonymous URL fetch like ROAD/Zenodo. Decide: redistributable mirror (if allowed) **or** opt-in credentials / cookie / DataPort AWS keys, then fetch `MisbehaviorX.zip` (~992 MB).
- [ ] **Raw → curated transform** — DataPort CSVs are raw BSM logs (`rv_id`, `hv_id`, `rv_pos_*`, `attack_type`, …). cpssbench expects engineered features (`speed_x`, `del_pos_x`, …) and attributes (`id`, `time_chunk`, `attack_gt`, `attack_name`). Implement (or import from RobIDS/VehiGAN prep) and write curated tables.
- [ ] **Layout normalize** — Map release `ambients/` → `ambient/`; nest curated outputs under `ambient/generated/` and `attacks/generated/`; keep raw under `_raw/` for audit.
- [ ] **Wire package download** — Add `download_misbehaviorx()` in `download.py`, register in `ensure_downloaded`, set `DatasetSpec.downloadable=True`, expose `python -m cpssbench download misbehaviorx`, add tiny curated fixtures + tests (no DataPort in CI).

Until the above is done, `MisbehaviorX(..., download=True)` will still fail unless curated CSVs/npy already exist locally.
