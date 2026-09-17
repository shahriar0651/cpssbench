"""Turn raw CAN CSVs into window-ready signal and attribute arrays."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .specs import DatasetSpec, normalize_filling


def _process_id(target_id, frame: pd.DataFrame) -> pd.DataFrame:
    per_id = frame[frame["ID"] == target_id].T.dropna().T
    rename = {}
    skip = {"ID", "Label", "Time"}
    for column in set(per_id.columns) - skip:
        cleaned = column.replace("Signal_", "Signal").replace("_of_ID", "")
        rename[column] = f"ID_{target_id}_" + cleaned.replace("Signal", "Sig_")
    renamed = per_id.rename(columns=rename)
    keep = [name for name in rename.values() if name in renamed.columns]
    return renamed[keep]


def _update_minmax(spec: DatasetSpec, signals: pd.DataFrame, scaler_path: Path) -> None:
    features = list(spec.features)
    present = [name for name in features if name in signals.columns]
    if not present:
        return
    try:
        stored = pd.read_csv(scaler_path, index_col=0)
        overall_min = stored["Min"].reindex(features)
        overall_max = stored["Max"].reindex(features)
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        overall_min = pd.Series(np.nan, index=features)
        overall_max = pd.Series(np.nan, index=features)

    current_min = signals[present].min(axis=0).to_numpy()
    current_max = signals[present].max(axis=0).to_numpy()
    stored_min = overall_min.loc[present].to_numpy(dtype=float)
    stored_max = overall_max.loc[present].to_numpy(dtype=float)
    overall_min.loc[present] = np.where(np.isnan(stored_min), current_min, np.minimum(stored_min, current_min))
    overall_max.loc[present] = np.where(np.isnan(stored_max), current_max, np.maximum(stored_max, current_max))
    pd.DataFrame({"Min": overall_min, "Max": overall_max}, index=features).to_csv(scaler_path)


def _sparse_one_file(
    spec: DatasetSpec,
    file_name: str,
    file_path: Path,
    scaler_path: Path,
    fit_scaler: bool,
    n_jobs: int,
) -> None:
    frame = pd.read_csv(file_path, skiprows=1, names=list(spec.org_features))
    frame["ID"] = frame["ID"].astype(str).str.replace("id", "", regex=False)

    total_elements = 75_000_000
    max_length = max(int(total_elements / max(len(spec.org_features), 1)), 1)
    n_chunks = int(np.ceil(len(frame) / max_length))
    chunk_length = int(np.ceil(len(frame) / max(n_chunks, 1)))

    for index in range(n_chunks):
        chunk = frame.iloc[index * chunk_length : min((index + 1) * chunk_length, len(frame))]
        pieces = Parallel(n_jobs=n_jobs)(
            delayed(_process_id)(target_id, chunk)
            for target_id in tqdm(chunk["ID"].unique(), desc=file_name, leave=False)
        )
        extended = pd.concat(pieces, axis=1)
        extended.insert(0, "File", file_name)
        extended[["ID", "Label", "Time"]] = chunk[["ID", "Label", "Time"]]
        extended = extended.sort_index()

        out = file_path.parent / "generated" / f"gen_{file_name}_{index + 1}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        extended.to_parquet(out, engine="pyarrow", compression="snappy")
        if fit_scaler:
            _update_minmax(spec, extended, scaler_path)


def _write_file_index(split_path: Path) -> dict[str, int]:
    csvs = sorted(glob.glob(str(split_path / "*.csv")))
    mapping = {Path(path).stem: index for index, path in enumerate(csvs)}
    with open(split_path / "file_index_dict.json", "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)
    return mapping


def apply_filling(signals: pd.DataFrame, filling: str) -> pd.DataFrame:
    """Fill intermittent CAN gaps, or leave them as NaN.

    Modes:
      - ``forward``: forward-fill then back-fill (dense windows)
      - ``none`` / ``nan``: keep missing bus samples as NaN (MAE-friendly)
      - ``zero``: replace missing values with 0.0
    """
    mode = normalize_filling(filling)
    if mode == "forward":
        return signals.ffill().bfill()
    if mode == "zero":
        return signals.fillna(0.0)
    return signals


def windows_dir(split_path: Path, filling: str) -> Path:
    """Per-filling cache directory for sig/att arrays (parquets stay under generated/)."""
    return Path(split_path) / "generated" / normalize_filling(filling)


def _npy_one_file(
    spec: DatasetSpec,
    file_name: str,
    parquet_path: Path,
    file_enum: dict[str, int],
    out_dir: Path,
) -> None:
    frame = pd.read_parquet(parquet_path, engine="pyarrow")
    frame = frame.replace(file_enum)
    signals = frame[list(spec.features)]
    attributes = frame[list(spec.attributes)].copy()
    attributes["ID"] = attributes["ID"].astype(int)

    total_elements = 100_000_000
    max_length = max(int(total_elements / max(len(spec.features), 1)), 1)
    n_chunks = int(np.ceil(len(signals) / max_length))
    out_dir.mkdir(parents=True, exist_ok=True)

    for index in range(n_chunks):
        start = int(index * max_length)
        stop = min(int((index + 1) * max_length), len(signals))
        raw_chunk = signals.iloc[start:stop]
        # Mask is computed before filling so forward-filled caches still know true bus gaps.
        obs_mask = raw_chunk.notna().to_numpy(dtype=np.float32)
        signal_chunk = apply_filling(raw_chunk, spec.filling)
        attr_chunk = attributes.iloc[start:stop]
        np.save(out_dir / f"sig_{file_name}_{index + 1}.npy", signal_chunk.to_numpy())
        np.save(out_dir / f"msk_{file_name}_{index + 1}.npy", obs_mask)
        np.save(out_dir / f"att_{file_name}_{index + 1}.npy", attr_chunk.to_numpy())


def _chunk_stems_ready(out_dir: Path, parquet_stem: str) -> bool:
    sigs = sorted(out_dir.glob(f"sig_{parquet_stem}_*.npy"))
    msks = sorted(out_dir.glob(f"msk_{parquet_stem}_*.npy"))
    atts = sorted(out_dir.glob(f"att_{parquet_stem}_*.npy"))
    return bool(sigs) and len(sigs) == len(msks) == len(atts)


def prepare_can_split(
    spec: DatasetSpec,
    split_path: Path,
    scaler_path: Path,
    fit_scaler: bool,
    n_jobs: int | None = None,
) -> Path:
    """Download-side preprocessing for one SynCAN/ROAD split directory.

    Returns the directory that holds ``sig_*.npy`` / ``msk_*.npy`` / ``att_*.npy``
    for ``spec.filling``. Observation masks are always written (pre-fill).
    """
    split_path = Path(split_path)
    if not any(split_path.glob("*.csv")):
        raise FileNotFoundError(
            f"No raw CSV files in {split_path}. Download the dataset or point root at an existing copy."
        )

    generated = split_path / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    workers = n_jobs if n_jobs is not None else min(8, os.cpu_count() or 1)
    filling = normalize_filling(spec.filling)
    out_dir = windows_dir(split_path, filling)

    if not list(generated.glob("gen_*.parquet")):
        for csv_path in sorted(split_path.glob("*.csv")):
            _sparse_one_file(spec, csv_path.stem, csv_path, scaler_path, fit_scaler, workers)
    elif fit_scaler and not scaler_path.exists():
        for parquet_path in sorted(generated.glob("gen_*.parquet")):
            frame = pd.read_parquet(parquet_path, engine="pyarrow", columns=list(spec.features))
            _update_minmax(spec, frame, scaler_path)

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing scaler {scaler_path}. Prepare the train/ambient split first so min/max can be fit."
        )

    file_enum = _write_file_index(split_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in sorted(generated.glob("gen_*.parquet")):
        if not _chunk_stems_ready(out_dir, parquet_path.stem):
            for pattern in (f"sig_{parquet_path.stem}_*.npy", f"msk_{parquet_path.stem}_*.npy", f"att_{parquet_path.stem}_*.npy"):
                for stale in out_dir.glob(pattern):
                    stale.unlink()
            _npy_one_file(spec, parquet_path.stem, parquet_path, file_enum, out_dir)
    return out_dir
