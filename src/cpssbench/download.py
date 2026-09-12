"""Download raw vehicular datasets into a local root directory."""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .specs import DatasetSpec


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be fetched or unpacked."""


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise DownloadError(
            f"'{name}' is required to download this dataset but was not found on PATH."
        )


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc


def split_dirname(split: str) -> str:
    key = split.strip().lower()
    mapping = {
        "train": "ambient",
        "training": "ambient",
        "ambient": "ambient",
        "test": "attacks",
        "testing": "attacks",
        "attack": "attacks",
        "attacks": "attacks",
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported split '{split}'. Use 'train'/'ambient' or 'test'/'attacks'."
        )
    return mapping[key]


def dataset_root(root: str | Path, name: str) -> Path:
    return Path(root).expanduser().resolve() / name


def split_dir(root: str | Path, name: str, split: str) -> Path:
    return dataset_root(root, name) / split_dirname(split)


def raw_csvs_present(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.csv"))


def download_syncan(destination: Path) -> None:
    """Clone SynCAN and unpack ambient/attacks next to each other."""
    _require_tool("git")
    destination.mkdir(parents=True, exist_ok=True)
    if raw_csvs_present(destination / "ambient") and raw_csvs_present(destination / "attacks"):
        print(f"SynCAN already present at {destination}")
        return

    clone_dir = destination / "_raw"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    print(f"Cloning SynCAN into {clone_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/etas/SynCAN.git", str(clone_dir)],
        check=True,
    )

    for pattern, folder in (("train_*.zip", "ambient"), ("test_*.zip", "attacks")):
        zips = sorted(clone_dir.glob(pattern))
        if not zips:
            raise DownloadError(f"SynCAN clone is missing {pattern} archives.")
        out = destination / folder
        out.mkdir(parents=True, exist_ok=True)
        for archive in zips:
            print(f"Extracting {archive.name} -> {out}")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(out)

    for normal in (destination / "attacks").glob("test_normal*"):
        if normal.is_file():
            normal.unlink()
    shutil.rmtree(clone_dir, ignore_errors=True)
    print(f"SynCAN downloaded to {destination}")


def download_road(destination: Path) -> None:
    """Download the ROAD signal-extraction release from Zenodo."""
    destination.mkdir(parents=True, exist_ok=True)
    if raw_csvs_present(destination / "ambient") and raw_csvs_present(destination / "attacks"):
        print(f"ROAD already present at {destination}")
        return

    archive = destination / "road.zip"
    _download_file("https://zenodo.org/records/10462796/files/road.zip", archive)
    extract_root = destination / "_raw"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_root)

    signal_dir = next(extract_root.rglob("signal_extractions"), None)
    source = signal_dir if signal_dir is not None else extract_root
    for folder in ("ambient", "attacks"):
        matches = [path for path in source.rglob(folder) if path.is_dir()]
        if not matches:
            raise DownloadError(
                f"ROAD archive does not contain an '{folder}' directory. "
                f"Inspect {extract_root} and place the CSVs manually."
            )
        target = destination / folder
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(matches[0]), str(target))

    # Original release includes non-ambient files inside ambient/.
    for extra in (destination / "ambient").iterdir():
        if extra.is_file() and not extra.name.startswith("ambient_"):
            extra.unlink()

    archive.unlink(missing_ok=True)
    shutil.rmtree(extract_root, ignore_errors=True)
    print(f"ROAD downloaded to {destination}")


def ensure_downloaded(spec: DatasetSpec, root: str | Path) -> Path:
    """Download ``spec`` under ``root/<name>`` and return that directory."""
    destination = dataset_root(root, spec.name)
    if spec.status == "planned":
        raise DownloadError(
            f"{spec.name} is registered but not implemented yet. {spec.notes}"
        )
    if not spec.downloadable:
        if raw_csvs_present(destination / "ambient") or raw_csvs_present(destination / "attacks"):
            return destination
        raise DownloadError(
            f"{spec.name} cannot be downloaded automatically. {spec.notes} "
            f"Expected files under {destination}."
        )
    if spec.name == "syncan":
        download_syncan(destination)
    elif spec.name == "road":
        download_road(destination)
    else:
        raise DownloadError(f"No downloader registered for {spec.name}.")
    return destination
