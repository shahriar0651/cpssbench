"""API tests that do not download the full datasets."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpssbench import list_datasets, load
from cpssbench.preprocess import apply_filling, windows_dir
from cpssbench.specs import get_spec, normalize_filling
from cpssbench.windows import WindowDataset


class RegistryTests(unittest.TestCase):
    def test_list_includes_ready_and_planned(self):
        names = {row["name"] for row in list_datasets()}
        self.assertIn("syncan", names)
        self.assertIn("road", names)
        self.assertIn("x-canids", names)

    def test_planned_dataset_is_not_loadable(self):
        with self.assertRaises(NotImplementedError):
            load("x-canids", download=False)

    def test_unknown_dataset(self):
        with self.assertRaises(KeyError):
            load("not-a-dataset")


class FillingTests(unittest.TestCase):
    def test_normalize_aliases(self):
        self.assertEqual(normalize_filling("nan"), "none")
        self.assertEqual(normalize_filling("ffill"), "forward")
        self.assertEqual(normalize_filling("0"), "zero")

    def test_apply_filling_modes(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [np.nan, 2.0, np.nan]})
        forward = apply_filling(frame, "forward")
        self.assertFalse(forward.isna().any().any())
        none = apply_filling(frame, "none")
        self.assertTrue(none.isna().any().any())
        zero = apply_filling(frame, "zero")
        self.assertEqual(float(zero.iloc[1, 0]), 0.0)

    def test_filling_override_can_only(self):
        can = get_spec("road").with_overrides(filling="none")
        self.assertEqual(can.filling, "none")
        with self.assertRaises(ValueError):
            get_spec("misbehaviorx").with_overrides(filling="none")

    def test_v2x_rejects_filling_kwarg(self):
        with self.assertRaises(ValueError):
            load("misbehaviorx", download=False, filling="none")

    def test_v2x_rejects_return_mask(self):
        with self.assertRaises(ValueError):
            load("misbehaviorx", download=False, return_mask=True)

    def test_windows_dir_is_per_filling(self):
        root = Path("/tmp/split")
        self.assertEqual(windows_dir(root, "nan"), root / "generated" / "none")


class WindowDatasetTests(unittest.TestCase):
    def _write_can_fixture(self, data_dir: Path, filling: str, signals: np.ndarray, mask: np.ndarray, attributes: np.ndarray, features):
        generated = data_dir / "generated" / filling
        generated.mkdir(parents=True)
        np.save(generated / "sig_ambient_1.npy", signals)
        np.save(generated / "msk_ambient_1.npy", mask)
        np.save(generated / "att_ambient_1.npy", attributes)
        pd.DataFrame({"Min": np.zeros(len(features)), "Max": np.ones(len(features))}, index=features).to_csv(
            data_dir / "scaler.csv"
        )
        with open(data_dir / "file_index_dict.json", "w", encoding="utf-8") as handle:
            json.dump({"ambient": 0}, handle)

    def test_getitem_matches_mnist_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            spec = get_spec("road").with_overrides(window_size=4, step_size=2)
            features = list(spec.features)
            n_rows = 20
            signals = np.linspace(0, 1, n_rows * len(features), dtype=np.float32).reshape(n_rows, len(features))
            mask = np.ones_like(signals, dtype=np.float32)
            attributes = np.zeros((n_rows, 4), dtype=np.float32)
            attributes[10:, 2] = 1
            self._write_can_fixture(data_dir, "forward", signals, mask, attributes, features)

            dataset = WindowDataset(spec, data_dir, data_dir / "scaler.csv", return_meta=False)
            window, label = dataset[0]
            self.assertEqual(tuple(window.shape), spec.input_shape)
            self.assertIsInstance(label, int)
            self.assertEqual(label, 0)

            window, label, meta = WindowDataset(spec, data_dir, data_dir / "scaler.csv", return_meta=True)[-1]
            self.assertEqual(label, 1)
            self.assertIn("file", meta)
            self.assertIn("obs_mask", meta)
            self.assertEqual(tuple(meta["obs_mask"].shape), tuple(window.shape))
            self.assertTrue(torch.is_tensor(window))

    def test_forward_fill_with_return_mask_zeros_gaps(self):
        """Forward-filled cache + return_mask → x zeros at original gaps, mask is 0/1."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            spec = get_spec("road").with_overrides(window_size=4, step_size=2, filling="forward")
            features = list(spec.features)
            n_rows = 20
            # Simulate forward-filled dense signals, but mask remembers original gaps.
            signals = np.full((n_rows, len(features)), 0.8, dtype=np.float32)
            mask = np.zeros_like(signals, dtype=np.float32)
            mask[0, 0] = 1.0
            mask[2, 1] = 1.0
            attributes = np.zeros((n_rows, 4), dtype=np.float32)
            self._write_can_fixture(data_dir, "forward", signals, mask, attributes, features)

            x, y, obs = WindowDataset(spec, data_dir, data_dir / "scaler.csv", return_mask=True)[0]
            self.assertEqual(tuple(x.shape), spec.input_shape)
            self.assertEqual(tuple(obs.shape), tuple(x.shape))
            self.assertEqual(obs.dtype, torch.float32)
            self.assertTrue(torch.all((obs == 0) | (obs == 1)))
            # Gaps must be zero in x even though the cache was forward-filled.
            self.assertTrue(torch.all(x[obs == 0] == 0))
            self.assertGreater(float(x[obs == 1].abs().sum()), 0.0)
            # Without return_mask, dense filled values are kept.
            x_dense, _ = WindowDataset(spec, data_dir, data_dir / "scaler.csv", return_mask=False)[0]
            self.assertFalse(torch.all(x_dense == 0))

    def test_mae_convention_none_filling(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            spec = get_spec("road").with_overrides(window_size=4, step_size=2, filling="none")
            features = list(spec.features)
            n_rows = 20
            signals = np.full((n_rows, len(features)), np.nan, dtype=np.float32)
            signals[0, 0] = 0.5
            signals[2, 1] = 0.25
            mask = (~np.isnan(signals)).astype(np.float32)
            attributes = np.zeros((n_rows, 4), dtype=np.float32)
            self._write_can_fixture(data_dir, "none", signals, mask, attributes, features)

            x, y, obs = WindowDataset(spec, data_dir, data_dir / "scaler.csv", return_mask=True)[0]
            self.assertFalse(torch.isnan(x).any())
            self.assertTrue(torch.all(x[obs == 0] == 0))
            self.assertEqual(tuple(obs.shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main()
