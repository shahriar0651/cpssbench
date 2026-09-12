"""API tests that do not download the full datasets."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpsbench import list_datasets, load
from cpsbench.specs import get_spec
from cpsbench.windows import WindowDataset


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


class WindowDatasetTests(unittest.TestCase):
    def test_getitem_matches_mnist_contract(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            generated = data_dir / "generated"
            generated.mkdir()
            spec = get_spec("road").with_overrides(window_size=4, step_size=2)
            features = list(spec.features)
            n_rows = 20
            signals = np.linspace(0, 1, n_rows * len(features), dtype=np.float32).reshape(n_rows, len(features))
            attributes = np.zeros((n_rows, 4), dtype=np.float32)
            attributes[:, 2] = 0
            attributes[10:, 2] = 1
            attributes[:, 3] = 0
            np.save(generated / "sig_ambient_1.npy", signals)
            np.save(generated / "att_ambient_1.npy", attributes)
            pd.DataFrame({"Min": np.zeros(len(features)), "Max": np.ones(len(features))}, index=features).to_csv(
                Path(tmp) / "scaler.csv"
            )
            with open(data_dir / "file_index_dict.json", "w", encoding="utf-8") as handle:
                json.dump({"ambient": 0}, handle)

            dataset = WindowDataset(spec, data_dir, Path(tmp) / "scaler.csv", return_meta=False)
            window, label = dataset[0]
            self.assertEqual(tuple(window.shape), spec.input_shape)
            self.assertIsInstance(label, int)
            self.assertEqual(label, 0)

            window, label, meta = WindowDataset(spec, data_dir, Path(tmp) / "scaler.csv", return_meta=True)[-1]
            self.assertEqual(label, 1)
            self.assertIn("file", meta)
            self.assertTrue(torch.is_tensor(window))


if __name__ == "__main__":
    unittest.main()
