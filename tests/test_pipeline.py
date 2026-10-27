"""Use the one synthetic generator to exercise the researcher workflow."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from bgt_esre.attention import export_attention
from bgt_esre.cli import main
from bgt_esre.config import Config
from bgt_esre.data import Dataset, load_dataset, make_synthetic
from bgt_esre.splits import create_splits, load_or_create_splits, validate_splits
from bgt_esre.training import load_checkpoint, metrics, predict, train_cv

torch.set_num_threads(1)


def tiny_config():
    config = Config()
    config.model.hidden_dim, config.model.num_heads, config.model.num_layers = 8, 2, 1
    config.training.folds, config.training.epochs, config.training.batch_size = 3, 2, 8
    return config


class DataAndSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = make_synthetic(self.root / "data.npz", 30, 8, 24)

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_and_generator_reproducibility(self):
        other = make_synthetic(self.root / "same.npz", 30, 8, 24)
        self.assertEqual(self.dataset.fingerprint(), other.fingerprint())
        self.assertEqual(
            self.dataset.fingerprint(),
            load_dataset(self.root / "data.npz").fingerprint(),
        )
        with self.assertRaises(FileExistsError):
            make_synthetic(self.root / "data.npz")

    def test_schema_checks(self):
        data = self.dataset
        with self.assertRaisesRegex(ValueError, "symmetric"):
            broken = data.connectivity.copy()
            broken[0, 0, 1] += 1
            Dataset(broken, data.coords)
        with self.assertRaisesRegex(ValueError, "unique"):
            Dataset(
                data.connectivity,
                data.coords,
                subject_ids=np.repeat("duplicate", len(data)),
            )
        with self.assertRaisesRegex(ValueError, "labels"):
            Dataset(data.connectivity, data.coords, labels=np.repeat(2, len(data)))
        with self.assertRaises(ValueError):
            Dataset(data.connectivity, np.ones((4, 3)))

    def test_config_rejects_typos_and_invalid_geometry(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"model": {"hidden_dims": 32}})
        with self.assertRaises(ValueError):
            Config.from_dict({"model": {"hidden_dim": 12, "num_heads": 4}})
        with self.assertRaises(ValueError):
            Config.from_dict({"graph": {"measures": ["ediff", "ediff"]}})

    def test_outer_tests_are_disjoint_and_splits_are_reproducible(self):
        config = tiny_config().training
        folds = create_splits(self.dataset, config)
        self.assertEqual(folds, create_splits(self.dataset, config))
        tests = [index for fold in folds for index in fold["test"]]
        self.assertEqual(sorted(tests), list(range(len(self.dataset))))
        for fold in folds:
            self.assertFalse(set(fold["train"]) & set(fold["validation"]))
            self.assertFalse(set(fold["train"]) & set(fold["test"]))

    def test_group_partitions_never_share_groups(self):
        self.dataset.groups = np.array(
            [f"family_{i // 2}" for i in range(len(self.dataset))]
        )
        folds = create_splits(self.dataset, tiny_config().training)
        for fold in folds:
            groups = [
                set(self.dataset.groups[fold[key]])
                for key in ("train", "validation", "test")
            ]
            self.assertFalse(
                groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
            )

    def test_saved_splits_reject_reordered_or_modified_data(self):
        out = self.root / "run"
        out.mkdir()
        load_or_create_splits(self.dataset, tiny_config().training, out)
        self.dataset.connectivity = self.dataset.connectivity[::-1].copy()
        with self.assertRaisesRegex(ValueError, "values/order"):
            load_or_create_splits(self.dataset, tiny_config().training, out)

    def test_split_validator_rejects_repeated_tests(self):
        config = tiny_config().training
        folds = create_splits(self.dataset, config)
        folds[1] = deepcopy(folds[0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_splits(folds, self.dataset, config.folds)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = make_synthetic(cls.root / "data.npz", 30, 8, 24)
        cls.config = tiny_config()
        with redirect_stdout(io.StringIO()):
            cls.summary = train_cv(cls.dataset, cls.config, cls.root / "run", "cpu")
        cls.checkpoint = cls.root / "run" / "fold_0" / "best.pt"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_selected_checkpoint_reproduces_test_metrics(self):
        result = predict(
            self.dataset, self.checkpoint, self.root / "evaluate", "cpu", evaluate=True
        )
        self.assertEqual(result["metrics"], self.summary["folds"][0]["test"])
        with open(self.root / "run" / "predictions.csv") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(self.dataset))
        self.assertEqual(len(set(row["subject_id"] for row in rows)), len(rows))
        self.assertTrue(np.isfinite(self.summary["mean_std"]["auc"]["mean"]))

    def test_unlabeled_prediction_and_atlas_validation(self):
        unlabeled = Dataset(
            self.dataset.connectivity,
            self.dataset.coords,
            self.dataset.features,
            subject_ids=self.dataset.subject_ids,
        )
        probs = predict(unlabeled, self.checkpoint, self.root / "predict", "cpu")
        self.assertEqual(probs.shape, (len(self.dataset), 2))
        np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
        unlabeled.coords = unlabeled.coords[::-1].copy()
        with self.assertRaisesRegex(ValueError, "atlas"):
            load_checkpoint(self.checkpoint, unlabeled, "cpu")

    def test_attention_uses_saved_test_subjects_once(self):
        rows = export_attention(
            self.dataset, self.root / "run", self.root / "attention", "cpu", 0.1
        )
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(row["receiver"] != row["sender"] for row in rows))
        self.assertTrue(
            all(
                row["present_0"] + row["present_1"] <= len(self.dataset) for row in rows
            )
        )
        self.assertTrue((self.root / "attention" / "ecospace.png").is_file())
        metadata = json.loads((self.root / "attention" / "metadata.json").read_text())
        self.assertEqual(metadata["partition"], "outer test only")

    def test_profile_additive_and_connectivity_ablation_train(self):
        config = tiny_config()
        config.model.input_mode, config.model.attention = "profile", "additive"
        config.graph.weighting = "connectivity"
        config.graph.measures = ("ebc", "comm")
        config.training.epochs = 1
        similarity = Dataset(
            np.maximum(self.dataset.connectivity, 0),
            self.dataset.coords,
            labels=self.dataset.labels,
            subject_ids=self.dataset.subject_ids,
        )
        with redirect_stdout(io.StringIO()):
            result = train_cv(similarity, config, self.root / "profile", "cpu")
        self.assertEqual(len(result["folds"]), 3)

    def test_all_pairs_cli_uses_identical_partitions(self):
        import yaml

        config = tiny_config()
        config.training.epochs, config.training.folds = 1, 2
        path = self.root / "tiny.yaml"
        path.write_text(yaml.safe_dump(config.to_dict()))
        with redirect_stdout(io.StringIO()):
            main(
                [
                    "train",
                    "--data",
                    str(self.root / "data.npz"),
                    "--config",
                    str(path),
                    "--out",
                    str(self.root / "pairs"),
                    "--all-pairs",
                    "--device",
                    "cpu",
                ]
            )
        summaries = json.loads((self.root / "pairs" / "pairs.json").read_text())
        self.assertEqual(len(summaries), 10)
        expected = json.loads((self.root / "pairs" / "splits.json").read_text())[
            "folds"
        ]
        for name in summaries:
            actual = json.loads(
                (self.root / "pairs" / name / "splits.json").read_text()
            )["folds"]
            self.assertEqual(actual, expected)

    def test_nonempty_output_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            train_cv(self.dataset, self.config, self.root / "run")

    def test_single_class_external_metrics_have_defined_accuracy(self):
        result = metrics(np.zeros(3, dtype=int), np.array([[0.8, 0.2]] * 3))
        self.assertEqual(result["accuracy"], 1)
        self.assertTrue(np.isnan(result["auc"]))


if __name__ == "__main__":
    unittest.main()
