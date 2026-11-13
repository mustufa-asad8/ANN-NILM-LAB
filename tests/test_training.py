import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nilm_ccp import train as nilm_ccp


class EvaluationSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = argparse.Namespace(
            region="austin",
            home_ids=[10, 20, 30, 40, 50],
            test_home=50,
            window=120,
            train_stride=20,
            epochs=20,
            patience=5,
            batch_size=512,
            seed=466,
        )

    def test_cli_default_window_matches_committed_experiment(self) -> None:
        with patch.object(sys, "argv", ["train.py"]):
            args = nilm_ccp.parse_args()
        self.assertEqual(args.window, 120)

    def test_evaluation_requires_saved_training_configuration(self) -> None:
        metadata = {
            "region": self.args.region,
            "models": list(nilm_ccp.MODEL_NAMES),
            "home_ids": self.args.home_ids,
            "test_home": self.args.test_home,
            "window_minutes": self.args.window,
            "train_stride": self.args.train_stride,
            "max_epochs": self.args.epochs,
            "early_stopping_patience": self.args.patience,
            "batch_size": self.args.batch_size,
            "seed": self.args.seed,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.args.output_dir = Path(temporary_directory)
            metadata_path = self.args.output_dir / "metadata" / "run_metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            nilm_ccp.validate_evaluation_setup(self.args)

            self.args.window = 60
            with self.assertRaisesRegex(ValueError, "original training configuration"):
                nilm_ccp.validate_evaluation_setup(self.args)

    def test_checkpoint_preprocessing_must_match_data(self) -> None:
        checkpoint = {
            "input_mean_kw": 1.0,
            "input_std_kw": 2.0,
            "on_thresholds_kw": {
                "air1": 0.05,
                "furnace1": 0.05,
                "drye1": 0.357,
            },
        }
        thresholds = dict(checkpoint["on_thresholds_kw"])
        nilm_ccp.validate_checkpoint_preprocessing(
            checkpoint, 1.0, 2.0, thresholds, "MLP"
        )
        thresholds["drye1"] = 0.4
        with self.assertRaisesRegex(ValueError, "ON thresholds"):
            nilm_ccp.validate_checkpoint_preprocessing(
                checkpoint, 1.0, 2.0, thresholds, "MLP"
            )

    def test_all_model_architectures_return_appliance_outputs(self) -> None:
        for model_name in nilm_ccp.MODEL_NAMES:
            with self.subTest(model=model_name):
                model = nilm_ccp.build_model(model_name, window_size=120)
                status_logits, estimated_power = model(
                    nilm_ccp.torch.zeros((2, 120), dtype=nilm_ccp.torch.float32)
                )
                self.assertEqual(tuple(status_logits.shape), (2, 3))
                self.assertEqual(tuple(estimated_power.shape), (2, 3))
                self.assertTrue(nilm_ccp.torch.all(estimated_power >= 0))


if __name__ == "__main__":
    unittest.main()
