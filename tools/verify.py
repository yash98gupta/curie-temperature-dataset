"""Verify the released dataset, model, preprocessing, predictions, and metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def release_bytes(path):
    content = path.read_bytes()
    return content if path.suffix == ".png" else content.replace(b"\r\n", b"\n")


root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--write-manifest", action="store_true")
arguments = parser.parse_args()
results = json.loads((root / "tools/results.json").read_text())
preprocessing = json.loads((root / "tools/preprocessing.json").read_text())
source = root / "curie_temperature_dataset.csv"
assert hashlib.sha256(release_bytes(source)).hexdigest() == results["dataset_sha256"]
dataset = pd.read_csv(source)
train_indices = preprocessing["train_indices"]
test_indices = preprocessing["test_indices"]
assert len(train_indices) == 536 and len(test_indices) == 134
assert set(train_indices).isdisjoint(test_indices)
assert set(train_indices).union(test_indices) == set(range(len(dataset)))
features = dataset[preprocessing["features"]].to_numpy(dtype=float)
scaler = StandardScaler().fit(features[train_indices])
np.testing.assert_allclose(scaler.mean_, preprocessing["mean"])
np.testing.assert_allclose(scaler.scale_, preprocessing["scale"])
model = XGBRegressor()
model.load_model(root / "tools/xgboost_seed42.json")
predicted = np.expm1(model.predict(scaler.transform(features[test_indices])))
reported = dataset.tc.iloc[test_indices].to_numpy()
stored = pd.read_csv(root / "tools/test_predictions.csv")
assert stored.row_index.tolist() == test_indices
assert stored.formula.tolist() == dataset.formula.iloc[test_indices].tolist()
np.testing.assert_array_equal(stored.reported_K, reported)
np.testing.assert_allclose(stored.predicted_K, predicted, rtol=1e-7, atol=1e-4)
np.testing.assert_allclose(r2_score(reported, predicted), results["table1"]["XGBoost"]["test_R2"])
np.testing.assert_allclose(mean_absolute_error(reported, predicted), results["table1"]["XGBoost"]["MAE"])
for scores in results["table1"].values():
    for scale in ("K", "log"):
        np.testing.assert_allclose(np.mean(scores[f"cv_{scale}_folds"]), scores[f"cv_{scale}_mean"])
        np.testing.assert_allclose(np.std(scores[f"cv_{scale}_folds"]), scores[f"cv_{scale}_std"])
manifest_path = root / "SHA256SUMS.json"
if arguments.write_manifest:
    names = [".gitattributes", ".gitignore", "README.md", "LICENSE", "LICENSE-DATA.txt", "requirements.txt",
             "curie_temperature_dataset.csv", "tools/analysis.py", "tools/verify.py", "tools/results.json",
             "tools/preprocessing.json", "tools/xgboost_seed42.json", "tools/test_predictions.csv"]
    names.extend(path.relative_to(root).as_posix() for path in sorted((root / "tools/figs_new").glob("*.png")))
    manifest = {name: hashlib.sha256(release_bytes(root / name)).hexdigest() for name in names}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
for name, digest in json.loads(manifest_path.read_text()).items():
    assert hashlib.sha256(release_bytes(root / name)).hexdigest() == digest, name
print("PASS: dataset hash, split, scaler, saved predictions, test metrics, CV summaries, and release manifest")