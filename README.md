# Magnetic Ordering Temperatures: Dataset and Reproducible Models

Supporting data and numerical evaluation for manuscript MRSC-D-26-00309,
"Machine Learning Prediction of Magnetic Ordering Temperatures: Accelerating
the Search for Above-Room-Temperature Magnets for Energy-Efficient Computing."

Authors: Yash Gupta, Aryan Gupta, and Sai Siddhardha Singamaneni.

## Dataset

The unchanged CSV contains 670 rows, 40 material classes, and 132 precomputed
Magpie composition descriptors. Temperatures span 1-1,388 K (median 340 K).
The legacy column `tc` contains **magnetic ordering temperatures**, pooling
Curie temperatures of ferromagnets/ferrimagnets and Neel temperatures of
antiferromagnets; it is not exclusively a Curie-temperature target.

Columns include `formula`, `tc`, `material_class`, `reference`, `composition`,
and the 132 columns beginning with `MagpieData`. No rows were added, removed,
or relabelled for the revision. `tools/results.json` records the input SHA-256.

## Reproduce

The recorded evaluation used Python 3.14.0 on Windows. From the repository root:

```sh
python -m pip install -r requirements.txt
python tools/verify.py
python tools/analysis.py
```

`verify.py` checks the existing model and results without retraining.
`analysis.py` reruns all five models, cross-validation, bootstrap calculations,
seed sensitivity, SHAP attribution, learning curves, and figure generation.
It overwrites the generated files in `tools/`. Package versions are pinned;
platform or dependency differences may produce small numerical differences.

## Evaluation

- The original study design is retained: 670 rows, 132 descriptors, five model
  families, an 80/20 random row split, and main seed 42. The released script
  documents the recalculation used to resolve numerical inconsistencies.
- Models learn `ln(1 + T_ord/K)`; predictions use `expm1` to return to kelvin.
- Feature scaling is fitted only on training data, including within each CV
  fold and learning-curve subset. Five-fold CV is performed once within the
  536 training rows. The held-out set contains 134 rows.
- XGBoost: test R-squared 0.837, MAE 97 K, RMSE 136 K. CV R-squared is
  0.622 +/- 0.070 on the log scale and 0.765 +/- 0.047 on the kelvin scale.
- Bootstrap: 2,000 paired resamples of 134 rows, seed 42, percentile intervals,
  without refitting. These are intervals for aggregate metrics, not individual
  prediction intervals. Ten-seed sensitivity uses seeds 0-9 for split and model.
- Per-fold/per-seed scores, all model metrics, software versions, and the
  exact CSV checksum are in `tools/results.json`.

## Saved Artifacts

`tools/xgboost_seed42.json` stores the fitted model; `tools/preprocessing.json`
stores feature order, scaler statistics, and split indices.
`tools/test_predictions.csv` stores held-out formulas, targets, and predictions.
`tools/figs_new/` contains the figure panels and graphical abstract.
`SHA256SUMS.json` identifies the files in the versioned release.

## Scope and Limitations

Source labels `recent` (23 rows) and `various` (12 rows) are not individual
primary citations. The release contains neither per-entry experimental
uncertainties nor complete magnetic-order labels or upstream conversion logs.
Nine equivalent-composition groups occur under different formula strings;
four span the main train/test split. These random-row results are not a
composition-grouped test of generalization to unseen chemistry.

Composition-only descriptors cannot distinguish polymorphs. Elemental-moment
SHAP attribution does not measure ionic orbital occupancy or prove a particular
exchange mechanism. Predictions support exploratory screening, not device
qualification. The optional timing block is not a matched DFT benchmark.

## Licensing and Citation

Code and trained model weights: MIT, see `LICENSE`.
Compiled data, derived numerical results, and figures: CC BY 4.0, see
`LICENSE-DATA.txt`. Rights in the underlying cited publications are unchanged.
Please cite the authors, repository, and the specific release/commit used.