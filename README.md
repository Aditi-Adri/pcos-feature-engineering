# PCOS: Representation vs Algorithm

Does feature engineering matter more than the choice of model? This tests it on a PCOS
dataset of 1,000 patients.

The labels follow a fixed rule, so 100% accuracy is possible. **These are not clinical
results.**

## Result

Seven algorithms, same data, different features:

| Algorithm | raw | scaled | engineered |
|---|---:|---:|---:|
| Gaussian Naive Bayes | 72.1% | 67.4% | 100% |
| SVM (RBF) | 80.1% | 96.5% | 100% |
| K-Nearest Neighbours | 80.6% | 96.4% | 100% |
| Logistic Regression | 91.0% | 91.0% | 100% |
| Decision Tree | 99.9% | 99.9% | 100% |
| Random Forest | 99.9% | 99.9% | 100% |
| HistGradientBoosting | 99.9% | 99.9% | 100% |

On raw data the algorithms differ by 33 points. After adding four threshold columns they all
match. The model was never the problem.

Scaling helps SVM and KNN by 16 points and does nothing for the trees, which is expected.

## Why 100% is possible

The labels come from a rule:

```python
PCOS = (Menstrual_Irregularity == 1) & (BMI > 25) & (Testosterone > 40) & (Antral_Follicle_Count > 10)
```

It matches all 1,000 labels exactly. There is no noise, so a perfect score is possible. The
data is synthetic.

## Is it leaking?

No. The main check: flipping 5 / 10 / 20 / 40% of labels at random drops accuracy to
95 / 90 / 80 / 60%. A leaking pipeline would stay near 100%.

Also checked: no test row shares a feature vector with training, the scaler is fit on
training data only, and all 5 folds find the same cut-points.

## Preprocessing

The raw file passed all 13 quality checks, so most standard steps were tested and rejected.
Four were needed: rename a column for XGBoost, scale the continuous columns, split early and
stratified, and turn the rule into features.

The cut-points are learned inside `fit()` by `ClinicalThresholdEncoder`, so they are re-fit
on every fold and a validation fold never affects its own score.

## Models

XGBoost, CatBoost and a PyTorch MLP. All three get 100% on the held-out 200 patients, so
each notebook also measures decision margin, which is the only thing that still separates
them.

Two notes:
- XGBoost early-stops on the **last** metric in `eval_metric`. With `error` last it stopped
  at one tree. Reordering to `["error", "logloss"]` fixed it.
- The GPU was slower than the CPU every time. The data is 45 KB, so setup costs more than
  the work. This is measured, not assumed.

## Files

```
pcos_dataset.csv     1,000 patients
eda.ipynb            finds the rule
preprocess.ipynb     audit, split, 5 feature versions
model/               XGBoost, CatBoost, PyTorch MLP + results
data/processed/      the 5 feature versions
artifacts/           cut-points and scaler values as JSON
```

Run order: `eda.ipynb` → `preprocess.ipynb` → any notebook in `model/`. All notebooks are
saved with outputs, so the charts show on GitHub without running anything.

There are no `.py` files and no pickled pipelines. A transformer defined in a notebook saves
as `__main__.ClinicalThresholdEncoder` and won't load anywhere else. The JSON recipe in
`artifacts/` works from plain pandas instead.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

The model notebooks use CUDA. For CPU, change `device="cuda"` to `"cpu"` (XGBoost),
`task_type="GPU"` to `"CPU"` (CatBoost), and `torch.device("cuda")` to `"cpu"` (PyTorch).
Same results, and faster here. Run them one at a time on a 6 GB card.

## Limitations

- Not clinical performance. The labels are rule-generated, so finding the rule is correct by
  definition. Real PCOS diagnosis is not deterministic.
- The dataset is synthetic. Clean uniform features are a property of the generator.
- `lean` and `count` are close to lookup tables. Four binary columns give 16 combinations, so
  every test patient matches one seen in training. Not leakage, but their 100% doesn't show
  generalisation. `raw` and `engineered` are the stronger case.
- `Menstrual_Irregularity` is constant within the positive class, so the class covariance is
  singular. QDA and some LDA solvers need `reg_param > 0`.
- No hyperparameter search. Accuracy is already at the ceiling, so tuning would measure
  nothing.

## Next

Run the same pipeline on real data, such as the Kerala PCOS dataset (~540 patients). The
method transfers directly, and there the model comparison would have a winner.

## License

MIT. The dataset is synthetic and has no real patient data.
