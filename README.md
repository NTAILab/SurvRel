# SurvRel - Relation-Aware Beran Survival Estimator

> **Code accompanying a paper at IITI26.**
> The citation block will be added here later.

**Single-file implementation** of neural survival analysis with relational data support.
The single file `surv_rel.py` contains both an **importable module** (classes, functions,
dataclasses) and a **full CLI interface** (five commands: `train`, `predict`, `evaluate`,
`make-relations-csv`, `benchmark`).

## What it is

The module implements a Beran estimator whose kernel is built on top of a neural network.
It takes into account not only the patients' features but also **relation matrices** between
samples (e.g., "treated at the same clinic", "shares a genotype", or any non-numeric links).
It predicts the survival function S(t), from which the expected event time is derived.

The kernel comes in two flavors:

| `model_type` | Description |
|---|---|
| `mlp` (default) | Feature MLP encoder + MLP head that merges the distance with the relation matrix |
| `gaussian` | Learnable anisotropic Gaussian kernel over features + learnable relation weights |

The main class is `RelationAwareBeranEstimator` (inherits from `sklearn.base.BaseEstimator`,
which makes it compatible with the scikit-learn ecosystem: `get_params`, cross-validation, etc.).

## Dependencies

```bash
pip install numpy pandas torch scikit-learn scikit-survival
# optional (for the benchmark command):
pip install optuna
```

## Data format

| Object | Format |
|---|---|
| Features `X` | `float32` ndarray of shape `(n_samples, n_features)` |
| Target `y` | numpy record array with fields `cens` (bool, 1 = event) and `time` (float). Created via `make_survival_array(times, events)` |
| Relations `relations` | `float32` ndarray of shape `(n_rows, n_background, n_relations)`. For training — `(n_train, n_train, R)`; for prediction — `(n_test, n_train, R)`. If there are no relations, `R = 0` (empty third axis) is allowed |

Relation matrices can be stored in `.npy`/`.npz` or CSV. Two CSV layouts are supported:
a single matrix with `n_background` columns, or grouped headers of the form
`rel0__0, rel0__1, ..., rel0__N, rel1__0, ...`, where the suffix `__<idx>` is the index
of a background sample.

---

## Using it as a module

### Training

```python
import numpy as np
from surv_rel import (
    RelationAwareBeranEstimator,
    make_survival_array,
)

X = np.random.randn(200, 5).astype(np.float32)
times = np.random.exponential(scale=10, size=200)
events = np.random.binomial(1, 0.7, size=200).astype(bool)
y = make_survival_array(times, events)

# a single relation matrix (200 x 200 x 1)
relations = (np.random.rand(200, 200, 1) > 0.5).astype(np.float32)

estimator = RelationAwareBeranEstimator(
    model_type="mlp",
    n_epochs=300,
    batch_size=32,
    learning_rate=1e-3,
    device="cpu",
    random_state=123,
    verbose=True,
)
estimator.fit(X, y, relations)
```

### Prediction

```python
X_new = np.random.randn(50, 5).astype(np.float32)
test_relations = (np.random.rand(50, 200, 1) > 0.5).astype(np.float32)

# survival table: S(t) for every point of the time grid
survival_table, time_grid = estimator.predict_survival_table(X_new, test_relations)

# survival functions as pd.Series
survival_curves = estimator.predict_survival_function(X_new, test_relations, return_array=False)

# expected event time
expected_times = estimator.predict_expected_time(X_new, test_relations)
# same as estimator.predict(X_new, test_relations)
```

### Quality metrics

```python
from surv_rel import calc_survival_metrics

metrics = calc_survival_metrics(y_train, y_test, survival_table, time_grid)
# {'C-index': ..., 'IBS': ..., 'CDAUC': ...}
```

The metrics use the "safe" variants from `scikit-survival`: IBS and cumulative dynamic AUC
are restricted to the adequate time region where the estimates are valid.

### Preparing data from tables (CSV)

```python
from surv_rel import prepare_table_data

prepared = prepare_table_data(
    train_csv="train.csv",
    time_col="time",
    event_col="event",
    feature_cols=["age", "bmi"],            # optional; otherwise all non-service columns are used
    relation_cat_cols=["clinic", "group"],  # categorical relations are built automatically
    test_csv="test.csv",
)

estimator = RelationAwareBeranEstimator()
estimator.fit(prepared.X, prepared.y, prepared.train_relations)
survival_table, time_grid = estimator.predict_survival_table(
    prepared.test_X, prepared.test_relations
)
```

Relations derived from categorical columns are binary match matrices
(`build_categorical_relations`); missing values are replaced with a `__MISSING__` marker.

### Serialization

```python
from surv_rel import save_pickle, load_pickle

save_pickle(estimator, "model.pkl")
model = load_pickle("model.pkl")
```

---

## CLI

All commands are invoked as `python surv_rel.py <command> [options]`.

### 1. `train` — training

```bash
python surv_rel.py train \
    --train-csv train.csv \
    --time-col time \
    --event-col event \
    --feature-cols age,bmi,marker \
    --relation-categorical-cols clinic \
    --model-type mlp \
    --n-epochs 300 \
    --learning-rate 1e-3 \
    --device cpu \
    --test-csv test.csv \
    --model-out model.pkl \
    --survival-out survival_test.csv \
    --expected-time-out expected_test.csv
```

Main model options (shared by all `add_shared_model_args`): `--model-type`,
`--n-epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`,
`--inverse-temperature`, `--loss-k`, `--embed-dim`, `--embed-layers`,
`--merge-hidden-dim`, `--merge-layers`, `--validation-fraction`, `--device`,
`--random-state`, `--verbose`, `--no-balance-events`, `--no-tied-times-correction`.

### 2. `predict` — inference with a saved model

```bash
python surv_rel.py predict \
    --model-in model.pkl \
    --test-csv new_data.csv \
    --feature-cols age,bmi,marker \
    --train-csv train.csv \
    --relation-categorical-cols clinic \
    --survival-out survival.csv \
    --expected-time-out expected.csv
```

If both output files are omitted, the survival table is printed to stdout.

### 3. `evaluate` — scoring saved predictions

```bash
python surv_rel.py evaluate \
    --train-csv train.csv \
    --test-csv test.csv \
    --time-col time \
    --event-col event \
    --survival-csv survival.csv \
    --expected-time-csv expected.csv \
    --output-csv metrics.csv
```

### 4. `make-relations-csv` — building relation matrices as CSV

```bash
python surv_rel.py make-relations-csv \
    --train-csv train.csv \
    --relation-categorical-cols clinic,group \
    --train-out train_relations.csv \
    --test-csv test.csv \
    --test-out test_relations.csv \
    --relation-prefix rel
```

### 5. `benchmark` — model comparison with Optuna tuning

```bash
python surv_rel.py benchmark \
    --train-csv train.csv \
    --time-col time \
    --event-col event \
    --models gaussian,mlp \
    --metric C-index \
    --cv-splits 5 \
    --optuna-trials 15 \
    --output-csv benchmark_results.csv
```

For each model (`gaussian`, `mlp`) and the RandomSurvivalForest baseline, an Optuna
search over the hyperparameter grid is run with stratified cross-validation; the results
include C-index, IBS, CDAUC, and the best parameters.

---

## Implementation details

- **Training**: leave-one-out within a batch (each sample is predicted from the other
  samples of the batch with a diagonal attention mask), or against a background set.
- **Loss function** `k_survival_loss`: log of a "smeared" survival step of width `k`
  for event samples, and log of S(t) for censored ones.
- **Class balancing** of event/censoring via `WeightedRandomSampler`.
- **Tied-time correction** (`tied_times_correction`): aggregation of survival steps for
  coinciding times, both before and after prediction.
- **Validation** is stratified by the event indicator (`validation_fraction`);
  the loss history is stored in `estimator.history_`.
- Relations without source data (`n_relations = 0`) are handled correctly by the model.

