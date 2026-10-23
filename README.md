# Massimo Stocker -- ETAI project (COMPAS two-year recidivism)

## Overview

This is my pipeline for the ETAI course. We predict whether someone gets rearrested within two years (`two_year_recid`) using ProPublica's COMPAS data, the same data behind the 2016 investigation into the risk score US courts used for bail and sentencing. I started from Sofia's repo (https://github.com/sofiacper/ETAI-Pipeline) and every week I copy in what she changes in class and then play around with it myself.

The goal here is less the prediction itself than whether the model is equally fair to everyone. For that reason `race` is never a feature: it is used only at the end to check false positive rates and compare them with COMPAS's own score. The full data dictionary is in `data/README.md`.

Summary after week 7: the honest accuracy of nearly every model tried sits around 0.67-0.68, regardless of model complexity or amount of tuning. The sections below argue that the limit is the data, not the models.

## Project structure

```
.
├── main.py                  # entry point: runs the whole pipeline
├── config.yaml              # every setting lives here
├── requirements.txt
├── src/
│   ├── data.py              # loading
│   ├── preprocessing.py     # cleaning, imputation/encoding/scaling recipe, locked test set split (weeks 3-4)
│   ├── model.py             # all the models + build_pipeline() = preprocessing + model in one estimator (weeks 5-7)
│   ├── evaluate.py          # cross-validation, metrics, out-of-fold reports, fairness check, paired test (weeks 4-7)
│   ├── tuning.py            # Optuna tuning + nested cross-validation (week 5)
│   └── results.py           # saves the report of each run + the scoreboard row (week 7)
├── results/                 # made automatically: one report per run + runs.csv (not in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md
```

## Pipeline progress (week by week)

This is the log of what changed each week and what I did on top of it. The "what I did myself" parts are my own experiments, the rest is what Sofia showed in class.

### Week 2 -- baseline

Starting point: one train/test split, `dropna()` for missing values, one-hot encoding, logistic regression, and a first fairness check (false positive rate by race, ours vs COMPAS). The weak spots are on purpose: it throws away rows, it judges the model on one single split, and it has no idea what is going on with the data.

### Week 3 -- EDA and preprocessing

Before touching the model we diagnosed the data (`01_eda_introduction`). What came out of it:

- `priors_count` and `prior_offenses` were read as text because of placeholder tokens like `-`. Same story with `?` and `n/a` in `race` and `sex`. Those are missing values in disguise, so they become `NaN`.
- Impossible values (age -3, decile score 23, negative juvenile counts, 250 priors) also become `NaN`. "Impossible but not missing" is still missing.
- Missingness is tested instead of guessed: chi-square plus Cramér's V of each "was it missing" flag against the other categorical columns. `age`, `juv_fel_count`, `race`, `sex` come out MCAR (nothing explains who is missing). `priors_count` and `c_charge_degree` are tied to `age_cat`, so I treat them as MNAR.
- Duplicates: 72 exact duplicate rows, all with a repeated `id`. I check both ways (whole row and `id`) because one check alone can miss a typo'd duplicate.
- Redundant columns: `prior_offenses` and `age_in_months` are exact copies (r = 1.00) of `priors_count` and `age`. `juvenile_total` is the sum of the three `juv_*` columns, which only VIF catches. All three dropped.
- Preprocessing got built as a recipe (`src/preprocessing.py`): clean, impute depending on the mechanism, encode, scale. For the two MNAR columns I impute **and** keep a `_was_missing` flag so the model can still see the pattern.
- The encoder/scaler pair was picked with a grid over 15 repeated splits and checked against the runner-up with a paired comparison. Sofia flagged herself that this grid isn't leak-safe, we only did it that way for this week, and from week 4 it's decided inside the pipeline.

The details of every decision are in the "Preprocessing decisions" table below.

### Week 4 -- cross-validation and the locked test set

This week changed the most. The one-line rule behind it: *from the split onward, held-out rows imitate the future, so anything that learns from data may only look at training rows.*

- **Locked test set**: 20% of the data (stratified, seed 42) is set aside by `split_dev_test()` and never scored by the pipeline, not for fitting, not for tuning, not for choosing. Everything else is the development set. I don't touch `test_set` in the config anymore, a new seed would silently give me a different, already-peeked test set.
- **Stratified 5-fold CV of the whole pipeline** on the development set. Whole pipeline means imputer, encoder, scaler and model are all re-fitted inside every fold, otherwise the validation fold has already influenced the preprocessing and the score is quietly too good. I report each fold, mean ± std and the train-validation gap (a big gap = memorising).
- Out-of-fold predictions: every row is predicted by the model that did not train on it, and the classification report and fairness check use those.
- Target encoding is now sklearn's `TargetEncoder` (it cross-fits, so a row's own label never leaks into its own encoding). Robust scaling because the count columns have a few extreme values.
- Two fixes in cleaning: real `NaN`s in categorical columns were being turned into the text `"nan"` (a fake category, so 229 `c_charge_degree` gaps were never imputed or flagged), fixed by adding `"nan"` to `placeholder_tokens`. And `clean_dataset()` no longer drops rows, duplicates are removed by a separate `drop_duplicate_rows()` on training data only, because on new data every row needs a prediction.
- Added `dummy` (always predicts the majority class) as the floor to beat, and `random_forest`.

### Week 5 -- hyperparameter tuning

- New `src/tuning.py`: **Optuna** with a seeded TPE sampler. Each trial is a 5-fold CV of the whole pipeline on the development set, same folds for every trial.
- **Nested CV** for the honest number: the whole tuning is repeated inside each outer fold and scored on the outer fold it never saw. The best trial's score is shown next to it as an "optimism check", because the best of many noisy scores is always a bit lucky (winner's curse).
- The final model is the tuned pipeline refit on all development rows.
- `build_pipeline()` in `model.py` (preprocessing + model in one place), `tuning` section in the config, `optuna` in `requirements.txt`.

How to read the three tables of a tuned run: table 1 is one tree with fixed settings (CV), table 2 is the nested CV (the honest score of "tune, then refit"), table 3 / the tuning report is the Optuna search on the whole development set that picks the settings I actually keep. If table 3's best trial is much better than the nested score, the tuned parameters were partly lucky. If they're close, the settings are fine.

### Week 6 -- ensembles

- `random_forest` (many trees on bootstrap samples with random features per split, averaged), `oblique_random_forest` (each tree splits on random linear combinations of features, built with `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (shallow trees one after another, each focusing on what the last ones got wrong).
- Search spaces for all three. I don't tune `n_estimators` for the forests, more trees never overfit, they only cost time. AdaBoost is the opposite, its number of rounds does overfit, so there it gets tuned together with the learning rate and the depth of the base trees.
- `corrected_paired_test()` in `evaluate.py` (Nadeau & Bengio). Plain t-tests on CV folds are too optimistic because the folds share most of their training rows, this one corrects for that.
- Sofia showed that a fully grown forest is *not* the best forest on this data. The trees all memorise the same noisy people, so averaging doesn't remove it. Limiting `min_samples_leaf` fixes most of it.

### Week 7 -- gradient boosting and evaluating with probabilities

- Three new models in `_MODELS`: `gradient_boosting` (each new shallow regression tree fits the residual of the log-loss left by the previous ones), `hist_gradient_boosting` (same idea on binned features, much faster and handles missing values itself) and `naive_bayes` (a quick baseline). Search spaces for all three.
- The evaluation now keeps out-of-fold **probabilities**, not just yes/no labels, and reports eight metrics: accuracy, balanced accuracy, precision, recall, F1, F-beta (beta 2), ROC-AUC and average precision. One of them is the **selection metric** (`cv.selection_metric`) that Optuna maximises and runs are ranked by. I picked ROC-AUC, see below.
- `build_scorers()` in `evaluate.py`, and the classification report now has a confusion matrix.
- **Scoreboard**: every `python main.py` adds a row to `results/runs.csv` with the model, its settings, the score on each fold, the no-model floor and the untuned score of the same model. That way I can compare any two runs with `corrected_paired_test()`.
- The printed output is shorter now, the full tables go to the file in `results/`.
- No new packages this week.

## What I tried myself (weeks 5-6)

These are experiments on top of the class material. The first rows are nested CV on the development set with accuracy as the metric (this was before the ROC-AUC switch). The rows from "Week 3" on were run after the switch, so they report ROC-AUC (tuned, 15 trials, unless it says untuned). The locked test set was never used.

| What I changed | What happened |
|---|---|
| Baseline decision tree, tuned | nested accuracy 0.675 ± 0.020 |
| Random forest, tuned | 0.682 ± 0.017 |
| Logistic regression (I added a `C` search space) | 0.671 ± 0.015 |
| Different metric for tuning (balanced accuracy, F1, ROC-AUC) | the recall vs false-positive trade-off moved, the accuracy barely did. F1 picked a very shallow tree. |
| `class_weight: balanced` | recall went up to about 0.62, but so did the false positive rate for every group (roughly 0.24 up to 0.38 for the larger groups; the very small groups, n=6 and n=21, were lower and too small to trust). So it's a choice about which mistake you prefer, not a free improvement. |
| Removing `sex` / `age_cat` | no visible effect |
| Other encoders (onehot, ordinal, count) | no visible effect |
| More trials (60) and more inner folds (10) | plateaus at about 0.68 |
| Week 3: mean instead of median imputation (tree) | ROC-AUC 0.713 vs 0.710, nothing really |
| Week 3: removing the `_was_missing` flags | tree 0.711, logistic regression 0.718 (same as with them). The flags don't seem to help here |
| Week 4: standard scaler instead of robust (logistic regression) | identical, 0.718 |
| Week 4: 10 outer folds instead of 5 (logistic regression) | same mean, 0.718, but the std went up from 0.014 to 0.020 |
| Week 4: one-hot encoder instead of target (tree) | 0.718 vs 0.710. Slightly better, but inside the std so I kept target encoding |
| Week 5: 5 trials vs 40 trials (tree) | 0.712 vs 0.715, 15 gave 0.710. Trial count doesn't matter much for the tree |
| Week 6: `min_samples_leaf` in the random forest, untuned | 1: 0.684, 20: 0.727, 60: 0.726. The leaf size matters, more than anything else I changed |
| Week 6: AdaBoost with depth-3 base trees instead of stumps, untuned | 0.719 vs 0.723 for stumps, not better |
| Week 7: gradient boosting `subsample` 0.7, and `learning_rate` 0.05 with 300 trees, untuned | 0.728 and 0.726 vs 0.727 for the defaults |
| Week 7: `class_weight: balanced` in HistGradientBoosting, untuned | 0.717, same as without. Makes sense, it changes the threshold more than the ranking |

**Conclusion:** five very different models all end up at about 0.68, and more tuning doesn't move it. The limit is therefore most likely the information in the 10 features, not the model or the tuning. This matches the week 6 notebook.

## Preprocessing decisions

*(From the week 3 EDA notebook, `01_eda_introduction`. Walkthrough of the recipe: `02_preprocessing`.)*

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR, tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR, tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (it is not a model feature anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out of range or negative) | domain rule | converted to `NaN` before imputation |
| `sex`, `race`, `c_charge_degree`, `score_text` | inconsistent spelling (casing, abbreviations) | data entry | one spelling per category |
| whole rows | 72 exact duplicates, all with a repeated `id` | data entry | dropped (training data only, before the split) |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant (r = 1.00, or an exact sum that only VIF catches) | multicollinearity | dropped |
| `id`, `decile_score`, `score_text`, `race` | not model inputs | | kept aside: COMPAS's own score only for comparison, `race` only for the fairness check |

**Encoder and scaler:** target encoding (one compact column per feature, and sklearn's version cross-fits so it doesn't leak) and robust scaling (median/IQR, so the few extreme counts don't set the scale). Trees don't need scaling at all, but I keep one recipe for every model. The other options (`onehot`, `ordinal`, `count`, `none`, `standard`, `minmax`) are a one-line change in the config.

One thing I did not do: I haven't tried KNN imputation or my own engineered features (for example priors per year of adulthood) from the challenges. That is the first thing I'd try next, because if the limit is the information in the features, that's where a gain could come from.

## Model evaluation

**Locked test set + stratified 5-fold CV.** The test set (20%) is never scored in the pipeline. Everything is decided on the development set, with CV of the whole pipeline. I report mean ± std over the folds, not a single split, because one 75/25 split can easily move by several accuracy points, which is more than most of the improvements I'm trying to measure.

**Selection metric: ROC-AUC.** All eight metrics are reported, but tuning maximises one. I chose ROC-AUC before comparing models because COMPAS is a risk score that is used to *rank* people, and ROC-AUC scores the ranking without deciding on a threshold. Picking the threshold is a separate decision. The other metrics depend on the 0.5 cut-off. If the question were "never miss a reoffender" I'd use a recall-heavy F-beta instead, and the "best" model would be a different one (Sofia showed that F-beta 2 picks a shallow tree that flags a lot more people but ranks worse). Another reason: ROC-AUC uses the whole ranking, so it separates models more clearly than accuracy, where the fold-to-fold noise is bigger.

**Comparing two models.** I only call a winner when the corrected paired test on the same folds says so. With 5 folds, a gap of around one point is usually just noise.

```python
import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]
corrected_paired_test(runs.loc[0, folds], runs.loc[1, folds], test_train_ratio=1 / 4)   # 5 folds: 1 / (5 - 1)
```

**The scoreboard.** `results/runs.csv` has one row per run. Two runs are only comparable if they used the same data, cleaning and folds, so if I change any of those I delete the file and start again.

| Model | Untuned (ROC-AUC) | Tuned, nested CV (ROC-AUC) | No-model floor |
|---|---|---|---|
| Dummy | 0.500 | n/a | 0.500 |
| Logistic regression | 0.718 | 0.718 ± 0.014 | 0.500 |
| Naive Bayes | 0.685 | 0.705 ± 0.011 | 0.500 |
| Decision tree | 0.612 | 0.710 ± 0.010 | 0.500 |
| Random forest | 0.684 | 0.727 ± 0.016 | 0.500 |
| Oblique random forest | 0.686 | 0.727 ± 0.015 | 0.500 |
| AdaBoost | 0.723 | 0.727 ± 0.015 | 0.500 |
| Gradient boosting | 0.727 | 0.728 ± 0.015 | 0.500 |
| HistGradientBoosting | 0.717 | 0.727 ± 0.016 | 0.500 |

(All numbers come from one run per model with 15 Optuna trials, nested 5-fold CV, in `results/runs.csv`. The tuned column is the nested CV score, mean ± std over the 5 outer folds.)

## Hyperparameter tuning

With `tuning.enabled: true`, `python main.py` does two things:

1. **Nested CV**: for each of the 5 outer folds, a full Optuna study on the outer-training rows only (each trial is a 5-fold CV of the whole pipeline), refit the best candidate and score it on the outer fold. That's the honest estimate.
2. **Tuning on the whole development set**: the same study once on every development row. Those hyperparameters are what the final model uses.

The best trial's score is printed too but it's optimistic (maximum of noisy scores), so I report the nested one. The search spaces live in `config.yaml` under `tuning.search_spaces`, one per model, with names like `model__max_depth`. Cost is `(cv.n_splits + 1) x n_trials x tuning.n_splits` fits. For the ensembles that's minutes, so I use `n_trials: 5` while testing and 15 for the real runs.

I added a `logistic_regression` search space (just `C`, log scale 0.001 to 100) which isn't in Sofia's repo. I also wrote a cheat sheet in `config.yaml` with sensible ranges for each model's parameters so I don't have to look them up every time.

## Models

Set `model.type` in `config.yaml`. Each one has an example of its params in the comments and its own search space.

| `model.type` | What it is | What the tuner chooses |
|---|---|---|
| `dummy` | always predicts the majority class | nothing (it's the floor) |
| `logistic_regression` | linear model | `C` (my addition) |
| `naive_bayes` | Gaussian Naive Bayes, fast baseline | `var_smoothing` |
| `decision_tree` | one tree | `max_depth`, `min_samples_leaf`, `criterion` |
| `random_forest` | many trees, bootstrap + random features, averaged | `max_depth`, `min_samples_leaf`, `max_features` |
| `oblique_random_forest` | same, but splits on random combinations of features | tree depth and leaf size, `n_components` |
| `adaboost` | shallow trees in sequence, re-weighting the rows it got wrong | base-tree depth, `n_estimators`, `learning_rate` |
| `gradient_boosting` | trees in sequence, each fitted to the gradient of the loss | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample` |
| `hist_gradient_boosting` | same on binned features, faster, handles missing values | `max_iter`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` |

Forests are about independent trees averaged (they fix variance, more trees never hurt). Boosting is about trees built one after another (it fixes bias, but more rounds *can* overfit, so rounds, learning rate and depth are tuned together).

## Best model

The nested CV ROC-AUC of the tuned models falls into three groups. Random forest, oblique random forest, AdaBoost, gradient boosting and HistGradientBoosting all land at 0.727-0.728, within 0.001 of each other, far below the fold-to-fold std of about 0.015. `corrected_paired_test()` cannot separate them (gradient boosting vs random forest: p = 0.62, vs HistGradientBoosting: p = 0.46). Logistic regression (0.718), the tree (0.710) and Naive Bayes (0.705) sit one to two points lower. The test does separate logistic regression from gradient boosting (gap 0.010, p = 0.005), so with only 5 folds I read that as a small but real difference. Against the tree the gap is larger (0.018) but borderline (p = 0.056).

So the models are not all tied: the five ensembles are, and the simpler ones are slightly worse. Among the five I would pick the simplest, since I can't claim any of them is better. I'd take the random forest, which has the fewest interacting settings to tune. The number to report for it is its nested CV score, 0.727 ± 0.016, never the best-trial score and never anything from the test set. The practical gain over logistic regression is about one point of ROC-AUC. If readability mattered more than that point, logistic regression would be a defensible choice, and I'd say so openly.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run.

**When `requirements.txt` changes** (a week adds a package, e.g. week 5 added `optuna`), reinstall once, with the venv active: `pip install -r requirements.txt`. Otherwise `python main.py` fails with `ModuleNotFoundError`. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Environment Troubleshooting

Two Windows issues come up often enough to note here -- if you hit either, this saves you re-diagnosing it from scratch.

**PowerShell blocks the venv activation script, every new terminal.** The `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above only fixes it for that one terminal window -- close it and it's back. For a fix that actually sticks across sessions, run this **once**, instead:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
If it still doesn't stick (common on locked-down school/lab machines with a Group Policy that resets it on every logon), skip PowerShell entirely: use **Git Bash** (`source venv/Scripts/activate`) or **cmd.exe** (`venv\Scripts\activate.bat`) instead -- neither is affected by PowerShell's execution policy.

**Windows blocks the terminal/Python from reading or writing files in Documents (or Desktop/Pictures).** Shows up as an "Access is denied" error, or a silent failure to create/update a file, only when the project sits inside one of those folders. Two independent settings can cause this -- check both:
- **Windows Security -> Virus & threat protection -> Manage ransomware protection** -- turn off **Controlled folder access**, or add your terminal/Python/editor to its allowed-apps list.
- **Settings -> Privacy & security -> File system** -- make sure the terminal/Python has access.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the hyperparameters if `tuning.enabled` is on (week 5, minutes instead of seconds for the ensembles of week 6), and prints, in order:
- the model, the selection metric and the CV design, plus the size of the development set and of the locked test set (which is **not** used)
- **default hyperparameters**: CV score on the selection metric, the train-validation gap and the no-model floor
- with tuning on, **tuning**: the nested CV score (the honest one) and the chosen hyperparameters
- **all metrics**: mean ± std of every metric
- **scoreboard**: this run next to the earlier runs on the same data and metric, best first
- the final model (refit on all development rows) and where the full report was saved

The per-fold tables, the tuning report, the confusion matrix, the classification report and the false-positive-rate-by-race comparison with COMPAS's own score go to the results file.

All of this is also saved to a timestamped file in `results/` (e.g. `results/run_20261023_213517.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Dataset

See `data/README.md`.
