# Machine-Learning-26-Spring-Final-Project-CSI500-Stock-Selection-Competition-[README.md](https://github.com/user-attachments/files/27714330/README.md)
# Week 2 Submission — Minjun Cai

## Model Overview

A 12-model ensemble (2 XGBoost + 2 LightGBM configurations × 3 random seeds) trained on 43 hand-crafted price/volume features to predict 5-day forward returns across CSI500 constituents. The portfolio selects the top-35 stocks by predicted score and assigns equal weights.

**Key parameters:**
- Target: 5-day forward return (`target_5d`)
- Features: 43 (technical + market-relative + theme flags)
- Portfolio: top-35 stocks, equal weight (~2.86% each)
- Training window: most recent **120 trading days** (~6 months)
- Time-decay sample weights: half-life 90 days
- Train/val split: 5-day embargo between train and validation sets

---

## Environment

**Python:** 3.12.5

Install all dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins exact versions:

```
xgboost==2.0.3
lightgbm==4.6.0
pandas==3.0.2
numpy==1.26.4
scipy==1.17.1
pyarrow>=14.0
```

---

## Data

Place the provided data files in a `data/` directory **one level above** this folder (i.e., at the project root):

```
project_root/
├── data/
│   ├── prices.parquet
│   ├── fundamentals.parquet
│   ├── index.parquet
│   └── constituents.csv
└── submission2_final/
    ├── gen_submission_week2.py   ← run this to reproduce the submission
    ├── backtest_week2.py
    ├── advanced_features.py
    ├── my_model.py
    ├── score_submission.py
    ├── requirements.txt
    └── submission_week2_final.csv
```

---

## Reproducing the Submitted Portfolio

Run from the **project root** (one level above this folder):

```bash
python submission2_final/gen_submission_week2.py
```

This will:
1. Build the 5-day target feature panel (43 features)
2. Take the most recent 120 trading days as the training window
3. Split into train / validation sets with a 5-day embargo
4. Train 12 models (6 XGBoost + 6 LightGBM)
5. Score all eligible CSI500 stocks as of the latest date in `prices.parquet`
6. Output `submission_week2_final.csv` in this folder

**Expected output:** 35 stocks, equal weight 0.028571, weight sum = 1.0

---

## Reproducibility Notes

| Setting | Value |
|---------|-------|
| Random seeds | 42, 123, 7 (all XGBoost and LightGBM models) |
| `MAX_TRAIN_DAYS` | **120** trading days (non-default — required for exact reproduction) |
| XGBoost `tree_method` | `"hist"` (deterministic) |
| LightGBM `n_jobs` | `1` (single-threaded, eliminates non-determinism) |
| Feature engineering | Fully deterministic rolling/ewm calculations |

---

## Self-Test: Rolling Backtest Results (15 windows)

Run the self-test with:

```bash
python submission2_final/backtest_week2.py
```

Each of the 15 windows has a distinct train / validation / test split with a 5-day embargo enforced between train and validation sets. Results:

| Metric | 120-day window (this model) | Full-history baseline |
|--------|----------------------------|----------------------|
| Overall win rate | 67% | 80% |
| Mean excess return | +1.57% | +1.28% |
| **Recent 5-window excess** | **+2.42%** | +2.21% |
| **Recent 5-window win rate** | **100%** | 100% |
| Mean val IC | +0.044 | +0.037 |

The 120-day window trades overall win rate for better recent performance (+2.42% vs +2.21%), reflecting adaptation to the current market regime. The evaluation window (May 11–15) falls within the "recent 5 windows" horizon, making recent performance the more relevant metric.

---

## File Descriptions

| File | Description |
|------|-------------|
| `gen_submission_week2.py` | **Main entry point.** Builds features, trains models, outputs portfolio CSV |
| `advanced_features.py` | Feature engineering: 43 price/volume/market-relative features |
| `my_model.py` | Portfolio construction, time-decay weights, train/val split utilities |
| `backtest_week2.py` | **Self-test.** 15-window rolling backtest with train/val/test splits |
| `score_submission.py` | Single-window realized excess return calculator |
| `tune_params.py` | Grid search over TOP_K and HALF_LIFE hyperparameters |
| `tune_ensemble.py` | Grid search over XGBoost/LightGBM ensemble ratios |
| `requirements.txt` | Pinned dependency versions |
| `submission_week2_final.csv` | **Submitted portfolio** (35 stocks, equal weight) |
