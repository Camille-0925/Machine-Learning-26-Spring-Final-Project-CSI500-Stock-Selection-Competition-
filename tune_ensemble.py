"""
tune_ensemble.py — 测试不同 XGB/LGB 比例对回测表现的影响
面板只建一次，循环测各种 ensemble 配置。
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb

import my_model as mm
from gen_submission_week2 import build_panel_5d, make_split_5d, predict, TARGET_COL
from advanced_features import build_features, get_feature_columns
from score_submission import score_window

DATA_DIR  = Path("data")
N_WINDOWS = 15
HOLD_DAYS = 5
SEEDS     = [42, 123, 7]

# ── XGB 配置（3套，第3套是新增的）────────────────────────────────────────────
XGB_PARAMS = [
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
     "subsample": 0.7, "colsample_bytree": 0.7,
     "min_child_weight": 15, "reg_alpha": 0.5, "reg_lambda": 2.0},
    {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.02,
     "subsample": 0.6, "colsample_bytree": 0.6,
     "min_child_weight": 20, "reg_alpha": 1.0, "reg_lambda": 3.0},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.025,
     "subsample": 0.75, "colsample_bytree": 0.65,
     "min_child_weight": 10, "reg_alpha": 0.3, "reg_lambda": 1.5},
]

# ── LGB 配置（2套）────────────────────────────────────────────────────────────
LGB_PARAMS = [
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
     "subsample": 0.7, "colsample_bytree": 0.7, "num_leaves": 31,
     "subsample_freq": 1, "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1},
    {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.02,
     "subsample": 0.6, "colsample_bytree": 0.6, "num_leaves": 20,
     "subsample_freq": 1, "reg_alpha": 1.0, "reg_lambda": 3.0, "verbose": -1},
]

# ── Ensemble 配置方案 ─────────────────────────────────────────────────────────
ENSEMBLE_CONFIGS = {
    "XGB_only (6模型)":      {"xgb": XGB_PARAMS[:2], "lgb": []},
    "XGB_heavy 3:1 (12模型)": {"xgb": XGB_PARAMS[:3], "lgb": LGB_PARAMS[:1]},
    "Current 1:1 (12模型) ◀": {"xgb": XGB_PARAMS[:2], "lgb": LGB_PARAMS[:2]},
    "LGB_heavy 1:3 (12模型)": {"xgb": XGB_PARAMS[:1], "lgb": LGB_PARAMS[:2] + [
        # 第3套LGB
        {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.025,
         "subsample": 0.75, "colsample_bytree": 0.65, "num_leaves": 40,
         "subsample_freq": 1, "reg_alpha": 0.3, "reg_lambda": 1.5, "verbose": -1}
    ]},
}


def train_models_custom(train_df, val_df, feature_cols, xgb_cfgs, lgb_cfgs):
    sw = mm.time_decay_weights(train_df)
    models = []
    for params in xgb_cfgs:
        for seed in SEEDS:
            m = xgb.XGBRegressor(**params, random_state=seed, tree_method="hist")
            m.fit(train_df[feature_cols], train_df[TARGET_COL],
                  sample_weight=sw,
                  eval_set=[(val_df[feature_cols], val_df[TARGET_COL])],
                  verbose=False)
            models.append(m)
    for params in lgb_cfgs:
        for seed in SEEDS:
            m = lgb.LGBMRegressor(**params, random_state=seed)
            m.fit(train_df[feature_cols], train_df[TARGET_COL],
                  sample_weight=sw,
                  eval_set=[(val_df[feature_cols], val_df[TARGET_COL])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(-1)])
            models.append(m)
    return models


def run_backtest(panel5, feature_cols, prices, index_df, xgb_cfgs, lgb_cfgs):
    trading_dates = np.sort(panel5["date"].unique())
    min_start = 5 + mm.VAL_DAYS + mm.EMBARGO_DAYS + 30
    last_idx  = len(trading_dates) - 1 - HOLD_DAYS
    pred_indices = []
    idx = last_idx
    while idx >= min_start and len(pred_indices) < N_WINDOWS:
        pred_indices.append(idx)
        idx -= HOLD_DAYS
    pred_indices = sorted(pred_indices)

    rows = []
    for pred_idx in pred_indices:
        as_of   = pd.Timestamp(trading_dates[pred_idx])
        w_start = pd.Timestamp(trading_dates[pred_idx + 1])
        w_end   = pd.Timestamp(trading_dates[min(pred_idx + HOLD_DAYS, len(trading_dates)-1)])
        try:
            train_df, val_df = make_split_5d(panel5, as_of, feature_cols)
        except RuntimeError:
            continue
        if len(train_df) < 100 or len(val_df) < 10:
            continue

        models = train_models_custom(train_df, val_df, feature_cols, xgb_cfgs, lgb_cfgs)
        val_pred = predict(models, val_df, feature_cols)
        val_ic, _ = spearmanr(val_df[TARGET_COL], val_pred)

        pred_df = panel5[panel5["date"] == as_of].dropna(subset=feature_cols).copy()
        pred_df = pred_df[(pred_df["turnover_ma20"] >= mm.MIN_TURNOVER) & (pred_df["volume"] > 0)].copy()
        if len(pred_df) < 35:
            continue

        pred_df["score"] = predict(models, pred_df, feature_cols)
        scores  = pred_df.set_index("stock_code")["score"]
        weights = mm.build_portfolio(scores, top_k=35, blend_alpha=0.0)

        realized = score_window(weights, prices, index_df, w_start, w_end)
        rows.append({"val_ic": val_ic, "excess_return": realized["excess_return"]})

    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        "win_rate":     (df["excess_return"] > 0).mean(),
        "mean_excess":  df["excess_return"].mean(),
        "near5_excess": df.tail(5)["excess_return"].mean(),
        "near5_win":    (df.tail(5)["excess_return"] > 0).mean(),
        "mean_ic":      df["val_ic"].mean(),
        "n_models":     len(xgb_cfgs) * len(SEEDS) + len(lgb_cfgs) * len(SEEDS),
    }


def main():
    prices   = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])

    print(">> 构建特征面板（只建一次）...")
    panel5 = build_panel_5d(prices)
    panel3 = build_features(prices)
    feature_cols = get_feature_columns(panel3)
    del panel3
    print(f"   特征数: {len(feature_cols)}\n")

    print("=" * 72)
    print(f"{'配置':<26}  {'胜率':>5}  {'均超额':>8}  {'近5窗超额':>10}  {'近5窗胜率':>9}  {'IC':>7}")
    print("-" * 72)

    best_near5 = -999
    best_name  = ""
    all_results = {}

    for name, cfg in ENSEMBLE_CONFIGS.items():
        print(f"  运行: {name} ...", end="", flush=True)
        r = run_backtest(panel5, feature_cols, prices, index_df,
                         cfg["xgb"], cfg["lgb"])
        all_results[name] = r
        print(f"\r  {name:<26}  {r['win_rate']:>4.0%}  {r['mean_excess']*100:>+7.3f}%  "
              f"{r['near5_excess']*100:>+9.3f}%  {r['near5_win']:>8.0%}  {r['mean_ic']:>+6.4f}")
        if r["near5_excess"] > best_near5:
            best_near5 = r["near5_excess"]
            best_name  = name

    print("-" * 72)
    print(f"\n  最优配置（近5窗超额最高）: {best_name}  ({best_near5*100:+.3f}%)")
    print("\n>> 完成。")


if __name__ == "__main__":
    main()
