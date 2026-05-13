"""
gen_submission_week2.py — Submission 2 Final

核心改动：
- FORWARD_HORIZON=5（对齐评估窗口May 11-15）
- top_k=35, blend_alpha=0.0（等权，回测验证最优）
- MAX_TRAIN_DAYS=120（只用最近120个交易日训练，回测验证近期表现更优）
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb

import advanced_features as af
from my_model import (
    build_portfolio, time_decay_weights,
    MIN_TURNOVER, VAL_DAYS, EMBARGO_DAYS,
)
from advanced_features import get_feature_columns, prediction_frame

DATA_DIR       = Path("data")
TOP_K          = 35
ALPHA          = 0.0
HOLD_DAYS      = 5
MAX_TRAIN_DAYS = 120   # 只用最近120个交易日训练（约6个月）

XGB_PARAMS = [
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
     "subsample": 0.7, "colsample_bytree": 0.7,
     "min_child_weight": 15, "reg_alpha": 0.5, "reg_lambda": 2.0},
    {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.02,
     "subsample": 0.6, "colsample_bytree": 0.6,
     "min_child_weight": 20, "reg_alpha": 1.0, "reg_lambda": 3.0},
]
LGB_PARAMS = [
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
     "subsample": 0.7, "colsample_bytree": 0.7, "num_leaves": 31,
     "subsample_freq": 1, "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1},
    {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.02,
     "subsample": 0.6, "colsample_bytree": 0.6, "num_leaves": 20,
     "subsample_freq": 1, "reg_alpha": 1.0, "reg_lambda": 3.0, "verbose": -1},
]
SEEDS      = [42, 123, 7]
TARGET_COL = "target_5d"


def make_split_5d(panel5, as_of, feature_cols, target_col=TARGET_COL,
                  fh=5, val_days=VAL_DAYS, embargo_days=EMBARGO_DAYS,
                  max_train_days=MAX_TRAIN_DAYS):
    """FORWARD_HORIZON=5 的无泄漏 split，支持训练窗口长度上限。"""
    trading_dates = np.sort(panel5["date"].unique())
    idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx  = max(0, idx - fh)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    # 训练起始日：最多往前看 max_train_days 个交易日
    start_idx   = max(0, cutoff_idx - max_train_days)
    train_start = pd.Timestamp(trading_dates[start_idx])

    pool = panel5[
        (panel5["date"] >= train_start) &
        (panel5["date"] <= train_cutoff)
    ].dropna(subset=feature_cols + [target_col]).copy()

    all_dates = np.sort(pool["date"].unique())
    if len(all_dates) < val_days + embargo_days + 20:
        raise RuntimeError("历史数据不足")

    val_start = pd.Timestamp(all_dates[-val_days])
    train_end = pd.Timestamp(all_dates[-(val_days + embargo_days + 1)])
    train_df  = pool[pool["date"] <= train_end].copy()
    val_df    = pool[pool["date"] >= val_start].copy()
    return train_df, val_df


def build_panel_5d(prices):
    orig_fh = af.FORWARD_HORIZON
    orig_tc = af.TARGET_COLUMN
    af.FORWARD_HORIZON = 5
    af.TARGET_COLUMN   = TARGET_COL
    panel = af.build_features(prices)
    af.FORWARD_HORIZON = orig_fh
    af.TARGET_COLUMN   = orig_tc
    return panel


def train_models(train_df, val_df, feature_cols):
    sw = time_decay_weights(train_df)
    models = []
    for params in XGB_PARAMS:
        for seed in SEEDS:
            m = xgb.XGBRegressor(**params, random_state=seed, tree_method="hist")
            m.fit(train_df[feature_cols], train_df[TARGET_COL],
                  sample_weight=sw,
                  eval_set=[(val_df[feature_cols], val_df[TARGET_COL])],
                  verbose=False)
            models.append(m)
    for params in LGB_PARAMS:
        for seed in SEEDS:
            m = lgb.LGBMRegressor(**params, random_state=seed, n_jobs=1)
            m.fit(train_df[feature_cols], train_df[TARGET_COL],
                  sample_weight=sw,
                  eval_set=[(val_df[feature_cols], val_df[TARGET_COL])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(-1)])
            models.append(m)
    return models


def predict(models, df, feature_cols):
    return np.mean([m.predict(df[feature_cols]) for m in models], axis=0)


def main():
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    print(">> 构建5日目标特征面板...")
    panel5 = build_panel_5d(prices)

    from advanced_features import build_features as _bf
    _panel3_tmp = _bf(prices)
    feature_cols = get_feature_columns(_panel3_tmp)
    del _panel3_tmp

    as_of = pd.Timestamp(panel5["date"].max())
    print(f">> 预测日期: {as_of.date()}")
    print(f"   特征数: {len(feature_cols)}")
    print(f"   训练窗口: 最近 {MAX_TRAIN_DAYS} 个交易日")

    train_df, val_df = make_split_5d(panel5, as_of, feature_cols)

    print(f"   训练集: {train_df['date'].min().date()} → {train_df['date'].max().date()}  ({len(train_df)}行)")
    print(f"   验证集: {val_df['date'].min().date()} → {val_df['date'].max().date()}  ({len(val_df)}行)")

    print(">> 训练模型（12个，5日目标）...")
    models = train_models(train_df, val_df, feature_cols)

    from scipy.stats import spearmanr
    val_pred = predict(models, val_df, feature_cols)
    ic, _ = spearmanr(val_df[TARGET_COL], val_pred)
    print(f"   val_IC (5d): {ic:+.4f}")

    from advanced_features import build_features as _bf2
    _p3 = _bf2(prices)
    pred_df = prediction_frame(_p3, as_of=as_of).copy()
    del _p3
    pred_df = pred_df[(pred_df["turnover_ma20"] >= MIN_TURNOVER) & (pred_df["volume"] > 0)].copy()
    pred_df["score"] = predict(models, pred_df, feature_cols)
    scores  = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=TOP_K, blend_alpha=ALPHA)

    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})

    const = pd.read_csv(DATA_DIR / "constituents.csv")
    const["stock_code"] = const["stock_code"].astype(str).str.zfill(6)
    result = out.merge(const[["stock_code", "stock_name"]], on="stock_code", how="left")
    print(f"\n  入选{TOP_K}只股票:")
    for _, row in result.iterrows():
        print(f"    {row['stock_code']}  {row['stock_name']}  {row['weight']:.4f}")

    out_path = Path(__file__).parent / "submission_week2_final.csv"
    out[["stock_code", "weight"]].to_csv(out_path, index=False)
    print(f"\n>> 保存至: {out_path}")
    print(f"   股票数: {len(out)}  权重和: {out['weight'].sum():.6f}  最大权重: {out['weight'].max():.4f}")


if __name__ == "__main__":
    main()
