"""
backtest_week2.py — 滚动回测（5日持仓期，5日目标，Week2 模型参数）

每个窗口输出：train / val / test 时间段 + val_IC + 超额收益
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from gen_submission_week2 import (
    build_panel_5d, make_split_5d, train_models, predict, TARGET_COL,
)
from advanced_features import build_features, get_feature_columns
from my_model import build_portfolio, MIN_TURNOVER, VAL_DAYS, EMBARGO_DAYS
from score_submission import score_window
from scipy.stats import spearmanr

DATA_DIR  = Path("data")
N_WINDOWS = 15
HOLD_DAYS = 5
TOP_K     = 35


def main():
    prices   = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])

    print(">> 构建特征面板（5日目标）...")
    panel5 = build_panel_5d(prices)

    # feature_cols 从 panel3 取，避免 target_5d 混入
    panel3 = build_features(prices)
    feature_cols = get_feature_columns(panel3)
    del panel3

    trading_dates = np.sort(panel5["date"].unique())
    min_start = 5 + VAL_DAYS + EMBARGO_DAYS + 30
    last_idx  = len(trading_dates) - 1 - HOLD_DAYS
    pred_indices = []
    idx = last_idx
    while idx >= min_start and len(pred_indices) < N_WINDOWS:
        pred_indices.append(idx)
        idx -= HOLD_DAYS
    pred_indices = sorted(pred_indices)

    rows = []
    print(f"\n{'as_of':>12}  {'train':>22}  {'val':>22}  {'test':>22}  {'val_IC':>7}  {'超额收益':>9}")
    print("  " + "-" * 110)

    for pred_idx in pred_indices:
        as_of   = pd.Timestamp(trading_dates[pred_idx])
        w_start = pd.Timestamp(trading_dates[pred_idx + 1])
        w_end   = pd.Timestamp(trading_dates[min(pred_idx + HOLD_DAYS, len(trading_dates) - 1)])

        try:
            train_df, val_df = make_split_5d(panel5, as_of, feature_cols)
        except RuntimeError:
            continue

        if len(train_df) < 100 or len(val_df) < 10:
            continue

        models = train_models(train_df, val_df, feature_cols)

        val_pred = predict(models, val_df, feature_cols)
        val_ic, _ = spearmanr(val_df[TARGET_COL], val_pred)

        pred_df = panel5[panel5["date"] == as_of].dropna(subset=feature_cols).copy()
        pred_df = pred_df[(pred_df["turnover_ma20"] >= MIN_TURNOVER) & (pred_df["volume"] > 0)].copy()
        if len(pred_df) < TOP_K:
            continue
        pred_df["score"] = predict(models, pred_df, feature_cols)
        scores  = pred_df.set_index("stock_code")["score"]
        weights = build_portfolio(scores, top_k=TOP_K, blend_alpha=0.0)

        realized = score_window(weights, prices, index_df, w_start, w_end)
        excess   = realized["excess_return"]

        train_str = f"{train_df['date'].min().date()} ~ {train_df['date'].max().date()}"
        val_str   = f"{val_df['date'].min().date()} ~ {val_df['date'].max().date()}"
        test_str  = f"{w_start.date()} ~ {w_end.date()}"

        flag = "✅" if excess > 0 else "❌"
        print(f"  {as_of.date()}  {train_str:>22}  {val_str:>22}  {test_str:>22}  {val_ic:>+7.4f}  {excess*100:>+8.3f}%  {flag}")

        rows.append({
            "as_of": as_of.date(),
            "train_start": train_df["date"].min().date(),
            "train_end":   train_df["date"].max().date(),
            "val_start":   val_df["date"].min().date(),
            "val_end":     val_df["date"].max().date(),
            "test_start":  w_start.date(),
            "test_end":    w_end.date(),
            "val_ic":      val_ic,
            "excess_return": excess,
        })

    df = pd.DataFrame(rows)
    print("  " + "-" * 110)
    print(f"  {'均值':>12}  {' '*66}  {df['val_ic'].mean():>+7.4f}  {df['excess_return'].mean()*100:>+8.3f}%")
    print(f"  {'胜率':>12}  {' '*66}  {(df['val_ic']>0).mean():>7.0%}  {(df['excess_return']>0).mean():>8.0%}")
    print(f"\n  近5窗均超额: {df.tail(5)['excess_return'].mean()*100:+.3f}%  近5窗胜率: {(df.tail(5)['excess_return']>0).mean():.0%}")

    df.to_csv("backtest_week2_results.csv", index=False)
    print(">> 结果保存至 backtest_week2_results.csv")


if __name__ == "__main__":
    main()
