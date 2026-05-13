"""
tune_params.py — 对 TOP_K 和 HALF_LIFE_DAYS 做网格搜索，面板只建一次。
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import my_model as mm
from gen_submission_week2 import build_panel_5d, make_split_5d, train_models, predict, TARGET_COL
from advanced_features import build_features, get_feature_columns
from score_submission import score_window

DATA_DIR  = Path("data")
N_WINDOWS = 15
HOLD_DAYS = 5

# ── 参数网格 ──────────────────────────────────────────────────────────────────
TOP_K_VALUES      = [30, 35, 40]
HALF_LIFE_VALUES  = [60, 90, 120]

def run_backtest(panel5, feature_cols, prices, index_df, top_k, half_life):
    """运行一次完整的15窗口回测，返回 summary dict。"""
    mm.HALF_LIFE_DAYS = half_life          # 动态覆盖全局变量

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
        as_of  = pd.Timestamp(trading_dates[pred_idx])
        w_start = pd.Timestamp(trading_dates[pred_idx + 1])
        w_end   = pd.Timestamp(trading_dates[min(pred_idx + HOLD_DAYS, len(trading_dates)-1)])
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
        pred_df = pred_df[(pred_df["turnover_ma20"] >= mm.MIN_TURNOVER) & (pred_df["volume"] > 0)].copy()
        if len(pred_df) < top_k:
            continue

        pred_df["score"] = predict(models, pred_df, feature_cols)
        scores  = pred_df.set_index("stock_code")["score"]
        weights = mm.build_portfolio(scores, top_k=top_k, blend_alpha=0.0)

        realized = score_window(weights, prices, index_df, w_start, w_end)
        rows.append({
            "val_ic":        val_ic,
            "excess_return": realized["excess_return"],
        })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        "n":          len(df),
        "mean_ic":    df["val_ic"].mean(),
        "win_rate":   (df["excess_return"] > 0).mean(),
        "mean_excess":df["excess_return"].mean(),
        "near5_excess": df.tail(5)["excess_return"].mean(),
        "near5_win":    (df.tail(5)["excess_return"] > 0).mean(),
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

    # ── 实验1：固定 HALF_LIFE=90，遍历 TOP_K ──────────────────────────────────
    print("=" * 70)
    print("实验1：TOP_K 对比（HALF_LIFE=90 固定）")
    print(f"{'TOP_K':>6}  {'胜率':>6}  {'均超额':>8}  {'近5窗超额':>10}  {'近5窗胜率':>9}  {'val_IC':>7}")
    print("-" * 70)
    results_topk = {}
    for k in TOP_K_VALUES:
        r = run_backtest(panel5, feature_cols, prices, index_df, top_k=k, half_life=90)
        results_topk[k] = r
        marker = " ◀ 当前" if k == 35 else ""
        print(f"  k={k:<4}  {r['win_rate']:>5.0%}  {r['mean_excess']*100:>+7.3f}%  "
              f"{r['near5_excess']*100:>+9.3f}%  {r['near5_win']:>8.0%}  {r['mean_ic']:>+7.4f}{marker}")

    # ── 实验2：固定 TOP_K=35，遍历 HALF_LIFE ─────────────────────────────────
    print()
    print("=" * 70)
    print("实验2：HALF_LIFE 对比（TOP_K=35 固定）")
    print(f"{'HL':>6}  {'胜率':>6}  {'均超额':>8}  {'近5窗超额':>10}  {'近5窗胜率':>9}  {'val_IC':>7}")
    print("-" * 70)
    for hl in HALF_LIFE_VALUES:
        r = run_backtest(panel5, feature_cols, prices, index_df, top_k=35, half_life=hl)
        marker = " ◀ 当前" if hl == 90 else ""
        print(f" hl={hl:<4}  {r['win_rate']:>5.0%}  {r['mean_excess']*100:>+7.3f}%  "
              f"{r['near5_excess']*100:>+9.3f}%  {r['near5_win']:>8.0%}  {r['mean_ic']:>+7.4f}{marker}")

    print()
    print(">> 完成。")


if __name__ == "__main__":
    main()
