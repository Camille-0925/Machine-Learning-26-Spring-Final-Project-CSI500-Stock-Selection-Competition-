"""
my_model.py — CSI500 股票选择：多模型集成 GBDT

策略设计：
- 特征：advanced_features.py 的 39 个多时间尺度信号
  （超短期动量、波动率 regime、流动性、K-bar 形态）
- 模型：2 XGBoost + 2 LightGBM 配置 × 3 seeds = 12 模型集成
- 预测目标：3 日远期收益（与评估窗口匹配）
- 训练权重：时间衰减（半衰期 60 天，近期数据权重更高）
- 组合构建：top-40 股票，混合排名/等权加权，10% 上限

用法：
  python my_model.py                        # 全流程：回测 + 生成提交文件
  python my_model.py --skip-backtest        # 跳过回测，直接生成提交文件
  python my_model.py --out submissions/week1.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import spearmanr

from advanced_features import (
    build_features,
    get_feature_columns,
    training_frame,
    prediction_frame,
    FORWARD_HORIZON,
    TARGET_COLUMN,
)
from score_submission import score_window

# ── 全局配置 ─────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent / "data"
GLOBAL_SEED  = 2026
VAL_DAYS     = 5         # 实验最优：vd=5 累计超额+13.68% IR+3.76（vd=10为+8.27%/+2.63）
EMBARGO_DAYS = max(5, FORWARD_HORIZON + 2)
MIN_STOCKS   = 30
MAX_WEIGHT   = 0.10
DEFAULT_TOP_K = 30        # top_k实验：k=30胜率70%、累计超额+8.27%、IR+2.63（k=40为60%/+6.43%/+2.32）
HALF_LIFE_DAYS = 90       # 完整回测最优：hl=90 累计超额+6.43%，年化IR+2.32
MIN_TURNOVER   = 0.0005   # 流动性过滤下限

ENSEMBLE_SEEDS = [2026, 42, 99]

XGB_CONFIGS = [
    {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.04,
     "subsample": 0.8, "colsample_bytree": 0.7,
     "min_child_weight": 15, "reg_alpha": 0.5, "reg_lambda": 2.0},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03,
     "subsample": 0.7, "colsample_bytree": 0.8,
     "min_child_weight": 20, "reg_alpha": 1.0, "reg_lambda": 3.0},
]
LGB_CONFIGS = [
    {"n_estimators": 400, "num_leaves": 31, "max_depth": 4,
     "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7,
     "subsample_freq": 1, "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1},
    {"n_estimators": 500, "num_leaves": 20, "max_depth": 3,
     "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.8,
     "subsample_freq": 1, "reg_alpha": 1.0, "reg_lambda": 3.0, "verbose": -1},
]


def _set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def add_index_features(panel: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """
    在截面标准化之后把 CSI500 大盘 regime 信号 merge 进 panel。
    因为这些特征每天对所有股票相同，截面 z-score 会把它们变成0，
    所以必须在 build_features（内含标准化）之后添加。
    """
    idx = index_df.sort_values("date").copy()
    c = idx["close"]
    ret = c.pct_change()

    idx["mkt_ret_5d"]     = c.pct_change(5)
    idx["mkt_ret_20d"]    = c.pct_change(20)
    idx["mkt_vol5"]       = ret.rolling(5).std()
    idx["mkt_vol20"]      = ret.rolling(20).std()
    idx["mkt_vol_ratio"]  = idx["mkt_vol5"] / (idx["mkt_vol20"] + 1e-8)
    ma60 = c.rolling(60).mean()
    idx["mkt_above_ma60"] = (c > ma60).astype(float)
    delta = ret
    gain  = delta.clip(lower=0).rolling(10).mean()
    loss  = (-delta.clip(upper=0)).rolling(10).mean()
    idx["mkt_rsi_10"]     = 100 - 100 / (1 + gain / (loss + 1e-8))

    mkt_cols = ["date", "mkt_ret_5d", "mkt_ret_20d", "mkt_vol_ratio",
                "mkt_above_ma60", "mkt_rsi_10"]
    panel = panel.merge(idx[mkt_cols], on="date", how="left")
    return panel


# ── 评估函数 ──────────────────────────────────────────────────────────────────
def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """每日截面 Spearman 相关，跨日平均。"""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def compute_metrics(results: pd.DataFrame) -> dict:
    """从回测结果计算汇总指标。"""
    ex = results["excess_return"]
    ic = results["validation_ic"]
    ir = ex.mean() / (ex.std() + 1e-8) * np.sqrt(252 / 3)
    ic_ir = ic.mean() / (ic.std() + 1e-8)

    # 最大回撤（基于组合累计净值）
    cum = (1 + results["portfolio_return"]).cumprod()
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    return {
        "windows":       len(results),
        "mean_IC":       ic.mean(),
        "IC_IR":         ic_ir,
        "hit_rate":      (ic > 0).mean(),
        "mean_excess":   ex.mean(),
        "excess_std":    ex.std(),
        "ann_IR":        ir,
        "win_rate":      (ex > 0).mean(),
        "max_drawdown":  max_dd,
        "cum_excess":    (1 + ex).prod() - 1,
    }


# ── 训练工具 ──────────────────────────────────────────────────────────────────
def time_decay_weights(train_df: pd.DataFrame) -> np.ndarray:
    """近期样本权重更大，半衰期 HALF_LIFE_DAYS 天。"""
    if HALF_LIFE_DAYS <= 0:
        return np.ones(len(train_df), dtype=float)
    dates = pd.to_datetime(train_df["date"])
    age = (dates.max() - dates).dt.days.astype(float)
    w = np.power(0.5, age / HALF_LIFE_DAYS)
    return (w / w.mean()).astype(float)


def make_split(panel: pd.DataFrame, as_of: pd.Timestamp):
    """时间序列划分：[训练] [embargo] [验证]，保证无泄露。
    返回 train_df, val_df, train_end, val_start（用于报告日期范围）。
    """
    trading_dates = np.sort(panel["date"].unique())
    idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("历史数据不足，无法划分训练/验证集。")

    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_start = pd.Timestamp(all_dates[0])
    train_df = pool[pool["date"] <= train_end].copy()
    val_df   = pool[pool["date"] >= val_start].copy()
    return train_df, val_df, train_start, train_end, val_start, train_cutoff


def train_ensemble(train_df: pd.DataFrame, val_df: pd.DataFrame,
                   feature_cols: list[str]) -> list:
    """训练 12 个模型（2 XGB + 2 LGB）× 3 seeds。"""
    models = []
    sw = time_decay_weights(train_df)
    for seed in ENSEMBLE_SEEDS:
        for cfg in XGB_CONFIGS:
            m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1,
                                  early_stopping_rounds=40,
                                  random_state=seed, **cfg)
            m.fit(train_df[feature_cols], train_df[TARGET_COLUMN],
                  sample_weight=sw,
                  eval_set=[(val_df[feature_cols], val_df[TARGET_COLUMN])],
                  verbose=False)
            models.append(m)
        for cfg in LGB_CONFIGS:
            m = lgb.LGBMRegressor(n_jobs=-1, random_state=seed, **cfg)
            m.fit(train_df[feature_cols], train_df[TARGET_COLUMN],
                  sample_weight=sw,
                  eval_set=[(val_df[feature_cols], val_df[TARGET_COLUMN])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
            models.append(m)
    return models


def ensemble_predict(models: list, df: pd.DataFrame,
                     feature_cols: list[str]) -> np.ndarray:
    return np.mean([m.predict(df[feature_cols]) for m in models], axis=0)


# ── 组合构建 ──────────────────────────────────────────────────────────────────
def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K,
                    blend_alpha: float = 0.6) -> pd.Series:
    """
    top_k 股票，blend_alpha * rank权重 + (1-blend_alpha) * 等权，10% 上限。
    blend_alpha=0.6：比纯等权更集中，比纯排名权重更分散。
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k 必须 >= {MIN_STOCKS}")
    scores = scores.dropna()
    if len(scores) < top_k:
        raise ValueError(f"有效股票数 {len(scores)} < top_k={top_k}")

    chosen = scores.sort_values(ascending=False).head(top_k)
    rank_w = np.arange(top_k, 0, -1, dtype=float)
    rank_w /= rank_w.sum()
    eq_w = np.full(top_k, 1.0 / top_k)
    raw = blend_alpha * rank_w + (1 - blend_alpha) * eq_w
    w = pd.Series(raw / raw.sum(), index=chosen.index)

    for _ in range(top_k + 5):
        over = w > MAX_WEIGHT + 1e-12
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()

    w = w.clip(0, MAX_WEIGHT)
    w /= w.sum()
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w <= MAX_WEIGHT + 1e-9).all()
    assert (w > 0).sum() >= MIN_STOCKS
    return w


# ── 单窗口预测 ────────────────────────────────────────────────────────────────
def generate_submission(panel: pd.DataFrame, as_of: pd.Timestamp,
                        top_k: int, feature_cols: list[str],
                        verbose: bool = False):
    train_df, val_df, train_start, train_end, val_start, val_end = make_split(panel, as_of)
    models = train_ensemble(train_df, val_df, feature_cols)

    val_pred = ensemble_predict(models, val_df, feature_cols)
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred,
                 val_df["date"].to_numpy())

    pred_df = prediction_frame(panel, as_of=as_of).copy()
    pred_df = pred_df[
        (pred_df["turnover_ma20"] >= MIN_TURNOVER) & (pred_df["volume"] > 0)
    ].copy()

    if verbose:
        print(f"  as_of={as_of.date()}  universe={len(pred_df)}  IC={ic:+.4f}")

    if len(pred_df) < top_k:
        raise RuntimeError(f"流动性过滤后仅 {len(pred_df)} 只，不足 top_k={top_k}。")

    pred_df["score"] = ensemble_predict(models, pred_df, feature_cols)
    weights = build_portfolio(
        pred_df.set_index("stock_code")["score"], top_k=top_k
    )
    sub = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    split_info = {
        "train_start": train_start, "train_end": train_end,
        "val_start":   val_start,   "val_end":   val_end,
    }
    return sub, ic, split_info


# ── 滚动回测 ──────────────────────────────────────────────────────────────────
def run_backtest(panel: pd.DataFrame, prices: pd.DataFrame,
                 index_df: pd.DataFrame, top_k: int,
                 feature_cols: list[str], n_windows: int = 20,
                 hold_days: int = 3) -> pd.DataFrame:
    """滚动 3 日持有期回测，评估模型的超额收益和 IC。"""
    trading_dates = np.sort(panel["date"].unique())
    min_start = FORWARD_HORIZON + VAL_DAYS + EMBARGO_DAYS + 30
    # 从最后一个完整3日窗口倒推，确保最后一个test_end = 数据最后一天
    last_pred_idx = len(trading_dates) - 1 - hold_days
    pred_indices = []
    idx = last_pred_idx
    while idx >= min_start and len(pred_indices) < n_windows:
        pred_indices.append(idx)
        idx -= hold_days
    pred_indices = sorted(pred_indices)
    rows = []

    for pred_idx in pred_indices:
        as_of = pd.Timestamp(trading_dates[pred_idx])
        try:
            sub, ic, split = generate_submission(panel, as_of, top_k, feature_cols,
                                                  verbose=False)
        except RuntimeError as e:
            print(f"  跳过 {as_of.date()}: {e}")
            continue

        weights = sub.set_index("stock_code")["weight"]
        w_start = pd.Timestamp(trading_dates[pred_idx + 1])
        w_end   = pd.Timestamp(
            trading_dates[min(pred_idx + hold_days, len(trading_dates) - 1)]
        )
        realized = score_window(weights, prices, index_df, w_start, w_end)
        rows.append({
            "train_start":      split["train_start"].strftime("%Y-%m-%d"),
            "train_end":        split["train_end"].strftime("%Y-%m-%d"),
            "val_start":        split["val_start"].strftime("%Y-%m-%d"),
            "val_end":          split["val_end"].strftime("%Y-%m-%d"),
            "test_start":       realized["start"],
            "test_end":         realized["end"],
            "test_days":        realized["trading_days"],
            "validation_ic":    ic,
            "portfolio_return": realized["portfolio_return"],
            "benchmark_return": realized["benchmark_return"],
            "excess_return":    realized["excess_return"],
        })

    return pd.DataFrame(rows)


# ── 打印指标 ──────────────────────────────────────────────────────────────────
def print_metrics(m: dict) -> None:
    print("\n" + "=" * 45)
    print("  回测指标汇总")
    print("=" * 45)
    print(f"  回测窗口数         : {m['windows']}")
    print(f"  平均 Rank IC       : {m['mean_IC']:+.4f}")
    print(f"  IC IR (IC均值/波动): {m['IC_IR']:+.3f}")
    print(f"  IC 正值率 (Hit)    : {m['hit_rate']:.1%}")
    print(f"  平均 3日超额收益   : {m['mean_excess']:+.4%}")
    print(f"  超额收益标准差     : {m['excess_std']:.4%}")
    print(f"  年化信息比率 IR    : {m['ann_IR']:+.3f}")
    print(f"  胜率 (超额>0)      : {m['win_rate']:.1%}")
    print(f"  最大回撤           : {m['max_drawdown']:+.4%}")
    print(f"  累计超额收益       : {m['cum_excess']:+.4%}")
    print("=" * 45 + "\n")


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    _set_seed()

    p = argparse.ArgumentParser()
    p.add_argument("--prices",        default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index",         default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--top-k",         type=int, default=DEFAULT_TOP_K)
    p.add_argument("--windows",       type=int, default=20,
                   help="回测滚动窗口数（越多越准确，越慢）")
    p.add_argument("--skip-backtest", action="store_true",
                   help="跳过回测，直接生成提交文件")
    p.add_argument("--out",           default="submissions/my_submission.csv")
    args = p.parse_args()

    # 加载数据
    print(">> 加载价格数据...")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    print(f"   {len(prices):,} 行，{prices['stock_code'].nunique()} 只股票，"
          f"{prices['date'].min().date()} ~ {prices['date'].max().date()}")

    # 加载基本面（可选）
    fund_path = DATA_DIR / "fundamentals.parquet"
    fundamentals = None
    if fund_path.exists():
        print(">> 加载基本面数据（PE/PB）...")
        fundamentals = pd.read_parquet(fund_path)
        fundamentals["date"] = pd.to_datetime(fundamentals["date"])
        print(f"   {len(fundamentals):,} 行")

    # 构建特征面板
    print(">> 构建特征面板（advanced_features，39个因子）...")
    panel = build_features(prices, fundamentals=fundamentals)
    feature_cols = get_feature_columns(panel)
    print(f"   面板大小: {panel.shape}，特征数: {len(feature_cols)}")

    # ── 滚动回测（自测）──────────────────────────────────────────────────────
    if not args.skip_backtest:
        print(f"\n>> 滚动回测（{args.windows} 个 3日窗口）...")
        results = run_backtest(
            panel, prices, index_df,
            top_k=args.top_k, feature_cols=feature_cols,
            n_windows=args.windows, hold_days=3,
        )
        if results.empty:
            print("  ⚠️ 回测未产生任何窗口，请检查数据或减少 --windows")
        else:
            results.to_csv("backtest_results.csv", index=False)
            print("\n" + results.to_string(index=False,
                                           float_format=lambda x: f"{x:.4f}"))
            m = compute_metrics(results)
            print_metrics(m)
            print(">> 回测结果已保存至 backtest_results.csv")

    # ── 生成最终提交文件 ──────────────────────────────────────────────────────
    print("\n>> 训练最终模型并生成提交文件...")
    latest = pd.Timestamp(panel["date"].max())
    print(f"   预测日期: {latest.date()}")

    sub, final_ic, _ = generate_submission(
        panel, as_of=latest, top_k=args.top_k,
        feature_cols=feature_cols, verbose=True,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

    print(f"\n>> 提交文件已写入: {out_path}")
    print(f"   股票数: {len(sub)}  val_IC: {final_ic:+.4f}")
    print(f"   权重: min={sub['weight'].min():.4f}  "
          f"max={sub['weight'].max():.4f}  sum={sub['weight'].sum():.6f}")
    print("\n>> 请运行以下命令验证格式：")
    print(f"   python validate_submission.py {out_path}")


if __name__ == "__main__":
    main()
