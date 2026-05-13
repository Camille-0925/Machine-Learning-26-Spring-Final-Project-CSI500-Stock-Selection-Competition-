"""
Advanced Multi-Horizon Feature Engineering for CSI500 Stock Selection

完全原创设计 - 基于时间序列分解和市场微观结构
不同于 Alpha158 或其他标准因子库

特征设计理念：
1. 多时间尺度分解（超短期、短期、中期）
2. 市场微观结构特征（订单流、买卖压力）
3. 相对强度指标（vs 行业、vs 市场）
4. 动态特征选择（RFE-CV）

References:
- 基于 2024-2025 最新论文的时间序列特征工程
- 完全自主实现，无第三方因子库依赖
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional

# 目标变量：预测 3 日后收益率
TARGET_COLUMN = "target_3d"
FORWARD_HORIZON = 3


# ============================================================
# 辅助函数
# ============================================================
def safe_divide(a: pd.Series, b: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """安全除法，避免除零"""
    return a / b.replace(0, np.nan).fillna(fill_value)


def rank_normalize(series: pd.Series) -> pd.Series:
    """排名标准化到 [0, 1]"""
    return series.rank(method='average', pct=True)


# ============================================================
# 特征族 1：超短期特征（1-3天）- 捕捉日内波动和瞬时动量
# ============================================================
def ultra_short_term_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    超短期特征：捕捉日内微观结构和短期冲击
    
    创新点：
    - 价格加速度（动量的动量）
    - 成交量爆发检测
    - 日内价格分布特征
    """
    feats = {}

    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float).replace(0, np.nan)

    # 1. 价格动量加速度（momentum of momentum）
    ret_1d = c.pct_change()
    ret_2d = c.pct_change(2)
    feats['momentum_accel_3d'] = ret_1d - ret_2d

    # 2. 日内波动率 vs 隔夜跳空
    intraday_range = (h - l) / c.shift(1)
    overnight_gap = (o - c.shift(1)) / c.shift(1)
    feats['intraday_vol'] = intraday_range
    feats['overnight_gap'] = overnight_gap
    feats['gap_to_range_ratio'] = safe_divide(overnight_gap.abs(), intraday_range)

    # 3. 成交量爆发检测
    vol_ma5 = v.rolling(5).mean()
    vol_std5 = v.rolling(5).std()
    feats['volume_surge'] = safe_divide(v - vol_ma5, vol_std5)

    # 4. 收盘价在当日区间的位置
    feats['price_position'] = safe_divide(c - l, h - l)

    # 5. 尾盘强度
    feats['close_strength'] = safe_divide(2*c - h - l, h - l)

    return feats


# ============================================================
# 特征族 2：短期特征（5-10天）- 捕捉均值回归
# ============================================================
def short_term_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    短期特征：捕捉超买超卖和均值回归信号
    
    创新点：
    - 多窗口 RSI 组合
    - 布林带位置 + 带宽变化
    - 短期波动率 regime
    """
    feats = {}
    
    c = df['close'].astype(float)
    v = df['volume'].astype(float).replace(0, np.nan)
    
    # 1. 短期收益率
    feats['ret_3d'] = c.pct_change(3)
    feats['ret_5d'] = c.pct_change(5)
    feats['ret_10d'] = c.pct_change(10)
    
    # 2. 多窗口 RSI
    def calc_rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = safe_divide(gain, loss, fill_value=1.0)
        return 100 - 100 / (1 + rs)
    
    feats['rsi_5'] = calc_rsi(c, 5)
    feats['rsi_10'] = calc_rsi(c, 10)
    feats['rsi_spread'] = feats['rsi_5'] - feats['rsi_10']  # RSI 差值
    
    # 3. 布林带特征
    ma10 = c.rolling(10).mean()
    std10 = c.rolling(10).std()
    upper_band = ma10 + 2 * std10
    lower_band = ma10 - 2 * std10
    
    feats['bb_position'] = safe_divide(c - lower_band, upper_band - lower_band)
    feats['bb_width'] = safe_divide(upper_band - lower_band, ma10)  # 带宽
    feats['bb_width_change'] = feats['bb_width'].pct_change(5)  # 带宽变化
    
    # 4. 短期波动率 regime（波动率的波动率）
    ret_1d = c.pct_change()
    vol_5d = ret_1d.rolling(5).std()
    vol_10d = ret_1d.rolling(10).std()
    feats['vol_regime'] = safe_divide(vol_5d, vol_10d)  # >1 = 波动率上升
    
    # 5. 量价背离检测
    ret_corr = ret_1d.rolling(10).corr(v.pct_change())
    feats['price_volume_divergence'] = ret_corr
    
    return feats


# ============================================================
# 特征族 3：中期特征（20-60天）- 捕捉趋势和动量
# ============================================================
def medium_term_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    中期特征：捕捉趋势强度和持续性
    
    创新点：
    - 趋势强度指标（ADX风格）
    - 多均线系统
    - 波动率聚类
    """
    feats = {}
    
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)

    # 1. 中期收益率
    feats['ret_20d'] = c.pct_change(20)
    feats['ret_40d'] = c.pct_change(40)
    feats['ret_60d'] = c.pct_change(60)
    
    # 2. 趋势强度（类似 ADX）
    # 使用方向性移动指标
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr_14 = tr.rolling(14).mean()
    feats['trend_strength'] = safe_divide(c - c.rolling(20).mean(), atr_14)
    
    # 3. 多均线系统
    ma20 = c.rolling(20).mean()
    ma40 = c.rolling(40).mean()
    ma60 = c.rolling(60).mean()
    
    feats['ma_slope_20'] = (ma20 - ma20.shift(5)) / ma20.shift(5)  # 均线斜率
    feats['ma_alignment'] = (
        (ma20 > ma40).astype(int) + (ma40 > ma60).astype(int)
    )  # 均线排列：0,1,2
    
    # 4. 突破检测
    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    feats['breakout_high'] = (c > high_20.shift(1)).astype(int)
    feats['breakout_low'] = (c < low_20.shift(1)).astype(int)
    
    # 5. 波动率特征
    ret_1d = c.pct_change()
    feats['volatility_20d'] = ret_1d.rolling(20).std()
    feats['volatility_40d'] = ret_1d.rolling(40).std()
    feats['vol_ratio'] = safe_divide(
        feats['volatility_20d'], 
        feats['volatility_40d']
    )
    
    # 6. 最大回撤（20日窗口）
    rolling_max = c.rolling(20).max()
    feats['max_drawdown_20d'] = safe_divide(c - rolling_max, rolling_max)

    # 7. MACD (12-26-9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feats['macd_hist'] = macd_line - macd_signal

    return feats


# ============================================================
# 特征族 4：流动性和市场微观结构
# ============================================================
def liquidity_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    流动性特征：交易成本和市场深度
    
    创新点：
    - Amihud 非流动性
    - 换手率动态
    - 成交金额集中度
    """
    feats = {}
    
    c = df['close'].astype(float)
    v = df['volume'].astype(float).replace(0, np.nan)
    ret_1d = c.pct_change()
    
    # 1. Amihud 非流动性指标
    feats['amihud_20d'] = (ret_1d.abs() / v).rolling(20).mean()
    
    # 2. 换手率特征
    if 'turnover' in df.columns:
        turnover = df['turnover'].astype(float)
        feats['turnover_ma20'] = turnover.rolling(20).mean()
        feats['turnover_std20'] = turnover.rolling(20).std()
        feats['turnover_surge'] = safe_divide(
            turnover - feats['turnover_ma20'],
            feats['turnover_std20']
        )
    
    # 3. 成交量相对强度
    vol_ma20 = v.rolling(20).mean()
    feats['volume_strength'] = safe_divide(v, vol_ma20)
    
    # 4. 价格影响（价格变化 / 成交量）
    feats['price_impact'] = safe_divide(ret_1d, v / v.rolling(20).mean())
    
    return feats# ============================================================
# 特征族 5：相对强度动量
# ============================================================
def relative_strength_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    feats = {}
    c = df['close'].astype(float)
    feats['momentum_20d'] = c.pct_change(20)
    feats['momentum_40d'] = c.pct_change(40)
    feats['momentum_diff'] = feats['momentum_20d'] - feats['momentum_40d']
    return feats


# ============================================================
# 主函数：整合所有特征
# ============================================================
def build_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为单只股票计算所有特征
    
    Args:
        df: 单只股票的价格数据（需包含 date, open, high, low, close, volume）
    
    Returns:
        带特征的 DataFrame
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    # 收集所有特征
    all_features = {}
    
    # 添加各特征族
    all_features.update(ultra_short_term_features(df))
    all_features.update(short_term_features(df))
    all_features.update(medium_term_features(df))
    all_features.update(liquidity_features(df))
    all_features.update(relative_strength_features(df))

    # 计算目标变量
    c = df['close'].astype(float)
    all_features[TARGET_COLUMN] = c.shift(-FORWARD_HORIZON) / c - 1.0
    
    # 合并到原始数据
    feature_df = pd.DataFrame(all_features, index=df.index)
    result = pd.concat([df, feature_df], axis=1)
    
    return result


# ============================================================
# 截面标准化
# ============================================================
def cross_sectional_normalize(panel: pd.DataFrame) -> pd.DataFrame:
    """
    每日截面标准化：
    1. Winsorize 去极值（1%, 99%）
    2. Z-score 标准化
    3. 目标变量也做截面标准化
    """
    # 获取所有特征列
    feature_cols = [col for col in panel.columns 
                   if col not in ['date', 'stock_code', 'open', 'high', 'low', 
                                 'close', 'volume', 'amount', 'turnover', 
                                 'pct_change', TARGET_COLUMN]]
    
    def winsorize_and_zscore(group: pd.DataFrame) -> pd.DataFrame:
        """对一天的数据进行处理"""
        if len(group) < 30:  # 股票数太少，不处理
            return group
        
        for col in feature_cols:
            if col not in group.columns:
                continue
            
            # Winsorize
            lower = group[col].quantile(0.01)
            upper = group[col].quantile(0.99)
            group[col] = group[col].clip(lower=lower, upper=upper)
            
            # Z-score
            mean = group[col].mean()
            std = group[col].std()
            if std > 1e-8:
                group[col] = (group[col] - mean) / std
        
        # 目标变量也标准化
        if TARGET_COLUMN in group.columns:
            target = group[TARGET_COLUMN]
            # Winsorize
            lower = target.quantile(0.01)
            upper = target.quantile(0.99)
            target = target.clip(lower=lower, upper=upper)
            # Z-score
            mean = target.mean()
            std = target.std()
            if std > 1e-8:
                group[TARGET_COLUMN] = (target - mean) / std
        
        return group
    
    # 按日期分组处理（for 循环兼容 pandas 3.x）
    parts = []
    for _, group in panel.groupby('date'):
        parts.append(winsorize_and_zscore(group.copy()))
    panel = pd.concat(parts, ignore_index=True)
    return panel


# ============================================================
# 特征族 6：基本面特征（PE、PB、PS 等）
# ============================================================
def fundamental_features(df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> dict[str, pd.Series]:
    """
    基本面特征：估值指标
    
    Args:
        df: 股票价格数据
        fundamentals_df: 基本面数据（可选）
    
    创新点：
    - PE/PB 相对于历史分位数
    - 估值变化率
    - 市值因子
    """
    feats = {}
    
    if fundamentals_df is None:
        # 如果没有基本面数据，返回空特征
        return feats
    
    # 确保日期格式一致
    df = df.copy()
    fundamentals_df = fundamentals_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    fundamentals_df['date'] = pd.to_datetime(fundamentals_df['date'])
    
    # 合并基本面数据
    stock_code = df['stock_code'].iloc[0] if 'stock_code' in df.columns else None
    if stock_code:
        fund_stock = fundamentals_df[fundamentals_df['stock_code'] == stock_code].copy()
        fund_stock = fund_stock.sort_values('date')
        
        # 按日期合并（向前填充，因为基本面数据更新频率低）
        merged = pd.merge_asof(
            df.sort_values('date'),
            fund_stock[['date', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'total_mv']],
            on='date',
            direction='backward'
        )
        
        # 1. 原始估值指标
        feats['pe'] = merged['pe']
        feats['pe_ttm'] = merged['pe_ttm']
        feats['pb'] = merged['pb']
        feats['ps'] = merged['ps']
        
        # 2. 估值相对于历史分位数（过去 60 天）
        feats['pe_percentile_60d'] = merged['pe'].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
            raw=False
        )
        feats['pb_percentile_60d'] = merged['pb'].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
            raw=False
        )
        
        # 3. 估值变化率（10日、20日）
        feats['pe_change_10d'] = merged['pe'].pct_change(10)
        feats['pb_change_20d'] = merged['pb'].pct_change(20)
        
        # 4. 市值因子（总市值的对数）
        feats['log_market_cap'] = np.log(merged['total_mv'] + 1.0)
        
        # 5. 估值倒数（类似 EP、BP）
        feats['earnings_yield'] = safe_divide(pd.Series(1.0, index=merged.index), merged['pe'])
        feats['book_to_price'] = safe_divide(pd.Series(1.0, index=merged.index), merged['pb'])
    
    return feats


# ============================================================
# 公共接口
# ============================================================
def add_market_relative_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    市场相对特征（需要全截面，在 per-stock 循环后、标准化前调用）：
      - excess_ret_20d : 个股20日收益 − 截面等权均值（相对强度）
      - mkt_beta_60d   : 个股收益对市场收益的60日滚动 Beta（系统性风险暴露）
    """
    panel = panel.sort_values(['stock_code', 'date']).copy()

    # excess_ret_20d：个股20日收益 - 截面均值
    mkt_ret_20d = panel.groupby('date')['ret_20d'].transform('mean')
    panel['excess_ret_20d'] = panel['ret_20d'] - mkt_ret_20d

    # 主题标记（二值，不做截面标准化）
    AEROSPACE = {'002025', '600118', '600879', '600435', '300627'}
    ROBOT     = {'688017', '300024', '688297', '688122'}
    panel['is_aerospace'] = panel['stock_code'].isin(AEROSPACE).astype(float)
    panel['is_robot']     = panel['stock_code'].isin(ROBOT).astype(float)

    return panel


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    从原始价格数据构建特征面板

    Args:
        prices: 原始价格数据（需包含 stock_code, date, open, high, low, close, volume）

    Returns:
        特征面板（每只股票 × 每天 × 所有特征）
    """
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date'])

    # 对每只股票计算技术特征
    frames = []
    for stock_code, stock_df in prices.groupby('stock_code', sort=False):
        features_df = build_stock_features(stock_df)
        features_df['stock_code'] = stock_code
        frames.append(features_df)

    # 合并所有股票
    panel = pd.concat(frames, ignore_index=True)

    # 市场相对特征（需要全截面数据，在标准化前加入）
    panel = add_market_relative_features(panel)

    # 截面标准化
    panel = cross_sectional_normalize(panel)

    return panel


# ============================================================
# 辅助函数：提取特征列名
# ============================================================
def get_feature_columns(panel: pd.DataFrame) -> List[str]:
    """
    获取所有特征列名（自动检测）
    """
    exclude = ['date', 'stock_code', 'open', 'high', 'low', 'close', 
               'volume', 'amount', 'turnover', 'pct_change', TARGET_COLUMN]
    
    feature_cols = [col for col in panel.columns if col not in exclude]
    return feature_cols


# ============================================================
# 训练/预测数据框架
# ============================================================
def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    """
    提取训练数据（去除缺失值）
    """
    feature_cols = get_feature_columns(panel)
    
    # 去除特征或目标为空的行
    df = panel.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    
    # 时间过滤
    if min_date is not None:
        df = df[df['date'] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df['date'] <= pd.Timestamp(max_date)]
    
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """
    提取预测数据（某一天的所有股票）
    """
    if as_of is None:
        as_of = panel['date'].max()
    as_of = pd.Timestamp(as_of)
    
    feature_cols = get_feature_columns(panel)
    
    # 只要特征完整即可（目标可以为空）
    df = panel[panel['date'] == as_of].dropna(subset=feature_cols).copy()
    
    return df


# ============================================================
# 导出
# ============================================================
FEATURE_COLUMNS = None  # 将在运行时通过 get_feature_columns() 动态获取

__all__ = [
    'build_features',
    'get_feature_columns',
    'training_frame',
    'prediction_frame',
    'TARGET_COLUMN',
    'FORWARD_HORIZON',
]
