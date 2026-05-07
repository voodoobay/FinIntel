"""
财务异常检测模块 - Financial Anomaly Detection
Z-Score / IQR / 同比异常 / Benford定律
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def zscore_anomalies(series: pd.Series, threshold: float = 2.0) -> pd.DataFrame:
    """
    Z-Score 异常检测：偏离均值超过 N 个标准差的点标记为异常。
    适用于近似正态分布的财务数据。
    """
    s = series.dropna()
    if len(s) < 3:
        return pd.DataFrame()

    mean, std = s.mean(), s.std(ddof=1)
    if std == 0:
        return pd.DataFrame()

    z = ((s - mean) / std).abs()
    anomalies = s[z > threshold]
    result = pd.DataFrame({'值': anomalies, 'Z分数': z[anomalies.index].round(2)})
    result['偏离方向'] = result['值'].apply(lambda x: '偏高' if x > mean else '偏低')
    return result.sort_values('Z分数', ascending=False)


def iqr_anomalies(series: pd.Series, multiplier: float = 1.5) -> pd.DataFrame:
    """
    IQR (四分位距) 异常检测：对偏态分布更稳健。
    """
    s = series.dropna()
    if len(s) < 4:
        return pd.DataFrame()

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.DataFrame()

    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    mask = (s < lower) | (s > upper)
    anomalies = s[mask]
    result = pd.DataFrame({'值': anomalies})
    result['偏离方向'] = result['值'].apply(lambda x: '偏高' if x > upper else '偏低')
    return result.sort_values('值', ascending=False)


def yoy_anomalies(current: pd.Series, prior: pd.Series, threshold_pct: float = 0.3) -> pd.DataFrame:
    """
    同比异常：同比变动超过阈值的科目标记为异常。
    适用于月度/季度财务数据同比检查。
    """
    if len(current) != len(prior):
        return pd.DataFrame({'error': ['两期数据长度不一致']})

    df = pd.DataFrame({'本期': current, '上年同期': prior})
    df['同比变动额'] = df['本期'] - df['上年同期']
    df['同比变动率'] = (df['同比变动额'] / df['上年同期'].abs()).replace([np.inf, -np.inf], np.nan)
    anomalies = df[df['同比变动率'].abs() > threshold_pct].copy()
    anomalies['同比变动率'] = anomalies['同比变动率'].round(4)
    return anomalies.sort_values('同比变动率', key=abs, ascending=False)


def benford_test(series: pd.Series) -> Dict:
    """
    Benford 定律检验：自然产生的财务数据首位数字应符合 Benford 分布。
    显著偏离可能暗示人为操纵。常用于审计场景。
    """
    def first_digit(x):
        try:
            x = abs(float(x))
            if x == 0:
                return 0
            return int(str(x).replace('.', '').lstrip('0')[0])
        except (ValueError, TypeError):
            return 0

    s = series.dropna()
    if len(s) < 50:
        return {'error': '样本量不足，建议至少50条数据'}

    digits = s.apply(first_digit)
    digit_counts = digits.value_counts().sort_index()
    digit_counts = digit_counts[digit_counts.index.isin(range(1, 10))]

    # Benford 理论分布
    benford_expected = {d: np.log10(1 + 1 / d) for d in range(1, 10)}
    total = digit_counts.sum()
    observed = {d: round(digit_counts.get(d, 0) / total, 4) for d in range(1, 10)}

    # 最大偏差
    max_deviation = 0
    max_dev_digit = 0
    for d in range(1, 10):
        dev = abs(observed[d] - benford_expected[d])
        if dev > max_deviation:
            max_deviation = dev
            max_dev_digit = d

    suspicious = max_deviation > 0.1

    return {
        '样本量': len(s),
        '首位数字分布_实际': observed,
        '首位数字分布_理论': {d: round(v, 4) for d, v in benford_expected.items()},
        '最大偏差数字': max_dev_digit,
        '最大偏差值': round(max_deviation, 4),
        '是否存在异常': '是，首位数分布偏离 Benford 定律，建议进一步审计' if suspicious else '未发现明显异常',
    }
