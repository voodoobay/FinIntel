"""
财务统计分析模块 - Financial Statistical Analysis
描述性统计 / 趋势分析 / 相关性分析 / 杜邦分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


def descriptive_stats(series: pd.Series) -> Dict[str, float]:
    """对单个财务指标做描述性统计"""
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        '样本量': len(s),
        '均值': float(s.mean()),
        '中位数': float(s.median()),
        '标准差': float(s.std(ddof=1)) if len(s) > 1 else 0,
        '最小值': float(s.min()),
        '最大值': float(s.max()),
        '偏度': float(s.skew()) if len(s) > 2 else 0,
        '峰度': float(s.kurtosis()) if len(s) > 3 else 0,
        '变异系数': float(s.std(ddof=1) / s.mean()) if len(s) > 1 and s.mean() != 0 else 0,
    }


def trend_analysis(series: pd.Series) -> Dict:
    """趋势分析：线性回归 + 年平均增长率 CAGR"""
    s = series.dropna()
    if len(s) < 2:
        return {'error': '数据点不足，至少需要2期'}

    x = np.arange(len(s))
    y = s.values.astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # CAGR
    cagr = (y[-1] / y[0]) ** (1 / (len(y) - 1)) - 1 if y[0] != 0 else None

    # 趋势方向
    if p_value < 0.05:
        direction = '显著上升' if slope > 0 else '显著下降'
    else:
        direction = '趋势不显著'

    return {
        '趋势方向': direction,
        '回归斜率': round(float(slope), 4),
        'R平方': round(float(r_value ** 2), 4),
        'P值': round(float(p_value), 4),
        '复合年增长率_CAGR': round(float(cagr), 4) if cagr is not None else None,
        '期数': len(s),
    }


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """计算财务指标间的相关系数矩阵 (Pearson)"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(4)


def dupont_analysis(financials: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    杜邦分析：ROE = 销售净利率 × 总资产周转率 × 权益乘数
    """
    def _get(row_name):
        try:
            if row_name in financials.index:
                val = financials.loc[row_name].iloc[-1]
                return float(val) if not pd.isna(val) else None
        except (KeyError, IndexError, TypeError):
            pass
        return None

    revenue = _get('营业收入') or _get('营业总收入')
    net_profit = _get('净利润') or _get('归属于母公司股东的净利润')
    total_assets = _get('资产总计') or _get('总资产')
    equity = _get('所有者权益合计') or _get('归属于母公司股东权益合计')

    npm = net_profit / revenue if revenue and net_profit and revenue != 0 else None  # 销售净利率
    tat = revenue / total_assets if revenue and total_assets and total_assets != 0 else None  # 总资产周转率
    em = total_assets / equity if total_assets and equity and equity != 0 else None  # 权益乘数
    roe = npm * tat * em if all([npm, tat, em]) else None

    return {
        '净资产收益率_ROE': round(roe, 4) if roe is not None else None,
        '销售净利率': round(npm, 4) if npm is not None else None,
        '总资产周转率': round(tat, 4) if tat is not None else None,
        '权益乘数': round(em, 4) if em is not None else None,
        '分解验证': f'{npm:.4f} × {tat:.4f} × {em:.4f} = {roe:.4f}' if all([npm, tat, em, roe]) else '数据不足',
    }


def variance_analysis(actual: pd.Series, budget: pd.Series) -> pd.DataFrame:
    """预算与实际差异分析"""
    df = pd.DataFrame({
        '实际': actual,
        '预算': budget,
        '差异': actual - budget,
    })
    df['差异率'] = (df['差异'] / df['预算'].abs()).replace([np.inf, -np.inf], np.nan).round(4)
    return df
