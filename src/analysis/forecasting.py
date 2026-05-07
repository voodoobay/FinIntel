"""
财务预测模块 - Financial Forecasting
移动平均 / 指数平滑 / 线性回归预测 / 简单季节性分解
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression


def moving_average(series: pd.Series, window: int = 3) -> pd.Series:
    """移动平均平滑"""
    return series.rolling(window=window, min_periods=1).mean()


def exponential_smoothing(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    """一次指数平滑"""
    s = series.dropna()
    result = [s.iloc[0]]
    for v in s.iloc[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return pd.Series(result, index=s.index)


def linear_forecast(series: pd.Series, periods: int = 3) -> Dict:
    """
    线性回归预测未来 N 期，返回预测值和置信区间。
    """
    s = series.dropna()
    if len(s) < 3:
        return {'error': '数据点不足，至少需要3期'}

    x = np.arange(len(s)).reshape(-1, 1)
    y = s.values.astype(float)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    # 残差标准差
    residuals = y - y_pred
    residual_std = np.std(residuals, ddof=2) if len(residuals) > 2 else 0

    # 预测未来
    future_x = np.arange(len(s), len(s) + periods).reshape(-1, 1)
    future_pred = model.predict(future_x)
    # 95% 预测区间
    ci = 1.96 * residual_std * np.sqrt(1 + 1 / len(s))

    return {
        '模型': f'y = {model.coef_[0]:.4f} × t + {model.intercept_:.4f}',
        'R平方': round(float(model.score(x, y)), 4),
        '历史拟合值': [round(v, 2) for v in y_pred],
        '预测值': {f'第{i+1}期': round(float(future_pred[i]), 2) for i in range(periods)},
        '95%置信下限': {f'第{i+1}期': round(float(future_pred[i] - ci), 2) for i in range(periods)},
        '95%置信上限': {f'第{i+1}期': round(float(future_pred[i] + ci), 2) for i in range(periods)},
    }


def seasonal_decompose(series: pd.Series, period: int = 4) -> Dict:
    """
    简单季节性分解（移动平均法）：分解为趋势 + 季节性 + 残差
    period: 周期长度（季度数据=4，月度数据=12）
    """
    s = series.dropna()
    if len(s) < period * 2:
        return {'error': f'数据点不足，至少需要 {period * 2} 期'}

    # 趋势（中心化移动平均）
    trend = s.rolling(window=period, center=True, min_periods=1).mean()

    # 去趋势
    detrended = s - trend

    # 季节性（各周期位置的平均）
    seasonal = np.zeros(len(s))
    for i in range(period):
        idx = list(range(i, len(s), period))
        if idx:
            seasonal[idx] = detrended.iloc[idx].mean()

    # 残差
    residual = s - trend - seasonal

    return {
        '趋势': [round(v, 2) for v in trend.values],
        '季节性': [round(v, 2) for v in seasonal],
        '残差': [round(v, 2) for v in residual.values],
        '季节性强度': round(1 - residual.var() / (s - trend).var(), 4) if (s - trend).var() != 0 else 0,
    }


def forecast_summary(series: pd.Series, periods: int = 3, series_name: str = '指标') -> str:
    """生成预测摘要报告"""
    s = series.dropna()
    if len(s) < 3:
        return f'{series_name}: 数据不足无法预测'

    forecast = linear_forecast(s, periods)
    if 'error' in forecast:
        return forecast['error']

    lines = [f'【{series_name} 预测报告】']
    lines.append(f'模型: {forecast["模型"]}')
    lines.append(f'拟合优度 R² = {forecast["R平方"]}')
    lines.append('未来预测:')
    for i in range(periods):
        k = f'第{i+1}期'
        lines.append(f'  {k}: {forecast["预测值"][k]} (95%CI: [{forecast["95%置信下限"][k]}, {forecast["95%置信上限"][k]}])')

    return '\n'.join(lines)
