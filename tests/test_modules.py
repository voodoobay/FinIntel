"""
核心模块烟雾测试 - 无需 API Key
运行: python tests/test_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.data.loader import load_financial_data, get_summary, detect_statement_type
from src.analysis.ratios import compute_all_ratios, interpret_ratios
from src.analysis.statistics import (
    descriptive_stats, trend_analysis, correlation_matrix,
    dupont_analysis, variance_analysis,
)
from src.analysis.anomaly import zscore_anomalies, iqr_anomalies, yoy_anomalies, benford_test
from src.analysis.forecasting import linear_forecast, forecast_summary, moving_average, exponential_smoothing
from src.analysis.financial_statements import (
    common_size_bs, common_size_is, horizontal_analysis, vertical_analysis_bs,
)


def test_data_loader():
    print('=== 数据加载模块 ===')
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data')
    for f in os.listdir(sample_dir):
        fp = os.path.join(sample_dir, f)
        df = load_financial_data(fp)
        stype = detect_statement_type(df)
        print(f'\n文件: {f}')
        print(get_summary(df))
        print(f'识别类型: {stype}')
    print('[PASS] Data loader\n')


def test_ratios():
    print('=== 财务比率模块 ===')
    fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data', '利润表_示例.csv')
    df = load_financial_data(fp)
    ratios = compute_all_ratios(df)
    print(interpret_ratios(ratios))
    print('[PASS] Ratios\n')


def _find_key(df, keyword):
    """按关键词模糊匹配索引名称"""
    for idx in df.index:
        if keyword in str(idx):
            return idx
    return None


def test_statistics():
    print('=== 统计分析模块 ===')
    fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data', '利润表_示例.csv')
    df = load_financial_data(fp)

    # Descriptive stats
    revenue_key = _find_key(df, '营业收入')
    if revenue_key:
        revenue = df.loc[revenue_key]
        stats = descriptive_stats(revenue)
        print(f'{revenue_key} 描述统计:', stats)

        # Trend
        trend = trend_analysis(revenue)
        print(f'{revenue_key} 趋势:', trend)

    # Dupont
    fp2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data', '资产负债表_示例.csv')
    bs = load_financial_data(fp2)
    dupont = dupont_analysis(bs)
    print('杜邦分析:', dupont)
    print('[PASS] Statistics\n')


def test_anomaly():
    print('=== 异常检测模块 ===')
    fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data', '利润表_示例.csv')
    df = load_financial_data(fp)

    revenue_key = _find_key(df, '营业收入')
    if revenue_key:
        revenue = df.loc[revenue_key]

        z_anom = zscore_anomalies(revenue)
        print(f'{revenue_key} Z-Score 异常:', z_anom)

        iqr_anom = iqr_anomalies(revenue)
        print(f'{revenue_key} IQR 异常:', iqr_anom)

    # Benford
    all_vals = pd.Series(df.values.flatten()).dropna()
    bf = benford_test(all_vals)
    print(f'Benford 检验: {bf.get("是否存在异常", bf.get("error", ""))}')
    print('[PASS] Anomaly\n')


def test_forecasting():
    print('=== 预测模块 ===')
    fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data', '利润表_示例.csv')
    df = load_financial_data(fp)
    revenue_key = _find_key(df, '营业收入')
    if revenue_key:
        revenue = df.loc[revenue_key]
        forecast = linear_forecast(revenue, periods=2)
        print(f'{revenue_key} 预测: {forecast}')
    print('[PASS] Forecasting\n')


def test_fs_analysis():
    print('=== 财务报表分析模块 ===')
    fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data', '利润表_示例.csv')
    df = load_financial_data(fp)

    cs = common_size_is(df)
    print('共同比分析 (利润表) 前3行:')
    print(cs.iloc[:3])

    ha = horizontal_analysis(df)
    print('水平分析 列名:', list(ha.columns))
    print('[PASS] FS Analysis\n')


if __name__ == '__main__':
    try:
        test_data_loader()
        test_ratios()
        test_statistics()
        test_anomaly()
        test_forecasting()
        test_fs_analysis()
        print('=' * 50)
        print('ALL TESTS PASSED')
    except Exception as e:
        print(f'TEST FAILED: {e}')
        import traceback
        traceback.print_exc()
