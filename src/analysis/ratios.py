"""
财务比率分析模块 - Financial Ratio Analysis
盈利能力 / 偿债能力 / 营运能力 / 发展能力 / 现金流量
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_all_ratios(financials: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    从财务报表数据计算全部关键财务比率。
    financials 应包含行索引为科目名称、列为期间的 DataFrame。
    """
    ratios = {}

    # ---- 盈利能力 (Profitability) ----
    ratios.update(_profitability_ratios(financials))
    # ---- 偿债能力 (Solvency) ----
    ratios.update(_solvency_ratios(financials))
    # ---- 营运能力 (Operating Efficiency) ----
    ratios.update(_efficiency_ratios(financials))
    # ---- 发展能力 (Growth) ----
    ratios.update(_growth_ratios(financials))
    # ---- 每股指标 (Per-Share) ----
    ratios.update(_per_share_ratios(financials))

    return {k: round(v, 4) if v is not None else None for k, v in ratios.items()}


def _safe_div(a, b):
    """安全除法，分母为 0 或缺失返回 None"""
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return None
        return a / b
    except (TypeError, ZeroDivisionError):
        return None


def _get(df, row_name):
    """安全取数，取最后一列（最近期间）"""
    try:
        if row_name in df.index:
            val = df.loc[row_name].iloc[-1] if hasattr(df.loc[row_name], 'iloc') else df.loc[row_name]
            return float(val) if not pd.isna(val) else None
    except (KeyError, IndexError, TypeError):
        pass
    return None


def _profitability_ratios(df: pd.DataFrame) -> dict:
    revenue = _get(df, '营业收入') or _get(df, '营业总收入')
    net_profit = _get(df, '净利润') or _get(df, '归属于母公司股东的净利润')
    total_profit = _get(df, '利润总额')
    operating_profit = _get(df, '营业利润')
    total_assets = _get(df, '资产总计') or _get(df, '总资产')
    equity = _get(df, '所有者权益合计') or _get(df, '归属于母公司股东权益合计')
    cost = _get(df, '营业成本')

    return {
        '销售毛利率': _safe_div(revenue - cost, revenue) if revenue and cost else None,
        '销售净利率': _safe_div(net_profit, revenue),
        '总资产报酬率_ROA': _safe_div(total_profit, total_assets),
        '净资产收益率_ROE': _safe_div(net_profit, equity),
        '营业利润率': _safe_div(operating_profit, revenue),
    }


def _solvency_ratios(df: pd.DataFrame) -> dict:
    current_assets = _get(df, '流动资产合计')
    current_liabilities = _get(df, '流动负债合计')
    total_liabilities = _get(df, '负债合计')
    total_assets = _get(df, '资产总计') or _get(df, '总资产')
    equity = _get(df, '所有者权益合计') or _get(df, '归属于母公司股东权益合计')
    inventory = _get(df, '存货')
    monetary = _get(df, '货币资金')
    accounts_receivable = _get(df, '应收账款')
    ebitda = _get(df, '息税折旧摊销前利润') or _get(df, '利润总额')

    quick_assets = None
    if current_assets is not None and inventory is not None:
        quick_assets = current_assets - inventory

    return {
        '流动比率': _safe_div(current_assets, current_liabilities),
        '速动比率': _safe_div(quick_assets, current_liabilities),
        '现金比率': _safe_div(monetary, current_liabilities),
        '资产负债率': _safe_div(total_liabilities, total_assets),
        '产权比率': _safe_div(total_liabilities, equity),
        '权益乘数': _safe_div(total_assets, equity),
    }


def _efficiency_ratios(df: pd.DataFrame) -> dict:
    revenue = _get(df, '营业收入') or _get(df, '营业总收入')
    cost = _get(df, '营业成本')
    total_assets = _get(df, '资产总计') or _get(df, '总资产')
    current_assets = _get(df, '流动资产合计')
    inventory = _get(df, '存货')
    accounts_receivable = _get(df, '应收账款')
    fixed_assets = _get(df, '固定资产')

    return {
        '总资产周转率': _safe_div(revenue, total_assets),
        '流动资产周转率': _safe_div(revenue, current_assets),
        '存货周转率': _safe_div(cost, inventory),
        '应收账款周转率': _safe_div(revenue, accounts_receivable),
        '固定资产周转率': _safe_div(revenue, fixed_assets),
    }


def _growth_ratios(df: pd.DataFrame) -> dict:
    """计算同比增长率，需至少两期数据"""
    if df.shape[1] < 2:
        return {}

    def yoy(row_name):
        try:
            if row_name in df.index:
                vals = df.loc[row_name]
                if hasattr(vals, 'iloc') and len(vals) >= 2:
                    a, b = float(vals.iloc[-1]), float(vals.iloc[-2])
                    return _safe_div(a - b, abs(b)) if b != 0 else None
        except (KeyError, IndexError, TypeError):
            pass
        return None

    return {
        '营业收入增长率': yoy('营业收入') or yoy('营业总收入'),
        '净利润增长率': yoy('净利润') or yoy('归属于母公司股东的净利润'),
        '总资产增长率': yoy('资产总计') or yoy('总资产'),
    }


def _per_share_ratios(df: pd.DataFrame) -> dict:
    return {
        '基本每股收益_EPS': _get(df, '基本每股收益'),
        '每股净资产': _get(df, '每股净资产'),
    }


def interpret_ratios(ratios: Dict[str, Optional[float]]) -> str:
    """对计算结果给出中文解读"""
    lines = []
    benchmarks = {
        '流动比率': (2.0, '一般认为 >2 较为安全'),
        '速动比率': (1.0, '一般认为 >1 较为安全'),
        '资产负债率': (0.6, '一般认为 40%-60% 较为合理，超过70%需关注'),
        '销售毛利率': (0.3, '因行业而异，一般越高越好'),
        '净资产收益率_ROE': (0.15, '一般认为 >15% 较为优秀'),
        '总资产周转率': (0.8, '因行业而异，越高说明资产利用效率越高'),
    }

    for name, value in ratios.items():
        if value is None:
            continue
        if name in benchmarks:
            threshold, note = benchmarks[name]
            symbol = '✓' if value >= threshold else '⚠'
            lines.append(f"{symbol} {name}: {value:.4f}  {note}")
        else:
            lines.append(f"  {name}: {value:.4f}")

    return '\n'.join(lines) if lines else '暂无足够数据计算财务比率'
