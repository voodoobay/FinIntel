"""
财务报表结构分析模块 - Financial Statement Structural Analysis
共同比分析 / 垂直分析 / 水平分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def common_size_bs(balance_sheet: pd.DataFrame) -> pd.DataFrame:
    """
    资产负债表共同比分析（以总资产为100%）
    """
    total_assets_row = None
    for name in ['资产总计', '总资产']:
        if name in balance_sheet.index:
            total_assets_row = name
            break
    if total_assets_row is None:
        return balance_sheet

    return balance_sheet.div(balance_sheet.loc[total_assets_row]) * 100


def common_size_is(income_stmt: pd.DataFrame) -> pd.DataFrame:
    """
    利润表共同比分析（以营业收入为100%）
    """
    revenue_row = None
    for name in ['营业收入', '营业总收入']:
        if name in income_stmt.index:
            revenue_row = name
            break
    if revenue_row is None:
        return income_stmt

    return income_stmt.div(income_stmt.loc[revenue_row]) * 100


def horizontal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    水平分析（环比变动）：每列相对于前一列的变动率和变动额。
    """
    result = pd.DataFrame(index=df.index)
    for i in range(df.shape[1]):
        if i == 0:
            result[f'{df.columns[i]}_值'] = df.iloc[:, i]
        else:
            prev, curr = df.iloc[:, i - 1], df.iloc[:, i]
            result[f'{df.columns[i]}_变动率'] = ((curr - prev) / prev.abs()).replace([np.inf, -np.inf], np.nan).round(4)
            result[f'{df.columns[i]}_变动额'] = (curr - prev).round(2)
    return result


def vertical_analysis_bs(df: pd.DataFrame) -> pd.DataFrame:
    """
    资产负债表垂直分析：各项占总资产的比重。
    返回带占比的新 DataFrame。
    """
    total_assets = None
    for name in ['资产总计', '总资产']:
        if name in df.index:
            total_assets = df.loc[name]
            break
    if total_assets is None:
        return df

    pct = df.div(total_assets) * 100
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        result[f'{col}_金额'] = df[col]
        result[f'{col}_占比%'] = pct[col].round(2)
    return result


def vertical_analysis_is(df: pd.DataFrame) -> pd.DataFrame:
    """利润表垂直分析：各项占营业收入的比重"""
    revenue = None
    for name in ['营业收入', '营业总收入']:
        if name in df.index:
            revenue = df.loc[name]
            break
    if revenue is None:
        return df

    pct = df.div(revenue) * 100
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        result[f'{col}_金额'] = df[col]
        result[f'{col}_占比%'] = pct[col].round(2)
    return result
