"""
数据加载模块 - Data Loader
支持 CSV / Excel 格式的财务报表和交易数据加载
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple


def load_financial_data(filepath: str) -> pd.DataFrame:
    """
    加载财务报表数据。
    支持 CSV 和 Excel 格式。
    自动将第一列设为行索引（科目名称），列名为期间。
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'文件不存在: {filepath}')

    if path.suffix.lower() in ('.csv',):
        df = pd.read_csv(filepath, index_col=0, encoding='utf-8')
    elif path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath, index_col=0)
    else:
        raise ValueError(f'不支持的文件格式: {path.suffix}，请使用 CSV 或 Excel')

    # 清理数据：移除全空行/列
    df = df.dropna(how='all').dropna(axis=1, how='all')
    # 数值列强制转换
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_transactions(filepath: str) -> pd.DataFrame:
    """
    加载交易流水数据。
    支持 CSV / Excel，自动解析日期列。
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'文件不存在: {filepath}')

    if path.suffix.lower() in ('.csv',):
        df = pd.read_csv(filepath, encoding='utf-8')
    elif path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f'不支持的文件格式: {path.suffix}')

    # 尝试自动识别日期列
    for col in df.columns:
        if any(kw in str(col).lower() for kw in ['日期', 'date', '时间', 'time']):
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 数值列转换
    for col in df.columns:
        if any(kw in str(col).lower() for kw in ['金额', 'amount', '数量', 'qty', '单价', 'price']):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def detect_statement_type(df: pd.DataFrame) -> str:
    """
    自动识别财务报表类型：资产负债表 / 利润表 / 现金流量表 / 未知
    """
    index_lower = [str(i).lower() for i in df.index]

    bs_keywords = ['资产总计', '负债合计', '所有者权益', '流动资产', '非流动资产', '流动负债']
    is_keywords = ['营业收入', '营业成本', '净利润', '营业利润', '利润总额', '销售费用']
    cf_keywords = ['经营活动现金流', '投资活动现金流', '筹资活动现金流', '现金及现金等价物']

    bs_score = sum(1 for kw in bs_keywords if any(kw in i for i in index_lower))
    is_score = sum(1 for kw in is_keywords if any(kw in i for i in index_lower))
    cf_score = sum(1 for kw in cf_keywords if any(kw in i for i in index_lower))

    scores = {'资产负债表': bs_score, '利润表': is_score, '现金流量表': cf_score}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else '未识别'


def get_summary(df: pd.DataFrame) -> str:
    """生成数据概览"""
    shape = df.shape
    statement_type = detect_statement_type(df)
    lines = [
        f'数据维度: {shape[0]} 行 × {shape[1]} 列',
        f'报表类型: {statement_type}',
        f'涵盖期间: {list(df.columns)}',
        f'主要科目: {list(df.index[:10])}...' if len(df.index) > 10 else f'科目列表: {list(df.index)}',
    ]
    missing = df.isnull().sum().sum()
    if missing > 0:
        lines.append(f'缺失值: {missing} 个')
    return '\n'.join(lines)
