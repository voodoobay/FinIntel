"""
现金流量表分析模块 - Cash Flow Statement Analysis
现金流质量 / 自由现金流 / 现金流比率 / 现金流结构
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def _get(df, row_name):
    for idx in df.index:
        if row_name in str(idx):
            val = df.loc[idx].iloc[-1] if hasattr(df.loc[idx], 'iloc') else df.loc[idx]
            return float(val) if not pd.isna(val) else None
    return None


def analyze_cashflow_structure(cf: pd.DataFrame) -> Dict:
    """分析现金流结构：三大活动的贡献比例和健康度"""
    ocf = _get(cf, '经营活动产生的现金流量净额')
    icf = _get(cf, '投资活动产生的现金流量净额')
    fcf_val = _get(cf, '筹资活动产生的现金流量净额')

    result = {
        '经营活动现金流净额': ocf,
        '投资活动现金流净额': icf,
        '筹资活动现金流净额': fcf_val,
    }

    if all(v is not None for v in [ocf, icf, fcf_val]):
        total_inflow = abs(ocf) + abs(icf) + abs(fcf_val)
        if total_inflow > 0:
            result['经营活动占比'] = round(ocf / total_inflow * 100, 2)
            result['投资活动占比'] = round(icf / total_inflow * 100, 2)
            result['筹资活动占比'] = round(fcf_val / total_inflow * 100, 2)

        # 现金流类型诊断
        if ocf > 0 and icf < 0 and fcf_val:
            result['现金流类型'] = '健康成长型：经营造血，投资扩张'
        elif ocf > 0 and icf > 0 and fcf_val < 0:
            result['现金流类型'] = '偿债收缩型：经营+投资回款，正在还债'
        elif ocf < 0 and icf < 0 and fcf_val > 0:
            result['现金流类型'] = '融资扩张型：靠借钱支撑投资，风险较高'
        elif ocf > 0 and icf < 0 and fcf_val < 0:
            result['现金流类型'] = '稳健型：经营造血同时投资+偿债'
        else:
            result['现金流类型'] = '混合型，需结合具体行业判断'

    return result


def compute_fcf(cf: pd.DataFrame) -> Dict[str, Optional[float]]:
    """计算自由现金流 (FCF)"""
    ocf = _get(cf, '经营活动产生的现金流量净额')
    capex = _get(cf, '购建固定资产、无形资产支付的现金')

    fcf = ocf - capex if ocf is not None and capex is not None else None

    return {
        '经营活动现金流净额_OCF': ocf,
        '资本支出_CapEx': capex,
        '自由现金流_FCF': fcf,
        'FCF占OCF比例': round(fcf / ocf * 100, 2) if fcf and ocf and ocf != 0 else None,
    }


def cashflow_ratios(cf: pd.DataFrame, is_df: pd.DataFrame = None, bs_df: pd.DataFrame = None) -> Dict[str, Optional[float]]:
    """
    现金流量相关比率。
    可选传入利润表和资产负债表做交叉分析。
    """
    ratios = {}

    ocf = _get(cf, '经营活动产生的现金流量净额')
    revenue = None
    net_profit = None
    total_debt = None
    current_liab = None
    total_assets = None

    # 从现金流量表提取
    investing_out = _get(cf, '投资活动现金流出小计')
    financing_out = _get(cf, '筹资活动现金流出小计')

    # 从利润表提取
    if is_df is not None:
        revenue = _get(is_df, '营业收入')
        net_profit = _get(is_df, '净利润') or _get(is_df, '归属于母公司股东的净利润')

    # 从资产负债表提取
    if bs_df is not None:
        total_debt = _get(bs_df, '负债合计')
        current_liab = _get(bs_df, '流动负债合计')
        total_assets_val = _get(bs_df, '资产总计') or _get(bs_df, '总资产')

    # 现金流质量比率
    ratios['营业收入现金含量'] = round(ocf / revenue, 4) if ocf and revenue and revenue != 0 else None
    ratios['净利润现金含量'] = round(ocf / net_profit, 4) if ocf and net_profit and net_profit != 0 else None

    # 偿债能力
    ratios['现金流负债比率'] = round(ocf / total_debt, 4) if ocf and total_debt and total_debt != 0 else None
    ratios['现金流流动负债比率'] = round(ocf / current_liab, 4) if ocf and current_liab and current_liab != 0 else None

    # FCF
    capex = _get(cf, '购建固定资产、无形资产支付的现金')
    fcf = ocf - capex if ocf is not None and capex is not None else None
    ratios['自由现金流_FCF'] = fcf

    return ratios


def cashflow_summary(cf: pd.DataFrame, is_df: pd.DataFrame = None, bs_df: pd.DataFrame = None) -> str:
    """生成现金流量综合分析报告"""
    lines = ['【现金流量分析报告】']

    # 结构分析
    structure = analyze_cashflow_structure(cf)
    lines.append('\n一、现金流结构')
    lines.append(f'  经营活动现金流净额: {_fmt(structure.get("经营活动现金流净额"))}')
    lines.append(f'  投资活动现金流净额: {_fmt(structure.get("投资活动现金流净额"))}')
    lines.append(f'  筹资活动现金流净额: {_fmt(structure.get("筹资活动现金流净额"))}')
    lines.append(f'  类型诊断: {structure.get("现金流类型", "N/A")}')

    # FCF
    fcf_data = compute_fcf(cf)
    lines.append('\n二、自由现金流')
    lines.append(f'  OCF: {_fmt(fcf_data.get("经营活动现金流净额_OCF"))}')
    lines.append(f'  CapEx: {_fmt(fcf_data.get("资本支出_CapEx"))}')
    lines.append(f'  FCF: {_fmt(fcf_data.get("自由现金流_FCF"))}')
    if fcf_data.get('FCF占OCF比例'):
        lines.append(f'  FCF/OCF: {fcf_data["FCF占OCF比例"]}%')

    # 质量比率
    cr = cashflow_ratios(cf, is_df, bs_df)
    lines.append('\n三、现金流质量')
    for k, v in cr.items():
        if v is not None:
            lines.append(f'  {k}: {round(v, 4)}')

    return '\n'.join(lines)


def _fmt(val):
    if val is None:
        return 'N/A'
    if abs(val) >= 1e8:
        return f'{val / 1e8:.2f} 亿'
    elif abs(val) >= 1e4:
        return f'{val / 1e4:.2f} 万'
    return f'{val:.2f}'
