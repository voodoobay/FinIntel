"""
Agent 工具定义 - 将分析模块封装为 Claude API 可调用的 tools
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.data.loader import load_financial_data, load_transactions, get_summary
from src.analysis.ratios import compute_all_ratios, interpret_ratios
from src.analysis.statistics import (
    descriptive_stats, trend_analysis, correlation_matrix,
    dupont_analysis, variance_analysis,
)
from src.analysis.anomaly import (
    zscore_anomalies, iqr_anomalies, yoy_anomalies, benford_test,
)
from src.analysis.forecasting import linear_forecast, forecast_summary, exponential_smoothing
from src.analysis.financial_statements import (
    common_size_bs, common_size_is, horizontal_analysis, vertical_analysis_bs, vertical_analysis_is,
)


# 全局数据缓存
_data_cache: Dict[str, pd.DataFrame] = {}


def _cache_key(filename: str) -> str:
    return filename


def tool_load_financials(file_path: str) -> str:
    """
    加载财务报表文件（CSV 或 Excel），自动识别报表类型。
    文件应包含：第一列为科目名称，后续列为各期数据。
    """
    try:
        df = load_financial_data(file_path)
        _data_cache[_cache_key(file_path)] = df
        return f'已加载: {file_path}\n{get_summary(df)}'
    except Exception as e:
        return f'加载失败: {str(e)}'


def tool_load_transactions(file_path: str) -> str:
    """加载交易流水数据"""
    try:
        df = load_transactions(file_path)
        _data_cache[_cache_key(file_path)] = df
        cols = list(df.columns)
        return f'已加载交易数据: {file_path}\n行数: {len(df)}, 列: {cols}'
    except Exception as e:
        return f'加载失败: {str(e)}'


def tool_financial_ratios(file_path: str) -> str:
    """计算全部关键财务比率并给出解读"""
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    df = _data_cache[file_path]
    ratios = compute_all_ratios(df)
    return interpret_ratios(ratios)


def tool_dupont_analysis(file_path: str) -> str:
    """杜邦分析：分解 ROE 为销售净利率×总资产周转率×权益乘数"""
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    result = dupont_analysis(_data_cache[file_path])
    lines = ['【杜邦分析】']
    for k, v in result.items():
        lines.append(f'{k}: {v}')
    return '\n'.join(lines)


def tool_horizontal_analysis(file_path: str) -> str:
    """水平分析：各科目环比变动率和变动额"""
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    df = _data_cache[file_path]
    result = horizontal_analysis(df)
    return result.to_string()


def tool_vertical_analysis(file_path: str) -> str:
    """垂直分析：各项占总资产/营业收入的比重"""
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    df = _data_cache[file_path]
    from src.data.loader import detect_statement_type
    stype = detect_statement_type(df)
    if stype == '利润表':
        result = vertical_analysis_is(df)
    else:
        result = vertical_analysis_bs(df)
    return result.to_string()


def tool_trend_analysis(file_path: str, item_name: str) -> str:
    """
    对指定科目做趋势分析（线性回归 + CAGR）。
    参数 item_name: 科目名称，如 '营业收入'、'净利润'
    """
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    df = _data_cache[file_path]
    if item_name not in df.index:
        similar = [i for i in df.index if item_name in str(i)]
        hint = f' 相似科目: {similar}' if similar else ''
        return f'未找到科目 "{item_name}"。{hint}'
    series = df.loc[item_name]
    result = trend_analysis(series)
    lines = [f'【{item_name} 趋势分析】']
    for k, v in result.items():
        lines.append(f'{k}: {v}')
    return '\n'.join(lines)


def tool_descriptive_stats(file_path: str, item_name: str) -> str:
    """对指定科目做描述性统计（均值、标准差、偏度、峰度等）"""
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    df = _data_cache[file_path]
    if item_name not in df.index:
        return f'未找到科目 "{item_name}"'
    result = descriptive_stats(df.loc[item_name])
    lines = [f'【{item_name} 描述性统计】']
    for k, v in result.items():
        lines.append(f'{k}: {v}')
    return '\n'.join(lines)


def tool_anomaly_detect(file_path: str, item_name: str = None, method: str = 'zscore') -> str:
    """
    异常检测。method 可选: zscore / iqr / yoy / benford
    - zscore/iqr: 需指定 item_name
    - benford: 无需 item_name，检测整个数据集的首位数字分布
    """
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    df = _data_cache[file_path]

    if method == 'benford':
        all_vals = pd.Series(df.values.flatten()).dropna()
        result = benford_test(all_vals)
        return json.dumps(result, ensure_ascii=False, indent=2)

    if item_name is None:
        return 'zscore/iqr 方法需要指定 item_name 参数'
    if item_name not in df.index:
        return f'未找到科目 "{item_name}"'

    series = df.loc[item_name]
    if method == 'zscore':
        anomalies = zscore_anomalies(series)
    elif method == 'iqr':
        anomalies = iqr_anomalies(series)
    elif method == 'yoy':
        if df.shape[1] < 2:
            return '同比分析需要至少两期数据'
        anomalies = yoy_anomalies(series.iloc[:, -1] if hasattr(series, 'iloc') else series, series.iloc[:, -2])
    else:
        return f'不支持的方法: {method}'

    if anomalies.empty:
        return f'未检测到异常 ({method} 方法)'
    return f'【{item_name} 异常检测 ({method})】\n{anomalies.to_string()}'


def tool_forecast(file_path: str, item_name: str, periods: int = 3) -> str:
    """对指定科目做线性预测"""
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    df = _data_cache[file_path]
    if item_name not in df.index:
        return f'未找到科目 "{item_name}"'
    return forecast_summary(df.loc[item_name], periods, item_name)


def tool_correlation(file_path: str) -> str:
    """计算财务指标间的相关系数矩阵"""
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    corr = correlation_matrix(_data_cache[file_path].T)
    if corr.empty:
        return '无法计算相关性（需要至少两个数值列）'
    return corr.to_string()


def tool_common_size(file_path: str) -> str:
    """共同比分析"""
    if file_path not in _data_cache:
        return '请先使用 load_financials 加载数据'
    df = _data_cache[file_path]
    from src.data.loader import detect_statement_type
    stype = detect_statement_type(df)
    if stype == '利润表':
        result = common_size_is(df)
    else:
        result = common_size_bs(df)
    return result.to_string()


# ---- 工具注册表 ----

TOOL_DEFINITIONS = [
    {
        "name": "load_financials",
        "description": "加载财务报表文件（CSV或Excel）。文件格式：第一列为科目名称，后续列为各期数值。加载后可进行比率分析、趋势分析、异常检测等操作。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "财务数据文件的路径"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "load_transactions",
        "description": "加载交易流水数据（CSV或Excel），用于明细级别的分析。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "交易数据文件路径"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "financial_ratios",
        "description": "计算全部关键财务比率：盈利能力（毛利率、净利率、ROA、ROE）、偿债能力（流动比率、速动比率、资产负债率）、营运能力（各类周转率）、发展能力（增长率）。自动给出行业基准对比解读。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "dupont_analysis",
        "description": "杜邦分析：将ROE分解为销售净利率×总资产周转率×权益乘数，帮助理解ROE的驱动因素。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "horizontal_analysis",
        "description": "水平分析（环比分析）：计算各科目每期相对于前一期的变动率和变动额，用于识别异常变动。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "vertical_analysis",
        "description": "垂直分析（结构分析）：资产负债表以总资产为100%计算各科目占比；利润表以营业收入为100%。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "trend_analysis",
        "description": "对指定财务科目做趋势分析：线性回归拟合 + 复合年增长率CAGR，判断趋势是否显著。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"},
                "item_name": {"type": "string", "description": "科目名称，如'营业收入'、'净利润'、'资产总计'"}
            },
            "required": ["file_path", "item_name"]
        }
    },
    {
        "name": "descriptive_stats",
        "description": "对指定财务科目做描述性统计：均值、中位数、标准差、偏度、峰度、变异系数等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"},
                "item_name": {"type": "string", "description": "科目名称"}
            },
            "required": ["file_path", "item_name"]
        }
    },
    {
        "name": "anomaly_detect",
        "description": "异常检测。支持四种方法：zscore（Z分数法，适合正态分布）、iqr（四分位距法，稳健）、yoy（同比异常检测）、benford（Benford定律检验，用于审计）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"},
                "item_name": {"type": "string", "description": "科目名称（benford方法不需要）"},
                "method": {"type": "string", "enum": ["zscore", "iqr", "yoy", "benford"], "description": "异常检测方法：zscore/iqr/yoy/benford"}
            },
            "required": ["file_path", "method"]
        }
    },
    {
        "name": "forecast",
        "description": "对指定财务指标做线性回归预测，输出预测值及95%置信区间。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"},
                "item_name": {"type": "string", "description": "要预测的科目名称"},
                "periods": {"type": "integer", "description": "预测期数，默认3期"}
            },
            "required": ["file_path", "item_name"]
        }
    },
    {
        "name": "correlation",
        "description": "计算各财务指标之间的Pearson相关系数矩阵，用于发现指标间的关联关系。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "common_size",
        "description": "共同比分析：将报表各科目转化为百分比形式，便于跨公司、跨期对比。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "已加载的财务数据文件路径"}
            },
            "required": ["file_path"]
        }
    },
]


TOOL_HANDLERS = {
    "load_financials": tool_load_financials,
    "load_transactions": tool_load_transactions,
    "financial_ratios": tool_financial_ratios,
    "dupont_analysis": tool_dupont_analysis,
    "horizontal_analysis": tool_horizontal_analysis,
    "vertical_analysis": tool_vertical_analysis,
    "trend_analysis": tool_trend_analysis,
    "descriptive_stats": tool_descriptive_stats,
    "anomaly_detect": tool_anomaly_detect,
    "forecast": tool_forecast,
    "correlation": tool_correlation,
    "common_size": tool_common_size,
}


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """执行工具调用并返回结果"""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f'未知工具: {tool_name}'
    try:
        return handler(**tool_input)
    except Exception as e:
        return f'工具执行出错 ({tool_name}): {str(e)}'
