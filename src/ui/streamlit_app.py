"""
财智分析 Agent - Streamlit Web UI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd

from src.agent.financial_agent import FinancialAgent
from src.data.loader import load_financial_data, get_summary
from src.analysis.ratios import compute_all_ratios, interpret_ratios
from src.analysis.statistics import trend_analysis, dupont_analysis
from src.analysis.anomaly import zscore_anomalies, benford_test

st.set_page_config(
    page_title='财智分析 Agent',
    page_icon='📊',
    layout='wide',
)

# ---- 初始化 Session State ----
if 'agent' not in st.session_state:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        try:
            st.session_state.agent = FinancialAgent(api_key=api_key)
            st.session_state.agent_ready = True
        except Exception:
            st.session_state.agent_ready = False
    else:
        st.session_state.agent_ready = False

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df_cache' not in st.session_state:
    st.session_state.df_cache = {}
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# ---- 侧边栏 ----
with st.sidebar:
    st.title('📊 财智分析 Agent')
    st.caption('FinIntel: AI-Powered Financial Analysis')

    if not st.session_state.agent_ready:
        api_key_input = st.text_input('Anthropic API Key', type='password')
        if api_key_input:
            try:
                st.session_state.agent = FinancialAgent(api_key=api_key_input)
                st.session_state.agent_ready = True
                st.success('已连接')
                st.rerun()
            except Exception as e:
                st.error(f'连接失败: {e}')
    else:
        st.success('Agent 就绪')

    st.divider()

    # 文件上传
    st.subheader('📁 上传财务数据')
    uploaded = st.file_uploader(
        '支持 CSV / Excel',
        type=['csv', 'xlsx', 'xls'],
        key='file_uploader',
    )

    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded, index_col=0)
            else:
                df = pd.read_excel(uploaded, index_col=0)

            df = df.dropna(how='all').dropna(axis=1, how='all')
            file_key = uploaded.name
            st.session_state.df_cache[file_key] = df
            st.session_state.current_file = file_key
            st.success(f'已加载: {file_key}')
        except Exception as e:
            st.error(f'加载失败: {e}')

    # 示例数据
    st.divider()
    st.subheader('📦 加载示例数据')
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'sample_data')
    if os.path.isdir(sample_dir):
        samples = os.listdir(sample_dir)
        for s in samples:
            if st.button(f'📄 {s}', key=f'sample_{s}'):
                fp = os.path.join(sample_dir, s)
                df = load_financial_data(fp)
                st.session_state.df_cache[s] = df
                st.session_state.current_file = s
                st.success(f'已加载: {s}')
                st.rerun()

    # 重置
    if st.button('🔄 重置对话'):
        if st.session_state.agent_ready:
            st.session_state.agent.reset()
        st.session_state.messages = []
        st.rerun()

# ---- 主区域 ----
st.title('财智分析 Agent')

# 显示当前数据概况
current = st.session_state.current_file
if current and current in st.session_state.df_cache:
    df = st.session_state.df_cache[current]
    with st.expander(f'📋 当前数据: {current}  ({df.shape[0]}行 × {df.shape[1]}列)', expanded=False):
        st.dataframe(df, use_container_width=True)
        st.caption(get_summary(df))

# ---- 快捷分析卡片 ----
if current and current in st.session_state.df_cache:
    st.subheader('⚡ 快捷分析')
    cols = st.columns(5)
    df = st.session_state.df_cache[current]

    with cols[0]:
        if st.button('📊 财务比率', use_container_width=True):
            ratios = compute_all_ratios(df)
            interpretation = interpret_ratios(ratios)
            st.session_state.messages.append({'role': 'assistant', 'content': f'## 财务比率分析\n\n{interpretation}'})
            st.rerun()

    with cols[1]:
        if st.button('🔍 杜邦分析', use_container_width=True):
            result = dupont_analysis(df)
            lines = '## 杜邦分析\n\n'
            for k, v in result.items():
                lines += f'- **{k}**: {v}\n'
            st.session_state.messages.append({'role': 'assistant', 'content': lines})
            st.rerun()

    with cols[2]:
        if st.button('📈 趋势分析', use_container_width=True):
            if df.shape[1] >= 2:
                first_item = df.index[0]
                result = trend_analysis(df.loc[first_item])
                lines = f'## {first_item} 趋势分析\n\n'
                for k, v in result.items():
                    lines += f'- **{k}**: {v}\n'
                st.session_state.messages.append({'role': 'assistant', 'content': lines})
            st.rerun()

    with cols[3]:
        if st.button('⚠️ 异常检测', use_container_width=True):
            if df.shape[1] >= 2:
                first_item = df.index[0]
                anomalies = zscore_anomalies(df.loc[first_item])
                if anomalies.empty:
                    lines = f'## {first_item} 异常检测\n\n未检测到异常'
                else:
                    lines = f'## {first_item} 异常检测\n\n{anomalies.to_markdown()}'
                st.session_state.messages.append({'role': 'assistant', 'content': lines})
            st.rerun()

    with cols[4]:
        if st.button('🔢 Benford检验', use_container_width=True):
            vals = pd.Series(df.values.flatten()).dropna()
            result = benford_test(vals)
            lines = '## Benford 定律检验\n\n'
            for k, v in result.items():
                lines += f'- **{k}**: {v}\n'
            st.session_state.messages.append({'role': 'assistant', 'content': lines})
            st.rerun()

# ---- 对话区域 ----
st.subheader('💬 对话')

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ---- 输入 ----
prompt = st.chat_input('输入分析问题，或引用当前数据提问...')

if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    if st.session_state.agent_ready and current:
        with st.chat_message('assistant'):
            with st.spinner('分析中...'):
                response = st.session_state.agent.chat(
                    f'当前已加载数据文件: {current}\n\n用户问题: {prompt}'
                )
            st.markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
    elif not st.session_state.agent_ready:
        st.error('请先在侧边栏配置 API Key')
    else:
        st.warning('请先上传数据文件或加载示例数据')
