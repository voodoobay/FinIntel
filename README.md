# FinIntel — AI 财务智能分析 Agent

基于 LLM Tool-Use 模式的智能财务分析系统。上传财务报表（资产负债表 / 利润表 / 现金流量表），Agent 自动完成**比率计算、趋势分析、异常检测、财务预测**等全套分析流程，并生成专业解读报告。

## 技术栈

`Python` `Claude API` `Tool-Use / Function Calling` `pandas` `scikit-learn` `Streamlit`

## 能力矩阵

| 分析类别 | 具体能力 | 对应工具 |
|---------|---------|---------|
| 财务比率 | 盈利能力 / 偿债能力 / 营运能力 / 发展能力 / 每股指标 | 自动计算全部关键比率 + 行业基准对比 |
| 杜邦分析 | ROE 分解为 销售净利率 × 总资产周转率 × 权益乘数 | 驱动因素识别 |
| 趋势分析 | 线性回归 + CAGR 复合增长率 + 显著性检验 | P 值判断趋势是否显著 |
| 结构分析 | 共同比分析 / 垂直分析 / 水平分析（环比变动） | 跨期可比 / 同行可比 |
| 异常检测 | Z-Score / IQR / 同比异常 / Benford 定律 | 审计与风控场景 |
| 财务预测 | 线性回归预测 + 95% 置信区间 | 营收 / 利润 / 费用预测 |
| 描述统计 | 均值 / 中位数 / 标准差 / 偏度 / 峰度 / 变异系数 | 数据质量评估 |

## 项目结构

```
FinIntel/
├── src/
│   ├── main.py                      # CLI 交互入口
│   ├── agent/
│   │   ├── financial_agent.py       # Agent 核心（Tool-Use 循环）
│   │   └── tools.py                 # 12 个工具定义 + 处理器
│   ├── analysis/
│   │   ├── ratios.py                # 财务比率引擎
│   │   ├── statistics.py            # 统计分析 + 杜邦分解
│   │   ├── anomaly.py               # 异常检测（Z-Score / IQR / Benford）
│   │   ├── forecasting.py           # 时间序列预测
│   │   └── financial_statements.py  # 结构分析（共同比 / 水平 / 垂直）
│   ├── data/
│   │   └── loader.py                # 数据加载（CSV / Excel 自动识别）
│   └── ui/
│       └── streamlit_app.py         # Web UI
├── sample_data/                     # 示例财务数据
├── tests/                           # 单元测试
├── requirements.txt
└── .env.example                     # 配置模板
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

支持 Anthropic 官方 API 或兼容接口（如 DeepSeek），配置示例：

```env
ANTHROPIC_API_KEY=sk-ant-api03-xxx
CLAUDE_MODEL=claude-sonnet-4-6
# 可选：自定义 API 端点
# ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
```

### 3. 启动

**命令行模式：**
```bash
python src/main.py
```

进入交互对话，输入分析请求即可：
```
分析 sample_data/利润表_示例.csv
ROE 下降的原因是什么？
预测明年营收，给置信区间
```

**Web 界面：**
```bash
streamlit run src/ui/streamlit_app.py
```

浏览器打开 `http://localhost:8501`，拖拽上传 CSV/Excel 或使用示例数据。

### 4. 运行测试

```bash
python tests/test_modules.py
```

## 示例输出

```
📊 利润表全面分析报告（2021-2024）

一、盈利能力快览
  销售毛利率: 从 30.07% 降至 27.96%  ⚠️ 四年累计下滑 2.11 个百分点
  净资产收益率: ROE 稳定在 22%-24% 区间

二、趋势诊断
  营业收入 CAGR: 11.57%  R²=0.952  P=0.024  显著上升
  营业成本 CAGR: 12.69%  ⚠️ 成本增速持续快于收入

三、异常排查
  Benford 检验: 首位数分布偏离 Benford 定律，建议审计关注
  投资收益波动: 2024 年由盈转亏 (-127%)

四、预测
  2025 年净利润预测: 15.68 亿  (95%CI: 15.08 ~ 16.28)

五、管理建议
  🔴 毛利率下滑 → 启动成本专项审计
  🟡 信用减值扩大 → 加强账龄管理
  🟡 Benford 异常 → 审查数据录入流程
```

## 设计理念

- **LLM 做大脑，不做计算器**：财务计算由确定性 Python 函数完成，LLM 负责语义理解、工具编排、结果解读
- **12 个工具，一个循环**：Agent 自主决定调用哪些工具、以什么顺序调、如何组合结果
- **ERP 视角**：分析结论对接管理动作，指明系统改进方向

## License

MIT
