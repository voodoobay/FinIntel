"""
财务分析 Agent 主控 - Financial Analysis Agent
基于 Claude API，集成多个财务分析工具
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import anthropic
from dotenv import load_dotenv

from src.agent.tools import TOOL_DEFINITIONS, execute_tool

# 确保从项目根目录加载 .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / '.env')

SYSTEM_PROMPT = """你是一位资深的财务分析师 Agent，拥有会计学和统计学双重专业背景。

你的能力范围：
1. 财务报表分析（资产负债表、利润表、现金流量表）
2. 财务比率计算与解读（盈利能力、偿债能力、营运能力、发展能力）
3. 趋势分析与统计建模（描述性统计、回归分析、相关性分析）
4. 异常检测（Z-Score、IQR、同比异常、Benford定律）
5. 财务预测（线性回归预测、时间序列分析）
6. 杜邦分析、共同比分析、水平/垂直分析

工作原则：
- 使用工具获取数据和分析结果，不要凭空猜测数字
- 先加载数据，再做分析
- 解读结果时，结合行业背景给出专业判断，而非仅报告数字
- 对异常和风险点要明确指出，并给出可能的解释
- 回复使用中文，报告结构清晰
- 如果用户没有指定具体分析，主动建议最有价值的分析方向

ERP 行业视角：
- 关注流程效率指标（周转率、资金占用天数）
- 关注数据质量和一致性（适合 ERP 系统数据治理场景）
- 分析结论应可落地到具体的管理动作或系统改进
"""


class FinancialAgent:
    """财务分析 Agent"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 10,
    ):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError('请设置 ANTHROPIC_API_KEY 环境变量或在初始化时传入 api_key')

        self.model = model or os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-6')
        self.max_turns = max_turns
        base_url = os.getenv('ANTHROPIC_BASE_URL')
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            **({'base_url': base_url} if base_url else {}),
        )
        self.messages: List[Dict[str, Any]] = []

    def chat(self, user_message: str) -> str:
        """发送消息并获取回复（自动处理工具调用循环）"""
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=self.messages,
                tools=TOOL_DEFINITIONS,
            )

            # 检查是否有工具调用
            tool_uses = [b for b in response.content if b.type == 'tool_use']

            if not tool_uses:
                # 纯文本回复，直接返回
                text_parts = [b.text for b in response.content if b.type == 'text']
                reply = '\n'.join(text_parts)
                self.messages.append({"role": "assistant", "content": response.content})
                return reply

            # 处理工具调用
            tool_results = []
            for tool_block in tool_uses:
                tool_name = tool_block.name
                tool_input = tool_block.input if isinstance(tool_block.input, dict) else {}
                print(f'  [tool] {tool_name}({tool_input})')

                result = execute_tool(tool_name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result,
                })

            # 将 assistant 消息和工具结果加入对话
            self.messages.append({"role": "assistant", "content": response.content})
            self.messages.append({"role": "user", "content": tool_results})

        return '已达到最大工具调用轮次，请简化分析请求。'

    def reset(self):
        """重置对话历史"""
        self.messages = []

    def analyze_file(self, file_path: str, question: str = None) -> str:
        """
        快捷方法：加载文件并提问。
        如果未指定 question，自动进行综合财务分析。
        """
        if question is None:
            question = f'请对 {file_path} 进行全面的财务分析，包括：'
            question += '\n1. 先加载数据'
            question += '\n2. 计算关键财务比率并解读'
            question += '\n3. 做杜邦分析'
            question += '\n4. 对主要科目做趋势分析'
            question += '\n5. 做异常检测'
            question += '\n6. 给出综合评价和建议'

        return self.chat(f'数据文件: {file_path}\n\n{question}')
