#!/usr/bin/env python3
"""
财智分析 Agent - CLI 入口
FinIntel: AI-Powered Financial Analysis Agent
"""

import sys
import os

# 确保 Windows 终端正确处理 UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from src.agent.financial_agent import FinancialAgent

console = Console()


def main():
    console.print(Panel.fit(
        '[bold cyan]财智分析 Agent[/bold cyan] — AI 财务分析助手\n'
        '[dim]FinIntel: Financial Intelligence Agent[/dim]\n\n'
        '基于 Claude API  |  财务比率  |  趋势分析  |  异常检测  |  预测  |  杜邦分析',
        border_style='cyan',
    ))

    # 检查 API Key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        console.print('[red]请先设置 ANTHROPIC_API_KEY 环境变量[/red]')
        console.print('创建 .env 文件并添加: [yellow]ANTHROPIC_API_KEY=your-key-here[/yellow]')
        sys.exit(1)

    console.print('[green]✓[/green] API Key 已配置')
    console.print('[green]✓[/green] 模型: ' + os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-6'))

    try:
        agent = FinancialAgent(api_key=api_key)
    except Exception as e:
        console.print(f'[red]Agent 初始化失败: {e}[/red]')
        sys.exit(1)

    console.print('\n[dim]支持的操作:[/dim]')
    console.print('  • 加载文件后自动分析：[yellow]分析 利润表_示例.csv[/yellow]')
    console.print('  • 直接提问：[yellow]ROE 下降可能是什么原因？[/yellow]')
    console.print('  • 输入 [yellow]reset[/yellow] 重置对话，[yellow]quit[/yellow] 退出\n')

    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data')
    if os.path.isdir(sample_dir):
        samples = os.listdir(sample_dir)
        if samples:
            console.print(f'[dim]示例数据文件: {", ".join(samples)}[/dim]\n')

    while True:
        try:
            user_input = console.input('[bold cyan]你[/bold cyan]: ').strip()
        except (EOFError, KeyboardInterrupt):
            console.print('\n[dim]再见！[/dim]')
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            console.print('[dim]再见！[/dim]')
            break

        if user_input.lower() == 'reset':
            agent.reset()
            console.print('[green]对话已重置[/green]')
            continue

        with console.status('[cyan]思考中...[/cyan]'):
            try:
                response = agent.chat(user_input)
            except Exception as e:
                response = f'[red]请求失败: {e}[/red]'

        console.print(f'[bold green]Agent[/bold green]:')
        console.print(Markdown(response))
        console.print()


if __name__ == '__main__':
    main()
