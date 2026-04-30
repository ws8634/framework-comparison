import asyncio
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from agent_compare.langchain_impl import (
    LangChainConcurrentScenario,
    LangChainMultiInstanceScenario,
    LangChainMultiAgentScenario,
)
from agent_compare.agentscope_impl import (
    AgentScopeConcurrentScenario,
    AgentScopeMultiInstanceScenario,
    AgentScopeMultiAgentScenario,
)


class Framework(str, Enum):
    LANGCHAIN = "langchain"
    AGENTSCOPE = "agentscope"
    BOTH = "both"


class ScenarioType(str, Enum):
    CONCURRENT = "concurrent"
    MULTI_INSTANCE = "multi_instance"
    MULTI_AGENT = "multi_agent"
    ALL = "all"


app = typer.Typer(
    name="agent-compare",
    help="LangChain vs AgentScope 框架选型对比实验工具",
)
console = Console()


def get_artifacts_dir() -> Path:
    artifacts_dir = Path.cwd() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def save_result(
    result: Any,
    framework: str,
    scenario: str,
    timestamp: str,
) -> Path:
    artifacts_dir = get_artifacts_dir()
    filename = f"{framework}_{scenario}_{timestamp}.json"
    filepath = artifacts_dir / filename
    
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
    
    return filepath


@dataclass
class SummaryStats:
    framework: str
    scenario: str
    total_duration_ms: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    instances_managed: int = 0
    messages_exchanged: int = 0
    success_rate: float = 100.0
    output_path: str = ""


async def run_concurrent_scenario(
    framework: Framework,
    num_tasks: int,
    max_concurrency: Optional[int],
) -> Dict[str, Any]:
    results = {}
    
    if framework in [Framework.LANGCHAIN, Framework.BOTH]:
        console.print(f"[cyan]运行 LangChain 并发调度场景 (tasks={num_tasks}, concurrency={max_concurrency})...[/cyan]")
        scenario = LangChainConcurrentScenario()
        result = await scenario.run(
            num_tasks=num_tasks,
            max_concurrency=max_concurrency,
        )
        results["langchain"] = result
    
    if framework in [Framework.AGENTSCOPE, Framework.BOTH]:
        console.print(f"[cyan]运行 AgentScope 并发调度场景 (tasks={num_tasks}, concurrency={max_concurrency})...[/cyan]")
        scenario = AgentScopeConcurrentScenario()
        result = await scenario.run(
            num_tasks=num_tasks,
            max_concurrency=max_concurrency,
        )
        results["agentscope"] = result
    
    return results


async def run_multi_instance_scenario(
    framework: Framework,
    num_instances: int,
    prompts_per_instance: int,
) -> Dict[str, Any]:
    results = {}
    
    if framework in [Framework.LANGCHAIN, Framework.BOTH]:
        console.print(f"[cyan]运行 LangChain 多实例管理场景 (instances={num_instances}, prompts={prompts_per_instance})...[/cyan]")
        scenario = LangChainMultiInstanceScenario()
        result = await scenario.run(
            num_instances=num_instances,
            prompts_per_instance=prompts_per_instance,
        )
        results["langchain"] = result
    
    if framework in [Framework.AGENTSCOPE, Framework.BOTH]:
        console.print(f"[cyan]运行 AgentScope 多实例管理场景 (instances={num_instances}, prompts={prompts_per_instance})...[/cyan]")
        scenario = AgentScopeMultiInstanceScenario()
        result = await scenario.run(
            num_instances=num_instances,
            prompts_per_instance=prompts_per_instance,
        )
        results["agentscope"] = result
    
    return results


async def run_multi_agent_scenario(
    framework: Framework,
    fail_subtask: Optional[str],
) -> Dict[str, Any]:
    results = {}
    
    if framework in [Framework.LANGCHAIN, Framework.BOTH]:
        console.print(f"[cyan]运行 LangChain 多智能体协作场景 (fail_subtask={fail_subtask})...[/cyan]")
        scenario = LangChainMultiAgentScenario()
        result = await scenario.run(
            fail_subtask=fail_subtask,
        )
        results["langchain"] = result
    
    if framework in [Framework.AGENTSCOPE, Framework.BOTH]:
        console.print(f"[cyan]运行 AgentScope 多智能体协作场景 (fail_subtask={fail_subtask})...[/cyan]")
        scenario = AgentScopeMultiAgentScenario()
        result = await scenario.run(
            fail_subtask=fail_subtask,
        )
        results["agentscope"] = result
    
    return results


def print_summary_table(summaries: List[SummaryStats]) -> None:
    table = Table(title="实验结果摘要")
    
    table.add_column("Framework", style="cyan")
    table.add_column("Scenario", style="green")
    table.add_column("Duration (ms)", style="yellow")
    table.add_column("Completed", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Success Rate", style="magenta")
    
    for summary in summaries:
        table.add_row(
            summary.framework,
            summary.scenario,
            f"{summary.total_duration_ms:.2f}",
            str(summary.tasks_completed),
            str(summary.tasks_failed),
            f"{summary.success_rate:.1f}%",
        )
    
    console.print(table)


def generate_aggregate_summary(
    all_results: Dict[str, Dict[str, Any]],
    output_paths: Dict[str, Dict[str, str]],
    timestamp: str,
) -> Dict[str, Any]:
    aggregate = {
        "timestamp": timestamp,
        "summary": {},
        "results": {},
        "output_paths": output_paths,
    }
    
    for scenario_name, framework_results in all_results.items():
        scenario_summary = {}
        for framework_name, result in framework_results.items():
            result_dict = result.to_dict() if hasattr(result, "to_dict") else result
            
            if scenario_name == "concurrent":
                scenario_summary[framework_name] = {
                    "total_tasks": result_dict.get("total_tasks", 0),
                    "tasks_completed": result_dict.get("tasks_completed", 0),
                    "tasks_failed": result_dict.get("tasks_failed", 0),
                    "total_duration_ms": result_dict.get("total_duration_ms", 0),
                    "actual_concurrency": result_dict.get("actual_concurrency", 0),
                }
            elif scenario_name == "multi_instance":
                scenario_summary[framework_name] = {
                    "total_instances": result_dict.get("total_instances", 0),
                    "instances_isolated": result_dict.get("instances_isolated", False),
                    "total_duration_ms": result_dict.get("total_duration_ms", 0),
                }
            elif scenario_name == "multi_agent":
                scenario_summary[framework_name] = {
                    "success": result_dict.get("success", False),
                    "errors_captured": result_dict.get("errors_captured", []),
                    "messages_exchanged": len(result_dict.get("message_logs", [])),
                    "total_duration_ms": result_dict.get("total_duration_ms", 0),
                }
        
        aggregate["summary"][scenario_name] = scenario_summary
        aggregate["results"][scenario_name] = {
            fw: (r.to_dict() if hasattr(r, "to_dict") else r)
            for fw, r in framework_results.items()
        }
    
    return aggregate


@app.command()
def run(
    framework: Framework = typer.Option(
        Framework.BOTH,
        "--framework",
        "-f",
        help="要运行的框架 (langchain, agentscope, 或 both)",
        case_sensitive=False,
    ),
    scenario: ScenarioType = typer.Option(
        ScenarioType.ALL,
        "--scenario",
        "-s",
        help="要运行的场景 (concurrent, multi_instance, multi_agent, 或 all)",
        case_sensitive=False,
    ),
    num_tasks: int = typer.Option(
        10,
        "--num-tasks",
        "-n",
        help="并发调度场景的任务数量",
    ),
    max_concurrency: Optional[int] = typer.Option(
        5,
        "--max-concurrency",
        "-c",
        help="最大并发数 (None 表示不限制)",
    ),
    num_instances: int = typer.Option(
        5,
        "--num-instances",
        "-i",
        help="多实例管理场景的实例数量",
    ),
    prompts_per_instance: int = typer.Option(
        3,
        "--prompts-per-instance",
        "-p",
        help="每个实例的交互轮数",
    ),
    fail_subtask: Optional[str] = typer.Option(
        None,
        "--fail-subtask",
        help="指定要模拟失败的子任务ID (如 st-002)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="输出目录 (默认为 ./artifacts)",
    ),
) -> None:
    """运行框架对比实验
    
    支持并发调度、多实例管理、多智能体协作三个场景。
    """
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    if output_dir:
        artifacts_dir = Path(output_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    else:
        artifacts_dir = get_artifacts_dir()
    
    console.print("[bold green]开始运行 Agent 框架对比实验[/bold green]")
    console.print(f"[dim]输出目录: {artifacts_dir}[/dim]")
    console.print()
    
    all_results = {}
    output_paths = {}
    summaries = []
    
    async def run_all():
        nonlocal all_results, output_paths, summaries
        
        if scenario in [ScenarioType.CONCURRENT, ScenarioType.ALL]:
            results = await run_concurrent_scenario(
                framework=framework,
                num_tasks=num_tasks,
                max_concurrency=max_concurrency,
            )
            all_results["concurrent"] = results
            output_paths["concurrent"] = {}
            
            for fw_name, result in results.items():
                filepath = save_result(result, fw_name, "concurrent", timestamp)
                output_paths["concurrent"][fw_name] = str(filepath)
                
                result_dict = result.to_dict()
                summary = SummaryStats(
                    framework=fw_name,
                    scenario="concurrent",
                    total_duration_ms=result_dict.get("total_duration_ms", 0),
                    tasks_completed=result_dict.get("tasks_completed", 0),
                    tasks_failed=result_dict.get("tasks_failed", 0),
                    success_rate=(
                        result_dict.get("tasks_completed", 0) /
                        max(1, result_dict.get("total_tasks", 1)) * 100
                    ),
                    output_path=str(filepath),
                )
                summaries.append(summary)
            
            console.print()
        
        if scenario in [ScenarioType.MULTI_INSTANCE, ScenarioType.ALL]:
            results = await run_multi_instance_scenario(
                framework=framework,
                num_instances=num_instances,
                prompts_per_instance=prompts_per_instance,
            )
            all_results["multi_instance"] = results
            output_paths["multi_instance"] = {}
            
            for fw_name, result in results.items():
                filepath = save_result(result, fw_name, "multi_instance", timestamp)
                output_paths["multi_instance"][fw_name] = str(filepath)
                
                result_dict = result.to_dict()
                summary = SummaryStats(
                    framework=fw_name,
                    scenario="multi_instance",
                    total_duration_ms=result_dict.get("total_duration_ms", 0),
                    tasks_completed=result_dict.get("total_instances", 0),
                    tasks_failed=0,
                    instances_managed=result_dict.get("total_instances", 0),
                    success_rate=100.0,
                    output_path=str(filepath),
                )
                summaries.append(summary)
            
            console.print()
        
        if scenario in [ScenarioType.MULTI_AGENT, ScenarioType.ALL]:
            results = await run_multi_agent_scenario(
                framework=framework,
                fail_subtask=fail_subtask,
            )
            all_results["multi_agent"] = results
            output_paths["multi_agent"] = {}
            
            for fw_name, result in results.items():
                filepath = save_result(result, fw_name, "multi_agent", timestamp)
                output_paths["multi_agent"][fw_name] = str(filepath)
                
                result_dict = result.to_dict()
                success = result_dict.get("success", False)
                errors = result_dict.get("errors_captured", [])
                summary = SummaryStats(
                    framework=fw_name,
                    scenario="multi_agent",
                    total_duration_ms=result_dict.get("total_duration_ms", 0),
                    tasks_completed=1 if success else 0,
                    tasks_failed=0 if success else 1,
                    messages_exchanged=len(result_dict.get("message_logs", [])),
                    success_rate=100.0 if success else 0.0,
                    output_path=str(filepath),
                )
                summaries.append(summary)
            
            console.print()
    
    asyncio.run(run_all())
    
    aggregate = generate_aggregate_summary(all_results, output_paths, timestamp)
    aggregate_path = artifacts_dir / f"aggregate_{timestamp}.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False, default=str)
    
    console.print("[bold green]实验完成！[/bold green]")
    console.print()
    
    print_summary_table(summaries)
    
    console.print()
    console.print(f"[bold]汇总输出:[/bold] {aggregate_path}")
    console.print("[dim]详细结果请查看 artifacts/ 目录下的 JSON 文件[/dim]")


@app.command()
def version() -> None:
    """显示版本信息"""
    from agent_compare import __version__
    console.print(f"[cyan]agent-compare v{__version__}[/cyan]")


if __name__ == "__main__":
    app()
