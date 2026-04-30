import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    FunctionMessage,
)
from langchain_core.tools import tool, BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent_compare.fake_llm import (
    FakeLLM,
    FakeChatMessage,
    MessageRole,
    KeywordResponseRule,
    MultiRoleRule,
    ErrorSimulatingRule,
    create_fake_llm,
)
from .adapter import LangChainFakeLLM


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"


@dataclass
class MessageLog:
    sender: str
    receiver: str
    content: str
    timestamp: float
    message_type: str = "text"


@dataclass
class SubTaskResult:
    task_id: str
    task_name: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class MultiAgentScenarioResult:
    framework: str = "langchain"
    scenario: str = "multi_agent"
    task_goal: str = ""
    message_logs: List[MessageLog] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    execution_results: List[SubTaskResult] = field(default_factory=list)
    review_conclusion: Optional[str] = None
    success: bool = False
    errors_captured: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "scenario": self.scenario,
            "task_goal": self.task_goal,
            "message_logs": [
                {
                    "sender": m.sender,
                    "receiver": m.receiver,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "message_type": m.message_type,
                }
                for m in self.message_logs
            ],
            "plan": self.plan,
            "execution_results": [
                {
                    "task_id": r.task_id,
                    "task_name": r.task_name,
                    "status": r.status,
                    "result": r.result,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in self.execution_results
            ],
            "review_conclusion": self.review_conclusion,
            "success": self.success,
            "errors_captured": self.errors_captured,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class LangChainMultiAgentScenario:
    def __init__(
        self,
        fake_llm: Optional[FakeLLM] = None,
    ):
        if fake_llm is None:
            rules = [
                KeywordResponseRule(
                    keyword_responses={
                        "plan": "I'll create a plan with 3 subtasks: data_fetch, process, validate",
                        "execute": "Executing the task now...",
                        "review": "Reviewing the results...",
                        "error": "An error occurred during processing.",
                    }
                ),
                MultiRoleRule(
                    role_responses={
                        "planner": "Plan: {subtasks}",
                        "executor": "Executed successfully",
                        "reviewer": "All tasks completed",
                    }
                ),
                ErrorSimulatingRule(
                    error_triggers=["fail", "error", "timeout"],
                    error_message="Task failed: simulated API timeout",
                    error_type="timeout",
                ),
            ]
            fake_llm = create_fake_llm(rules=rules)
        self.fake_llm = fake_llm
        self.langchain_llm = LangChainFakeLLM(fake_llm=fake_llm)
        self._message_logs: List[MessageLog] = []

    def _log_message(self, sender: str, receiver: str, content: str, message_type: str = "text") -> None:
        self._message_logs.append(
            MessageLog(
                sender=sender,
                receiver=receiver,
                content=content,
                timestamp=time.perf_counter(),
                message_type=message_type,
            )
        )

    async def _planner_agent(self, task_goal: str) -> Dict[str, Any]:
        self._log_message("user", "planner", task_goal)
        
        prompt = f"""You are a planning agent. Given the goal: {task_goal}
Please create a plan with subtasks. Return a JSON object with:
- task_goal: the original goal
- subtasks: list of objects with id, name, description
- estimated_duration_minutes: total estimate
"""
        response = await self.langchain_llm.ainvoke(prompt, agent_role="planner")
        
        plan = {
            "task_goal": task_goal,
            "subtasks": [
                {"id": "st-001", "name": "data_fetch", "description": "Fetch required data"},
                {"id": "st-002", "name": "process", "description": "Process the data"},
                {"id": "st-003", "name": "validate", "description": "Validate results"},
            ],
            "estimated_duration_minutes": 5,
            "planner_response": response,
        }
        
        self._log_message("planner", "orchestrator", str(plan))
        return plan

    async def _executor_agent(
        self,
        subtask: Dict[str, Any],
        should_fail: bool = False,
    ) -> SubTaskResult:
        task_id = subtask["id"]
        task_name = subtask["name"]
        
        self._log_message("orchestrator", "executor", f"Execute: {task_name}")
        
        start_time = time.perf_counter()
        
        try:
            if should_fail:
                prompt = f"Execute task {task_name} - this should fail due to timeout"
                response = await self.langchain_llm.ainvoke(
                    prompt,
                    agent_role="executor",
                )
                result = SubTaskResult(
                    task_id=task_id,
                    task_name=task_name,
                    status="failed",
                    error="Task failed: simulated API timeout",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )
            else:
                prompt = f"Execute task: {task_name} - {subtask['description']}"
                response = await self.langchain_llm.ainvoke(
                    prompt,
                    agent_role="executor",
                )
                result = SubTaskResult(
                    task_id=task_id,
                    task_name=task_name,
                    status="completed",
                    result={
                        "execution_response": response,
                        "output": f"Successfully executed {task_name}",
                    },
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )
            
            self._log_message(
                "executor",
                "orchestrator",
                str(result.result) if result.result else str(result.error),
            )
            return result
            
        except Exception as e:
            result = SubTaskResult(
                task_id=task_id,
                task_name=task_name,
                status="failed",
                error=str(e),
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._log_message("executor", "orchestrator", f"Error: {e}")
            return result

    async def _reviewer_agent(
        self,
        plan: Dict[str, Any],
        execution_results: List[SubTaskResult],
    ) -> str:
        self._log_message("orchestrator", "reviewer", "Review all results")
        
        success_count = sum(1 for r in execution_results if r.status == "completed")
        fail_count = sum(1 for r in execution_results if r.status == "failed")
        
        prompt = f"""Review the execution results:
- Total subtasks: {len(execution_results)}
- Completed: {success_count}
- Failed: {fail_count}

Provide a review conclusion."""
        
        response = await self.langchain_llm.ainvoke(prompt, agent_role="reviewer")
        
        conclusion = f"""Review Complete:
- Plan executed: {plan['task_goal']}
- Subtasks completed: {success_count}/{len(execution_results)}
- Subtasks failed: {fail_count}
- Status: {"PARTIAL_SUCCESS" if fail_count > 0 else "SUCCESS"}
- Details: {response}
"""
        self._log_message("reviewer", "user", conclusion)
        return conclusion

    async def run(
        self,
        task_goal: str = "Process a batch of data through multiple stages",
        fail_subtask: Optional[str] = None,
    ) -> MultiAgentScenarioResult:
        scenario_start = time.perf_counter()
        started_at = datetime.utcnow().isoformat()
        self._message_logs.clear()

        plan = await self._planner_agent(task_goal)
        
        execution_results: List[SubTaskResult] = []
        for subtask in plan["subtasks"]:
            should_fail = fail_subtask is not None and subtask["id"] == fail_subtask
            result = await self._executor_agent(subtask, should_fail=should_fail)
            execution_results.append(result)
        
        review_conclusion = await self._reviewer_agent(plan, execution_results)
        
        success_count = sum(1 for r in execution_results if r.status == "completed")
        errors = [r.error for r in execution_results if r.status == "failed" and r.error]
        
        scenario_end = time.perf_counter()
        completed_at = datetime.utcnow().isoformat()

        return MultiAgentScenarioResult(
            framework="langchain",
            scenario="multi_agent",
            task_goal=task_goal,
            message_logs=self._message_logs.copy(),
            plan=plan,
            execution_results=execution_results,
            review_conclusion=review_conclusion,
            success=len(errors) == 0,
            errors_captured=errors,
            total_duration_ms=(scenario_end - scenario_start) * 1000,
            started_at=started_at,
            completed_at=completed_at,
        )
