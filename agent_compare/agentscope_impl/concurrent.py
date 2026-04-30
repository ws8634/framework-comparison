import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from agent_compare.fake_llm import (
    FakeLLM,
    FakeChatMessage,
    MessageRole,
    DelaySimulatingRule,
    HashBasedRule,
    create_fake_llm,
)
from .adapter import AgentScopeFakeModel


@dataclass
class TaskResult:
    task_id: str
    status: str
    started_at: float
    completed_at: float
    duration_ms: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcurrentScenarioResult:
    framework: str = "agentscope"
    scenario: str = "concurrent"
    total_tasks: int = 0
    max_concurrency: Optional[int] = None
    rate_limit: Optional[float] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_queued: int = 0
    tasks_rejected: int = 0
    total_duration_ms: float = 0.0
    avg_task_duration_ms: float = 0.0
    actual_concurrency: int = 0
    results: List[TaskResult] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "scenario": self.scenario,
            "total_tasks": self.total_tasks,
            "max_concurrency": self.max_concurrency,
            "rate_limit": self.rate_limit,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_queued": self.tasks_queued,
            "tasks_rejected": self.tasks_rejected,
            "total_duration_ms": self.total_duration_ms,
            "avg_task_duration_ms": self.avg_task_duration_ms,
            "actual_concurrency": self.actual_concurrency,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [
                {
                    "task_id": r.task_id,
                    "status": r.status,
                    "started_at": r.started_at,
                    "completed_at": r.completed_at,
                    "duration_ms": r.duration_ms,
                    "result": r.result,
                    "error": r.error,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }


class AgentScopeConcurrentScenario:
    def __init__(
        self,
        fake_llm: Optional[FakeLLM] = None,
        base_delay_ms: float = 50.0,
        jitter_ms: float = 25.0,
    ):
        self.base_delay_ms = base_delay_ms
        self.jitter_ms = jitter_ms
        if fake_llm is None:
            rules = [
                DelaySimulatingRule(base_delay_ms=base_delay_ms, jitter_ms=jitter_ms),
                HashBasedRule(),
            ]
            fake_llm = create_fake_llm(rules=rules)
        self.fake_llm = fake_llm
        self.agentscope_model = AgentScopeFakeModel(fake_llm=fake_llm)
        self._active_tasks: Set[str] = set()
        self._max_active_concurrency = 0

    async def _run_single_task(
        self,
        task_id: str,
        prompt: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs: Any,
    ) -> TaskResult:
        started_at = time.perf_counter()
        result = TaskResult(
            task_id=task_id,
            status="running",
            started_at=started_at,
            completed_at=started_at,
            duration_ms=0.0,
        )

        try:
            if semaphore:
                async with semaphore:
                    self._active_tasks.add(task_id)
                    current_active = len(self._active_tasks)
                    if current_active > self._max_active_concurrency:
                        self._max_active_concurrency = current_active
                    response = await self.agentscope_model.ainvoke(prompt)
                    self._active_tasks.discard(task_id)
            else:
                self._active_tasks.add(task_id)
                current_active = len(self._active_tasks)
                if current_active > self._max_active_concurrency:
                    self._max_active_concurrency = current_active
                response = await self.agentscope_model.ainvoke(prompt)
                self._active_tasks.discard(task_id)

            completed_at = time.perf_counter()
            result.status = "completed"
            result.completed_at = completed_at
            result.duration_ms = (completed_at - started_at) * 1000
            result.result = {
                "prompt": prompt,
                "response": response.get("content", ""),
                "llm_stats": self.fake_llm.get_stats(),
                "model_response": response,
            }
            result.metadata = {
                "queue_position": kwargs.get("queue_position"),
                "start_delay_ms": kwargs.get("start_delay_ms", 0),
            }

        except Exception as e:
            completed_at = time.perf_counter()
            result.status = "failed"
            result.completed_at = completed_at
            result.duration_ms = (completed_at - started_at) * 1000
            result.error = str(e)
            self._active_tasks.discard(task_id)

        return result

    async def run(
        self,
        num_tasks: int = 10,
        max_concurrency: Optional[int] = None,
        rate_limit: Optional[float] = None,
        prompts: Optional[List[str]] = None,
    ) -> ConcurrentScenarioResult:
        scenario_start = time.perf_counter()
        started_at = datetime.utcnow().isoformat()

        if prompts is None:
            prompts = [f"Task prompt #{i}: Please process this request" for i in range(num_tasks)]
        else:
            num_tasks = len(prompts)

        self._max_active_concurrency = 0
        self._active_tasks.clear()

        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        tasks = []
        for i, prompt in enumerate(prompts):
            task_id = f"task-{i:04d}"
            task = self._run_single_task(
                task_id=task_id,
                prompt=prompt,
                semaphore=semaphore,
                queue_position=i,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=False)

        scenario_end = time.perf_counter()
        completed_at = datetime.utcnow().isoformat()

        tasks_completed = sum(1 for r in results if r.status == "completed")
        tasks_failed = sum(1 for r in results if r.status == "failed")
        total_duration_ms = (scenario_end - scenario_start) * 1000

        avg_duration_ms = 0.0
        if tasks_completed > 0:
            completed_durations = [r.duration_ms for r in results if r.status == "completed"]
            avg_duration_ms = sum(completed_durations) / len(completed_durations)

        return ConcurrentScenarioResult(
            framework="agentscope",
            scenario="concurrent",
            total_tasks=num_tasks,
            max_concurrency=max_concurrency,
            rate_limit=rate_limit,
            tasks_completed=tasks_completed,
            tasks_failed=tasks_failed,
            tasks_queued=0,
            tasks_rejected=0,
            total_duration_ms=total_duration_ms,
            avg_task_duration_ms=avg_duration_ms,
            actual_concurrency=self._max_active_concurrency,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
        )
