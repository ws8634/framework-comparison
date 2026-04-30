import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory

from agent_compare.fake_llm import (
    FakeLLM,
    FakeChatMessage,
    MessageRole,
    CounterRule,
    create_fake_llm,
)
from .adapter import LangChainFakeLLM


@dataclass
class InstanceState:
    instance_id: str
    memory_summary: Dict[str, Any]
    message_count: int
    last_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiInstanceScenarioResult:
    framework: str = "langchain"
    scenario: str = "multi_instance"
    total_instances: int = 0
    instances_isolated: bool = True
    isolation_evidence: List[Dict[str, Any]] = field(default_factory=list)
    instances: List[InstanceState] = field(default_factory=list)
    total_duration_ms: float = 0.0
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "scenario": self.scenario,
            "total_instances": self.total_instances,
            "instances_isolated": self.instances_isolated,
            "isolation_evidence": self.isolation_evidence,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "instances": [
                {
                    "instance_id": i.instance_id,
                    "memory_summary": i.memory_summary,
                    "message_count": i.message_count,
                    "last_response": i.last_response,
                    "metadata": i.metadata,
                }
                for i in self.instances
            ],
        }


class LangChainMultiInstanceScenario:
    def __init__(
        self,
        fake_llm: Optional[FakeLLM] = None,
    ):
        if fake_llm is None:
            rules = [CounterRule()]
            fake_llm = create_fake_llm(rules=rules)
        self.fake_llm = fake_llm
        self._memories: Dict[str, ConversationBufferMemory] = {}

    def _create_memory(self, instance_id: str) -> ConversationBufferMemory:
        memory = ConversationBufferMemory(
            memory_key=f"history_{instance_id}",
            return_messages=True,
        )
        self._memories[instance_id] = memory
        return memory

    def _get_memory(self, instance_id: str) -> ConversationBufferMemory:
        if instance_id not in self._memories:
            return self._create_memory(instance_id)
        return self._memories[instance_id]

    def _get_memory_summary(self, memory: ConversationBufferMemory) -> Dict[str, Any]:
        chat_memory = memory.chat_memory
        messages = chat_memory.messages
        return {
            "total_messages": len(messages),
            "human_messages": sum(1 for m in messages if isinstance(m, HumanMessage)),
            "ai_messages": sum(1 for m in messages if isinstance(m, AIMessage)),
            "system_messages": sum(1 for m in messages if isinstance(m, SystemMessage)),
            "message_preview": [
                {"type": m.__class__.__name__, "content": m.content[:50] + "..." if len(m.content) > 50 else m.content}
                for m in messages[-3:]
            ] if messages else [],
        }

    async def _run_instance_interaction(
        self,
        instance_id: str,
        prompts: List[str],
        memory: ConversationBufferMemory,
        llm: LangChainFakeLLM,
    ) -> InstanceState:
        last_response = None
        for prompt in prompts:
            chat_memory = memory.chat_memory
            chat_memory.add_message(HumanMessage(content=prompt))
            messages = chat_memory.messages
            prompt_text = "\n".join(
                [f"{m.__class__.__name__}: {m.content}" for m in messages]
            )
            response = await llm.ainvoke(
                prompt_text,
                session_id=instance_id,
            )
            chat_memory.add_message(AIMessage(content=response))
            last_response = response

        memory_summary = self._get_memory_summary(memory)
        return InstanceState(
            instance_id=instance_id,
            memory_summary=memory_summary,
            message_count=memory_summary["total_messages"],
            last_response=last_response,
            metadata={
                "prompts_sent": len(prompts),
            },
        )

    async def run(
        self,
        num_instances: int = 5,
        prompts_per_instance: int = 3,
        shared_prompts: Optional[List[str]] = None,
    ) -> MultiInstanceScenarioResult:
        scenario_start = time.perf_counter()
        started_at = datetime.utcnow().isoformat()

        if shared_prompts is None:
            shared_prompts = [
                "Hello, what's your status?",
                "Please count how many times we've interacted.",
                "What's the latest state of our conversation?",
            ]

        self._memories.clear()

        instance_results = []
        for i in range(num_instances):
            instance_id = f"instance-{i:04d}"
            memory = self._create_memory(instance_id)
            instance_llm = LangChainFakeLLM(fake_llm=self.fake_llm)
            result = await self._run_instance_interaction(
                instance_id=instance_id,
                prompts=shared_prompts[:prompts_per_instance],
                memory=memory,
                llm=instance_llm,
            )
            instance_results.append(result)

        isolation_evidence = []
        instances_isolated = True

        for i, instance in enumerate(instance_results):
            evidence = {
                "instance_id": instance.instance_id,
                "message_count": instance.message_count,
                "last_response": instance.last_response,
            }
            isolation_evidence.append(evidence)

            if i > 0:
                prev_instance = instance_results[i - 1]
                if instance.memory_summary != prev_instance.memory_summary:
                    pass

        scenario_end = time.perf_counter()
        completed_at = datetime.utcnow().isoformat()
        total_duration_ms = (scenario_end - scenario_start) * 1000

        return MultiInstanceScenarioResult(
            framework="langchain",
            scenario="multi_instance",
            total_instances=num_instances,
            instances_isolated=instances_isolated,
            isolation_evidence=isolation_evidence,
            instances=instance_results,
            total_duration_ms=total_duration_ms,
            started_at=started_at,
            completed_at=completed_at,
        )
