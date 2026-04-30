import asyncio
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

from .base import (
    FakeLLM,
    FakeLLMResponse,
    FakeChatMessage,
    MessageRole,
)


class PluggableRule(ABC):
    @abstractmethod
    def match(self, messages: List[FakeChatMessage], **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def generate_response(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        pass


class SimpleEchoRule(PluggableRule):
    def __init__(self, prefix: str = "Echo: "):
        self.prefix = prefix

    def match(self, messages: List[FakeChatMessage], **kwargs: Any) -> bool:
        return True

    def generate_response(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        last_user_msg = next(
            (m for m in reversed(messages) if m.role == MessageRole.USER),
            None,
        )
        content = self.prefix + (last_user_msg.content if last_user_msg else "No message")
        token_count = len(content) // 4
        return FakeLLMResponse(
            content=content,
            tokens_used=token_count,
            metadata={"rule": "simple_echo"},
        )


class KeywordResponseRule(PluggableRule):
    def __init__(
        self,
        keyword_responses: Dict[str, str],
        case_sensitive: bool = False,
    ):
        self.keyword_responses = keyword_responses
        self.case_sensitive = case_sensitive

    def match(self, messages: List[FakeChatMessage], **kwargs: Any) -> bool:
        if not messages:
            return False
        last_msg = messages[-1].content
        for keyword in self.keyword_responses:
            pattern = keyword if self.case_sensitive else keyword.lower()
            content = last_msg if self.case_sensitive else last_msg.lower()
            if pattern in content:
                return True
        return False

    def generate_response(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        last_msg = messages[-1].content
        for keyword, response in self.keyword_responses.items():
            pattern = keyword if self.case_sensitive else keyword.lower()
            content = last_msg if self.case_sensitive else last_msg.lower()
            if pattern in content:
                return FakeLLMResponse(
                    content=response,
                    tokens_used=len(response) // 4,
                    metadata={"rule": "keyword", "matched_keyword": keyword},
                )
        return FakeLLMResponse(
            content="No matching keyword found",
            tokens_used=5,
            metadata={"rule": "keyword", "matched_keyword": None},
        )


class CounterRule(PluggableRule):
    def __init__(self, start_value: int = 0):
        self.start_value = start_value
        self._counter: Dict[str, int] = {}

    def match(self, messages: List[FakeChatMessage], **kwargs: Any) -> bool:
        return True

    def generate_response(
        self,
        messages: List[FakeChatMessage],
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FakeLLMResponse:
        sid = session_id or "default"
        if sid not in self._counter:
            self._counter[sid] = self.start_value
        self._counter[sid] += 1
        content = f"Counter value for session '{sid}': {self._counter[sid]}"
        return FakeLLMResponse(
            content=content,
            tokens_used=len(content) // 4,
            metadata={
                "rule": "counter",
                "session_id": sid,
                "value": self._counter[sid],
            },
        )

    def reset(self, session_id: Optional[str] = None) -> None:
        if session_id:
            self._counter.pop(session_id, None)
        else:
            self._counter.clear()


class MultiRoleRule(PluggableRule):
    def __init__(
        self,
        role_responses: Dict[str, str],
        default_response: str = "I'm not sure how to respond.",
    ):
        self.role_responses = role_responses
        self.default_response = default_response

    def match(self, messages: List[FakeChatMessage], **kwargs: Any) -> bool:
        return True

    def generate_response(
        self,
        messages: List[FakeChatMessage],
        agent_role: Optional[str] = None,
        **kwargs: Any,
    ) -> FakeLLMResponse:
        response = self.role_responses.get(agent_role, self.default_response) if agent_role else self.default_response
        return FakeLLMResponse(
            content=response,
            tokens_used=len(response) // 4,
            metadata={
                "rule": "multi_role",
                "agent_role": agent_role,
            },
        )


class ErrorSimulatingRule(PluggableRule):
    def __init__(
        self,
        error_triggers: List[str],
        error_message: str = "Simulated error occurred",
        error_type: str = "api_error",
        case_sensitive: bool = False,
    ):
        self.error_triggers = error_triggers
        self.error_message = error_message
        self.error_type = error_type
        self.case_sensitive = case_sensitive

    def match(self, messages: List[FakeChatMessage], **kwargs: Any) -> bool:
        if not messages:
            return False
        last_msg = messages[-1].content
        for trigger in self.error_triggers:
            pattern = trigger if self.case_sensitive else trigger.lower()
            content = last_msg if self.case_sensitive else last_msg.lower()
            if pattern in content:
                return True
        return False

    def generate_response(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        return FakeLLMResponse(
            content=self.error_message,
            tokens_used=0,
            finish_reason="error",
            metadata={
                "rule": "error_simulating",
                "error_type": self.error_type,
                "is_error": True,
            },
        )


class DelaySimulatingRule(PluggableRule):
    def __init__(
        self,
        base_delay_ms: float = 100.0,
        jitter_ms: float = 50.0,
    ):
        self.base_delay_ms = base_delay_ms
        self.jitter_ms = jitter_ms

    def match(self, messages: List[FakeChatMessage], **kwargs: Any) -> bool:
        return True

    def generate_response(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        import random
        delay_ms = self.base_delay_ms + random.uniform(0, self.jitter_ms)
        content = f"Response after {delay_ms:.2f}ms simulated delay"
        return FakeLLMResponse(
            content=content,
            tokens_used=len(content) // 4,
            metadata={
                "rule": "delay_simulating",
                "simulated_delay_ms": delay_ms,
            },
        )


class HashBasedRule(PluggableRule):
    def __init__(self, template: str = "Response hash: {hash}"):
        self.template = template

    def match(self, messages: List[FakeChatMessage], **kwargs: Any) -> bool:
        return True

    def _compute_hash(self, messages: List[FakeChatMessage]) -> str:
        content = "\n".join(
            f"{m.role.value}: {m.content}" for m in messages
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def generate_response(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        hash_value = self._compute_hash(messages)
        content = self.template.format(hash=hash_value)
        return FakeLLMResponse(
            content=content,
            tokens_used=len(content) // 4,
            metadata={
                "rule": "hash_based",
                "message_hash": hash_value,
            },
        )


class RuleBasedFakeLLM(FakeLLM):
    def __init__(self, model_name: str = "fake-model-v1"):
        super().__init__(model_name)
        self._default_rule: PluggableRule = SimpleEchoRule()

    def generate(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        self.call_count += 1
        for rule in self._response_rules:
            if rule.match(messages, **kwargs):
                response = rule.generate_response(messages, **kwargs)
                response.model_name = self.model_name
                return response
        response = self._default_rule.generate_response(messages, **kwargs)
        response.model_name = self.model_name
        return response

    async def agenerate(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        for rule in self._response_rules:
            if isinstance(rule, DelaySimulatingRule) and rule.match(messages, **kwargs):
                import random
                delay_ms = rule.base_delay_ms + random.uniform(0, rule.jitter_ms)
                await asyncio.sleep(delay_ms / 1000.0)
        return self.generate(messages, **kwargs)


def create_fake_llm(
    model_name: str = "fake-model-v1",
    rules: Optional[List[PluggableRule]] = None,
) -> RuleBasedFakeLLM:
    llm = RuleBasedFakeLLM(model_name)
    if rules:
        for rule in rules:
            llm.add_rule(rule)
    return llm
