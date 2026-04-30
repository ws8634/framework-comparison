from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict, List


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class FakeChatMessage:
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        return result


@dataclass
class FakeLLMResponse:
    content: str
    model_name: str = "fake-model-v1"
    tokens_used: int = 0
    finish_reason: str = "stop"
    function_call: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model_name": self.model_name,
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "function_call": self.function_call,
            "metadata": self.metadata,
        }


class FakeLLM(ABC):
    def __init__(self, model_name: str = "fake-model-v1"):
        self.model_name = model_name
        self.call_count = 0
        self._response_rules: List["PluggableRule"] = []

    @abstractmethod
    def generate(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages: List[FakeChatMessage],
        **kwargs: Any,
    ) -> FakeLLMResponse:
        pass

    def add_rule(self, rule: "PluggableRule") -> None:
        self._response_rules.append(rule)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "call_count": self.call_count,
        }
