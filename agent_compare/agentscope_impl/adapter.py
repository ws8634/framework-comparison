import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

from agent_compare.fake_llm import (
    FakeLLM,
    FakeLLMResponse,
    FakeChatMessage,
    MessageRole,
    create_fake_llm,
)


class AgentScopeFakeModel:
    def __init__(
        self,
        fake_llm: Optional[FakeLLM] = None,
        config_name: str = "fake-model",
        model_type: str = "text",
    ):
        if fake_llm is None:
            fake_llm = create_fake_llm()
        self.fake_llm = fake_llm
        self.config_name = config_name
        self.model_type = model_type
        self._model_name = fake_llm.model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def _convert_to_fake_messages(
        self,
        messages: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[FakeChatMessage]:
        if isinstance(messages, str):
            return [FakeChatMessage(role=MessageRole.USER, content=messages)]
        
        if isinstance(messages, dict):
            messages = [messages]
        
        result = []
        for msg in messages:
            role_str = msg.get("role", "user").lower()
            content = msg.get("content", "")
            name = msg.get("name")
            function_call = msg.get("function_call")
            
            if role_str == "system":
                role = MessageRole.SYSTEM
            elif role_str == "assistant":
                role = MessageRole.ASSISTANT
            elif role_str == "function":
                role = MessageRole.FUNCTION
            else:
                role = MessageRole.USER
            
            result.append(FakeChatMessage(
                role=role,
                content=content,
                name=name,
                function_call=function_call,
            ))
        return result

    def _format_response(
        self,
        response: FakeLLMResponse,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "content": response.content,
            "text": response.content,
            "finish_reason": response.finish_reason,
            "model_name": response.model_name,
            "tokens_used": response.tokens_used,
            "function_call": response.function_call,
            "metadata": response.metadata,
            "raw": response.to_dict(),
        }

    def __call__(
        self,
        messages: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        fake_messages = self._convert_to_fake_messages(messages)
        response = self.fake_llm.generate(fake_messages, **kwargs)
        return self._format_response(response, **kwargs)

    async def ainvoke(
        self,
        messages: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        fake_messages = self._convert_to_fake_messages(messages)
        response = await self.fake_llm.agenerate(fake_messages, **kwargs)
        return self._format_response(response, **kwargs)

    def invoke(
        self,
        messages: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.__call__(messages, **kwargs)

    def generate(
        self,
        messages: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.__call__(messages, **kwargs)

    async def agenerate(
        self,
        messages: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self.ainvoke(messages, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return self.fake_llm.get_stats()
