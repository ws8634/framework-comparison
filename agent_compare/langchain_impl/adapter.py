from typing import Any, Dict, List, Optional, Iterator

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    FunctionMessage,
)
from langchain_core.outputs import Generation, LLMResult

from agent_compare.fake_llm import (
    FakeLLM,
    FakeChatMessage,
    MessageRole,
    FakeLLMResponse,
    create_fake_llm,
)


class LangChainFakeLLM(LLM):
    fake_llm: FakeLLM

    def __init__(self, fake_llm: Optional[FakeLLM] = None, **kwargs: Any):
        if fake_llm is None:
            fake_llm = create_fake_llm()
        super().__init__(fake_llm=fake_llm, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "langchain-fake-llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.fake_llm.model_name,
            "call_count": self.fake_llm.call_count,
        }

    def _convert_message(self, message: BaseMessage) -> FakeChatMessage:
        if isinstance(message, HumanMessage):
            return FakeChatMessage(role=MessageRole.USER, content=message.content)
        elif isinstance(message, AIMessage):
            return FakeChatMessage(role=MessageRole.ASSISTANT, content=message.content)
        elif isinstance(message, SystemMessage):
            return FakeChatMessage(role=MessageRole.SYSTEM, content=message.content)
        elif isinstance(message, FunctionMessage):
            return FakeChatMessage(
                role=MessageRole.FUNCTION,
                content=message.content,
                name=message.name,
            )
        else:
            return FakeChatMessage(role=MessageRole.USER, content=message.content)

    def _convert_response(self, response: FakeLLMResponse) -> str:
        return response.content

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [FakeChatMessage(role=MessageRole.USER, content=prompt)]
        response = self.fake_llm.generate(messages, **kwargs)
        return self._convert_response(response)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [FakeChatMessage(role=MessageRole.USER, content=prompt)]
        response = await self.fake_llm.agenerate(messages, **kwargs)
        return self._convert_response(response)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            messages = [FakeChatMessage(role=MessageRole.USER, content=prompt)]
            response = self.fake_llm.generate(messages, **kwargs)
            generations.append(
                [Generation(text=self._convert_response(response))]
            )
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            messages = [FakeChatMessage(role=MessageRole.USER, content=prompt)]
            response = await self.fake_llm.agenerate(messages, **kwargs)
            generations.append(
                [Generation(text=self._convert_response(response))]
            )
        return LLMResult(generations=generations)

    def get_stats(self) -> Dict[str, Any]:
        return self.fake_llm.get_stats()
