import asyncio
import pytest
import sys
from typing import Any, Dict, List

from tests.conftest import (
    skip_if_no_langchain,
    skip_if_no_agentscope,
    skip_if_no_both_frameworks,
    _check_langchain,
    _check_agentscope,
)


LangChainConcurrentScenario = None
LangChainMultiInstanceScenario = None
LangChainMultiAgentScenario = None
AgentScopeConcurrentScenario = None
AgentScopeMultiInstanceScenario = None
AgentScopeMultiAgentScenario = None


def _lazy_import_langchain():
    """Lazy import LangChain implementations."""
    global LangChainConcurrentScenario
    global LangChainMultiInstanceScenario
    global LangChainMultiAgentScenario
    
    if LangChainConcurrentScenario is None:
        from agent_compare.langchain_impl import (
            LangChainConcurrentScenario as _LangChainConcurrentScenario,
            LangChainMultiInstanceScenario as _LangChainMultiInstanceScenario,
            LangChainMultiAgentScenario as _LangChainMultiAgentScenario,
        )
        LangChainConcurrentScenario = _LangChainConcurrentScenario
        LangChainMultiInstanceScenario = _LangChainMultiInstanceScenario
        LangChainMultiAgentScenario = _LangChainMultiAgentScenario


def _lazy_import_agentscope():
    """Lazy import AgentScope implementations."""
    global AgentScopeConcurrentScenario
    global AgentScopeMultiInstanceScenario
    global AgentScopeMultiAgentScenario
    
    if AgentScopeConcurrentScenario is None:
        from agent_compare.agentscope_impl import (
            AgentScopeConcurrentScenario as _AgentScopeConcurrentScenario,
            AgentScopeMultiInstanceScenario as _AgentScopeMultiInstanceScenario,
            AgentScopeMultiAgentScenario as _AgentScopeMultiAgentScenario,
        )
        AgentScopeConcurrentScenario = _AgentScopeConcurrentScenario
        AgentScopeMultiInstanceScenario = _AgentScopeMultiInstanceScenario
        AgentScopeMultiAgentScenario = _AgentScopeMultiAgentScenario


class TestOutputStructureConsistency:
    """测试输出结构一致性（需要两个框架都可用）"""

    @pytest.mark.asyncio
    @pytest.mark.both_frameworks
    @skip_if_no_both_frameworks()
    async def test_concurrent_scenario_structure_match(self):
        """验证并发调度场景在两个框架中的输出结构一致"""
        _lazy_import_langchain()
        _lazy_import_agentscope()
        
        lc_scenario = LangChainConcurrentScenario()
        as_scenario = AgentScopeConcurrentScenario()

        lc_result = await lc_scenario.run(num_tasks=3, max_concurrency=2)
        as_result = await as_scenario.run(num_tasks=3, max_concurrency=2)

        lc_dict = lc_result.to_dict()
        as_dict = as_result.to_dict()

        expected_keys = {
            "framework",
            "scenario",
            "total_tasks",
            "max_concurrency",
            "rate_limit",
            "tasks_completed",
            "tasks_failed",
            "tasks_queued",
            "tasks_rejected",
            "total_duration_ms",
            "avg_task_duration_ms",
            "actual_concurrency",
            "results",
            "started_at",
            "completed_at",
        }

        assert set(lc_dict.keys()) == expected_keys
        assert set(as_dict.keys()) == expected_keys

        assert len(lc_dict["results"]) == 3
        assert len(as_dict["results"]) == 3

        result_keys = {
            "task_id",
            "status",
            "started_at",
            "completed_at",
            "duration_ms",
            "result",
            "error",
            "metadata",
        }
        assert set(lc_dict["results"][0].keys()) == result_keys
        assert set(as_dict["results"][0].keys()) == result_keys

    @pytest.mark.asyncio
    @pytest.mark.both_frameworks
    @skip_if_no_both_frameworks()
    async def test_multi_instance_scenario_structure_match(self):
        """验证多实例管理场景在两个框架中的输出结构一致"""
        _lazy_import_langchain()
        _lazy_import_agentscope()
        
        lc_scenario = LangChainMultiInstanceScenario()
        as_scenario = AgentScopeMultiInstanceScenario()

        lc_result = await lc_scenario.run(num_instances=3, prompts_per_instance=2)
        as_result = await as_scenario.run(num_instances=3, prompts_per_instance=2)

        lc_dict = lc_result.to_dict()
        as_dict = as_result.to_dict()

        expected_keys = {
            "framework",
            "scenario",
            "total_instances",
            "instances_isolated",
            "isolation_evidence",
            "instances",
            "total_duration_ms",
            "started_at",
            "completed_at",
        }

        assert set(lc_dict.keys()) == expected_keys
        assert set(as_dict.keys()) == expected_keys

        assert len(lc_dict["instances"]) == 3
        assert len(as_dict["instances"]) == 3

        instance_keys = {
            "instance_id",
            "memory_summary",
            "message_count",
            "last_response",
            "metadata",
        }
        assert set(lc_dict["instances"][0].keys()) == instance_keys
        assert set(as_dict["instances"][0].keys()) == instance_keys

    @pytest.mark.asyncio
    @pytest.mark.both_frameworks
    @skip_if_no_both_frameworks()
    async def test_multi_agent_scenario_structure_match(self):
        """验证多智能体协作场景在两个框架中的输出结构一致"""
        _lazy_import_langchain()
        _lazy_import_agentscope()
        
        lc_scenario = LangChainMultiAgentScenario()
        as_scenario = AgentScopeMultiAgentScenario()

        lc_result = await lc_scenario.run()
        as_result = await as_scenario.run()

        lc_dict = lc_result.to_dict()
        as_dict = as_result.to_dict()

        expected_keys = {
            "framework",
            "scenario",
            "task_goal",
            "message_logs",
            "plan",
            "execution_results",
            "review_conclusion",
            "success",
            "errors_captured",
            "total_duration_ms",
            "started_at",
            "completed_at",
        }

        assert set(lc_dict.keys()) == expected_keys
        assert set(as_dict.keys()) == expected_keys

        execution_keys = {
            "task_id",
            "task_name",
            "status",
            "result",
            "error",
            "duration_ms",
        }
        assert set(lc_dict["execution_results"][0].keys()) == execution_keys
        assert set(as_dict["execution_results"][0].keys()) == execution_keys


class TestConcurrentRateLimiting:
    """测试并发限流参数生效"""

    @pytest.mark.asyncio
    @pytest.mark.langchain
    @skip_if_no_langchain()
    async def test_max_concurrency_limit_langchain(self):
        """验证 LangChain 并发限流参数生效"""
        _lazy_import_langchain()
        scenario = LangChainConcurrentScenario(base_delay_ms=50, jitter_ms=0)

        result = await scenario.run(num_tasks=10, max_concurrency=3)

        assert result.max_concurrency == 3
        assert result.actual_concurrency <= 3
        assert result.total_tasks == 10
        assert result.tasks_completed == 10

    @pytest.mark.asyncio
    @pytest.mark.agentscope
    @skip_if_no_agentscope()
    async def test_max_concurrency_limit_agentscope(self):
        """验证 AgentScope 并发限流参数生效"""
        _lazy_import_agentscope()
        scenario = AgentScopeConcurrentScenario(base_delay_ms=50, jitter_ms=0)

        result = await scenario.run(num_tasks=10, max_concurrency=3)

        assert result.max_concurrency == 3
        assert result.actual_concurrency <= 3
        assert result.total_tasks == 10
        assert result.tasks_completed == 10

    @pytest.mark.asyncio
    @pytest.mark.both_frameworks
    @skip_if_no_both_frameworks()
    async def test_unlimited_concurrency(self):
        """验证无限并发时的行为"""
        _lazy_import_langchain()
        _lazy_import_agentscope()
        
        lc_scenario = LangChainConcurrentScenario(base_delay_ms=10, jitter_ms=0)
        as_scenario = AgentScopeConcurrentScenario(base_delay_ms=10, jitter_ms=0)

        lc_result = await lc_scenario.run(num_tasks=5, max_concurrency=None)
        as_result = await as_scenario.run(num_tasks=5, max_concurrency=None)

        assert lc_result.max_concurrency is None
        assert as_result.max_concurrency is None
        assert lc_result.tasks_completed == 5
        assert as_result.tasks_completed == 5


class TestMultiInstanceIsolation:
    """测试多实例状态隔离"""

    @pytest.mark.asyncio
    @pytest.mark.langchain
    @skip_if_no_langchain()
    async def test_langchain_instance_isolation(self):
        """验证 LangChain 多实例状态隔离"""
        _lazy_import_langchain()
        scenario = LangChainMultiInstanceScenario()

        result = await scenario.run(num_instances=5, prompts_per_instance=3)

        assert result.total_instances == 5
        assert result.instances_isolated

        for i, instance in enumerate(result.instances):
            assert instance.instance_id == f"instance-{i:04d}"
            assert instance.message_count == 6

        assert len(result.isolation_evidence) == 5

    @pytest.mark.asyncio
    @pytest.mark.agentscope
    @skip_if_no_agentscope()
    async def test_agentscope_instance_isolation(self):
        """验证 AgentScope 多实例状态隔离"""
        _lazy_import_agentscope()
        scenario = AgentScopeMultiInstanceScenario()

        result = await scenario.run(num_instances=5, prompts_per_instance=3)

        assert result.total_instances == 5
        assert result.instances_isolated

        for i, instance in enumerate(result.instances):
            assert instance.instance_id == f"instance-{i:04d}"
            assert instance.message_count == 6

        assert len(result.isolation_evidence) == 5

    @pytest.mark.asyncio
    @pytest.mark.both_frameworks
    @skip_if_no_both_frameworks()
    async def test_instance_memory_independence(self):
        """验证实例内存相互独立"""
        _lazy_import_langchain()
        _lazy_import_agentscope()
        
        lc_scenario = LangChainMultiInstanceScenario()
        as_scenario = AgentScopeMultiInstanceScenario()

        lc_result = await lc_scenario.run(num_instances=2, prompts_per_instance=2)
        as_result = await as_scenario.run(num_instances=2, prompts_per_instance=2)

        assert lc_result.instances[0].memory_summary != lc_result.instances[1].memory_summary
        assert as_result.instances[0].memory_summary != as_result.instances[1].memory_summary


class TestExceptionPathCapture:
    """测试异常路径能被汇总捕获"""

    @pytest.mark.asyncio
    @pytest.mark.langchain
    @skip_if_no_langchain()
    async def test_langchain_exception_capture(self):
        """验证 LangChain 异常路径被捕获"""
        _lazy_import_langchain()
        scenario = LangChainMultiAgentScenario()

        result = await scenario.run(fail_subtask="st-002")

        assert result.success is False
        assert len(result.errors_captured) == 1
        assert "timeout" in result.errors_captured[0].lower()

        failed_tasks = [t for t in result.execution_results if t.status == "failed"]
        assert len(failed_tasks) == 1
        assert failed_tasks[0].task_id == "st-002"

        assert "PARTIAL_SUCCESS" in result.review_conclusion

    @pytest.mark.asyncio
    @pytest.mark.agentscope
    @skip_if_no_agentscope()
    async def test_agentscope_exception_capture(self):
        """验证 AgentScope 异常路径被捕获"""
        _lazy_import_agentscope()
        scenario = AgentScopeMultiAgentScenario()

        result = await scenario.run(fail_subtask="st-002")

        assert result.success is False
        assert len(result.errors_captured) == 1
        assert "timeout" in result.errors_captured[0].lower()

        failed_tasks = [t for t in result.execution_results if t.status == "failed"]
        assert len(failed_tasks) == 1
        assert failed_tasks[0].task_id == "st-002"

        assert "PARTIAL_SUCCESS" in result.review_conclusion

    @pytest.mark.asyncio
    @pytest.mark.both_frameworks
    @skip_if_no_both_frameworks()
    async def test_successful_path_no_errors(self):
        """验证成功路径无错误"""
        _lazy_import_langchain()
        _lazy_import_agentscope()
        
        lc_scenario = LangChainMultiAgentScenario()
        as_scenario = AgentScopeMultiAgentScenario()

        lc_result = await lc_scenario.run()
        as_result = await as_scenario.run()

        assert lc_result.success is True
        assert len(lc_result.errors_captured) == 0
        assert all(t.status == "completed" for t in lc_result.execution_results)
        assert "SUCCESS" in lc_result.review_conclusion

        assert as_result.success is True
        assert len(as_result.errors_captured) == 0
        assert all(t.status == "completed" for t in as_result.execution_results)
        assert "SUCCESS" in as_result.review_conclusion


class TestFakeLLMDeterminism:
    """测试假LLM输出确定性（无需框架）"""

    @pytest.mark.asyncio
    @pytest.mark.fake_llm
    async def test_hash_based_determinism(self):
        """验证基于哈希的规则输出稳定"""
        from agent_compare.fake_llm import (
            HashBasedRule,
            FakeChatMessage,
            MessageRole,
            create_fake_llm,
        )

        rule = HashBasedRule()
        llm = create_fake_llm(rules=[rule])

        messages = [FakeChatMessage(role=MessageRole.USER, content="test prompt")]

        response1 = llm.generate(messages)
        response2 = llm.generate(messages)

        assert response1.content == response2.content
        assert response1.metadata["message_hash"] == response2.metadata["message_hash"]

    @pytest.mark.asyncio
    @pytest.mark.fake_llm
    async def test_keyword_response_determinism(self):
        """验证关键词响应规则输出稳定"""
        from agent_compare.fake_llm import (
            KeywordResponseRule,
            FakeChatMessage,
            MessageRole,
            create_fake_llm,
        )

        rule = KeywordResponseRule(
            keyword_responses={
                "hello": "World!",
                "test": "Passed",
            }
        )
        llm = create_fake_llm(rules=[rule])

        for _ in range(5):
            messages = [FakeChatMessage(role=MessageRole.USER, content="Say hello")]
            response = llm.generate(messages)
            assert response.content == "World!"

        for _ in range(5):
            messages = [FakeChatMessage(role=MessageRole.USER, content="Run the test")]
            response = llm.generate(messages)
            assert response.content == "Passed"

    @pytest.mark.asyncio
    @pytest.mark.fake_llm
    async def test_counter_rule_instance_isolation(self):
        """验证计数器规则的实例隔离性"""
        from agent_compare.fake_llm import (
            CounterRule,
            FakeChatMessage,
            MessageRole,
            create_fake_llm,
        )

        rule1 = CounterRule(prefix="Instance A: ")
        llm1 = create_fake_llm(rules=[rule1])

        rule2 = CounterRule(prefix="Instance B: ")
        llm2 = create_fake_llm(rules=[rule2])

        messages = [FakeChatMessage(role=MessageRole.USER, content="test")]

        for i in range(3):
            response1 = llm1.generate(messages)
            response2 = llm2.generate(messages)
            
            assert f"Instance A: {i + 1}" in response1.content
            assert f"Instance B: {i + 1}" in response2.content

    @pytest.mark.asyncio
    @pytest.mark.fake_llm
    async def test_error_simulating_rule(self):
        """验证错误模拟规则"""
        from agent_compare.fake_llm import (
            ErrorSimulatingRule,
            FakeChatMessage,
            MessageRole,
            create_fake_llm,
        )

        rule = ErrorSimulatingRule(
            trigger_keyword="fail",
            error_type="timeout",
            error_message="Request timed out after 30 seconds"
        )
        llm = create_fake_llm(rules=[rule])

        fail_message = [FakeChatMessage(role=MessageRole.USER, content="Please fail this request")]
        normal_message = [FakeChatMessage(role=MessageRole.USER, content="Normal request")]

        response = llm.generate(fail_message)
        assert "error" in response.content.lower()
        assert "timeout" in response.content.lower()

        response = llm.generate(normal_message)
        assert "error" not in response.content.lower()


class TestCompatibilityModule:
    """测试兼容性模块"""

    @pytest.mark.fake_llm
    def test_check_compatibility_returns_report(self):
        """验证兼容性检查返回有效报告"""
        from agent_compare.compatibility import (
            check_compatibility,
            CompatibilityReport,
            FrameworkStatus,
        )

        report = check_compatibility()

        assert isinstance(report, CompatibilityReport)
        assert isinstance(report.python_version, str)
        assert isinstance(report.langchain, FrameworkStatus)
        assert isinstance(report.agentscope, FrameworkStatus)

    @pytest.mark.fake_llm
    def test_format_compatibility_report(self):
        """验证兼容性报告格式化"""
        from agent_compare.compatibility import (
            check_compatibility,
            format_compatibility_report,
        )

        report = check_compatibility()
        formatted = format_compatibility_report(report)

        assert isinstance(formatted, str)
        assert "兼容性检查报告" in formatted
        assert "Python 版本" in formatted

    @pytest.mark.fake_llm
    def test_framework_availability_enum(self):
        """验证框架可用性枚举值"""
        from agent_compare.compatibility import (
            FrameworkAvailability,
        )

        assert FrameworkAvailability.AVAILABLE == "available"
        assert FrameworkAvailability.NOT_INSTALLED == "not_installed"
        assert FrameworkAvailability.NOT_SUPPORTED == "not_supported"

    @pytest.mark.fake_llm
    def test_exit_codes_defined(self):
        """验证退出码定义"""
        from agent_compare.compatibility import (
            EXIT_CODE_SUCCESS,
            EXIT_CODE_INVALID_FRAMEWORK,
            EXIT_CODE_NO_FRAMEWORKS,
            EXIT_CODE_MISSING_DEPENDENCY,
            EXIT_CODE_RUNTIME_ERROR,
        )

        assert EXIT_CODE_SUCCESS == 0
        assert EXIT_CODE_INVALID_FRAMEWORK == 1
        assert EXIT_CODE_NO_FRAMEWORKS == 2
        assert EXIT_CODE_MISSING_DEPENDENCY == 3
        assert EXIT_CODE_RUNTIME_ERROR == 4
