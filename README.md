# Agent 框架选型对比

## 项目目标

本项目旨在通过**可运行的最小化实现**，对比 LangChain 与 AgentScope 两个主流 Agent 框架在以下三个核心维度的差异：

- **并发调度**：同时运行多个独立任务的能力
- **多实例管理**：维护多个相互独立状态的 Agent 实例
- **多智能体协作**：多个 Agent 协同完成复杂任务

**设计原则**：
1. **无外部依赖**：所有示例不依赖真实 LLM API，使用可预测的假模型层
2. **输出一致性**：两套框架针对同一输入生成相同结构的 JSON 结果
3. **可复现性**：相同输入产生稳定输出，便于横向对比

---

## 环境要求

- Python 版本：>= 3.9, < 3.12
- 系统：Linux / macOS / Windows

---

## 安装依赖

### 方式一：开发者模式（推荐）

```bash
# 克隆仓库后进入目录
cd 12-agent-framework-comparison

# 以开发者模式安装
pip install -e .

# 安装开发依赖（用于运行测试）
pip install -e ".[dev]"
```

### 方式二：直接安装依赖

```bash
pip install langchain langchain-core agentscope pydantic typer rich aiohttp asyncio-throttle pytest pytest-asyncio
```

---

## 快速开始

### 一键运行所有场景

```bash
# 运行两个框架的所有对比场景
python -m agent_compare.cli run

# 或使用安装后的命令
agent-compare run
```

运行后将在终端显示摘要表格，并在 `artifacts/` 目录生成完整的 JSON 结果文件。

### 运行特定场景

```bash
# 仅运行并发调度场景
agent-compare run --scenario concurrent

# 仅运行多实例管理场景
agent-compare run --scenario multi_instance

# 仅运行多智能体协作场景
agent-compare run --scenario multi_agent
```

### 运行特定框架

```bash
# 仅运行 LangChain
agent-compare run --framework langchain

# 仅运行 AgentScope
agent-compare run --framework agentscope

# 运行两者（默认）
agent-compare run --framework both
```

---

## 调整实验规模

通过 CLI 参数可以调整各场景的规模：

### 并发调度场景参数

```bash
# 运行 50 个任务，最大并发 10
agent-compare run \
    --scenario concurrent \
    --num-tasks 50 \
    --max-concurrency 10

# 不限制并发（默认：限制为 5）
agent-compare run \
    --scenario concurrent \
    --num-tasks 200 \
    --max-concurrency 0
```

### 多实例管理场景参数

```bash
# 管理 10 个实例，每个实例进行 5 轮交互
agent-compare run \
    --scenario multi_instance \
    --num-instances 10 \
    --prompts-per-instance 5
```

### 多智能体协作场景参数

```bash
# 正常执行路径
agent-compare run --scenario multi_agent

# 模拟异常路径（指定子任务失败）
agent-compare run \
    --scenario multi_agent \
    --fail-subtask st-002
```

---

## 输出文件结构

运行后，`artifacts/` 目录会生成以下文件：

```
artifacts/
├── aggregate_20260430_143022.json          # 总汇总（包含所有场景的对比）
├── langchain_concurrent_20260430_143022.json
├── langchain_multi_instance_20260430_143022.json
├── langchain_multi_agent_20260430_143022.json
├── agentscope_concurrent_20260430_143022.json
├── agentscope_multi_instance_20260430_143022.json
└── agentscope_multi_agent_20260430_143022.json
```

### 输出格式说明

两套框架的输出 JSON 结构完全一致，便于直接对比：

**并发调度场景输出**：
```json
{
  "framework": "langchain",
  "scenario": "concurrent",
  "total_tasks": 10,
  "max_concurrency": 5,
  "tasks_completed": 10,
  "tasks_failed": 0,
  "total_duration_ms": 254.32,
  "actual_concurrency": 5,
  "results": [
    {
      "task_id": "task-0000",
      "status": "completed",
      "duration_ms": 75.2,
      "result": { "prompt": "...", "response": "..." }
    }
  ]
}
```

**多智能体协作场景输出**（含异常捕获）：
```json
{
  "framework": "agentscope",
  "scenario": "multi_agent",
  "success": false,
  "errors_captured": ["Task failed: simulated API timeout"],
  "plan": { "task_goal": "...", "subtasks": [...] },
  "execution_results": [
    { "task_id": "st-001", "status": "completed", "result": {...} },
    { "task_id": "st-002", "status": "failed", "error": "..." }
  ],
  "message_logs": [
    { "sender": "user", "receiver": "planner", "content": "...", "timestamp": 12345.678 }
  ],
  "review_conclusion": "Review Complete:\n- Status: PARTIAL_SUCCESS"
}
```

---

## 运行测试

项目包含完整的测试套件，验证核心功能：

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试类
pytest tests/test_framework_comparison.py::TestOutputStructureConsistency -v
pytest tests/test_framework_comparison.py::TestConcurrentRateLimiting -v
pytest tests/test_framework_comparison.py::TestMultiInstanceIsolation -v
pytest tests/test_framework_comparison.py::TestExceptionPathCapture -v

# 带覆盖率报告
pytest tests/ -v --cov=agent_compare
```

### 测试覆盖范围

| 测试类 | 验证内容 |
|--------|----------|
| `TestOutputStructureConsistency` | 两个框架输出 JSON 结构完全一致 |
| `TestConcurrentRateLimiting` | 最大并发数参数生效 |
| `TestMultiInstanceIsolation` | 各实例状态独立隔离 |
| `TestExceptionPathCapture` | 异常路径被正确捕获 |
| `TestFakeLLMDeterminism` | 假模型输出稳定可复现 |

---

## 复现实验结果

按照以下步骤可在本地离线完整复现所有实验：

### 步骤 1：准备环境

```bash
# 1. 确保使用 Python 3.9-3.11
python --version

# 2. 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -e ".[dev]"
```

### 步骤 2：运行完整实验

```bash
# 运行所有场景（两个框架对比）
agent-compare run

# 或使用更大规模的参数
agent-compare run \
    --num-tasks 50 \
    --max-concurrency 10 \
    --num-instances 10 \
    --prompts-per-instance 5

# 模拟异常路径
agent-compare run --fail-subtask st-002
```

### 步骤 3：查看结果

```bash
# 查看生成的文件
ls -la artifacts/

# 查看汇总结果
cat artifacts/aggregate_*.json | python -m json.tool
```

### 步骤 4：运行验证测试

```bash
# 运行所有测试验证功能正确性
pytest tests/ -v
```

---

## 项目结构

```
12-agent-framework-comparison/
├── README.md                           # 本文件
├── pyproject.toml                      # 依赖管理
├── agent_compare/
│   ├── __init__.py
│   ├── cli.py                          # CLI 入口
│   ├── fake_llm/
│   │   ├── __init__.py
│   │   ├── base.py                     # 假 LLM 基类定义
│   │   └── rules.py                    # 可插拔响应规则
│   ├── langchain_impl/
│   │   ├── __init__.py
│   │   ├── adapter.py                  # LangChain 假模型适配器
│   │   ├── concurrent.py               # LangChain 并发场景
│   │   ├── multi_instance.py           # LangChain 多实例场景
│   │   └── multi_agent.py              # LangChain 多智能体场景
│   └── agentscope_impl/
│       ├── __init__.py
│       ├── adapter.py                  # AgentScope 假模型适配器
│       ├── concurrent.py               # AgentScope 并发场景
│       ├── multi_instance.py           # AgentScope 多实例场景
│       └── multi_agent.py              # AgentScope 多智能体场景
├── tests/
│   ├── __init__.py
│   └── test_framework_comparison.py    # 测试用例
├── docs/
│   └── FRAMEWORK_COMPARISON.md         # 详细对比报告
└── artifacts/                          # 运行输出（自动生成）
```

---

## 核心要点

### 1. LangChain 框架底层设计

- **设计哲学**：链式调用 (Chain-oriented)
- **核心抽象**：`Runnable`, `Chain`, `Agent`
- **执行模型**：同步优先，异步可选
- **Memory 管理**：分散式，开发者负责
- **工具调用**：Function Calling 格式
- **可观测性**：分层 Callback Handler

### 2. AgentScope 框架底层设计

- **设计哲学**：消息驱动 (Message-oriented)
- **核心抽象**：`Agent`, `Msg`, `Pipeline`
- **执行模型**：异步原生，Actor 模型
- **Memory 管理**：集中式，框架内置
- **工具调用**：Tool 装饰器 + 注册表
- **可观测性**：Monitor + 分布式追踪

### 3. 并发调度维度对比

| 维度 | LangChain | AgentScope |
|------|-----------|-------------|
| 并发原语 | `asyncio.gather`, `batch()` | `ParallelPipeline`, 异步消息 |
| 内置限流 | ❌ 无，需自行实现 | ⚠️ 部分支持 (Pipeline 级) |
| 分布式支持 | 需自行实现 | ✅ 内置 RPC 机制 |

### 4. 多实例管理维度对比

| 维度 | LangChain | AgentScope |
|------|-----------|-------------|
| 隔离机制 | 对象隔离（开发者负责） | ✅ 内置 Memory 隔离 |
| 共享控制 | 隐式（引用共享） | ✅ 显式配置 |
| 自动持久化 | ❌ 无，需自行实现 | ✅ 内置 save/load |

### 5. 多智能体协作维度对比

| 维度 | LangChain | AgentScope |
|------|-----------|-------------|
| 协作模式 | LangGraph (State), Plan-and-Execute | ✅ Pipeline, 广播, Hub-and-Spoke |
| 工作流编排 | 需 LangGraph 扩展 | ✅ 内置多种 Pipeline |
| 并行执行 | 需 asyncio 实现 | ✅ ParallelPipeline |

### 6. 选型适配场景建议

**推荐 LangChain**：
- 快速原型开发与实验
- 与现有 Python 生态集成
- 单 Agent 复杂任务
- 需要精细控制的场景

**推荐 AgentScope**：
- 多智能体系统设计
- 异步高并发场景
- 状态持久化需求
- 生产级多智能体应用

**详细对比报告**：请查看 `docs/FRAMEWORK_COMPARISON.md`

---

## 参考实现

本项目包含完整的可运行实现：

### 假 LLM 层

位于 `agent_compare/fake_llm/`：

- **`base.py`**：定义了 `FakeLLM`, `FakeLLMResponse`, `FakeChatMessage` 等核心抽象
- **`rules.py`**：提供多种可插拔的响应规则：
  - `SimpleEchoRule`：简单回显
  - `KeywordResponseRule`：关键词匹配响应
  - `CounterRule`：计数器（用于验证实例隔离）
  - `MultiRoleRule`：多角色差异化响应
  - `ErrorSimulatingRule`：错误模拟
  - `DelaySimulatingRule`：延迟模拟（用于并发测试）
  - `HashBasedRule`：哈希基响应（确保确定性）

### 框架适配器

位于 `agent_compare/langchain_impl/adapter.py` 和 `agent_compare/agentscope_impl/adapter.py`：

- **`LangChainFakeLLM`**：继承自 LangChain 的 `LLM` 基类，将假模型适配到 LangChain 接口
- **`AgentScopeFakeModel`**：模拟 AgentScope 的 `ModelWrapper` 接口

### 场景实现

三个场景在两套框架中分别实现，输出完全一致的 JSON 结构：

| 场景 | LangChain 实现 | AgentScope 实现 |
|------|---------------|-----------------|
| 并发调度 | `LangChainConcurrentScenario` | `AgentScopeConcurrentScenario` |
| 多实例管理 | `LangChainMultiInstanceScenario` | `AgentScopeMultiInstanceScenario` |
| 多智能体协作 | `LangChainMultiAgentScenario` | `AgentScopeMultiAgentScenario` |

### CLI 入口

位于 `agent_compare/cli.py`，使用 `typer` 框架实现：

```bash
# 查看帮助
agent-compare --help
agent-compare run --help

# 查看版本
agent-compare version
```

---

## 依赖版本

| 包名 | 版本范围 | 用途 |
|------|---------|------|
| langchain | >=0.1.0 | LangChain 核心框架 |
| langchain-core | >=0.1.0 | LangChain 核心抽象 |
| agentscope | >=0.1.0 | AgentScope 框架 |
| pydantic | >=2.0.0 | 数据验证 |
| typer | >=0.9.0 | CLI 框架 |
| rich | >=13.0.0 | 终端美化输出 |
| aiohttp | >=3.9.0 | 异步 HTTP |
| pytest | >=7.0.0 | 测试框架 |
| pytest-asyncio | >=0.21.0 | 异步测试支持 |

---

## 常见问题

### Q1: 运行失败时会有什么提示？

运行失败时会有明确的报错信息和非零退出码：

- **参数非法**：`typer` 会显示参数类型错误和使用说明
- **写入 artifacts 失败**：捕获异常并打印详细错误信息
- **依赖缺失**：Python 会抛出 `ImportError`，提示缺少的包名

### Q2: 如何验证两个框架的输出结构一致？

运行测试套件中的结构一致性测试：

```bash
pytest tests/test_framework_comparison.py::TestOutputStructureConsistency -v
```

### Q3: 假模型的输出是确定性的吗？

是的，本项目的假模型实现了确定性输出：

- `HashBasedRule`：基于输入内容的 MD5 哈希生成响应
- `KeywordResponseRule`：相同关键词返回固定响应
- `CounterRule`：基于 session_id 独立计数

运行以下测试验证：

```bash
pytest tests/test_framework_comparison.py::TestFakeLLMDeterminism -v
```

### Q4: 是否可以接入真实的 LLM API？

本项目设计为**离线可运行**，默认使用假模型。如果需要接入真实 API，可以：

1. 替换 `LangChainFakeLLM` 和 `AgentScopeFakeModel` 为真实模型实现
2. 保持场景实现不变，因为它们依赖的是抽象接口
3. 注意：需要配置 API 密钥（环境变量或配置文件）

---

## 许可证

本项目仅用于学习和研究目的。

---

## 贡献

欢迎提交 Issue 和 PR 来改进本项目。

