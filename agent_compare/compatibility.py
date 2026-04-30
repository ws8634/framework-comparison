import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class FrameworkAvailability(str, Enum):
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    NOT_SUPPORTED = "not_supported"


@dataclass
class FrameworkStatus:
    name: str
    availability: FrameworkAvailability
    version: Optional[str] = None
    error_message: Optional[str] = None

    def is_available(self) -> bool:
        return self.availability == FrameworkAvailability.AVAILABLE


@dataclass
class CompatibilityReport:
    python_version: str
    python_version_info: tuple
    langchain: FrameworkStatus
    agentscope: FrameworkStatus

    def get_available_frameworks(self) -> list:
        available = []
        if self.langchain.is_available():
            available.append("langchain")
        if self.agentscope.is_available():
            available.append("agentscope")
        return available

    def has_at_least_one_framework(self) -> bool:
        return (
            self.langchain.is_available()
            or self.agentscope.is_available()
        )


def get_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_langchain() -> FrameworkStatus:
    try:
        import langchain
        version = getattr(langchain, "__version__", "unknown")
        return FrameworkStatus(
            name="langchain",
            availability=FrameworkAvailability.AVAILABLE,
            version=version,
        )
    except ImportError as e:
        return FrameworkStatus(
            name="langchain",
            availability=FrameworkAvailability.NOT_INSTALLED,
            error_message=str(e),
        )
    except Exception as e:
        return FrameworkStatus(
            name="langchain",
            availability=FrameworkAvailability.NOT_SUPPORTED,
            error_message=str(e),
        )


def check_agentscope() -> FrameworkStatus:
    python_major = sys.version_info.major
    python_minor = sys.version_info.minor
    
    if python_major >= 3 and python_minor >= 13:
        return FrameworkStatus(
            name="agentscope",
            availability=FrameworkAvailability.NOT_SUPPORTED,
            error_message=(
                f"AgentScope 不支持 Python {python_major}.{python_minor}。"
                f"当前支持的最高 Python 版本为 3.12。"
            ),
        )
    
    try:
        import agentscope
        version = getattr(agentscope, "__version__", "unknown")
        return FrameworkStatus(
            name="agentscope",
            availability=FrameworkAvailability.AVAILABLE,
            version=version,
        )
    except ImportError as e:
        return FrameworkStatus(
            name="agentscope",
            availability=FrameworkAvailability.NOT_INSTALLED,
            error_message=str(e),
        )
    except Exception as e:
        return FrameworkStatus(
            name="agentscope",
            availability=FrameworkAvailability.NOT_SUPPORTED,
            error_message=str(e),
        )


def check_compatibility() -> CompatibilityReport:
    return CompatibilityReport(
        python_version=get_python_version(),
        python_version_info=(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
        langchain=check_langchain(),
        agentscope=check_agentscope(),
    )


def format_compatibility_report(report: CompatibilityReport) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append("兼容性检查报告")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Python 版本: {report.python_version}")
    lines.append("")
    lines.append("框架状态:")
    lines.append("")
    
    for fw_name, fw_status in [
        ("LangChain", report.langchain),
        ("AgentScope", report.agentscope),
    ]:
        status_icon = "✅" if fw_status.is_available() else "❌"
        lines.append(f"  {status_icon} {fw_name}")
        lines.append(f"     状态: {fw_status.availability.value}")
        if fw_status.version:
            lines.append(f"     版本: {fw_status.version}")
        if fw_status.error_message:
            lines.append(f"     问题: {fw_status.error_message}")
        lines.append("")
    
    available = report.get_available_frameworks()
    if available:
        lines.append(f"可用框架: {', '.join(available)}")
    else:
        lines.append("⚠️  没有可用的框架！")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


def get_installation_guide(report: CompatibilityReport) -> str:
    lines = []
    lines.append("安装指南:")
    lines.append("")
    
    if not report.langchain.is_available():
        lines.append("安装 LangChain:")
        lines.append("  pip install -e \".[langchain]\"")
        lines.append("  或")
        lines.append("  pip install langchain langchain-core")
        lines.append("")
    
    if not report.agentscope.is_available():
        if sys.version_info >= (3, 13):
            lines.append("⚠️  AgentScope 不支持 Python 3.13+")
            lines.append("   请使用 Python 3.9-3.12 运行 AgentScope 相关测试")
        else:
            lines.append("安装 AgentScope:")
            lines.append("  pip install -e \".[agentscope]\"")
            lines.append("  或")
            lines.append("  pip install agentscope")
        lines.append("")
    
    lines.append("安装所有依赖（Python < 3.13）:")
    lines.append("  pip install -e \".[langchain,agentscope,dev]\"")
    lines.append("")
    lines.append("仅安装 LangChain 和开发依赖（Python >= 3.13）:")
    lines.append("  pip install -e \".[langchain,dev]\"")
    
    return "\n".join(lines)


EXIT_CODE_SUCCESS = 0
EXIT_CODE_INVALID_FRAMEWORK = 1
EXIT_CODE_NO_FRAMEWORKS = 2
EXIT_CODE_MISSING_DEPENDENCY = 3
EXIT_CODE_RUNTIME_ERROR = 4
