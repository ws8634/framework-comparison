import sys
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "langchain: Tests that require LangChain framework"
    )
    config.addinivalue_line(
        "markers", "agentscope: Tests that require AgentScope framework"
    )
    config.addinivalue_line(
        "markers", "both_frameworks: Tests that require both LangChain and AgentScope"
    )
    config.addinivalue_line(
        "markers", "fake_llm: Tests for fake LLM implementation (no framework required)"
    )


@pytest.fixture(scope="session")
def langchain_available():
    """Check if LangChain is available."""
    try:
        import langchain
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def agentscope_available():
    """Check if AgentScope is available and supported."""
    if sys.version_info >= (3, 13):
        return False
    try:
        import agentscope
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def both_frameworks_available(langchain_available, agentscope_available):
    """Check if both frameworks are available."""
    return langchain_available and agentscope_available


def skip_if_no_langchain():
    """Decorator to skip tests that require LangChain."""
    return pytest.mark.skipif(
        not _check_langchain(),
        reason="LangChain not installed or not available",
    )


def skip_if_no_agentscope():
    """Decorator to skip tests that require AgentScope."""
    return pytest.mark.skipif(
        not _check_agentscope(),
        reason="AgentScope not installed or Python >= 3.13",
    )


def skip_if_no_both_frameworks():
    """Decorator to skip tests that require both frameworks."""
    return pytest.mark.skipif(
        not (_check_langchain() and _check_agentscope()),
        reason="Both LangChain and AgentScope required",
    )


def _check_langchain():
    """Internal function to check LangChain availability."""
    try:
        import langchain
        return True
    except ImportError:
        return False


def _check_agentscope():
    """Internal function to check AgentScope availability."""
    if sys.version_info >= (3, 13):
        return False
    try:
        import agentscope
        return True
    except ImportError:
        return False


def get_available_frameworks():
    """Get list of available frameworks."""
    available = []
    if _check_langchain():
        available.append("langchain")
    if _check_agentscope():
        available.append("agentscope")
    return available


def get_installation_guide():
    """Get installation guide based on current environment."""
    lines = []
    lines.append("Installation Guide:")
    lines.append("")
    
    if not _check_langchain():
        lines.append("To install LangChain:")
        lines.append("  pip install -e \".[langchain]\"")
        lines.append("  or")
        lines.append("  pip install langchain langchain-core")
        lines.append("")
    
    if not _check_agentscope():
        if sys.version_info >= (3, 13):
            lines.append("⚠️  AgentScope does not support Python 3.13+")
            lines.append("   Use Python 3.9-3.12 to run AgentScope tests")
        else:
            lines.append("To install AgentScope:")
            lines.append("  pip install -e \".[agentscope]\"")
            lines.append("  or")
            lines.append("  pip install agentscope")
        lines.append("")
    
    lines.append("To install all dependencies (Python < 3.13):")
    lines.append("  pip install -e \".[langchain,agentscope,dev]\"")
    lines.append("")
    lines.append("To install only LangChain and dev dependencies (Python >= 3.13):")
    lines.append("  pip install -e \".[langchain,dev]\"")
    
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def print_framework_info():
    """Print framework availability info at session start."""
    pass


def pytest_sessionstart(session):
    """Print framework availability at session start."""
    print("\n" + "=" * 60)
    print("Framework Availability Check")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")
    
    langchain_ok = _check_langchain()
    agentscope_ok = _check_agentscope()
    
    print(f"\nLangChain: {'✅ Available' if langchain_ok else '❌ Unavailable'}")
    print(f"AgentScope: {'✅ Available' if agentscope_ok else '❌ Unavailable'}")
    
    if sys.version_info >= (3, 13):
        print(f"\n⚠️  Note: Python >= 3.13 detected")
        print(f"   AgentScope is not supported on Python 3.13+")
        print(f"   Only LangChain tests will be run")
    
    available = get_available_frameworks()
    if available:
        print(f"\nAvailable frameworks: {', '.join(available)}")
    else:
        print(f"\n❌ No frameworks available!")
        print(f"\n{get_installation_guide()}")
    
    print("=" * 60 + "\n")
