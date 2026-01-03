"""
AIONIC - Autonomous Intelligence Orchestration Network

A production-grade multi-agent framework supporting:
- Dynamic role-switching agents
- Tool orchestration
- Risk-aware execution permissions
- Explainable reasoning
- Memory, logging, and governance
"""

__version__ = "0.1.0"
__author__ = "AIONIC Team"

from .core.agent import AgentBase
from .core.orchestrator import Orchestrator
from .core.task import Task, TaskStatus, TaskPriority
from .core.context import Context
from .memory.memory_store import MemoryStore
from .security.autonomy_policy import AutonomyPolicy, RiskTier

# LLM Integration
from .llm.router import LLMRouter, RoutingStrategy
from .llm.registry import ProviderRegistry
from .llm.models import ModelConfig, LLMResponse, ChatMessage, EmbeddingResponse
from .llm.base import LLMProviderBase
from .llm.config import LLMConfigLoader, load_llm_from_name

# Builder (Beginner Mode)
from .builder import (
    AgentBuilder,
    AgentConfig,
    ToolBuilder,
    ToolConfig,
    ProjectBlueprint,
    create_blueprint,
    DiscoveryRegistry,
)

__all__ = [
    "AgentBase",
    "Orchestrator",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "Context",
    "MemoryStore",
    "AutonomyPolicy",
    "RiskTier",
    # LLM Integration
    "LLMRouter",
    "RoutingStrategy",
    "ProviderRegistry",
    "ModelConfig",
    "LLMResponse",
    "ChatMessage",
    "EmbeddingResponse",
    "LLMProviderBase",
    "LLMConfigLoader",
    "load_llm_from_name",
    # Builder (Beginner Mode)
    "AgentBuilder",
    "AgentConfig",
    "ToolBuilder",
    "ToolConfig",
    "ProjectBlueprint",
    "create_blueprint",
    "DiscoveryRegistry",
]

