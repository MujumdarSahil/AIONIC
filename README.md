# AIONIC - Autonomous Intelligence Orchestration Network

A production-grade multi-agent framework supporting dynamic role-switching agents, tool orchestration, risk-aware execution permissions, and explainable reasoning.

## Features

- **Dynamic Role-Switching Agents**: Agents adapt their roles based on competence and performance
- **Tool Orchestration**: Extensible tool interface with permission-based access control
- **Risk-Aware Execution**: Multi-tier permission system based on agent role and action risk
- **Explainable Reasoning**: Comprehensive logging and reasoning trails
- **Memory & Governance**: Persistent memory system and audit logging
- **Modular Architecture**: Clean, extensible design for custom agents and tools

## Architecture

```
AIONIC Framework
├── Core Components
│   ├── AgentBase - Base agent abstraction
│   ├── Orchestrator - Task planning and agent coordination
│   ├── Task - Task lifecycle management
│   ├── Context - Execution context sharing
│   └── ToolInterface - Extensible tool abstraction
├── Memory System
│   ├── MemoryStore - Persistent memory interface
│   └── AuditLogger - Comprehensive audit trail
├── Security
│   └── AutonomyPolicy - Risk-aware permission system
├── Agents
│   ├── BaseAgent - Default agent implementation
│   ├── RAGAgent - Retrieval-Augmented Generation
│   ├── AutomationAgent - Task automation specialist
│   └── ResearchAgent - Research and analysis
└── Tools
    └── Base tools (WebSearch, FileRead, DataAnalysis, etc.)
```

## Quick Start

```python
from aionic.core.orchestrator import Orchestrator
from aionic.core.task import TaskPriority
from aionic.core.tool import ToolRegistry
from aionic.memory.memory_store import InMemoryMemoryStore
from aionic.memory.audit_logger import InMemoryAuditLogger
from aionic.security.autonomy_policy import AutonomyPolicy
from aionic.agents.rag_agent import RAGAgent
from aionic.tools.base_tools import WebSearchTool

# Setup components
memory_store = InMemoryMemoryStore()
audit_logger = InMemoryAuditLogger()
autonomy_policy = AutonomyPolicy()
tool_registry = ToolRegistry()

# Register tools
tool_registry.register(WebSearchTool())

# Create agent
agent = RAGAgent(
    agent_id="rag_1",
    name="RAG Agent",
    memory=memory_store,
    autonomy_policy=autonomy_policy,
    tool_registry=tool_registry,
)

# Create orchestrator
orchestrator = Orchestrator(
    agents=[agent],
    tool_registry=tool_registry,
    memory_store=memory_store,
    autonomy_policy=autonomy_policy,
    audit_logger=audit_logger,
)

# Submit and execute task
task = orchestrator.submit_task(
    description="Research AI trends",
    objective="Find recent information about AI trends in 2024",
    priority=TaskPriority.HIGH,
)

result = orchestrator.execute_task(task.task_id)
print(result)
```

## Installation

```bash
pip install -e .
```

## Documentation

See `docs/` directory for detailed documentation.

## Beginner Mode - No-Code Agent Creation

AIONIC now includes a beginner-friendly builder system that allows you to create agents, tools, and complete projects using only natural language descriptions - no coding required!

### Creating Agents from Natural Language

```python
from aionic.builder import AgentBuilder
from aionic.core.tool import ToolRegistry
from aionic.memory.memory_store import InMemoryMemoryStore
from aionic.security.autonomy_policy import AutonomyPolicy
from aionic.tools.base_tools import WebSearchTool, FileReadTool
from aionic.llm.config import LLMConfigLoader

# Setup
tool_registry = ToolRegistry()
tool_registry.register(WebSearchTool())
tool_registry.register(FileReadTool())

memory_store = InMemoryMemoryStore()
autonomy_policy = AutonomyPolicy()
llm_config = LLMConfigLoader()
llm_router = llm_config.create_router()

# Create agent builder
builder = AgentBuilder(
    tool_registry=tool_registry,
    memory_store=memory_store,
    autonomy_policy=autonomy_policy,
)

# Create agent from natural language description
agent = builder.create_from_description(
    "Create an agent named Market Analyst who analyzes startup funding trends "
    "and uses web search + file reading tools",
    llm_router=llm_router,
)

print(f"Created: {agent.name}")
print(f"Goal: {agent.goal}")
print(f"Tools: {agent.get_available_tools()}")
```

### Creating Tools from Natural Language

```python
from aionic.builder import ToolBuilder

builder = ToolBuilder()

# Create tool from description
tool = builder.create_from_description(
    "Create a tool that searches Wikipedia for articles",
    tool_name="wikipedia_search",
)

# Use the tool
result = tool.execute(query="Python programming")
print(result.data)
```

### Creating Projects from Blueprints

```python
from aionic.builder import create_blueprint

# Create a complete RAG Research Assistant project
blueprint = create_blueprint(
    "rag_research_assistant",
    tool_registry=tool_registry,
    memory_store=memory_store,
    autonomy_policy=autonomy_policy,
)

print(f"Project: {blueprint.name}")
print(f"Agents: {len(blueprint.agents)}")
```

### Available Blueprints

- `rag_research_assistant` - Complete RAG system for document research
- `recommendation_engine` - Personalized recommendation agent
- `cyber_threat_analysis` - Security threat analysis agent
- `knowledge_graph_explorer` - Knowledge graph navigation agent
- `multi_agent_workflow` - Orchestrated multi-agent system

### Unified LLM Configuration

All LLM providers are configured in one place:

```python
from aionic.llm.config import LLMConfigLoader, load_llm_from_name

# Load configuration (reads from .env file)
config = LLMConfigLoader()

# Create router with all configured providers
router = config.create_router()

# Load specific model by friendly name
provider = load_llm_from_name("gpt-4")
# or
provider = load_llm_from_name("claude-3")
# or
provider = load_llm_from_name("gemini-pro")
```

### Environment Configuration

Create a `.env` file in your project root:

```env
# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_ENABLED=true

# Claude (Anthropic)
ANTHROPIC_API_KEY=your_key_here
CLAUDE_ENABLED=true

# Gemini (Google)
GEMINI_API_KEY=your_key_here
GEMINI_ENABLED=true

# Groq
GROQ_API_KEY=your_key_here
GROQ_ENABLED=true

# Local/Ollama (optional)
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_MODEL=llama2
LOCAL_ENABLED=true
```

### Research-Publishable Properties

Every generated agent and tool includes:
- **Metadata**: Version, creation timestamp, configuration trace
- **Reproducibility**: Exportable JSON configurations
- **Audit Logging**: All decisions and actions logged
- **Telemetry**: Routing decisions, execution latency, cost estimation
- **Deterministic Generation**: Same input produces same output

## Examples

See `src/aionic/examples/` for example implementations:
- `basic_example.py` - Basic framework usage
- `role_switching_example.py` - Agent role adaptation demonstration
- `builder_demo.py` - Beginner mode examples (NEW!)

## License

MIT License

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

