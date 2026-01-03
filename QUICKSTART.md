# AIONIC Framework - Quick Start Guide

## Installation

```bash
# Clone or navigate to project directory
cd AIONIC

# Install in development mode
pip install -e .
```

## Basic Usage

### 1. Setup Components

```python
from aionic.core.orchestrator import Orchestrator
from aionic.core.task import TaskPriority
from aionic.core.tool import ToolRegistry
from aionic.memory.memory_store import InMemoryMemoryStore
from aionic.memory.audit_logger import InMemoryAuditLogger
from aionic.security.autonomy_policy import AutonomyPolicy
from aionic.agents.rag_agent import RAGAgent
from aionic.tools.base_tools import WebSearchTool

# Initialize core components
memory_store = InMemoryMemoryStore()
audit_logger = InMemoryAuditLogger()
autonomy_policy = AutonomyPolicy()
tool_registry = ToolRegistry()

# Register tools
tool_registry.register(WebSearchTool())
```

### 2. Create Agents

```python
# Create a RAG agent
rag_agent = RAGAgent(
    agent_id="rag_1",
    name="My RAG Agent",
    memory=memory_store,
    autonomy_policy=autonomy_policy,
    tool_registry=tool_registry,
    initial_competence=0.6,
)
```

### 3. Create Orchestrator

```python
orchestrator = Orchestrator(
    agents=[rag_agent],
    tool_registry=tool_registry,
    memory_store=memory_store,
    autonomy_policy=autonomy_policy,
    audit_logger=audit_logger,
)
```

### 4. Submit and Execute Tasks

```python
# Submit a task
task = orchestrator.submit_task(
    description="Research Python frameworks",
    objective="Find information about Python web frameworks",
    priority=TaskPriority.HIGH,
)

# Execute the task
result = orchestrator.execute_task(task.task_id)
print(result)
```

### 5. Monitor System State

```python
# Get system state
state = orchestrator.get_system_state()
print(f"Total tasks: {state['total_tasks']}")
print(f"Completed: {state['completed_tasks']}")

# Check agent status
for agent_id, info in state['agents'].items():
    print(f"{info['name']}: {info['role']} (competence: {info['competence_score']:.2f})")
```

## Creating Custom Agents

```python
from aionic.core.agent import AgentBase
from aionic.core.task import Task
from aionic.core.context import Context

class MyCustomAgent(AgentBase):
    @property
    def agent_type(self) -> str:
        return "custom"
    
    def reason(self, task: Task, context: Context) -> dict:
        # Implement your reasoning logic
        return {
            "strategy": "my_custom_strategy",
            "steps": ["step1", "step2"],
        }
    
    def execute_task(self, task: Task, context: Context) -> any:
        # Implement your execution logic
        reasoning = self.reason(task, context)
        # Execute steps...
        return {"result": "success"}
```

## Creating Custom Tools

```python
from aionic.core.tool import ToolInterface, ToolResult, ToolCategory

class MyCustomTool(ToolInterface):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "Description of my tool"
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.INFORMATION
    
    @property
    def parameters(self) -> dict:
        return {
            "param1": {
                "type": "string",
                "description": "Parameter description",
                "required": True,
            },
        }
    
    @property
    def risk_tier(self) -> str:
        return "LOW"  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    
    def execute(self, **kwargs) -> ToolResult:
        # Implement tool logic
        param1 = kwargs.get("param1")
        # ... tool execution ...
        return ToolResult(success=True, data={"result": "data"})
```

## Running Examples

```bash
# Basic example
python src/aionic/examples/basic_example.py

# Role switching example
python src/aionic/examples/role_switching_example.py
```

## Key Concepts

### Roles
- **Junior** (0.0-0.4): Basic tasks, low-risk tools
- **Associate** (0.4-0.6): Standard tasks, medium-risk tools
- **Senior** (0.6-0.8): Complex tasks, high-risk tools
- **Expert** (0.8-0.95): Advanced tasks, critical tools (with approval)
- **Architect** (0.95+): System-level tasks

### Risk Tiers
- **NONE**: No risk
- **LOW**: Minimal risk
- **MEDIUM**: Moderate risk
- **HIGH**: Significant risk
- **CRITICAL**: Critical risk (requires approval)

### Task States
- PENDING → ASSIGNED → IN_PROGRESS → COMPLETED
- Can also be: BLOCKED, FAILED, CANCELLED

## Next Steps

- Read `docs/ARCHITECTURE.md` for detailed architecture
- Check `docs/RESEARCH_PAPER_FRAMEWORK.md` for research details
- Explore examples in `src/aionic/examples/`
- See `PROJECT_OVERVIEW.md` for complete feature list

## Support

For questions or issues, refer to the documentation or create an issue in the repository.

