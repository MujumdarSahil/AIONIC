# AIONIC Architecture Documentation

## High-Level Architecture

AIONIC follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│              (User Code, Examples, Integrations)             │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                        │
│                  (Orchestrator, Task Planning)               │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
│  Agent Layer │    │  Agent Layer │    │  Agent Layer │
│  (RAG)       │    │ (Automation) │    │  (Research)  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                        │
│    (Memory, Security, Tools, Logging, Context Management)    │
└─────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Core Components (`src/aionic/core/`)

#### 1.1 AgentBase (`core/agent.py`)
**Purpose**: Abstract base class for all agents

**Key Attributes:**
- `agent_id`: Unique identifier
- `name`: Human-readable name
- `goal`: Agent's primary objective
- `state`: AgentState with competence, role, metrics
- `memory`: MemoryStore instance
- `autonomy_policy`: Permission policy
- `tool_registry`: Available tools

**Key Methods:**
- `reason(task, context)`: Generate reasoning and plan
- `execute_task(task, context)`: Execute task
- `use_tool(tool_name, **kwargs)`: Execute tool with permission checking
- `update_competence(task_success)`: Update competence score

**Role System:**
- Junior (0.0-0.4)
- Associate (0.4-0.6)
- Senior (0.6-0.8)
- Expert (0.8-0.95)
- Architect (0.95+)

#### 1.2 Orchestrator (`core/orchestrator.py`)
**Purpose**: Central coordination system

**Responsibilities:**
1. **Task Management**: Create, track, and manage tasks
2. **Agent Selection**: Score-based agent assignment
3. **Task Planning**: Decompose complex tasks
4. **Collaboration**: Coordinate multi-agent tasks
5. **Conflict Resolution**: Arbitrate agent conflicts

**Key Methods:**
- `submit_task(...)`: Create new task
- `assign_task(task, agent_id)`: Assign task to agent
- `execute_task(task_id)`: Execute task
- `select_agent(task)`: Select best agent for task
- `coordinate_collaboration(...)`: Setup multi-agent collaboration

#### 1.3 Task (`core/task.py`)
**Purpose**: Task representation and lifecycle

**States:**
- PENDING → ASSIGNED → IN_PROGRESS → COMPLETED
- Can also be: BLOCKED, FAILED, CANCELLED

**Attributes:**
- Task metadata (id, description, objective)
- Status and timestamps
- Dependencies and subtasks
- Result and error information

#### 1.4 Context (`core/context.py`)
**Purpose**: Shared execution context

**Features:**
- Mutable state dictionary
- Metadata storage
- Context hierarchy (parent contexts)
- Type classification (TASK, SESSION, WORKFLOW, RESEARCH)

#### 1.5 ToolInterface (`core/tool.py`)
**Purpose**: Extensible tool abstraction

**Tool Interface:**
- `name`: Unique tool identifier
- `description`: Human-readable description
- `category`: Tool category (INFORMATION, COMPUTATION, etc.)
- `parameters`: JSON Schema parameter definition
- `risk_tier`: Risk level (NONE, LOW, MEDIUM, HIGH, CRITICAL)
- `execute(**kwargs)`: Execute tool logic

**ToolRegistry**: Central registry for tool discovery and management

### 2. Memory System (`src/aionic/memory/`)

#### 2.1 MemoryStore (`memory/memory_store.py`)
**Purpose**: Persistent memory for agents

**Memory Types:**
- **Episodic**: Specific events/experiences
- **Semantic**: Facts and knowledge
- **Procedural**: How-to knowledge
- **Working**: Temporary active memory

**Operations:**
- `store(...)`: Store memory with importance score
- `retrieve(...)`: Retrieve memories by query/type
- `update_importance(...)`: Adjust memory importance
- `delete(...)`: Remove memory

**Implementation**: `InMemoryMemoryStore` (extensible to persistent storage)

#### 2.2 AuditLogger (`memory/audit_logger.py`)
**Purpose**: Comprehensive audit trail

**Log Categories:**
- TASK, AGENT, TOOL, ORCHESTRATION, SECURITY, MEMORY, SYSTEM

**Log Levels:**
- DEBUG, INFO, WARNING, ERROR, CRITICAL

**Features:**
- Structured logging with metadata
- Filtering by category, level, agent, task
- Convenience methods for common events

**Implementation**: `InMemoryAuditLogger` (extensible to persistent logging)

### 3. Security (`src/aionic/security/`)

#### 3.1 AutonomyPolicy (`security/autonomy_policy.py`)
**Purpose**: Risk-aware permission system

**Risk Tiers:**
- NONE, LOW, MEDIUM, HIGH, CRITICAL

**Permission Rules:**
- Role-based: Minimum role required for risk tier
- Competence-based: Minimum competence score
- Tool-specific: Custom rules per agent-tool combination

**Features:**
- Blacklist/whitelist agents
- Approval requirements for CRITICAL operations
- Policy customization

### 4. Agents (`src/aionic/agents/`)

#### 4.1 BaseAgent (`agents/base_agent.py`)
**Purpose**: Default concrete agent implementation

**Features:**
- Default reasoning logic
- Tool usage patterns
- Memory integration
- Competence tracking

#### 4.2 RAGAgent (`agents/rag_agent.py`)
**Purpose**: Retrieval-Augmented Generation specialist

**Capabilities:**
- Document retrieval
- Knowledge base querying
- Context-aware generation
- Semantic search

#### 4.3 AutomationAgent (`agents/automation_agent.py`)
**Purpose**: Task automation specialist

**Capabilities:**
- Workflow execution
- Batch processing
- Script execution (with permissions)
- Process orchestration

#### 4.4 ResearchAgent (`agents/research_agent.py`)
**Purpose**: Research and analysis specialist

**Capabilities:**
- Multi-source information gathering
- Source verification
- Analysis and synthesis
- Report generation

### 5. Tools (`src/aionic/tools/`)

#### 5.1 Base Tools (`tools/base_tools.py`)

**WebSearchTool**: Web search capability (LOW risk)
**FileReadTool**: File reading (MEDIUM risk)
**DataAnalysisTool**: Statistical analysis (LOW risk)
**CodeExecutionTool**: Code execution (CRITICAL risk)
**DatabaseQueryTool**: Database queries (HIGH risk)

### 6. Examples (`src/aionic/examples/`)

- `basic_example.py`: Simple framework usage
- `role_switching_example.py`: Role adaptation demonstration

## Agent Lifecycle

### Initialization
1. Create agent with initial competence
2. Register tools
3. Set goal and configuration

### Task Execution Cycle
1. **Task Submission**: User/orchestrator submits task
2. **Task Planning**: Orchestrator decomposes task (if needed)
3. **Agent Selection**: Orchestrator selects best agent
4. **Permission Check**: Verify agent can execute required tools
5. **Task Assignment**: Assign task to agent
6. **Reasoning**: Agent generates reasoning and plan
7. **Execution**: Agent executes task using tools
8. **Competence Update**: Update agent competence based on outcome
9. **Role Update**: Update role if competence crosses threshold
10. **Memory Storage**: Store execution result in memory
11. **Audit Logging**: Log all activities

### Role Adaptation
- **Promotion**: Competence increases → role upgrade
- **Demotion**: Competence decreases → role downgrade
- **Hysteresis**: Thresholds prevent oscillation

## Data Flow

```
Task → Orchestrator → Agent Selection → Permission Check
                                          ↓
                                    Agent Execution
                                          ↓
                                    Tool Execution
                                          ↓
                                    Result → Memory
                                          ↓
                                    Competence Update
                                          ↓
                                    Role Update (if needed)
```

## Extension Points

### Adding a New Agent
1. Inherit from `AgentBase`
2. Implement `reason()` method
3. Implement `execute_task()` method
4. Register required tools
5. Set agent type

### Adding a New Tool
1. Inherit from `ToolInterface`
2. Implement all abstract properties
3. Implement `execute()` method
4. Define risk tier
5. Register with `ToolRegistry`

### Adding Custom Memory Backend
1. Inherit from `MemoryStore`
2. Implement all abstract methods
3. Use in agent initialization

### Customizing Permission Policy
1. Extend `AutonomyPolicy`
2. Override `can_execute_tool()` method
3. Add custom rules

## Design Principles

1. **Separation of Concerns**: Clear boundaries between components
2. **Dependency Inversion**: Depend on abstractions, not concretions
3. **Open/Closed**: Open for extension, closed for modification
4. **Single Responsibility**: Each class has one clear purpose
5. **Interface Segregation**: Small, focused interfaces
6. **Minimal Dependencies**: Framework stays lightweight

## Scalability Considerations

### Current Limitations
- In-memory storage (memory and logs)
- Single-node execution
- Synchronous task execution

### Future Extensions
- Database-backed memory stores
- Distributed agent execution
- Async task execution
- Message queue integration
- Caching layers

## Security Considerations

1. **Tool Execution**: All tool executions go through permission checks
2. **Risk Stratification**: Multi-tier risk system
3. **Approval Workflows**: CRITICAL operations require approval
4. **Audit Trail**: Complete logging of all activities
5. **Blacklisting**: Ability to blacklist agents/tools
6. **Input Validation**: Tool parameter validation

## Performance Considerations

1. **Agent Selection**: O(n) scoring for n agents
2. **Memory Retrieval**: O(m) for m memories (can be optimized with indexing)
3. **Tool Execution**: Varies by tool (external dependencies)
4. **Orchestration Overhead**: Minimal (<5% of task time)

## Testing Strategy

1. **Unit Tests**: Each component independently
2. **Integration Tests**: Component interactions
3. **End-to-End Tests**: Full task execution flows
4. **Performance Tests**: Benchmarking and profiling
5. **Security Tests**: Permission system validation

