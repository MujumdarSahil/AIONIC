"""
Basic Example - Simple demonstration of AIONIC framework usage.

Shows how to:
- Initialize core components
- Create agents
- Register tools
- Submit and execute tasks
- Monitor system state
"""

from aionic.core.orchestrator import Orchestrator
from aionic.core.task import TaskPriority
from aionic.core.tool import ToolRegistry
from aionic.memory.memory_store import InMemoryMemoryStore
from aionic.memory.audit_logger import InMemoryAuditLogger
from aionic.security.autonomy_policy import AutonomyPolicy
from aionic.agents.base_agent import BaseAgent
from aionic.agents.rag_agent import RAGAgent
from aionic.agents.research_agent import ResearchAgent
from aionic.tools.base_tools import (
    WebSearchTool,
    FileReadTool,
    DataAnalysisTool,
)


def setup_system():
    """Setup AIONIC system with basic components."""
    
    # Initialize core components
    memory_store = InMemoryMemoryStore()
    audit_logger = InMemoryAuditLogger()
    autonomy_policy = AutonomyPolicy()
    tool_registry = ToolRegistry()
    
    # Register tools
    tool_registry.register(WebSearchTool())
    tool_registry.register(FileReadTool())
    tool_registry.register(DataAnalysisTool())
    
    # Create agents
    base_agent = BaseAgent(
        agent_id="agent_base_1",
        name="Base Agent",
        goal="Execute general tasks",
        memory=memory_store,
        autonomy_policy=autonomy_policy,
        tool_registry=tool_registry,
        initial_competence=0.6,
    )
    base_agent.register_tools(["web_search", "data_analysis"])
    
    rag_agent = RAGAgent(
        agent_id="agent_rag_1",
        name="RAG Agent",
        memory=memory_store,
        autonomy_policy=autonomy_policy,
        tool_registry=tool_registry,
        initial_competence=0.7,
    )
    
    research_agent = ResearchAgent(
        agent_id="agent_research_1",
        name="Research Agent",
        memory=memory_store,
        autonomy_policy=autonomy_policy,
        tool_registry=tool_registry,
        initial_competence=0.65,
    )
    
    agents = [base_agent, rag_agent, research_agent]
    
    # Create orchestrator
    orchestrator = Orchestrator(
        agents=agents,
        tool_registry=tool_registry,
        memory_store=memory_store,
        autonomy_policy=autonomy_policy,
        audit_logger=audit_logger,
    )
    
    return orchestrator, memory_store, audit_logger


def run_basic_example():
    """Run basic example demonstration."""
    
    print("=" * 60)
    print("AIONIC Framework - Basic Example")
    print("=" * 60)
    
    # Setup system
    orchestrator, memory_store, audit_logger = setup_system()
    
    # Submit tasks
    print("\n1. Submitting tasks...")
    
    task1 = orchestrator.submit_task(
        description="Search for information about Python",
        objective="Find recent information about Python programming language",
        priority=TaskPriority.NORMAL,
    )
    print(f"   Task created: {task1.task_id}")
    
    task2 = orchestrator.submit_task(
        description="Research multi-agent systems",
        objective="What are the key characteristics of multi-agent systems?",
        priority=TaskPriority.HIGH,
    )
    print(f"   Task created: {task2.task_id}")
    
    # Execute tasks
    print("\n2. Executing tasks...")
    
    try:
        result1 = orchestrator.execute_task(task1.task_id)
        print(f"   Task {task1.task_id} completed")
        print(f"   Result keys: {list(result1.keys()) if isinstance(result1, dict) else 'N/A'}")
    except Exception as e:
        print(f"   Task {task1.task_id} failed: {e}")
    
    try:
        result2 = orchestrator.execute_task(task2.task_id)
        print(f"   Task {task2.task_id} completed")
        print(f"   Result keys: {list(result2.keys()) if isinstance(result2, dict) else 'N/A'}")
    except Exception as e:
        print(f"   Task {task2.task_id} failed: {e}")
    
    # Display system state
    print("\n3. System State:")
    state = orchestrator.get_system_state()
    print(f"   Total tasks: {state['total_tasks']}")
    print(f"   Completed tasks: {state['completed_tasks']}")
    print(f"   Registered agents: {state['registered_agents']}")
    
    # Display agent information
    print("\n4. Agent Status:")
    for agent_id, agent_info in state['agents'].items():
        print(f"   {agent_info['name']} ({agent_id}):")
        print(f"      Role: {agent_info['role']}")
        print(f"      Competence: {agent_info['competence_score']:.2f}")
        print(f"      Success Rate: {agent_info['success_rate']:.2f}")
        print(f"      Tasks Completed: {agent_info['tasks_completed']}")
    
    # Display recent logs
    print("\n5. Recent Logs:")
    recent_logs = audit_logger.get_recent_logs(limit=5)
    for log in recent_logs:
        print(f"   [{log.level.value.upper()}] {log.message}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_basic_example()

