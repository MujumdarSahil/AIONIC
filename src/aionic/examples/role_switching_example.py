"""
Role Switching Example - Demonstrate agent role promotion/demotion.

Shows how agents adapt their roles based on task outcomes
and competence scores.
"""

from aionic.core.orchestrator import Orchestrator
from aionic.core.task import TaskPriority
from aionic.core.tool import ToolRegistry
from aionic.memory.memory_store import InMemoryMemoryStore
from aionic.memory.audit_logger import InMemoryAuditLogger
from aionic.security.autonomy_policy import AutonomyPolicy
from aionic.agents.base_agent import BaseAgent
from aionic.tools.base_tools import WebSearchTool, DataAnalysisTool


def setup_role_switching_demo():
    """Setup system for role switching demonstration."""
    
    memory_store = InMemoryMemoryStore()
    audit_logger = InMemoryAuditLogger()
    autonomy_policy = AutonomyPolicy()
    tool_registry = ToolRegistry()
    
    # Register tools
    tool_registry.register(WebSearchTool())
    tool_registry.register(DataAnalysisTool())
    
    # Create agent with low initial competence (will be Junior)
    agent = BaseAgent(
        agent_id="agent_demo_1",
        name="Learning Agent",
        goal="Learn and improve through task execution",
        memory=memory_store,
        autonomy_policy=autonomy_policy,
        tool_registry=tool_registry,
        initial_competence=0.3,  # Low - will start as Junior
    )
    agent.register_tools(["web_search", "data_analysis"])
    
    orchestrator = Orchestrator(
        agents=[agent],
        tool_registry=tool_registry,
        memory_store=memory_store,
        autonomy_policy=autonomy_policy,
        audit_logger=audit_logger,
    )
    
    return orchestrator, agent


def run_role_switching_example():
    """Demonstrate role switching through task execution."""
    
    print("=" * 60)
    print("AIONIC Framework - Role Switching Example")
    print("=" * 60)
    
    orchestrator, agent = setup_role_switching_demo()
    
    # Show initial state
    print("\nInitial Agent State:")
    print(f"  Role: {agent.role.value}")
    print(f"  Competence: {agent.competence_score:.3f}")
    
    # Execute multiple tasks to observe role changes
    print("\nExecuting tasks to observe role progression...")
    
    task_count = 5
    for i in range(task_count):
        task = orchestrator.submit_task(
            description=f"Task {i+1}: Search for information",
            objective=f"Find information about topic {i+1}",
            priority=TaskPriority.NORMAL,
        )
        
        try:
            orchestrator.execute_task(task.task_id)
            
            # Display agent state after each task
            print(f"\nAfter Task {i+1}:")
            print(f"  Role: {agent.role.value}")
            print(f"  Competence: {agent.competence_score:.3f}")
            print(f"  Success Rate: {agent.state.success_rate:.3f}")
            
        except Exception as e:
            print(f"\nTask {i+1} failed: {e}")
            print(f"  Role: {agent.role.value}")
            print(f"  Competence: {agent.competence_score:.3f}")
    
    print("\n" + "=" * 60)
    print("Role switching demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_role_switching_example()

