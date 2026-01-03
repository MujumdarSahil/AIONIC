"""
Builder Demo - Demonstration of beginner-friendly agent and tool creation.

Shows how to use the AIONIC Builder to create agents, tools, and projects
using only natural language descriptions - no coding required!

Examples:
1. Create an agent from description
2. Create a tool from description
3. Create a complete project from blueprint
"""

from aionic.builder import AgentBuilder, ToolBuilder, create_blueprint
from aionic.core.tool import ToolRegistry
from aionic.memory.memory_store import InMemoryMemoryStore
from aionic.security.autonomy_policy import AutonomyPolicy
from aionic.tools.base_tools import WebSearchTool, FileReadTool, DataAnalysisTool
from aionic.llm.config import LLMConfigLoader


def demo_agent_creation():
    """Demonstrate creating agents from natural language."""
    print("=" * 60)
    print("Demo 1: Creating Agents from Natural Language")
    print("=" * 60)
    
    # Setup
    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    tool_registry.register(FileReadTool())
    tool_registry.register(DataAnalysisTool())
    
    memory_store = InMemoryMemoryStore()
    autonomy_policy = AutonomyPolicy()
    
    # Create LLM router
    llm_config = LLMConfigLoader()
    llm_router = llm_config.create_router()
    
    builder = AgentBuilder(
        tool_registry=tool_registry,
        memory_store=memory_store,
        autonomy_policy=autonomy_policy,
    )
    
    # Example 1: Market Analyst Agent
    print("\n1. Creating Market Analyst Agent...")
    agent1 = builder.create_from_description(
        "Create an agent named Market Analyst who analyzes startup funding trends "
        "and uses web search + file reading tools",
        llm_router=llm_router,
    )
    print(f"   ✓ Created: {agent1.name} (ID: {agent1.agent_id})")
    print(f"   ✓ Type: {agent1.agent_type}")
    print(f"   ✓ Goal: {agent1.goal}")
    print(f"   ✓ Tools: {agent1.get_available_tools()}")
    
    # Example 2: RAG Agent
    print("\n2. Creating RAG Research Assistant...")
    agent2 = builder.create_from_description(
        "Create a document-summarization agent with RAG capabilities using Gemini + Groq",
        llm_router=llm_router,
    )
    print(f"   ✓ Created: {agent2.name} (ID: {agent2.agent_id})")
    print(f"   ✓ Type: {agent2.agent_type}")
    print(f"   ✓ Goal: {agent2.goal}")
    
    # Export configurations
    print("\n3. Exporting agent configurations...")
    config1 = builder.export_config(agent1.agent_id)
    config2 = builder.export_config(agent2.agent_id)
    print(f"   ✓ Agent 1 config: {list(config1.keys())}")
    print(f"   ✓ Agent 2 config: {list(config2.keys())}")
    
    return builder, agent1, agent2


def demo_tool_creation():
    """Demonstrate creating tools from natural language."""
    print("\n" + "=" * 60)
    print("Demo 2: Creating Tools from Natural Language")
    print("=" * 60)
    
    builder = ToolBuilder()
    
    # Example 1: Wikipedia Search Tool
    print("\n1. Creating Wikipedia Search Tool...")
    tool1 = builder.create_from_description(
        "Create a tool that searches Wikipedia for articles",
        tool_name="wikipedia_search",
    )
    print(f"   ✓ Created: {tool1.name}")
    print(f"   ✓ Category: {tool1.category.value}")
    print(f"   ✓ Risk Tier: {tool1.risk_tier}")
    print(f"   ✓ Parameters: {list(tool1.parameters.keys())}")
    
    # Example 2: Email Sender Tool
    print("\n2. Creating Email Sender Tool...")
    tool2 = builder.create_from_description(
        "Create a tool that sends emails to recipients",
        tool_name="email_sender",
    )
    print(f"   ✓ Created: {tool2.name}")
    print(f"   ✓ Category: {tool2.category.value}")
    print(f"   ✓ Risk Tier: {tool2.risk_tier}")
    
    # Test tool execution
    print("\n3. Testing tool execution...")
    result = tool1.execute(query="Python programming")
    print(f"   ✓ Tool executed: {result.success}")
    if result.data:
        print(f"   ✓ Result: {str(result.data)[:100]}...")
    
    return builder, tool1, tool2


def demo_project_blueprint():
    """Demonstrate creating projects from blueprints."""
    print("\n" + "=" * 60)
    print("Demo 3: Creating Projects from Blueprints")
    print("=" * 60)
    
    # Setup
    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    tool_registry.register(FileReadTool())
    tool_registry.register(DataAnalysisTool())
    
    memory_store = InMemoryMemoryStore()
    autonomy_policy = AutonomyPolicy()
    
    # Create RAG Research Assistant blueprint
    print("\n1. Creating RAG Research Assistant Project...")
    blueprint = create_blueprint(
        "rag_research_assistant",
        tool_registry=tool_registry,
        memory_store=memory_store,
        autonomy_policy=autonomy_policy,
    )
    print(f"   ✓ Project: {blueprint.name}")
    print(f"   ✓ Description: {blueprint.description}")
    print(f"   ✓ Agents: {len(blueprint.agents)}")
    for agent_info in blueprint.agents:
        print(f"     - {agent_info['name']} ({agent_info['agent_id']})")
    print(f"   ✓ Tools: {len(blueprint.tools)}")
    print(f"   ✓ LLM Config: {blueprint.llm_config}")
    
    # Create Multi-agent Workflow blueprint
    print("\n2. Creating Multi-agent Workflow Project...")
    blueprint2 = create_blueprint(
        "multi_agent_workflow",
        tool_registry=tool_registry,
        memory_store=memory_store,
        autonomy_policy=autonomy_policy,
    )
    print(f"   ✓ Project: {blueprint2.name}")
    print(f"   ✓ Agents: {len(blueprint2.agents)}")
    for agent_info in blueprint2.agents:
        print(f"     - {agent_info['name']} ({agent_info['agent_id']})")
    
    return blueprint, blueprint2


def demo_discovery():
    """Demonstrate discovery capabilities."""
    print("\n" + "=" * 60)
    print("Demo 4: Discovery and Registry")
    print("=" * 60)
    
    # LLM Config Discovery
    llm_config = LLMConfigLoader()
    print("\n1. Available LLM Providers:")
    providers = llm_config.list_available_providers()
    for provider in providers:
        print(f"   ✓ {provider}")
    
    print("\n2. Available Models:")
    models = llm_config.list_available_models()
    for model in models[:10]:  # Show first 10
        print(f"   ✓ {model}")
    if len(models) > 10:
        print(f"   ... and {len(models) - 10} more")
    
    # Tool Registry Discovery
    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    tool_registry.register(FileReadTool())
    tool_registry.register(DataAnalysisTool())
    
    print("\n3. Available Tools:")
    tools = tool_registry.list_tool_names()
    for tool in tools:
        tool_obj = tool_registry.get(tool)
        print(f"   ✓ {tool} ({tool_obj.category.value}, {tool_obj.risk_tier})")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AIONIC Builder Demo - Beginner-Friendly Agent Creation")
    print("=" * 60)
    print("\nThis demo shows how to create agents, tools, and projects")
    print("using only natural language - no coding required!\n")
    
    try:
        # Demo 1: Agent Creation
        builder, agent1, agent2 = demo_agent_creation()
        
        # Demo 2: Tool Creation
        tool_builder, tool1, tool2 = demo_tool_creation()
        
        # Demo 3: Project Blueprints
        blueprint1, blueprint2 = demo_project_blueprint()
        
        # Demo 4: Discovery
        demo_discovery()
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  • Agents can be created from natural language descriptions")
        print("  • Tools are automatically inferred and configured")
        print("  • Projects can be created from pre-configured blueprints")
        print("  • All components auto-register in the system")
        print("  • Configurations are exportable for reproducibility")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

