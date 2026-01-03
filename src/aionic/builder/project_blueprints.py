"""
Project Blueprints - Pre-configured project templates.

Provides ready-to-use project configurations for common use cases:
- RAG Research Assistant
- Recommendation Engine Agent
- Cyber-threat Analysis Agent
- Knowledge-graph Explorer
- Multi-agent Workflow Executor

Users can create complete projects with agents, tools, and configurations
using simple natural language commands.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .agent_builder import AgentBuilder, AgentConfig
from .tool_builder import ToolBuilder
from ..core.tool import ToolRegistry
from ..memory.memory_store import MemoryStore
from ..security.autonomy_policy import AutonomyPolicy
from ..llm.config import LLMConfigLoader
from ..llm.router import LLMRouter


@dataclass
class ProjectBlueprint:
    """
    Blueprint for a complete project configuration.
    
    Attributes:
        name: Blueprint name
        description: Blueprint description
        agents: List of agent configurations
        tools: List of tool configurations
        llm_config: LLM configuration
        routing_strategy: LLM routing strategy
        metadata: Additional metadata
        version: Blueprint version
        created_at: Creation timestamp
    """
    name: str
    description: str
    agents: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    llm_config: Dict[str, Any] = field(default_factory=dict)
    routing_strategy: str = "quality_optimized"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize blueprint to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "agents": self.agents,
            "tools": self.tools,
            "llm_config": self.llm_config,
            "routing_strategy": self.routing_strategy,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at,
        }


class BlueprintManager:
    """Manager for project blueprints."""
    
    BLUEPRINTS: Dict[str, Dict[str, Any]] = {
        "rag_research_assistant": {
            "name": "RAG Research Assistant",
            "description": "Complete RAG system for document research and retrieval",
            "agents": [
                {
                    "description": "Create a RAG Research Assistant agent that retrieves and analyzes documents using web search and file reading tools",
                    "name": "RAG Research Assistant",
                }
            ],
            "tools": ["web_search", "file_read"],
            "llm_config": {
                "primary_model": "gpt-4",
                "fallback_models": ["gpt-3.5-turbo", "claude-3"],
            },
            "routing_strategy": "quality_optimized",
        },
        "recommendation_engine": {
            "name": "Recommendation Engine Agent",
            "description": "Agent for generating personalized recommendations",
            "agents": [
                {
                    "description": "Create a Recommendation Engine agent that analyzes user preferences and generates recommendations using data analysis tools",
                    "name": "Recommendation Engine",
                }
            ],
            "tools": ["data_analysis"],
            "llm_config": {
                "primary_model": "gpt-4",
                "fallback_models": ["gpt-3.5-turbo"],
            },
            "routing_strategy": "cost_optimized",
        },
        "cyber_threat_analysis": {
            "name": "Cyber-threat Analysis Agent",
            "description": "Agent for analyzing cybersecurity threats and vulnerabilities",
            "agents": [
                {
                    "description": "Create a Cyber-threat Analysis agent that researches security threats using web search and data analysis tools",
                    "name": "Cyber-threat Analyst",
                }
            ],
            "tools": ["web_search", "data_analysis", "file_read"],
            "llm_config": {
                "primary_model": "claude-3",
                "fallback_models": ["gpt-4"],
            },
            "routing_strategy": "quality_optimized",
        },
        "knowledge_graph_explorer": {
            "name": "Knowledge-graph Explorer",
            "description": "Agent for exploring and querying knowledge graphs",
            "agents": [
                {
                    "description": "Create a Knowledge-graph Explorer agent that queries and navigates knowledge graphs using database and search tools",
                    "name": "Knowledge-graph Explorer",
                }
            ],
            "tools": ["database_query", "web_search"],
            "llm_config": {
                "primary_model": "gpt-4",
                "fallback_models": ["claude-3"],
            },
            "routing_strategy": "quality_optimized",
        },
        "multi_agent_workflow": {
            "name": "Multi-agent Workflow Executor",
            "description": "Orchestrated multi-agent system for complex workflows",
            "agents": [
                {
                    "description": "Create a Research Agent that gathers information using web search",
                    "name": "Research Agent",
                },
                {
                    "description": "Create an Analysis Agent that processes data using data analysis tools",
                    "name": "Analysis Agent",
                },
                {
                    "description": "Create an Automation Agent that executes workflows using code execution tools",
                    "name": "Automation Agent",
                },
            ],
            "tools": ["web_search", "data_analysis", "code_execution"],
            "llm_config": {
                "primary_model": "gpt-4",
                "fallback_models": ["gpt-3.5-turbo", "claude-3"],
            },
            "routing_strategy": "provider_fallback",
        },
    }
    
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        memory_store: Optional[MemoryStore] = None,
        autonomy_policy: Optional[AutonomyPolicy] = None,
        llm_config_loader: Optional[LLMConfigLoader] = None,
    ):
        """
        Initialize blueprint manager.
        
        Args:
            tool_registry: Tool registry
            memory_store: Memory store
            autonomy_policy: Autonomy policy
            llm_config_loader: LLM config loader
        """
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory_store = memory_store
        self.autonomy_policy = autonomy_policy or AutonomyPolicy()
        self.llm_config_loader = llm_config_loader or LLMConfigLoader()
        
        self.agent_builder = AgentBuilder(
            tool_registry=self.tool_registry,
            memory_store=self.memory_store,
            autonomy_policy=self.autonomy_policy,
        )
        self.tool_builder = ToolBuilder()
    
    def create_blueprint(
        self,
        blueprint_name: str,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> ProjectBlueprint:
        """
        Create a project from a blueprint.
        
        Args:
            blueprint_name: Name of blueprint to use
            custom_config: Optional custom configuration overrides
            
        Returns:
            ProjectBlueprint instance
            
        Example:
            blueprint = manager.create_blueprint("rag_research_assistant")
        """
        blueprint_def = self.BLUEPRINTS.get(blueprint_name)
        if not blueprint_def:
            raise ValueError(f"Blueprint '{blueprint_name}' not found")
        
        # Merge with custom config
        if custom_config:
            blueprint_def = {**blueprint_def, **custom_config}
        
        # Create LLM router
        llm_router = self.llm_config_loader.create_router(
            default_strategy=self._get_routing_strategy(blueprint_def.get("routing_strategy", "quality_optimized")),
        )
        
        # Create agents
        agent_configs = []
        for agent_desc in blueprint_def.get("agents", []):
            if isinstance(agent_desc, dict):
                description = agent_desc.get("description", "")
                agent = self.agent_builder.create_from_description(
                    description,
                    memory_store=self.memory_store,
                    autonomy_policy=self.autonomy_policy,
                    llm_router=llm_router,
                )
                agent_configs.append({
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "config": self.agent_builder.export_config(agent.agent_id),
                })
            else:
                # Legacy string format
                agent = self.agent_builder.create_from_description(
                    agent_desc,
                    memory_store=self.memory_store,
                    autonomy_policy=self.autonomy_policy,
                    llm_router=llm_router,
                )
                agent_configs.append({
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "config": self.agent_builder.export_config(agent.agent_id),
                })
        
        # Create tools (if not already in registry)
        tool_configs = []
        for tool_name in blueprint_def.get("tools", []):
            # Check if tool exists in registry
            if not self.tool_registry.get(tool_name):
                # Create tool from description
                tool_desc = f"Create a {tool_name.replace('_', ' ')} tool"
                tool = self.tool_builder.create_from_description(tool_desc, tool_name=tool_name)
                self.tool_registry.register(tool)
                tool_configs.append(self.tool_builder.export_config(tool._config.tool_id))
        
        return ProjectBlueprint(
            name=blueprint_def["name"],
            description=blueprint_def["description"],
            agents=agent_configs,
            tools=tool_configs,
            llm_config=blueprint_def.get("llm_config", {}),
            routing_strategy=blueprint_def.get("routing_strategy", "quality_optimized"),
            metadata={
                "blueprint_name": blueprint_name,
                "created_from": "blueprint",
            },
        )
    
    def _get_routing_strategy(self, strategy_name: str):
        """Get routing strategy enum from name."""
        from ..llm.router import RoutingStrategy
        strategy_map = {
            "cost_optimized": RoutingStrategy.COST_OPTIMIZED,
            "speed_optimized": RoutingStrategy.SPEED_OPTIMIZED,
            "quality_optimized": RoutingStrategy.QUALITY_OPTIMIZED,
            "provider_fallback": RoutingStrategy.PROVIDER_FALLBACK,
        }
        return strategy_map.get(strategy_name, RoutingStrategy.QUALITY_OPTIMIZED)
    
    def list_blueprints(self) -> List[str]:
        """List available blueprint names."""
        return list(self.BLUEPRINTS.keys())
    
    def get_blueprint_info(self, blueprint_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a blueprint."""
        return self.BLUEPRINTS.get(blueprint_name)


def create_blueprint(
    blueprint_name: str,
    tool_registry: Optional[ToolRegistry] = None,
    memory_store: Optional[MemoryStore] = None,
    autonomy_policy: Optional[AutonomyPolicy] = None,
    custom_config: Optional[Dict[str, Any]] = None,
) -> ProjectBlueprint:
    """
    Convenience function to create a project blueprint.
    
    Args:
        blueprint_name: Name of blueprint
        tool_registry: Optional tool registry
        memory_store: Optional memory store
        autonomy_policy: Optional autonomy policy
        custom_config: Optional custom configuration
        
    Returns:
        ProjectBlueprint instance
    """
    manager = BlueprintManager(
        tool_registry=tool_registry,
        memory_store=memory_store,
        autonomy_policy=autonomy_policy,
    )
    return manager.create_blueprint(blueprint_name, custom_config)

