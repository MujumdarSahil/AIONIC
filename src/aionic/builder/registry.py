"""
Enhanced Registry System - Auto-registration and discovery.

Provides enhanced discovery capabilities for agents, tools, and LLM providers
created through the builder system.
"""

from typing import Dict, List, Optional, Any
from ..core.tool import ToolRegistry
from ..llm.registry import ProviderRegistry
from ..llm.config import LLMConfigLoader


class DiscoveryRegistry:
    """
    Enhanced registry with discovery capabilities.
    
    Provides unified discovery for:
    - Created agents (from builder)
    - Registered tools
    - Available LLM providers
    - Compatibility matrix
    """
    
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        provider_registry: Optional[ProviderRegistry] = None,
        llm_config_loader: Optional[LLMConfigLoader] = None,
    ):
        """
        Initialize discovery registry.
        
        Args:
            tool_registry: Tool registry
            provider_registry: Provider registry
            llm_config_loader: LLM config loader
        """
        self.tool_registry = tool_registry or ToolRegistry()
        self.provider_registry = provider_registry
        self.llm_config_loader = llm_config_loader or LLMConfigLoader()
        self._agent_registry: Dict[str, Any] = {}
    
    def register_agent(self, agent_id: str, agent: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an agent in the discovery registry.
        
        Args:
            agent_id: Agent identifier
            agent: Agent instance
            metadata: Optional metadata
        """
        self._agent_registry[agent_id] = {
            "agent": agent,
            "metadata": metadata or {},
        }
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents.
        
        Returns:
            List of agent information dictionaries
        """
        agents = []
        for agent_id, entry in self._agent_registry.items():
            agent = entry["agent"]
            agents.append({
                "agent_id": agent_id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "goal": agent.goal,
                "role": agent.role.value,
                "competence_score": agent.competence_score,
                "tools": agent.get_available_tools(),
                "metadata": entry["metadata"],
            })
        return agents
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools.
        
        Returns:
            List of tool information dictionaries
        """
        tools = []
        for tool in self.tool_registry.list_tools():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "risk_tier": tool.risk_tier,
                "parameters": list(tool.parameters.keys()),
            })
        return tools
    
    def list_llm_providers(self) -> List[Dict[str, Any]]:
        """
        List all available LLM providers.
        
        Returns:
            List of provider information dictionaries
        """
        providers = []
        
        # From config loader
        available_providers = self.llm_config_loader.list_available_providers()
        for provider_name in available_providers:
            config = self.llm_config_loader.get_provider_config(provider_name)
            if config:
                providers.append({
                    "name": provider_name,
                    "enabled": config.enabled,
                    "has_api_key": config.api_key is not None,
                })
        
        # From registry if available
        if self.provider_registry:
            for provider_name in self.provider_registry.list_providers():
                metadata = self.provider_registry.get_metadata(provider_name)
                if metadata:
                    providers.append({
                        "name": provider_name,
                        "display_name": metadata.display_name,
                        "supported_models": metadata.supported_models,
                        "supports_embeddings": metadata.supports_embeddings,
                        "cost_per_1k_tokens": metadata.cost_per_1k_tokens,
                    })
        
        return providers
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        return self.llm_config_loader.list_available_models()
    
    def get_compatibility_matrix(self) -> Dict[str, Any]:
        """
        Get compatibility matrix showing agent-tool-LLM compatibility.
        
        Returns:
            Compatibility matrix dictionary
        """
        matrix = {
            "agents": len(self._agent_registry),
            "tools": len(self.tool_registry.list_tool_names()),
            "llm_providers": len(self.llm_config_loader.list_available_providers()),
            "llm_models": len(self.llm_config_loader.list_available_models()),
            "agent_tool_compatibility": {},
        }
        
        # Build agent-tool compatibility
        for agent_id, entry in self._agent_registry.items():
            agent = entry["agent"]
            matrix["agent_tool_compatibility"][agent_id] = {
                "agent_name": agent.name,
                "available_tools": agent.get_available_tools(),
                "tool_count": len(agent.get_available_tools()),
            }
        
        return matrix
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all registered components."""
        return {
            "agents": {
                "count": len(self._agent_registry),
                "list": [a["name"] for a in self.list_agents()],
            },
            "tools": {
                "count": len(self.tool_registry.list_tool_names()),
                "list": [t["name"] for t in self.list_tools()],
            },
            "llm_providers": {
                "count": len(self.llm_config_loader.list_available_providers()),
                "list": [p["name"] for p in self.list_llm_providers()],
            },
            "llm_models": {
                "count": len(self.llm_config_loader.list_available_models()),
                "list": self.llm_config_loader.list_available_models(),
            },
            "compatibility_matrix": self.get_compatibility_matrix(),
        }

