"""
Tool Interface - Extensible tool abstraction for agent capabilities.

Defines the interface that all tools must implement, enabling
agents to interact with external systems and services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolCategory(Enum):
    """Categories of tools."""
    INFORMATION = "information"
    COMPUTATION = "computation"
    AUTOMATION = "automation"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    CREATIVE = "creative"
    SYSTEM = "system"


@dataclass
class ToolResult:
    """
    Result from tool execution.
    
    Attributes:
        success: Whether execution was successful
        data: Result data
        error: Error message if failed
        metadata: Additional execution metadata
        execution_time_ms: Execution time in milliseconds
    """
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    execution_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ToolInterface(ABC):
    """
    Abstract base class for all tools.
    
    All tools must implement this interface to be usable by agents.
    Tools define their capabilities, parameters, and execution logic.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        """Tool category classification."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Tool parameter schema in JSON Schema format.
        
        Returns:
            Dictionary mapping parameter names to their schema definitions.
        """
        pass
    
    @property
    @abstractmethod
    def risk_tier(self) -> str:
        """
        Risk tier of this tool (used for permission checking).
        
        Returns:
            Risk tier string (e.g., "LOW", "MEDIUM", "HIGH", "CRITICAL")
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with provided parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate parameters before execution.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Basic validation against schema
        required_params = {
            name: schema for name, schema in self.parameters.items()
            if schema.get("required", False)
        }
        
        for param_name, schema in required_params.items():
            if param_name not in kwargs:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "risk_tier": self.risk_tier,
        }


class ToolRegistry:
    """
    Central registry for managing available tools.
    
    Maintains a catalog of registered tools and provides
    lookup and discovery capabilities.
    """
    
    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, ToolInterface] = {}
    
    def register(self, tool: ToolInterface) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[ToolInterface]:
        """
        Get tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolInterface]:
        """
        List all registered tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool instances
        """
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools
    
    def list_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

