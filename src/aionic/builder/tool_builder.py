"""
Tool Builder - Natural language tool creation system.

Allows users to create tools using only natural language descriptions,
automatically inferring parameters, risk tiers, and implementation patterns.

Example:
    builder = ToolBuilder()
    tool = builder.create_from_description(
        "Create a tool that searches Wikipedia for articles"
    )
"""

import re
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..core.tool import ToolInterface, ToolResult, ToolCategory


@dataclass
class ToolConfig:
    """
    Configuration for a tool generated from natural language.
    
    Attributes:
        tool_id: Unique tool identifier
        name: Tool name
        description: Tool description
        category: Tool category
        parameters: Parameter schema
        risk_tier: Risk tier (LOW, MEDIUM, HIGH, CRITICAL)
        implementation_type: Type of implementation (simple, api, custom)
        metadata: Additional metadata
        version: Configuration version
        created_at: Creation timestamp
    """
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    risk_tier: str = "LOW"
    implementation_type: str = "simple"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "risk_tier": self.risk_tier,
            "implementation_type": self.implementation_type,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at,
        }


class GeneratedTool(ToolInterface):
    """Tool implementation generated from natural language description."""
    
    def __init__(self, config: ToolConfig, executor_func: Optional[Any] = None):
        """
        Initialize generated tool.
        
        Args:
            config: Tool configuration
            executor_func: Optional custom executor function
        """
        self._config = config
        self._executor_func = executor_func
    
    @property
    def name(self) -> str:
        return self._config.name
    
    @property
    def description(self) -> str:
        return self._config.description
    
    @property
    def category(self) -> ToolCategory:
        return self._config.category
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return self._config.parameters
    
    @property
    def risk_tier(self) -> str:
        return self._config.risk_tier
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute tool with provided parameters."""
        if self._executor_func:
            try:
                result = self._executor_func(**kwargs)
                return ToolResult(
                    success=True,
                    data=result,
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e),
                )
        else:
            # Default placeholder implementation
            return ToolResult(
                success=True,
                data={
                    "message": f"Tool '{self.name}' executed with parameters: {kwargs}",
                    "note": "This is a generated tool placeholder. Implement custom executor for production use.",
                },
                metadata={"tool_id": self._config.tool_id},
            )


class ToolBuilder:
    """
    Builder for creating tools from natural language descriptions.
    
    Parses natural language descriptions to infer:
    - Tool name and description
    - Required parameters
    - Risk tier
    - Category
    - Implementation pattern
    """
    
    # Category inference patterns
    CATEGORY_PATTERNS: Dict[ToolCategory, List[str]] = {
        ToolCategory.INFORMATION: ["search", "lookup", "find", "get", "retrieve", "read", "fetch"],
        ToolCategory.RESEARCH: ["research", "investigate", "analyze", "study", "explore"],
        ToolCategory.COMPUTATION: ["calculate", "compute", "math", "numeric", "statistics"],
        ToolCategory.AUTOMATION: ["automate", "execute", "run", "process", "workflow"],
        ToolCategory.COMMUNICATION: ["send", "message", "email", "notify", "communicate"],
        ToolCategory.ANALYSIS: ["analyze", "analysis", "evaluate", "assess", "examine"],
        ToolCategory.CREATIVE: ["generate", "create", "write", "compose", "design"],
        ToolCategory.SYSTEM: ["system", "file", "directory", "process", "command"],
    }
    
    # Risk tier inference patterns
    RISK_KEYWORDS: Dict[str, List[str]] = {
        "CRITICAL": ["delete", "remove", "execute code", "system", "admin", "root", "critical"],
        "HIGH": ["write", "modify", "update", "change", "database", "file write"],
        "MEDIUM": ["read", "query", "search", "file read", "access"],
        "LOW": ["search", "lookup", "get", "retrieve", "fetch", "read-only"],
    }
    
    def __init__(self):
        """Initialize tool builder."""
        self._created_tools: Dict[str, ToolConfig] = {}
    
    def create_from_description(
        self,
        description: str,
        tool_name: Optional[str] = None,
        executor_func: Optional[Any] = None,
    ) -> GeneratedTool:
        """
        Create a tool from a natural language description.
        
        Args:
            description: Natural language description of the tool
            tool_name: Optional tool name (inferred if None)
            executor_func: Optional custom executor function
            
        Returns:
            Created tool instance
            
        Example:
            tool = builder.create_from_description(
                "Create a tool that searches Wikipedia for articles"
            )
        """
        # Parse description
        config = self._parse_description(description, tool_name)
        
        # Validate and adjust config
        config = self._validate_config(config)
        
        # Create tool instance
        tool = GeneratedTool(config, executor_func)
        
        # Store config
        self._created_tools[config.tool_id] = config
        
        return tool
    
    def _parse_description(self, description: str, tool_name: Optional[str] = None) -> ToolConfig:
        """Parse natural language description into ToolConfig."""
        description_lower = description.lower()
        
        # Extract or infer tool name
        if tool_name is None:
            tool_name = self._extract_name(description)
        
        # Infer category
        category = self._infer_category(description_lower)
        
        # Extract parameters
        parameters = self._extract_parameters(description)
        
        # Infer risk tier
        risk_tier = self._infer_risk_tier(description_lower)
        
        # Infer implementation type
        implementation_type = self._infer_implementation_type(description_lower)
        
        # Generate tool ID
        tool_id = f"tool_{tool_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        return ToolConfig(
            tool_id=tool_id,
            name=tool_name,
            description=description,
            category=category,
            parameters=parameters,
            risk_tier=risk_tier,
            implementation_type=implementation_type,
        )
    
    def _extract_name(self, description: str) -> str:
        """Extract tool name from description."""
        # Look for patterns like "tool that X", "tool for X", "X tool"
        patterns = [
            r"tool\s+(?:that|for|to)\s+([a-z]+(?:\s+[a-z]+)*)",
            r"([a-z]+(?:\s+[a-z]+)*)\s+tool",
            r"create\s+a\s+([a-z]+(?:\s+[a-z]+)*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Convert to snake_case
                name = name.replace(" ", "_")
                return name
        
        # Default: use first few words
        words = description.split()[:3]
        return "_".join(w.lower() for w in words)
    
    def _infer_category(self, description_lower: str) -> ToolCategory:
        """Infer tool category from description."""
        for category, keywords in self.CATEGORY_PATTERNS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        return ToolCategory.INFORMATION  # Default
    
    def _extract_parameters(self, description: str) -> Dict[str, Dict[str, Any]]:
        """Extract parameters from description."""
        parameters = {}
        
        # Look for common parameter patterns
        param_patterns = [
            (r"query|search\s+term", "query", "string"),
            (r"url|link|address", "url", "string"),
            (r"file\s+path|path\s+to\s+file", "file_path", "string"),
            (r"text|content|message", "text", "string"),
            (r"number|count|limit|max", "limit", "integer"),
        ]
        
        description_lower = description.lower()
        for pattern, param_name, param_type in param_patterns:
            if re.search(pattern, description_lower):
                parameters[param_name] = {
                    "type": param_type,
                    "description": f"{param_name.replace('_', ' ').title()} parameter",
                    "required": True,
                }
        
        # If no parameters found, add a generic "input" parameter
        if not parameters:
            parameters["input"] = {
                "type": "string",
                "description": "Input parameter",
                "required": True,
            }
        
        return parameters
    
    def _infer_risk_tier(self, description_lower: str) -> str:
        """Infer risk tier from description."""
        for risk_tier, keywords in self.RISK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return risk_tier
        return "LOW"  # Default to safe
    
    def _infer_implementation_type(self, description_lower: str) -> str:
        """Infer implementation type from description."""
        if any(kw in description_lower for kw in ["api", "http", "rest", "endpoint"]):
            return "api"
        elif any(kw in description_lower for kw in ["custom", "complex", "advanced"]):
            return "custom"
        else:
            return "simple"
    
    def _validate_config(self, config: ToolConfig) -> ToolConfig:
        """Validate and adjust tool configuration."""
        # Ensure name is valid (alphanumeric and underscores)
        config.name = re.sub(r"[^a-z0-9_]", "_", config.name.lower())
        
        # Ensure risk tier is valid
        if config.risk_tier not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            config.risk_tier = "LOW"
        
        # Ensure at least one parameter
        if not config.parameters:
            config.parameters = {
                "input": {
                    "type": "string",
                    "description": "Input parameter",
                    "required": True,
                }
            }
        
        return config
    
    def get_config(self, tool_id: str) -> Optional[ToolConfig]:
        """Get configuration for a created tool."""
        return self._created_tools.get(tool_id)
    
    def list_created_tools(self) -> List[str]:
        """List IDs of all created tools."""
        return list(self._created_tools.keys())
    
    def export_config(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Export tool configuration as JSON-serializable dict."""
        config = self._created_tools.get(tool_id)
        if config:
            return config.to_dict()
        return None

