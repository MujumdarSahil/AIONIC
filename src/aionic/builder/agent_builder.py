"""
Agent Builder - Natural language agent creation system.

Allows users to create agents using only natural language descriptions,
automatically inferring role, goal, profile, autonomy level, and tool mappings.

Example:
    builder = AgentBuilder()
    agent = builder.create_from_description(
        "Create an agent named Market Analyst who analyzes startup funding trends "
        "and uses web search + file writer tools"
    )
"""

import re
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

from ..core.agent import AgentBase, AgentRole
from ..agents.base_agent import BaseAgent
from ..agents.rag_agent import RAGAgent
from ..agents.research_agent import ResearchAgent
from ..agents.automation_agent import AutomationAgent
from ..core.tool import ToolRegistry, ToolCategory
from ..security.autonomy_policy import AutonomyPolicy, RiskTier
from ..memory.memory_store import MemoryStore


@dataclass
class AgentConfig:
    """
    Configuration for an agent generated from natural language.
    
    Attributes:
        agent_id: Unique agent identifier
        name: Agent name
        goal: Agent's primary goal
        agent_type: Type of agent (base, rag, research, automation)
        initial_competence: Initial competence score
        tools: List of tool names to register
        autonomy_level: Autonomy level (conservative, moderate, aggressive)
        profile: Agent profile/description
        metadata: Additional metadata
        version: Configuration version
        created_at: Creation timestamp
    """
    agent_id: str
    name: str
    goal: str
    agent_type: str = "base"
    initial_competence: float = 0.5
    tools: List[str] = field(default_factory=list)
    autonomy_level: str = "moderate"
    profile: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "goal": self.goal,
            "agent_type": self.agent_type,
            "initial_competence": self.initial_competence,
            "tools": self.tools,
            "autonomy_level": self.autonomy_level,
            "profile": self.profile,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at,
        }


class AgentBuilder:
    """
    Builder for creating agents from natural language descriptions.
    
    Parses natural language descriptions to infer:
    - Agent name and role
    - Goal and objectives
    - Required tools and capabilities
    - Autonomy level
    - Agent type (RAG, research, automation, etc.)
    """
    
    # Tool name mappings from natural language
    TOOL_KEYWORDS: Dict[str, List[str]] = {
        "web_search": ["web search", "search", "internet search", "google", "browse"],
        "file_read": ["file read", "read file", "read document", "file"],
        "file_write": ["file write", "write file", "save file", "write document"],
        "data_analysis": ["analyze", "analysis", "data analysis", "statistics", "compute"],
        "code_execution": ["code", "execute code", "run code", "programming"],
        "database_query": ["database", "query database", "sql", "db"],
    }
    
    # Agent type inference patterns
    AGENT_TYPE_PATTERNS: Dict[str, List[str]] = {
        "rag": ["rag", "retrieval", "document", "knowledge base", "semantic search", "embedding"],
        "research": ["research", "investigate", "study", "analyze trends", "market research"],
        "automation": ["automate", "workflow", "process", "system operation", "task automation"],
        "base": [],  # Default
    }
    
    # Autonomy level inference
    AUTONOMY_KEYWORDS: Dict[str, List[str]] = {
        "conservative": ["safe", "conservative", "careful", "low risk", "restricted"],
        "aggressive": ["aggressive", "autonomous", "independent", "full control", "unrestricted"],
        "moderate": [],  # Default
    }
    
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        memory_store: Optional[MemoryStore] = None,
        autonomy_policy: Optional[AutonomyPolicy] = None,
    ):
        """
        Initialize agent builder.
        
        Args:
            tool_registry: Tool registry for agent creation
            memory_store: Memory store for agents
            autonomy_policy: Autonomy policy for agents
        """
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory_store = memory_store
        self.autonomy_policy = autonomy_policy or AutonomyPolicy()
        self._created_agents: Dict[str, AgentConfig] = {}
    
    def create_from_description(
        self,
        description: str,
        agent_id: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
        autonomy_policy: Optional[AutonomyPolicy] = None,
        llm_router: Optional[Any] = None,
    ) -> BaseAgent:
        """
        Create an agent from a natural language description.
        
        Args:
            description: Natural language description of the agent
            agent_id: Optional agent ID (generated if None)
            memory_store: Optional memory store (uses builder default if None)
            autonomy_policy: Optional autonomy policy (uses builder default if None)
            llm_router: Optional LLM router for the agent
            
        Returns:
            Created agent instance
            
        Example:
            agent = builder.create_from_description(
                "Create a Market Analyst agent that analyzes startup funding trends "
                "using web search and file reading tools"
            )
        """
        # Parse description
        config = self._parse_description(description, agent_id)
        
        # Validate and adjust config
        config = self._validate_config(config)
        
        # Create agent instance
        agent = self._create_agent_instance(
            config,
            memory_store or self.memory_store,
            autonomy_policy or self.autonomy_policy,
            llm_router,
        )
        
        # Store config
        self._created_agents[config.agent_id] = config
        
        return agent
    
    def _parse_description(self, description: str, agent_id: Optional[str] = None) -> AgentConfig:
        """Parse natural language description into AgentConfig."""
        description_lower = description.lower()
        
        # Extract agent name
        name = self._extract_name(description)
        
        # Extract goal
        goal = self._extract_goal(description, name)
        
        # Infer agent type
        agent_type = self._infer_agent_type(description_lower)
        
        # Extract tools
        tools = self._extract_tools(description_lower)
        
        # Infer autonomy level
        autonomy_level = self._infer_autonomy_level(description_lower)
        
        # Infer initial competence based on description
        initial_competence = self._infer_competence(description_lower, agent_type)
        
        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = f"agent_{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        return AgentConfig(
            agent_id=agent_id,
            name=name,
            goal=goal,
            agent_type=agent_type,
            initial_competence=initial_competence,
            tools=tools,
            autonomy_level=autonomy_level,
            profile=description,
        )
    
    def _extract_name(self, description: str) -> str:
        """Extract agent name from description."""
        # Look for patterns like "named X", "called X", "agent X"
        patterns = [
            r"named\s+([A-Z][a-zA-Z\s]+?)(?:\s+who|\s+that|\s+with|$)",
            r"called\s+([A-Z][a-zA-Z\s]+?)(?:\s+who|\s+that|\s+with|$)",
            r"agent\s+([A-Z][a-zA-Z\s]+?)(?:\s+who|\s+that|\s+with|$)",
            r"create\s+an?\s+([A-Z][a-zA-Z\s]+?)(?:\s+agent|\s+who|\s+that|\s+with|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up name
                name = re.sub(r"\s+", " ", name)
                return name
        
        # Default: use first few words
        words = description.split()[:3]
        return " ".join(words).title()
    
    def _extract_goal(self, description: str, name: str) -> str:
        """Extract or infer agent goal from description."""
        # Look for explicit goal statements
        goal_patterns = [
            r"who\s+(.+?)(?:\s+and\s+uses|\s+with|\s+using|$)",
            r"that\s+(.+?)(?:\s+and\s+uses|\s+with|\s+using|$)",
            r"to\s+(.+?)(?:\s+and\s+uses|\s+with|\s+using|$)",
        ]
        
        for pattern in goal_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                goal = match.group(1).strip()
                # Remove tool references
                goal = re.sub(r"\s+(uses?|with|using).*$", "", goal, flags=re.IGNORECASE)
                if goal:
                    return goal.capitalize()
        
        # Infer from agent type and name
        if "analyst" in name.lower():
            return "Analyze data and provide insights"
        elif "researcher" in name.lower() or "research" in name.lower():
            return "Research topics and gather information"
        elif "assistant" in name.lower():
            return "Assist users with tasks and questions"
        else:
            return f"Execute tasks related to {name.lower()}"
    
    def _infer_agent_type(self, description_lower: str) -> str:
        """Infer agent type from description."""
        for agent_type, keywords in self.AGENT_TYPE_PATTERNS.items():
            if agent_type == "base":
                continue
            for keyword in keywords:
                if keyword in description_lower:
                    return agent_type
        return "base"
    
    def _extract_tools(self, description_lower: str) -> List[str]:
        """Extract requested tools from description."""
        tools = []
        
        for tool_name, keywords in self.TOOL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    tools.append(tool_name)
                    break
        
        # Also check for explicit tool mentions
        tool_mentions = re.findall(r"(web\s+search|file\s+(read|write)|data\s+analysis|code|database)", description_lower)
        for mention in tool_mentions:
            mention_text = mention[0] if isinstance(mention, tuple) else mention
            for tool_name, keywords in self.TOOL_KEYWORDS.items():
                if any(kw in mention_text for kw in keywords):
                    if tool_name not in tools:
                        tools.append(tool_name)
        
        return tools
    
    def _infer_autonomy_level(self, description_lower: str) -> str:
        """Infer autonomy level from description."""
        for level, keywords in self.AUTONOMY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return level
        return "moderate"
    
    def _infer_competence(self, description_lower: str, agent_type: str) -> float:
        """Infer initial competence score."""
        base_competence = {
            "rag": 0.6,
            "research": 0.65,
            "automation": 0.7,
            "base": 0.5,
        }.get(agent_type, 0.5)
        
        # Adjust based on keywords
        if any(kw in description_lower for kw in ["expert", "advanced", "senior", "professional"]):
            base_competence += 0.15
        elif any(kw in description_lower for kw in ["junior", "basic", "simple", "beginner"]):
            base_competence -= 0.15
        
        return max(0.3, min(0.9, base_competence))
    
    def _validate_config(self, config: AgentConfig) -> AgentConfig:
        """Validate and adjust agent configuration."""
        # Validate tools exist in registry
        valid_tools = []
        for tool_name in config.tools:
            if self.tool_registry.get(tool_name):
                valid_tools.append(tool_name)
            else:
                print(f"Warning: Tool '{tool_name}' not found in registry, skipping")
        
        config.tools = valid_tools
        
        # Validate autonomy level
        if config.autonomy_level not in ["conservative", "moderate", "aggressive"]:
            config.autonomy_level = "moderate"
        
        # Ensure memory store is available
        if self.memory_store is None:
            from ..memory.memory_store import InMemoryMemoryStore
            self.memory_store = InMemoryMemoryStore()
        
        return config
    
    def _create_agent_instance(
        self,
        config: AgentConfig,
        memory_store: MemoryStore,
        autonomy_policy: AutonomyPolicy,
        llm_router: Optional[Any],
    ) -> BaseAgent:
        """Create agent instance from config."""
        # Select agent class based on type
        agent_classes = {
            "rag": RAGAgent,
            "research": ResearchAgent,
            "automation": AutomationAgent,
            "base": BaseAgent,
        }
        
        agent_class = agent_classes.get(config.agent_type, BaseAgent)
        
        # Create agent with appropriate parameters
        if config.agent_type == "rag":
            agent = agent_class(
                agent_id=config.agent_id,
                name=config.name,
                memory=memory_store,
                autonomy_policy=autonomy_policy,
                tool_registry=self.tool_registry,
                initial_competence=config.initial_competence,
            )
        elif config.agent_type == "research":
            agent = agent_class(
                agent_id=config.agent_id,
                name=config.name,
                memory=memory_store,
                autonomy_policy=autonomy_policy,
                tool_registry=self.tool_registry,
                initial_competence=config.initial_competence,
            )
        elif config.agent_type == "automation":
            agent = agent_class(
                agent_id=config.agent_id,
                name=config.name,
                memory=memory_store,
                autonomy_policy=autonomy_policy,
                tool_registry=self.tool_registry,
                initial_competence=config.initial_competence,
            )
        else:
            agent = agent_class(
                agent_id=config.agent_id,
                name=config.name,
                goal=config.goal,
                memory=memory_store,
                autonomy_policy=autonomy_policy,
                tool_registry=self.tool_registry,
                initial_competence=config.initial_competence,
            )
        
        # Register tools
        if config.tools:
            agent.register_tools(config.tools)
        
        # Set LLM router if provided
        if llm_router:
            agent.llm_router = llm_router
        
        return agent
    
    def get_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get configuration for a created agent."""
        return self._created_agents.get(agent_id)
    
    def list_created_agents(self) -> List[str]:
        """List IDs of all created agents."""
        return list(self._created_agents.keys())
    
    def export_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Export agent configuration as JSON-serializable dict."""
        config = self._created_agents.get(agent_id)
        if config:
            return config.to_dict()
        return None

