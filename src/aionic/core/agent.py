"""
Agent Base - Core agent abstraction with role-switching and competence scoring.

Defines the base class for all agents, including role management,
competence tracking, memory integration, and tool usage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum

from ..memory.memory_store import MemoryStore
from ..security.autonomy_policy import AutonomyPolicy
from .tool import ToolInterface, ToolRegistry
from .task import Task, TaskStatus
from .context import Context
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.router import LLMRouter
    from ..llm.models import ModelConfig, ChatMessage, RoutingStrategy


class AgentRole(Enum):
    """Agent role levels based on competence and responsibility."""
    JUNIOR = "junior"          # 0.0 - 0.4 competence
    ASSOCIATE = "associate"    # 0.4 - 0.6 competence
    SENIOR = "senior"          # 0.6 - 0.8 competence
    EXPERT = "expert"          # 0.8 - 0.95 competence
    ARCHITECT = "architect"    # 0.95+ competence


@dataclass
class AgentState:
    """
    Agent runtime state and metrics.
    
    Attributes:
        competence_score: Current competence score (0.0 - 1.0)
        role: Current agent role
        tasks_completed: Total tasks completed
        tasks_failed: Total tasks failed
        success_rate: Historical success rate
        last_activity: Timestamp of last activity
        current_task_id: ID of currently executing task
    """
    
    competence_score: float = 0.5
    role: AgentRole = AgentRole.ASSOCIATE
    tasks_completed: int = 0
    tasks_failed: int = 0
    success_rate: float = 0.0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    current_task_id: Optional[str] = None
    
    def update_competence(self, task_success: bool, weight: float = 0.1) -> None:
        """
        Update competence score based on task outcome.
        
        Args:
            task_success: Whether task was successful
            weight: Learning rate for competence adjustment
        """
        if task_success:
            self.tasks_completed += 1
            # Positive reinforcement
            self.competence_score = min(1.0, self.competence_score + weight * (1.0 - self.competence_score))
        else:
            self.tasks_failed += 1
            # Negative feedback (less aggressive)
            self.competence_score = max(0.0, self.competence_score - weight * 0.5 * self.competence_score)
        
        total = self.tasks_completed + self.tasks_failed
        if total > 0:
            self.success_rate = self.tasks_completed / total
        
        self._update_role()
        self.last_activity = datetime.utcnow()
    
    def _update_role(self) -> None:
        """Update role based on current competence score."""
        score = self.competence_score
        if score >= 0.95:
            self.role = AgentRole.ARCHITECT
        elif score >= 0.8:
            self.role = AgentRole.EXPERT
        elif score >= 0.6:
            self.role = AgentRole.SENIOR
        elif score >= 0.4:
            self.role = AgentRole.ASSOCIATE
        else:
            self.role = AgentRole.JUNIOR


class AgentBase(ABC):
    """
    Abstract base class for all agents in AIONIC.
    
    Agents are autonomous entities that execute tasks using tools,
    maintain memory, and adapt their roles based on performance.
    
    Key Features:
    - Role-based execution with dynamic promotion/demotion
    - Competence scoring and adaptive learning
    - Memory integration for context retention
    - Tool usage with permission checking
    - Explainable reasoning and decision-making
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        goal: str,
        memory: MemoryStore,
        autonomy_policy: AutonomyPolicy,
        tool_registry: ToolRegistry,
        initial_competence: float = 0.5,
        llm_router: Optional["LLMRouter"] = None,
    ):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            goal: Agent's primary objective/goal
            memory: Memory store for agent
            autonomy_policy: Policy for execution permissions
            tool_registry: Registry of available tools
            initial_competence: Initial competence score (0.0-1.0)
            llm_router: Optional LLM router for text generation
        """
        self.agent_id = agent_id
        self.name = name
        self.goal = goal
        self.memory = memory
        self.autonomy_policy = autonomy_policy
        self.tool_registry = tool_registry
        self.llm_router = llm_router
        self.state = AgentState(competence_score=initial_competence)
        self.state._update_role()
        
        # Agent-specific tools (subset of registry)
        self._available_tools: Set[str] = set()
        
        # Reasoning and decision log
        self.reasoning_log: List[Dict[str, Any]] = []
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Type/category of agent (e.g., 'rag', 'automation', 'research')."""
        pass
    
    @property
    def role(self) -> AgentRole:
        """Current agent role."""
        return self.state.role
    
    @property
    def competence_score(self) -> float:
        """Current competence score."""
        return self.state.competence_score
    
    def register_tool(self, tool_name: str) -> None:
        """
        Register a tool for use by this agent.
        
        Args:
            tool_name: Name of tool in registry
        """
        if self.tool_registry.get(tool_name):
            self._available_tools.add(tool_name)
        else:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
    
    def register_tools(self, tool_names: List[str]) -> None:
        """Register multiple tools at once."""
        for tool_name in tool_names:
            self.register_tool(tool_name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of tool names available to this agent."""
        return list(self._available_tools)
    
    @abstractmethod
    def reason(self, task: Task, context: Context) -> Dict[str, Any]:
        """
        Generate reasoning and plan for task execution.
        
        This is the core decision-making method that each agent
        must implement based on its specialized logic.
        
        Args:
            task: Task to reason about
            context: Execution context
            
        Returns:
            Dictionary containing reasoning, plan, and decisions
        """
        pass
    
    @abstractmethod
    def execute_task(self, task: Task, context: Context) -> Any:
        """
        Execute a task and return result.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Task execution result
        """
        pass
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool with permission checking.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            PermissionError: If tool execution not permitted
            ValueError: If tool not available or invalid parameters
        """
        # Check if tool is available to agent
        if tool_name not in self._available_tools:
            raise ValueError(f"Tool '{tool_name}' not available to agent '{self.name}'")
        
        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        # Check permissions
        if not self.autonomy_policy.can_execute_tool(
            agent_id=self.agent_id,
            tool_name=tool_name,
            tool_risk_tier=tool.risk_tier,
            agent_role=self.role.value,
            agent_competence=self.competence_score,
        ):
            raise PermissionError(
                f"Agent '{self.name}' not permitted to execute tool '{tool_name}' "
                f"(risk tier: {tool.risk_tier})"
            )
        
        # Validate parameters
        if not tool.validate_parameters(**kwargs):
            raise ValueError(f"Invalid parameters for tool '{tool_name}'")
        
        # Execute tool
        result = tool.execute(**kwargs)
        
        # Log tool usage
        self._log_reasoning(
            action="tool_execution",
            tool_name=tool_name,
            parameters=kwargs,
            result=result.success,
            error=result.error,
        )
        
        return result
    
    def _log_reasoning(self, **kwargs) -> None:
        """Log reasoning step or decision."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "role": self.role.value,
            "competence": self.competence_score,
            **kwargs,
        }
        self.reasoning_log.append(log_entry)
    
    def update_competence(self, task_success: bool) -> None:
        """
        Update agent competence based on task outcome.
        
        Args:
            task_success: Whether task was successfully completed
        """
        self.state.update_competence(task_success)
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of agent's reasoning log."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "current_role": self.role.value,
            "competence_score": self.competence_score,
            "success_rate": self.state.success_rate,
            "tasks_completed": self.state.tasks_completed,
            "tasks_failed": self.state.tasks_failed,
            "reasoning_entries": len(self.reasoning_log),
            "available_tools": list(self._available_tools),
        }
    
    def generate_text(
        self,
        prompt: str,
        config: Optional["ModelConfig"] = None,
        strategy: Optional["RoutingStrategy"] = None,
        require_permission: bool = True,
    ) -> str:
        """
        Generate text using LLM router.
        
        This is a convenience method that agents can use to generate text
        without directly depending on LLM providers. The router handles
        provider selection, failover, and logging.
        
        Args:
            prompt: Input prompt text
            config: Optional model configuration
            strategy: Optional routing strategy (uses router default if None)
            require_permission: Whether to check autonomy policy before generation
            
        Returns:
            Generated text content
            
        Raises:
            RuntimeError: If LLM router not configured or all providers fail
            PermissionError: If permission check fails and require_permission is True
        """
        if not self.llm_router:
            raise RuntimeError(
                "LLM router not configured. Pass llm_router parameter to agent constructor."
            )
        
        # Check autonomy policy if required
        # LLM generation is considered a LOW risk action by default
        if require_permission:
            if not self.autonomy_policy.can_execute_tool(
                agent_id=self.agent_id,
                tool_name="llm_generate",
                tool_risk_tier="low",
                agent_role=self.role.value,
                agent_competence=self.competence_score,
            ):
                raise PermissionError(
                    f"Agent '{self.name}' not permitted to use LLM generation"
                )
        
        # Get task metadata from current context if available
        task_metadata = None
        if hasattr(self, 'state') and self.state.current_task_id:
            task_metadata = {"task_id": self.state.current_task_id}
        
        # Generate text
        response = self.llm_router.generate(
            prompt=prompt,
            config=config,
            strategy=strategy,
            task_metadata=task_metadata,
        )
        
        # Log LLM usage
        self._log_reasoning(
            action="llm_generate",
            prompt=prompt[:200],  # Truncate for logging
            response_length=len(response.content),
            provider=response.provider,
            model=response.model,
            usage=response.usage,
        )
        
        return response.content
    
    def chat(
        self,
        messages: List["ChatMessage"],
        config: Optional["ModelConfig"] = None,
        strategy: Optional["RoutingStrategy"] = None,
        require_permission: bool = True,
    ) -> str:
        """
        Chat with LLM using message history.
        
        Args:
            messages: List of chat messages (conversation history)
            config: Optional model configuration
            strategy: Optional routing strategy
            require_permission: Whether to check autonomy policy
            
        Returns:
            Generated response content
            
        Raises:
            RuntimeError: If LLM router not configured or all providers fail
            PermissionError: If permission check fails
        """
        if not self.llm_router:
            raise RuntimeError(
                "LLM router not configured. Pass llm_router parameter to agent constructor."
            )
        
        # Check autonomy policy if required
        if require_permission:
            if not self.autonomy_policy.can_execute_tool(
                agent_id=self.agent_id,
                tool_name="llm_chat",
                tool_risk_tier="low",
                agent_role=self.role.value,
                agent_competence=self.competence_score,
            ):
                raise PermissionError(
                    f"Agent '{self.name}' not permitted to use LLM chat"
                )
        
        # Get task metadata
        task_metadata = None
        if hasattr(self, 'state') and self.state.current_task_id:
            task_metadata = {"task_id": self.state.current_task_id}
        
        # Generate response
        response = self.llm_router.chat(
            messages=messages,
            config=config,
            strategy=strategy,
            task_metadata=task_metadata,
        )
        
        # Log LLM usage
        self._log_reasoning(
            action="llm_chat",
            message_count=len(messages),
            response_length=len(response.content),
            provider=response.provider,
            model=response.model,
            usage=response.usage,
        )
        
        return response.content
    
    def embed(
        self,
        text: str,
        config: Optional["ModelConfig"] = None,
        strategy: Optional["RoutingStrategy"] = None,
        require_permission: bool = True,
    ) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Input text to embed
            config: Optional model configuration
            strategy: Optional routing strategy
            require_permission: Whether to check autonomy policy
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            RuntimeError: If LLM router not configured or embeddings not supported
            PermissionError: If permission check fails
        """
        if not self.llm_router:
            raise RuntimeError(
                "LLM router not configured. Pass llm_router parameter to agent constructor."
            )
        
        # Check autonomy policy if required
        if require_permission:
            if not self.autonomy_policy.can_execute_tool(
                agent_id=self.agent_id,
                tool_name="llm_embed",
                tool_risk_tier="low",
                agent_role=self.role.value,
                agent_competence=self.competence_score,
            ):
                raise PermissionError(
                    f"Agent '{self.name}' not permitted to use LLM embeddings"
                )
        
        # Get task metadata
        task_metadata = None
        if hasattr(self, 'state') and self.state.current_task_id:
            task_metadata = {"task_id": self.state.current_task_id}
        
        # Generate embeddings
        response = self.llm_router.embed(
            text=text,
            config=config,
            strategy=strategy,
            task_metadata=task_metadata,
        )
        
        # Log embedding usage
        self._log_reasoning(
            action="llm_embed",
            text_length=len(text),
            embedding_dim=len(response.embeddings[0]) if response.embeddings else 0,
            provider=response.provider,
            model=response.model,
            usage=response.usage,
        )
        
        # Return first embedding vector (most common case)
        if response.embeddings:
            return response.embeddings[0]
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "goal": self.goal,
            "agent_type": self.agent_type,
            "role": self.role.value,
            "competence_score": self.competence_score,
            "success_rate": self.state.success_rate,
            "tasks_completed": self.state.tasks_completed,
            "tasks_failed": self.state.tasks_failed,
            "available_tools": list(self._available_tools),
            "has_llm_router": self.llm_router is not None,
        }

