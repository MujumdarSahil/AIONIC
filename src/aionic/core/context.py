"""
Context - Shared execution context for agents and tasks.

Maintains state, configuration, and environment information
across agent interactions within a task execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum


class ContextType(Enum):
    """Types of execution contexts."""
    TASK = "task"
    SESSION = "session"
    WORKFLOW = "workflow"
    RESEARCH = "research"


@dataclass
class Context:
    """
    Execution context shared across agents and tasks.
    
    Attributes:
        context_id: Unique identifier for this context
        context_type: Type of context (task, session, workflow, etc.)
        metadata: Arbitrary key-value metadata
        created_at: Timestamp of context creation
        updated_at: Timestamp of last update
        state: Mutable state dictionary for runtime data
        parent_context: Optional parent context for nested execution
    """
    
    context_id: str
    context_type: ContextType = ContextType.TASK
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    state: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional['Context'] = None
    
    def update(self, key: str, value: Any) -> None:
        """Update context state and timestamp."""
        self.state[key] = value
        self.updated_at = datetime.utcnow()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context state."""
        return self.state.get(key, default)
    
    def merge(self, other: Dict[str, Any]) -> None:
        """Merge dictionary into context state."""
        self.state.update(other)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "context_id": self.context_id,
            "context_type": self.context_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "state": self.state,
        }

