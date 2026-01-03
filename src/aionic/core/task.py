"""
Task - Core task representation and lifecycle management.

Defines tasks that agents execute, including status tracking,
priority management, and result storage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """Task execution status states."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """
    Represents a task to be executed by agents.
    
    Attributes:
        task_id: Unique task identifier
        description: Human-readable task description
        objective: Clear goal statement
        priority: Task priority level
        status: Current execution status
        assigned_agent_id: ID of agent assigned to task
        created_at: Task creation timestamp
        started_at: Task execution start timestamp
        completed_at: Task completion timestamp
        result: Task execution result (if completed)
        error: Error information (if failed)
        metadata: Additional task metadata
        dependencies: List of task IDs this task depends on
        subtasks: List of subtasks spawned from this task
        context: Execution context for this task
    """
    
    task_id: str
    description: str
    objective: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    context: Optional['Context'] = None
    
    def start(self, agent_id: str) -> None:
        """Mark task as started and assign agent."""
        self.status = TaskStatus.IN_PROGRESS
        self.assigned_agent_id = agent_id
        self.started_at = datetime.utcnow()
    
    def complete(self, result: Any = None) -> None:
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.utcnow()
    
    def fail(self, error: str) -> None:
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()
    
    def block(self, reason: str) -> None:
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED
        self.metadata["block_reason"] = reason
    
    def can_execute(self) -> bool:
        """Check if task can be executed (dependencies satisfied)."""
        # In real implementation, check dependency status
        return self.status == TaskStatus.PENDING or self.status == TaskStatus.ASSIGNED
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "objective": self.objective,
            "priority": self.priority.value,
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata,
            "dependencies": self.dependencies,
        }

