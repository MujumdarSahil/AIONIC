"""
Audit Logger - Comprehensive logging and audit trail system.

Provides structured logging for all agent activities, decisions,
and system events for governance and debugging.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(Enum):
    """Categories of log entries."""
    TASK = "task"
    AGENT = "agent"
    TOOL = "tool"
    ORCHESTRATION = "orchestration"
    SECURITY = "security"
    MEMORY = "memory"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """
    Single log entry.
    
    Attributes:
        log_id: Unique log entry identifier
        timestamp: Log timestamp
        level: Log severity level
        category: Log category
        agent_id: Related agent ID (if applicable)
        task_id: Related task ID (if applicable)
        message: Log message
        metadata: Additional structured data
    """
    
    log_id: str
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize log entry to dictionary."""
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "message": self.message,
            "metadata": self.metadata,
        }


class AuditLogger(ABC):
    """
    Abstract base class for audit logging.
    
    Provides structured logging interface for all system activities.
    """
    
    @abstractmethod
    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log an entry.
        
        Args:
            level: Log severity level
            category: Log category
            message: Log message
            agent_id: Related agent ID
            task_id: Related task ID
            metadata: Additional metadata
            
        Returns:
            Log entry ID
        """
        pass
    
    # Convenience methods for common log types
    def log_task_created(self, task_id: str, description: str, priority: str) -> str:
        """Log task creation."""
        return self.log(
            LogLevel.INFO,
            LogCategory.TASK,
            f"Task created: {description}",
            task_id=task_id,
            metadata={"priority": priority, "description": description},
        )
    
    def log_task_assigned(
        self,
        task_id: str,
        agent_id: str,
        agent_name: str,
        reason: str,
    ) -> str:
        """Log task assignment."""
        return self.log(
            LogLevel.INFO,
            LogCategory.ORCHESTRATION,
            f"Task assigned to {agent_name}",
            agent_id=agent_id,
            task_id=task_id,
            metadata={"agent_name": agent_name, "reason": reason},
        )
    
    def log_task_started(self, task_id: str, agent_id: str) -> str:
        """Log task start."""
        return self.log(
            LogLevel.INFO,
            LogCategory.TASK,
            "Task execution started",
            agent_id=agent_id,
            task_id=task_id,
        )
    
    def log_task_completed(self, task_id: str, agent_id: str, result: Any) -> str:
        """Log task completion."""
        return self.log(
            LogLevel.INFO,
            LogCategory.TASK,
            "Task completed successfully",
            agent_id=agent_id,
            task_id=task_id,
            metadata={"result": str(result)[:200]},  # Truncate long results
        )
    
    def log_task_failed(self, task_id: str, agent_id: str, error: str) -> str:
        """Log task failure."""
        return self.log(
            LogLevel.ERROR,
            LogCategory.TASK,
            f"Task failed: {error}",
            agent_id=agent_id,
            task_id=task_id,
            metadata={"error": error},
        )
    
    def log_tool_execution(
        self,
        agent_id: str,
        tool_name: str,
        success: bool,
        error: Optional[str] = None,
    ) -> str:
        """Log tool execution."""
        level = LogLevel.INFO if success else LogLevel.WARNING
        message = f"Tool '{tool_name}' executed {'successfully' if success else 'with error'}"
        return self.log(
            level,
            LogCategory.TOOL,
            message,
            agent_id=agent_id,
            metadata={"tool_name": tool_name, "success": success, "error": error},
        )
    
    def log_permission_denied(
        self,
        agent_id: str,
        action: str,
        reason: str,
    ) -> str:
        """Log permission denial."""
        return self.log(
            LogLevel.WARNING,
            LogCategory.SECURITY,
            f"Permission denied: {action}",
            agent_id=agent_id,
            metadata={"action": action, "reason": reason},
        )
    
    def log_collaboration_started(
        self,
        task_id: str,
        agent_ids: List[str],
    ) -> str:
        """Log collaboration start."""
        return self.log(
            LogLevel.INFO,
            LogCategory.ORCHESTRATION,
            f"Collaboration started with {len(agent_ids)} agents",
            task_id=task_id,
            metadata={"agent_ids": agent_ids},
        )
    
    def log_conflict_arbitrated(
        self,
        task_id: str,
        conflicting_agents: List[str],
        selected_agent: str,
        conflict_type: str,
    ) -> str:
        """Log conflict arbitration."""
        return self.log(
            LogLevel.WARNING,
            LogCategory.ORCHESTRATION,
            f"Conflict arbitrated: selected {selected_agent}",
            task_id=task_id,
            metadata={
                "conflicting_agents": conflicting_agents,
                "selected_agent": selected_agent,
                "conflict_type": conflict_type,
            },
        )


class InMemoryAuditLogger(AuditLogger):
    """
    In-memory implementation of AuditLogger.
    
    Stores logs in memory. For production, use persistent storage.
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize in-memory logger.
        
        Args:
            max_entries: Maximum number of log entries to retain
        """
        self._logs: List[LogEntry] = []
        self._counter = 0
        self.max_entries = max_entries
    
    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an entry."""
        self._counter += 1
        log_id = f"log_{self._counter}_{datetime.utcnow().timestamp()}"
        
        entry = LogEntry(
            log_id=log_id,
            timestamp=datetime.utcnow(),
            level=level,
            category=category,
            agent_id=agent_id,
            task_id=task_id,
            message=message,
            metadata=metadata or {},
        )
        
        self._logs.append(entry)
        
        # Maintain size limit
        if len(self._logs) > self.max_entries:
            self._logs = self._logs[-self.max_entries:]
        
        return log_id
    
    def get_logs(
        self,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """
        Retrieve logs with filters.
        
        Args:
            level: Filter by log level
            category: Filter by category
            agent_id: Filter by agent ID
            task_id: Filter by task ID
            limit: Maximum number of entries
            
        Returns:
            List of log entries
        """
        logs = self._logs
        
        if level:
            logs = [l for l in logs if l.level == level]
        if category:
            logs = [l for l in logs if l.category == category]
        if agent_id:
            logs = [l for l in logs if l.agent_id == agent_id]
        if task_id:
            logs = [l for l in logs if l.task_id == task_id]
        
        return logs[-limit:]
    
    def get_recent_logs(self, limit: int = 100) -> List[LogEntry]:
        """Get most recent log entries."""
        return self._logs[-limit:]

