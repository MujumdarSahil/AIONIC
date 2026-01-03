"""
Memory Store - Persistent memory system for agents.

Provides structured storage and retrieval of agent memories,
contexts, and knowledge.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(Enum):
    """Types of memories."""
    EPISODIC = "episodic"      # Specific events/experiences
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    WORKING = "working"        # Temporary active memory


@dataclass
class Memory:
    """
    Single memory entry.
    
    Attributes:
        memory_id: Unique memory identifier
        agent_id: Agent that created this memory
        memory_type: Type of memory
        content: Memory content
        metadata: Additional metadata
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        access_count: Number of times accessed
        importance_score: Importance score (0.0-1.0)
    """
    
    memory_id: str
    agent_id: str
    memory_type: MemoryType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    importance_score: float = 0.5
    
    def access(self) -> None:
        """Mark memory as accessed."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class MemoryStore(ABC):
    """
    Abstract base class for memory storage.
    
    Provides interface for storing and retrieving agent memories
    with support for different memory types and search capabilities.
    """
    
    @abstractmethod
    def store(
        self,
        agent_id: str,
        content: Any,
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: float = 0.5,
    ) -> str:
        """
        Store a memory.
        
        Args:
            agent_id: Agent identifier
            content: Memory content
            memory_type: Type of memory
            metadata: Additional metadata
            importance_score: Importance score (0.0-1.0)
            
        Returns:
            Memory ID of stored memory
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        agent_id: str,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Retrieve memories for an agent.
        
        Args:
            agent_id: Agent identifier
            query: Optional search query
            memory_type: Optional type filter
            limit: Maximum number of memories to return
            
        Returns:
            List of memories
        """
        pass
    
    @abstractmethod
    def retrieve_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID."""
        pass
    
    @abstractmethod
    def update_importance(self, memory_id: str, score: float) -> None:
        """Update importance score of a memory."""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        pass


class InMemoryMemoryStore(MemoryStore):
    """
    In-memory implementation of MemoryStore.
    
    Suitable for testing and small-scale deployments.
    For production, use a persistent implementation (e.g., database-backed).
    """
    
    def __init__(self):
        """Initialize empty in-memory store."""
        self._memories: Dict[str, Memory] = {}
        self._agent_memories: Dict[str, List[str]] = {}  # agent_id -> [memory_ids]
        self._counter = 0
    
    def store(
        self,
        agent_id: str,
        content: Any,
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: float = 0.5,
    ) -> str:
        """Store a memory."""
        self._counter += 1
        memory_id = f"mem_{self._counter}_{datetime.utcnow().timestamp()}"
        
        memory = Memory(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            importance_score=importance_score,
        )
        
        self._memories[memory_id] = memory
        
        if agent_id not in self._agent_memories:
            self._agent_memories[agent_id] = []
        self._agent_memories[agent_id].append(memory_id)
        
        return memory_id
    
    def retrieve(
        self,
        agent_id: str,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Retrieve memories for an agent."""
        if agent_id not in self._agent_memories:
            return []
        
        memory_ids = self._agent_memories[agent_id]
        memories = [self._memories[mid] for mid in memory_ids if mid in self._memories]
        
        # Filter by type if specified
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        # Simple query matching (in production, use proper search/indexing)
        if query:
            query_lower = query.lower()
            memories = [
                m for m in memories
                if query_lower in str(m.content).lower()
                or query_lower in str(m.metadata).lower()
            ]
        
        # Sort by importance and recency
        memories.sort(
            key=lambda m: (m.importance_score, m.created_at.timestamp()),
            reverse=True,
        )
        
        # Mark as accessed
        for memory in memories[:limit]:
            memory.access()
        
        return memories[:limit]
    
    def retrieve_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID."""
        memory = self._memories.get(memory_id)
        if memory:
            memory.access()
        return memory
    
    def update_importance(self, memory_id: str, score: float) -> None:
        """Update importance score."""
        if memory_id in self._memories:
            self._memories[memory_id].importance_score = max(0.0, min(1.0, score))
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id not in self._memories:
            return False
        
        memory = self._memories[memory_id]
        
        # Remove from agent's memory list
        if memory.agent_id in self._agent_memories:
            self._agent_memories[memory.agent_id] = [
                mid for mid in self._agent_memories[memory.agent_id]
                if mid != memory_id
            ]
        
        del self._memories[memory_id]
        return True
    
    def get_agent_memory_count(self, agent_id: str) -> int:
        """Get total number of memories for an agent."""
        return len(self._agent_memories.get(agent_id, []))

