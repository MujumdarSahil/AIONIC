"""
Orchestrator - Central coordination system for multi-agent task execution.

Manages task planning, agent assignment, collaboration, and arbitration.
Implements intelligent task decomposition and agent orchestration.
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass

from .agent import AgentBase, AgentRole
from .task import Task, TaskStatus, TaskPriority
from .context import Context, ContextType
from .tool import ToolRegistry
from ..memory.memory_store import MemoryStore
from ..security.autonomy_policy import AutonomyPolicy
from ..memory.audit_logger import AuditLogger


@dataclass
class Assignment:
    """
    Agent-task assignment record.
    
    Attributes:
        task_id: Task identifier
        agent_id: Assigned agent identifier
        assigned_at: Assignment timestamp
        reason: Reasoning for assignment
        priority: Assignment priority
    """
    
    task_id: str
    agent_id: str
    assigned_at: datetime
    reason: str
    priority: int


class Orchestrator:
    """
    Central orchestrator for multi-agent task execution.
    
    Responsibilities:
    - Task decomposition and planning
    - Agent selection and assignment
    - Task scheduling and dependency resolution
    - Agent collaboration coordination
    - Conflict arbitration
    - Execution monitoring
    """
    
    def __init__(
        self,
        agents: List[AgentBase],
        tool_registry: ToolRegistry,
        memory_store: MemoryStore,
        autonomy_policy: AutonomyPolicy,
        audit_logger: AuditLogger,
    ):
        """
        Initialize orchestrator.
        
        Args:
            agents: List of available agents
            tool_registry: Registry of available tools
            memory_store: Shared memory store
            autonomy_policy: Autonomy and permission policy
            audit_logger: Audit logging system
        """
        self.agents = {agent.agent_id: agent for agent in agents}
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.autonomy_policy = autonomy_policy
        self.audit_logger = audit_logger
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.assignments: Dict[str, Assignment] = {}
        
        # Execution state
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        
        # Collaboration tracking
        self.collaboration_graph: Dict[str, Set[str]] = {}  # task_id -> {agent_ids}
    
    def submit_task(
        self,
        description: str,
        objective: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict] = None,
        dependencies: Optional[List[str]] = None,
    ) -> Task:
        """
        Submit a new task for execution.
        
        Args:
            description: Task description
            objective: Clear objective statement
            priority: Task priority level
            metadata: Additional task metadata
            dependencies: List of task IDs this depends on
            
        Returns:
            Created Task instance
        """
        task_id = f"task_{datetime.utcnow().timestamp()}_{len(self.tasks)}"
        
        task = Task(
            task_id=task_id,
            description=description,
            objective=objective,
            priority=priority,
            metadata=metadata or {},
            dependencies=dependencies or [],
            context=Context(
                context_id=f"ctx_{task_id}",
                context_type=ContextType.TASK,
            ),
        )
        
        self.tasks[task_id] = task
        self.audit_logger.log_task_created(task_id, description, priority.value)
        
        return task
    
    def plan_task(self, task: Task) -> List[Task]:
        """
        Decompose task into subtasks.
        
        This implements task decomposition logic - breaking down
        complex tasks into manageable subtasks.
        
        Args:
            task: Task to decompose
            
        Returns:
            List of subtasks (may be empty if task cannot be decomposed)
        """
        # Base implementation: check if task should be decomposed
        # Subclasses or plugins can override for specific decomposition logic
        
        # Simple heuristic: if task has dependencies, create subtasks for dependencies
        subtasks = []
        if task.dependencies:
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    subtasks.append(self.tasks[dep_id])
        
        return subtasks
    
    def select_agent(self, task: Task) -> Optional[str]:
        """
        Select best agent for task execution.
        
        Implements intelligent agent selection based on:
        - Agent competence and role
        - Task requirements and complexity
        - Agent availability
        - Specialization match
        
        Args:
            task: Task to assign
            
        Returns:
            Agent ID of selected agent, or None if no suitable agent
        """
        # Filter available agents (not currently executing tasks)
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state.current_task_id is None
        ]
        
        if not available_agents:
            return None
        
        # Score agents for this task
        agent_scores = {}
        for agent in available_agents:
            score = self._score_agent_for_task(agent, task)
            agent_scores[agent.agent_id] = score
        
        # Select highest scoring agent
        if agent_scores:
            best_agent_id = max(agent_scores.items(), key=lambda x: x[1])[0]
            return best_agent_id
        
        return None
    
    def _score_agent_for_task(self, agent: AgentBase, task: Task) -> float:
        """
        Score agent suitability for task.
        
        Args:
            agent: Agent to score
            task: Task to score against
            
        Returns:
            Suitability score (0.0 - 1.0)
        """
        score = 0.0
        
        # Base score from competence
        score += agent.competence_score * 0.4
        
        # Role-based weighting
        role_weights = {
            AgentRole.JUNIOR: 0.2,
            AgentRole.ASSOCIATE: 0.4,
            AgentRole.SENIOR: 0.6,
            AgentRole.EXPERT: 0.8,
            AgentRole.ARCHITECT: 1.0,
        }
        score += role_weights.get(agent.role, 0.4) * 0.3
        
        # Success rate
        score += agent.state.success_rate * 0.2
        
        # Availability (already filtered, but can add recency weight)
        time_since_activity = (datetime.utcnow() - agent.state.last_activity).total_seconds()
        availability_weight = min(1.0, 1.0 / (1.0 + time_since_activity / 3600))  # Decay over hours
        score += availability_weight * 0.1
        
        return min(1.0, score)
    
    def assign_task(self, task: Task, agent_id: Optional[str] = None) -> bool:
        """
        Assign task to agent.
        
        Args:
            task: Task to assign
            agent_id: Optional specific agent ID, otherwise auto-select
            
        Returns:
            True if assignment successful, False otherwise
        """
        if task.task_id in self.assignments:
            return False  # Already assigned
        
        if agent_id is None:
            agent_id = self.select_agent(task)
        
        if agent_id is None or agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        assignment = Assignment(
            task_id=task.task_id,
            agent_id=agent_id,
            assigned_at=datetime.utcnow(),
            reason=f"Selected based on competence score: {agent.competence_score:.2f}",
            priority=task.priority.value,
        )
        
        self.assignments[task.task_id] = assignment
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = agent_id
        agent.state.current_task_id = task.task_id
        
        self.audit_logger.log_task_assigned(
            task.task_id,
            agent_id,
            agent.name,
            assignment.reason,
        )
        
        return True
    
    def execute_task(self, task_id: str) -> Any:
        """
        Execute a task with its assigned agent.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task execution result
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        task = self.tasks[task_id]
        
        if task.task_id not in self.assignments:
            if not self.assign_task(task):
                raise ValueError(f"Could not assign task '{task_id}' to any agent")
        
        assignment = self.assignments[task.task_id]
        agent = self.agents[assignment.agent_id]
        
        # Start task execution
        task.start(assignment.agent_id)
        self.active_tasks.add(task_id)
        
        self.audit_logger.log_task_started(task_id, assignment.agent_id)
        
        try:
            # Execute task
            context = task.context or Context(
                context_id=f"ctx_{task_id}",
                context_type=ContextType.TASK,
            )
            
            result = agent.execute_task(task, context)
            
            # Mark task as completed
            task.complete(result)
            agent.update_competence(True)
            agent.state.current_task_id = None
            
            self.active_tasks.remove(task_id)
            self.completed_tasks.add(task_id)
            
            self.audit_logger.log_task_completed(task_id, assignment.agent_id, result)
            
            return result
            
        except Exception as e:
            # Handle task failure
            error_msg = str(e)
            task.fail(error_msg)
            agent.update_competence(False)
            agent.state.current_task_id = None
            
            self.active_tasks.remove(task_id)
            
            self.audit_logger.log_task_failed(task_id, assignment.agent_id, error_msg)
            
            raise
    
    def coordinate_collaboration(
        self,
        task_id: str,
        required_agents: List[str],
    ) -> bool:
        """
        Coordinate multiple agents working on the same task.
        
        Args:
            task_id: Task requiring collaboration
            required_agents: List of agent IDs needed
            
        Returns:
            True if collaboration setup successful
        """
        if task_id not in self.tasks:
            return False
        
        # Verify all agents exist and are available
        available_agents = set()
        for agent_id in required_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if agent.state.current_task_id is None:
                    available_agents.add(agent_id)
        
        if len(available_agents) < len(required_agents):
            return False
        
        # Register collaboration
        self.collaboration_graph[task_id] = available_agents
        
        self.audit_logger.log_collaboration_started(
            task_id,
            list(available_agents),
        )
        
        return True
    
    def arbitrate_conflict(
        self,
        task_id: str,
        conflicting_agents: List[str],
        conflict_type: str,
    ) -> Optional[str]:
        """
        Arbitrate conflict between agents.
        
        Args:
            task_id: Task with conflict
            conflicting_agents: Agent IDs in conflict
            conflict_type: Type of conflict
            
        Returns:
            Selected agent ID, or None if arbitration failed
        """
        # Simple arbitration: select agent with highest competence
        best_agent = None
        best_score = -1.0
        
        for agent_id in conflicting_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                score = agent.competence_score
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
        
        if best_agent:
            self.audit_logger.log_conflict_arbitrated(
                task_id,
                conflicting_agents,
                best_agent,
                conflict_type,
            )
        
        return best_agent
    
    def get_system_state(self) -> Dict:
        """Get current orchestrator system state."""
        return {
            "total_tasks": len(self.tasks),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "registered_agents": len(self.agents),
            "available_agents": len([
                a for a in self.agents.values()
                if a.state.current_task_id is None
            ]),
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            },
            "agents": {
                agent_id: agent.to_dict()
                for agent_id, agent in self.agents.items()
            },
        }

