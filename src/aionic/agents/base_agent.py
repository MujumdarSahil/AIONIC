"""
Base Agent Implementation - Concrete agent base class.

Provides default implementations for common agent operations
while allowing specialization through inheritance.
"""

from typing import Any, Dict, List
import uuid

from ..core.agent import AgentBase
from ..core.task import Task, TaskStatus
from ..core.context import Context
from ..memory.memory_store import MemoryStore, MemoryType
from ..security.autonomy_policy import AutonomyPolicy
from ..core.tool import ToolRegistry


class BaseAgent(AgentBase):
    """
    Concrete base agent with default reasoning and execution logic.
    
    This class provides a working implementation that can be
    extended for specialized agents (RAG, Automation, etc.).
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
        agent_type: str = "base",
    ):
        """Initialize base agent."""
        super().__init__(
            agent_id=agent_id,
            name=name,
            goal=goal,
            memory=memory,
            autonomy_policy=autonomy_policy,
            tool_registry=tool_registry,
            initial_competence=initial_competence,
        )
        self._agent_type = agent_type
    
    @property
    def agent_type(self) -> str:
        """Agent type identifier."""
        return self._agent_type
    
    def reason(self, task: Task, context: Context) -> Dict[str, Any]:
        """
        Generate reasoning and plan for task execution.
        
        Default implementation:
        1. Analyze task objective
        2. Retrieve relevant memories
        3. Select appropriate tools
        4. Create execution plan
        """
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(
            agent_id=self.agent_id,
            query=task.objective,
            limit=5,
        )
        
        # Analyze available tools
        available_tools = self.get_available_tools()
        
        # Determine execution strategy
        strategy = self._determine_strategy(task, available_tools, relevant_memories)
        
        reasoning = {
            "task_id": task.task_id,
            "objective": task.objective,
            "strategy": strategy,
            "tools_selected": strategy.get("tools", []),
            "steps": strategy.get("steps", []),
            "relevant_memories_count": len(relevant_memories),
            "confidence": self._calculate_confidence(task, available_tools),
        }
        
        # Log reasoning
        self._log_reasoning(
            action="reasoning",
            task_id=task.task_id,
            reasoning=reasoning,
        )
        
        return reasoning
    
    def _determine_strategy(
        self,
        task: Task,
        available_tools: List[str],
        memories: List,
    ) -> Dict[str, Any]:
        """Determine execution strategy for task."""
        # Simple heuristic-based strategy
        objective_lower = task.objective.lower()
        
        tools_needed = []
        steps = []
        
        # Analyze objective to determine needed tools
        if any(keyword in objective_lower for keyword in ["search", "find", "research"]):
            if "web_search" in available_tools:
                tools_needed.append("web_search")
                steps.append("Search for information")
        
        if any(keyword in objective_lower for keyword in ["analyze", "calculate", "compute"]):
            if "data_analysis" in available_tools:
                tools_needed.append("data_analysis")
                steps.append("Perform data analysis")
        
        if any(keyword in objective_lower for keyword in ["read", "file", "document"]):
            if "file_read" in available_tools:
                tools_needed.append("file_read")
                steps.append("Read relevant files")
        
        # Default step if no specific tools identified
        if not steps:
            steps.append("Execute task objective")
        
        return {
            "tools": tools_needed,
            "steps": steps,
            "approach": "sequential",
        }
    
    def _calculate_confidence(self, task: Task, available_tools: List[str]) -> float:
        """Calculate confidence in ability to complete task."""
        # Base confidence from competence
        confidence = self.competence_score * 0.6
        
        # Boost if we have relevant tools
        objective_lower = task.objective.lower()
        if any(keyword in objective_lower for keyword in ["search", "find"]):
            if "web_search" in available_tools:
                confidence += 0.2
        if any(keyword in objective_lower for keyword in ["analyze"]):
            if "data_analysis" in available_tools:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def execute_task(self, task: Task, context: Context) -> Any:
        """
        Execute task with reasoning and tool usage.
        
        Default implementation:
        1. Generate reasoning/plan
        2. Execute plan steps
        3. Store results in memory
        4. Return result
        """
        # Generate reasoning
        reasoning = self.reason(task, context)
        
        # Execute plan
        result_data = {
            "task_id": task.task_id,
            "objective": task.objective,
            "reasoning": reasoning,
            "steps_executed": [],
            "results": {},
        }
        
        try:
            # Execute strategy steps
            for step in reasoning.get("steps", []):
                step_result = self._execute_step(step, task, context, reasoning)
                result_data["steps_executed"].append(step)
                result_data["results"][step] = step_result
            
            # Store in memory
            self.memory.store(
                agent_id=self.agent_id,
                content={
                    "task": task.objective,
                    "result": result_data,
                    "success": True,
                },
                memory_type=MemoryType.EPISODIC,
                importance_score=0.7,
            )
            
            # Update context
            context.update("task_result", result_data)
            context.update("executed_by", self.agent_id)
            
            return result_data
            
        except Exception as e:
            # Store failure in memory
            self.memory.store(
                agent_id=self.agent_id,
                content={
                    "task": task.objective,
                    "error": str(e),
                    "success": False,
                },
                memory_type=MemoryType.EPISODIC,
                importance_score=0.5,
            )
            
            raise
    
    def _execute_step(
        self,
        step: str,
        task: Task,
        context: Context,
        reasoning: Dict[str, Any],
    ) -> Any:
        """Execute a single step from the plan."""
        step_lower = step.lower()
        
        # Map steps to tool execution
        if "search" in step_lower:
            # Extract query from task objective
            query = task.objective  # Simplified - should extract better query
            tool_result = self.use_tool("web_search", query=query, max_results=5)
            return tool_result.data if tool_result.success else None
        
        elif "analyze" in step_lower or "analysis" in step_lower:
            # Placeholder - would need actual data to analyze
            return {"status": "analysis_placeholder"}
        
        elif "read" in step_lower:
            # Would need file path from context or task metadata
            if "file_path" in task.metadata:
                tool_result = self.use_tool("file_read", file_path=task.metadata["file_path"])
                return tool_result.data if tool_result.success else None
            return {"status": "no_file_path_provided"}
        
        else:
            # Generic step execution
            return {"status": "completed", "step": step}

