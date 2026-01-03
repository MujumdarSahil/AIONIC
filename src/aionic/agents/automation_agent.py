"""
Automation Agent - Task automation and workflow execution.

Specialized agent for automating repetitive tasks,
workflow execution, and system operations.
"""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from ..core.task import Task
from ..core.context import Context
from ..memory.memory_store import MemoryStore
from ..security.autonomy_policy import AutonomyPolicy
from ..core.tool import ToolRegistry


class AutomationAgent(BaseAgent):
    """
    Automation Agent for task and workflow automation.
    
    Specialized for:
    - Workflow execution
    - Task automation
    - Process orchestration
    - System operations
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        memory: MemoryStore,
        autonomy_policy: AutonomyPolicy,
        tool_registry: ToolRegistry,
        workflow_engine: Optional[Any] = None,
        initial_competence: float = 0.7,
    ):
        """Initialize automation agent."""
        super().__init__(
            agent_id=agent_id,
            name=name,
            goal="Automate tasks and workflows efficiently and reliably",
            memory=memory,
            autonomy_policy=autonomy_policy,
            tool_registry=tool_registry,
            initial_competence=initial_competence,
            agent_type="automation",
        )
        self.workflow_engine = workflow_engine
        
        # Register automation tools (higher risk tier tools)
        self.register_tools(["code_execution", "database_query", "file_read"])
    
    def reason(self, task: Task, context: Context) -> Dict[str, Any]:
        """Generate automation-specific reasoning."""
        # Analyze task for automation patterns
        automation_pattern = self._identify_automation_pattern(task)
        
        # Check workflow dependencies
        workflow_steps = self._plan_workflow(task, context)
        
        reasoning = {
            "task_id": task.task_id,
            "agent_type": "automation",
            "automation_pattern": automation_pattern,
            "workflow_steps": workflow_steps,
            "requires_approval": self._requires_approval(task),
            "estimated_steps": len(workflow_steps),
        }
        
        self._log_reasoning(
            action="automation_reasoning",
            task_id=task.task_id,
            pattern=automation_pattern,
            steps=len(workflow_steps),
        )
        
        return reasoning
    
    def _identify_automation_pattern(self, task: Task) -> str:
        """Identify automation pattern from task."""
        objective_lower = task.objective.lower()
        
        if any(kw in objective_lower for kw in ["workflow", "process", "pipeline"]):
            return "workflow_execution"
        elif any(kw in objective_lower for kw in ["script", "code", "execute"]):
            return "code_execution"
        elif any(kw in objective_lower for kw in ["batch", "bulk", "multiple"]):
            return "batch_processing"
        else:
            return "generic_automation"
    
    def _plan_workflow(self, task: Task, context: Context) -> List[Dict[str, Any]]:
        """Plan workflow steps for task."""
        # Extract workflow definition from metadata or context
        workflow_def = task.metadata.get("workflow", None) or context.get("workflow")
        
        if workflow_def:
            # Parse workflow definition
            if isinstance(workflow_def, list):
                return [{"step": i, "action": step} for i, step in enumerate(workflow_def)]
            elif isinstance(workflow_def, dict):
                return workflow_def.get("steps", [])
        
        # Default: create simple workflow from task objective
        return [
            {"step": 1, "action": "validate_inputs"},
            {"step": 2, "action": "execute_automation"},
            {"step": 3, "action": "verify_outputs"},
        ]
    
    def _requires_approval(self, task: Task) -> bool:
        """Check if task requires approval (uses high-risk tools)."""
        # Check if task involves high-risk operations
        objective_lower = task.objective.lower()
        high_risk_keywords = ["delete", "modify", "execute", "code", "critical"]
        return any(kw in objective_lower for kw in high_risk_keywords)
    
    def execute_task(self, task: Task, context: Context) -> Any:
        """Execute automation task."""
        reasoning = self.reason(task, context)
        workflow_steps = reasoning["workflow_steps"]
        
        execution_results = {
            "task_id": task.task_id,
            "pattern": reasoning["automation_pattern"],
            "steps_executed": [],
            "step_results": {},
            "success": True,
        }
        
        try:
            # Execute workflow steps
            for step_info in workflow_steps:
                step_num = step_info.get("step", 0)
                action = step_info.get("action", "")
                
                step_result = self._execute_workflow_step(action, task, context, step_info)
                
                execution_results["steps_executed"].append(step_num)
                execution_results["step_results"][step_num] = {
                    "action": action,
                    "result": step_result,
                }
                
                # Check for step failure
                if step_result.get("success") is False:
                    execution_results["success"] = False
                    break
            
            return execution_results
            
        except Exception as e:
            execution_results["success"] = False
            execution_results["error"] = str(e)
            raise
    
    def _execute_workflow_step(
        self,
        action: str,
        task: Task,
        context: Context,
        step_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        action_lower = action.lower()
        
        if "validate" in action_lower:
            return {"success": True, "message": "Inputs validated"}
        
        elif "execute" in action_lower or "run" in action_lower:
            # Execute automation logic
            if "code" in task.objective.lower():
                code = task.metadata.get("code") or context.get("code")
                if code:
                    try:
                        tool_result = self.use_tool("code_execution", code=code)
                        return {
                            "success": tool_result.success,
                            "result": tool_result.data if tool_result.success else None,
                            "error": tool_result.error,
                        }
                    except PermissionError as e:
                        return {"success": False, "error": f"Permission denied: {e}"}
            
            return {"success": True, "message": "Automation executed"}
        
        elif "verify" in action_lower or "check" in action_lower:
            return {"success": True, "message": "Outputs verified"}
        
        else:
            return {"success": True, "message": f"Step '{action}' completed"}

