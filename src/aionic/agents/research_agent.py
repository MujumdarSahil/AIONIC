"""
Research Agent - Research and analysis specialist.

Specialized agent for conducting research, gathering information,
synthesizing findings, and generating reports.
"""

from typing import Any, Dict, List

from .base_agent import BaseAgent
from ..core.task import Task
from ..core.context import Context
from ..memory.memory_store import MemoryStore, MemoryType
from ..security.autonomy_policy import AutonomyPolicy
from ..core.tool import ToolRegistry


class ResearchAgent(BaseAgent):
    """
    Research Agent for conducting research and analysis.
    
    Specialized for:
    - Information gathering
    - Source verification
    - Analysis and synthesis
    - Report generation
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        memory: MemoryStore,
        autonomy_policy: AutonomyPolicy,
        tool_registry: ToolRegistry,
        initial_competence: float = 0.65,
    ):
        """Initialize research agent."""
        super().__init__(
            agent_id=agent_id,
            name=name,
            goal="Conduct thorough research and provide well-sourced analysis",
            memory=memory,
            autonomy_policy=autonomy_policy,
            tool_registry=tool_registry,
            initial_competence=initial_competence,
            agent_type="research",
        )
        
        # Register research tools
        self.register_tools(["web_search", "data_analysis", "file_read"])
    
    def reason(self, task: Task, context: Context) -> Dict[str, Any]:
        """Generate research-specific reasoning."""
        # Extract research questions
        research_questions = self._extract_research_questions(task)
        
        # Plan research methodology
        methodology = {
            "questions": research_questions,
            "approach": "multi_source_verification",
            "steps": [
                "Formulate research questions",
                "Gather information from multiple sources",
                "Verify source credibility",
                "Analyze and synthesize findings",
                "Generate comprehensive report",
            ],
            "source_count_target": len(research_questions) * 3,  # 3 sources per question
        }
        
        reasoning = {
            "task_id": task.task_id,
            "agent_type": "research",
            "research_questions": research_questions,
            "methodology": methodology,
            "confidence": self._calculate_research_confidence(task),
        }
        
        self._log_reasoning(
            action="research_reasoning",
            task_id=task.task_id,
            questions=research_questions,
            methodology=methodology,
        )
        
        return reasoning
    
    def _extract_research_questions(self, task: Task) -> List[str]:
        """Extract research questions from task."""
        # Simple extraction - can be enhanced with NLP
        objective = task.objective
        
        # Split by question marks or keywords
        questions = []
        if "?" in objective:
            questions = [q.strip() for q in objective.split("?") if q.strip()]
        else:
            # Extract key topics
            questions = [objective]  # Single main question
        
        return questions[:5]  # Limit to 5 questions
    
    def _calculate_research_confidence(self, task: Task) -> float:
        """Calculate confidence for research task."""
        base_confidence = self.competence_score * 0.8
        
        # Boost if we have web search capability
        if "web_search" in self.get_available_tools():
            base_confidence += 0.15
        
        # Slight boost for data analysis capability
        if "data_analysis" in self.get_available_tools():
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def execute_task(self, task: Task, context: Context) -> Any:
        """Execute research task."""
        reasoning = self.reason(task, context)
        research_questions = reasoning["research_questions"]
        
        # Gather information for each question
        research_findings = []
        
        for question in research_questions:
            # Search for information
            try:
                search_result = self.use_tool("web_search", query=question, max_results=5)
                if search_result.success:
                    findings = {
                        "question": question,
                        "sources": search_result.data.get("results", []),
                        "source_count": len(search_result.data.get("results", [])),
                    }
                    research_findings.append(findings)
            except Exception as e:
                self._log_reasoning(
                    action="research_error",
                    question=question,
                    error=str(e),
                )
        
        # Synthesize findings
        synthesis = self._synthesize_findings(research_questions, research_findings)
        
        # Generate report
        report = {
            "task_id": task.task_id,
            "research_questions": research_questions,
            "findings": research_findings,
            "synthesis": synthesis,
            "total_sources": sum(f.get("source_count", 0) for f in research_findings),
            "reasoning": reasoning,
        }
        
        # Store in memory
        self.memory.store(
            agent_id=self.agent_id,
            content={
                "research_topic": task.objective,
                "report": report,
                "questions": research_questions,
            },
            memory_type=MemoryType.SEMANTIC,
            importance_score=0.9,
        )
        
        context.update("research_report", report)
        return report
    
    def _synthesize_findings(
        self,
        questions: List[str],
        findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synthesize research findings into coherent summary."""
        # Placeholder synthesis - in production, use LLM or advanced NLP
        synthesis = {
            "summary": f"Research conducted on {len(questions)} question(s) with {len(findings)} finding set(s)",
            "key_points": [
                f"Question {i+1}: {q}" for i, q in enumerate(questions)
            ],
            "source_quality": "verified" if len(findings) > 0 else "insufficient",
        }
        return synthesis

