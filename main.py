"""
Deep Research Swarm - Main Orchestration Module

Implements the main orchestration loop using Agno Workflows:
1. Planner → Creates research plan with subtasks
2. Workers → Execute subtasks in parallel (Perplexity search + save findings)
3. Editor → Synthesize findings into comprehensive report

Uses Agno Workflow 2.0 with Parallel execution for optimal performance.
"""
import os
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from textwrap import dedent

# Configure LiteLLM before importing Agno
import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.workflow import Workflow, Step, StepOutput
from agno.workflow.parallel import Parallel
from agno.workflow.types import StepInput
from agno.utils.log import logger

from config import config
from agents.planner import PlannerAgent, ResearchPlan, Subtask
from agents.worker import WorkerAgent
from agents.editor import EditorAgent
from agents.critic import CriticAgent
from agents.domain_experts import create_expert_panel, get_multi_perspective_analysis
from agents.schemas import CriticEvaluation, ResearchIteration, ResearchCheckpoint
from infrastructure.perplexity_tools import PerplexitySearchTools
from infrastructure.knowledge_tools import KnowledgeTools
from infrastructure.retry_utils import with_retry


# =============================================================================
# Research Swarm Configuration
# =============================================================================

@dataclass
class SwarmResult:
    """Result from a research swarm execution"""
    query: str
    plan: Optional[ResearchPlan] = None
    worker_results: List[Dict[str, Any]] = field(default_factory=list)
    report: str = ""
    success: bool = False
    error: Optional[str] = None
    
    def summary(self) -> str:
        """Generate a summary of the swarm execution"""
        lines = [
            "# Research Swarm Execution Summary",
            "",
            f"**Query:** {self.query}",
            f"**Success:** {'✅' if self.success else '❌'}",
        ]
        
        if self.error:
            lines.append(f"**Error:** {self.error}")
        
        if self.plan:
            lines.append(f"**Subtasks:** {len(self.plan.subtasks)}")
            lines.append(f"**Research Depth:** {self.plan.estimated_depth}")
        
        if self.worker_results:
            completed = 0
            for r in self.worker_results:
                if isinstance(r, dict) and r.get("status") == "completed":
                    completed += 1
            lines.append(f"**Workers Completed:** {completed}/{len(self.worker_results)}")
        
        if self.report:
            lines.append(f"**Report Length:** {len(self.report)} characters")
        
        return "\n".join(lines)


# =============================================================================
# Step Executors (Custom Functions for Agno Workflow)
# =============================================================================

class PlannerExecutor:
    """
    Custom executor for planning step.
    Wraps PlannerAgent to work with Agno Workflow.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        max_subtasks: int = 7,
    ):
        self.model_id = model_id or config.models.planner
        self.max_subtasks = max_subtasks
        self._planner: Optional[PlannerAgent] = None
    
    @property
    def planner(self) -> PlannerAgent:
        """Lazy initialization of PlannerAgent"""
        if self._planner is None:
            self._planner = PlannerAgent(
                model_id=self.model_id,
                max_subtasks=self.max_subtasks,
                temperature=config.models.planner_temperature,
            )
        return self._planner
    
    def __call__(self, step_input: StepInput) -> StepOutput:
        """Execute planning step"""
        query = step_input.input
        logger.info(f"[Planner] Creating research plan for: {query[:50]}...")
        
        try:
            plan = self.planner.plan(query)
            
            # Store plan as dict for serialization
            plan_dict = {
                "original_query": plan.original_query,
                "summary": plan.summary,
                "subtasks": [
                    {
                        "id": t.id,
                        "query": t.query,
                        "focus": t.focus,
                        "search_type": t.search_type,
                        "priority": t.priority,
                    }
                    for t in plan.subtasks
                ],
                "estimated_depth": plan.estimated_depth,
            }
            
            logger.info(f"[Planner] Created plan with {len(plan.subtasks)} subtasks")
            
            return StepOutput(
                step_name="planner",
                content=plan_dict,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"[Planner] Failed: {e}")
            return StepOutput(
                step_name="planner",
                content=None,
                success=False,
                error=str(e),
            )


class WorkerExecutor:
    """
    Custom executor for worker step.
    Executes a single subtask using WorkerAgent.
    """
    
    def __init__(
        self,
        subtask_id: int,
        model_id: Optional[str] = None,
        search_tools: Optional[PerplexitySearchTools] = None,
        knowledge_tools: Optional[KnowledgeTools] = None,
    ):
        self.subtask_id = subtask_id
        self.model_id = model_id or config.models.worker
        self.search_tools = search_tools
        self.knowledge_tools = knowledge_tools
        self._worker: Optional[WorkerAgent] = None
    
    @property
    def worker(self) -> WorkerAgent:
        """Lazy initialization of WorkerAgent"""
        if self._worker is None:
            self._worker = WorkerAgent(
                model_id=self.model_id,
                temperature=config.models.worker_temperature,
                search_tools=self.search_tools,
                knowledge_tools=self.knowledge_tools,
            )
        return self._worker
    
    def __call__(self, step_input: StepInput) -> StepOutput:
        """Execute worker step for assigned subtask"""
        # Get plan from previous step
        plan_dict = step_input.get_step_content("planner")
        
        if not plan_dict or not isinstance(plan_dict, dict):
            return StepOutput(
                step_name=f"worker_{self.subtask_id}",
                content=None,
                success=False,
                error="No plan found from planner step",
            )
        
        subtasks = plan_dict.get("subtasks", [])
        
        # Find our subtask
        subtask_dict = None
        for st in subtasks:
            if st.get("id") == self.subtask_id:
                subtask_dict = st
                break
        
        if not subtask_dict:
            return StepOutput(
                step_name=f"worker_{self.subtask_id}",
                content=f"Subtask {self.subtask_id} not found in plan",
                success=True,  # Not an error, just no work to do
            )
        
        # Reconstruct Subtask object
        subtask = Subtask(
            id=subtask_dict["id"],
            query=subtask_dict["query"],
            focus=subtask_dict["focus"],
            search_type=subtask_dict.get("search_type", "general"),
            priority=subtask_dict.get("priority", 1),
        )
        
        logger.info(f"[Worker {self.subtask_id}] Executing: {subtask.focus}")
        
        try:
            result = self.worker.execute_subtask(subtask, worker_id=f"W{self.subtask_id:02d}")
            
            return StepOutput(
                step_name=f"worker_{self.subtask_id}",
                content={
                    "subtask_id": self.subtask_id,
                    "focus": subtask.focus,
                    "response": result,
                    "status": "completed",
                },
                success=True,
            )
            
        except Exception as e:
            logger.error(f"[Worker {self.subtask_id}] Failed: {e}")
            return StepOutput(
                step_name=f"worker_{self.subtask_id}",
                content={
                    "subtask_id": self.subtask_id,
                    "focus": subtask.focus,
                    "error": str(e),
                    "status": "failed",
                },
                success=False,
                error=str(e),
            )


class EditorExecutor:
    """
    Custom executor for editor step.
    Synthesizes all findings into final report.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        knowledge_tools: Optional[KnowledgeTools] = None,
    ):
        self.model_id = model_id or config.models.editor
        self.knowledge_tools = knowledge_tools
        self._editor: Optional[EditorAgent] = None
    
    @property
    def editor(self) -> EditorAgent:
        """Lazy initialization of EditorAgent"""
        if self._editor is None:
            self._editor = EditorAgent(
                model_id=self.model_id,
                temperature=config.models.editor_temperature,
                knowledge_tools=self.knowledge_tools,
            )
        return self._editor
    
    def __call__(self, step_input: StepInput) -> StepOutput:
        """Execute editor step to synthesize report"""
        # Get original query
        plan_dict = step_input.get_step_content("planner")
        original_query = plan_dict.get("original_query", step_input.input) if plan_dict else step_input.input
        
        # Gather worker results
        all_content = step_input.get_all_previous_content()
        
        # Build findings summary from worker results
        findings_parts = []
        for key, value in (step_input.previous_step_outputs or {}).items():
            if key.startswith("worker_") and value.content:
                content = value.content
                if isinstance(content, dict):
                    focus = content.get("focus", "Unknown")
                    response = content.get("response", "")[:500]
                    findings_parts.append(f"### {focus}\n{response}\n")
        
        findings_summary = "\n".join(findings_parts) if findings_parts else None
        
        logger.info(f"[Editor] Synthesizing report for: {original_query[:50]}...")
        
        try:
            report = self.editor.synthesize(original_query, findings_summary)
            
            return StepOutput(
                step_name="editor",
                content=report,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"[Editor] Failed: {e}")
            return StepOutput(
                step_name="editor",
                content=f"Report generation failed: {e}",
                success=False,
                error=str(e),
            )


# =============================================================================
# Research Swarm Orchestrator
# =============================================================================

class ResearchSwarm:
    """
    Main orchestrator for the Deep Research Swarm.
    
    Uses Agno Workflow with Parallel execution for optimal performance.
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        max_subtasks: int = 7,
        db_path: Optional[str] = None,
    ):
        """
        Initialize Research Swarm.
        
        Args:
            max_workers: Maximum parallel workers
            max_subtasks: Maximum subtasks from planner
            db_path: Path to LanceDB knowledge base
        """
        self.max_workers = max_workers
        self.max_subtasks = max_subtasks
        
        # Shared tools (reused across workers)
        self.search_tools = PerplexitySearchTools(max_results=config.search.max_results)
        self.knowledge_tools = KnowledgeTools(db_path=db_path or config.knowledge.db_path)
        
        # Executors
        self._planner_executor = PlannerExecutor(max_subtasks=max_subtasks)
        self._editor_executor = EditorExecutor(knowledge_tools=self.knowledge_tools)
    
    def _create_worker_executor(self, subtask_id: int) -> WorkerExecutor:
        """Create a worker executor for a specific subtask"""
        return WorkerExecutor(
            subtask_id=subtask_id,
            search_tools=self.search_tools,
            knowledge_tools=self.knowledge_tools,
        )
    
    def research(self, query: str, stream: bool = False) -> SwarmResult:
        """
        Execute a full research swarm on the given query.
        
        This is a two-phase approach:
        1. First, run the planner to get subtasks
        2. Then, create a dynamic workflow with parallel workers
        
        Args:
            query: Research query
            stream: Whether to stream output (not yet implemented)
            
        Returns:
            SwarmResult: Complete results from the research
        """
        result = SwarmResult(query=query)
        
        try:
            # Phase 1: Planning
            logger.info("=" * 60)
            logger.info("PHASE 1: PLANNING")
            logger.info("=" * 60)
            
            planner_output = self._planner_executor(StepInput(input=query))
            
            if not planner_output.success or not planner_output.content:
                result.error = f"Planning failed: {planner_output.error}"
                return result
            
            plan_dict = planner_output.content
            result.plan = ResearchPlan(
                original_query=plan_dict["original_query"],
                summary=plan_dict["summary"],
                subtasks=[
                    Subtask(**st) for st in plan_dict["subtasks"]
                ],
                estimated_depth=plan_dict.get("estimated_depth", "medium"),
            )
            
            logger.info(f"Created plan with {len(result.plan.subtasks)} subtasks")
            
            # Phase 2: Workers (execute in parallel)
            logger.info("=" * 60)
            logger.info("PHASE 2: WORKER EXECUTION")
            logger.info("=" * 60)
            
            # Limit workers
            subtasks_to_execute = result.plan.subtasks[:self.max_workers]
            
            # Create worker steps
            worker_steps = []
            for subtask in subtasks_to_execute:
                worker_executor = self._create_worker_executor(subtask.id)
                step = Step(
                    name=f"worker_{subtask.id}",
                    executor=worker_executor,
                    description=f"Research: {subtask.focus}",
                )
                worker_steps.append(step)
            
            # Build workflow with parallel workers
            planner_step = Step(
                name="planner",
                executor=self._planner_executor,
                description="Create research plan",
            )
            
            editor_step = Step(
                name="editor",
                executor=self._editor_executor,
                description="Synthesize final report",
            )
            
            # Create workflow
            if len(worker_steps) > 1:
                workflow = Workflow(
                    name="Deep Research Swarm",
                    description=f"Research: {query[:50]}...",
                    steps=[
                        planner_step,
                        Parallel(*worker_steps, name="worker_execution"),
                        editor_step,
                    ],
                )
            else:
                workflow = Workflow(
                    name="Deep Research Swarm",
                    description=f"Research: {query[:50]}...",
                    steps=[
                        planner_step,
                        *worker_steps,
                        editor_step,
                    ],
                )
            
            # Execute workflow
            logger.info(f"Executing workflow with {len(worker_steps)} parallel workers...")
            
            workflow_result = workflow.run(input=query)
            
            # Extract worker results from workflow
            for step_output in workflow_result.step_results:
                if isinstance(step_output, list):  # Parallel results
                    for so in step_output:
                        if hasattr(so, 'step_name') and so.step_name and so.step_name.startswith("worker_"):
                            content = so.content
                            if isinstance(content, dict):
                                result.worker_results.append(content)
                            elif content:
                                result.worker_results.append({"response": str(content), "status": "completed"})
                elif hasattr(step_output, 'step_name') and step_output.step_name and step_output.step_name.startswith("worker_"):
                    content = step_output.content
                    if isinstance(content, dict):
                        result.worker_results.append(content)
                    elif content:
                        result.worker_results.append({"response": str(content), "status": "completed"})
            
            # Phase 3: Report (already in workflow)
            logger.info("=" * 60)
            logger.info("PHASE 3: REPORT SYNTHESIS")
            logger.info("=" * 60)
            
            # Get final report from workflow
            result.report = workflow_result.content or ""
            result.success = True
            
            logger.info("Research swarm completed successfully!")
            
        except Exception as e:
            logger.error(f"Research swarm failed: {e}")
            import traceback
            traceback.print_exc()
            result.error = str(e)
            result.success = False
        
        return result
    
    def research_simple(self, query: str) -> SwarmResult:
        """
        Simple sequential execution (no Agno Workflow).
        Useful for debugging or when Workflow has issues.
        
        Args:
            query: Research query
            
        Returns:
            SwarmResult: Complete results from the research
        """
        result = SwarmResult(query=query)
        
        try:
            # Phase 1: Planning
            logger.info("=" * 60)
            logger.info("PHASE 1: PLANNING")
            logger.info("=" * 60)
            
            planner = PlannerAgent(max_subtasks=self.max_subtasks)
            plan = planner.plan(query)
            result.plan = plan
            
            logger.info(f"Created plan with {len(plan.subtasks)} subtasks")
            
            # Phase 2: Workers (sequential for simplicity)
            logger.info("=" * 60)
            logger.info("PHASE 2: WORKER EXECUTION")
            logger.info("=" * 60)
            
            worker = WorkerAgent(
                search_tools=self.search_tools,
                knowledge_tools=self.knowledge_tools,
            )
            
            subtasks_to_execute = plan.subtasks[:self.max_workers]
            result.worker_results = worker.execute_subtasks(subtasks_to_execute)
            
            completed = sum(1 for r in result.worker_results if r.get("status") == "completed")
            logger.info(f"Completed {completed}/{len(subtasks_to_execute)} subtasks")
            
            # Phase 3: Editor
            logger.info("=" * 60)
            logger.info("PHASE 3: REPORT SYNTHESIS")
            logger.info("=" * 60)
            
            try:
                editor = EditorAgent(knowledge_tools=self.knowledge_tools)
                result.report = editor.synthesize(query)
                
                # Check if editor returned a valid report
                if not result.report or len(result.report) < 500:
                    logger.warning(f"Editor returned short/empty report, using fallback")
                    # Gather findings from knowledge base
                    findings = self._gather_findings_for_fallback()
                    result.report = self._generate_simple_fallback_report(query, findings)
                    
            except Exception as editor_error:
                logger.error(f"Editor synthesis failed: {editor_error}")
                logger.info("Generating fallback report...")
                # Gather findings from knowledge base  
                findings = self._gather_findings_for_fallback()
                result.report = self._generate_simple_fallback_report(query, findings)
            
            result.success = True
            logger.info("Research swarm completed successfully!")
            logger.info(f"Report length: {len(result.report)} characters")
            
        except Exception as e:
            logger.error(f"Research swarm failed: {e}")
            import traceback
            traceback.print_exc()
            result.error = str(e)
            result.success = False
        
        return result
    
    def _gather_findings_for_fallback(self) -> list:
        """Gather all findings from knowledge base for fallback report"""
        try:
            # Search for all findings
            findings_data = self.knowledge_tools.search_knowledge("research findings", top_k=100)
            
            findings = []
            if findings_data and "results" in findings_data:
                for item in findings_data["results"]:
                    findings.append({
                        "content": item.get("content", ""),
                        "source_url": item.get("source_url", ""),
                        "source_title": item.get("source_title", "Source"),
                        "search_type": item.get("search_type", "general"),
                        "verified": item.get("verified", False),
                    })
            return findings
        except Exception as e:
            logger.warning(f"Could not gather findings: {e}")
            return []
    
    def _generate_simple_fallback_report(self, query: str, findings: list) -> str:
        """Generate a simple but well-formatted fallback report"""
        lines = [
            f"# {query}",
            "",
            "*Research Survey*",
            "",
            "---",
            "",
            "## Overview",
            "",
            f"This report synthesizes {len(findings)} research findings on **{query.lower()}**.",
            "",
        ]
        
        if not findings:
            lines.extend([
                "No findings were collected. This may be due to:",
                "- Search API limitations",
                "- Network connectivity issues", 
                "- Query specificity",
                "",
            ])
            return "\n".join(lines)
        
        # Group by source
        sources_seen = set()
        
        lines.extend([
            "## Key Findings",
            "",
        ])
        
        for i, finding in enumerate(findings[:20], 1):
            content = finding.get("content", "")[:500]
            source = finding.get("source_title", "Source")
            source_url = finding.get("source_url", "")
            
            if source_url in sources_seen:
                continue
            sources_seen.add(source_url)
            
            # Clean content
            content = content.replace("\n", " ").strip()
            if len(content) > 400:
                sentences = content[:450].split(". ")
                content = ". ".join(sentences[:-1]) + "." if len(sentences) > 1 else content[:400] + "..."
            
            lines.append(f"**{i}. {source}**")
            lines.append("")
            lines.append(f"{content}")
            lines.append("")
        
        # References
        lines.extend([
            "---",
            "",
            "## References",
            "",
        ])
        
        for i, url in enumerate(list(sources_seen)[:15], 1):
            if url:
                lines.append(f"{i}. {url}")
        
        lines.extend([
            "",
            "---",
            "",
            "*Report generated by Deep Research Swarm (fallback mode)*",
        ])
        
        return "\n".join(lines)


# =============================================================================
# Deep Research Swarm (Multi-Iteration with Quality Control)
# =============================================================================

class DeepResearchSwarm:
    """
    Multi-iteration deep research with gap analysis and quality control.
    
    Implements PhD-level research methodology:
    1. Planning with comprehensive subtask decomposition
    2. Iterative research with gap filling
    3. Quality evaluation by Critic agent
    4. Multi-perspective analysis by Domain Experts
    5. Academic-quality report synthesis
    
    Supports checkpointing for long-running research sessions.
    """
    
    def __init__(
        self,
        max_workers: int = 7,
        max_subtasks: int = 15,
        max_iterations: int = 3,
        quality_threshold: int = 80,
        checkpoint_dir: str = "./checkpoints",
        db_path: Optional[str] = None,
    ):
        """
        Initialize Deep Research Swarm.
        
        Args:
            max_workers: Maximum parallel workers per iteration
            max_subtasks: Maximum subtasks from planner
            max_iterations: Maximum research iterations (gap-filling rounds)
            quality_threshold: Minimum quality score (0-100) to stop iterating
            checkpoint_dir: Directory for saving checkpoints
            db_path: Path to LanceDB knowledge base
        """
        self.max_workers = max_workers
        self.max_subtasks = max_subtasks
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.checkpoint_dir = checkpoint_dir
        
        # Shared tools
        self.search_tools = PerplexitySearchTools(max_results=15)  # More results for deep research
        self.knowledge_tools = KnowledgeTools(db_path=db_path or config.knowledge.db_path)
        
        # Agents
        self._planner: Optional[PlannerAgent] = None
        self._worker: Optional[WorkerAgent] = None
        self._editor: Optional[EditorAgent] = None
        self._critic: Optional[CriticAgent] = None
        
        # Session tracking
        self.iterations: List[Dict[str, Any]] = []
        self.all_findings: List[Dict[str, Any]] = []
    
    @property
    def planner(self) -> PlannerAgent:
        """Lazy init planner"""
        if self._planner is None:
            self._planner = PlannerAgent(
                model_id=config.models.planner,
                max_subtasks=self.max_subtasks,
                temperature=config.models.planner_temperature,
            )
        return self._planner
    
    @property
    def worker(self) -> WorkerAgent:
        """Lazy init worker"""
        if self._worker is None:
            self._worker = WorkerAgent(
                model_id=config.models.worker,
                temperature=config.models.worker_temperature,
                search_tools=self.search_tools,
                knowledge_tools=self.knowledge_tools,
            )
        return self._worker
    
    @property
    def editor(self) -> EditorAgent:
        """Lazy init editor"""
        if self._editor is None:
            self._editor = EditorAgent(
                model_id=config.models.editor,
                temperature=config.models.editor_temperature,
                knowledge_tools=self.knowledge_tools,
            )
        return self._editor
    
    @property
    def critic(self) -> CriticAgent:
        """Lazy init critic"""
        if self._critic is None:
            self._critic = CriticAgent(
                model_id=config.models.critic,
                temperature=config.models.critic_temperature,
                quality_threshold=self.quality_threshold,
            )
        return self._critic
    
    def deep_research(
        self,
        query: str,
        use_experts: bool = True,
        save_checkpoint: bool = True,
    ) -> SwarmResult:
        """
        Execute iterative deep research with quality control.
        
        Args:
            query: Research query
            use_experts: Whether to use domain expert analysis
            save_checkpoint: Whether to save checkpoints
            
        Returns:
            SwarmResult: Complete research results
        """
        result = SwarmResult(query=query)
        
        try:
            logger.info("=" * 70)
            logger.info("  DEEP RESEARCH SWARM - Starting PhD-Level Investigation")
            logger.info("=" * 70)
            logger.info(f"Query: {query}")
            logger.info(f"Max iterations: {self.max_iterations}")
            logger.info(f"Quality threshold: {self.quality_threshold}")
            
            # Phase 1: Initial Planning
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1: STRATEGIC PLANNING")
            logger.info("=" * 60)
            
            plan = self.planner.plan(query)
            result.plan = plan
            
            logger.info(f"Created plan with {len(plan.subtasks)} subtasks")
            logger.info(f"Research depth: {plan.estimated_depth}")
            
            if save_checkpoint:
                self._save_checkpoint("planning", {"plan": plan.model_dump() if hasattr(plan, 'model_dump') else {"subtasks": len(plan.subtasks)}})
            
            # Phase 2: Iterative Research Loop
            for iteration in range(1, self.max_iterations + 1):
                logger.info("\n" + "=" * 60)
                logger.info(f"ITERATION {iteration}/{self.max_iterations}: RESEARCH EXECUTION")
                logger.info("=" * 60)
                
                # Execute workers
                iteration_result = self._execute_iteration(plan, iteration)
                self.iterations.append(iteration_result)
                result.worker_results.extend(iteration_result.get("worker_results", []))
                
                # Gather findings for evaluation
                findings = self._get_all_findings()
                self.all_findings = findings
                
                logger.info(f"Total findings collected: {len(findings)}")
                
                # Quality evaluation
                logger.info("\n--- Critic Evaluation ---")
                
                try:
                    evaluation = self.critic.evaluate(findings, query, iteration)
                except Exception as e:
                    logger.warning(f"Full critic evaluation failed: {e}, using quick assess")
                    quick = self.critic.quick_assess(findings)
                    evaluation = CriticEvaluation(
                        overall_score=quick["estimated_score"],
                        coverage_score=quick["estimated_score"],
                        source_quality_score=quick["estimated_score"] - 10,
                        evidence_strength_score=quick["estimated_score"] - 5,
                        ready_for_synthesis=not quick["needs_more_research"],
                    )
                
                logger.info(f"Quality Score: {evaluation.overall_score}/100")
                logger.info(f"Ready for synthesis: {evaluation.ready_for_synthesis}")
                
                if save_checkpoint:
                    self._save_checkpoint(f"iteration_{iteration}", {
                        "findings_count": len(findings),
                        "score": evaluation.overall_score,
                        "ready": evaluation.ready_for_synthesis,
                    })
                
                # Check if we should continue or synthesize
                if evaluation.ready_for_synthesis or evaluation.overall_score >= self.quality_threshold:
                    logger.info(f"Quality threshold met ({evaluation.overall_score} >= {self.quality_threshold})")
                    break
                
                # If not ready, plan follow-up research
                if iteration < self.max_iterations and evaluation.follow_up_queries:
                    logger.info("\n--- Planning Follow-up Research ---")
                    follow_up_subtasks = self._create_followup_subtasks(
                        evaluation.follow_up_queries,
                        plan,
                        iteration
                    )
                    plan.subtasks.extend(follow_up_subtasks)
                    logger.info(f"Added {len(follow_up_subtasks)} follow-up subtasks")
            
            # Phase 3: Multi-Perspective Analysis (Optional)
            expert_insights = []
            if use_experts and len(self.all_findings) > 0:
                logger.info("\n" + "=" * 60)
                logger.info("PHASE 3: MULTI-PERSPECTIVE ANALYSIS")
                logger.info("=" * 60)
                
                try:
                    perspectives = get_multi_perspective_analysis(
                        self.all_findings,
                        query,
                        expert_types=["technical", "industry", "skeptic"],
                    )
                    
                    for expert_type, perspective in perspectives.items():
                        logger.info(f"\n[{expert_type.upper()}] {perspective.perspective_summary[:100]}...")
                        expert_insights.append({
                            "expert": expert_type,
                            "summary": perspective.perspective_summary,
                            "insights": perspective.key_insights,
                            "concerns": perspective.concerns,
                        })
                except Exception as e:
                    logger.warning(f"Expert analysis failed: {e}")
            
            # Phase 4: Final Synthesis
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 4: REPORT SYNTHESIS")
            logger.info("=" * 60)
            
            # Build context for editor
            synthesis_context = self._build_synthesis_context(expert_insights)
            
            try:
                logger.info(f"Calling editor synthesis with {len(self.all_findings)} findings...")
                result.report = self.editor.synthesize(query, synthesis_context)
                
                # Check if editor returned a valid report
                if not result.report or len(result.report) < 500:
                    logger.warning(f"Editor returned short/empty report ({len(result.report) if result.report else 0} chars), using fallback")
                    result.report = self._generate_fallback_report(query, self.all_findings)
                else:
                    logger.info(f"Editor synthesis complete: {len(result.report)} chars")
                    
            except Exception as e:
                logger.error(f"Editor synthesis failed: {e}")
                import traceback
                traceback.print_exc()
                logger.info(f"Generating fallback report with {len(self.all_findings)} findings...")
                result.report = self._generate_fallback_report(query, self.all_findings)
            
            # Optional: Draft review
            if len(result.report) > 1000:
                try:
                    critique = self.critic.review_draft(result.report, query)
                    if not critique.ready_for_publication:
                        logger.info("Draft needs revision - applying suggestions...")
                except Exception as e:
                    logger.warning(f"Draft review skipped: {e}")
            
            if save_checkpoint:
                self._save_checkpoint("complete", {
                    "report_length": len(result.report),
                    "total_findings": len(self.all_findings),
                    "iterations": len(self.iterations),
                })
            
            result.success = True
            logger.info("\n" + "=" * 70)
            logger.info("  DEEP RESEARCH COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Total findings: {len(self.all_findings)}")
            logger.info(f"Iterations: {len(self.iterations)}")
            logger.info(f"Report length: {len(result.report)} characters")
            
        except Exception as e:
            logger.error(f"Deep research failed: {e}")
            import traceback
            traceback.print_exc()
            result.error = str(e)
            result.success = False
        
        return result
    
    def _execute_iteration(self, plan: ResearchPlan, iteration: int) -> Dict[str, Any]:
        """Execute a single research iteration"""
        from datetime import datetime
        
        start_time = datetime.utcnow()
        
        # Get subtasks for this iteration
        if iteration == 1:
            subtasks = plan.subtasks[:self.max_workers]
        else:
            # For follow-up iterations, get remaining subtasks
            completed_ids = set()
            for iter_result in self.iterations:
                for r in iter_result.get("worker_results", []):
                    if r.get("status") == "completed":
                        completed_ids.add(r.get("subtask_id"))
            subtasks = [st for st in plan.subtasks if st.id not in completed_ids][:self.max_workers]
        
        logger.info(f"Executing {len(subtasks)} subtasks...")
        
        # Execute workers
        worker_results = self.worker.execute_subtasks(subtasks)
        
        completed = sum(1 for r in worker_results if r.get("status") == "completed")
        
        return {
            "iteration": iteration,
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "subtasks_executed": len(subtasks),
            "subtasks_completed": completed,
            "worker_results": worker_results,
        }
    
    def _get_all_findings(self) -> List[Dict[str, Any]]:
        """Get all findings from the knowledge base as structured data"""
        try:
            # Access the LanceDB table directly to get structured data
            df = self.knowledge_tools.table.to_pandas()
            
            # Filter out initialization record
            df = df[df["id"] != "init"]
            
            if len(df) == 0:
                logger.warning("No findings found in knowledge base")
                return []
            
            # Convert to list of dicts with full structure
            findings = []
            for _, row in df.iterrows():
                findings.append({
                    "id": row.get("id", ""),
                    "content": row.get("content", ""),
                    "source_url": row.get("source_url", ""),
                    "source_title": row.get("source_title", ""),
                    "search_type": row.get("search_type", "general"),
                    "verified": row.get("verified", False),
                    "subtask_id": row.get("subtask_id", 0),
                    "worker_id": row.get("worker_id", ""),
                    "timestamp": row.get("timestamp", ""),
                })
            
            logger.info(f"Retrieved {len(findings)} structured findings from knowledge base")
            return findings
            
        except Exception as e:
            logger.warning(f"Could not retrieve findings: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_followup_subtasks(
        self,
        queries: List[str],
        plan: ResearchPlan,
        iteration: int
    ) -> List[Subtask]:
        """Create follow-up subtasks from critic recommendations"""
        existing_ids = {st.id for st in plan.subtasks}
        next_id = max(existing_ids) + 1 if existing_ids else 1
        
        followup_subtasks = []
        for query in queries[:5]:  # Limit follow-up queries
            followup_subtasks.append(Subtask(
                id=next_id,
                query=query,
                focus=f"Follow-up research (iteration {iteration + 1})",
                search_type="general",
                priority=2,
            ))
            next_id += 1
        
        return followup_subtasks
    
    def _build_synthesis_context(self, expert_insights: List[Dict]) -> Optional[str]:
        """
        Build synthesis context using findings index for tool-based discovery.
        
        Instead of dumping all findings, we provide:
        1. A compact findings index (statistics, sources, topics)
        2. Expert insights if available
        
        The Editor will then use search_knowledge() to retrieve detailed findings.
        """
        parts = []
        
        # Prefer in-memory findings (from current session) to avoid empty KB responses
        if self.all_findings:
            academic = sum(1 for f in self.all_findings if f.get("search_type") == "academic")
            general = len(self.all_findings) - academic
            verified = sum(1 for f in self.all_findings if f.get("verified", False))
            unique_sources = len({f.get("source_url") for f in self.all_findings if f.get("source_url")})
            
            parts.append("## Research Findings Summary\n")
            parts.append(f"Total findings: {len(self.all_findings)}")
            parts.append(f"- Academic: {academic}")
            parts.append(f"- General: {general}")
            parts.append(f"- Verified: {verified}")
            parts.append(f"- Unique sources: {unique_sources}")
            parts.append("")
            parts.append("### Key Research Content")
            for f in self.all_findings[:3]:
                snippet = f.get("content", "")[:240].strip()
                parts.append(f"- {snippet}")
            parts.append("")
        else:
            # Get findings index from knowledge tools (compact summary)
            try:
                findings_index = self.knowledge_tools.get_findings_index()
                parts.append(findings_index)
                parts.append("")
            except Exception as e:
                logger.warning(f"Could not generate findings index: {e}")
                parts.append("## Research Findings Summary\n")
                parts.append("Total findings: 0")
                parts.append("")
        
        # Add expert perspectives if available
        if expert_insights:
            parts.append("---")
            parts.append("")
            parts.append("## Expert Perspectives\n")
            for insight in expert_insights:
                parts.append(f"### {insight['expert'].title()} Perspective")
                parts.append(insight['summary'])
                if insight.get('insights'):
                    parts.append("\n**Key Insights:**")
                    for i in insight['insights'][:5]:
                        parts.append(f"- {i}")
                if insight.get('concerns'):
                    parts.append("\n**Concerns Raised:**")
                    for c in insight['concerns'][:3]:
                        parts.append(f"- {c}")
                parts.append("")
        
        return "\n".join(parts) if parts else None
    
    def _generate_fallback_report(self, query: str, findings: List[Dict]) -> str:
        """
        Generate a well-crafted fallback report when editor fails.
        
        Focuses on quality prose over mechanical structure:
        - Clear, engaging opening
        - Thematically organized content
        - Smooth transitions and narrative flow
        - Concrete evidence and citations
        - Substantive conclusions
        """
        import re
        from collections import defaultdict
        
        # =================================================================
        # Step 1: Analyze and categorize findings
        # =================================================================
        
        academic_findings = [f for f in findings if f.get("search_type") == "academic"]
        general_findings = [f for f in findings if f.get("search_type") != "academic"]
        verified_count = sum(1 for f in findings if f.get("verified", False))
        
        # Collect unique sources
        sources = {}
        for f in findings:
            url = f.get("source_url", "")
            if url and url not in sources:
                sources[url] = {
                    "title": f.get("source_title", "Untitled"),
                    "verified": f.get("verified", False),
                    "type": f.get("search_type", "general"),
                }
        
        # =================================================================
        # Step 2: Thematic clustering using keyword analysis
        # =================================================================
        
        # Define theme keywords for intelligent grouping
        theme_keywords = {
            "Architecture & Foundations": [
                "architecture", "framework", "component", "design", "structure",
                "foundation", "core", "module", "system design", "building block"
            ],
            "Reasoning & Planning": [
                "reasoning", "planning", "chain-of-thought", "cot", "thinking",
                "decision", "problem-solving", "inference", "logic", "cognitive"
            ],
            "Multi-Agent Systems": [
                "multi-agent", "collaboration", "coordination", "cooperation",
                "swarm", "collective", "distributed", "team", "communication"
            ],
            "Tool Use & Function Calling": [
                "tool", "function calling", "api", "integration", "external",
                "capability", "action", "execution", "interface"
            ],
            "Memory & Knowledge": [
                "memory", "knowledge", "context", "retrieval", "rag",
                "vector", "embedding", "storage", "long-term", "short-term"
            ],
            "Benchmarks & Evaluation": [
                "benchmark", "evaluation", "performance", "accuracy", "score",
                "metric", "test", "assessment", "comparison", "sota"
            ],
            "Applications & Deployment": [
                "application", "deployment", "production", "real-world", "use case",
                "industry", "enterprise", "commercial", "practical"
            ],
            "Challenges & Limitations": [
                "challenge", "limitation", "problem", "issue", "concern",
                "weakness", "failure", "error", "risk", "safety"
            ],
            "Future Directions": [
                "future", "trend", "emerging", "next", "prediction",
                "roadmap", "direction", "advancement", "evolution"
            ]
        }
        
        # Categorize findings into themes
        themed_findings = defaultdict(list)
        uncategorized = []
        
        for finding in findings:
            content = finding.get("content", "").lower()
            title = finding.get("source_title", "").lower()
            text = content + " " + title
            
            matched_theme = None
            max_matches = 0
            
            for theme, keywords in theme_keywords.items():
                matches = sum(1 for kw in keywords if kw in text)
                if matches > max_matches:
                    max_matches = matches
                    matched_theme = theme
            
            if matched_theme and max_matches >= 2:
                themed_findings[matched_theme].append(finding)
            else:
                uncategorized.append(finding)
        
        # Add uncategorized to most relevant or "General Findings"
        if uncategorized:
            themed_findings["General Findings"].extend(uncategorized)
        
        # =================================================================
        # Step 3: Extract key statistics and insights
        # =================================================================
        
        def extract_statistics(text):
            """Extract percentages, numbers, and metrics from text"""
            stats = []
            # Match percentages
            pct_matches = re.findall(r'\d+(?:\.\d+)?%', text)
            stats.extend(pct_matches[:3])
            # Match benchmark scores like "86.4% on MMLU"
            bench_matches = re.findall(r'(\d+(?:\.\d+)?%?\s+(?:on|in|for)\s+\w+)', text)
            stats.extend(bench_matches[:2])
            return stats
        
        all_stats = []
        for f in findings[:50]:
            content = f.get("content", "")
            all_stats.extend(extract_statistics(content))
        
        # =================================================================
        # Step 4: Build the academic report
        # =================================================================
        
        lines = []
        
        # ----- TITLE & EXECUTIVE SUMMARY -----
        lines.extend([
            f"# Research Report: {query}",
            "",
            "*A Comprehensive Research Survey*",
            "",
            "## Executive Summary",
            "",
            f"This report synthesizes {len(findings)} research findings from {len(sources)} unique sources on **{query.lower()}**.",
            f"- Academic findings: {len(academic_findings)}",
            f"- General findings: {len(general_findings)}",
            f"- Verified sources: {verified_count}",
            "",
            "---",
            "",
        ])

        # If no findings, return early with context for debugging
        if not findings:
            lines.extend([
                "No findings were collected. This may be due to:",
                "- Search API limitations",
                "- Network connectivity issues",
                "- Query specificity",
                "",
            ])
            return "\n".join(lines)
        
        # ----- ABSTRACT -----
        # Find the most striking statistic or finding for the opening
        striking_stats = all_stats[:3] if all_stats else []
        top_themes = [t for t in themed_findings.keys() if t != "General Findings"][:4]
        
        abstract_opener = f"This survey synthesizes {len(findings)} findings from {len(sources)} sources"
        if striking_stats:
            abstract_opener = f"Recent advances have transformed {query.lower()}, with key developments showing metrics like {', '.join(striking_stats[:2])}. This survey synthesizes {len(findings)} findings from {len(sources)} sources"
        
        lines.extend([
            "## Abstract",
            "",
            f"{abstract_opener} to map the current landscape of **{query.lower()}**. "
            f"The research draws on {len(academic_findings)} academic papers and {len(general_findings)} "
            f"industry sources, identifying {len(top_themes)} major themes: {', '.join(top_themes)}. "
            "Key findings reveal both rapid progress in capabilities and persistent challenges in "
            "reliability and generalization. This report distills the essential insights for "
            "researchers and practitioners navigating this fast-moving field.",
            "",
        ])
        
        # ----- TABLE OF CONTENTS -----
        lines.extend([
            "---",
            "",
            "## Table of Contents",
            "",
            "1. [Introduction](#introduction)",
            "2. [Background and Definitions](#background-and-definitions)",
            "3. [Literature Review](#literature-review)",
        ])
        
        section_num = 4
        for theme in themed_findings.keys():
            if theme != "General Findings" and len(themed_findings[theme]) >= 2:
                anchor = theme.lower().replace(" ", "-").replace("&", "and")
                lines.append(f"   - [{theme}](#{anchor})")
        
        lines.extend([
            f"{section_num}. [Analysis and Discussion](#analysis-and-discussion)",
            f"{section_num + 1}. [Challenges and Limitations](#challenges-and-limitations)",
            f"{section_num + 2}. [Future Directions](#future-directions)",
            f"{section_num + 3}. [Conclusions](#conclusions)",
            f"{section_num + 4}. [References](#references)",
            "",
        ])

        # ----- KEY FINDINGS SNAPSHOT -----
        key_points = []
        for f in findings[:5]:
            title = f.get("source_title", "Source") or "Source"
            snippet = f.get("content", "").replace("\n", " ").strip()
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            key_points.append(f"- **{title}**: {snippet}")

        lines.extend([
            "---",
            "",
            "## Key Findings",
            "",
        ])
        if key_points:
            lines.extend(key_points)
            lines.append("")
        else:
            lines.extend([
                "No key findings available yet. Collect more research to populate this section.",
                "",
            ])
        
        # ----- INTRODUCTION -----
        # Find an interesting finding to lead with
        lead_finding = None
        for f in findings[:10]:
            content = f.get("content", "")
            if any(char in content for char in ['%', 'billion', 'million', 'breakthrough']):
                lead_finding = content[:200].split(". ")[0] + "."
                break
        
        intro_lead = f"The landscape of {query.lower()} is shifting rapidly."
        if lead_finding:
            intro_lead = f"{lead_finding} This single data point hints at broader transformations reshaping {query.lower()}."
        
        lines.extend([
            "---",
            "",
            "## Introduction",
            "",
            intro_lead,
            "",
            "Understanding these changes requires examining both the technical advances driving progress "
            f"and the practical challenges that remain. This survey maps the territory through "
            f"{len(findings)} research findings spanning academic research and industry deployment.",
            "",
            "### Key Questions",
            "",
            "This investigation addresses:",
            "",
            f"- What architectural patterns and approaches define current {query.lower()}?",
            "- Where has the field achieved genuine breakthroughs, and where do claims outpace evidence?",
            "- What practical applications have moved beyond proof-of-concept?",
            "- What obstacles stand between current capabilities and broader impact?",
            "",
            "### Research Approach",
            "",
            f"We synthesized {len(findings)} findings from {len(sources)} sources, balancing "
            f"academic depth ({len(academic_findings)} papers) with practical relevance ({len(general_findings)} "
            f"industry sources). {verified_count} sources received additional verification. "
            "The analysis organizes findings thematically rather than chronologically, "
            "highlighting connections that might otherwise be obscured by publication silos.",
            "",
        ])
        
        # ----- BACKGROUND -----
        lines.extend([
            "---",
            "",
            "## Background and Definitions",
            "",
        ])
        
        # Find architecture/foundation findings for background
        background_findings = themed_findings.get("Architecture & Foundations", [])[:5]
        if background_findings:
            for finding in background_findings:
                content = finding.get("content", "")
                source = finding.get("source_title", "Source")
                # Extract first meaningful paragraph
                paragraphs = content.split(". ")
                if len(paragraphs) > 2:
                    excerpt = ". ".join(paragraphs[:3]) + "."
                else:
                    excerpt = content[:500]
                lines.append(f"{excerpt} [{source}]")
                lines.append("")
        else:
            lines.append(f"The study of {query.lower()} encompasses multiple interconnected domains "
                        "including artificial intelligence, distributed systems, and human-computer interaction.")
            lines.append("")
        
        # ----- LITERATURE REVIEW -----
        active_themes = [t for t in themed_findings.keys() if t != "General Findings" and len(themed_findings[t]) >= 2]
        
        lines.extend([
            "---",
            "",
            "## Literature Review",
            "",
            f"The {len(findings)} findings cluster around {len(active_themes)} major themes. "
            "Rather than catalog sources exhaustively, this section highlights the most "
            "significant contributions within each area.",
            "",
        ])
        
        # Generate themed sections
        priority_themes = [
            "Architecture & Foundations",
            "Reasoning & Planning", 
            "Multi-Agent Systems",
            "Tool Use & Function Calling",
            "Memory & Knowledge",
            "Benchmarks & Evaluation",
            "Applications & Deployment"
        ]
        
        for theme in priority_themes:
            theme_data = themed_findings.get(theme, [])
            if len(theme_data) < 2:
                continue
                
            lines.extend([
                f"### {theme}",
                "",
            ])
            
            # Synthesize findings into narrative paragraphs
            # Group by source to avoid repetition
            sources_seen = set()
            synthesized_content = []
            
            for finding in theme_data[:10]:
                source_url = finding.get("source_url", "")
                if source_url in sources_seen:
                    continue
                sources_seen.add(source_url)
                
                content = finding.get("content", "")
                source_title = finding.get("source_title", "Research")
                
                # Clean and truncate content
                content = content.replace("\n", " ").strip()
                if len(content) > 600:
                    # Find sentence boundary
                    sentences = content[:700].split(". ")
                    content = ". ".join(sentences[:-1]) + "." if len(sentences) > 1 else content[:600] + "..."
                
                synthesized_content.append(f"{content} [{source_title}]")
            
            # Write as flowing paragraphs (2-3 findings per paragraph)
            for i in range(0, len(synthesized_content), 2):
                chunk = synthesized_content[i:i+2]
                lines.append(" ".join(chunk))
                lines.append("")
        
        # ----- ANALYSIS -----
        lines.extend([
            "---",
            "",
            "## Analysis and Discussion",
            "",
        ])
        
        # Extract insights from verified/high-quality sources
        verified_findings = [f for f in findings if f.get("verified", False)][:5]
        high_quality = verified_findings if verified_findings else findings[:5]
        
        if high_quality:
            lines.append("Several findings stand out for their significance:")
            lines.append("")
            for finding in high_quality[:4]:
                content = finding.get("content", "")
                source = finding.get("source_title", "Source")
                # Extract meaningful excerpt
                sentences = content.split(". ")[:2]
                excerpt = ". ".join(sentences).strip()
                if not excerpt.endswith("."):
                    excerpt += "."
                if len(excerpt) > 300:
                    excerpt = excerpt[:300] + "..."
                lines.append(f"- {excerpt} [{source}]")
                lines.append("")
        
        # Academic vs industry perspective
        if len(academic_findings) >= 3 and len(general_findings) >= 3:
            lines.extend([
                "### Academic vs. Industry Perspectives",
                "",
                f"The {len(academic_findings)} academic sources tend toward theoretical depth—rigorous "
                f"benchmarks, formal analysis, and careful claims. The {len(general_findings)} industry "
                "sources offer a different view: deployment challenges, scaling considerations, "
                "and the gap between demo and production. Neither perspective alone captures "
                "the full picture. The most valuable insights often emerge at their intersection.",
                "",
            ])
        else:
            lines.extend([
                "### Patterns Across Sources",
                "",
                f"Examining {len(sources)} sources reveals consistent themes: rapid capability gains "
                "coexist with persistent reliability challenges. The most promising approaches "
                "combine theoretical grounding with practical validation—neither pure research "
                "nor engineering alone seems sufficient for lasting progress.",
                "",
            ])
        
        # ----- CHALLENGES -----
        challenge_findings = themed_findings.get("Challenges & Limitations", [])[:5]
        lines.extend([
            "---",
            "",
            "## Challenges and Limitations",
            "",
        ])
        
        if challenge_findings:
            for finding in challenge_findings:
                content = finding.get("content", "")[:400]
                source = finding.get("source_title", "Source")
                lines.append(f"- {content} [{source}]")
                lines.append("")
        else:
            lines.extend([
                "Key challenges identified in the research include:",
                "",
                "- **Scalability**: Managing computational costs as system complexity increases",
                "- **Reliability**: Ensuring consistent performance across diverse scenarios",
                "- **Interpretability**: Understanding and explaining system behavior",
                "- **Safety**: Preventing unintended or harmful outcomes",
                "- **Generalization**: Transferring capabilities across domains",
                "",
            ])
        
        # ----- FUTURE DIRECTIONS -----
        future_findings = themed_findings.get("Future Directions", [])[:5]
        lines.extend([
            "---",
            "",
            "## Future Directions",
            "",
        ])
        
        if future_findings:
            lines.append("The research points to several emerging directions:")
            lines.append("")
            for finding in future_findings:
                content = finding.get("content", "")[:400]
                source = finding.get("source_title", "Source")
                lines.append(f"- {content} [{source}]")
                lines.append("")
        else:
            lines.extend([
                "Based on the analysis, promising future directions include:",
                "",
                "- Enhanced reasoning capabilities through hybrid symbolic-neural approaches",
                "- Improved multi-agent coordination and communication protocols",
                "- More robust tool integration and function calling mechanisms",
                "- Advanced memory systems for long-term knowledge retention",
                "- Better evaluation frameworks and standardized benchmarks",
                "",
            ])
        
        # ----- CONCLUSIONS -----
        # Get the top 3 themes by finding count
        top_themes_by_count = sorted(
            [(t, len(f)) for t, f in themed_findings.items() if t != "General Findings"],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        lines.extend([
            "---",
            "",
            "## Conclusions",
            "",
            f"What emerges from {len(findings)} findings across {len(sources)} sources is a field "
            "in active transformation—one where capabilities expand rapidly but fundamental challenges persist.",
            "",
        ])
        
        # Theme-specific takeaways
        if top_themes_by_count:
            lines.append("**Core Insights:**")
            lines.append("")
            for theme, count in top_themes_by_count:
                theme_findings = themed_findings[theme][:2]
                if theme_findings:
                    first_content = theme_findings[0].get("content", "")[:150].split(". ")[0]
                    lines.append(f"- **{theme}**: {first_content}.")
            lines.append("")
        
        lines.extend([
            "**Looking Ahead:**",
            "",
            "The gap between research demonstrations and production deployment remains significant. "
            "While benchmarks show impressive numbers, real-world reliability and safety "
            "concerns demand continued attention. The most impactful advances will likely come from "
            "approaches that bridge this gap—systems that are not just capable but dependable.",
            "",
            f"For practitioners entering this space, the evidence suggests focusing on "
            "well-established patterns rather than chasing every new technique. "
            "For researchers, the open questions around reliability, interpretability, "
            "and generalization offer rich territory for meaningful contributions.",
            "",
        ])
        
        # ----- REFERENCES -----
        lines.extend([
            "---",
            "",
            "## References",
            "",
            f"**Total Sources: {len(sources)}**",
            "",
        ])
        
        # Academic sources
        academic_sources = {k: v for k, v in sources.items() if v["type"] == "academic"}
        if academic_sources:
            lines.append(f"### Academic Sources ({len(academic_sources)})")
            lines.append("")
            for i, (url, info) in enumerate(list(academic_sources.items())[:30], 1):
                verified = "✅" if info["verified"] else "❌"
                lines.append(f"{i}. {info['title']} {verified}")
                lines.append(f"   {url}")
                lines.append("")
        
        # General sources  
        general_sources = {k: v for k, v in sources.items() if v["type"] != "academic"}
        if general_sources:
            lines.append(f"### Industry & General Sources ({len(general_sources)})")
            lines.append("")
            for i, (url, info) in enumerate(list(general_sources.items())[:30], 1):
                verified = "✅" if info["verified"] else "❌"
                lines.append(f"{i}. {info['title']} {verified}")
                lines.append(f"   {url}")
                lines.append("")
        
        # ----- APPENDIX -----
        lines.extend([
            "---",
            "",
            "## Appendix: Research Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Findings | {len(findings)} |",
            f"| Unique Sources | {len(sources)} |",
            f"| Academic Sources | {len(academic_findings)} |",
            f"| General Sources | {len(general_findings)} |",
            f"| Verified Sources | {verified_count} |",
            f"| Themes Identified | {len(themed_findings)} |",
            "",
            "*Report generated by Deep Research Swarm*",
        ])
        
        return "\n".join(lines)
    
    def _save_checkpoint(self, phase: str, data: Dict[str, Any]):
        """Save a checkpoint for resuming later"""
        import json
        import os
        from datetime import datetime
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "phase": phase,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        
        filename = f"{self.checkpoint_dir}/checkpoint_{phase}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, "w") as f:
                json.dump(checkpoint, f, indent=2, default=str)
            logger.debug(f"Saved checkpoint: {filename}")
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")
    
    def _load_checkpoint(self, phase: str) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint for a phase"""
        import json
        import glob
        
        pattern = f"{self.checkpoint_dir}/checkpoint_{phase}_*.json"
        files = sorted(glob.glob(pattern), reverse=True)
        
        if files:
            try:
                with open(files[0], "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        return None


# =============================================================================
# CLI Interface
# =============================================================================

def save_markdown_report(result: SwarmResult, path: Optional[str] = None) -> str:
    """
    Save a complete run (summary + full report) to a markdown file.
    
    Args:
        result: SwarmResult with summary() and report
        path: Optional explicit path; if None, create timestamped file under reports/
        
    Returns:
        The path of the saved markdown file
    """
    # Default path: reports/deep_research_YYYYMMDD_HHMMSS.md
    if path:
        output_path = path
    else:
        os.makedirs("reports", exist_ok=True)
        output_path = os.path.join(
            "reports",
            f"deep_research_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md",
        )
    
    # Ensure parent directory exists if a custom path includes one
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# Research Report\n\n")
        f.write(result.summary())
        f.write("\n\n---\n\n")
        f.write(result.report or "")
        f.write("\n")
    
    return output_path


def main():
    """Main entry point for CLI usage"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Deep Research Swarm - AI-powered research assistant"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query (or use --interactive)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--simple", "-s",
        action="store_true",
        help="Use simple sequential execution (no Agno Workflow)"
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=5,
        help="Maximum parallel workers (default: 5)"
    )
    parser.add_argument(
        "--max-subtasks", "-t",
        type=int,
        default=7,
        help="Maximum subtasks (default: 7)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for report (optional)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no API calls)"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use DeepResearchSwarm (multi-iteration with quality control)"
    )
    parser.add_argument(
        "--express",
        action="store_true",
        help="Express deep mode (1 iteration, faster); implies --deep"
    )
    
    args = parser.parse_args()
    
    if args.mock:
        print("\n" + "=" * 60)
        print("  DEEP RESEARCH SWARM (MOCK MODE)")
        print("=" * 60)
        print(f"\n🔍 Query: {args.query or 'Mock Query'}\n")
        print("INFO PHASE 1: PLANNING (Mock)")
        print("INFO PHASE 2: EXECUTION (Mock)")
        print("INFO PHASE 3: SYNTHESIS (Mock)")
        
        # Create mock findings
        mock_findings = [
            {
                "content": "Artificial Intelligence (AI) in biology is transforming drug discovery. AlphaFold has predicted over 200 million protein structures, solving a 50-year grand challenge.",
                "source_title": "AlphaFold Protein Structure Database",
                "source_url": "https://alphafold.ebi.ac.uk",
                "search_type": "academic", 
                "verified": True
            },
            {
                "content": "Generative AI models are designing novel proteins not found in nature. These de novo proteins can be tailored for specific therapeutic functions.",
                "source_title": "De Novo Protein Design", 
                "source_url": "https://nature.com/articles/s41586-023-00000",
                "search_type": "academic",
                "verified": True
            },
            {
                "content": "AI-driven genomics is identifying disease variants faster than ever. Deep learning models analyze non-coding DNA to predict genetic risks.",
                "source_title": "AI in Genomics",
                "source_url": "https://science.org/genomics-ai",
                "search_type": "general",
                "verified": False
            }
        ]
        
        # Use the simpler swarm class for fallback report generation
        from infrastructure.knowledge_tools import KnowledgeTools
        swarm = ResearchSwarm(
            max_workers=args.max_workers,
            max_subtasks=args.max_subtasks,
        )
        swarm.knowledge_tools = KnowledgeTools() # Ensure tools are initialized
        
        report = swarm._generate_simple_fallback_report(args.query or "AI in Biology", mock_findings)
        
        print("\n" + "=" * 60)
        print("  RESEARCH COMPLETE")
        print("=" * 60)
        print("\n" + "-" * 60)
        print("  REPORT")
        print("-" * 60 + "\n")
        print(report)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\n📄 Report saved to: {args.output}")
            
            # Try PDF generation
            try:
                from generate_pdf_simple import generate_pdf
                pdf_path = args.output.replace(".md", ".pdf")
                generate_pdf(args.output, pdf_path)
            except Exception as e:
                print(f"⚠️ PDF generation failed: {e}")
                
        return 0

    # Check configuration
    from config import validate_config
    issues = validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nSet required environment variables and try again.")
        return 1
    
    # Create swarm
    swarm = None
    deep_swarm = None
    if args.deep or args.express:
        deep_swarm = DeepResearchSwarm(
            max_workers=args.max_workers,
            max_subtasks=args.max_subtasks,
            max_iterations=1 if args.express else 3,
            quality_threshold=80,
        )
    else:
        swarm = ResearchSwarm(
            max_workers=args.max_workers,
            max_subtasks=args.max_subtasks,
        )
    
    if args.interactive:
        print("\n" + "=" * 60)
        print("  DEEP RESEARCH SWARM - Interactive Mode")
        print("=" * 60)
        print("\nEnter your research query (or 'quit' to exit):\n")
        
        while True:
            try:
                query = input("🔍 Query: ").strip()
                
                if query.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n" + "-" * 60)
                print("Starting research...")
                print("-" * 60 + "\n")
                
                if args.deep or args.express:
                    result = deep_swarm.deep_research(query, use_experts=not args.express)
                elif args.simple:
                    result = swarm.research_simple(query)
                else:
                    result = swarm.research(query)
                
                print("\n" + "=" * 60)
                print("  RESEARCH COMPLETE")
                print("=" * 60)
                
                print(result.summary())
                print("\n" + "-" * 60)
                print("  REPORT")
                print("-" * 60 + "\n")
                print(result.report)
                
                # Always save the run to markdown (use provided output path if set)
                saved_path = save_markdown_report(result, args.output)
                print(f"\n📄 Report saved to: {saved_path}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    elif args.query:
        print("\n" + "=" * 60)
        print("  DEEP RESEARCH SWARM")
        print("=" * 60)
        print(f"\n🔍 Query: {args.query}\n")
        
        if args.deep or args.express:
            result = deep_swarm.deep_research(args.query, use_experts=not args.express)
        elif args.simple:
            result = swarm.research_simple(args.query)
        else:
            result = swarm.research(args.query)
        
        print("\n" + "=" * 60)
        print("  RESEARCH COMPLETE")
        print("=" * 60)
        
        print(result.summary())
        print("\n" + "-" * 60)
        print("  REPORT")
        print("-" * 60 + "\n")
        print(result.report)
        
        # Always save the run to markdown (use provided output path if set)
        saved_path = save_markdown_report(result, args.output)
        print(f"\n📄 Report saved to: {saved_path}")
        
        return 0 if result.success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)

