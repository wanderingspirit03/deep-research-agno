"""
Agents Layer - Deep Research Swarm Agents

Provides specialized agents for the research swarm:
- Planner Agent: Decomposes research queries into subtasks
- Worker Agent: Executes searches and saves findings
- Editor Agent: Synthesizes findings into final report
- Critic Agent: Evaluates research quality and identifies gaps
- Domain Expert Agents: Multi-perspective analysis
"""
from .planner import PlannerAgent, ResearchPlan, Subtask
from .worker import WorkerAgent, create_worker_agent
from .editor import EditorAgent
from .critic import CriticAgent
from .domain_experts import (
    DomainExpertAgent,
    create_expert_agent,
    create_expert_panel,
    get_multi_perspective_analysis,
    list_expert_types,
    EXPERT_CONFIGS,
)
from .schemas import (
    # Enums
    ResearchPhase,
    SearchType,
    SourceAuthority,
    # Deep planning
    DeepSubtask,
    DeepResearchPlan,
    # Quality findings
    QualityFinding,
    # Critic schemas
    GapAnalysis,
    CriticEvaluation,
    DraftCritique,
    # Iteration tracking
    ResearchIteration,
    ResearchSession,
    # Expert analysis
    ExpertPerspective,
    # Checkpointing
    ResearchCheckpoint,
)

__all__ = [
    # Core agents
    "PlannerAgent",
    "ResearchPlan",
    "Subtask",
    "WorkerAgent",
    "create_worker_agent",
    "EditorAgent",
    # New agents
    "CriticAgent",
    "DomainExpertAgent",
    "create_expert_agent",
    "create_expert_panel",
    "get_multi_perspective_analysis",
    "list_expert_types",
    "EXPERT_CONFIGS",
    # Schemas
    "ResearchPhase",
    "SearchType",
    "SourceAuthority",
    "DeepSubtask",
    "DeepResearchPlan",
    "QualityFinding",
    "GapAnalysis",
    "CriticEvaluation",
    "DraftCritique",
    "ResearchIteration",
    "ResearchSession",
    "ExpertPerspective",
    "ResearchCheckpoint",
]

