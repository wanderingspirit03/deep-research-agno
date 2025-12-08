"""
Agents Layer - Deep Research Swarm Agents

Provides specialized agents for the research swarm:
- Planner Agent: Decomposes research queries into subtasks
- Worker Agent: Executes searches and saves findings
- Editor Agent: Synthesizes findings into final report
- Critic Agent: Evaluates research quality and identifies gaps
- Domain Expert Agents: Multi-perspective analysis
- Data Analyst Agent: Structured data extraction and penetration metrics
- HITL Agent: Human-in-the-loop review via Slack
"""
from .planner import PlannerAgent, ResearchPlan, Subtask
from .worker import WorkerAgent, create_worker_agent
from .editor import EditorAgent
from .critic import CriticAgent
from .analyst import DataAnalystAgent
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
    AdoptionStatus,
    SearchStrategy,
    AgencyType,
    EvidenceSourceType,
    # Deep planning
    DeepSubtask,
    DeepResearchPlan,
    # Quality findings
    QualityFinding,
    # Agency adoption tracking
    AdoptionEvidence,
    AgencyAdoption,
    PenetrationMetrics,
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

# HITL Agent (optional - requires Slack)
try:
    from .hitl_agent import HitlAgent, HitlResult, Domain as HitlDomain
    _HITL_AVAILABLE = True
except ImportError:
    HitlAgent = None
    HitlResult = None
    HitlDomain = None
    _HITL_AVAILABLE = False

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
    "DataAnalystAgent",
    "DomainExpertAgent",
    "create_expert_agent",
    "create_expert_panel",
    "get_multi_perspective_analysis",
    "list_expert_types",
    "EXPERT_CONFIGS",
    # HITL Agent
    "HitlAgent",
    "HitlResult",
    "HitlDomain",
    # Enums
    "ResearchPhase",
    "SearchType",
    "SourceAuthority",
    "AdoptionStatus",
    "SearchStrategy",
    "AgencyType",
    "EvidenceSourceType",
    # Schemas
    "DeepSubtask",
    "DeepResearchPlan",
    "QualityFinding",
    "AdoptionEvidence",
    "AgencyAdoption",
    "PenetrationMetrics",
    "GapAnalysis",
    "CriticEvaluation",
    "DraftCritique",
    "ResearchIteration",
    "ResearchSession",
    "ExpertPerspective",
    "ResearchCheckpoint",
]

