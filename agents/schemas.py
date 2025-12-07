"""
Enhanced Pydantic schemas for deep research system.

Provides structured models for:
- Deep subtasks with multiple queries and metadata
- Quality-scored findings with source metadata
- Critic evaluations with gap analysis
- Research iteration tracking
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class ResearchPhase(str, Enum):
    """Research phases for organized investigation"""
    FOUNDATION = "foundation"      # Background, definitions, history
    CURRENT = "current"            # Current state, recent developments
    CRITICAL = "critical"          # Critical analysis, comparisons
    FUTURE = "future"              # Future directions, predictions
    SYNTHESIS = "synthesis"        # Integration and conclusions


class SearchType(str, Enum):
    """Types of search to perform"""
    ACADEMIC = "academic"          # Papers, journals, research
    GENERAL = "general"            # Web, news, documentation
    TECHNICAL = "technical"        # Code, APIs, specifications
    NEWS = "news"                  # Recent news and announcements


class SourceAuthority(str, Enum):
    """Authority level of a source"""
    PEER_REVIEWED = "peer_reviewed"      # Academic journals
    INSTITUTIONAL = "institutional"       # Universities, research orgs
    INDUSTRY = "industry"                # Major tech companies
    EXPERT_BLOG = "expert_blog"          # Known experts
    NEWS_OUTLET = "news_outlet"          # Major news sources
    DOCUMENTATION = "documentation"       # Official docs
    COMMUNITY = "community"              # Forums, blogs
    UNKNOWN = "unknown"


# =============================================================================
# Deep Subtask Schema
# =============================================================================

class DeepSubtask(BaseModel):
    """Enhanced subtask with multiple queries and metadata for deep research"""
    id: int = Field(..., description="Unique subtask ID")
    phase: ResearchPhase = Field(
        default=ResearchPhase.CURRENT,
        description="Research phase this subtask belongs to"
    )
    focus: str = Field(..., description="What aspect this subtask investigates")
    primary_query: str = Field(..., description="Main search query")
    alternative_queries: List[str] = Field(
        default_factory=list,
        description="3-5 alternative query variations for comprehensive coverage"
    )
    search_types: List[SearchType] = Field(
        default_factory=lambda: [SearchType.GENERAL],
        description="Types of searches to perform"
    )
    expected_sources: int = Field(
        default=5,
        description="Expected number of quality sources"
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="IDs of subtasks this depends on"
    )
    priority: int = Field(
        default=1,
        description="Priority 1-3 (1=highest)"
    )
    
    # Quality expectations
    min_findings: int = Field(default=3, description="Minimum findings required")
    target_findings: int = Field(default=7, description="Target number of findings")


class DeepResearchPlan(BaseModel):
    """Comprehensive research plan for deep investigation"""
    original_query: str = Field(..., description="The original user query")
    summary: str = Field(..., description="Brief summary of the research approach")
    methodology: str = Field(
        default="",
        description="Research methodology and approach"
    )
    subtasks: List[DeepSubtask] = Field(..., description="List of deep subtasks")
    estimated_depth: str = Field(
        default="deep",
        description="Research depth: 'shallow', 'medium', 'deep', 'exhaustive'"
    )
    estimated_duration_minutes: int = Field(
        default=20,
        description="Estimated time to complete research"
    )
    key_questions: List[str] = Field(
        default_factory=list,
        description="Key questions the research aims to answer"
    )


# =============================================================================
# Quality Finding Schema
# =============================================================================

class QualityFinding(BaseModel):
    """Research finding with quality metadata"""
    id: Optional[str] = Field(default=None, description="Unique finding ID")
    content: str = Field(..., description="The key information/insight")
    source_url: str = Field(..., description="URL of the source")
    source_title: str = Field(default="", description="Title of the source")
    
    # Quality metadata
    quality_score: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Quality score 1-5 (5=highest)"
    )
    authority_type: SourceAuthority = Field(
        default=SourceAuthority.UNKNOWN,
        description="Authority level of the source"
    )
    relevance_score: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Relevance to research query 1-5"
    )
    recency_score: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Recency of information 1-5"
    )
    
    # Extracted metadata
    key_statistics: List[str] = Field(
        default_factory=list,
        description="Statistics and data points extracted"
    )
    named_entities: List[str] = Field(
        default_factory=list,
        description="People, organizations, products mentioned"
    )
    key_claims: List[str] = Field(
        default_factory=list,
        description="Main claims or findings"
    )
    
    # Context
    subtask_id: int = Field(default=0, description="ID of the subtask")
    worker_id: str = Field(default="", description="ID of the worker")
    search_type: str = Field(default="general", description="Type of search used")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the finding was saved"
    )
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        return (
            self.quality_score * 0.4 +
            self.relevance_score * 0.4 +
            self.recency_score * 0.2
        )


# =============================================================================
# Critic Evaluation Schema
# =============================================================================

class GapAnalysis(BaseModel):
    """Analysis of research gaps"""
    gap_description: str = Field(..., description="Description of the gap")
    importance: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Importance of filling this gap 1-5"
    )
    suggested_queries: List[str] = Field(
        default_factory=list,
        description="Queries to fill this gap"
    )
    expected_sources: List[str] = Field(
        default_factory=list,
        description="Types of sources that might help"
    )


class CriticEvaluation(BaseModel):
    """Critic's evaluation of research quality"""
    # Overall scores (0-100)
    overall_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Overall research quality score"
    )
    coverage_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="How well the topic is covered"
    )
    source_quality_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Quality of sources used"
    )
    evidence_strength_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Strength of evidence and citations"
    )
    balance_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Balance of perspectives"
    )
    
    # Analysis
    strengths: List[str] = Field(
        default_factory=list,
        description="Strengths of the research"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Weaknesses to address"
    )
    critical_gaps: List[GapAnalysis] = Field(
        default_factory=list,
        description="Critical gaps in research"
    )
    
    # Recommendations
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="Recommended follow-up queries"
    )
    suggested_improvements: List[str] = Field(
        default_factory=list,
        description="Suggested improvements for the report"
    )
    
    # Decision
    ready_for_synthesis: bool = Field(
        default=False,
        description="Whether research is ready for final synthesis"
    )
    recommendation: str = Field(
        default="",
        description="Overall recommendation (continue, synthesize, or refocus)"
    )


class DraftCritique(BaseModel):
    """Critique of a draft report"""
    overall_quality: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Overall quality score"
    )
    
    # Structure
    structure_score: int = Field(default=0, ge=0, le=100)
    clarity_score: int = Field(default=0, ge=0, le=100)
    completeness_score: int = Field(default=0, ge=0, le=100)
    citation_score: int = Field(default=0, ge=0, le=100)
    
    # Issues
    factual_issues: List[str] = Field(
        default_factory=list,
        description="Potential factual errors or unsupported claims"
    )
    structural_issues: List[str] = Field(
        default_factory=list,
        description="Issues with report structure"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="Missing sections or elements"
    )
    
    # Improvements
    specific_edits: List[str] = Field(
        default_factory=list,
        description="Specific edits to make"
    )
    ready_for_publication: bool = Field(default=False)


# =============================================================================
# Research Iteration Tracking
# =============================================================================

class ResearchIteration(BaseModel):
    """Track progress of a research iteration"""
    iteration: int = Field(..., description="Iteration number (1-based)")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Progress
    findings_count: int = Field(default=0)
    sources_count: int = Field(default=0)
    subtasks_completed: int = Field(default=0)
    subtasks_total: int = Field(default=0)
    
    # Quality
    quality_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Quality score from critic"
    )
    gaps_remaining: int = Field(default=0)
    
    # Decisions
    should_continue: bool = Field(default=True)
    focus_areas: List[str] = Field(
        default_factory=list,
        description="Areas to focus on in next iteration"
    )


class ResearchSession(BaseModel):
    """Complete research session tracking"""
    session_id: str = Field(..., description="Unique session ID")
    query: str = Field(..., description="Original research query")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Progress
    iterations: List[ResearchIteration] = Field(default_factory=list)
    current_iteration: int = Field(default=0)
    
    # Results
    total_findings: int = Field(default=0)
    total_sources: int = Field(default=0)
    final_quality_score: int = Field(default=0)
    
    # Output
    report_generated: bool = Field(default=False)
    report_length: int = Field(default=0)
    
    @property
    def duration_minutes(self) -> float:
        """Calculate session duration in minutes"""
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds() / 60


# =============================================================================
# Expert Analysis Schema
# =============================================================================

class ExpertPerspective(BaseModel):
    """Analysis from a domain expert perspective"""
    expert_type: str = Field(..., description="Type of expert (technical, industry, etc.)")
    perspective_summary: str = Field(..., description="Summary of expert's perspective")
    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights from this perspective"
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="Concerns or criticisms"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations from this perspective"
    )
    confidence_score: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Confidence in this analysis 1-5"
    )


# =============================================================================
# Checkpoint Schema
# =============================================================================

class ResearchCheckpoint(BaseModel):
    """Checkpoint for resuming long-running research"""
    checkpoint_id: str = Field(..., description="Unique checkpoint ID")
    session_id: str = Field(..., description="Research session ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # State
    phase: str = Field(..., description="Current phase (planning, research, synthesis)")
    iteration: int = Field(default=1)
    
    # Data
    plan: Optional[Dict[str, Any]] = Field(default=None, description="Serialized research plan")
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Collected findings so far"
    )
    worker_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Worker execution results"
    )
    evaluations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Critic evaluations"
    )
    
    # Metadata
    can_resume: bool = Field(default=True)
    resume_instructions: str = Field(default="")



