"""
Enhanced Pydantic schemas for deep research system.

Provides structured models for:
- Deep subtasks with multiple queries and metadata
- Quality-scored findings with source metadata
- Critic evaluations with gap analysis
- Research iteration tracking
- Agency adoption tracking (for market penetration research)
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


class AdoptionStatus(str, Enum):
    """Adoption status labels for agency/entity tracking"""
    CONFIRMED = "confirmed"              # Clear, unambiguous evidence of active use
    PROBABLE = "probable"                # Strong evidence but not explicitly confirmed
    PILOT = "pilot"                      # Piloting or trialing (limited deployment)
    NOT_ADOPTED = "not_adopted"          # Searched and no evidence found
    NO_DATA = "no_data"                  # Insufficient public records to determine


class SearchStrategy(str, Enum):
    """Search strategy types for targeted research"""
    GENERAL_WEB = "general_web"                    # Standard web search
    ACADEMIC = "academic"                          # Academic/scholarly sources
    DORK_POLICE_REPORTS = "dork_police_reports"    # Google dork for police reports with disclaimers
    DORK_PROCUREMENT = "dork_procurement"          # Google dork for procurement portals
    DORK_COUNCIL_MINUTES = "dork_council_minutes"  # Google dork for council/board minutes
    DORK_NEWS = "dork_news"                        # Google dork for news coverage
    DORK_GOVERNMENT = "dork_government"            # Google dork for .gov sites
    PROCUREMENT_PORTAL = "procurement_portal"      # Direct procurement portal search
    OPEN_DATA_PORTAL = "open_data_portal"          # Public records/open data portals
    PRESS_RELEASE = "press_release"                # Official press releases


class AgencyType(str, Enum):
    """Types of law enforcement agencies"""
    MUNICIPAL_POLICE = "Municipal Police"
    COUNTY_SHERIFF = "County Sheriff"
    STATE_POLICE = "State Police"
    HIGHWAY_PATROL = "Highway Patrol"
    FEDERAL = "Federal"
    OTHER = "Other"


class EvidenceSourceType(str, Enum):
    """Types of evidence sources for adoption detection"""
    POLICE_REPORT = "police_report"              # Incident/arrest reports with disclaimers
    PROCUREMENT_CONTRACT = "procurement_contract"  # RFPs, contracts, purchase orders
    COUNCIL_MINUTES = "council_minutes"           # Council/board meeting minutes
    BUDGET_DOCUMENT = "budget_document"           # Budget presentations, fiscal documents
    PRESS_RELEASE = "press_release"               # Official announcements
    NEWS_ARTICLE = "news_article"                 # Local/national news coverage
    POLICY_MEMO = "policy_memo"                   # Internal policies made public
    AXON_ANNOUNCEMENT = "axon_announcement"       # Axon's own press releases
    OTHER = "other"


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
# Agency Adoption Schema (for market penetration research)
# =============================================================================

class AdoptionEvidence(BaseModel):
    """Evidence supporting agency adoption status"""
    evidence_type: EvidenceSourceType = Field(
        default=EvidenceSourceType.OTHER,
        description="Type of evidence source"
    )
    url: str = Field(default="", description="URL of evidence source")
    title: str = Field(default="", description="Title/description of evidence")
    excerpt: str = Field(default="", description="Relevant excerpt from source")
    date_found: Optional[str] = Field(default=None, description="Date the evidence was found/published")
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this evidence (0-1)"
    )
    
    # Specific fields for police reports
    has_ai_disclaimer: bool = Field(default=False, description="Contains AI/DraftOne disclaimer")
    disclaimer_text: Optional[str] = Field(default=None, description="Exact disclaimer text if found")


class AgencyAdoption(BaseModel):
    """
    Structured record for tracking agency adoption of a product/service.
    
    Designed for tracking Axon AI Era Plan / DraftOne adoption but can be
    generalized to other market penetration research.
    """
    # Agency identification
    agency_name: str = Field(..., description="Full name of the agency")
    state: str = Field(..., description="US state code (e.g., 'CA', 'TX')")
    agency_type: AgencyType = Field(
        default=AgencyType.MUNICIPAL_POLICE,
        description="Type of law enforcement agency"
    )
    
    # Demographics
    officer_count: Optional[int] = Field(
        default=None,
        description="Number of sworn officers (active duty)"
    )
    population_served: Optional[int] = Field(
        default=None,
        description="Population served by the agency"
    )
    
    # Adoption status
    status: AdoptionStatus = Field(
        default=AdoptionStatus.NO_DATA,
        description="Current adoption status"
    )
    status_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the status determination (0-1)"
    )
    
    # Evidence collection
    evidence: List[AdoptionEvidence] = Field(
        default_factory=list,
        description="Supporting evidence for the status"
    )
    evidence_count: int = Field(default=0, description="Number of evidence sources found")
    
    # Timeline tracking
    first_adoption_date: Optional[str] = Field(
        default=None,
        description="Earliest date of adoption evidence (ISO format)"
    )
    last_verified_date: Optional[str] = Field(
        default=None,
        description="Most recent verification date (ISO format)"
    )
    
    # Research metadata
    search_completed: bool = Field(default=False, description="Whether comprehensive search was performed")
    data_access_notes: Optional[str] = Field(
        default=None,
        description="Notes about data availability (e.g., 'no open data portal')"
    )
    
    def add_evidence(self, evidence: AdoptionEvidence):
        """Add evidence and update count"""
        self.evidence.append(evidence)
        self.evidence_count = len(self.evidence)
    
    @property
    def has_confirmed_adoption(self) -> bool:
        """Check if there's confirmed adoption evidence"""
        return self.status == AdoptionStatus.CONFIRMED
    
    @property
    def best_evidence_url(self) -> Optional[str]:
        """Get the highest-confidence evidence URL"""
        if not self.evidence:
            return None
        sorted_evidence = sorted(self.evidence, key=lambda e: e.confidence_score, reverse=True)
        return sorted_evidence[0].url if sorted_evidence else None


class PenetrationMetrics(BaseModel):
    """Aggregated penetration metrics for market research"""
    total_agencies: int = Field(default=0, description="Total agencies in universe")
    confirmed_count: int = Field(default=0, description="Agencies with confirmed adoption")
    probable_count: int = Field(default=0, description="Agencies with probable adoption")
    pilot_count: int = Field(default=0, description="Agencies in pilot/trial")
    not_adopted_count: int = Field(default=0, description="Agencies confirmed not adopting")
    no_data_count: int = Field(default=0, description="Agencies with insufficient data")
    
    # Officer-weighted metrics
    total_officers: int = Field(default=0, description="Total officers across all agencies")
    officers_confirmed: int = Field(default=0, description="Officers in confirmed agencies")
    officers_probable: int = Field(default=0, description="Officers in probable agencies")
    
    # Penetration percentages
    @property
    def penetration_confirmed(self) -> float:
        """Percentage of agencies with confirmed adoption"""
        return (self.confirmed_count / self.total_agencies * 100) if self.total_agencies > 0 else 0.0
    
    @property
    def penetration_confirmed_plus_probable(self) -> float:
        """Percentage including both confirmed and probable"""
        return ((self.confirmed_count + self.probable_count) / self.total_agencies * 100) if self.total_agencies > 0 else 0.0
    
    @property
    def officer_penetration_confirmed(self) -> float:
        """Percentage of officers in confirmed agencies"""
        return (self.officers_confirmed / self.total_officers * 100) if self.total_officers > 0 else 0.0
    
    @property
    def coverage_rate(self) -> float:
        """Percentage of agencies actually researched (have data)"""
        researched = self.total_agencies - self.no_data_count
        return (researched / self.total_agencies * 100) if self.total_agencies > 0 else 0.0
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Generate summary dictionary for reporting"""
        return {
            "total_agencies": self.total_agencies,
            "confirmed_adoptions": self.confirmed_count,
            "probable_adoptions": self.probable_count,
            "pilot_programs": self.pilot_count,
            "not_adopted": self.not_adopted_count,
            "no_data": self.no_data_count,
            "penetration_confirmed_pct": round(self.penetration_confirmed, 2),
            "penetration_confirmed_plus_probable_pct": round(self.penetration_confirmed_plus_probable, 2),
            "officer_penetration_pct": round(self.officer_penetration_confirmed, 2),
            "research_coverage_pct": round(self.coverage_rate, 2),
        }


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
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.GENERAL_WEB,
        description="Primary search strategy for this subtask"
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
    
    # Geographic/entity targeting (for market research)
    target_state: Optional[str] = Field(default=None, description="Target US state for geographic decomposition")
    target_agencies: List[str] = Field(default_factory=list, description="Specific agencies to research")
    
    # Dork-specific fields
    dork_pattern: Optional[str] = Field(
        default=None,
        description="Google dork pattern to use (e.g., 'site:.gov \"Axon DraftOne\"')"
    )


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



