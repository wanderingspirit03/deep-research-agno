"""
Configuration for Deep Research Swarm
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# HITL (Human-in-the-Loop) Configuration
# =============================================================================

@dataclass(frozen=True)
class HitlChannel:
    """Static smart-channel definition used for domain-based routing."""
    domain: str
    channel_id: Optional[str]
    description: str = ""


def _default_hitl_channels() -> List[HitlChannel]:
    """Smart defaults for Slack routing - domain-specific channels."""
    return [
        HitlChannel(
            domain="econ",
            channel_id=os.getenv("HITL_CHANNEL_ECON"),
            description="Economics, macro/micro, trade, labor, policy, and markets",
        ),
        HitlChannel(
            domain="fin_analytics",
            channel_id=os.getenv("HITL_CHANNEL_FIN"),
            description="Financial analytics, modeling, valuations, and risk",
        ),
        HitlChannel(
            domain="forecasting",
            channel_id=os.getenv("HITL_CHANNEL_FORECAST"),
            description="Forecasting, time-series, demand planning, and scenario planning",
        ),
        HitlChannel(
            domain="swe",
            channel_id=os.getenv("HITL_CHANNEL_SWE"),
            description="Software engineering, systems design, and debugging",
        ),
        HitlChannel(
            domain="general",
            channel_id=os.getenv("HITL_DEFAULT_CHANNEL"),
            description="General fallback review channel",
        ),
    ]


@dataclass
class HitlConfig:
    """Human-in-the-Loop (Slack) configuration."""
    enabled: bool = field(default_factory=lambda: os.getenv("HITL_ENABLED", "false").lower() == "true")
    slack_token: Optional[str] = field(default_factory=lambda: os.getenv("SLACK_TOKEN"))
    default_channel: str = field(default_factory=lambda: os.getenv("HITL_DEFAULT_CHANNEL", ""))
    smart_channels: List[HitlChannel] = field(default_factory=_default_hitl_channels)
    timeout_seconds: int = 300  # 5 minutes default
    poll_interval_seconds: float = 2.0
    confidence_threshold: float = 0.7  # Below this, escalate to HITL
    force_domains: List[str] = field(default_factory=list)  # Always use HITL for these domains
    model_id: str = "openai/gpt-5-mini-2025-08-07"  # Model for HITL agent
    temperature: float = 0.2


@dataclass
class ObservabilityConfig:
    """LMNR (Laminar) observability configuration."""
    lmnr_enabled: bool = field(default_factory=lambda: bool(os.getenv("LMNR_PROJECT_API_KEY")))
    lmnr_project_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LMNR_PROJECT_API_KEY"))


# =============================================================================
# Domain Filters
# =============================================================================

ACADEMIC_DOMAINS = [
    "arxiv.org",
    "nature.com",
    "ieee.org",
    "sciencedirect.com",
    "springer.com",
    "pubmed.ncbi.nlm.nih.gov",
    "acm.org",
    "wiley.com",
    "jstor.org",
    "scholar.google.com",
    "researchgate.net",
]

DENYLIST_DOMAINS = [
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "instagram.com",
    "linkedin.com",
]


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ModelConfig:
    """LLM model configuration"""
    # Available models on isara proxy (use openai/ prefix for LiteLLM routing):
    # - openai/gpt-5-2025-08-07 (powerful)
    # - openai/gpt-5-mini-2025-08-07 (fast)
    # - openai/gpt-5-nano-2025-08-07 (fastest)
    # - openai/gpt-5-codex (code)
    # - openai/claude-sonnet-4-5-20250929
    # - openai/claude-haiku-4-5-20251001 (fast)
    # - openai/claude-opus-4-1-20250805
    # - openai/claude-opus-4-5-20251101 (most powerful)
    
    # Strategy: Claude Opus 4.5 for all agents (highest quality)
    # Note: Use openai/ prefix to route through LiteLLM proxy with OpenAI-compatible API
    planner: str = "openai/claude-opus-4-5-20251101"  # Opus 4.5 for strategic planning
    worker: str = "openai/claude-opus-4-5-20251101"  # Opus 4.5 for research workers
    editor: str = "openai/claude-opus-4-5-20251101"  # Opus 4.5 for synthesis and writing
    critic: str = "openai/claude-opus-4-5-20251101"  # Opus 4.5 for quality evaluation
    embedding: str = "openai/text-embedding-3-large"
    
    # Temperature settings
    planner_temperature: float = 0.3
    worker_temperature: float = 0.5
    editor_temperature: float = 0.7
    critic_temperature: float = 0.2  # Lower for more consistent evaluation
    
    # Timeout settings (30 minutes = 1800 seconds for deep research)
    planner_timeout: int = 1800
    worker_timeout: int = 1800
    editor_timeout: int = 1800
    critic_timeout: int = 1800
    
    # Retry settings
    max_retries: int = 3
    retry_delay_base: float = 5.0


@dataclass
class SearchConfig:
    """Perplexity Search configuration"""
    max_results: int = 10
    batch_size: int = 5  # Max queries per batch
    country: str = "US"
    return_snippets: bool = True
    
    # Domain filtering
    academic_domains: List[str] = field(default_factory=lambda: ACADEMIC_DOMAINS)
    denylist_domains: List[str] = field(default_factory=lambda: DENYLIST_DOMAINS)
    
    # Rate limiting
    retry_attempts: int = 3
    retry_delay_base: float = 1.0  # Exponential backoff base


@dataclass
class DaytonaConfig:
    """Daytona sandbox configuration"""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("DAYTONA_API_KEY"))
    target: str = field(default_factory=lambda: os.getenv("DAYTONA_TARGET", "us"))
    auto_cleanup: bool = True
    timeout: int = 120  # seconds


@dataclass
class KnowledgeConfig:
    """LanceDB knowledge base configuration"""
    db_path: str = field(default_factory=lambda: os.getenv("LANCEDB_PATH", "./research_kb"))
    
    # OpenAI embedding model (use openai/ prefix for LiteLLM proxy routing)
    # "openai/text-embedding-3-large" (3072 dims, best quality)
    # "openai/text-embedding-3-small" (1536 dims, faster)
    embedding_model: str = "openai/text-embedding-3-large"
    embedding_dimensions: int = 3072
    
    top_k_default: int = 10


@dataclass
class SwarmConfig:
    """Swarm orchestration configuration"""
    max_subtasks: int = 10  # Increased for more comprehensive research (10×8=80 findings)
    max_workers: int = 5
    worker_timeout: int = 1800  # seconds (30 minutes for deep research)
    enable_parallel: bool = True


@dataclass
class DeepResearchConfig:
    """Deep research mode configuration for PhD-level research"""
    max_iterations: int = 3           # Research loop iterations
    quality_threshold: int = 80       # Min score (0-100) to stop iterating
    checkpoint_enabled: bool = True   # Enable checkpointing for long research
    checkpoint_dir: str = "./checkpoints"
    
    # Research depth settings (reduced to prevent context window overflow)
    min_sources_per_subtask: int = 3
    target_findings_per_subtask: int = 5
    max_findings_per_subtask: int = 8  # 7 subtasks * 8 = 56 max findings
    
    # Quality settings
    require_academic_sources: bool = True
    min_academic_ratio: float = 0.3  # At least 30% academic sources
    
    # Report settings
    min_report_words: int = 3000
    target_report_words: int = 7000
    include_methodology: bool = True
    include_limitations: bool = True


@dataclass
class Config:
    """Main configuration container"""
    models: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    daytona: DaytonaConfig = field(default_factory=DaytonaConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    deep_research: DeepResearchConfig = field(default_factory=DeepResearchConfig)
    hitl: HitlConfig = field(default_factory=HitlConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    # API Keys (loaded from environment)
    perplexity_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("PERPLEXITY_API_KEY")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    
    # LiteLLM Proxy (isara platform)
    litellm_api_base: Optional[str] = field(
        default_factory=lambda: os.getenv("LITELLM_API_BASE")
    )
    litellm_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("LITELLM_API_KEY")
    )


# Global config instance
config = Config()


# =============================================================================
# Validation
# =============================================================================

def validate_config() -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    if not config.perplexity_api_key:
        issues.append("PERPLEXITY_API_KEY not set")
    
    # Either direct LLM keys or LiteLLM proxy
    has_llm = (
        config.openai_api_key or 
        config.anthropic_api_key or 
        (config.litellm_api_base and config.litellm_api_key)
    )
    if not has_llm:
        issues.append("LLM access required: set OPENAI_API_KEY, ANTHROPIC_API_KEY, or LITELLM_API_BASE+LITELLM_API_KEY")
    
    return issues


def validate_hitl_config() -> List[str]:
    """Validate HITL configuration and return list of issues (optional feature)"""
    issues = []
    
    if config.hitl.enabled:
        if not config.hitl.slack_token:
            issues.append("HITL enabled but SLACK_TOKEN not set")
        if not config.hitl.default_channel:
            issues.append("HITL enabled but HITL_DEFAULT_CHANNEL not set")
    
    return issues


def validate_observability_config() -> List[str]:
    """Validate observability configuration and return list of issues (optional feature)"""
    issues = []
    
    # Observability is auto-enabled if key is present, so no validation needed
    # Just informational
    
    return issues


if __name__ == "__main__":
    # Quick config check
    print("=== Deep Research Swarm Configuration ===\n")
    
    issues = validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Core configuration valid!")
    
    # Check HITL
    hitl_issues = validate_hitl_config()
    if config.hitl.enabled:
        if hitl_issues:
            print("\n⚠️  HITL Issues:")
            for issue in hitl_issues:
                print(f"   - {issue}")
        else:
            print("✅ HITL configuration valid!")
    else:
        print("ℹ️  HITL disabled (set HITL_ENABLED=true to enable)")
    
    # Check Observability
    if config.observability.lmnr_enabled:
        print("✅ LMNR observability enabled")
    else:
        print("ℹ️  LMNR observability disabled (set LMNR_PROJECT_API_KEY to enable)")
    
    print(f"\n📊 Settings:")
    print(f"   Planner Model: {config.models.planner}")
    print(f"   Worker Model: {config.models.worker}")
    print(f"   Editor Model: {config.models.editor}")
    print(f"   Critic Model: {config.models.critic}")
    print(f"   Max Subtasks: {config.swarm.max_subtasks}")
    print(f"   Max Workers: {config.swarm.max_workers}")
    print(f"   Search Max Results: {config.search.max_results}")
    print(f"\n🔄 HITL Settings:")
    print(f"   Enabled: {config.hitl.enabled}")
    print(f"   Confidence Threshold: {config.hitl.confidence_threshold}")
    print(f"   Timeout: {config.hitl.timeout_seconds}s")
    print(f"\n📈 Observability:")
    print(f"   LMNR Enabled: {config.observability.lmnr_enabled}")

