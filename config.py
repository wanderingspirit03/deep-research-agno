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
    # Available models on isara proxy:
    # - gpt-5-2025-08-07 (powerful)
    # - gpt-5-mini-2025-08-07 (fast)
    # - gpt-5-nano-2025-08-07 (fastest)
    # - gpt-5-codex (code)
    # - claude-sonnet-4-5-20250929
    # - claude-haiku-4-5-20251001 (fast)
    # - claude-opus-4-1-20250805
    # - claude-opus-4-5-20251101 (most powerful)
    
    # Strategy: GPT-5 full for quality (planner/editor), Mini for speed (workers)
    planner: str = "gpt-5-2025-08-07"  # Full GPT-5 for strategic planning
    worker: str = "gpt-5-mini-2025-08-07"  # Fast for parallel search workers
    editor: str = "gpt-5-2025-08-07"  # Full GPT-5 for synthesis and writing
    critic: str = "gpt-5-mini-2025-08-07"  # Fast for quality evaluation
    embedding: str = "text-embedding-3-small"
    
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
    
    # OpenAI embedding model
    # "text-embedding-3-large" (3072 dims, best quality)
    # "text-embedding-3-small" (1536 dims, faster)
    embedding_model: str = "text-embedding-3-large"
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


if __name__ == "__main__":
    # Quick config check
    print("=== Deep Research Swarm Configuration ===\n")
    
    issues = validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Configuration valid!")
    
    print(f"\n📊 Settings:")
    print(f"   Planner Model: {config.models.planner}")
    print(f"   Worker Model: {config.models.worker}")
    print(f"   Editor Model: {config.models.editor}")
    print(f"   Max Subtasks: {config.swarm.max_subtasks}")
    print(f"   Max Workers: {config.swarm.max_workers}")
    print(f"   Search Max Results: {config.search.max_results}")

