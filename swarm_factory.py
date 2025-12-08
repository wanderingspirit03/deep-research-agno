"""
Swarm Factory - Factory for creating and configuring research swarms

Provides convenient factory functions for creating pre-configured
research swarms with different characteristics:
- Quick research (fast, fewer subtasks)
- Deep research (thorough, more subtasks)
- Academic research (focused on scholarly sources)
- Technical research (code and documentation focus)
"""
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from config import config
from main import ResearchSwarm, SwarmResult, DeepResearchSwarm
from agents.planner import PlannerAgent
from agents.worker import WorkerAgent
from agents.editor import EditorAgent
from infrastructure.perplexity_tools import PerplexitySearchTools, ACADEMIC_DOMAINS
from infrastructure.knowledge_tools import KnowledgeTools


# =============================================================================
# Preset Configurations
# =============================================================================

@dataclass
class SwarmPreset:
    """Configuration preset for a research swarm"""
    name: str
    description: str
    max_workers: int
    max_subtasks: int
    planner_model: str
    worker_model: str
    editor_model: str
    search_max_results: int
    academic_focus: bool = False
    # Deep research settings
    max_iterations: int = 1
    quality_threshold: int = 70
    use_experts: bool = False


# Predefined presets
# Strategy: Claude Opus 4.5 for all agents (highest quality)
PRESETS: Dict[str, SwarmPreset] = {
    "quick": SwarmPreset(
        name="Quick Research",
        description="Fast research with minimal depth - good for simple queries",
        max_workers=3,
        max_subtasks=3,
        planner_model="openai/claude-opus-4-5-20251101",
        worker_model="openai/claude-opus-4-5-20251101",
        editor_model="openai/claude-opus-4-5-20251101",
        search_max_results=5,
    ),
    "balanced": SwarmPreset(
        name="Balanced Research",
        description="Balance between speed and depth - good for most queries",
        max_workers=5,
        max_subtasks=5,
        planner_model="openai/claude-opus-4-5-20251101",
        worker_model="openai/claude-opus-4-5-20251101",
        editor_model="openai/claude-opus-4-5-20251101",
        search_max_results=10,
    ),
    "deep": SwarmPreset(
        name="Deep Research",
        description="Thorough research with maximum depth - for complex topics",
        max_workers=7,
        max_subtasks=10,
        planner_model="openai/claude-opus-4-5-20251101",
        worker_model="openai/claude-opus-4-5-20251101",
        editor_model="openai/claude-opus-4-5-20251101",
        search_max_results=15,
    ),
    "academic": SwarmPreset(
        name="Academic Research",
        description="Focused on scholarly sources - for research papers",
        max_workers=5,
        max_subtasks=7,
        planner_model="openai/claude-opus-4-5-20251101",
        worker_model="openai/claude-opus-4-5-20251101",
        editor_model="openai/claude-opus-4-5-20251101",
        search_max_results=10,
        academic_focus=True,
    ),
    "technical": SwarmPreset(
        name="Technical Research",
        description="Code and documentation focus - for technical queries",
        max_workers=5,
        max_subtasks=5,
        planner_model="openai/claude-opus-4-5-20251101",
        worker_model="openai/claude-opus-4-5-20251101",
        editor_model="openai/claude-opus-4-5-20251101",
        search_max_results=10,
    ),
    "deep_research": SwarmPreset(
        name="PhD-Level Deep Research",
        description="Multi-iteration research with quality control - for comprehensive analysis (20-30 min)",
        max_workers=7,
        max_subtasks=15,
        planner_model="openai/claude-opus-4-5-20251101",
        worker_model="openai/claude-opus-4-5-20251101",
        editor_model="openai/claude-opus-4-5-20251101",
        search_max_results=15,
        academic_focus=True,
        max_iterations=3,
        quality_threshold=80,
        use_experts=True,
    ),
    "express_deep": SwarmPreset(
        name="Express Deep Research",
        description="Quick deep research with 1 iteration - for faster comprehensive results (5-10 min)",
        max_workers=5,
        max_subtasks=7,
        planner_model="openai/claude-opus-4-5-20251101",
        worker_model="openai/claude-opus-4-5-20251101",
        editor_model="openai/claude-opus-4-5-20251101",
        search_max_results=10,
        academic_focus=True,
        max_iterations=1,
        quality_threshold=70,
        use_experts=False,
    ),
}


# =============================================================================
# Factory Functions
# =============================================================================

def create_swarm(
    preset: str = "balanced",
    db_path: Optional[str] = None,
    **overrides,
) -> ResearchSwarm:
    """
    Create a research swarm with a preset configuration.
    
    Args:
        preset: Preset name ("quick", "balanced", "deep", "academic", "technical")
        db_path: Custom database path (optional)
        **overrides: Override specific preset values
        
    Returns:
        ResearchSwarm: Configured research swarm
        
    Example:
        >>> swarm = create_swarm("deep")
        >>> result = swarm.research("Quantum computing advances")
        
        >>> swarm = create_swarm("quick", max_workers=2)
        >>> result = swarm.research("Simple topic")
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    
    cfg = PRESETS[preset]
    
    # Apply overrides
    max_workers = overrides.get("max_workers", cfg.max_workers)
    max_subtasks = overrides.get("max_subtasks", cfg.max_subtasks)
    
    return ResearchSwarm(
        max_workers=max_workers,
        max_subtasks=max_subtasks,
        db_path=db_path,
    )


def quick_research(query: str, db_path: Optional[str] = None) -> SwarmResult:
    """
    Execute a quick research query.
    
    Args:
        query: Research query
        db_path: Custom database path (optional)
        
    Returns:
        SwarmResult: Research results
    """
    swarm = create_swarm("quick", db_path=db_path)
    return swarm.research_simple(query)


def deep_research(query: str, db_path: Optional[str] = None, express: bool = False) -> SwarmResult:
    """
    Execute a deep research query with multi-iteration quality control.
    
    This is the SOTA research mode for comprehensive, PhD-level investigations.
    
    Args:
        query: Research query
        db_path: Custom database path (optional)
        express: If True, use faster 1-iteration mode (5-10 min vs 20-30 min)
        
    Returns:
        SwarmResult: Research results with comprehensive report
    """
    preset_name = "express_deep" if express else "deep_research"
    preset = PRESETS[preset_name]
    
    swarm = DeepResearchSwarm(
        max_workers=preset.max_workers,
        max_subtasks=preset.max_subtasks,
        max_iterations=preset.max_iterations,
        quality_threshold=preset.quality_threshold,
        db_path=db_path,
    )
    
    return swarm.deep_research(query, use_experts=preset.use_experts)


def academic_research(query: str, db_path: Optional[str] = None) -> SwarmResult:
    """
    Execute an academic-focused research query.
    
    Args:
        query: Research query
        db_path: Custom database path (optional)
        
    Returns:
        SwarmResult: Research results
    """
    swarm = create_swarm("academic", db_path=db_path)
    return swarm.research(query)


# =============================================================================
# Advanced Factory
# =============================================================================

class SwarmBuilder:
    """
    Builder pattern for creating customized research swarms.
    
    Example:
        >>> swarm = (SwarmBuilder()
        ...     .with_workers(5)
        ...     .with_subtasks(7)
        ...     .with_planner_model("gpt-5-2025-08-07")
        ...     .with_editor_model("claude-opus-4-5-20251101")
        ...     .with_db_path("./custom_kb")
        ...     .build())
        >>> result = swarm.research("Complex query")
    """
    
    def __init__(self):
        self._max_workers = config.swarm.max_workers
        self._max_subtasks = config.swarm.max_subtasks
        self._db_path = config.knowledge.db_path
        self._planner_model = config.models.planner
        self._worker_model = config.models.worker
        self._editor_model = config.models.editor
        self._search_max_results = config.search.max_results
    
    def with_workers(self, count: int) -> "SwarmBuilder":
        """Set maximum number of parallel workers"""
        self._max_workers = count
        return self
    
    def with_subtasks(self, count: int) -> "SwarmBuilder":
        """Set maximum number of subtasks"""
        self._max_subtasks = count
        return self
    
    def with_db_path(self, path: str) -> "SwarmBuilder":
        """Set knowledge base database path"""
        self._db_path = path
        return self
    
    def with_planner_model(self, model: str) -> "SwarmBuilder":
        """Set planner LLM model"""
        self._planner_model = model
        return self
    
    def with_worker_model(self, model: str) -> "SwarmBuilder":
        """Set worker LLM model"""
        self._worker_model = model
        return self
    
    def with_editor_model(self, model: str) -> "SwarmBuilder":
        """Set editor LLM model"""
        self._editor_model = model
        return self
    
    def with_search_results(self, count: int) -> "SwarmBuilder":
        """Set maximum search results per query"""
        self._search_max_results = count
        return self
    
    def from_preset(self, preset: str) -> "SwarmBuilder":
        """Initialize from a preset configuration"""
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        
        cfg = PRESETS[preset]
        self._max_workers = cfg.max_workers
        self._max_subtasks = cfg.max_subtasks
        self._planner_model = cfg.planner_model
        self._worker_model = cfg.worker_model
        self._editor_model = cfg.editor_model
        self._search_max_results = cfg.search_max_results
        return self
    
    def build(self) -> ResearchSwarm:
        """Build the configured research swarm"""
        return ResearchSwarm(
            max_workers=self._max_workers,
            max_subtasks=self._max_subtasks,
            db_path=self._db_path,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def list_presets() -> Dict[str, Dict[str, Any]]:
    """
    List all available presets with their configurations.
    
    Returns:
        Dict mapping preset names to their configs
    """
    return {
        name: {
            "name": preset.name,
            "description": preset.description,
            "max_workers": preset.max_workers,
            "max_subtasks": preset.max_subtasks,
            "planner_model": preset.planner_model,
            "worker_model": preset.worker_model,
            "editor_model": preset.editor_model,
        }
        for name, preset in PRESETS.items()
    }


def print_presets():
    """Print all available presets in a formatted table"""
    print("\n" + "=" * 70)
    print("  RESEARCH SWARM PRESETS")
    print("=" * 70 + "\n")
    
    for name, preset in PRESETS.items():
        print(f"📋 {name.upper()}: {preset.name}")
        print(f"   {preset.description}")
        print(f"   Workers: {preset.max_workers} | Subtasks: {preset.max_subtasks}")
        print(f"   Planner: {preset.planner_model}")
        print(f"   Editor: {preset.editor_model}")
        print()


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print_presets()
    
    print("\n" + "=" * 70)
    print("  TESTING SWARM FACTORY")
    print("=" * 70 + "\n")
    
    # Test builder pattern
    print("Testing SwarmBuilder...")
    
    swarm = (SwarmBuilder()
        .from_preset("quick")
        .with_workers(2)
        .with_db_path("./test_factory_kb")
        .build())
    
    print(f"✅ Created swarm with {swarm.max_workers} workers")
    print(f"✅ Database path: {swarm.knowledge_tools.db_path}")
    
    print("\n✅ Swarm factory tests passed!")

