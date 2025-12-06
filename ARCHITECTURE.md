# Deep Research Swarm - System Architecture

> **A PhD-Level Multi-Agent Research System**

This document provides a comprehensive overview of the Deep Research Swarm architecture—a sophisticated multi-agent system designed for conducting thorough, academically-rigorous research using AI agents orchestrated with the Agno framework.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Agent Layer](#agent-layer)
5. [Infrastructure Layer](#infrastructure-layer)
6. [Data Flow](#data-flow)
7. [Execution Modes](#execution-modes)
8. [Configuration System](#configuration-system)
9. [Quality Control Pipeline](#quality-control-pipeline)
10. [Knowledge Management](#knowledge-management)

---

## System Overview

The Deep Research Swarm is a **multi-agent research orchestration system** that breaks down complex research queries into manageable subtasks, executes them in parallel using specialized AI agents, and synthesizes findings into comprehensive academic-quality reports.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Query Decomposition** | Breaks complex queries into 7-15 focused subtasks |
| **Parallel Execution** | Up to 7 worker agents operating simultaneously |
| **Quality Control** | Critic agent evaluates research coverage (0-100 score) |
| **Iterative Refinement** | Multiple research iterations with gap analysis |
| **Multi-Perspective Analysis** | Domain experts provide diverse viewpoints |
| **Knowledge Persistence** | Vector-based storage with semantic search |
| **Academic Synthesis** | 5,000-10,000 word reports with proper citations |

### Technology Stack

- **Agent Framework**: [Agno](https://github.com/agno-ai/agno) (Workflow 2.0 with Parallel execution)
- **LLM Backend**: LiteLLM proxy (Claude Opus 4.5, Claude Haiku 4.5, GPT-5)
- **Search API**: Perplexity Search API
- **Vector Database**: LanceDB
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEEP RESEARCH SWARM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   USER      │────▶│              ORCHESTRATION LAYER                │   │
│  │   QUERY     │     │  ┌─────────────────────────────────────────┐    │   │
│  └─────────────┘     │  │         ResearchSwarm / DeepResearchSwarm│    │   │
│                      │  │  • Workflow management                   │    │   │
│                      │  │  • Phase coordination                    │    │   │
│                      │  │  • Checkpoint handling                   │    │   │
│                      │  └─────────────────────────────────────────┘    │   │
│                      └─────────────────────────────────────────────────┘   │
│                                          │                                  │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         AGENT LAYER                                   │  │
│  │                                                                       │  │
│  │  ┌──────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐  │  │
│  │  │ PLANNER  │──▶│     WORKERS      │──▶│  CRITIC  │──▶│  EDITOR  │  │  │
│  │  │  AGENT   │   │  (Parallel x7)   │   │  AGENT   │   │  AGENT   │  │  │
│  │  └──────────┘   └──────────────────┘   └──────────┘   └──────────┘  │  │
│  │       │                  │                   │              │        │  │
│  │       │                  │                   │              │        │  │
│  │       │         ┌────────┴────────┐          │              │        │  │
│  │       │         │                 │          │              │        │  │
│  │       │    ┌────▼────┐     ┌─────▼─────┐    │              │        │  │
│  │       │    │ DOMAIN  │     │  DOMAIN   │    │              │        │  │
│  │       │    │ EXPERT  │     │  EXPERT   │    │              │        │  │
│  │       │    │(Tech)   │     │(Industry) │    │              │        │  │
│  │       │    └─────────┘     └───────────┘    │              │        │  │
│  │       │                                      │              │        │  │
│  └───────┼──────────────────────────────────────┼──────────────┼────────┘  │
│          │                                      │              │           │
│          ▼                                      ▼              ▼           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      INFRASTRUCTURE LAYER                             │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │  │
│  │  │   PERPLEXITY    │  │   LANCEDB       │  │   LITELLM       │       │  │
│  │  │   SEARCH TOOLS  │  │   KNOWLEDGE KB  │  │   PROXY         │       │  │
│  │  │                 │  │                 │  │                 │       │  │
│  │  │ • search()      │  │ • save_finding()│  │ • Claude Opus   │       │  │
│  │  │ • batch_search()│  │ • search_know() │  │ • Claude Haiku  │       │  │
│  │  │ • academic()    │  │ • list_sources()│  │ • GPT-5 Mini    │       │  │
│  │  │ • general()     │  │ • get_finding() │  │                 │       │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│                                          │                                  │
│                                          ▼                                  │
│                              ┌─────────────────────┐                       │
│                              │   RESEARCH REPORT   │                       │
│                              │   (5,000-10,000     │                       │
│                              │    words)           │                       │
│                              └─────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Entry Points

| File | Purpose |
|------|---------|
| `main.py` | Main orchestration module with `ResearchSwarm` and `DeepResearchSwarm` classes |
| `swarm_factory.py` | Factory functions for creating pre-configured research swarms |
| `run_express_research.py` | Quick-start script for express research mode |

### 2. Swarm Types

#### ResearchSwarm (Basic)
Single-iteration research with parallel worker execution:
- Phase 1: Planning
- Phase 2: Parallel Worker Execution
- Phase 3: Report Synthesis

#### DeepResearchSwarm (Advanced)
Multi-iteration research with quality control:
- Phase 1: Strategic Planning
- Phase 2: Iterative Research Loop (up to 3 iterations)
- Phase 3: Multi-Perspective Analysis (Domain Experts)
- Phase 4: Academic Report Synthesis

---

## Agent Layer

### Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT HIERARCHY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌───────────────┐                          │
│                      │    PLANNER    │  ◄── Strategic Reasoning │
│                      │  Claude Opus  │      (Temperature: 0.3)  │
│                      └───────┬───────┘                          │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                  │
│              ▼               ▼               ▼                  │
│       ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│       │ WORKER 1 │    │ WORKER 2 │    │ WORKER N │  (up to 7)  │
│       │  Haiku   │    │  Haiku   │    │  Haiku   │             │
│       └────┬─────┘    └────┬─────┘    └────┬─────┘             │
│            │               │               │                    │
│            └───────────────┼───────────────┘                    │
│                            ▼                                    │
│                      ┌───────────────┐                          │
│                      │    CRITIC     │  ◄── Quality Evaluation  │
│                      │  Claude Opus  │      (Temperature: 0.2)  │
│                      └───────┬───────┘                          │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│       ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│       │TECHNICAL │    │ INDUSTRY │    │ SKEPTIC  │  (Optional) │
│       │ EXPERT   │    │  EXPERT  │    │ EXPERT   │             │
│       └──────────┘    └──────────┘    └──────────┘             │
│                              │                                  │
│                              ▼                                  │
│                      ┌───────────────┐                          │
│                      │    EDITOR     │  ◄── Report Synthesis    │
│                      │  Claude Opus  │      (Temperature: 0.7)  │
│                      └───────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Individual Agent Specifications

#### 🎯 Planner Agent (`agents/planner.py`)

**Purpose**: Decomposes complex research queries into structured subtasks.

**Model**: Claude Opus 4.5 (strategic reasoning)

**Capabilities**:
- Analyzes user research queries
- Creates 7-15 focused subtasks organized by research phase
- Assigns search strategies (academic vs. general)
- Prioritizes subtasks (1=critical, 2=important, 3=supplementary)

**Output Schema**:
```python
class ResearchPlan:
    original_query: str
    summary: str
    subtasks: List[Subtask]  # Each with id, query, focus, search_type, priority
    estimated_depth: str     # "shallow" | "medium" | "deep"
```

**Four-Phase Research Structure**:
1. **Foundation** (Priority 1): Background, definitions, history
2. **Current State** (Priority 1): State-of-the-art, recent developments
3. **Critical Analysis** (Priority 2): Limitations, competing approaches
4. **Future Directions** (Priority 2-3): Trends, predictions, implications

---

#### ⚡ Worker Agent (`agents/worker.py`)

**Purpose**: Executes individual research subtasks by searching and saving findings.

**Model**: Claude Haiku 4.5 (fast parallel execution)

**Capabilities**:
- Executes Perplexity searches (academic/general)
- Evaluates and filters search results
- Extracts comprehensive content from sources
- Saves findings to knowledge base with metadata

**Workflow**:
1. **Phase 1 - Search**: Execute primary query + alternative variations
2. **Phase 2 - Extract**: Deep URL extraction for full content
3. **Phase 3 - Save**: Store 5-12 comprehensive findings (1000-3000 chars each)

**Quality Scoring** (before saving):
- 5: Peer-reviewed paper, official documentation
- 4: Major tech company blog, institutional report
- 3: Reputable news outlet, expert blog
- 2: Forum posts, opinion pieces
- 1: Unknown sources, marketing material

---

#### 🔬 Critic Agent (`agents/critic.py`)

**Purpose**: Evaluates research quality and identifies gaps.

**Model**: Claude Opus 4.5 (rigorous analysis)

**Evaluation Criteria** (0-100 scores):
| Criterion | Description |
|-----------|-------------|
| **Coverage** | Does research address all key aspects? |
| **Source Quality** | Are sources authoritative and recent? |
| **Evidence Strength** | Are claims supported by data? |
| **Balance** | Are multiple perspectives represented? |

**Output Schema**:
```python
class CriticEvaluation:
    overall_score: int          # 0-100
    coverage_score: int
    source_quality_score: int
    evidence_strength_score: int
    balance_score: int
    critical_gaps: List[GapAnalysis]
    follow_up_queries: List[str]
    ready_for_synthesis: bool   # True if score >= threshold
    recommendation: str         # "synthesize" | "continue" | "refocus"
```

---

#### 📝 Editor Agent (`agents/editor.py`)

**Purpose**: Synthesizes findings into comprehensive academic reports.

**Model**: Claude Opus 4.5 (high-quality writing)

**Report Structure** (5,000-10,000 words):
1. **Abstract** (200-300 words)
2. **Introduction** (500-800 words)
3. **Background & Definitions** (400-600 words)
4. **Literature Review** (2000-3500 words) - Multiple themed subsections
5. **Critical Analysis** (800-1200 words)
6. **Future Directions** (500-800 words)
7. **Conclusions** (400-600 words)
8. **References** - Complete source list

**Workflow**:
1. Review findings index
2. Plan report sections based on themes
3. For each section: `search_knowledge()` → extract facts → write with citations
4. Call `list_sources()` for reference list

---

#### 🎓 Domain Expert Agents (`agents/domain_experts.py`)

**Purpose**: Provide specialized perspectives for multi-faceted analysis.

**Expert Types**:

| Expert | Role | Focus Areas |
|--------|------|-------------|
| **Technical** | Senior ML Researcher | Algorithm design, benchmarks, limitations |
| **Industry** | VP of Engineering | Production readiness, costs, integration |
| **Skeptic** | Critical Researcher | Reproducibility, overhype, hidden assumptions |
| **Futurist** | Trend Analyst | Long-term trajectory, societal implications |
| **Academic** | University Professor | Literature positioning, methodology |

**Output Schema**:
```python
class ExpertPerspective:
    expert_type: str
    perspective_summary: str
    key_insights: List[str]
    concerns: List[str]
    recommendations: List[str]
    confidence_score: int  # 1-5
```

---

## Infrastructure Layer

### Perplexity Search Tools (`infrastructure/perplexity_tools.py`)

**Purpose**: Web search capabilities with domain filtering.

**Tools Exposed**:
| Tool | Description |
|------|-------------|
| `search(query)` | Basic web search |
| `batch_search(queries)` | Multiple queries (max 5) |
| `search_academic(query)` | Filtered to academic domains |
| `search_general(query)` | Excludes low-quality sources |

**Academic Domains** (allowlist):
- arxiv.org, nature.com, ieee.org, sciencedirect.com, springer.com
- pubmed.ncbi.nlm.nih.gov, acm.org, wiley.com, jstor.org

**Blocked Domains** (denylist):
- pinterest.com, quora.com, reddit.com, facebook.com, twitter.com

---

### Knowledge Tools (`infrastructure/knowledge_tools.py`)

**Purpose**: Vector-based storage and semantic search using LanceDB.

**Tools Exposed**:
| Tool | Description |
|------|-------------|
| `save_finding(...)` | Store finding with embeddings |
| `search_knowledge(query)` | Semantic search over findings |
| `list_sources()` | Get all unique source URLs |
| `get_finding(id)` | Retrieve specific finding |
| `get_findings_index()` | Compact overview for planning |

**Finding Schema**:
```python
{
    "id": str,              # UUID
    "content": str,         # Finding text (500-2000 chars)
    "source_url": str,
    "source_title": str,
    "subtask_id": int,
    "worker_id": str,
    "timestamp": str,
    "verified": bool,
    "search_type": str,     # "academic" | "general"
    "vector": List[float],  # 3072-dim embedding
}
```

---

## Data Flow

### Complete Research Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  USER QUERY                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 1: PLANNING                                               │   │
│  │  ┌─────────┐                                                     │   │
│  │  │ Planner │──▶ ResearchPlan {subtasks: [S1, S2, ..., S15]}     │   │
│  │  └─────────┘                                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 2: PARALLEL RESEARCH (Iteration Loop)                     │   │
│  │                                                                  │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                │   │
│  │  │Worker 1│  │Worker 2│  │Worker 3│  │Worker N│                │   │
│  │  │   S1   │  │   S2   │  │   S3   │  │   SN   │                │   │
│  │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘                │   │
│  │      │           │           │           │                      │   │
│  │      ▼           ▼           ▼           ▼                      │   │
│  │  ┌─────────────────────────────────────────┐                    │   │
│  │  │          PERPLEXITY SEARCH              │                    │   │
│  │  │  search_academic() / search_general()   │                    │   │
│  │  └───────────────────┬─────────────────────┘                    │   │
│  │                      │                                          │   │
│  │                      ▼                                          │   │
│  │  ┌─────────────────────────────────────────┐                    │   │
│  │  │          LANCEDB KNOWLEDGE BASE         │                    │   │
│  │  │  save_finding() → vector embeddings     │                    │   │
│  │  └───────────────────┬─────────────────────┘                    │   │
│  │                      │                                          │   │
│  │                      ▼                                          │   │
│  │  ┌─────────────────────────────────────────┐                    │   │
│  │  │            CRITIC EVALUATION            │                    │   │
│  │  │  Score: 85/100 | Ready: true            │                    │   │
│  │  └─────────────────────────────────────────┘                    │   │
│  │                      │                                          │   │
│  │         ┌────────────┴────────────┐                             │   │
│  │         │                         │                             │   │
│  │    score < 80              score >= 80                          │   │
│  │         │                         │                             │   │
│  │         ▼                         ▼                             │   │
│  │  ┌─────────────┐           ┌─────────────┐                      │   │
│  │  │  CONTINUE   │           │   PROCEED   │                      │   │
│  │  │ Gap-filling │           │ to Phase 3  │                      │   │
│  │  └─────────────┘           └─────────────┘                      │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3: MULTI-PERSPECTIVE ANALYSIS (Optional)                  │   │
│  │                                                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │   │
│  │  │Technical │  │ Industry │  │ Skeptic  │                       │   │
│  │  │ Expert   │  │  Expert  │  │ Expert   │                       │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                       │   │
│  │       └─────────────┼─────────────┘                             │   │
│  │                     ▼                                           │   │
│  │            Expert Perspectives[]                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 4: SYNTHESIS                                              │   │
│  │                                                                  │   │
│  │  ┌─────────┐                                                     │   │
│  │  │ Editor  │──▶ search_knowledge() ──▶ Write Sections           │   │
│  │  └─────────┘                                                     │   │
│  │                     │                                            │   │
│  │                     ▼                                            │   │
│  │  ┌─────────────────────────────────────────┐                    │   │
│  │  │         RESEARCH REPORT                 │                    │   │
│  │  │  • 5,000-10,000 words                   │                    │   │
│  │  │  • Academic structure                   │                    │   │
│  │  │  • Inline citations [Source]            │                    │   │
│  │  │  • Complete reference list              │                    │   │
│  │  └─────────────────────────────────────────┘                    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Modes

### Available Presets (`swarm_factory.py`)

| Preset | Workers | Subtasks | Duration | Use Case |
|--------|---------|----------|----------|----------|
| **quick** | 3 | 3 | 2-5 min | Simple queries |
| **balanced** | 5 | 5 | 5-10 min | Most queries |
| **deep** | 7 | 10 | 10-15 min | Complex topics |
| **academic** | 5 | 7 | 10-15 min | Research papers |
| **technical** | 5 | 5 | 5-10 min | Technical queries |
| **deep_research** | 7 | 15 | 20-30 min | PhD-level |
| **express_deep** | 5 | 7 | 5-10 min | Quick comprehensive |

### Usage Examples

```python
from swarm_factory import quick_research, deep_research, create_swarm

# Quick research
result = quick_research("What is transformer architecture?")

# Deep research (PhD-level)
result = deep_research("State of AI agents in 2024", express=False)

# Custom configuration
swarm = create_swarm("academic", max_workers=3)
result = swarm.research("Neural network pruning techniques")
```

---

## Configuration System

### Configuration Hierarchy (`config.py`)

```python
@dataclass
class Config:
    models: ModelConfig          # LLM model settings
    search: SearchConfig         # Perplexity search settings
    daytona: DaytonaConfig       # Sandbox settings
    knowledge: KnowledgeConfig   # LanceDB settings
    swarm: SwarmConfig           # Orchestration settings
    deep_research: DeepResearchConfig  # Deep research mode
```

### Model Configuration

| Agent | Default Model | Temperature | Purpose |
|-------|---------------|-------------|---------|
| Planner | claude-opus-4-5-20251101 | 0.3 | Strategic planning |
| Worker | claude-haiku-4-5-20251001 | 0.5 | Fast parallel work |
| Editor | claude-opus-4-5-20251101 | 0.7 | Creative synthesis |
| Critic | claude-opus-4-5-20251101 | 0.2 | Consistent evaluation |

### Environment Variables

```bash
# Required
PERPLEXITY_API_KEY=pplx-xxx        # Perplexity Search API
OPENAI_API_KEY=sk-xxx              # Embeddings
LITELLM_API_BASE=https://xxx       # LLM proxy
LITELLM_API_KEY=xxx                # LLM authentication

# Optional
LANCEDB_PATH=./research_kb         # Knowledge base path
DAYTONA_API_KEY=xxx                # Code sandbox (optional)
```

---

## Quality Control Pipeline

### Critic Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUALITY CONTROL PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ITERATION 1                                                    │
│  ┌─────────────┐                                               │
│  │  Findings   │──▶ Critic Evaluate ──▶ Score: 65/100          │
│  │  (20 docs)  │                       Gaps: 3 critical        │
│  └─────────────┘                       Ready: NO               │
│        │                                                        │
│        ▼                                                        │
│  Generate follow-up queries from gaps                           │
│        │                                                        │
│        ▼                                                        │
│  ITERATION 2                                                    │
│  ┌─────────────┐                                               │
│  │  Findings   │──▶ Critic Evaluate ──▶ Score: 78/100          │
│  │  (35 docs)  │                       Gaps: 1 critical        │
│  └─────────────┘                       Ready: NO               │
│        │                                                        │
│        ▼                                                        │
│  Continue gap-filling...                                        │
│        │                                                        │
│        ▼                                                        │
│  ITERATION 3                                                    │
│  ┌─────────────┐                                               │
│  │  Findings   │──▶ Critic Evaluate ──▶ Score: 85/100          │
│  │  (48 docs)  │                       Gaps: 0 critical        │
│  └─────────────┘                       Ready: YES ✓            │
│        │                                                        │
│        ▼                                                        │
│  PROCEED TO SYNTHESIS                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quality Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Overall Score** | ≥ 80 | Combined quality metric |
| **Coverage Score** | ≥ 70 | Topic coverage completeness |
| **Academic Ratio** | ≥ 30% | Minimum academic sources |
| **Critical Gaps** | 0 | No gaps with importance ≥ 4 |

---

## Knowledge Management

### LanceDB Schema

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANCEDB FINDINGS TABLE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Finding Record                                          │   │
│  │  ─────────────────────────────────────────────────────   │   │
│  │  id: "abc12345"                                          │   │
│  │  content: "GPT-4 achieved 86.4% on MMLU..."             │   │
│  │  source_url: "https://arxiv.org/abs/2303.08774"         │   │
│  │  source_title: "GPT-4 Technical Report"                 │   │
│  │  subtask_id: 3                                          │   │
│  │  worker_id: "W03"                                       │   │
│  │  timestamp: "2024-01-15T10:30:00Z"                      │   │
│  │  verified: true                                         │   │
│  │  search_type: "academic"                                │   │
│  │  vector: [0.023, -0.156, ..., 0.089]  (3072 dims)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Semantic Search Flow:                                          │
│  ─────────────────────                                          │
│  Query: "language model benchmarks"                             │
│       │                                                         │
│       ▼                                                         │
│  OpenAI Embedding (text-embedding-3-large)                      │
│       │                                                         │
│       ▼                                                         │
│  Vector Similarity Search                                       │
│       │                                                         │
│       ▼                                                         │
│  Top-K Results (ranked by cosine similarity)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Checkpointing System

For long-running research sessions, the system saves checkpoints:

```
checkpoints/
├── checkpoint_planning_20241204_195256.json
├── checkpoint_iteration_1_20241204_201011.json
├── checkpoint_iteration_2_20241204_203517.json
└── checkpoint_complete_20241204_205242.json
```

Checkpoint contents:
- Research phase (planning, research, synthesis)
- Current iteration number
- Serialized research plan
- Collected findings
- Critic evaluations
- Resume instructions

---

## File Structure Summary

```
deep research system isara/
├── main.py                    # Main orchestration (ResearchSwarm, DeepResearchSwarm)
├── swarm_factory.py           # Factory functions and presets
├── config.py                  # Configuration management
├── run_express_research.py    # Quick-start script
│
├── agents/
│   ├── __init__.py           # Agent exports
│   ├── planner.py            # Query decomposition
│   ├── worker.py             # Search & save findings
│   ├── editor.py             # Report synthesis
│   ├── critic.py             # Quality evaluation
│   ├── domain_experts.py     # Multi-perspective analysis
│   └── schemas.py            # Pydantic data models
│
├── infrastructure/
│   ├── __init__.py
│   ├── perplexity_tools.py   # Web search toolkit
│   ├── knowledge_tools.py    # LanceDB vector storage
│   ├── parallel_tools.py     # URL extraction
│   ├── retry_utils.py        # Retry decorators
│   └── daytona_tools.py      # Code sandbox (optional)
│
├── research_kb/              # LanceDB database
│   └── findings.lance/
│
├── checkpoints/              # Research session checkpoints
│
└── tests/
    ├── test_agents.py
    ├── test_tools.py
    └── test_orchestration.py
```

---

## Summary

The Deep Research Swarm is a sophisticated multi-agent system that combines:

1. **Intelligent Planning**: Claude Opus decomposes queries into comprehensive research subtasks
2. **Parallel Execution**: Multiple Claude Haiku workers search simultaneously
3. **Quality Control**: Iterative refinement with critic evaluation and gap analysis
4. **Multi-Perspective Analysis**: Domain experts provide diverse viewpoints
5. **Academic Synthesis**: 5,000-10,000 word reports with proper citations
6. **Persistent Knowledge**: Vector-based storage for semantic retrieval

This architecture enables PhD-level research quality while maintaining reasonable execution times (5-30 minutes depending on mode).

