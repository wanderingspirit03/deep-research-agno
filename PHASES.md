# Deep Research Swarm - Development Phases

## Overview

Building a Recursive Deep Research Swarm using:
- **Agno** - Multi-agent orchestration framework
- **Perplexity Search API** - Web search with multi-query batching
- **Daytona** - Secure sandboxed code execution
- **LanceDB** - Vector storage for shared agent knowledge
- **LiteLLM** - Unified LLM interface

---

## Phase 1: Foundation ✅ COMPLETE

### Goals
- [x] Create project structure
- [x] Set up configuration system
- [x] Implement Perplexity Search Tools (basic)
- [x] Test basic search functionality

### Files
```
recursive_deep_research/
├── requirements.txt
├── .env.example
├── config.py
└── test_search.py (basic validation)
```

### Success Criteria ✅
- ✅ Can run a simple Perplexity search query
- ✅ Can run multi-query batch searches
- ✅ Results are properly parsed and returned
- ✅ Domain filtering works (academic vs general)

---

## Phase 2: Tools Layer ✅ COMPLETE

### Goals
- [x] Complete PerplexitySearchTools with domain filtering
- [x] Implement DaytonaSandboxTools
- [x] Implement KnowledgeTools with LanceDB

### Files
```
infrastructure/
├── __init__.py
├── perplexity_tools.py   # search(), batch_search(), search_academic(), search_general()
├── daytona_tools.py      # run_code(), verify_url(), scrape_content(), cleanup()
└── knowledge_tools.py    # save_finding(), search_knowledge(), list_sources(), get_finding()
```

### Success Criteria ✅
- ✅ Domain filtering works (academic vs general)
- ✅ Can execute code in Daytona sandbox
- ✅ Can save/retrieve findings from LanceDB

---

## Phase 3: Agents Layer ✅ COMPLETE

### Goals
- [x] Implement Planner Agent
- [x] Implement Worker Agent factory
- [x] Implement Editor Agent

### Files
```
agents/
├── __init__.py
├── planner.py          # PlannerAgent with ResearchPlan/Subtask structured output
├── worker.py           # WorkerAgent factory + execute_subtask/execute_subtasks
└── editor.py           # EditorAgent with synthesize/quick_summary
```

### Success Criteria ✅
- ✅ Planner decomposes queries into subtasks (structured ResearchPlan output)
- ✅ Workers execute searches and save findings (uses Perplexity + LanceDB)
- ✅ Editor synthesizes final report (comprehensive markdown reports with citations)

---

## Phase 4: Orchestration ✅ COMPLETE

### Goals
- [x] Implement main orchestration loop
- [x] Add error handling and retries
- [x] Implement parallel worker execution

### Files
```
├── main.py              # ResearchSwarm orchestrator with Agno Workflow
├── swarm_factory.py     # Factory for creating pre-configured swarms
└── test_orchestration.py # Phase 4 test suite (8/8 tests passing)
```

### Success Criteria ✅
- ✅ End-to-end query → report flow works
- ✅ Graceful error handling with fallbacks
- ✅ Workers execute in parallel using Agno Workflow Parallel

---

## Phase 5: Deep Research Upgrade ✅ COMPLETE

### Goals
- [x] Add 30-minute timeout support for long research sessions
- [x] Implement exponential backoff retry utilities
- [x] Create Critic Agent for quality evaluation
- [x] Create Domain Expert Agents for multi-perspective analysis
- [x] Enhanced Pydantic schemas for deep research
- [x] Iterative research loop with gap analysis
- [x] Checkpoint support for long-running research
- [x] PhD-level enhanced prompts for all agents

### New Files
```
agents/
├── critic.py           # CriticAgent - evaluates quality, identifies gaps
├── domain_experts.py   # DomainExpertAgent - multi-perspective analysis
└── schemas.py          # DeepSubtask, QualityFinding, CriticEvaluation, etc.

infrastructure/
└── retry_utils.py      # with_retry, with_async_retry, RetryContext

main.py                 # Added DeepResearchSwarm class
swarm_factory.py        # Added deep_research, express_deep presets
```

### Success Criteria ✅
- ✅ 1800s (30 min) timeout configured for all agents
- ✅ Retry with exponential backoff (3 retries, 5s base)
- ✅ CriticAgent provides quality scores and gap analysis
- ✅ 5 domain expert types (technical, industry, skeptic, futurist, academic)
- ✅ DeepResearchSwarm with iterative research loop
- ✅ Checkpoint save/load for resuming research
- ✅ Enhanced prompts for PhD-level research depth

---

## Phase 6: Production Ready

### Goals
- [ ] Add comprehensive logging with structured output
- [ ] Implement LiteLLM fallback chains
- [ ] Add streaming output for real-time progress
- [ ] Performance optimization and caching
- [ ] Web UI dashboard

### Success Criteria
- Production-ready system
- Robust error recovery
- Good observability
- User-friendly interface

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Project Structure | ✅ Done | requirements.txt, config.py |
| Config System | ✅ Done | Dataclasses with env loading, DeepResearchConfig |
| Perplexity Tools | ✅ Done | Multi-query + domain filtering |
| Daytona Tools | ✅ Done | Code execution, URL verify, scraping |
| Knowledge Tools | ✅ Done | LanceDB vector storage, semantic search |
| Planner Agent | ✅ Done | Query decomposition with PhD-level prompts |
| Worker Agent | ✅ Done | Multi-query search + detailed extraction |
| Editor Agent | ✅ Done | Academic survey format reports |
| Critic Agent | ✅ Done | Quality evaluation, gap analysis |
| Domain Experts | ✅ Done | 5 expert perspectives |
| Orchestration | ✅ Done | ResearchSwarm + DeepResearchSwarm |
| Retry Utils | ✅ Done | Exponential backoff, retriable error detection |
| Checkpointing | ✅ Done | Save/load for long research |
| Production Ready | 🔴 Not Started | Phase 6 |

---

## Deep Research Quick Start

### Express Deep Research (5-10 min)
```python
from swarm_factory import deep_research

result = deep_research(
    "What are the latest advances in large language models?",
    express=True  # 1 iteration, faster
)
print(result.report)
```

### Full PhD-Level Research (20-30 min)
```python
from swarm_factory import deep_research

result = deep_research(
    "Comprehensive analysis of transformer architecture evolution",
    express=False  # 3 iterations, thorough
)
print(result.report)
```

### Custom Configuration
```python
from main import DeepResearchSwarm

swarm = DeepResearchSwarm(
    max_workers=7,
    max_subtasks=15,
    max_iterations=3,
    quality_threshold=80,
    checkpoint_dir="./my_checkpoints",
)

result = swarm.deep_research(
    "Your research query",
    use_experts=True,
    save_checkpoint=True,
)
```

---

## API Notes

### Perplexity Search API
```python
from perplexity import Perplexity

client = Perplexity()

# Single query
search = client.search.create(
    query="latest AI developments",
    max_results=5
)

# Multi-query (up to 5)
search = client.search.create(
    query=["query1", "query2", "query3"],
    max_results=10
)

# Results
for result in search.results:
    print(result.title, result.url, result.snippet)
```

**Note**: Domain filtering not built-in, we implement post-filtering.

### Daytona SDK
```python
from daytona import Daytona, DaytonaConfig

daytona = Daytona(DaytonaConfig(api_key="..."))
sandbox = daytona.create()

response = sandbox.process.code_run('print("hello")')
print(response.result)

sandbox.delete()
```

### Agno Agent with LiteLLM (isara proxy)
```python
import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.agent import Agent
from agno.models.litellm import LiteLLM

# Available models:
# gpt-5-2025-08-07, gpt-5-mini-2025-08-07, gpt-5-nano-2025-08-07, gpt-5-codex
# claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001, claude-opus-4-1-20250805, claude-opus-4-5-20251101

agent = Agent(
    model=LiteLLM(
        id="gpt-5-mini-2025-08-07",
        api_base="https://model-proxy.development.research-platform.isara.io/v1",
        api_key=os.getenv("LITELLM_API_KEY"),
    ),
    tools=[MyCustomToolkit()],
    markdown=True
)
```

