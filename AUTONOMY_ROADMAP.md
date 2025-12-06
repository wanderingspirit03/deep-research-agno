## Deep Research Swarm – Autonomy & Quality Roadmap

This document outlines how we can evolve the Deep Research Swarm into a **more autonomous, higher‑quality, and still safe/cost‑controlled** system.

It focuses on:
- Where the current design is already strong
- Where autonomy and robustness are limited
- Concrete, incremental improvements to agents, tools, and orchestration

---

## 1. Current Architecture (Recap)

**Agents:**
- **PlannerAgent** (`planner.py`):  
  - Decomposes the user query into up to `max_subtasks` subtasks  
  - Decides: focus, search type (academic/general), priority, estimated depth
- **WorkerAgent** (`worker.py`):  
  - Executes subtasks using tools:
    - `PerplexitySearchTools` → web/academic search
    - `KnowledgeTools` → save & search findings in LanceDB
    - `ParallelExtractTools` (optional) → URL extraction
- **EditorAgent** (`editor.py`):  
  - Uses `KnowledgeTools` to search the KB and synthesize a narrative report
  - Now uses two‑pass synthesis (plan sections → targeted searches → write)
- **CriticAgent** (`critic.py`):  
  - Scores the **set of findings** passed in by the orchestrator
  - Evaluates coverage, quality, gaps; can also review a draft
- **DomainExpertAgent** (`domain_experts.py`):  
  - Analyzes the same findings from multiple roles: technical, industry, skeptic, futurist, academic

**Data flow (simplified):**
1. Planner creates a `ResearchPlan` (subtasks).
2. Workers execute subtasks in parallel and call `save_finding()` → LanceDB.
3. Orchestrator collects findings from the KB.
4. Critic + Expert panel evaluate the **collected findings** (no tools).
5. Editor uses `KnowledgeTools` to search & write the final report.

**Key design choice:**  
Only **Workers** and **Editor** have tools. **Planner, Critic, Experts** are *pure reasoning agents* that operate on data injected by the orchestrator.

---

## 2. Strengths of the Current Design

- **Clear separation of concerns**
  - Workers = “data acquisition & extraction”
  - Editor = “narrative synthesis”
  - Critic/Experts = “evaluation and perspective”
  - Planner = “strategy”

- **Deterministic evaluation context**
  - Critic and Experts see exactly the findings that Workers produced.
  - They cannot silently fetch extra sources that the Editor never uses.
  - This avoids mismatches between what is evaluated and what is actually in the report.

- **Cost and latency control**
  - Expensive external calls (Perplexity, extraction) are concentrated in Workers and Editor.
  - Critic + Experts are just additional LLM calls over existing data.

- **Recent robustness improvements**
  - `max_subtasks` increased to 10 for better coverage.
  - `max_findings_per_subtask` reduced to 8 to avoid context overflow.
  - `quality_score` field on findings, with ranking in `search_knowledge()`.
  - Two‑pass Editor synthesis (plan sections → targeted searches → write).

This is a solid foundation for a high‑quality, resource‑aware system.

---

## 3. Limitations & Opportunities

Despite the strengths, several autonomy and quality limitations remain:

- **No independent fact‑checking**
  - Critic and Domain Experts cannot:
    - Pull *additional* evidence to confirm or refute a finding.
    - Look for missing but obvious perspectives (safety, regulation, negative results, etc.).
  - If Workers miss something, Critic/Experts can only say “coverage is weak,” not fix it.

- **Planner has a fixed subtask budget**
  - Uses a static `max_subtasks` (now 10).
  - Does not adapt subtask count to query complexity (simple vs very broad questions).

- **Orchestrator caps parallelism at `max_workers`**
  - Currently `max_workers = 5`, even if the plan has 10 subtasks.
  - Multi‑iteration loop exists but is still somewhat rigid.

- **Editor is the only component with direct KB search**
  - Critic and Experts have to rely on what Workers stored, even if:
    - Some sources are low‑quality.
    - Coverage is skewed (e.g., mostly industry PR, little primary literature).

- **No explicit global “research budget”**
  - There is no clear token/API budget model per run (only implicit via config).
  - As autonomy increases, we must keep cost/safety constraints explicit.

These are good candidates for targeted, incremental improvements.

---

## 4. Upgrade 1 – Constrained Tools for Critic & Experts

### 4.1 Goals

- Allow **Critic** and **DomainExpertAgent** to:
  - Perform **limited, focused fact‑checking**.
  - Patch obvious coverage gaps in the Worker output.
  - Do this without blowing up context window or cost.

- Preserve:
  - Determinism of the main research result.
  - Clear logging of any additional evidence they used.

### 4.2 Proposed Tool Access

- **Read‑only access to `KnowledgeTools`:**
  - `search_knowledge(query, top_k=5–8, sort_by_quality=True)`
  - `list_sources()` for understanding existing coverage.

- **(Optional) Tiny Perplexity budget**
  - A *very small* number of extra web queries (e.g., 1–3) for:
    - Confirming controversial or surprising claims.
    - Searching for negative trials / safety issues that Workers might skip.

### 4.3 Critic Workflow (Upgraded)

1. Orchestrator passes `findings` as today.
2. Critic:
   - Analyzes findings for coverage and quality.
   - Identifies **concrete gaps** (e.g., “no clinical trial evidence”, “no regulatory context”).
3. For each major gap (up to a small limit, e.g. 2–3):
   - Runs `search_knowledge("query keywords for missing angle", top_k=5)`.
   - Optionally 1 Perplexity query if KB results are thin.
4. Emits:
   - Updated evaluation with **gap‑patching evidence**.
   - Suggestions to Planner for follow‑up subtasks:
     - e.g., “Add a subtask on FDA/EMA regulatory guidance for AI‑enabled devices.”

### 4.4 Domain Expert Workflow (Upgraded)

1. Receives same `findings` + optional `context` as now.
2. May perform a small number of targeted KB searches:
   - Technical expert: deep dives on algorithmic limitations, benchmarks.
   - Industry expert: market adoption, deals, commercialization.
   - Skeptic: failure modes, negative results, reproducibility issues.
3. Returns a structured `ExpertPerspective` that:
   - References both worker findings and any extra KB lookups.
   - Explicitly marks any conclusions that rely on **extra** evidence.

### 4.5 Guardrails

- Configurable per‑run budgets:
  - `critic_max_kb_searches`, `critic_max_web_queries`.
  - `expert_max_kb_searches` per expert.
- All extra queries logged in checkpoints and summaries.
- Keep Critic/Experts **read‑only**: no `save_finding()` to avoid side effects.

---

## 5. Upgrade 2 – Dynamic Planning & Worker Allocation

### 5.1 Goals

- Make the **Planner** adjust the number of subtasks to query complexity:
  - Simple, narrow question → 3–5 subtasks.
  - Complex, cross‑domain question → 10–15 subtasks.

- Allow the orchestrator to:
  - Adjust **worker scheduling** based on plan size and resource budget.

### 5.2 Planner Changes (Conceptual)

- Update planning instructions to:
  - “Create **between 3 and 15 subtasks**, depending on how broad and complex the query is.”
  - Provide explicit heuristics:
    - If the query mentions **multiple domains** (biology + regulation + applications) → more subtasks.
    - If the query is **very narrow / specific** → fewer, deeper subtasks.

- Keep a global clamp:
  - `MIN_SUBTASKS <= len(plan.subtasks) <= MAX_SUBTASKS`
  - Configurable in `SwarmConfig`.

### 5.3 Orchestrator Changes

- Use plan length to drive worker planning:
  - Parallelism still capped by `max_workers`.
  - But multi‑iteration loop explicitly plans remaining subtasks.

- Example policy:
  - If `len(subtasks) <= max_workers`: run once, all parallel.
  - If `len(subtasks) > max_workers`: schedule in rounds, but also:
    - Prioritize Priority 1 subtasks in earlier iterations.
    - Optionally skip low‑priority subtasks if Critic says coverage is already sufficient.

---

## 6. Upgrade 3 – Iterative Gap‑Filling Loop

We already have `max_iterations` and a critic loop. We can make it more **goal‑directed**:

1. **Iteration 1**:
   - Planner → initial subtasks.
   - Workers → findings.
   - Critic/Experts → evaluation + gap report.

2. **Planner (Refinement)**:
   - Takes Critic/Expert feedback as input.
   - Generates **follow‑up subtasks** specifically targeting gaps.

3. **Iteration 2+**:
   - Workers focus only on new subtasks.
   - Editor waits until:
     - `overall_score >= quality_threshold` *or*
     - `iteration == max_iterations`.

4. **Editor**:
   - Uses all collected findings, but can **weight** sections based on:
     - Richness of evidence.
     - Critic’s confidence.

This creates a genuine **research loop**:  
_plan → search → critique → refine → search → synthesize_.

---

## 7. Upgrade 4 – Richer Evaluation & Metrics

With `quality_score` already introduced in `KnowledgeTools`, we can:

- **At Critic level:**
  - Report metrics like:
    - Coverage by theme (background, SOTA, limitations, future).
    - Source diversity (academic vs general, primary vs secondary).
    - Quality distribution (how many findings above 0.7).
  - Identify **over‑reliance** on certain source types:
    - e.g., “heavy dependence on industry PR vs peer‑reviewed work.”

- **At Editor level:**
  - Bias `search_knowledge()` toward high‑`quality_score` findings.
  - Provide structured “evidence tables” per section in optional appendices.

---

## 8. Upgrade 5 – Safety, Cost & Observability

As autonomy grows, we should make **budgets and observability explicit**:

- **Budgets per run:**
  - Max Perplexity calls (workers + critic + experts).
  - Max tokens per agent (planner/editor/critic/experts).
  - Hard caps per iteration and per run.

- **Logging & checkpoints:**
  - Log **which agent** made **which external call**, with:
    - Query string
    - Tool used
    - Result summary (e.g., top source URLs)
  - Include Critic/Expert extra searches in checkpoints.

- **Failure handling:**
  - If Critic/Expert search fails (API issue, rate limit), they should:
    - Explicitly note in evaluation: “Could not verify X due to tool failure.”
    - Avoid silently degrading trust in findings.

---

## 9. Phased Implementation Plan

### Phase 1 – Low‑Risk Enhancements

- Already done or easy to add:
  - `quality_score` for findings + quality‑aware search.
  - Two‑pass Editor synthesis.
  - Increased `max_subtasks` to 10 for more coverage.

**Next low‑risk steps:**
- Add **read‑only `KnowledgeTools`** to Critic and DomainExpertAgent.
- Add small, config‑backed budgets:
  - `critic_max_kb_searches`, `expert_max_kb_searches`.

### Phase 2 – Smarter Planning & Looping

- Update Planner instructions to choose **3–15 subtasks** based on complexity.
- Make the orchestrator’s multi‑iteration loop:
  - Use Critic feedback to generate follow‑up subtasks.
  - Optionally stop early when `overall_score >= quality_threshold`.

### Phase 3 – Optional Web‑Level Fact‑Checking

- Introduce **very constrained** Perplexity access for Critic/Experts:
  - 1–3 extra queries per run.
  - Primarily for checking:
    - Negative/failed trials.
    - Regulatory warnings.
    - Contradictory evidence.

---

## 10. Summary

The current system already has a clean architecture:
- Workers and Editor hold tools and side effects.
- Planner, Critic, and Domain Experts reason over well‑defined inputs.

To make it **more autonomous and robust** without losing control, we should:

- Give Critic and Experts **constrained, read‑only tool access** for targeted gap‑filling.
- Make Planner and the orchestrator **dynamic**, adapting subtasks and iterations to query complexity and evaluation feedback.
- Use `quality_score` and richer metrics to guide both search and synthesis.
- Introduce clear **budgets and logging** as autonomy and external calls increase.

This roadmap keeps the spirit of the current design—clarity, separation of concerns, and safety—while pushing the swarm closer to a genuinely **self‑directed research collaborator**.


