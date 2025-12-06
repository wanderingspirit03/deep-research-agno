# Deep Dive & Verify - Worker Agent Enhancement

## Overview

This document describes the **"Deep Dive & Verify"** pattern for the Worker Agent, transforming it from a snippet aggregator into a true research agent that visits sources, reads full content, and extracts verified findings.

---

## Current State vs. Target State

### Current State (Shallow)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Perplexity Search                                                  │
│       │                                                             │
│       ▼                                                             │
│  Returns: title, url, snippet (200-500 chars)                       │
│       │                                                             │
│       ▼                                                             │
│  Worker saves snippet directly ──► Knowledge Base                   │
│       │                                                             │
│       ▼                                                             │
│  Problem: Snippet may be inaccurate, hallucinated, or outdated      │
│  Problem: No verification that URL even exists                      │
│  Problem: Missing deeper insights from full article                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Target State (Deep Dive & Verify)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Perplexity Search                                                  │
│       │                                                             │
│       ▼                                                             │
│  Returns: title, url, snippet ──► Treated as "LEADS" not facts      │
│       │                                                             │
│       ▼                                                             │
│  Worker ranks & selects top 3-5 promising URLs                      │
│       │                                                             │
│       ▼                                                             │
│  For each selected URL:                                             │
│       │                                                             │
│       ├──► verify_url(url) ──► Check HTTP status, get real title    │
│       │         │                                                   │
│       │         ▼                                                   │
│       │    URL dead? Skip it. URL live? Continue.                   │
│       │                                                             │
│       ├──► scrape_content(url) ──► Get full article text (5000+)    │
│       │         │                                                   │
│       │         ▼                                                   │
│       │    LLM reads full text, extracts key findings               │
│       │         │                                                   │
│       │         ▼                                                   │
│       └──► save_finding() with verified=True                        │
│                 │                                                   │
│                 ▼                                                   │
│           Knowledge Base (high-quality, verified findings)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Process Flow

### Phase 1: Search & Rank

**Input:** Subtask query (e.g., "EV battery recycling environmental impact")

**Process:**
1. Worker calls `search_academic()` or `search_general()` based on subtask type
2. Receives 10 search results with URLs and snippets
3. Worker uses LLM reasoning to **rank and select** top 3-5 URLs based on:
   - Source authority (academic domains, government sites, reputable news)
   - Relevance to the specific subtask focus
   - Recency (prefer recent publications)
   - Snippet quality (specific data vs. vague claims)

**Output:** Ordered list of 3-5 URLs to investigate deeply

### Phase 2: Verify URLs

**Input:** List of candidate URLs

**Process:**
1. For each URL, call `verify_url(url)` via Daytona sandbox
2. Check response:
   - **HTTP 200:** URL is live → continue
   - **HTTP 404/5xx:** URL is dead → skip, log warning
   - **Timeout:** URL unreachable → skip, log warning
3. Compare returned page title with expected title (sanity check)

**Output:** Filtered list of verified, accessible URLs

### Phase 3: Scrape Full Content

**Input:** Verified URLs

**Process:**
1. For each verified URL, call `scrape_content(url, max_chars=8000)`
2. Daytona sandbox fetches page, strips HTML, returns clean text
3. Handle edge cases:
   - **Paywall detected:** Log as "partial access", use available content
   - **PDF link:** Note limitation (future: add PDF parsing)
   - **JavaScript-heavy page:** May return limited content

**Output:** Full text content for each URL (up to 8000 chars)

### Phase 4: Extract Verified Findings

**Input:** Full text content + original subtask focus

**Process:**
1. Worker LLM reads the full scraped content
2. Extracts **specific, factual findings** that relate to the subtask:
   - Statistics and numbers (with context)
   - Methodologies and approaches
   - Key conclusions and claims
   - Dates, names, organizations mentioned
3. **Cross-references** with original snippet:
   - Does the full text support the snippet's claims?
   - Are there additional insights not in the snippet?
   - Any contradictions or nuances?

**Extraction Prompt Template:**
```
You have scraped the full content from: {url}
Title: {title}

Original search snippet claimed: "{snippet}"

Your task for subtask "{subtask_focus}":
1. Verify if the snippet's claims are supported by the full text
2. Extract 2-4 specific, factual findings from this source
3. Include exact quotes, statistics, or data points when available
4. Note the publication date if mentioned

For each finding, provide:
- The specific fact or insight
- A direct quote or data point supporting it
- Your confidence level (high/medium/low)
```

**Output:** 2-4 detailed findings per URL, grounded in actual source text

### Phase 5: Save Verified Findings

**Input:** Extracted findings with metadata

**Process:**
1. For each finding, call `save_finding()` with:
   ```python
   save_finding(
       content="[Detailed finding with supporting quote/data]",
       source_url=url,
       source_title=verified_title,
       subtask_id=subtask.id,
       worker_id=worker_id,
       search_type=subtask.search_type,
       verified=True  # ← Now actually verified!
   )
   ```

**Output:** High-quality, verified findings in Knowledge Base

---

## Data Model Enhancement

### Current Finding Schema

```python
{
    "id": str,
    "content": str,           # Snippet-level content
    "source_url": str,
    "source_title": str,
    "subtask_id": int,
    "worker_id": str,
    "timestamp": str,
    "verified": bool,         # Always False currently
    "search_type": str,
    "vector": list,
}
```

### Enhanced Finding Schema

```python
{
    "id": str,
    "content": str,           # Deep extracted content
    "source_url": str,
    "source_title": str,
    "subtask_id": int,
    "worker_id": str,
    "timestamp": str,
    "verified": bool,         # True if URL visited & content confirmed
    "search_type": str,
    "vector": list,
    
    # NEW FIELDS
    "verification_status": str,    # "verified" | "partial" | "failed"
    "http_status": int,            # 200, 404, etc.
    "content_depth": str,          # "snippet" | "full_scrape" | "partial_scrape"
    "source_quote": str,           # Direct quote from source (optional)
    "confidence": str,             # "high" | "medium" | "low"
    "scraped_chars": int,          # How much content was retrieved
}
```

---

## Worker Agent Instructions (Updated)

```python
DEEP_VERIFY_INSTRUCTIONS = """
You are Research Worker {worker_id}, assigned to investigate: **{subtask.focus}**

## Your Enhanced Process

### Step 1: Search
Use `{search_method}` with query: "{subtask.query}"

### Step 2: Rank & Select
From the search results, select the TOP 3-5 most promising URLs based on:
- Source authority (prefer .edu, .gov, arxiv, nature, ieee, etc.)
- Relevance to your specific focus
- Recency of information
- Specificity of the snippet (data > vague claims)

### Step 3: Verify & Scrape (for each selected URL)
a) Call `verify_url(url)` to confirm the page exists
b) If verified, call `scrape_content(url)` to get full text
c) Read the full content carefully

### Step 4: Extract Deep Findings
From the full scraped content, extract specific findings:
- Look for statistics, percentages, dates, names
- Find direct quotes that support key claims
- Note methodologies or data sources mentioned
- Identify conclusions supported by evidence

### Step 5: Save Verified Findings
For EACH valuable finding, call `save_finding` with:
- content: Detailed finding with supporting evidence
- source_url: The verified URL
- source_title: Title from verification
- verified: True (you visited the source!)
- Include specific quotes or data points in the content

## Quality Standards
- Only save findings you extracted from ACTUAL page content
- Include at least one specific data point or quote per finding
- If scraping fails, note it and use snippet as fallback (verified=False)
- Aim for 2-4 HIGH-QUALITY findings per source (quality > quantity)
"""
```

---

## Error Handling

### URL Verification Failures

| Scenario | Action |
|----------|--------|
| HTTP 404 | Skip URL, log warning, try next candidate |
| HTTP 403 (Forbidden) | Skip URL, likely paywall |
| HTTP 5xx | Retry once, then skip |
| Timeout | Skip URL, log as unreachable |
| SSL Error | Skip URL, log security issue |

### Scraping Failures

| Scenario | Action |
|----------|--------|
| Empty content | Use snippet as fallback, mark `content_depth="snippet"` |
| Partial content (<500 chars) | Use what's available, mark `content_depth="partial_scrape"` |
| JavaScript-only page | Use snippet as fallback, log limitation |
| PDF link | Log as "PDF not supported", use snippet |

### Fallback Strategy

```python
if scrape_successful and len(content) > 500:
    # Full deep dive
    extract_findings_from_full_text(content)
    save_finding(verified=True, content_depth="full_scrape")
    
elif scrape_partial and len(content) > 100:
    # Partial scrape
    extract_findings_from_partial(content)
    save_finding(verified=True, content_depth="partial_scrape")
    
else:
    # Fallback to snippet
    save_finding_from_snippet(snippet)
    save_finding(verified=False, content_depth="snippet")
```

---

## Performance Considerations

### Time Estimates

| Step | Time per URL | Parallel? |
|------|--------------|-----------|
| Search | 1-2s | N/A (single call) |
| verify_url | 0.5-2s | Yes |
| scrape_content | 2-5s | Yes |
| LLM extraction | 3-8s | Yes |
| save_finding | 0.5s | Yes |

**Total per subtask (3 URLs):** ~30-60 seconds

### Optimization Strategies

1. **Parallel URL verification:** Verify all 3-5 URLs simultaneously
2. **Parallel scraping:** Scrape multiple URLs at once
3. **Batch LLM calls:** Send all scraped content in one prompt (if within token limit)
4. **Caching:** Cache scraped content for URLs seen in previous subtasks
5. **Smart URL selection:** Prioritize URLs that appear in multiple search results

---

## Token Usage Analysis

### Current (Snippet-only)

| Component | Tokens |
|-----------|--------|
| Search results | ~500 |
| Worker reasoning | ~200 |
| Save findings | ~100 |
| **Total per subtask** | **~800 tokens** |

### Enhanced (Deep Verify)

| Component | Tokens |
|-----------|--------|
| Search results | ~500 |
| URL selection reasoning | ~300 |
| Scraped content (3 URLs × 2000) | ~6000 |
| Extraction per URL (3 × 500) | ~1500 |
| Save findings | ~300 |
| **Total per subtask** | **~8600 tokens** |

**Cost increase:** ~10x more tokens per subtask
**Quality increase:** Significantly higher (verified, detailed findings)

---

## Implementation Checklist

### Phase 1: Core Changes
- [ ] Update Worker instructions with deep verify process
- [ ] Add URL ranking/selection logic
- [ ] Integrate `verify_url()` into Worker flow
- [ ] Integrate `scrape_content()` into Worker flow
- [ ] Add extraction prompt for full-text analysis

### Phase 2: Schema Updates
- [ ] Add new fields to Knowledge Base schema
- [ ] Update `save_finding()` to accept new fields
- [ ] Migrate existing findings (set `content_depth="snippet"`)

### Phase 3: Error Handling
- [ ] Implement fallback chain (full → partial → snippet)
- [ ] Add retry logic for transient failures
- [ ] Log verification failures for analysis

### Phase 4: Testing
- [ ] Test with academic URLs (arxiv, nature)
- [ ] Test with news URLs (paywalls, JS-heavy)
- [ ] Test with dead/broken URLs
- [ ] Benchmark time and token usage

### Phase 5: Optimization
- [ ] Add parallel verification
- [ ] Add parallel scraping
- [ ] Implement URL caching
- [ ] Tune `max_chars` for scraping

---

## Example: Before vs. After

### Before (Snippet-only)

```
Finding: "Electric vehicle batteries have a recycling rate of approximately 
         5% globally according to recent studies."
Source: https://example.com/ev-recycling
Verified: False
```

**Problems:**
- Is 5% accurate? We don't know.
- What year is "recent"? Unknown.
- Which studies? Not specified.
- Does the URL even work? Not checked.

### After (Deep Verify)

```
Finding: "According to the International Energy Agency's Global EV Outlook 
         2024 report, the current recycling rate for EV lithium-ion batteries 
         is estimated at 5% globally, though this is projected to reach 
         15-20% by 2030 as recycling infrastructure scales. The report notes 
         that 'end-of-life battery volumes will increase 50-fold by 2040, 
         making recycling economically viable at scale.'"
         
Source: https://www.iea.org/reports/global-ev-outlook-2024
Title: Global EV Outlook 2024 – Analysis - IEA
Verified: True
HTTP Status: 200
Content Depth: full_scrape
Scraped Chars: 6,234
Confidence: high
Source Quote: "end-of-life battery volumes will increase 50-fold by 2040"
```

**Improvements:**
- Specific source identified (IEA)
- Exact report cited (2024)
- Direct quote included
- Additional context (2030 projection)
- URL confirmed accessible
- High confidence rating

---

## Decision Points

### How many URLs to deep-dive per subtask?

| Option | Trade-off |
|--------|-----------|
| 1-2 URLs | Fast but may miss important sources |
| 3-5 URLs | Balanced (recommended) |
| 5-10 URLs | Thorough but slow and expensive |

**Recommendation:** 3 URLs for general, 5 URLs for academic subtasks

### Max chars to scrape per URL?

| Option | Trade-off |
|--------|-----------|
| 2,000 chars | Fast, may miss key content |
| 5,000 chars | Balanced (recommended) |
| 10,000 chars | Very thorough but high token cost |

**Recommendation:** 5,000 chars default, 8,000 for academic papers

### When to skip verification?

Consider skipping deep verification for:
- Time-sensitive queries (breaking news)
- High-volume exploratory searches
- Trusted domains (arxiv, nature, .gov) - still verify but trust snippets more

---

## Next Steps

1. **Review this design** - Does this match your vision?
2. **Decide on parameters** - URLs per subtask, chars per scrape
3. **Implement Phase 1** - Core Worker changes
4. **Test incrementally** - Start with one subtask type
5. **Measure quality improvement** - Compare reports before/after

---

## Questions for Discussion

1. Should we implement a **confidence threshold** (skip low-confidence findings)?
2. Should the **Editor weight** verified findings higher than unverified?
3. Do we need **PDF parsing** support for academic papers?
4. Should we add **domain-specific extraction** (e.g., special handling for arxiv)?

