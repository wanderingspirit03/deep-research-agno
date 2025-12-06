"""
Worker Agent - Executes research subtasks by searching and saving findings

The Worker Agent:
1. Receives a subtask from the planner
2. Searches using Perplexity (academic or general)
3. Evaluates and filters results
4. Saves valuable findings to the knowledge base
"""
import os
from typing import Optional, List
from textwrap import dedent

import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.utils.log import logger

from infrastructure.perplexity_tools import PerplexitySearchTools
from infrastructure.knowledge_tools import KnowledgeTools
from infrastructure.retry_utils import with_retry
try:
    from infrastructure.parallel_tools import ParallelExtractTools
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
from .planner import Subtask


# =============================================================================
# Worker Agent Factory
# =============================================================================

def create_worker_agent(
    worker_id: str,
    subtask: Subtask,
    model_id: str = "claude-haiku-4-5-20251001",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    search_tools: Optional[PerplexitySearchTools] = None,
    knowledge_tools: Optional[KnowledgeTools] = None,
    extract_tools: Optional["ParallelExtractTools"] = None,
) -> Agent:
    """
    Factory function to create a Worker Agent for a specific subtask.
    
    Args:
        worker_id: Unique identifier for this worker
        subtask: The subtask to execute
        model_id: LLM model ID (default: claude-haiku-4-5)
        api_base: LiteLLM API base URL
        api_key: LiteLLM API key
        temperature: Model temperature
        search_tools: Perplexity search toolkit (shared)
        knowledge_tools: Knowledge base toolkit (shared)
        extract_tools: Parallel URL extraction toolkit (shared)
        
    Returns:
        Agent: Configured Agno Agent ready to execute the subtask
    """
    api_base = api_base or os.getenv("LITELLM_API_BASE")
    api_key = api_key or os.getenv("LITELLM_API_KEY")
    
    if not api_base or not api_key:
        raise ValueError("LITELLM_API_BASE and LITELLM_API_KEY must be set")
    
    # Create tools if not provided
    if search_tools is None:
        search_tools = PerplexitySearchTools(max_results=10)
    
    if knowledge_tools is None:
        knowledge_tools = KnowledgeTools()
    
    # Create extract tools if available and not provided
    if extract_tools is None and PARALLEL_AVAILABLE:
        try:
            extract_tools = ParallelExtractTools()
        except Exception as e:
            logger.warning(f"Could not initialize Parallel extract tools: {e}")
            extract_tools = None
    
    # Build instructions based on subtask
    instructions = _build_worker_instructions(worker_id, subtask, has_extract=extract_tools is not None)
    
    logger.info(f"Creating Worker Agent {worker_id} for subtask {subtask.id}: {subtask.focus}")
    
    # Build tools list
    tools = [search_tools, knowledge_tools]
    if extract_tools:
        tools.append(extract_tools)
        logger.info(f"Worker {worker_id} has URL extraction enabled")
    
    agent = Agent(
        name=f"Research Worker {worker_id}",
        model=LiteLLM(
            id=model_id,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            top_p=None,  # Claude doesn't accept both temperature and top_p
        ),
        description=f"Research worker specialized in: {subtask.focus}",
        instructions=instructions,
        tools=tools,
        markdown=True,
        debug_mode=True,
        debug_level=2,
    )
    
    return agent


def _build_worker_instructions(worker_id: str, subtask: Subtask, has_extract: bool = False) -> List[str]:
    """Build worker-specific instructions for deep research"""
    search_method = "search_academic" if subtask.search_type == "academic" else "search_general"
    alt_method = "search_general" if subtask.search_type == "academic" else "search_academic"
    
    # Build extraction instructions if available
    if has_extract:
        extraction_instructions = """
            ## PHASE 2: Deep URL Extraction (CRITICAL)
            
            After searching, use `extract_for_research` to get FULL CONTENT from the best URLs.
            This is the key to comprehensive research - search snippets are NOT enough!
            
            For each promising URL from search results:
            ```
            extract_for_research(
                url="<the source URL>",
                research_query="{subtask.query}",
                subtask_focus="{subtask.focus}"
            )
            ```
            
            This extracts:
            - Full article/page content (thousands of words)
            - Relevant excerpts focused on your research
            - Complete context and details
            
            **Extract at least 3-5 URLs** from your search results to get comprehensive content.
            Prioritize academic papers, official docs, and high-quality sources for extraction.
            """
    else:
        extraction_instructions = """
            ## Note on Content Extraction
            
            URL extraction tools are not available. Extract as much as possible from search snippets.
            """
    
    return [
        dedent(f"""
            You are Research Worker {worker_id}, a meticulous research assistant conducting
            PhD-level investigation on: **{subtask.focus}**
            
            ## Your Mission
            Execute deep research on subtask #{subtask.id}: `{subtask.query}`
            
            ## PHASE 1: Search for Sources
            
            Use multiple search queries to find sources:
            
            1. **Primary Search**: Use `{search_method}` with: "{subtask.query}"
            
            2. **Alternative Queries** (if primary yields <5 results):
               - Rephrase using synonyms
               - Add "2024" or "latest" for recency
               - Add "survey" or "review" for overview sources
               - Try `{alt_method}` for different source types
            
            Collect URLs from search results for deep extraction.
            {extraction_instructions}
            
            ## PHASE 3: Save Comprehensive Findings
            
            After extracting content, save COMPREHENSIVE findings to the knowledge base.
            
            ### Finding Content (EXTRACT EVERYTHING RELEVANT):
            Your content field should include:
            
            1. **All key claims and findings** - Every important statement
            2. **All statistics, numbers, metrics, percentages** - Be exhaustive
            3. **All named entities** - People, organizations, products, models, tools
            4. **Methodology details** - How studies were conducted, sample sizes
            5. **Dates and timeframes** - Publication date, study periods, projections
            6. **Direct quotes** - Important statements when available
            7. **Comparisons** - How things compare to alternatives/baselines
            8. **Limitations mentioned** - Any caveats or constraints noted
            9. **Technical specifications** - Architecture details, parameters
            10. **Results and outcomes** - What happened, what was achieved
            
            **IMPORTANT**: Each finding should be 1000-3000 characters (150-500 words).
            With full URL extraction, you have access to complete article content.
            Extract ALL relevant information - do not artificially truncate.
            
            ### Save Finding With:
            - content: COMPREHENSIVE extraction (1000-3000 chars from extracted content)
            - source_url: The URL
            - source_title: The title
            - subtask_id: {subtask.id}
            - worker_id: "{worker_id}"
            - search_type: "{subtask.search_type}"
            - verified: false
            
            ## Target Output
            
            - **Minimum**: 5 comprehensive findings (each 1000+ chars)
            - **Target**: 8-12 findings with deep extraction
            - **Each finding should be INFORMATION-DENSE** - from full article content
            - **Multiple findings per source is OK** - break down long articles
            
            ## Quality Scoring
            
            Before saving, rate source quality (don't save <3):
            - 5: Peer-reviewed paper, official documentation
            - 4: Major tech company blog, institutional report
            - 3: Reputable news outlet, expert blog
            - 2: Forum posts, opinion pieces
            - 1: Unknown sources, marketing material
            
            ## Final Summary Format
            
            After completing all phases:
            ```
            ## Worker {worker_id} Summary
            
            **URLs Extracted**: [count]
            **Findings Saved**: [count]
            **Total Content**: [approximate word count]
            **Source Types**: [academic/industry/news breakdown]
            
            **Key Discoveries**:
            1. [Most important finding with detail]
            2. [Second most important]
            3. [Third most important]
            
            **Gaps Identified**:
            - [What couldn't be found]
            - [Areas needing more research]
            
            **Confidence Level**: [High/Medium/Low]
            ```
        """).strip(),
    ]


# =============================================================================
# Worker Agent Wrapper Class
# =============================================================================

class WorkerAgent:
    """
    Worker Agent wrapper for executing research subtasks.
    
    Provides a convenient interface for creating and running
    worker agents on subtasks.
    """
    
    def __init__(
        self,
        model_id: str = "claude-haiku-4-5-20251001",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        search_tools: Optional[PerplexitySearchTools] = None,
        knowledge_tools: Optional[KnowledgeTools] = None,
    ):
        """
        Initialize Worker Agent wrapper.
        
        Args:
            model_id: LLM model ID (default: gpt-5-mini)
            api_base: LiteLLM API base URL
            api_key: LiteLLM API key
            temperature: Model temperature
            search_tools: Perplexity search toolkit (shared across workers)
            knowledge_tools: Knowledge base toolkit (shared across workers)
        """
        self.model_id = model_id
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.temperature = temperature
        
        # Shared tools across workers
        self.search_tools = search_tools or PerplexitySearchTools(max_results=10)
        self.knowledge_tools = knowledge_tools or KnowledgeTools()
        
        # Counter for unique worker IDs
        self._worker_counter = 0
    
    def _next_worker_id(self) -> str:
        """Generate next worker ID"""
        self._worker_counter += 1
        return f"W{self._worker_counter:02d}"
    
    @with_retry(max_retries=3, base_delay=3.0)
    def execute_subtask(self, subtask: Subtask, worker_id: Optional[str] = None) -> str:
        """
        Execute a single research subtask.
        
        Args:
            subtask: The subtask to execute
            worker_id: Optional worker ID (auto-generated if not provided)
            
        Returns:
            str: Worker's response/summary
        """
        if worker_id is None:
            worker_id = self._next_worker_id()
        
        logger.info(f"Worker {worker_id} executing subtask {subtask.id}: {subtask.focus}")
        
        # Create worker agent
        agent = create_worker_agent(
            worker_id=worker_id,
            subtask=subtask,
            model_id=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=self.temperature,
            search_tools=self.search_tools,
            knowledge_tools=self.knowledge_tools,
        )
        
        # Execute the subtask
        prompt = f"""
Execute your assigned research subtask:

**Subtask #{subtask.id}:** {subtask.focus}
**Query:** {subtask.query}
**Search Type:** {subtask.search_type}

Search for information, evaluate results, and save all valuable findings to the knowledge base.
        """.strip()
        
        response = agent.run(prompt)
        
        logger.info(f"Worker {worker_id} completed subtask {subtask.id}")
        
        return response.content
    
    def execute_subtasks(self, subtasks: List[Subtask]) -> List[dict]:
        """
        Execute multiple subtasks sequentially.
        
        Args:
            subtasks: List of subtasks to execute
            
        Returns:
            List[dict]: Results for each subtask
        """
        results = []
        
        for subtask in subtasks:
            worker_id = self._next_worker_id()
            
            try:
                response = self.execute_subtask(subtask, worker_id=worker_id)
                results.append({
                    "subtask_id": subtask.id,
                    "worker_id": worker_id,
                    "focus": subtask.focus,
                    "status": "completed",
                    "response": response,
                })
            except Exception as e:
                logger.error(f"Worker {worker_id} failed on subtask {subtask.id}: {e}")
                results.append({
                    "subtask_id": subtask.id,
                    "worker_id": worker_id,
                    "focus": subtask.focus,
                    "status": "failed",
                    "error": str(e),
                })
        
        return results


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Worker Agent Test ===\n")
    
    # Check API config
    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not api_base or not api_key:
        print("❌ LITELLM_API_BASE and LITELLM_API_KEY must be set")
        exit(1)
    
    if not perplexity_key:
        print("❌ PERPLEXITY_API_KEY must be set")
        exit(1)
    
    print(f"✅ LiteLLM API Base: {api_base}")
    print(f"✅ API Key: {api_key[:15]}...")
    print(f"✅ Perplexity Key: {perplexity_key[:10]}...")
    
    # Create test subtask
    test_subtask = Subtask(
        id=1,
        query="GPT-5 capabilities and benchmarks 2024",
        focus="GPT-5 performance analysis",
        search_type="general",
        priority=1
    )
    
    # Create worker
    worker = WorkerAgent()
    
    print(f"\n🔍 Executing subtask: {test_subtask.focus}\n")
    print("-" * 50)
    
    result = worker.execute_subtask(test_subtask)
    
    print("\n📝 Worker Response:")
    print("-" * 50)
    print(result)

