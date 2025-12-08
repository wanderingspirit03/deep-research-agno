"""
Planner Agent - Decomposes research queries into actionable subtasks

The Planner Agent:
1. Analyzes the user's research query
2. Breaks it down into focused subtasks
3. Assigns search strategies (academic vs general)
4. Prioritizes subtasks by importance
"""
import os
from typing import List, Optional
from textwrap import dedent

from pydantic import BaseModel, Field
from typing import Optional

import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.utils.log import logger

from infrastructure.retry_utils import with_retry
from infrastructure.observability import observe


# =============================================================================
# Structured Output Schemas
# =============================================================================

class Subtask(BaseModel):
    """A single research subtask"""
    id: int = Field(..., description="Unique subtask ID (1, 2, 3, ...)")
    query: str = Field(..., description="Specific search query for this subtask")
    focus: str = Field(..., description="What aspect this subtask investigates")
    search_type: str = Field(
        default="general",
        description="Search type: 'academic' for papers/research, 'general' for web/news"
    )
    search_strategy: str = Field(
        default="general_web",
        description="""Search strategy to use:
        - 'general_web': Standard web search
        - 'academic': Scholarly/academic sources
        - 'dork_police_reports': Google dork for police reports with AI disclaimers
        - 'dork_procurement': Google dork for procurement/contract documents
        - 'dork_council_minutes': Google dork for council/board meeting minutes
        - 'dork_news': Google dork for news coverage
        - 'dork_government': Google dork for .gov sites
        - 'procurement_portal': Direct procurement portal search
        - 'open_data_portal': Public records/open data search
        """
    )
    dork_pattern: Optional[str] = Field(
        default=None,
        description="Google dork pattern if search_strategy is a dork type (e.g., 'site:.gov \"Axon DraftOne\"')"
    )
    target_state: Optional[str] = Field(
        default=None,
        description="Target US state code for geographic decomposition (e.g., 'CA', 'TX')"
    )
    target_agencies: List[str] = Field(
        default_factory=list,
        description="Specific agencies to focus on in this subtask"
    )
    priority: int = Field(
        default=1,
        description="Priority 1-3 (1=highest, 3=lowest)"
    )


class ResearchPlan(BaseModel):
    """Complete research plan with subtasks"""
    original_query: str = Field(..., description="The original user query")
    summary: str = Field(..., description="Brief summary of the research approach")
    subtasks: List[Subtask] = Field(..., description="List of research subtasks")
    estimated_depth: str = Field(
        default="medium",
        description="Research depth: 'shallow', 'medium', or 'deep'"
    )
    geographic_scope: Optional[str] = Field(
        default=None,
        description="Geographic scope (e.g., 'US nationwide', 'California', 'Top 50 agencies')"
    )
    decomposition_strategy: Optional[str] = Field(
        default=None,
        description="How the research was decomposed (e.g., 'by_state', 'by_agency_type', 'by_evidence_type')"
    )


# =============================================================================
# Planner Agent
# =============================================================================

class PlannerAgent:
    """
    Planner Agent - Decomposes research queries into subtasks.
    
    Uses GPT-5 Mini via LiteLLM to analyze queries and create
    structured research plans.
    """
    
    def __init__(
        self,
        model_id: str = "gpt-5-mini-2025-08-07",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_subtasks: int = 7,
        temperature: float = 0.3,
    ):
        """
        Initialize Planner Agent.
        
        Args:
            model_id: LLM model ID (default: gpt-5-mini)
            api_base: LiteLLM API base URL
            api_key: LiteLLM API key
            max_subtasks: Maximum number of subtasks to generate
            temperature: Model temperature (lower = more focused)
        """
        self.model_id = model_id
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.max_subtasks = max_subtasks
        self.temperature = temperature
        
        # Lazy-initialized agent
        self._agent: Optional[Agent] = None
    
    @property
    def agent(self) -> Agent:
        """Lazy initialization of Agno Agent"""
        if self._agent is None:
            if not self.api_base or not self.api_key:
                raise ValueError("LITELLM_API_BASE and LITELLM_API_KEY must be set")
            
            logger.info(f"Creating Planner Agent with model: {self.model_id}")
            
            self._agent = Agent(
                name="Research Planner",
                model=LiteLLM(
                    id=self.model_id,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    top_p=None,  # Claude doesn't accept both temperature and top_p
                ),
                description="Expert research strategist that decomposes complex queries into actionable research subtasks.",
                instructions=self._get_instructions(),
                output_schema=ResearchPlan,
                markdown=True,
                debug_mode=True,
                debug_level=2,
            )
        
        return self._agent
    
    def _get_instructions(self) -> List[str]:
        """Get planner agent instructions"""
        return [
            dedent(f"""
                You are an expert research strategist with PhD-level research planning skills.
                Your job is to decompose complex research queries into comprehensive, 
                academically rigorous subtasks for deep investigation.
                
                **Current Date: December 2025** - Use this for temporal context in your planning.
                
                ## Research Planning Philosophy
                
                Think like a PhD researcher planning a literature review. Your plan should
                enable thorough understanding from foundations to cutting-edge developments.
                
                ## CRITICAL: Exhaustive Geographic/Entity Decomposition
                
                For market research, adoption studies, or geographic analysis:
                
                **DECOMPOSE LARGE SCOPES INTO SPECIFIC, MANAGEABLE CHUNKS.**
                
                Examples:
                - "US law enforcement" → Break into subtasks by state or top agencies
                - "Fortune 500 companies" → Break by industry or top 50 companies
                - "University adoption" → Break by region or top research universities
                
                **Create MANY focused subtasks** rather than few broad ones.
                For nationwide US research, consider:
                - Top 10 agencies by officer count (NYPD, LAPD, CPD, etc.)
                - Regional breakdown (Northeast, Southeast, Midwest, West)
                - State-specific searches for high-population states (CA, TX, FL, NY)
                
                ## Google Dorks for Targeted Research
                
                Use **Google Dorks** (advanced search operators) to find specific document types:
                
                ### Dork Patterns for Law Enforcement Research:
                
                **Police Reports with AI Disclaimers:**
                ```
                site:.gov "Axon DraftOne" OR "AI-assisted report" OR "AI-generated"
                site:police.gov "drafted using AI" OR "DraftOne"
                filetype:pdf "incident report" "Axon" "AI" 
                ```
                
                **Procurement Documents:**
                ```
                site:.gov "Axon AI Era Plan" contract OR procurement OR RFP
                site:bidnet.com OR site:govwin.com "Axon DraftOne"
                filetype:pdf "purchase order" "Axon" "AI Era"
                ```
                
                **Council/Board Minutes:**
                ```
                site:.gov "city council" OR "board of supervisors" "Axon DraftOne"
                site:.gov "meeting minutes" "AI report writing" police
                filetype:pdf "council agenda" "Axon AI"
                ```
                
                **News Coverage:**
                ```
                site:localnews.com OR site:patch.com "police" "Axon DraftOne"
                "police department" "adopts" "Axon AI" OR "DraftOne"
                ```
                
                **Government Sites:**
                ```
                site:.gov "Axon" "artificial intelligence" police
                site:state.gov OR site:city.gov "DraftOne" OR "AI Era Plan"
                ```
                
                ### Search Strategy Selection:
                
                Set `search_strategy` to one of:
                - `general_web`: Standard web search
                - `academic`: Scholarly sources
                - `dork_police_reports`: Use dork for police reports
                - `dork_procurement`: Use dork for procurement docs
                - `dork_council_minutes`: Use dork for meeting minutes
                - `dork_news`: Use dork for news coverage
                - `dork_government`: Use dork for .gov sites
                
                When using dork strategies, set `dork_pattern` with the specific pattern.
                
                ## Four-Phase Research Structure
                
                Organize subtasks into four research phases:
                
                ### Phase 1: FOUNDATION (Priority 1)
                - Background knowledge and definitions
                - Historical context and evolution
                - Key concepts and terminology
                - Foundational papers/sources
                
                ### Phase 2: CURRENT STATE (Priority 1)
                - State-of-the-art approaches
                - Recent developments (last 2-3 years)
                - Key players and institutions
                - Performance benchmarks and comparisons
                
                ### Phase 3: CRITICAL ANALYSIS (Priority 2)
                - Limitations and challenges
                - Competing approaches and trade-offs
                - Open problems and debates
                - Practical vs theoretical gaps
                
                ### Phase 4: FUTURE DIRECTIONS (Priority 2-3)
                - Emerging trends and research directions
                - Predictions and roadmaps
                - Potential breakthroughs
                - Societal and ethical implications
                
                ## Subtask Requirements
                
                For each subtask provide:
                - Clear, specific focus area
                - Search-engine optimized query (7-15 words)
                - Search type: 'academic' for papers/research, 'general' for industry/news
                - Search strategy: Choose from dork types for specific document hunting
                - Dork pattern: If using dork strategy, provide the exact pattern
                - Target state/agencies: For geographic decomposition
                - Priority: 1 (essential), 2 (important), 3 (supplementary)
                
                ## Quality Guidelines
                
                - Create {self.max_subtasks} comprehensive subtasks
                - Balance academic (40%) and general (60%) sources
                - Include multiple perspectives (technical, practical, critical)
                - Ensure subtasks are mutually exclusive but collectively exhaustive
                - Include at least one subtask for limitations/challenges
                - Include at least one subtask for future directions
                - **For market/adoption research**: Include geographic decomposition
                - **Use dork strategies** when searching for specific document types
                
                ## Query Optimization Tips
                
                - Use specific technical terms
                - Include year qualifiers for recent work (e.g., "2025")
                - Add context words ("survey", "benchmark", "limitations")
                - Avoid overly broad queries
                - For entity-specific research, name the entities explicitly
                - Use quotes for exact phrase matching
            """).strip(),
        ]
    
    @observe(name="planner.plan")
    @with_retry(max_retries=3, base_delay=5.0)
    def plan(self, query: str) -> ResearchPlan:
        """
        Create a research plan for the given query.
        
        Args:
            query: The user's research query
            
        Returns:
            ResearchPlan: Structured plan with subtasks
        """
        logger.info(f"Planning research for: {query[:50]}...")
        
        prompt = f"""
Create a comprehensive research plan for the following query:

**Query:** {query}

Generate a structured plan with up to {self.max_subtasks} focused subtasks.
For each subtask, specify whether to use 'academic' or 'general' search.
        """.strip()
        
        response = self.agent.run(prompt)
        
        # Handle None response (LLM call failed)
        if response is None:
            logger.error(f"LLM returned None response for model: {self.model_id}")
            raise RuntimeError(f"LLM call failed - model '{self.model_id}' returned no response. Check if model is available on proxy.")
        
        # The response.content should be the ResearchPlan object
        if isinstance(response.content, ResearchPlan):
            plan = response.content
        elif isinstance(response.content, dict):
            plan = ResearchPlan(**response.content)
        else:
            # Parse from string if needed
            import json
            try:
                data = json.loads(str(response.content))
                plan = ResearchPlan(**data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse plan: {e}")
                # Create minimal fallback plan
                plan = ResearchPlan(
                    original_query=query,
                    summary="Direct search approach",
                    subtasks=[
                        Subtask(
                            id=1,
                            query=query,
                            focus="Main query",
                            search_type="general",
                            priority=1
                        )
                    ],
                    estimated_depth="shallow"
                )
        
        logger.info(f"Created plan with {len(plan.subtasks)} subtasks")
        return plan
    
    def plan_to_markdown(self, plan: ResearchPlan) -> str:
        """Convert a research plan to markdown format"""
        lines = [
            f"# Research Plan",
            f"",
            f"**Query:** {plan.original_query}",
            f"",
            f"**Summary:** {plan.summary}",
            f"",
            f"**Depth:** {plan.estimated_depth}",
            f"",
            f"## Subtasks ({len(plan.subtasks)})",
            f"",
        ]
        
        for task in sorted(plan.subtasks, key=lambda t: (t.priority, t.id)):
            priority_emoji = {1: "🔴", 2: "🟡", 3: "🟢"}.get(task.priority, "⚪")
            search_emoji = "📚" if task.search_type == "academic" else "🌐"
            
            lines.extend([
                f"### {task.id}. {task.focus} {priority_emoji}",
                f"",
                f"- **Query:** `{task.query}`",
                f"- **Search:** {search_emoji} {task.search_type}",
                f"- **Priority:** {task.priority}",
                f"",
            ])
        
        return "\n".join(lines)


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Planner Agent Test ===\n")
    
    # Check API config
    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not api_base or not api_key:
        print("❌ LITELLM_API_BASE and LITELLM_API_KEY must be set")
        exit(1)
    
    print(f"✅ API Base: {api_base}")
    print(f"✅ API Key: {api_key[:15]}...")
    
    # Create planner
    planner = PlannerAgent(max_subtasks=5)
    
    # Test planning
    test_query = "What are the latest advances in large language models and their real-world applications?"
    
    print(f"\n📋 Planning for: {test_query}\n")
    print("-" * 50)
    
    plan = planner.plan(test_query)
    
    print(planner.plan_to_markdown(plan))

