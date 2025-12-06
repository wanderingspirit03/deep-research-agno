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

import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.utils.log import logger

from infrastructure.retry_utils import with_retry


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
                
                ## Research Planning Philosophy
                
                Think like a PhD researcher planning a literature review. Your plan should
                enable thorough understanding from foundations to cutting-edge developments.
                
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
                - Priority: 1 (essential), 2 (important), 3 (supplementary)
                
                ## Quality Guidelines
                
                - Create {self.max_subtasks} comprehensive subtasks
                - Balance academic (40%) and general (60%) sources
                - Include multiple perspectives (technical, practical, critical)
                - Ensure subtasks are mutually exclusive but collectively exhaustive
                - Include at least one subtask for limitations/challenges
                - Include at least one subtask for future directions
                
                ## Query Optimization Tips
                
                - Use specific technical terms
                - Include year qualifiers for recent work (e.g., "2024")
                - Add context words ("survey", "benchmark", "limitations")
                - Avoid overly broad queries
            """).strip(),
        ]
    
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

