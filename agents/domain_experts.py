"""
Domain Expert Agents - Multiple perspectives for comprehensive analysis.

Provides specialized expert agents that analyze research from different viewpoints:
- Technical Expert: ML researcher perspective, focuses on technical depth
- Industry Expert: VP Engineering perspective, focuses on practical applications
- Skeptic Expert: Critical researcher, challenges claims and finds weaknesses
- Futurist Expert: Trend analyst, focuses on future implications
"""
import os
from typing import Optional, List, Dict, Any
from textwrap import dedent

import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.utils.log import logger

from .schemas import ExpertPerspective


# =============================================================================
# Expert Configurations
# =============================================================================

EXPERT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "technical": {
        "name": "Technical Expert",
        "role": "Senior ML Researcher",
        "perspective": "Deep technical analysis and methodology critique",
        "focus_areas": [
            "Algorithm design and complexity",
            "Benchmark validity and methodology",
            "Technical limitations and constraints",
            "Implementation challenges",
            "Mathematical foundations",
        ],
        "questions_to_ask": [
            "What are the underlying algorithms and their complexity?",
            "Are the benchmarks and evaluations rigorous?",
            "What are the technical limitations not mentioned?",
            "How does this compare to state-of-the-art technically?",
            "What implementation details are missing?",
        ],
    },
    "industry": {
        "name": "Industry Expert",
        "role": "VP of Engineering at AI Company",
        "perspective": "Practical applications and business implications",
        "focus_areas": [
            "Production readiness and scalability",
            "Cost and resource requirements",
            "Integration challenges",
            "Competitive landscape",
            "Market timing and adoption",
        ],
        "questions_to_ask": [
            "Is this production-ready or still research?",
            "What are the real-world deployment costs?",
            "How does this fit into existing systems?",
            "What's the competitive advantage?",
            "Who are the key players and what's their strategy?",
        ],
    },
    "skeptic": {
        "name": "Skeptic Expert",
        "role": "Critical Research Scientist",
        "perspective": "Challenge claims and identify weaknesses",
        "focus_areas": [
            "Reproducibility concerns",
            "Overhyped claims vs reality",
            "Hidden assumptions and biases",
            "Methodological flaws",
            "Conflicting evidence",
        ],
        "questions_to_ask": [
            "What claims are not well-supported by evidence?",
            "Are there reproducibility issues?",
            "What assumptions are hidden in this research?",
            "What are critics saying about this?",
            "Is this overhyped relative to actual capabilities?",
        ],
    },
    "futurist": {
        "name": "Futurist Expert",
        "role": "Technology Trend Analyst",
        "perspective": "Future implications and emerging directions",
        "focus_areas": [
            "Long-term trajectory",
            "Emerging research directions",
            "Potential breakthroughs",
            "Societal implications",
            "Regulatory landscape",
        ],
        "questions_to_ask": [
            "What's the 5-year trajectory for this field?",
            "What emerging directions are most promising?",
            "What could be the next breakthrough?",
            "What are the societal and ethical implications?",
            "How might regulation affect this?",
        ],
    },
    "academic": {
        "name": "Academic Expert",
        "role": "University Professor",
        "perspective": "Academic rigor and research context",
        "focus_areas": [
            "Literature positioning",
            "Novel contributions",
            "Research methodology",
            "Citation networks",
            "Educational value",
        ],
        "questions_to_ask": [
            "How does this relate to the broader literature?",
            "What's truly novel vs incremental?",
            "Is the methodology sound?",
            "Who are the key researchers in this area?",
            "What are the open research questions?",
        ],
    },
}


# =============================================================================
# Domain Expert Agent
# =============================================================================

class DomainExpertAgent:
    """
    Domain Expert Agent - Provides specialized perspective on research.
    
    Each expert type has a unique viewpoint and focuses on different
    aspects of the research findings.
    """
    
    def __init__(
        self,
        expert_type: str,
        model_id: str = "gpt-5-mini-2025-08-07",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
    ):
        """
        Initialize Domain Expert Agent.
        
        Args:
            expert_type: Type of expert (technical, industry, skeptic, futurist, academic)
            model_id: LLM model ID
            api_base: LiteLLM API base URL
            api_key: LiteLLM API key
            temperature: Model temperature
        """
        if expert_type not in EXPERT_CONFIGS:
            raise ValueError(f"Unknown expert type: {expert_type}. Available: {list(EXPERT_CONFIGS.keys())}")
        
        self.expert_type = expert_type
        self.config = EXPERT_CONFIGS[expert_type]
        self.model_id = model_id
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.temperature = temperature
        
        self._agent: Optional[Agent] = None
    
    @property
    def agent(self) -> Agent:
        """Lazy initialization of expert agent"""
        if self._agent is None:
            if not self.api_base or not self.api_key:
                raise ValueError("LITELLM_API_BASE and LITELLM_API_KEY must be set")
            
            logger.info(f"Creating {self.config['name']} with model: {self.model_id}")
            
            self._agent = Agent(
                name=self.config['name'],
                model=LiteLLM(
                    id=self.model_id,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    top_p=None,  # Claude doesn't accept both temperature and top_p
                ),
                description=f"{self.config['role']} providing {self.config['perspective']}",
                instructions=self._get_instructions(),
                output_schema=ExpertPerspective,
                markdown=True,
                debug_mode=True,
                debug_level=2,
            )
        
        return self._agent
    
    def _get_instructions(self) -> List[str]:
        """Get expert-specific instructions"""
        focus_areas = "\n".join(f"- {area}" for area in self.config['focus_areas'])
        questions = "\n".join(f"- {q}" for q in self.config['questions_to_ask'])
        
        return [
            dedent(f"""
                You are a {self.config['role']}, providing expert analysis from a 
                {self.config['perspective']} perspective.
                
                ## Your Focus Areas
                {focus_areas}
                
                ## Questions You Should Address
                {questions}
                
                ## Analysis Guidelines
                
                1. **Be Specific**: Provide concrete insights, not generic observations
                2. **Use Evidence**: Reference specific findings when making claims
                3. **Acknowledge Uncertainty**: Note when you're speculating vs. certain
                4. **Add Value**: Provide insights that wouldn't be obvious to non-experts
                5. **Be Critical**: Don't just summarize - analyze and critique
                
                ## Output Format
                
                Provide:
                1. A summary of your expert perspective (2-3 sentences)
                2. Key insights from your viewpoint (3-5 specific points)
                3. Concerns or criticisms you have (2-4 points)
                4. Recommendations based on your expertise (2-3 actionable items)
                5. Confidence score (1-5) in your analysis
            """).strip(),
        ]
    
    def analyze(
        self,
        findings: List[Dict[str, Any]],
        query: str,
        context: Optional[str] = None,
    ) -> ExpertPerspective:
        """
        Analyze research findings from expert perspective.
        
        Args:
            findings: List of research findings
            query: The original research query
            context: Optional additional context
            
        Returns:
            ExpertPerspective: Expert's analysis
        """
        logger.info(f"[{self.config['name']}] Analyzing {len(findings)} findings...")
        
        # Format findings
        findings_text = self._format_findings(findings)
        
        prompt = f"""
Analyze the following research findings from your expert perspective.

**Research Query:** {query}

**Your Role:** {self.config['role']}
**Your Perspective:** {self.config['perspective']}

{f"**Additional Context:** {context}" if context else ""}

**Research Findings:**
{findings_text}

---

Provide your expert analysis focusing on your specialized areas of expertise.
What insights, concerns, and recommendations do you have?
        """.strip()
        
        response = self.agent.run(prompt)
        
        if isinstance(response.content, ExpertPerspective):
            perspective = response.content
        elif isinstance(response.content, dict):
            perspective = ExpertPerspective(**response.content)
        else:
            # Fallback
            perspective = ExpertPerspective(
                expert_type=self.expert_type,
                perspective_summary="Analysis could not be fully parsed.",
                key_insights=[],
                concerns=[],
                recommendations=[],
                confidence_score=2,
            )
        
        # Ensure expert_type is set
        perspective.expert_type = self.expert_type
        
        logger.info(f"[{self.config['name']}] Analysis complete (confidence: {perspective.confidence_score})")
        
        return perspective
    
    def _format_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Format findings for analysis prompt"""
        lines = []
        
        for i, finding in enumerate(findings[:15], 1):  # Limit to 15 most relevant
            content = finding.get("content", "")[:400]
            source = finding.get("source_title", "") or finding.get("source_url", "")
            
            lines.append(f"{i}. {content}")
            if source:
                lines.append(f"   Source: {source}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_expert_agent(expert_type: str, **kwargs) -> DomainExpertAgent:
    """
    Factory function to create a domain expert agent.
    
    Args:
        expert_type: Type of expert (technical, industry, skeptic, futurist, academic)
        **kwargs: Additional arguments passed to DomainExpertAgent
        
    Returns:
        DomainExpertAgent: Configured expert agent
    """
    return DomainExpertAgent(expert_type=expert_type, **kwargs)


def create_expert_panel(
    expert_types: Optional[List[str]] = None,
    **kwargs,
) -> List[DomainExpertAgent]:
    """
    Create a panel of domain experts for multi-perspective analysis.
    
    Args:
        expert_types: List of expert types (default: technical, industry, skeptic)
        **kwargs: Additional arguments passed to each expert
        
    Returns:
        List of configured expert agents
    """
    if expert_types is None:
        expert_types = ["technical", "industry", "skeptic"]
    
    return [create_expert_agent(et, **kwargs) for et in expert_types]


def get_multi_perspective_analysis(
    findings: List[Dict[str, Any]],
    query: str,
    expert_types: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, ExpertPerspective]:
    """
    Get analysis from multiple expert perspectives.
    
    Args:
        findings: Research findings to analyze
        query: Original research query
        expert_types: List of expert types to consult
        **kwargs: Additional arguments for expert creation
        
    Returns:
        Dict mapping expert type to their perspective
    """
    experts = create_expert_panel(expert_types, **kwargs)
    
    perspectives = {}
    for expert in experts:
        try:
            perspective = expert.analyze(findings, query)
            perspectives[expert.expert_type] = perspective
        except Exception as e:
            logger.error(f"Expert {expert.expert_type} failed: {e}")
    
    return perspectives


def list_expert_types() -> List[Dict[str, str]]:
    """
    List available expert types and their descriptions.
    
    Returns:
        List of expert type information
    """
    return [
        {
            "type": expert_type,
            "name": config["name"],
            "role": config["role"],
            "perspective": config["perspective"],
        }
        for expert_type, config in EXPERT_CONFIGS.items()
    ]


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Domain Expert Agents Test ===\n")
    
    # List available experts
    print("📋 Available Expert Types:")
    for expert in list_expert_types():
        print(f"   • {expert['type']}: {expert['name']}")
        print(f"     Role: {expert['role']}")
        print(f"     Focus: {expert['perspective']}")
        print()
    
    # Check API config
    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not api_base or not api_key:
        print("❌ LITELLM_API_BASE and LITELLM_API_KEY must be set for full test")
        exit(0)
    
    print(f"✅ API Base: {api_base}")
    
    # Create an expert
    expert = create_expert_agent("technical")
    print(f"\n✅ Created {expert.config['name']}")
    
    # Test with sample findings (no actual LLM call for quick test)
    print("\n✅ Domain expert agents ready!")
    print("Note: Full analysis requires LLM call - skipped for quick test")

