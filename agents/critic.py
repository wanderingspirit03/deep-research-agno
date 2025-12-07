"""
Critic Agent - Evaluates research quality and identifies gaps.

The Critic Agent:
1. Evaluates the quality and coverage of research findings
2. Identifies gaps and missing perspectives
3. Recommends follow-up queries to fill gaps
4. Determines when research is ready for synthesis
5. Reviews draft reports for improvements
6. Optionally escalates to human reviewers via HITL

HITL (Human-in-the-Loop) Integration:
- When enabled, low-confidence evaluations can be escalated to human reviewers
- Supports configurable confidence thresholds
- Merges human feedback with LLM evaluation
"""
import os
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from textwrap import dedent

import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.utils.log import logger

from config import config
from infrastructure.observability import observe
from .schemas import (
    CriticEvaluation,
    DraftCritique,
    GapAnalysis,
    QualityFinding,
)

if TYPE_CHECKING:
    from .hitl_agent import HitlAgent, HitlResult


# =============================================================================
# Critic Agent
# =============================================================================

class CriticAgent:
    """
    Critic Agent - Evaluates research quality and identifies gaps.
    
    Acts as a rigorous academic reviewer to ensure research quality
    meets PhD-level standards before synthesis.
    
    Supports optional HITL (Human-in-the-Loop) escalation for low-confidence
    evaluations or forced human review.
    """
    
    def __init__(
        self,
        model_id: str = "gpt-5-2025-08-07",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,  # Low for consistent evaluation
        quality_threshold: int = 80,
        hitl_enabled: bool = False,
        hitl_agent: Optional["HitlAgent"] = None,
        hitl_confidence_threshold: float = 0.7,
    ):
        """
        Initialize Critic Agent.
        
        Args:
            model_id: LLM model ID
            api_base: LiteLLM API base URL
            api_key: LiteLLM API key
            temperature: Model temperature (lower = more consistent)
            quality_threshold: Minimum score (0-100) to pass evaluation
            hitl_enabled: Whether HITL escalation is enabled
            hitl_agent: Pre-initialized HITL agent (lazy-created if None)
            hitl_confidence_threshold: Score range that triggers HITL (e.g., 60-80)
        """
        self.model_id = model_id
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.temperature = temperature
        self.quality_threshold = quality_threshold
        
        # HITL integration
        self.hitl_enabled = hitl_enabled
        self._hitl_agent = hitl_agent
        self.hitl_confidence_threshold = hitl_confidence_threshold
        
        # Lazy-initialized agents
        self._evaluation_agent: Optional[Agent] = None
        self._critique_agent: Optional[Agent] = None
    
    @property
    def hitl_agent(self) -> Optional["HitlAgent"]:
        """Lazy initialization of HITL agent."""
        if self._hitl_agent is None and self.hitl_enabled:
            try:
                from .hitl_agent import HitlAgent
                self._hitl_agent = HitlAgent()
                logger.info("[Critic] HITL agent initialized")
            except Exception as e:
                logger.warning(f"[Critic] Could not initialize HITL agent: {e}")
                self.hitl_enabled = False
        return self._hitl_agent
    
    @property
    def evaluation_agent(self) -> Agent:
        """Agent for evaluating research findings"""
        if self._evaluation_agent is None:
            if not self.api_base or not self.api_key:
                raise ValueError("LITELLM_API_BASE and LITELLM_API_KEY must be set")
            
            logger.info(f"Creating Critic Evaluation Agent with model: {self.model_id}")
            
            self._evaluation_agent = Agent(
                name="Research Critic",
                model=LiteLLM(
                    id=self.model_id,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    top_p=None,  # Claude doesn't accept both temperature and top_p
                ),
                description="Rigorous academic critic that evaluates research quality and identifies gaps.",
                instructions=self._get_evaluation_instructions(),
                output_schema=CriticEvaluation,
                markdown=True,
                debug_mode=True,
                debug_level=2,
            )
        
        return self._evaluation_agent
    
    @property
    def critique_agent(self) -> Agent:
        """Agent for critiquing draft reports"""
        if self._critique_agent is None:
            if not self.api_base or not self.api_key:
                raise ValueError("LITELLM_API_BASE and LITELLM_API_KEY must be set")
            
            self._critique_agent = Agent(
                name="Draft Critic",
                model=LiteLLM(
                    id=self.model_id,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    top_p=None,  # Claude doesn't accept both temperature and top_p
                ),
                description="Academic editor that critiques draft reports for quality and accuracy.",
                instructions=self._get_critique_instructions(),
                output_schema=DraftCritique,
                markdown=True,
                debug_mode=True,
                debug_level=2,
            )
        
        return self._critique_agent
    
    def _get_evaluation_instructions(self) -> List[str]:
        """Instructions for research evaluation"""
        return [
            dedent(f"""
                You are a rigorous academic research critic with PhD-level expertise.
                Your job is to evaluate research quality and identify gaps that need filling.
                
                ## Evaluation Criteria
                
                ### 1. Coverage Analysis (0-100)
                - Does the research address all key aspects of the query?
                - Are foundational concepts explained?
                - Is the current state of the art covered?
                - Are future directions discussed?
                - Score guide: <40 = major gaps, 40-60 = adequate, 60-80 = good, 80+ = comprehensive
                
                ### 2. Source Quality (0-100)
                - Are sources authoritative (peer-reviewed, institutional, industry leaders)?
                - Is there a mix of source types (academic + industry + news)?
                - Are sources recent and relevant?
                - Are claims properly attributed?
                - Score guide: <40 = poor sources, 40-60 = adequate, 60-80 = good, 80+ = excellent
                
                ### 3. Evidence Strength (0-100)
                - Are claims supported by specific data, statistics, or citations?
                - Is there consensus among sources?
                - Are conflicting views acknowledged?
                - Score guide: <40 = weak evidence, 40-60 = adequate, 60-80 = strong, 80+ = rigorous
                
                ### 4. Balance (0-100)
                - Are multiple perspectives represented?
                - Are limitations and criticisms included?
                - Is the analysis objective rather than promotional?
                
                ## Gap Identification
                
                For each gap identified, provide:
                1. Clear description of what's missing
                2. Importance rating (1-5)
                3. Specific search queries to fill the gap
                4. Types of sources that would help
                
                ## Decision Making
                
                Set `ready_for_synthesis = true` only if:
                - Overall score >= {self.quality_threshold}
                - No critical gaps with importance >= 4
                - Coverage score >= 70
                
                Provide a clear recommendation:
                - "synthesize" - Ready for final report
                - "continue" - Need more research on specific areas
                - "refocus" - Research is off-track, need new approach
            """).strip(),
        ]
    
    def _get_critique_instructions(self) -> List[str]:
        """Instructions for draft report critique"""
        return [
            dedent("""
                You are an academic editor reviewing a research report draft.
                Provide detailed, constructive criticism to improve quality.
                
                ## Evaluation Areas
                
                ### Structure (0-100)
                - Logical flow of sections
                - Clear introduction and conclusion
                - Appropriate section headings
                - Smooth transitions
                
                ### Clarity (0-100)
                - Clear, precise language
                - Technical terms explained
                - Accessible to target audience
                - No ambiguous statements
                
                ### Completeness (0-100)
                - All key topics covered
                - Sufficient depth on main points
                - Important caveats included
                - Future directions discussed
                
                ### Citations (0-100)
                - Claims properly attributed
                - Sources clearly referenced
                - Citation format consistent
                - No unsupported claims
                
                ## Issue Identification
                
                Be specific about:
                1. **Factual Issues**: Claims that may be incorrect or unsupported
                2. **Structural Issues**: Problems with organization
                3. **Missing Elements**: Important content that's absent
                
                ## Specific Edits
                
                Provide actionable edits like:
                - "Add citation for claim about X in paragraph Y"
                - "Expand section on Z with more technical detail"
                - "Reorder sections to improve flow: A, B, C → A, C, B"
                
                ## Publication Readiness
                
                Set `ready_for_publication = true` only if:
                - Overall quality >= 80
                - No unresolved factual issues
                - All critical sections present
            """).strip(),
        ]
    
    @observe(name="critic.evaluate")
    def evaluate(
        self,
        findings: List[Dict[str, Any]],
        original_query: str,
        iteration: int = 1,
        force_hitl: bool = False,
    ) -> CriticEvaluation:
        """
        Evaluate research findings for quality and completeness.
        
        Args:
            findings: List of research findings (dicts with content, source, etc.)
            original_query: The original research query
            iteration: Current research iteration number
            force_hitl: Force human review regardless of score
            
        Returns:
            CriticEvaluation: Detailed evaluation with scores and gaps
        """
        logger.info(f"Evaluating {len(findings)} findings for: {original_query[:50]}...")
        
        # Step 1: Run LLM evaluation first
        evaluation = self._llm_evaluate(findings, original_query, iteration)
        
        # Step 2: Decide if HITL is needed
        needs_hitl = (
            force_hitl or
            (self.hitl_enabled and self._should_escalate(evaluation, original_query))
        )
        
        # Step 3: If HITL needed, get human feedback
        if needs_hitl and self.hitl_agent:
            logger.info("[Critic] Escalating to HITL for human review")
            try:
                hitl_result = self._run_hitl(evaluation, findings, original_query)
                if hitl_result:
                    evaluation = self._merge_hitl_feedback(evaluation, hitl_result)
                    logger.info(f"[Critic] HITL feedback merged, new score: {evaluation.overall_score}")
            except Exception as e:
                logger.warning(f"[Critic] HITL escalation failed: {e}")
        
        logger.info(
            f"Evaluation complete: score={evaluation.overall_score}, "
            f"ready={evaluation.ready_for_synthesis}, gaps={len(evaluation.critical_gaps)}"
        )
        
        return evaluation
    
    def _llm_evaluate(
        self,
        findings: List[Dict[str, Any]],
        original_query: str,
        iteration: int,
    ) -> CriticEvaluation:
        """Run LLM-based evaluation (internal method)."""
        # Format findings for evaluation
        findings_text = self._format_findings(findings)
        
        prompt = f"""
Evaluate the following research findings for the query.

**Original Query:** {original_query}

**Research Iteration:** {iteration}

**Number of Findings:** {len(findings)}

**Findings:**
{findings_text}

---

Provide a comprehensive evaluation including:
1. Scores for coverage, source quality, evidence strength, and balance
2. List of strengths and weaknesses
3. Critical gaps that need to be filled
4. Follow-up queries for the next iteration
5. Decision on whether research is ready for synthesis
        """.strip()
        
        response = self.evaluation_agent.run(prompt)
        
        # Parse response
        if isinstance(response.content, CriticEvaluation):
            evaluation = response.content
        elif isinstance(response.content, dict):
            evaluation = CriticEvaluation(**response.content)
        else:
            # Fallback evaluation
            logger.warning("Could not parse critic evaluation, using fallback")
            evaluation = self._create_fallback_evaluation(findings)
        
        return evaluation
    
    def _should_escalate(self, evaluation: CriticEvaluation, query: str) -> bool:
        """
        Determine if evaluation should be escalated to HITL.
        
        Escalation criteria:
        - Score is in "uncertain" range (not too low to be obvious failure,
          not high enough to pass)
        - Query is in a forced HITL domain
        """
        # Check if query is in a forced domain
        force_domains = config.hitl.force_domains if hasattr(config, 'hitl') else []
        query_lower = query.lower()
        for domain in force_domains:
            if domain.lower() in query_lower:
                logger.info(f"[Critic] Force HITL for domain: {domain}")
                return True
        
        # Check if score is in uncertain range
        min_uncertain = 60  # Below this, clearly needs more research
        max_uncertain = self.quality_threshold  # At or above this, passes
        
        if min_uncertain <= evaluation.overall_score < max_uncertain:
            logger.info(
                f"[Critic] Score {evaluation.overall_score} in uncertain range "
                f"({min_uncertain}-{max_uncertain}), escalating to HITL"
            )
            return True
        
        return False
    
    def _run_hitl(
        self,
        evaluation: CriticEvaluation,
        findings: List[Dict[str, Any]],
        query: str,
    ) -> Optional["HitlResult"]:
        """Run HITL review and return result."""
        if not self.hitl_agent:
            return None
        
        # Build evaluation summary for human review
        summary = f"""
## AI Evaluation Summary

**Overall Score:** {evaluation.overall_score}/100
**Coverage:** {evaluation.coverage_score}/100
**Source Quality:** {evaluation.source_quality_score}/100
**Evidence Strength:** {evaluation.evidence_strength_score}/100

### Strengths
{chr(10).join('- ' + s for s in (evaluation.strengths or [])[:5])}

### Weaknesses
{chr(10).join('- ' + w for w in (evaluation.weaknesses or [])[:5])}

### Recommendation
{evaluation.recommendation}

**Ready for synthesis:** {evaluation.ready_for_synthesis}
"""
        
        # Format some findings for context
        findings_summary = self._format_findings(findings[:10])
        
        question = f"Research quality evaluation for: {query}"
        response = f"{summary}\n\n## Sample Findings\n{findings_summary}"
        
        return self.hitl_agent.run_review(
            question=question,
            response=response,
            mode_hint="qualitative_review",
            meta={"query": query, "findings_count": len(findings)},
        )
    
    def _merge_hitl_feedback(
        self,
        evaluation: CriticEvaluation,
        hitl_result: "HitlResult",
    ) -> CriticEvaluation:
        """Merge HITL feedback into evaluation."""
        # Adjust score based on HITL result
        hitl_score = hitl_result.score * 10  # HITL returns 0-10, convert to 0-100
        
        # Weighted average: 60% LLM, 40% human
        new_score = int(0.6 * evaluation.overall_score + 0.4 * hitl_score)
        
        # Add HITL feedback to weaknesses if not approved
        new_weaknesses = list(evaluation.weaknesses or [])
        if not hitl_result.approved and hitl_result.feedback:
            new_weaknesses.append(f"[HITL] {hitl_result.feedback[:200]}")
        
        # Update ready_for_synthesis based on combined assessment
        new_ready = hitl_result.approved and new_score >= self.quality_threshold
        
        # Create updated evaluation
        return CriticEvaluation(
            overall_score=new_score,
            coverage_score=evaluation.coverage_score,
            source_quality_score=evaluation.source_quality_score,
            evidence_strength_score=evaluation.evidence_strength_score,
            balance_score=evaluation.balance_score,
            strengths=evaluation.strengths,
            weaknesses=new_weaknesses,
            critical_gaps=evaluation.critical_gaps,
            follow_up_queries=evaluation.follow_up_queries,
            ready_for_synthesis=new_ready,
            recommendation=evaluation.recommendation if new_ready else "continue",
        )
    
    @observe(name="critic.review_draft")
    def review_draft(self, draft: str, original_query: str) -> DraftCritique:
        """
        Review a draft report for improvements.
        
        Args:
            draft: The draft report text
            original_query: The original research query
            
        Returns:
            DraftCritique: Detailed critique with improvement suggestions
        """
        logger.info(f"Reviewing draft ({len(draft)} chars) for: {original_query[:50]}...")
        
        prompt = f"""
Review the following research report draft.

**Original Query:** {original_query}

**Draft Report:**
{draft}

---

Provide a detailed critique including:
1. Scores for structure, clarity, completeness, and citations
2. Any factual issues that need correction
3. Structural improvements needed
4. Missing elements to add
5. Specific edits to make
6. Whether the report is ready for publication
        """.strip()
        
        response = self.critique_agent.run(prompt)
        
        if isinstance(response.content, DraftCritique):
            critique = response.content
        elif isinstance(response.content, dict):
            critique = DraftCritique(**response.content)
        else:
            logger.warning("Could not parse draft critique, using fallback")
            critique = DraftCritique(
                overall_quality=70,
                structure_score=70,
                clarity_score=70,
                completeness_score=70,
                citation_score=70,
                ready_for_publication=False,
            )
        
        logger.info(f"Critique complete: quality={critique.overall_quality}")
        
        return critique
    
    def _format_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Format findings for evaluation prompt"""
        lines = []
        
        for i, finding in enumerate(findings, 1):
            content = finding.get("content", "")
            source = finding.get("source_url", "") or finding.get("source", "")
            title = finding.get("source_title", "")
            search_type = finding.get("search_type", "general")
            
            lines.append(f"### Finding {i}")
            lines.append(f"**Content:** {content[:500]}...")
            lines.append(f"**Source:** {title or source}")
            lines.append(f"**Type:** {search_type}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_fallback_evaluation(self, findings: List[Dict[str, Any]]) -> CriticEvaluation:
        """Create a fallback evaluation when parsing fails"""
        num_findings = len(findings)
        
        # Estimate scores based on quantity
        base_score = min(50 + num_findings * 5, 75)
        
        return CriticEvaluation(
            overall_score=base_score,
            coverage_score=base_score,
            source_quality_score=base_score - 10,
            evidence_strength_score=base_score - 5,
            balance_score=base_score - 5,
            strengths=["Multiple sources gathered"],
            weaknesses=["Evaluation could not fully parse findings"],
            critical_gaps=[
                GapAnalysis(
                    gap_description="Unable to fully assess research gaps",
                    importance=3,
                    suggested_queries=["Continue research on core topic"],
                )
            ],
            follow_up_queries=[],
            ready_for_synthesis=num_findings >= 10,
            recommendation="continue" if num_findings < 10 else "synthesize",
        )
    
    def quick_assess(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Quick assessment without full LLM evaluation.
        
        Useful for fast iteration decisions based on heuristics.
        
        Args:
            findings: List of research findings
            
        Returns:
            Dict with quick assessment metrics
        """
        num_findings = len(findings)
        
        # Count unique sources
        sources = set()
        academic_count = 0
        
        for f in findings:
            source = f.get("source_url", "") or f.get("source", "")
            if source:
                sources.add(source)
            if f.get("search_type") == "academic":
                academic_count += 1
        
        # Calculate metrics
        source_diversity = len(sources) / max(num_findings, 1)
        academic_ratio = academic_count / max(num_findings, 1)
        
        # Estimate quality
        estimated_score = min(
            40 +  # Base
            num_findings * 3 +  # More findings = better
            len(sources) * 2 +  # Source diversity
            academic_count * 5,  # Academic sources valuable
            95
        )
        
        return {
            "num_findings": num_findings,
            "num_sources": len(sources),
            "academic_ratio": academic_ratio,
            "source_diversity": source_diversity,
            "estimated_score": estimated_score,
            "needs_more_research": num_findings < 10 or len(sources) < 5,
            "needs_academic": academic_ratio < 0.3,
        }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Critic Agent Test ===\n")
    
    # Check API config
    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not api_base or not api_key:
        print("❌ LITELLM_API_BASE and LITELLM_API_KEY must be set")
        exit(1)
    
    print(f"✅ API Base: {api_base}")
    
    # Create critic
    critic = CriticAgent()
    
    # Test with sample findings
    test_findings = [
        {
            "content": "GPT-5 achieved 92.3% on MMLU benchmark, surpassing previous models.",
            "source_url": "https://openai.com/research/gpt5",
            "source_title": "GPT-5 Technical Report",
            "search_type": "general",
        },
        {
            "content": "Transformer architecture limitations include quadratic attention complexity.",
            "source_url": "https://arxiv.org/abs/2024.12345",
            "source_title": "Attention Mechanisms Survey",
            "search_type": "academic",
        },
    ]
    
    # Test quick assess
    print("\n📊 Quick Assessment:")
    quick = critic.quick_assess(test_findings)
    for k, v in quick.items():
        print(f"   {k}: {v}")
    
    print("\n✅ Critic agent initialized successfully!")
    print("Note: Full evaluation requires LLM call - skipped for quick test")

