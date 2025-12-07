"""
Slack HITL Wrapper - Thin convenience wrapper for HITL integration.

This module provides a simplified interface for integrating HITL into
the Deep Research Swarm's quality evaluation pipeline.
"""
import os
from typing import Any, Dict, List, Optional

from agno.utils.log import logger

from config import config
from agents.hitl_agent import HitlAgent, HitlResult


class SlackHitlWrapper:
    """
    Wrapper for HITL integration with simplified interface.
    
    Provides convenience methods for common HITL operations
    in the context of research quality evaluation.
    """
    
    def __init__(self, force_enabled: bool = False):
        """
        Initialize HITL wrapper.
        
        Args:
            force_enabled: Force HITL even if not enabled in config
        """
        self.enabled = config.hitl.enabled or force_enabled
        self._agent: Optional[HitlAgent] = None
        
        if self.enabled and not config.hitl.slack_token:
            logger.warning("[HITL] Enabled but SLACK_TOKEN not set - disabling")
            self.enabled = False
    
    @property
    def agent(self) -> Optional[HitlAgent]:
        """Lazy initialization of HITL agent."""
        if self._agent is None and self.enabled:
            try:
                self._agent = HitlAgent()
                logger.info("[HITL] Agent initialized successfully")
            except Exception as e:
                logger.error(f"[HITL] Failed to initialize agent: {e}")
                self.enabled = False
        return self._agent
    
    def review_findings(
        self,
        findings: List[Dict[str, Any]],
        query: str,
        evaluation_summary: str,
        domain_hint: Optional[str] = None,
    ) -> HitlResult:
        """
        Request human review of research findings.
        
        Args:
            findings: List of research findings to review
            query: Original research query
            evaluation_summary: AI-generated evaluation summary
            domain_hint: Optional domain hint for routing
            
        Returns:
            HitlResult: Human review result
        """
        if not self.enabled or not self.agent:
            raise RuntimeError("HITL is not enabled or agent not initialized")
        
        # Format findings for review
        findings_text = self._format_findings_for_review(findings)
        
        question = f"Research quality evaluation for: {query}"
        response = f"""
## AI Evaluation Summary
{evaluation_summary}

## Research Findings ({len(findings)} total)
{findings_text}
"""
        
        return self.agent.run_review(
            question=question,
            response=response,
            mode_hint="qualitative_review",
            domain_hint=domain_hint,
            meta={"query": query, "findings_count": len(findings)},
        )
    
    def review_draft(
        self,
        draft: str,
        query: str,
        critique_summary: Optional[str] = None,
        domain_hint: Optional[str] = None,
    ) -> HitlResult:
        """
        Request human review of a draft report.
        
        Args:
            draft: Draft report text
            query: Original research query
            critique_summary: Optional AI critique summary
            domain_hint: Optional domain hint for routing
            
        Returns:
            HitlResult: Human review result
        """
        if not self.enabled or not self.agent:
            raise RuntimeError("HITL is not enabled or agent not initialized")
        
        question = f"Draft report review for: {query}"
        response = draft
        
        if critique_summary:
            response = f"""
## AI Critique Summary
{critique_summary}

## Draft Report
{draft}
"""
        
        return self.agent.run_review(
            question=question,
            response=response,
            mode_hint="qualitative_review",
            domain_hint=domain_hint,
            meta={"query": query, "draft_length": len(draft)},
        )
    
    def get_score(
        self,
        content: str,
        question: str,
        domain_hint: Optional[str] = None,
    ) -> HitlResult:
        """
        Request a human score (0-10) for content.
        
        Args:
            content: Content to score
            question: Context question
            domain_hint: Optional domain hint for routing
            
        Returns:
            HitlResult: Human score result
        """
        if not self.enabled or not self.agent:
            raise RuntimeError("HITL is not enabled or agent not initialized")
        
        return self.agent.run_review(
            question=question,
            response=content,
            mode_hint="score_answer_0_to_10",
            domain_hint=domain_hint,
        )
    
    def _format_findings_for_review(self, findings: List[Dict[str, Any]]) -> str:
        """Format findings for human review."""
        lines = []
        
        for i, finding in enumerate(findings[:20], 1):  # Limit to 20 for readability
            content = finding.get("content", "")[:500]
            source = finding.get("source_title", "") or finding.get("source_url", "Unknown")
            search_type = finding.get("search_type", "general")
            
            lines.append(f"### Finding {i}: {source}")
            lines.append(f"**Type:** {search_type}")
            lines.append(f"{content}...")
            lines.append("")
        
        if len(findings) > 20:
            lines.append(f"*...and {len(findings) - 20} more findings*")
        
        return "\n".join(lines)


def run_hitl_review(
    content: str,
    question: str,
    mode: str = "review",
    domain_hint: Optional[str] = None,
    force_enabled: bool = False,
) -> Optional[HitlResult]:
    """
    Convenience function to run a HITL review.
    
    Args:
        content: Content to review
        question: Review question/context
        mode: Review mode - "review", "score", "select"
        domain_hint: Optional domain hint for routing
        force_enabled: Force HITL even if not enabled in config
        
    Returns:
        HitlResult if successful, None if HITL is disabled
    """
    wrapper = SlackHitlWrapper(force_enabled=force_enabled)
    
    if not wrapper.enabled:
        logger.info("[HITL] Skipping review - HITL not enabled")
        return None
    
    mode_hints = {
        "review": "qualitative_review",
        "score": "score_answer_0_to_10",
        "select": "select_best_answer",
        "open": "open_answer",
    }
    
    try:
        return wrapper.agent.run_review(
            question=question,
            response=content,
            mode_hint=mode_hints.get(mode, "qualitative_review"),
            domain_hint=domain_hint,
        )
    except Exception as e:
        logger.error(f"[HITL] Review failed: {e}")
        return None


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Slack HITL Wrapper Test ===\n")
    
    wrapper = SlackHitlWrapper()
    
    print(f"HITL Enabled: {wrapper.enabled}")
    print(f"Config enabled: {config.hitl.enabled}")
    print(f"Slack token set: {bool(config.hitl.slack_token)}")
    
    if wrapper.enabled:
        print("\n✅ HITL wrapper ready for use")
    else:
        print("\n❌ HITL not enabled - set HITL_ENABLED=true and SLACK_TOKEN in .env")
