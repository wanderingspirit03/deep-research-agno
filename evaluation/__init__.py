"""
Evaluation Layer - Deep Research Swarm Evaluation Tools

Provides evaluation and human-in-the-loop capabilities:
- slack_hitl: Thin wrapper for HITL integration
"""
from .slack_hitl import run_hitl_review, SlackHitlWrapper

__all__ = [
    "run_hitl_review",
    "SlackHitlWrapper",
]
