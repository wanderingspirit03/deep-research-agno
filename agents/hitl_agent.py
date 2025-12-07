"""
Slack HITL (Human-in-the-Loop) Agent for Deep Research Swarm.

Reimplements the legacy CLI-based HITL and channel router as a callable agent:
- Domain classification and channel routing with smart defaults
- Slack tools for select_best / score / qualitative review / open answer
- Aggregation via LiteLLM + Agno Agent
- Polling every 2s (configurable) with timeout

Ported from deep-research-clone/agents/hitl_agent.py, converted to use LiteLLM.
"""
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import litellm
litellm.drop_params = True

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.tools.slack import SlackTools
from agno.utils.log import logger

from config import HitlChannel, HitlConfig, config
from infrastructure.observability import observe


class Domain(str, Enum):
    """Research domain for routing to appropriate Slack channels."""
    ECON = "econ"
    FIN_ANALYTICS = "fin_analytics"
    FORECASTING = "forecasting"
    SWE = "swe"
    GENERAL = "general"


@dataclass
class DomainClassification:
    """Result of domain classification for a research question."""
    domain: Domain
    confidence: float
    reasoning: str
    secondary_domains: List[Domain] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.6

    @property
    def is_multidomain(self) -> bool:
        return len(self.secondary_domains) > 0


@dataclass
class HitlResult:
    """Result from a HITL review."""
    channel: str
    classification: Dict[str, Any]
    mode_hint: Optional[str]
    result: Dict[str, Any]
    score: int
    raw_content: Any
    approved: bool = False
    feedback: str = ""


CLASSIFICATION_SYSTEM_PROMPT = """You are a research domain classifier for a Human-in-the-Loop feedback routing system.

Your task: Analyze a research question and classify it into ONE primary domain. Be decisive and prefer specific domains over "general".

Available domains:
- ECON: Macroeconomics, microeconomics, trade, labor markets, policy, inflation, growth
- FIN_ANALYTICS: Financial analytics, modeling, valuation, risk, portfolio construction, capital markets
- FORECASTING: Forecasting, demand planning, time-series, scenario planning, predictive modeling of future outcomes
- SWE: Software engineering, coding, architecture, systems design, debugging, dev practices
- GENERAL: Only for trivial factual questions OR highly interdisciplinary questions where NO domain is clearly primary (rare - use sparingly!)

Instructions:
1. Read the question carefully and identify key subject matter
2. PREFER a specific domain - choose where domain experts would add the most value
3. Even if a question touches multiple domains, pick the PRIMARY one (the most important expertise needed)
4. Use GENERAL only as a last resort for truly ambiguous or trivial questions
5. Assess confidence: HIGH (0.8-1.0), MEDIUM (0.6-0.79), LOW (0.0-0.59)
6. Note any secondary domains if the question is interdisciplinary
7. Explain your reasoning briefly

Return ONLY valid JSON with keys: domain (lowercase), confidence (float 0-1), reasoning (string), secondary_domains (list of lowercase domain strings).
"""


ROUTER_SYSTEM_PROMPT = """
You are a Human-in-the-Loop (HITL) review orchestrator that uses Slack as the human interface.
Your job:
- Read the evaluation input and decide which HUMAN FEEDBACK MODE is most appropriate.
- Call exactly ONE feedback tool that posts to Slack, waits for humans, aggregates, and returns structured feedback.
- Turn the tool result into a clear, concise textual response for the caller.

You receive a single prompt that contains everything you need:
- input.question (str)
- input.response_raw (str)
- input.mode_hint (str | null)
- slack_channel (str) → pass this as the channel argument when you call a tool.

You have FOUR feedback modes, each mapped to a specific tool:
1. select_best_answer  →  request_select_best_from_slack(question, answers, channel)
2. score_answer_0_to_10  →  request_score_from_slack(question, answer, channel)
3. qualitative_review  →  request_review_from_slack(question, draft, channel)
4. open_answer  →  request_open_answer_from_slack(question, channel)

Decision rules:
- If input.mode_hint is present and valid, RESPECT it and choose the matching tool.
- Otherwise:
  - If you can identify 2 or more distinct candidate answers/options within input.response_raw (bullets, numbering, ordinals, separated clauses, etc.) → use select_best_answer with those inferred options.
  - Else if the question asks for a numeric score/0-10/rating → use score_answer_0_to_10 with input.response_raw.
  - Else if input.response_raw is present → use qualitative_review with input.response_raw.
  - Else → use open_answer.

Tool outputs:
- request_select_best_from_slack returns: { "mode": "select_best", "winner_index": int, "rationale": str }
- request_score_from_slack returns: { "mode": "score", "average_score": float, "notes": str }
- request_review_from_slack returns: { "mode": "text_feedback", "feedback": str }
- request_open_answer_from_slack returns: { "mode": "open_answer", "answer": str }

Final response:
- Always produce a single, human-readable text answer for the caller.
- Summarize what humans decided (winner, score, feedback, or answer) without exposing tool names.
"""


def _safe_json_parse(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Safely parse JSON string, returning fallback on failure."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return fallback


def _extract_score(raw: Any) -> Optional[int]:
    """Try to pull a 0–10 score from structured or freeform text."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in ("score", "average_score", "winner_index", "rating"):
            if key in raw:
                val = raw.get(key)
                if isinstance(val, (int, float)) and 0 <= val <= 10:
                    return int(round(val))
        if "content" in raw:
            nested = _extract_score(raw.get("content"))
            if nested is not None:
                return nested
    text = str(raw)
    import re
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        try:
            num = float(match.group(1))
            if 0 <= num <= 10:
                return int(round(num))
        except Exception:
            pass
    return None


def _fallback_score_from_feedback(feedback: str) -> int:
    """If no explicit score is present, use a simple sentiment heuristic to pick 0 or 10."""
    text = feedback.lower()
    positive_markers = ("good", "great", "approve", "best", "winner", "ok", "pass", "accepted")
    negative_markers = ("bad", "fail", "reject", "wrong", "issue", "error", "bug", "no reply")
    if any(token in text for token in negative_markers):
        return 0
    if any(token in text for token in positive_markers):
        return 10
    return 0


def _build_channel_map(channels: List[HitlChannel]) -> Dict[str, HitlChannel]:
    """Build a mapping from domain string to HitlChannel."""
    return {ch.domain: ch for ch in channels}


class ChannelRouter:
    """Domain-based channel router using an LLM classifier."""

    def __init__(self, model: LiteLLM, hitl_config: HitlConfig):
        self.model = model
        self.hitl_config = hitl_config
        self.channel_map = _build_channel_map(hitl_config.smart_channels)
        self.classifier = Agent(
            name="DomainClassifier",
            model=self.model,
            tools=[],
            instructions=CLASSIFICATION_SYSTEM_PROMPT,
        )

    def classify_question_domain(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        domain_hint: Optional[Domain] = None,
    ) -> DomainClassification:
        """Classify a question into a research domain."""
        if domain_hint is not None:
            logger.info(f"[HITL] Domain hint provided: {domain_hint.value}")
            return DomainClassification(
                domain=domain_hint,
                confidence=0.95,
                reasoning="Domain explicitly provided by caller",
                secondary_domains=[],
            )

        prompt_parts = [f"Classify this research question:\n\nQuestion: {question}"]
        if context:
            context_str = json.dumps(context, indent=2, ensure_ascii=False)
            prompt_parts.append(f"\nAdditional context:\n{context_str}")
        prompt_parts.append("\nReturn JSON with domain, confidence, reasoning, secondary_domains.")
        prompt = "\n".join(prompt_parts)

        try:
            result = self.classifier.run(prompt)
            content = str(result.content or "").strip()
            data = _safe_json_parse(
                content,
                {
                    "domain": "general",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse classification response",
                    "secondary_domains": [],
                },
            )
            domain_str = data.get("domain", "general").lower()
            try:
                domain = Domain(domain_str)
            except ValueError:
                logger.warning(f"[HITL] Unknown domain '{domain_str}', defaulting to GENERAL")
                domain = Domain.GENERAL

            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            secondary_domains = []
            for sec_domain_str in data.get("secondary_domains", []):
                try:
                    secondary_domains.append(Domain(sec_domain_str.lower()))
                except ValueError:
                    pass

            classification = DomainClassification(
                domain=domain,
                confidence=confidence,
                reasoning=reasoning,
                secondary_domains=secondary_domains,
            )
            logger.info(
                f"[HITL] Domain={classification.domain.value} "
                f"(conf={classification.confidence:.2f}) reason={classification.reasoning}"
            )
            return classification
        except Exception as e:
            logger.error(f"[HITL] Classification failed: {e}")
            return DomainClassification(
                domain=Domain.GENERAL,
                confidence=0.5,
                reasoning=f"Classification error: {str(e)}",
                secondary_domains=[],
            )

    def get_channel_for_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        domain_hint: Optional[Domain] = None,
        fallback_channel: Optional[str] = None,
    ) -> Tuple[str, DomainClassification]:
        """Get the appropriate Slack channel for a question."""
        classification = self.classify_question_domain(question, context, domain_hint)
        channel = self._get_channel_for_domain(
            classification.domain,
            use_default_if_missing=True,
            fallback_channel=fallback_channel,
        )
        if not classification.is_confident:
            logger.info(
                f"[HITL] Low confidence ({classification.confidence:.2f}) "
                f"for domain {classification.domain.value}, using default"
            )
            channel = fallback_channel or self._default_channel()
        return channel, classification

    def _default_channel(self) -> str:
        """Get the default/general channel."""
        channel = self._get_channel_for_domain(Domain.GENERAL, use_default_if_missing=False)
        if channel:
            return channel
        # Fallback to configured default
        if self.hitl_config.default_channel:
            return self.hitl_config.default_channel
        raise ValueError("No default HITL channel configured")

    def _get_channel_for_domain(
        self,
        domain: Domain,
        use_default_if_missing: bool = True,
        fallback_channel: Optional[str] = None,
    ) -> str:
        """Get the channel ID for a specific domain."""
        channel = None
        if domain.value in self.channel_map:
            channel = self.channel_map[domain.value].channel_id
        if channel is None and use_default_if_missing:
            return fallback_channel or self.hitl_config.default_channel or ""
        if channel is None:
            raise ValueError(f"No channel configured for domain: {domain.value}")
        return channel


class HitlAgent:
    """
    HITL review agent that provides programmatic access to human review.
    
    Uses LiteLLM for model access (via Agno) and Slack for human interaction.
    """

    def __init__(self, hitl_config: Optional[HitlConfig] = None):
        """
        Initialize HITL Agent.
        
        Args:
            hitl_config: HITL configuration (uses global config if not provided)
        """
        self.config = hitl_config or config.hitl
        
        if not self.config.slack_token:
            raise ValueError("HITL requires SLACK_TOKEN to be set")
        
        # Initialize LiteLLM model (via Agno)
        api_base = os.getenv("LITELLM_API_BASE")
        api_key = os.getenv("LITELLM_API_KEY")
        
        if not api_base or not api_key:
            raise ValueError("HITL requires LITELLM_API_BASE and LITELLM_API_KEY")
        
        self.model = LiteLLM(
            id=self.config.model_id,
            api_base=api_base,
            api_key=api_key,
            temperature=self.config.temperature,
            top_p=None,  # Claude compatibility
        )
        
        self.slack_tools = SlackTools(token=self.config.slack_token)
        
        self.aggregator = Agent(
            name="HitlAggregator",
            model=self.model,
            tools=[],
            instructions="You return strict JSON only. Do not call tools.",
        )
        
        self.router = ChannelRouter(self.model, self.config)
        self.tools = self._build_tools()
        
        self.router_agent = Agent(
            name="SlackHitlEvaluator",
            model=self.model,
            tools=self.tools,
            instructions=ROUTER_SYSTEM_PROMPT,
        )

    @observe(name="hitl.run_review")
    def run_review(
        self,
        question: str,
        response: str,
        mode_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> HitlResult:
        """
        Run a HITL review for a given question and response.
        
        Args:
            question: The research question being reviewed
            response: The AI-generated response to review
            mode_hint: Optional hint for feedback mode (select_best, score, review, open)
            domain_hint: Optional hint for domain routing
            meta: Optional metadata for context
            
        Returns:
            HitlResult: Structured result from human review
        """
        hint_enum = None
        if domain_hint:
            try:
                hint_enum = Domain(domain_hint)
            except ValueError:
                logger.warning(f"[HITL] Unknown domain hint '{domain_hint}', ignoring.")

        channel, classification = self.router.get_channel_for_question(
            question=question,
            context={"response": response, "meta": meta},
            domain_hint=hint_enum,
            fallback_channel=None,
        )

        router_prompt = self._build_router_prompt(
            question=question,
            response=response,
            mode_hint=mode_hint,
            channel=channel,
            meta=meta,
        )
        
        logger.info(f"[HITL] Dispatching HITL run to channel {channel}")
        
        run_result = self.router_agent.run(
            router_prompt,
            context={
                "input": {
                    "question": question,
                    "response": response,
                    "mode_hint": mode_hint,
                    "meta": meta,
                },
                "slack_channel": channel,
            },
        )
        
        raw_content = run_result.content if run_result else None
        if isinstance(raw_content, dict):
            content = raw_content
        else:
            try:
                content = json.loads(raw_content) if raw_content else {}
            except Exception:
                content = {"text": str(raw_content)}

        score = _extract_score(content)
        if score is None:
            score = _fallback_score_from_feedback(str(raw_content or ""))

        # Determine if approved (score >= 7 or positive feedback)
        approved = score >= 7
        feedback = content.get("feedback", "") or content.get("text", "") or str(raw_content or "")

        return HitlResult(
            channel=channel,
            classification={
                "domain": classification.domain.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "secondary_domains": [d.value for d in classification.secondary_domains],
            },
            mode_hint=mode_hint,
            result=content,
            score=score,
            raw_content=raw_content,
            approved=approved,
            feedback=feedback,
        )

    def _build_router_prompt(
        self,
        question: str,
        response: str,
        mode_hint: Optional[str],
        channel: str,
        meta: Optional[Dict[str, Any]],
    ) -> str:
        """Build the prompt for the router agent."""
        payload = {
            "question": question,
            "response_raw": response,
            "mode_hint": mode_hint,
            "meta": meta,
            "slack_channel": channel,
        }
        pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        return (
            "Decide which single Slack feedback tool to call and then summarize the result.\n"
            "Use the provided data and do not ask follow-up questions.\n"
            "Input JSON:\n"
            f"{pretty}"
        )

    def _aggregate_with_prompt(self, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        """Run aggregation prompt and parse result."""
        try:
            result = self.aggregator.run(prompt)
            content = str(result.content or "").strip()
            return _safe_json_parse(content, fallback)
        except Exception as e:
            logger.error(f"[HITL] Aggregation failed: {e}")
            return fallback

    def _post_message(self, text: str, channel: str) -> Tuple[str, str]:
        """Post a message to Slack and return channel_id and thread_ts."""
        resp = self.slack_tools.send_message(channel=channel, text=text)
        data = json.loads(resp)
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["channel"], data["ts"]

    def _poll_human_replies(self, channel_id: str, thread_ts: str) -> List[dict]:
        """Poll for human replies in a Slack thread."""
        poll_interval = self.config.poll_interval_seconds
        timeout_seconds = self.config.timeout_seconds
        start = time.time()
        latest_seen = "0"
        collected: List[dict] = []
        
        while True:
            resp = self.slack_tools.client.conversations_replies(
                channel=channel_id, ts=thread_ts, limit=50
            )
            messages: List[dict] = resp.get("messages", []) or []
            replies: List[dict] = messages[1:] if len(messages) > 1 else []
            human_replies = [msg for msg in replies if not self._is_bot_message(msg)]

            if human_replies:
                newest = max((msg.get("ts") or "0") for msg in human_replies)
                collected = human_replies
                if newest > latest_seen:
                    return sorted(human_replies, key=lambda m: m.get("ts", "0"), reverse=True)
            if time.time() - start > timeout_seconds:
                return sorted(collected, key=lambda m: m.get("ts", "0"), reverse=True)
            time.sleep(poll_interval)

    def _build_tools(self):
        """Build the Slack feedback tools."""
        
        def request_select_best_from_slack(question: str, answers: List[str], channel: str) -> Dict[str, Any]:
            """Request humans to select the best answer from options."""
            if not answers or len(answers) < 2:
                raise ValueError("select_best requires at least two answers.")
            answers_block = "\n".join(f"{idx + 1}. {ans}" for idx, ans in enumerate(answers))
            text = (
                "*HITL: pick the best answer*\n"
                f"*Question*: {question}\n"
                f"*Answers*:\n{answers_block}\n"
                "Reply with the number of the best answer and any rationale."
            )
            channel_id, thread_ts = self._post_message(text, channel)
            replies = self._poll_human_replies(channel_id, thread_ts)
            replies_text = self._format_replies_for_prompt(replies)
            prompt = (
                "Select the winning answer based on human replies.\n"
                "Return JSON with keys: mode, winner_index, rationale.\n"
                "winner_index is 0-based.\n"
                f"Question: {question}\n"
                f"Answers: {answers}\n"
                f"Replies (newest first):\n{replies_text}"
            )
            fallback = {"mode": "select_best", "winner_index": 0, "rationale": "No human replies before timeout."}
            data = self._aggregate_with_prompt(prompt, fallback)
            data.setdefault("mode", "select_best")
            return data

        def request_score_from_slack(question: str, answer: str, channel: str) -> Dict[str, Any]:
            """Request humans to score an answer 0-10."""
            text = (
                "*HITL: score the answer (0–10)*\n"
                f"*Question*: {question}\n"
                f"*Answer:* ```{answer}```\n"
                "Reply with a score 0–10 and a short reason."
            )
            channel_id, thread_ts = self._post_message(text, channel)
            replies = self._poll_human_replies(channel_id, thread_ts)
            replies_text = self._format_replies_for_prompt(replies)
            prompt = (
                "Aggregate human scores.\n"
                "Return JSON with keys: mode, average_score, notes.\n"
                "If no numeric score was given, set average_score to null and explain in notes.\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Replies (newest first):\n{replies_text}"
            )
            fallback = {"mode": "score", "average_score": None, "notes": "No human replies before timeout."}
            data = self._aggregate_with_prompt(prompt, fallback)
            data.setdefault("mode", "score")
            return data

        def request_review_from_slack(question: str, draft: str, channel: str) -> Dict[str, Any]:
            """Request humans to qualitatively review a draft."""
            text = (
                "*HITL: qualitative review*\n"
                f"*Question*: {question}\n"
                f"*Draft:* ```{draft}```\n"
                "Reply with edits, gaps, or approval/rejection."
            )
            channel_id, thread_ts = self._post_message(text, channel)
            replies = self._poll_human_replies(channel_id, thread_ts)
            replies_text = self._format_replies_for_prompt(replies)
            prompt = (
                "Summarize human feedback for the draft.\n"
                "Return JSON with keys: mode, feedback.\n"
                "feedback should capture key edits/notes; if no replies, explain that.\n"
                f"Question: {question}\n"
                f"Draft: {draft}\n"
                f"Replies (newest first):\n{replies_text}"
            )
            fallback = {"mode": "text_feedback", "feedback": "No human replies before timeout."}
            data = self._aggregate_with_prompt(prompt, fallback)
            data.setdefault("mode", "text_feedback")
            return data

        def request_open_answer_from_slack(question: str, channel: str) -> Dict[str, Any]:
            """Request humans to provide an open answer."""
            text = (
                "*HITL: open question*\n"
                f"*Question*: {question}\n"
                "Reply with your answer or suggestions."
            )
            channel_id, thread_ts = self._post_message(text, channel)
            replies = self._poll_human_replies(channel_id, thread_ts)
            replies_text = self._format_replies_for_prompt(replies)
            prompt = (
                "Aggregate the human answers.\n"
                "Return JSON with keys: mode, answer.\n"
                "If multiple replies, synthesize them; if none, state that.\n"
                f"Question: {question}\n"
                f"Replies (newest first):\n{replies_text}"
            )
            fallback = {"mode": "open_answer", "answer": "No human replies before timeout."}
            data = self._aggregate_with_prompt(prompt, fallback)
            data.setdefault("mode", "open_answer")
            return data

        return [
            request_select_best_from_slack,
            request_score_from_slack,
            request_review_from_slack,
            request_open_answer_from_slack,
        ]

    def _format_replies_for_prompt(self, replies: List[dict]) -> str:
        """Format Slack replies for the aggregation prompt."""
        return "\n".join(
            f"- {msg.get('user', 'user')} @ {msg.get('ts', '')}: {msg.get('text', '')}"
            for msg in replies
        )

    def _is_bot_message(self, msg: dict) -> bool:
        """Check if a message is from a bot."""
        if msg.get("bot_id"):
            return True
        if msg.get("subtype"):
            return True
        return False


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== HITL Agent Test ===\n")
    
    # Check config
    if not config.hitl.slack_token:
        print("❌ SLACK_TOKEN not set - HITL agent requires Slack")
        print("   Set SLACK_TOKEN in .env to test")
        exit(1)
    
    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not api_base or not api_key:
        print("❌ LITELLM_API_BASE and LITELLM_API_KEY must be set")
        exit(1)
    
    print(f"✅ Slack Token: {config.hitl.slack_token[:20]}...")
    print(f"✅ LiteLLM API Base: {api_base}")
    print(f"✅ Default Channel: {config.hitl.default_channel}")
    
    # Create HITL agent
    try:
        hitl = HitlAgent()
        print("✅ HITL Agent initialized successfully!")
        
        # Test domain classification (without Slack)
        print("\n📋 Testing domain classification...")
        channel, classification = hitl.router.get_channel_for_question(
            "What's the best approach for building a distributed microservices architecture?",
            fallback_channel=config.hitl.default_channel,
        )
        print(f"   Domain: {classification.domain.value}")
        print(f"   Confidence: {classification.confidence:.2f}")
        print(f"   Reasoning: {classification.reasoning}")
        print(f"   Channel: {channel}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
