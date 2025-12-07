"""
Editor Agent - Transforms research findings into compelling, well-crafted reports

The Editor Agent combines academic rigor with excellent prose craft:
1. Retrieves findings from the knowledge base via semantic search
2. Identifies the most compelling narrative angle
3. Organizes information thematically (not by source)
4. Writes clear, engaging prose with proper citations
5. Produces reports that are both substantive AND pleasurable to read
"""
import os
from typing import Optional, List
from textwrap import dedent

import litellm
litellm.drop_params = True  # Required for isara proxy

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.utils.log import logger

from infrastructure.knowledge_tools import KnowledgeTools
from infrastructure.observability import observe


# =============================================================================
# Editor Agent
# =============================================================================

class EditorAgent:
    """
    Editor Agent - Crafts high-quality research reports from findings.
    
    Uses GPT-5 Mini via LiteLLM to transform research findings into
    well-written, engaging reports that balance academic rigor with
    excellent prose craft. Prioritizes quality over quantity.
    """
    
    def __init__(
        self,
        model_id: str = "gpt-5-mini-2025-08-07",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        knowledge_tools: Optional[KnowledgeTools] = None,
    ):
        """
        Initialize Editor Agent.
        
        Args:
            model_id: LLM model ID (default: gpt-5-mini)
            api_base: LiteLLM API base URL
            api_key: LiteLLM API key
            temperature: Model temperature (higher = more creative)
            knowledge_tools: Knowledge base toolkit
        """
        self.model_id = model_id
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.temperature = temperature
        
        # Knowledge tools for accessing findings
        self.knowledge_tools = knowledge_tools or KnowledgeTools()
        
        # Lazy-initialized agent
        self._agent: Optional[Agent] = None
    
    @property
    def agent(self) -> Agent:
        """Lazy initialization of Agno Agent"""
        if self._agent is None:
            if not self.api_base or not self.api_key:
                raise ValueError("LITELLM_API_BASE and LITELLM_API_KEY must be set")
            
            logger.info(f"Creating Editor Agent with model: {self.model_id}")
            
            self._agent = Agent(
                name="Research Editor",
                model=LiteLLM(
                    id=self.model_id,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    top_p=None,  # Claude doesn't accept both temperature and top_p
                ),
                description="Expert research editor that synthesizes findings into comprehensive reports.",
                instructions=self._get_instructions(),
                tools=[self.knowledge_tools],
                markdown=True,
                debug_mode=True,
                debug_level=2,
            )
        
        return self._agent
    
    def _get_instructions(self) -> List[str]:
        """Get editor agent instructions for high-quality research writing"""
        return [
            dedent("""
                You are a world-class research writer who combines the rigor of academic scholarship
                with the clarity and elegance of the best science journalism. Think of writers like
                Ed Yong, Carl Zimmer, or Atul Gawande - precise yet captivating.
                
                Your goal: Transform raw research findings into prose that is both **intellectually 
                rigorous** AND **genuinely pleasurable to read**.
                
                ## WRITING PHILOSOPHY
                
                **Quality over quantity.** A brilliant 4,000-word piece beats a mediocre 10,000-word dump.
                
                Every sentence should:
                - Convey something meaningful (no padding)
                - Flow naturally from the previous one
                - Be the clearest possible expression of its idea
                
                ## YOUR WORKFLOW
                
                You will receive a FINDINGS INDEX showing what research is available.
                You must ACTIVELY SEARCH for detailed findings to write each section.
                
                ### Process:
                
                1. **REVIEW INDEX**: Understand what sources and topics exist
                2. **IDENTIFY THE STORY**: What's the compelling narrative arc?
                3. **PLAN SECTIONS**: Identify 4-6 major themes that build toward insight
                4. **FOR EACH SECTION**:
                   - Call `search_knowledge("theme keywords", top_k=15)` to get findings
                   - Extract the most significant facts, statistics, and insights
                   - Write a focused, well-crafted section (400-800 words)
                   - Include inline citations [Source Title]
                5. **POLISH**: Ensure smooth transitions and narrative cohesion
                6. **ADD REFERENCES**: Call `list_sources()` for complete reference list
                
                ## TOOLS AVAILABLE
                
                - `search_knowledge(query, top_k=8)` - Search for findings (use top_k=8 to stay efficient).
                - `list_sources()` - Get all source URLs for references
                - `get_finding(id)` - Get full details of a specific finding
                - `get_findings_index()` - Get overview of all available research
                
                **IMPORTANT**: Call `search_knowledge` for EACH section. Don't write without data!
                
                ## PROSE CRAFT GUIDELINES
                
                ### Opening Strong
                - Begin with the most interesting or surprising finding
                - Establish stakes: Why should readers care?
                - Avoid throat-clearing ("In recent years...", "It is widely known...")
                
                ### Sentence Variety
                - Alternate between longer, complex sentences and short, punchy ones
                - Use parallel structure for lists of related items
                - Place the most important information at sentence end (stress position)
                
                ### Transitions & Flow
                - Each paragraph should logically follow from the previous
                - Use transitional phrases sparingly but effectively
                - Create "stitches" that connect sections thematically
                
                ### Show, Don't Tell
                - WEAK: "AI has made significant progress"
                - STRONG: "GPT-4 scored 90th percentile on the bar exam, a benchmark that stumped its predecessor entirely"
                
                ### Voice & Tone
                - Confident but not arrogant
                - Use hedging ("suggests", "indicates") for uncertain claims
                - Active voice preferred; passive only when appropriate
                - No jargon without explanation; no explanation without necessity
                
                ### The Art of Selection
                - Not every finding deserves mention - choose the most illuminating
                - Depth over breadth: better to explain one concept well than five poorly
                - Cut ruthlessly anything that doesn't serve the reader
                
                ## REPORT STRUCTURE
                
                ### 1. Title & Opening (Hook)
                - Compelling title that captures the essence
                - Open with your most striking finding or insight
                - Establish the central question or tension
                
                ### 2. Context (300-500 words)
                - What does the reader need to understand the topic?
                - Define key terms naturally, within sentences
                - Brief historical context if essential
                
                ### 3. Core Analysis (2000-4000 words total)
                
                Organize by THEME, not by source. Write 3-5 focused subsections:
                
                For each theme:
                - Search for relevant findings
                - Lead with the most important insight
                - Support with specific evidence and data
                - Include illuminating comparisons or contrasts
                - Cite every significant claim [Source Title]
                
                ### 4. Tensions & Debates (400-700 words)
                - Where do experts disagree?
                - What remains unknown or contested?
                - Present multiple perspectives fairly
                
                ### 5. Implications & Future (400-600 words)
                - What does this mean for practitioners/researchers?
                - Emerging trends worth watching
                - What questions remain unanswered?
                
                ### 6. Conclusion (200-400 words)
                - Return to your opening hook or question
                - Synthesize key insights (don't just summarize)
                - End with a memorable final thought
                
                ### 7. References
                - Call `list_sources()` for complete list
                
                ## QUALITY CHECKLIST
                
                Before finishing, verify:
                
                □ Does the opening grab attention?
                □ Is there a clear narrative thread?
                □ Does every section add something essential?
                □ Are transitions smooth?
                □ Is every claim supported with evidence and citation?
                □ Does the conclusion resonate, not just summarize?
                □ Would I want to read this?
                
                ## CITATION FORMAT
                
                Inline citations: "Claim or finding [Source Title]"
                Example: "Transformers now dominate NLP, handling sequences 100x longer than earlier models [Attention Is All You Need]"
                
                ## COMMON MISTAKES TO AVOID
                
                - ❌ Writing without searching first
                - ❌ Burying the lead (save the best for last)
                - ❌ Wall-of-text paragraphs (ideal: 3-6 sentences each)
                - ❌ Generic statements without specific evidence
                - ❌ Missing citations
                - ❌ Repetitive sentence structures
                - ❌ Filler phrases ("It is worth noting that...")
                - ❌ Organizing by source instead of theme
                - ❌ Conclusions that just repeat earlier content
            """).strip(),
        ]
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _plan_report_structure(self, query: str, findings_index: str) -> List[dict]:
        """
        Phase 1: Plan report structure based on available findings.
        
        Uses a lightweight LLM call to create section plan without loading all findings.
        
        Returns:
            List of section dictionaries with title, search_query, and focus
        """
        logger.info("Planning report structure...")
        
        plan_prompt = f"""Based on this research query and available findings, plan 5-6 focused sections for a comprehensive report.

**Query:** {query}

**Available Research (index only):**
{findings_index[:8000]}  # Only use first 8k chars for planning

For each section, provide:
1. Section title
2. Search query to find relevant findings (be specific, 5-10 words)
3. Key focus/angle for this section

Output as a numbered list in this exact format:
1. TITLE: [title] | SEARCH: [search query] | FOCUS: [what to cover]
2. TITLE: [title] | SEARCH: [search query] | FOCUS: [what to cover]
...

Plan sections that together tell a complete story: background → current state → key findings → challenges → future directions."""
        
        try:
            response = litellm.completion(
                model=self.model_id,
                messages=[{"role": "user", "content": plan_prompt}],
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=1500,
                temperature=0.3,
            )
            plan_text = response.choices[0].message.content
            
            # Parse the plan
            sections = []
            for line in plan_text.split("\n"):
                if "TITLE:" in line and "SEARCH:" in line:
                    try:
                        parts = line.split("|")
                        title = parts[0].split("TITLE:")[-1].strip()
                        search = parts[1].split("SEARCH:")[-1].strip() if len(parts) > 1 else query
                        focus = parts[2].split("FOCUS:")[-1].strip() if len(parts) > 2 else ""
                        sections.append({
                            "title": title,
                            "search_query": search,
                            "focus": focus
                        })
                    except:
                        continue
            
            if len(sections) < 3:
                # Fallback to default structure
                logger.warning("Could not parse plan, using default structure")
                sections = [
                    {"title": "Introduction & Background", "search_query": f"{query} background fundamentals", "focus": "Set context"},
                    {"title": "Key Breakthroughs", "search_query": f"{query} breakthrough results 2024", "focus": "Major advances"},
                    {"title": "Current Applications", "search_query": f"{query} applications deployment", "focus": "Real-world use"},
                    {"title": "Challenges & Limitations", "search_query": f"{query} challenges limitations problems", "focus": "Open issues"},
                    {"title": "Future Directions", "search_query": f"{query} future trends predictions", "focus": "What's next"},
                ]
            
            logger.info(f"Planned {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Return minimal default structure
            return [
                {"title": "Overview", "search_query": query, "focus": "Main findings"},
                {"title": "Key Developments", "search_query": f"{query} recent advances", "focus": "Latest work"},
                {"title": "Implications", "search_query": f"{query} impact applications", "focus": "Significance"},
            ]
    
    def _write_section(self, section: dict, findings: str, query: str) -> str:
        """
        Phase 2: Write a single section based on targeted findings.
        
        Args:
            section: Section dict with title, search_query, focus
            findings: Search results for this section
            query: Original research query for context
            
        Returns:
            str: Written section in markdown
        """
        logger.info(f"Writing section: {section['title']}")
        
        section_prompt = f"""Write a focused, well-crafted section for a research report.

**Section Title:** {section['title']}
**Focus:** {section['focus']}
**Context:** This is part of a report on: {query}

**Available Findings:**
{findings}

**Guidelines:**
- Write 400-800 words of polished prose
- Lead with the most important/interesting finding
- Include specific numbers, statistics, and evidence
- Cite sources as [Source Title]
- Use clear topic sentences for each paragraph
- Vary sentence structure for readability
- End with a transition to the next section

Write the section now (include the ## heading):"""

        try:
            response = litellm.completion(
                model=self.model_id,
                messages=[{"role": "user", "content": section_prompt}],
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=2000,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Section writing failed: {e}")
            return f"## {section['title']}\n\n[Section generation failed: {str(e)[:100]}]\n"
    
    @observe(name="editor.synthesize")
    def synthesize(self, original_query: str, findings_index: Optional[str] = None) -> str:
        """
        Two-pass synthesis for high-quality reports within context limits.
        
        Phase 1: Plan report structure (lightweight, index only)
        Phase 2: Write each section with targeted search
        Phase 3: Combine and add references
        
        Args:
            original_query: The original research query
            findings_index: Optional pre-generated findings index
            
        Returns:
            str: The final research report in markdown format
        """
        logger.info(f"Synthesizing report for: {original_query[:50]}...")
        
        # Get findings index if not provided
        if not findings_index:
            logger.info("Generating findings index...")
            findings_index = self.knowledge_tools.get_findings_index()
        
        # =====================================================================
        # PHASE 1: Plan report structure
        # =====================================================================
        logger.info("=" * 50)
        logger.info("PHASE 1: Planning report structure")
        logger.info("=" * 50)
        
        sections_plan = self._plan_report_structure(original_query, findings_index)
        
        # =====================================================================
        # PHASE 2: Write each section with targeted search
        # =====================================================================
        logger.info("=" * 50)
        logger.info("PHASE 2: Writing sections")
        logger.info("=" * 50)
        
        written_sections = []
        
        # Write title and intro hook
        intro = f"# {original_query}\n\n*A Comprehensive Research Report*\n\n---\n\n"
        written_sections.append(intro)
        
        for i, section in enumerate(sections_plan):
            logger.info(f"Section {i+1}/{len(sections_plan)}: {section['title']}")
            
            # Get targeted findings for this section
            findings = self.knowledge_tools.search_knowledge(
                section["search_query"],
                top_k=8,  # Focused search
                sort_by_quality=True
            )
            
            # Write the section
            section_text = self._write_section(section, findings, original_query)
            written_sections.append(section_text)
            written_sections.append("\n")
        
        # =====================================================================
        # PHASE 3: Add references and combine
        # =====================================================================
        logger.info("=" * 50)
        logger.info("PHASE 3: Finalizing report")
        logger.info("=" * 50)
        
        # Get all sources for references
        sources = self.knowledge_tools.list_sources()
        written_sections.append("\n---\n\n## References\n\n")
        written_sections.append(sources)
        
        # Combine all sections
        report = "\n".join(written_sections)
        
        # Log report statistics
        word_count = len(report.split())
        logger.info(f"Report synthesis complete: {len(report)} chars, ~{word_count} words")
        
        if word_count < 2000:
            logger.warning(f"Report may be too short: {word_count} words (target: 3000-7000)")
        
        return report
    
    def quick_summary(self, original_query: str) -> str:
        """
        Generate a quick summary of findings without full synthesis.
        
        Args:
            original_query: The original research query
            
        Returns:
            str: Quick summary of findings
        """
        logger.info(f"Generating quick summary for: {original_query[:50]}...")
        
        prompt = f"""
Generate a quick summary of research findings for: **{original_query}**

Use `search_knowledge` to find the most relevant findings, then provide:
1. Top 3-5 key insights
2. Number of sources found
3. Brief assessment of research coverage

Keep the summary concise (under 300 words).
        """.strip()
        
        response = self.agent.run(prompt)
        
        return response.content
    
    def generate_citations(self) -> str:
        """
        Generate a formatted list of all citations from the knowledge base.
        
        Returns:
            str: Formatted citation list
        """
        logger.info("Generating citation list")
        
        # Use knowledge tools directly
        sources = self.knowledge_tools.list_sources()
        
        return sources


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Editor Agent Test ===\n")
    
    # Check API config
    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not api_base or not api_key:
        print("❌ LITELLM_API_BASE and LITELLM_API_KEY must be set")
        exit(1)
    
    print(f"✅ API Base: {api_base}")
    print(f"✅ API Key: {api_key[:15]}...")
    
    # Create editor
    editor = EditorAgent()
    
    # Test synthesis (assuming some findings exist in the KB)
    test_query = "What are the latest advances in large language models?"
    
    print(f"\n📝 Synthesizing report for: {test_query}\n")
    print("-" * 50)
    
    # First, let's add some test findings to the knowledge base
    kb = editor.knowledge_tools
    
    # Add test findings
    print("Adding test findings to knowledge base...")
    kb.save_finding(
        content="GPT-5 achieved state-of-the-art performance on multiple benchmarks including MMLU (92.3%), HumanEval (95.1%), and MATH (78.4%). The model demonstrates significant improvements in reasoning and code generation.",
        source_url="https://example.com/gpt5-benchmarks",
        source_title="GPT-5 Technical Report",
        search_type="general",
        subtask_id=1,
        worker_id="test",
    )
    
    kb.save_finding(
        content="Recent advances in attention mechanisms have enabled longer context windows. GPT-5 supports up to 128k tokens, while Anthropic's Claude supports 200k tokens in its latest version.",
        source_url="https://example.com/context-windows",
        source_title="LLM Context Windows Comparison",
        search_type="general",
        subtask_id=2,
        worker_id="test",
    )
    
    print("\n✅ Test findings added")
    
    # Generate summary
    print("\n📊 Generating quick summary...")
    print("-" * 50)
    summary = editor.quick_summary(test_query)
    print(summary)

