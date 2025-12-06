"""
Parallel API Tools - URL Content Extraction

Uses Parallel's Extract API to fetch full content from URLs:
- Full article/page content extraction
- Relevant excerpts based on search objective
- Handles multiple URLs in batch

API Reference: https://docs.parallel.ai/api-reference/extract-beta/extract
"""
import os
import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from agno.tools import Toolkit
from agno.utils.log import logger

# Parallel API is available via HTTP requests
PARALLEL_AVAILABLE = True
PARALLEL_API_URL = "https://api.parallel.ai/v1beta/extract"
PARALLEL_BETA_HEADER = "search-extract-2025-10-10"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractedContent:
    """Structured extracted content from a URL"""
    url: str
    title: str
    excerpts: List[str]
    full_content: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "excerpts": self.excerpts,
            "full_content": self.full_content,
            "error": self.error,
        }
    
    @property
    def content_length(self) -> int:
        """Total content length in characters"""
        return len(self.full_content) if self.full_content else 0
    
    @property
    def word_count(self) -> int:
        """Approximate word count"""
        return len(self.full_content.split()) if self.full_content else 0


# =============================================================================
# Parallel Extract Toolkit
# =============================================================================

class ParallelExtractTools(Toolkit):
    """
    Custom Agno Toolkit for Parallel API URL extraction.
    
    Features:
    - Extract full content from URLs
    - Get relevant excerpts based on research objective
    - Batch extraction for multiple URLs
    - Content summarization for knowledge base storage
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        include_excerpts: bool = True,
        include_full_content: bool = True,
    ):
        """
        Initialize Parallel Extract Tools.
        
        Args:
            api_key: Parallel API key (defaults to PARALLEL_API_KEY env var)
            include_excerpts: Whether to extract relevant excerpts
            include_full_content: Whether to extract full page content
        """
        self.api_key = api_key or os.getenv("PARALLEL_API_KEY")
        self.include_excerpts = include_excerpts
        self.include_full_content = include_full_content
        
        # Register tools with Toolkit
        tools = [
            self.extract_url,
            self.extract_urls,
            self.extract_for_research,
        ]
        
        super().__init__(name="parallel_extract", tools=tools)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Parallel API"""
        if not self.api_key:
            raise ValueError("PARALLEL_API_KEY not set")
        return {
            "x-api-key": self.api_key,
            "parallel-beta": PARALLEL_BETA_HEADER,
            "Content-Type": "application/json",
        }
    
    def _extract_content(
        self,
        urls: List[str],
        objective: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
    ) -> List[ExtractedContent]:
        """
        Internal method to extract content from URLs via Parallel API.
        
        Args:
            urls: List of URLs to extract content from
            objective: Research objective to focus extraction
            search_queries: Specific queries to focus excerpts on
            
        Returns:
            List of ExtractedContent objects
        """
        try:
            # Build request payload
            payload = {
                "urls": urls,
                "excerpts": self.include_excerpts,
                "full_content": self.include_full_content,
            }
            
            if objective:
                payload["objective"] = objective
            if search_queries:
                payload["search_queries"] = search_queries
            
            # Call Parallel API via HTTP
            response = requests.post(
                PARALLEL_API_URL,
                headers=self._get_headers(),
                json=payload,
                timeout=60,  # 60 second timeout for extraction
            )
            
            if response.status_code != 200:
                logger.error(f"Parallel API error: {response.status_code} - {response.text}")
                return [
                    ExtractedContent(
                        url=url,
                        title="Error",
                        excerpts=[],
                        full_content="",
                        error=f"HTTP {response.status_code}: {response.text[:200]}",
                    )
                    for url in urls
                ]
            
            data = response.json()
            results = []
            
            # Process successful results
            for item in data.get("results", []):
                content = ExtractedContent(
                    url=item.get("url", ""),
                    title=item.get("title", "Untitled"),
                    excerpts=item.get("excerpts", []) or [],
                    full_content=item.get("full_content", "") or "",
                )
                results.append(content)
                logger.info(f"Extracted {content.word_count} words from: {content.url[:50]}...")
            
            # Process errors
            for error in data.get("errors", []):
                results.append(ExtractedContent(
                    url=error.get("url", ""),
                    title="Error",
                    excerpts=[],
                    full_content="",
                    error=f"{error.get('error_type', 'unknown')}: {error.get('content', '')}",
                ))
                logger.warning(f"Extraction failed for {error.get('url')}: {error.get('error_type')}")
            
            return results
            
        except requests.exceptions.Timeout:
            logger.error(f"Parallel API timeout for URLs: {urls}")
            return [
                ExtractedContent(url=url, title="Error", excerpts=[], full_content="", error="Request timeout")
                for url in urls
            ]
        except Exception as e:
            logger.error(f"Parallel extraction error: {e}")
            return [
                ExtractedContent(url=url, title="Error", excerpts=[], full_content="", error=str(e))
                for url in urls
            ]
    
    # =========================================================================
    # Public Tool Methods (exposed to agents)
    # =========================================================================
    
    def extract_url(self, url: str, objective: Optional[str] = None) -> str:
        """
        Extract full content from a single URL.
        
        Use this to get comprehensive content from a source URL discovered during search.
        This extracts the FULL article/page content, not just a snippet.
        
        Args:
            url: The URL to extract content from
            objective: Optional research objective to focus extraction
        
        Returns:
            str: Extracted content with title, excerpts, and full text
        
        Example:
            >>> content = tools.extract_url(
            ...     "https://arxiv.org/abs/2303.08774",
            ...     objective="GPT-4 capabilities and benchmarks"
            ... )
        """
        logger.info(f"Extracting content from: {url}")
        
        results = self._extract_content([url], objective=objective)
        
        if not results:
            return f"## Extraction Failed\n\nNo content could be extracted from: {url}"
        
        content = results[0]
        
        if content.error:
            return f"## Extraction Error\n\n**URL:** {url}\n**Error:** {content.error}"
        
        # Format output
        output = [f"## Extracted Content\n"]
        output.append(f"**Title:** {content.title}")
        output.append(f"**URL:** {content.url}")
        output.append(f"**Word Count:** {content.word_count:,} words")
        output.append(f"**Character Count:** {content.content_length:,} chars\n")
        
        if content.excerpts:
            output.append("### Key Excerpts\n")
            for i, excerpt in enumerate(content.excerpts[:5], 1):
                output.append(f"{i}. {excerpt}\n")
        
        if content.full_content:
            output.append("### Full Content\n")
            # Include full content (truncate for display if very long)
            if len(content.full_content) > 15000:
                output.append(content.full_content[:15000])
                output.append(f"\n\n*[Content truncated - {content.word_count:,} total words]*")
            else:
                output.append(content.full_content)
        
        return "\n".join(output)
    
    def extract_urls(
        self,
        urls: List[str],
        objective: Optional[str] = None,
    ) -> str:
        """
        Extract full content from multiple URLs in batch.
        
        Use this to efficiently extract content from multiple source URLs at once.
        
        Args:
            urls: List of URLs to extract content from (max 10 recommended)
            objective: Optional research objective to focus extraction
        
        Returns:
            str: Extracted content from all URLs with summaries
        """
        # Limit batch size
        urls = urls[:10]
        logger.info(f"Batch extracting {len(urls)} URLs")
        
        results = self._extract_content(urls, objective=objective)
        
        if not results:
            return "## Batch Extraction Failed\n\nNo content could be extracted."
        
        # Format output
        output = [f"## Batch Extraction Results\n"]
        output.append(f"**URLs Processed:** {len(results)}")
        
        successful = [r for r in results if not r.error]
        failed = [r for r in results if r.error]
        
        output.append(f"**Successful:** {len(successful)}")
        output.append(f"**Failed:** {len(failed)}\n")
        
        total_words = sum(r.word_count for r in successful)
        output.append(f"**Total Words Extracted:** {total_words:,}\n")
        
        for i, content in enumerate(results, 1):
            output.append(f"---\n### {i}. {content.title[:80]}")
            output.append(f"**URL:** {content.url}")
            
            if content.error:
                output.append(f"**Error:** {content.error}\n")
            else:
                output.append(f"**Words:** {content.word_count:,}")
                
                if content.excerpts:
                    output.append(f"**Key Excerpt:** {content.excerpts[0][:500]}...")
                
                if content.full_content:
                    # Show preview of full content
                    preview = content.full_content[:1000].replace('\n', ' ')
                    output.append(f"**Content Preview:** {preview}...\n")
        
        return "\n".join(output)
    
    def extract_for_research(
        self,
        url: str,
        research_query: str,
        subtask_focus: Optional[str] = None,
    ) -> str:
        """
        Extract content from a URL optimized for research knowledge base.
        
        This is the PRIMARY method for deep research extraction.
        It extracts comprehensive content focused on your research objective,
        formatted for saving to the knowledge base.
        
        Args:
            url: The source URL to extract from
            research_query: The research query/topic being investigated
            subtask_focus: Specific subtask focus for targeted extraction
        
        Returns:
            str: Comprehensive extracted content ready for knowledge base
        
        Example:
            >>> content = tools.extract_for_research(
            ...     url="https://arxiv.org/abs/2303.08774",
            ...     research_query="AI agent capabilities 2024",
            ...     subtask_focus="Benchmark performance analysis"
            ... )
        """
        objective = f"Research on: {research_query}"
        if subtask_focus:
            objective += f" | Focus: {subtask_focus}"
        
        search_queries = [research_query]
        if subtask_focus:
            search_queries.append(subtask_focus)
        
        logger.info(f"Research extraction from: {url}")
        
        results = self._extract_content(
            [url],
            objective=objective,
            search_queries=search_queries,
        )
        
        if not results or results[0].error:
            error_msg = results[0].error if results else "Unknown error"
            return f"## Extraction Failed\n\n**URL:** {url}\n**Error:** {error_msg}"
        
        content = results[0]
        
        # Format for research use
        output = [f"## Research Content Extraction\n"]
        output.append(f"**Source:** {content.title}")
        output.append(f"**URL:** {content.url}")
        output.append(f"**Research Query:** {research_query}")
        if subtask_focus:
            output.append(f"**Subtask Focus:** {subtask_focus}")
        output.append(f"**Content Size:** {content.word_count:,} words / {content.content_length:,} chars\n")
        
        # Excerpts section (most relevant content)
        if content.excerpts:
            output.append("### Relevant Excerpts (High-Signal Content)\n")
            for i, excerpt in enumerate(content.excerpts, 1):
                output.append(f"**Excerpt {i}:**")
                output.append(f"{excerpt}\n")
        
        # Full content section
        if content.full_content:
            output.append("### Full Article Content\n")
            output.append(content.full_content)
        
        return "\n".join(output)
    
    def get_content_for_chunking(
        self,
        url: str,
        objective: Optional[str] = None,
    ) -> Optional[ExtractedContent]:
        """
        Get raw extracted content object for further processing (like chunking).
        
        This returns the ExtractedContent dataclass directly for use with
        chunking strategies or other processing pipelines.
        
        Args:
            url: URL to extract
            objective: Optional research objective
            
        Returns:
            ExtractedContent object or None if extraction failed
        """
        results = self._extract_content([url], objective=objective)
        
        if results and not results[0].error:
            return results[0]
        return None


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Parallel Extract Tools Test ===\n")
    
    # Check API key
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        print("❌ PARALLEL_API_KEY not set")
        print("   Get your API key from: https://parallel.ai")
        exit(1)
    
    print(f"✅ API key found: {api_key[:20]}...")
    
    # Initialize tools
    tools = ParallelExtractTools()
    
    # Test single extraction
    print("\n--- Single URL Extraction Test ---")
    test_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    result = tools.extract_url(test_url, objective="LLM autonomous agents")
    print(result[:3000] + "..." if len(result) > 3000 else result)
    
    # Test research extraction
    print("\n\n--- Research Extraction Test ---")
    result = tools.extract_for_research(
        url=test_url,
        research_query="AI agents architecture",
        subtask_focus="Planning and memory components"
    )
    print(result[:3000] + "..." if len(result) > 3000 else result)

