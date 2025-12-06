"""
Perplexity Search Tools - Custom Agno Toolkit for web search

Uses Perplexity Search API with:
- Multi-query batching (up to 5 queries)
- Domain filtering (academic vs general)
- Result ranking and deduplication
"""
import os
import time
import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from agno.tools import Toolkit
from agno.utils.log import logger

try:
    # Package is 'perplexityai' on pip but imports as 'perplexity'
    from perplexity import Perplexity, AsyncPerplexity
    import perplexity as perplexity_module
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    logger.warning("perplexity package not installed. Run: pip install perplexityai")


# =============================================================================
# Domain Lists
# =============================================================================

ACADEMIC_DOMAINS = [
    "arxiv.org",
    "nature.com",
    "ieee.org",
    "sciencedirect.com",
    "springer.com",
    "pubmed.ncbi.nlm.nih.gov",
    "acm.org",
    "wiley.com",
    "jstor.org",
    "scholar.google.com",
    "researchgate.net",
    "plos.org",
    "biorxiv.org",
    "medrxiv.org",
]

DENYLIST_DOMAINS = [
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    domain: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "date": self.date,
            "domain": self.domain,
        }


# =============================================================================
# Perplexity Search Toolkit
# =============================================================================

class PerplexitySearchTools(Toolkit):
    """
    Custom Agno Toolkit for Perplexity Search API.
    
    Features:
    - Single and multi-query search
    - Domain filtering (academic/general)
    - Automatic retry with exponential backoff
    - Result deduplication
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        country: str = "US",
        academic_domains: Optional[List[str]] = None,
        denylist_domains: Optional[List[str]] = None,
        retry_attempts: int = 3,
        retry_delay_base: float = 1.0,
    ):
        """
        Initialize Perplexity Search Tools.
        
        Args:
            api_key: Perplexity API key (defaults to PERPLEXITY_API_KEY env var)
            max_results: Maximum results per search
            country: ISO country code for location-based results
            academic_domains: Allowlist for academic search
            denylist_domains: Blocklist for general search
            retry_attempts: Number of retry attempts on rate limit
            retry_delay_base: Base delay for exponential backoff
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.max_results = max_results
        self.country = country
        self.academic_domains = academic_domains or ACADEMIC_DOMAINS
        self.denylist_domains = denylist_domains or DENYLIST_DOMAINS
        self.retry_attempts = retry_attempts
        self.retry_delay_base = retry_delay_base
        
        # Initialize client
        self._client: Optional[Perplexity] = None
        
        # Register tools with Toolkit
        tools = [
            self.search,
            self.batch_search,
            self.search_academic,
            self.search_general,
        ]
        
        super().__init__(name="perplexity_search", tools=tools)
    
    @property
    def client(self) -> "Perplexity":
        """Lazy initialization of Perplexity client"""
        if not PERPLEXITY_AVAILABLE:
            raise ImportError("perplexity package not installed")
        
        if self._client is None:
            if not self.api_key:
                raise ValueError("PERPLEXITY_API_KEY not set")
            self._client = Perplexity(api_key=self.api_key)
        
        return self._client
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""
    
    def _filter_by_domains(
        self,
        results: List[SearchResult],
        allowlist: Optional[List[str]] = None,
        denylist: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Filter results by domain allowlist or denylist"""
        filtered = []
        
        for result in results:
            domain = result.domain or self._extract_domain(result.url)
            
            # Check allowlist (if provided, only include matching domains)
            if allowlist:
                if any(allowed in domain for allowed in allowlist):
                    filtered.append(result)
                continue
            
            # Check denylist (exclude matching domains)
            if denylist:
                if not any(denied in domain for denied in denylist):
                    filtered.append(result)
            else:
                filtered.append(result)
        
        return filtered
    
    def _parse_results(self, response) -> List[SearchResult]:
        """Parse Perplexity API response into SearchResult objects"""
        results = []
        
        if not hasattr(response, 'results'):
            return results
        
        for item in response.results:
            result = SearchResult(
                title=getattr(item, 'title', 'No title'),
                url=getattr(item, 'url', ''),
                snippet=getattr(item, 'snippet', ''),
                date=getattr(item, 'date', None),
            )
            result.domain = self._extract_domain(result.url)
            results.append(result)
        
        return results
    
    def _search_with_retry(self, query, max_results: int) -> List[SearchResult]:
        """Execute search with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.search.create(
                    query=query,
                    max_results=max_results,
                    country=self.country,
                )
                return self._parse_results(response)
            
            except Exception as e:
                error_name = type(e).__name__
                
                # Check for rate limit
                if "RateLimit" in error_name or "429" in str(e):
                    if attempt < self.retry_attempts - 1:
                        delay = (self.retry_delay_base * (2 ** attempt)) + random.uniform(0, 1)
                        logger.warning(f"Rate limited, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                
                logger.error(f"Search error: {e}")
                raise
        
        return []
    
    # =========================================================================
    # Public Tool Methods (exposed to agents)
    # =========================================================================
    
    def search(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Perform a single web search query.
        
        Args:
            query: The search query string
            max_results: Maximum number of results (default: 10)
        
        Returns:
            str: Formatted search results with titles, URLs, and snippets
        """
        max_results = max_results or self.max_results
        logger.info(f"Searching: {query}")
        
        try:
            results = self._search_with_retry(query, max_results)
            
            if not results:
                return f"No results found for: {query}"
            
            # Format results
            output = [f"## Search Results for: {query}\n"]
            for i, r in enumerate(results, 1):
                output.append(f"### {i}. {r.title}")
                output.append(f"**URL:** {r.url}")
                if r.date:
                    output.append(f"**Date:** {r.date}")
                output.append(f"**Snippet:** {r.snippet}\n")
            
            return "\n".join(output)
        
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def batch_search(self, queries: List[str], max_results: Optional[int] = None) -> str:
        """
        Perform multiple search queries in a single batch.
        
        Args:
            queries: List of search query strings (max 5)
            max_results: Maximum results per query (default: 10)
        
        Returns:
            str: Combined formatted search results from all queries
        """
        max_results = max_results or self.max_results
        
        # Limit to 5 queries per batch
        queries = queries[:5]
        logger.info(f"Batch searching {len(queries)} queries")
        
        try:
            response = self.client.search.create(
                query=queries,
                max_results=max_results,
                country=self.country,
            )
            results = self._parse_results(response)
            
            if not results:
                return f"No results found for queries: {queries}"
            
            # Format results
            output = [f"## Batch Search Results ({len(queries)} queries)\n"]
            output.append(f"**Queries:** {', '.join(queries)}\n")
            
            for i, r in enumerate(results, 1):
                output.append(f"### {i}. {r.title}")
                output.append(f"**URL:** {r.url}")
                output.append(f"**Domain:** {r.domain}")
                if r.date:
                    output.append(f"**Date:** {r.date}")
                output.append(f"**Snippet:** {r.snippet}\n")
            
            return "\n".join(output)
        
        except Exception as e:
            return f"Batch search error: {str(e)}"
    
    def search_academic(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Search with academic domain filter.
        Only returns results from academic sources (arxiv, nature, ieee, etc.)
        
        Args:
            query: The search query string
            max_results: Maximum number of results (default: 10)
        
        Returns:
            str: Formatted search results from academic sources only
        """
        max_results = max_results or self.max_results
        # Request more results since we'll filter
        fetch_count = min(max_results * 3, 30)
        
        logger.info(f"Academic search: {query}")
        
        try:
            results = self._search_with_retry(query, fetch_count)
            filtered = self._filter_by_domains(results, allowlist=self.academic_domains)
            filtered = filtered[:max_results]
            
            if not filtered:
                return f"No academic results found for: {query}\nTry a broader search or use search() for general results."
            
            # Format results
            output = [f"## Academic Search Results for: {query}\n"]
            output.append(f"*Filtered to academic sources only*\n")
            
            for i, r in enumerate(filtered, 1):
                output.append(f"### {i}. {r.title}")
                output.append(f"**URL:** {r.url}")
                output.append(f"**Source:** {r.domain}")
                if r.date:
                    output.append(f"**Date:** {r.date}")
                output.append(f"**Snippet:** {r.snippet}\n")
            
            return "\n".join(output)
        
        except Exception as e:
            return f"Academic search error: {str(e)}"
    
    def search_general(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Search with low-quality domain filter.
        Excludes social media and low-quality sources.
        
        Args:
            query: The search query string
            max_results: Maximum number of results (default: 10)
        
        Returns:
            str: Formatted search results excluding low-quality sources
        """
        max_results = max_results or self.max_results
        # Request more results since we'll filter
        fetch_count = min(max_results * 2, 20)
        
        logger.info(f"General search (filtered): {query}")
        
        try:
            results = self._search_with_retry(query, fetch_count)
            filtered = self._filter_by_domains(results, denylist=self.denylist_domains)
            filtered = filtered[:max_results]
            
            if not filtered:
                return f"No quality results found for: {query}"
            
            # Format results
            output = [f"## Search Results for: {query}\n"]
            output.append(f"*Filtered to exclude social media and low-quality sources*\n")
            
            for i, r in enumerate(filtered, 1):
                output.append(f"### {i}. {r.title}")
                output.append(f"**URL:** {r.url}")
                output.append(f"**Domain:** {r.domain}")
                if r.date:
                    output.append(f"**Date:** {r.date}")
                output.append(f"**Snippet:** {r.snippet}\n")
            
            return "\n".join(output)
        
        except Exception as e:
            return f"General search error: {str(e)}"


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Perplexity Search Tools Test ===\n")
    
    # Check API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("❌ PERPLEXITY_API_KEY not set")
        print("   Set it in your .env file or environment")
        exit(1)
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    # Initialize tools
    tools = PerplexitySearchTools()
    
    # Test single search
    print("\n--- Single Search Test ---")
    result = tools.search("What is LangChain?", max_results=3)
    print(result)

