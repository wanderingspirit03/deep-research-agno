"""
Government Document Search Tools - Specialized toolkit for law enforcement research.

Provides tools for:
- Google Dork-based government document searches
- Procurement portal searching
- Open data portal access
- Police report and council minutes discovery
"""
import os
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from agno.tools import Toolkit
from agno.utils.log import logger

try:
    from perplexity import Perplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    logger.warning("perplexity package not installed")


# =============================================================================
# Dork Pattern Templates
# =============================================================================

DORK_TEMPLATES = {
    # Police reports with AI disclaimers
    "police_reports_draftone": [
        'site:.gov "Axon DraftOne" police',
        'site:.gov "Draft One" incident report',
        'site:.gov "AI-assisted report" police',
        'site:police.gov "drafted using AI"',
        'filetype:pdf "incident report" "Axon DraftOne"',
    ],
    
    # Procurement documents
    "procurement_axon": [
        'site:.gov "Axon AI Era Plan" contract',
        'site:.gov "Axon" procurement "artificial intelligence"',
        '"purchase order" "Axon" "AI Era Plan"',
        '"sole source" Axon "AI Era"',
        'site:bidnet.com "Axon DraftOne"',
    ],
    
    # Council/board minutes
    "council_minutes": [
        'site:.gov "city council" "Axon DraftOne"',
        'site:.gov "board of supervisors" "AI report writing" police',
        '"council agenda" "Axon" "AI" police',
        '"meeting minutes" "DraftOne" approval',
        'filetype:pdf "council" "Axon AI Era"',
    ],
    
    # News and press releases
    "news_coverage": [
        '"police department" "adopts" "Axon DraftOne"',
        '"sheriff\'s office" "AI-powered report writing"',
        '"law enforcement" "Axon AI Era Plan" deployment',
        'site:patch.com "police" "DraftOne"',
        '"press release" police "Axon AI"',
    ],
    
    # Generic government AI
    "government_ai": [
        'site:.gov police "artificial intelligence" report',
        'site:.gov "AI-generated" OR "AI-assisted" law enforcement',
        'site:state.*.us police "DraftOne" OR "AI Era"',
    ],
}

# State-specific domain patterns
STATE_DOMAINS = {
    "CA": ["ca.gov", "police.ca.gov", "lapd.com", "sfgov.org"],
    "TX": ["texas.gov", "dps.texas.gov", "houstontx.gov", "dallascityhall.com"],
    "FL": ["fl.gov", "fdle.state.fl.us", "miamidade.gov"],
    "NY": ["ny.gov", "nyc.gov", "nypd.org"],
    "IL": ["illinois.gov", "chicago.gov"],
    "PA": ["pa.gov", "philadelphiapolice.com"],
    "OH": ["ohio.gov", "columbus.gov"],
    "GA": ["georgia.gov", "atlantapd.org"],
    "NC": ["nc.gov", "charlotte.gov"],
    "MI": ["michigan.gov", "detroitmi.gov"],
    "AZ": ["az.gov", "phoenix.gov"],
    "WA": ["wa.gov", "seattle.gov"],
    "CO": ["colorado.gov", "denvergov.org"],
    "MA": ["mass.gov", "boston.gov"],
    "NV": ["nv.gov", "lvmpd.com"],
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DorkSearchResult:
    """Result from a dork-based search"""
    query_used: str
    title: str
    url: str
    snippet: str
    domain: str
    is_gov_domain: bool
    potential_evidence_type: str  # police_report, procurement, council, news, other
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_used": self.query_used,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
            "is_gov_domain": self.is_gov_domain,
            "potential_evidence_type": self.potential_evidence_type,
        }


# =============================================================================
# Government Document Search Toolkit
# =============================================================================

class GovernmentSearchTools(Toolkit):
    """
    Specialized toolkit for searching government documents related to
    law enforcement AI adoption.
    
    Features:
    - Pre-built Google Dork patterns for common document types
    - State-specific search targeting
    - Evidence type classification
    - AI disclaimer detection in results
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 15,
    ):
        """
        Initialize Government Search Tools.
        
        Args:
            api_key: Perplexity API key
            max_results: Maximum results per search
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.max_results = max_results
        self._client: Optional[Perplexity] = None
        
        # Register tools
        tools = [
            self.search_police_reports,
            self.search_procurement,
            self.search_council_minutes,
            self.search_government_news,
            self.search_state_agencies,
            self.build_custom_dork,
        ]
        
        super().__init__(name="government_search", tools=tools)
    
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
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except:
            return ""
    
    def _is_gov_domain(self, url: str) -> bool:
        """Check if URL is a government domain"""
        domain = self._extract_domain(url)
        gov_patterns = [".gov", "police.", "sheriff.", "pd.", "cityof"]
        return any(pattern in domain for pattern in gov_patterns)
    
    def _classify_evidence_type(self, title: str, snippet: str, url: str) -> str:
        """Classify the type of evidence based on content"""
        text = f"{title} {snippet}".lower()
        
        if any(kw in text for kw in ["incident report", "arrest report", "police report", "case number"]):
            return "police_report"
        elif any(kw in text for kw in ["procurement", "contract", "rfp", "purchase order", "bid"]):
            return "procurement"
        elif any(kw in text for kw in ["council", "board", "minutes", "agenda", "meeting"]):
            return "council_minutes"
        elif any(kw in text for kw in ["press release", "announces", "news", "article"]):
            return "news"
        elif any(kw in text for kw in ["policy", "memo", "directive", "guideline"]):
            return "policy"
        else:
            return "other"
    
    def _detect_ai_disclaimer_signals(self, text: str) -> bool:
        """Check if text contains AI disclaimer signals"""
        patterns = [
            r"axon\s*draftone",
            r"draft\s*one",
            r"ai[\s-]*assisted",
            r"ai[\s-]*generated",
            r"artificial\s*intelligence",
            r"drafted\s+using\s+ai",
            r"officer\s+(?:is\s+)?responsible",
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)
    
    def _search_with_dorks(
        self,
        dork_queries: List[str],
        evidence_type: str,
    ) -> List[DorkSearchResult]:
        """Execute multiple dork searches and aggregate results"""
        results = []
        seen_urls = set()
        
        for dork_query in dork_queries:
            try:
                response = self.client.search.create(
                    query=dork_query,
                    max_results=self.max_results,
                )
                
                if hasattr(response, 'results'):
                    for item in response.results:
                        url = getattr(item, 'url', '')
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        
                        title = getattr(item, 'title', 'No title')
                        snippet = getattr(item, 'snippet', '')
                        
                        result = DorkSearchResult(
                            query_used=dork_query,
                            title=title,
                            url=url,
                            snippet=snippet,
                            domain=self._extract_domain(url),
                            is_gov_domain=self._is_gov_domain(url),
                            potential_evidence_type=self._classify_evidence_type(title, snippet, url),
                        )
                        results.append(result)
                        
            except Exception as e:
                logger.warning(f"Dork search failed for '{dork_query}': {e}")
                continue
        
        return results
    
    def _format_results(self, results: List[DorkSearchResult], search_type: str) -> str:
        """Format search results for agent consumption"""
        if not results:
            return f"## Government Search Results ({search_type})\n\nNo results found."
        
        output = [f"## Government Search Results ({search_type})\n"]
        output.append(f"**Total Results:** {len(results)}")
        
        gov_count = sum(1 for r in results if r.is_gov_domain)
        output.append(f"**Government Domains:** {gov_count}")
        output.append("")
        
        # Group by evidence type
        by_type = {}
        for r in results:
            by_type.setdefault(r.potential_evidence_type, []).append(r)
        
        for etype, items in by_type.items():
            output.append(f"### {etype.replace('_', ' ').title()} ({len(items)} results)")
            output.append("")
            
            for i, r in enumerate(items[:10], 1):
                gov_badge = "🏛️" if r.is_gov_domain else "🌐"
                ai_badge = "🤖" if self._detect_ai_disclaimer_signals(r.snippet) else ""
                
                output.append(f"**{i}. {gov_badge}{ai_badge} {r.title}**")
                output.append(f"**URL:** {r.url}")
                output.append(f"**Domain:** {r.domain}")
                output.append(f"**Snippet:** {r.snippet[:300]}...")
                output.append("")
        
        return "\n".join(output)
    
    # =========================================================================
    # Public Tool Methods
    # =========================================================================
    
    def search_police_reports(
        self,
        additional_terms: Optional[str] = None,
        state: Optional[str] = None,
    ) -> str:
        """
        Search for police reports containing AI/DraftOne disclaimers.
        
        This is the PRIMARY search for finding direct evidence of DraftOne usage.
        
        Args:
            additional_terms: Additional search terms to include
            state: 2-letter state code to focus on (e.g., 'CA', 'TX')
        
        Returns:
            str: Formatted search results with highlighted AI disclaimer signals
        
        Example:
            >>> tools.search_police_reports()
            >>> tools.search_police_reports(state="CA")
        """
        logger.info(f"Searching police reports for AI disclaimers (state={state})")
        
        dork_queries = list(DORK_TEMPLATES["police_reports_draftone"])
        
        # Add state-specific queries
        if state and state.upper() in STATE_DOMAINS:
            state_domains = STATE_DOMAINS[state.upper()]
            for domain in state_domains[:2]:
                dork_queries.append(f'site:{domain} "DraftOne" OR "AI-assisted" report')
        
        # Add custom terms
        if additional_terms:
            dork_queries.append(f'{additional_terms} "Axon DraftOne" police report')
        
        results = self._search_with_dorks(dork_queries, "police_report")
        return self._format_results(results, "Police Reports with AI Disclaimers")
    
    def search_procurement(
        self,
        product_name: str = "Axon AI Era Plan",
        state: Optional[str] = None,
    ) -> str:
        """
        Search government procurement documents for AI product contracts.
        
        Args:
            product_name: Product name to search for
            state: 2-letter state code to focus on
        
        Returns:
            str: Formatted search results from procurement portals
        
        Example:
            >>> tools.search_procurement("Axon AI Era Plan", state="TX")
        """
        logger.info(f"Searching procurement for: {product_name}")
        
        dork_queries = list(DORK_TEMPLATES["procurement_axon"])
        
        # Add product-specific query
        dork_queries.insert(0, f'site:.gov "{product_name}" procurement OR contract')
        
        # Add state-specific
        if state:
            dork_queries.append(f'site:{state.lower()}.gov "{product_name}" contract')
        
        results = self._search_with_dorks(dork_queries, "procurement")
        return self._format_results(results, "Procurement Documents")
    
    def search_council_minutes(
        self,
        city: Optional[str] = None,
        state: Optional[str] = None,
    ) -> str:
        """
        Search city council and board meeting minutes for AI adoption discussions.
        
        Args:
            city: City name to focus on
            state: 2-letter state code
        
        Returns:
            str: Formatted search results from council/board meetings
        
        Example:
            >>> tools.search_council_minutes(city="Seattle", state="WA")
        """
        logger.info(f"Searching council minutes (city={city}, state={state})")
        
        dork_queries = list(DORK_TEMPLATES["council_minutes"])
        
        if city:
            dork_queries.insert(0, f'"{city}" "city council" "Axon" AI police')
        
        if state:
            dork_queries.append(f'site:{state.lower()}.gov council "DraftOne"')
        
        results = self._search_with_dorks(dork_queries, "council_minutes")
        return self._format_results(results, "Council/Board Minutes")
    
    def search_government_news(
        self,
        agency_name: Optional[str] = None,
        state: Optional[str] = None,
    ) -> str:
        """
        Search for news coverage and press releases about AI adoption.
        
        Args:
            agency_name: Specific agency to search for
            state: 2-letter state code
        
        Returns:
            str: Formatted news and press release results
        
        Example:
            >>> tools.search_government_news("Seattle Police Department", "WA")
        """
        logger.info(f"Searching news (agency={agency_name}, state={state})")
        
        dork_queries = list(DORK_TEMPLATES["news_coverage"])
        
        if agency_name:
            dork_queries.insert(0, f'"{agency_name}" "Axon DraftOne" OR "AI Era Plan"')
            dork_queries.insert(1, f'"{agency_name}" adopts "AI report writing"')
        
        results = self._search_with_dorks(dork_queries, "news")
        return self._format_results(results, "News Coverage")
    
    def search_state_agencies(
        self,
        state: str,
        top_n_agencies: int = 5,
    ) -> str:
        """
        Comprehensive search for AI adoption in a specific state's agencies.
        
        Searches across multiple evidence types for a single state.
        
        Args:
            state: 2-letter state code (required)
            top_n_agencies: Number of top agencies to focus on
        
        Returns:
            str: Comprehensive search results for the state
        
        Example:
            >>> tools.search_state_agencies("CA", top_n_agencies=10)
        """
        logger.info(f"Comprehensive state search: {state}")
        
        state_upper = state.upper()
        
        # Build state-specific queries
        dork_queries = [
            f'site:{state.lower()}.gov "Axon DraftOne" police',
            f'site:{state.lower()}.gov "AI Era Plan" law enforcement',
            f'"{state_upper}" police "DraftOne" adoption',
            f'"{state_upper}" sheriff "AI-assisted report"',
            f'site:.gov "{state_upper}" "Axon" artificial intelligence police',
        ]
        
        # Add state-specific domains
        if state_upper in STATE_DOMAINS:
            for domain in STATE_DOMAINS[state_upper]:
                dork_queries.append(f'site:{domain} "DraftOne" OR "AI Era"')
        
        results = self._search_with_dorks(dork_queries, "state_search")
        return self._format_results(results, f"State Search: {state_upper}")
    
    def build_custom_dork(
        self,
        base_query: str,
        site_filter: Optional[str] = None,
        filetype: Optional[str] = None,
        must_include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> str:
        """
        Build and execute a custom Google dork query.
        
        Args:
            base_query: Main search terms
            site_filter: Limit to specific site (e.g., '.gov', 'seattle.gov')
            filetype: Limit to file type (e.g., 'pdf', 'doc')
            must_include: List of terms that MUST be present (quoted)
            exclude: List of terms to exclude
        
        Returns:
            str: Formatted search results
        
        Example:
            >>> tools.build_custom_dork(
            ...     base_query="police AI report",
            ...     site_filter=".gov",
            ...     filetype="pdf",
            ...     must_include=["Axon", "DraftOne"]
            ... )
        """
        # Build dork query
        parts = []
        
        if site_filter:
            parts.append(f"site:{site_filter}")
        
        if filetype:
            parts.append(f"filetype:{filetype}")
        
        parts.append(base_query)
        
        if must_include:
            for term in must_include:
                parts.append(f'"{term}"')
        
        if exclude:
            for term in exclude:
                parts.append(f"-{term}")
        
        dork_query = " ".join(parts)
        logger.info(f"Custom dork: {dork_query}")
        
        results = self._search_with_dorks([dork_query], "custom")
        return self._format_results(results, f"Custom Search: {dork_query[:50]}...")


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Government Search Tools Test ===\n")
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("❌ PERPLEXITY_API_KEY not set")
        exit(1)
    
    print(f"✅ API key found")
    
    tools = GovernmentSearchTools()
    
    # Test police reports search
    print("\n--- Police Reports Search ---")
    results = tools.search_police_reports()
    print(results[:2000] + "..." if len(results) > 2000 else results)
